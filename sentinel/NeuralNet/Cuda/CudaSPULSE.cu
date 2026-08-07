#include "CudaSPULSE.hpp"

#include "CudaMatmul.hpp"
#include "CudaOps.hpp"
#include "CudaTransformerBlock.hpp"
#include "../Utils/SmokeLog.hpp"

#include <algorithm>
#include <cmath>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace {

void throwIfFailed(cudaError_t status, const char* what) {
    if (status == cudaSuccess) return;
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
}

int elementwiseBlocks(int elementCount, int threads = 256) {
    return (elementCount + threads - 1) / threads;
}

float clipScale(float scale, float scaleMin, float scaleMax) {
    if (scale < scaleMin) return scaleMin;
    if (scale > scaleMax) return scaleMax;
    return scale;
}

float computeScale(float energyFast, float energySlow, float epsilon, float scaleMin, float scaleMax) {
    return clipScale(std::sqrt(energySlow / (energyFast + epsilon)), scaleMin, scaleMax);
}

__global__ void spulseFusedStep(
    float* parameter,
    float* momentum,
    const float* gradient,
    const float* energy,
    float* sumSquares,
    int elementCount,
    float momentumBeta,
    float oneMinusMom,
    float gradientScale,
    float learningRate,
    float keep
) {
    __shared__ float shared[256];
    const float stepScale = energy[2];
    float local = 0.0f;
    for (int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
         index < elementCount;
         index += static_cast<int>(blockDim.x * gridDim.x)) {
        const float g = gradient[index] * gradientScale;
        const float m = momentumBeta * momentum[index] + oneMinusMom * g;
        momentum[index] = m;
        float value = parameter[index];
        if (keep != 1.0f)
            value *= keep;
        parameter[index] = value - learningRate * stepScale * m;
        local += g * g;
    }
    shared[threadIdx.x] = local;
    __syncthreads();
    for (int stride = static_cast<int>(blockDim.x) / 2; stride > 0; stride >>= 1) {
        if (static_cast<int>(threadIdx.x) < stride)
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        atomicAdd(sumSquares, shared[0]);
}

/// <summary>1-thread: energy EMA + write next lagged scale into energy[2]</summary>
__global__ void spulseCommitEnergyAndScale(
    float* energy,
    const float* sumSquares,
    float fastBeta,
    float slowBeta,
    float oneMinusFast,
    float oneMinusSlow,
    float epsilon,
    float scaleMin,
    float scaleMax
) {
    float eFast = energy[0];
    float eSlow = energy[1];
    const float g2 = *sumSquares;
    eFast = fastBeta * eFast + oneMinusFast * g2;
    eSlow = slowBeta * eSlow + oneMinusSlow * g2;
    energy[0] = eFast;
    energy[1] = eSlow;
    float scale = sqrtf(eSlow / (eFast + epsilon));
    if (scale < scaleMin) scale = scaleMin;
    if (scale > scaleMax) scale = scaleMax;
    energy[2] = scale;
}

void hostUpdateCore(
    float* parameter,
    float* momentum,
    float& energyFast,
    float& energySlow,
    float& scale,
    const float* gradient,
    size_t elementCount,
    float learningRate,
    float momentumBeta,
    float fastBeta,
    float slowBeta,
    float epsilon,
    float scaleMin,
    float scaleMax,
    float weightDecay,
    float gradientScale
) {
    const float oneMinusMom = 1.0f - momentumBeta;
    const float oneMinusFast = 1.0f - fastBeta;
    const float oneMinusSlow = 1.0f - slowBeta;
    const float stepScale = scale;
    float sumSquares = 0.0f;
    const float keep = 1.0f - learningRate * weightDecay;
    const float step = learningRate * stepScale;
    for (size_t i = 0; i < elementCount; ++i) {
        const float g = gradient[i] * gradientScale;
        momentum[i] = momentumBeta * momentum[i] + oneMinusMom * g;
        sumSquares += g * g;
        float value = parameter[i];
        if (keep != 1.0f)
            value *= keep;
        parameter[i] = value - step * momentum[i];
    }
    energyFast = fastBeta * energyFast + oneMinusFast * sumSquares;
    energySlow = slowBeta * energySlow + oneMinusSlow * sumSquares;
    scale = computeScale(energyFast, energySlow, epsilon, scaleMin, scaleMax);
}

void hostUpdateFromHalfCore(
    float* parameter,
    float* momentum,
    float& energyFast,
    float& energySlow,
    float& scale,
    const __half* gradHalf,
    size_t elementCount,
    float learningRate,
    float momentumBeta,
    float fastBeta,
    float slowBeta,
    float epsilon,
    float scaleMin,
    float scaleMax,
    float weightDecay,
    float gradientScale
) {
    const float oneMinusMom = 1.0f - momentumBeta;
    const float oneMinusFast = 1.0f - fastBeta;
    const float oneMinusSlow = 1.0f - slowBeta;
    const float stepScale = scale;
    float sumSquares = 0.0f;
    const float keep = 1.0f - learningRate * weightDecay;
    const float step = learningRate * stepScale;
    for (size_t i = 0; i < elementCount; ++i) {
        const float g = gradientScale * __half2float(gradHalf[i]);
        momentum[i] = momentumBeta * momentum[i] + oneMinusMom * g;
        sumSquares += g * g;
        float value = parameter[i];
        if (keep != 1.0f)
            value *= keep;
        parameter[i] = value - step * momentum[i];
    }
    energyFast = fastBeta * energyFast + oneMinusFast * sumSquares;
    energySlow = slowBeta * energySlow + oneMinusSlow * sumSquares;
    scale = computeScale(energyFast, energySlow, epsilon, scaleMin, scaleMax);
}

/// <summary>
/// HostSGD-shaped loop: master -= lr * scale * g. No momentum buffer.
/// Extra work vs HostSGD: accumulate ||g||² + update dual-horizon scale for next step.
/// </summary>
void hostUpdateFromHalfLite(
    float* parameter,
    float& energyFast,
    float& energySlow,
    float& scale,
    const __half* gradHalf,
    size_t elementCount,
    float learningRate,
    float fastBeta,
    float slowBeta,
    float epsilon,
    float scaleMin,
    float scaleMax,
    float weightDecay,
    float gradientScale
) {
    const float oneMinusFast = 1.0f - fastBeta;
    const float oneMinusSlow = 1.0f - slowBeta;
    const float step = learningRate * scale;
    const float keep = 1.0f - learningRate * weightDecay;
    float sumSquares = 0.0f;
    for (size_t i = 0; i < elementCount; ++i) {
        const float g = gradientScale * __half2float(gradHalf[i]);
        sumSquares += g * g;
        float value = parameter[i];
        if (keep != 1.0f)
            value *= keep;
        parameter[i] = value - step * g;
    }
    energyFast = fastBeta * energyFast + oneMinusFast * sumSquares;
    energySlow = slowBeta * energySlow + oneMinusSlow * sumSquares;
    scale = computeScale(energyFast, energySlow, epsilon, scaleMin, scaleMax);
}

} // namespace

bool CudaSpulseState::empty() const {
    return this->momentum.empty();
}

void CudaSpulseState::ensure(const CudaMatrix& parameter) {
    if (parameter.empty()) throw std::invalid_argument("CudaSpulseState::ensure empty parameter");
    if (!this->momentum.empty()
        && this->momentum.rows == parameter.rows
        && this->momentum.cols == parameter.cols
        && this->energy.deviceData != nullptr)
        return;
    this->momentum.ensureSize(parameter.rows, parameter.cols);
    CudaOps::zeroInPlace(this->momentum);
    // [0]=e_fast, [1]=e_slow, [2]=lagged scale (init 1)
    this->energy.ensureCapacity(3 * sizeof(float));
    const float init[3] = {0.0f, 0.0f, 1.0f};
    throwIfFailed(
        cudaMemcpy(this->energy.deviceData, init, 3 * sizeof(float), cudaMemcpyHostToDevice),
        "CudaSpulseState::ensure energy init");
}

void CudaSpulseState::free() {
    this->momentum.free();
    this->energy.free();
}

void CudaSpulseState::downloadInto(SpulseState& host) const {
    if (this->momentum.empty()) {
        host.clear();
        return;
    }
    this->momentum.downloadInto(host.momentum);
    float energyHost[3] = {0.0f, 0.0f, 1.0f};
    if (this->energy.deviceData != nullptr) {
        throwIfFailed(
            cudaMemcpy(energyHost, this->energy.deviceData, 3 * sizeof(float), cudaMemcpyDeviceToHost),
            "CudaSpulseState::downloadInto energy");
    }
    host.energyFast = energyHost[0];
    host.energySlow = energyHost[1];
    host.scale = energyHost[2];
}

void CudaSpulseState::uploadFrom(const SpulseState& host) {
    if (host.momentum.empty()) {
        this->free();
        return;
    }
    this->momentum.upload(host.momentum);
    this->energy.ensureCapacity(3 * sizeof(float));
    const float energyHost[3] = {host.energyFast, host.energySlow, host.scale};
    throwIfFailed(
        cudaMemcpy(this->energy.deviceData, energyHost, 3 * sizeof(float), cudaMemcpyHostToDevice),
        "CudaSpulseState::uploadFrom energy");
}

void CudaTransformerBlockSpulseStates::ensureFrom(const CudaTransformerBlock& block) {
    this->queryWeight.ensure(block.attention.queryWeight);
    this->keyWeight.ensure(block.attention.keyWeight);
    this->valueWeight.ensure(block.attention.valueWeight);
    this->attentionOutputWeight.ensure(block.attention.outputWeight);
    this->feedForwardGateWeight.ensure(block.feedForward.gateWeight);
    this->feedForwardUpWeight.ensure(block.feedForward.upWeight);
    this->feedForwardDownWeight.ensure(block.feedForward.downWeight);
}

void CudaTransformerBlockSpulseStates::free() {
    this->queryWeight.free();
    this->keyWeight.free();
    this->valueWeight.free();
    this->attentionOutputWeight.free();
    this->feedForwardGateWeight.free();
    this->feedForwardUpWeight.free();
    this->feedForwardDownWeight.free();
}

void CudaTransformerBlockHostSpulseStates::clear() {
    this->queryWeight.clear();
    this->keyWeight.clear();
    this->valueWeight.clear();
    this->attentionOutputWeight.clear();
    this->feedForwardGateWeight.clear();
    this->feedForwardUpWeight.clear();
    this->feedForwardDownWeight.clear();
}

CudaSpulse::CudaSpulse(
    float learningRate,
    float momentumBeta,
    float fastBeta,
    float slowBeta,
    float epsilon,
    float scaleMin,
    float scaleMax,
    float weightDecay,
    SpulseCoverage coverage,
    bool hostLightweight
)
    : learningRate(learningRate),
      momentumBeta(momentumBeta),
      fastBeta(fastBeta),
      slowBeta(slowBeta),
      epsilon(epsilon),
      scaleMin(scaleMin),
      scaleMax(scaleMax),
      weightDecay(weightDecay),
      coverage(coverage),
      hostLightweight(hostLightweight) {
    if (learningRate <= 0.0f) throw std::invalid_argument("CudaSpulse learningRate must be > 0");
    if (momentumBeta < 0.0f || momentumBeta >= 1.0f) throw std::invalid_argument("CudaSpulse momentumBeta must be in [0, 1)");
    if (fastBeta < 0.0f || fastBeta >= 1.0f) throw std::invalid_argument("CudaSpulse fastBeta must be in [0, 1)");
    if (slowBeta < 0.0f || slowBeta >= 1.0f) throw std::invalid_argument("CudaSpulse slowBeta must be in [0, 1)");
    if (slowBeta < fastBeta) throw std::invalid_argument("CudaSpulse slowBeta must be >= fastBeta");
    if (epsilon <= 0.0f) throw std::invalid_argument("CudaSpulse epsilon must be > 0");
    if (scaleMin <= 0.0f || scaleMax < scaleMin) throw std::invalid_argument("CudaSpulse invalid scale clip");
    if (weightDecay < 0.0f) throw std::invalid_argument("CudaSpulse weightDecay must be >= 0");
}

const char* CudaSpulse::coverageName(SpulseCoverage coverage) {
    switch (coverage) {
    case SpulseCoverage::Hybrid: return "Hybrid";
    case SpulseCoverage::Full: return "Full";
    }
    return "Unknown";
}

bool CudaSpulse::ownsHybridBlockWeights() const {
    return this->coverage == SpulseCoverage::Hybrid || this->coverage == SpulseCoverage::Full;
}

bool CudaSpulse::ownsFullModelWeights() const {
    return this->coverage == SpulseCoverage::Full;
}

void CudaSpulse::update(CudaMatrix& parameter, CudaSpulseState& state, const CudaMatrix& gradient, float gradientScale) {
    if (parameter.empty()) throw std::invalid_argument("CudaSpulse::update empty parameter");
    if (gradient.elementCount() != parameter.elementCount())
        throw std::invalid_argument("CudaSpulse::update gradient/parameter size mismatch");
    state.ensure(parameter);

    const int elementCount = static_cast<int>(parameter.elementCount());
    const int threads = 256;
    const int blocks = (std::min)(1024, elementwiseBlocks(elementCount, threads));
    const float oneMinusMom = 1.0f - this->momentumBeta;
    const float keep = 1.0f - this->learningRate * this->weightDecay;
    cudaStream_t stream = CudaMatmul::activeStream();

    this->sumSquaresScratch.ensureCapacity(sizeof(float));
    float* sumSquares = this->sumSquaresScratch.deviceData;
    CudaMatmul::memsetDevice(sumSquares, 0, sizeof(float));

    spulseFusedStep<<<blocks, threads, 0, stream>>>(
        parameter.buffer.deviceData,
        state.momentum.buffer.deviceData,
        gradient.buffer.deviceData,
        state.energy.deviceData,
        sumSquares,
        elementCount,
        this->momentumBeta,
        oneMinusMom,
        gradientScale,
        this->learningRate,
        keep);
    throwIfFailed(cudaGetLastError(), "spulseFusedStep");

    spulseCommitEnergyAndScale<<<1, 1, 0, stream>>>(
        state.energy.deviceData,
        sumSquares,
        this->fastBeta,
        this->slowBeta,
        1.0f - this->fastBeta,
        1.0f - this->slowBeta,
        this->epsilon,
        this->scaleMin,
        this->scaleMax);
    throwIfFailed(cudaGetLastError(), "spulseCommitEnergyAndScale");
}

void CudaSpulse::updateHost(Matrix& parameter, SpulseState& state, const Matrix& gradient, float gradientScale) const {
    if (parameter.empty()) throw std::invalid_argument("CudaSpulse::updateHost empty parameter");
    if (gradient.data.size() != parameter.data.size())
        throw std::invalid_argument("CudaSpulse::updateHost gradient/parameter size mismatch");
    state.ensure(parameter);
    hostUpdateCore(
        parameter.data.data(),
        state.momentum.data.data(),
        state.energyFast,
        state.energySlow,
        state.scale,
        gradient.data.data(),
        parameter.data.size(),
        this->learningRate,
        this->momentumBeta,
        this->fastBeta,
        this->slowBeta,
        this->epsilon,
        this->scaleMin,
        this->scaleMax,
        this->weightDecay,
        gradientScale);
}

void CudaSpulse::updateHostFromHalf(
    Matrix& parameter,
    SpulseState& state,
    const std::uint16_t* gradHalf,
    size_t elementCount,
    float gradientScale
) const {
    if (parameter.empty()) throw std::invalid_argument("CudaSpulse::updateHostFromHalf empty parameter");
    if (gradHalf == nullptr) throw std::invalid_argument("CudaSpulse::updateHostFromHalf null gradHalf");
    if (parameter.data.size() != elementCount)
        throw std::invalid_argument("CudaSpulse::updateHostFromHalf size mismatch");

    if (this->hostLightweight) {
        // Energy scalars only — do not allocate per-element momentum (HostSGD traffic).
        if (state.scale <= 0.0f)
            state.scale = 1.0f;
        hostUpdateFromHalfLite(
            parameter.data.data(),
            state.energyFast,
            state.energySlow,
            state.scale,
            reinterpret_cast<const __half*>(gradHalf),
            elementCount,
            this->learningRate,
            this->fastBeta,
            this->slowBeta,
            this->epsilon,
            this->scaleMin,
            this->scaleMax,
            this->weightDecay,
            gradientScale);
        return;
    }

    state.ensure(parameter);
    hostUpdateFromHalfCore(
        parameter.data.data(),
        state.momentum.data.data(),
        state.energyFast,
        state.energySlow,
        state.scale,
        reinterpret_cast<const __half*>(gradHalf),
        elementCount,
        this->learningRate,
        this->momentumBeta,
        this->fastBeta,
        this->slowBeta,
        this->epsilon,
        this->scaleMin,
        this->scaleMax,
        this->weightDecay,
        gradientScale);
}

void CudaSpulse::applyFusedHalfHostPieces(const std::vector<SpulseFusedHalfHostPiece>& pieces, float gradientScale) const {
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
#endif
    for (int pieceIndex = 0; pieceIndex < static_cast<int>(pieces.size()); ++pieceIndex) {
        const SpulseFusedHalfHostPiece& piece = pieces[static_cast<size_t>(pieceIndex)];
        if (piece.host == nullptr || piece.state == nullptr || piece.gradHalf == nullptr) continue;
        this->updateHostFromHalf(*piece.host, *piece.state, piece.gradHalf, piece.elementCount, gradientScale);
    }
}

void CudaSpulse::applyHybridBlockWeights(
    CudaTransformerBlock& block,
    CudaTransformerBlockGradients& blockGradients,
    CudaTransformerBlockSpulseStates& spulseStates,
    float gradientScale
) {
    if (!this->ownsHybridBlockWeights()) return;

    // Per-tensor updates (no QKV/gateUp fuse memcpy — that was Muon-only overhead).
    // Fused mirrors are refreshed by CudaLanguageModel::applyGradients → syncFusedMirrors.
    this->update(block.attention.queryWeight, spulseStates.queryWeight, blockGradients.queryWeight, gradientScale);
    this->update(block.attention.keyWeight, spulseStates.keyWeight, blockGradients.keyWeight, gradientScale);
    this->update(block.attention.valueWeight, spulseStates.valueWeight, blockGradients.valueWeight, gradientScale);
    this->update(block.attention.outputWeight, spulseStates.attentionOutputWeight, blockGradients.attentionOutputWeight, gradientScale);
    this->update(block.feedForward.gateWeight, spulseStates.feedForwardGateWeight, blockGradients.feedForwardGateWeight, gradientScale);
    this->update(block.feedForward.upWeight, spulseStates.feedForwardUpWeight, blockGradients.feedForwardUpWeight, gradientScale);
    this->update(block.feedForward.downWeight, spulseStates.feedForwardDownWeight, blockGradients.feedForwardDownWeight, gradientScale);
}

void CudaSpulse::runSmokeDemo(int parameterRows, int parameterCols) {
    if (!CudaMatmul::isAvailable()) {
        SmokeLog::skip("SPULSE");
        return;
    }
    if (parameterRows < 2 || parameterCols < 1)
        throw std::invalid_argument("CudaSpulse::runSmokeDemo invalid dims");

    Matrix hostParam(static_cast<size_t>(parameterRows), static_cast<size_t>(parameterCols), 0.0f);
    Matrix hostGrad(static_cast<size_t>(parameterRows), static_cast<size_t>(parameterCols), 0.0f);
    unsigned rng = 424242u;
    for (size_t index = 0; index < hostParam.data.size(); ++index) {
        rng = rng * 1664525u + 1013904223u;
        hostParam.data[index] = (static_cast<float>(rng >> 8) / 16777216.0f) * 2.0f - 1.0f;
        rng = rng * 1664525u + 1013904223u;
        hostGrad.data[index] = (static_cast<float>(rng >> 8) / 16777216.0f) * 0.02f - 0.01f;
    }

    CudaSpulse opt(1e-3f, 0.9f, 0.9f, 0.999f, 1e-8f, 0.25f, 4.0f, 0.0f, SpulseCoverage::Hybrid);
    Matrix hostParamRef = hostParam;
    SpulseState hostState;
    opt.updateHost(hostParamRef, hostState, hostGrad, 1.0f);

    CudaMatrix deviceParam;
    deviceParam.upload(hostParam);
    CudaMatrix deviceGrad;
    deviceGrad.upload(hostGrad);
    CudaSpulseState deviceState;
    opt.update(deviceParam, deviceState, deviceGrad, 1.0f);
    throwIfFailed(cudaDeviceSynchronize(), "CudaSpulse update synchronize");
    Matrix deviceDownloaded = deviceParam.download();

    float maxDiff = 0.0f;
    bool nonFinite = false;
    for (size_t index = 0; index < hostParamRef.data.size(); ++index) {
        if (!std::isfinite(hostParamRef.data[index]) || !std::isfinite(deviceDownloaded.data[index]))
            nonFinite = true;
        maxDiff = (std::max)(maxDiff, std::fabs(hostParamRef.data[index] - deviceDownloaded.data[index]));
    }

    SpulseState energyCheck;
    deviceState.downloadInto(energyCheck);
    const bool energyOk = std::isfinite(energyCheck.energyFast) && std::isfinite(energyCheck.energySlow)
        && energyCheck.energyFast >= 0.0f && energyCheck.energySlow >= 0.0f;

    SmokeLog::result(
        "SPULSE host↔GPU",
        "shape=%dx%d  maxDiff=%.3e  finite=%s  energyOk=%s  eFast=%.3e eSlow=%.3e",
        parameterRows,
        parameterCols,
        maxDiff,
        nonFinite ? "no" : "yes",
        energyOk ? "yes" : "no",
        energyCheck.energyFast,
        energyCheck.energySlow);

    if (nonFinite || !energyOk || maxDiff > 5e-4f)
        throw std::runtime_error("CudaSpulse::runSmokeDemo parity/finite check failed");
}
