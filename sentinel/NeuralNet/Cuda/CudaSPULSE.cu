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
    float keep,
    float momCorrection
) {
    __shared__ float shared[256];
    const float stepScale = energy[2] * momCorrection;
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

/// <summary>
/// Host-offload path: update <c>u</c> / energy on device and overwrite the grad buffer with
/// <c>delta = lr · lagged_scale · u</c> (masters live on host; θ is not updated here).
/// </summary>
__global__ void spulsePrepareHostDelta(
    float* gradientOrDelta,
    float* momentum,
    const float* energy,
    float* sumSquares,
    int elementCount,
    float momentumBeta,
    float oneMinusMom,
    float gradientScale,
    float learningRate,
    float momCorrection
) {
    __shared__ float shared[256];
    const float stepScale = energy[2] * momCorrection;
    float local = 0.0f;
    for (int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
         index < elementCount;
         index += static_cast<int>(blockDim.x * gridDim.x)) {
        const float g = gradientOrDelta[index] * gradientScale;
        const float m = momentumBeta * momentum[index] + oneMinusMom * g;
        momentum[index] = m;
        gradientOrDelta[index] = learningRate * stepScale * m;
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

__global__ void spulseFusedStepHalf(
    float* parameter,
    __half* momentum,
    const float* gradient,
    const float* energy,
    float* sumSquares,
    int elementCount,
    float momentumBeta,
    float oneMinusMom,
    float gradientScale,
    float learningRate,
    float keep,
    float momCorrection
) {
    __shared__ float shared[256];
    const float stepScale = energy[2] * momCorrection;
    float local = 0.0f;
    for (int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
         index < elementCount;
         index += static_cast<int>(blockDim.x * gridDim.x)) {
        const float g = gradient[index] * gradientScale;
        float m = momentumBeta * __half2float(momentum[index]) + oneMinusMom * g;
        if (!isfinite(m)) m = 0.0f;
        momentum[index] = __float2half_rn(m);
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

__global__ void spulsePrepareHostDeltaHalf(
    float* gradientOrDelta,
    __half* momentum,
    const float* energy,
    float* sumSquares,
    int elementCount,
    float momentumBeta,
    float oneMinusMom,
    float gradientScale,
    float learningRate,
    float momCorrection
) {
    __shared__ float shared[256];
    const float stepScale = energy[2] * momCorrection;
    float local = 0.0f;
    for (int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
         index < elementCount;
         index += static_cast<int>(blockDim.x * gridDim.x)) {
        const float g = gradientOrDelta[index] * gradientScale;
        float m = momentumBeta * __half2float(momentum[index]) + oneMinusMom * g;
        if (!isfinite(m)) m = 0.0f;
        momentum[index] = __float2half_rn(m);
        gradientOrDelta[index] = learningRate * stepScale * m;
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

/// <summary>One CUDA block = one absmax quant block. writeDelta=true → host-delta path (no θ update).</summary>
__global__ void spulseStepInt8(
    float* parameterOrDelta,
    signed char* momentumQ,
    float* momentumScales,
    const float* gradient,
    const float* energy,
    float* sumSquares,
    int elementCount,
    int blockSize,
    float momentumBeta,
    float oneMinusMom,
    float gradientScale,
    float learningRate,
    float keep,
    float momCorrection,
    int writeDelta
) {
    const int quantBlock = static_cast<int>(blockIdx.x);
    const int start = quantBlock * blockSize;
    if (start >= elementCount) return;
    const int count = (start + blockSize <= elementCount) ? blockSize : (elementCount - start);
    const int threadCount = static_cast<int>(blockDim.x);

    extern __shared__ float shared[];
    float* momFp = shared;
    float* reduceMax = shared + blockSize;
    float* reduceSum = reduceMax + threadCount;

    const float oldScale = momentumScales[quantBlock];
    const float stepScale = energy[2] * momCorrection;
    float localSum = 0.0f;

    for (int local = static_cast<int>(threadIdx.x); local < count; local += threadCount) {
        const int index = start + local;
        float m = (oldScale == 0.0f) ? 0.0f : static_cast<float>(momentumQ[index]) * oldScale;
        if (!isfinite(m)) m = 0.0f;
        const float g = (gradient != nullptr ? gradient[index] : parameterOrDelta[index]) * gradientScale;
        m = momentumBeta * m + oneMinusMom * g;
        if (!isfinite(m)) m = 0.0f;
        momFp[local] = m;
        localSum += g * g;

        if (writeDelta) {
            parameterOrDelta[index] = learningRate * stepScale * m;
        } else {
            float value = parameterOrDelta[index];
            if (keep != 1.0f)
                value *= keep;
            parameterOrDelta[index] = value - learningRate * stepScale * m;
        }
    }
    reduceSum[threadIdx.x] = localSum;

    float localMax = 0.0f;
    for (int local = static_cast<int>(threadIdx.x); local < count; local += threadCount)
        localMax = fmaxf(localMax, fabsf(momFp[local]));
    reduceMax[threadIdx.x] = localMax;
    __syncthreads();

    for (int stride = threadCount / 2; stride > 0; stride >>= 1) {
        if (static_cast<int>(threadIdx.x) < stride) {
            reduceMax[threadIdx.x] = fmaxf(reduceMax[threadIdx.x], reduceMax[threadIdx.x + stride]);
            reduceSum[threadIdx.x] += reduceSum[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(sumSquares, reduceSum[0]);
        const float absmax = reduceMax[0];
        const float newScale = (absmax > 0.0f) ? (absmax / 127.0f) : 0.0f;
        momentumScales[quantBlock] = newScale;
    }
    __syncthreads();

    const float newScale = momentumScales[quantBlock];
    for (int local = static_cast<int>(threadIdx.x); local < count; local += threadCount) {
        float quantized = (newScale == 0.0f) ? 0.0f : rintf(momFp[local] / newScale);
        quantized = fminf(127.0f, fmaxf(-127.0f, quantized));
        if (!isfinite(quantized)) quantized = 0.0f;
        momentumQ[start + local] = static_cast<signed char>(quantized);
    }
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
    float gradientScale,
    float momCorrection
) {
    const float oneMinusMom = 1.0f - momentumBeta;
    const float oneMinusFast = 1.0f - fastBeta;
    const float oneMinusSlow = 1.0f - slowBeta;
    const float stepScale = scale * momCorrection;
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
    float gradientScale,
    float momCorrection
) {
    const float oneMinusMom = 1.0f - momentumBeta;
    const float oneMinusFast = 1.0f - fastBeta;
    const float oneMinusSlow = 1.0f - slowBeta;
    const float stepScale = scale * momCorrection;
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
    return this->rows == 0 || this->cols == 0;
}

void CudaSpulseState::ensure(
    size_t rows,
    size_t cols,
    SpulseMomentumStorage storage,
    int int8BlockSize
) {
    if (rows == 0 || cols == 0) throw std::invalid_argument("CudaSpulseState::ensure empty shape");
    if (int8BlockSize <= 0) throw std::invalid_argument("CudaSpulseState::ensure int8BlockSize must be > 0");

    const size_t elementCount = rows * cols;
    const bool sameShape = this->rows == rows && this->cols == cols && this->energy.deviceData != nullptr
        && this->storage == storage
        && (storage != SpulseMomentumStorage::Int8 || this->int8BlockSize == int8BlockSize);
    if (sameShape) {
        if (storage == SpulseMomentumStorage::Fp32 && this->momentum.hasDeviceStorage()) return;
        if (storage == SpulseMomentumStorage::Fp16 && this->momentumHalf.deviceData != nullptr) return;
        if (storage == SpulseMomentumStorage::Int8 && this->momentumQ.deviceData != nullptr) return;
    }

    this->free();
    this->rows = rows;
    this->cols = cols;
    this->storage = storage;
    this->int8BlockSize = int8BlockSize;
    this->scaleCount = 0;

    if (storage == SpulseMomentumStorage::Fp32) {
        this->momentum.ensureSize(rows, cols);
        CudaOps::zeroInPlace(this->momentum);
    } else if (storage == SpulseMomentumStorage::Fp16) {
        this->momentumHalf.ensureCapacity(elementCount * sizeof(__half));
        throwIfFailed(
            cudaMemset(this->momentumHalf.deviceData, 0, elementCount * sizeof(__half)),
            "CudaSpulseState::ensure momentumHalf zero");
    } else {
        this->scaleCount = static_cast<int>((elementCount + static_cast<size_t>(int8BlockSize) - 1)
            / static_cast<size_t>(int8BlockSize));
        this->momentumQ.ensureCapacity(elementCount);
        this->momentumQ.zeroInPlace();
        this->momentumScales.ensureCapacity(static_cast<size_t>(this->scaleCount) * sizeof(float));
        throwIfFailed(
            cudaMemset(this->momentumScales.deviceData, 0, static_cast<size_t>(this->scaleCount) * sizeof(float)),
            "CudaSpulseState::ensure momentumScales zero");
    }

    this->energy.ensureCapacity(3 * sizeof(float));
    const float init[3] = {0.0f, 0.0f, 1.0f};
    throwIfFailed(
        cudaMemcpy(this->energy.deviceData, init, 3 * sizeof(float), cudaMemcpyHostToDevice),
        "CudaSpulseState::ensure energy init");
}

void CudaSpulseState::ensure(
    const CudaMatrix& parameter,
    SpulseMomentumStorage storage,
    int int8BlockSize
) {
    if (parameter.empty()) throw std::invalid_argument("CudaSpulseState::ensure empty parameter");
    this->ensure(parameter.rows, parameter.cols, storage, int8BlockSize);
}

void CudaSpulseState::free() {
    this->momentum.free();
    this->momentumHalf.free();
    this->momentumQ.free();
    this->momentumScales.free();
    this->energy.free();
    this->rows = 0;
    this->cols = 0;
    this->scaleCount = 0;
}

void CudaSpulseState::downloadInto(SpulseState& host) const {
    if (this->empty()) {
        host.clear();
        return;
    }
    host.momentum.ensureSize(this->rows, this->cols);
    const size_t elementCount = this->rows * this->cols;

    if (this->storage == SpulseMomentumStorage::Fp32) {
        this->momentum.downloadInto(host.momentum);
    } else if (this->storage == SpulseMomentumStorage::Fp16) {
        std::vector<__half> halfHost(elementCount);
        throwIfFailed(
            cudaMemcpy(halfHost.data(), this->momentumHalf.deviceData, elementCount * sizeof(__half), cudaMemcpyDeviceToHost),
            "CudaSpulseState::downloadInto momentumHalf");
        for (size_t i = 0; i < elementCount; ++i)
            host.momentum.data[i] = __half2float(halfHost[i]);
    } else {
        std::vector<signed char> qHost(elementCount);
        std::vector<float> scales(static_cast<size_t>(this->scaleCount), 0.0f);
        this->momentumQ.copyToHost(qHost.data(), elementCount);
        throwIfFailed(
            cudaMemcpy(scales.data(), this->momentumScales.deviceData, scales.size() * sizeof(float), cudaMemcpyDeviceToHost),
            "CudaSpulseState::downloadInto momentumScales");
        for (size_t i = 0; i < elementCount; ++i) {
            const int block = static_cast<int>(i / static_cast<size_t>(this->int8BlockSize));
            const float scale = scales[static_cast<size_t>(block)];
            host.momentum.data[i] = scale == 0.0f ? 0.0f : static_cast<float>(qHost[i]) * scale;
        }
    }

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
    this->ensure(host.momentum.rows, host.momentum.cols, this->storage, this->int8BlockSize);
    const size_t elementCount = host.momentum.data.size();

    if (this->storage == SpulseMomentumStorage::Fp32) {
        this->momentum.upload(host.momentum);
    } else if (this->storage == SpulseMomentumStorage::Fp16) {
        std::vector<__half> halfHost(elementCount);
        for (size_t i = 0; i < elementCount; ++i)
            halfHost[i] = __float2half_rn(host.momentum.data[i]);
        throwIfFailed(
            cudaMemcpy(this->momentumHalf.deviceData, halfHost.data(), elementCount * sizeof(__half), cudaMemcpyHostToDevice),
            "CudaSpulseState::uploadFrom momentumHalf");
    } else {
        std::vector<signed char> qHost(elementCount, 0);
        std::vector<float> scales(static_cast<size_t>(this->scaleCount), 0.0f);
        for (int block = 0; block < this->scaleCount; ++block) {
            const size_t start = static_cast<size_t>(block) * static_cast<size_t>(this->int8BlockSize);
            const size_t end = (std::min)(elementCount, start + static_cast<size_t>(this->int8BlockSize));
            float absmax = 0.0f;
            for (size_t i = start; i < end; ++i)
                absmax = (std::max)(absmax, std::fabs(host.momentum.data[i]));
            const float scale = absmax > 0.0f ? absmax / 127.0f : 0.0f;
            scales[static_cast<size_t>(block)] = scale;
            for (size_t i = start; i < end; ++i) {
                float q = scale == 0.0f ? 0.0f : std::round(host.momentum.data[i] / scale);
                q = (std::min)(127.0f, (std::max)(-127.0f, q));
                qHost[i] = static_cast<signed char>(q);
            }
        }
        this->momentumQ.copyFromHost(qHost.data(), elementCount);
        throwIfFailed(
            cudaMemcpy(this->momentumScales.deviceData, scales.data(), scales.size() * sizeof(float), cudaMemcpyHostToDevice),
            "CudaSpulseState::uploadFrom momentumScales");
    }

    this->energy.ensureCapacity(3 * sizeof(float));
    const float energyHost[3] = {host.energyFast, host.energySlow, host.scale};
    throwIfFailed(
        cudaMemcpy(this->energy.deviceData, energyHost, 3 * sizeof(float), cudaMemcpyHostToDevice),
        "CudaSpulseState::uploadFrom energy");
}

void CudaTransformerBlockSpulseStates::ensureFrom(
    const CudaTransformerBlock& block,
    SpulseMomentumStorage storage,
    int int8BlockSize
) {
    this->queryWeight.ensure(block.attention.queryWeight, storage, int8BlockSize);
    this->keyWeight.ensure(block.attention.keyWeight, storage, int8BlockSize);
    this->valueWeight.ensure(block.attention.valueWeight, storage, int8BlockSize);
    this->attentionOutputWeight.ensure(block.attention.outputWeight, storage, int8BlockSize);
    this->feedForwardGateWeight.ensure(block.feedForward.gateWeight, storage, int8BlockSize);
    this->feedForwardUpWeight.ensure(block.feedForward.upWeight, storage, int8BlockSize);
    this->feedForwardDownWeight.ensure(block.feedForward.downWeight, storage, int8BlockSize);
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
    bool hostLightweight,
    SpulseMomentumStorage momentumStorage,
    int int8BlockSize
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
      hostLightweight(hostLightweight),
      momentumStorage(momentumStorage),
      int8BlockSize(int8BlockSize),
      timeStep(0) {
    if (learningRate <= 0.0f) throw std::invalid_argument("CudaSpulse learningRate must be > 0");
    if (momentumBeta < 0.0f || momentumBeta >= 1.0f) throw std::invalid_argument("CudaSpulse momentumBeta must be in [0, 1)");
    if (fastBeta < 0.0f || fastBeta >= 1.0f) throw std::invalid_argument("CudaSpulse fastBeta must be in [0, 1)");
    if (slowBeta < 0.0f || slowBeta >= 1.0f) throw std::invalid_argument("CudaSpulse slowBeta must be in [0, 1)");
    if (slowBeta < fastBeta) throw std::invalid_argument("CudaSpulse slowBeta must be >= fastBeta");
    if (epsilon <= 0.0f) throw std::invalid_argument("CudaSpulse epsilon must be > 0");
    if (scaleMin <= 0.0f || scaleMax < scaleMin) throw std::invalid_argument("CudaSpulse invalid scale clip");
    if (weightDecay < 0.0f) throw std::invalid_argument("CudaSpulse weightDecay must be >= 0");
    if (int8BlockSize <= 0) throw std::invalid_argument("CudaSpulse int8BlockSize must be > 0");
}

void CudaSpulse::step() {
    ++this->timeStep;
}

float CudaSpulse::momentumBiasCorrection() const {
    if (this->timeStep <= 0) return 1.0f;
    const float oneMinusPow = 1.0f - std::pow(this->momentumBeta, static_cast<float>(this->timeStep));
    if (!(oneMinusPow > 0.0f)) return 1.0f;
    return 1.0f / oneMinusPow;
}

const char* CudaSpulse::coverageName(SpulseCoverage coverage) {
    switch (coverage) {
    case SpulseCoverage::Hybrid: return "Hybrid";
    case SpulseCoverage::Full: return "Full";
    }
    return "Unknown";
}

const char* CudaSpulse::momentumStorageName(SpulseMomentumStorage storage) {
    switch (storage) {
    case SpulseMomentumStorage::Fp32: return "Fp32";
    case SpulseMomentumStorage::Fp16: return "Fp16";
    case SpulseMomentumStorage::Int8: return "Int8";
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
    state.ensure(parameter, this->momentumStorage, this->int8BlockSize);

    const int elementCount = static_cast<int>(parameter.elementCount());
    const float oneMinusMom = 1.0f - this->momentumBeta;
    const float keep = 1.0f - this->learningRate * this->weightDecay;
    const float momCorrection = this->momentumBiasCorrection();
    cudaStream_t stream = CudaMatmul::activeStream();

    this->sumSquaresScratch.ensureCapacity(sizeof(float));
    float* sumSquares = this->sumSquaresScratch.deviceData;
    CudaMatmul::memsetDevice(sumSquares, 0, sizeof(float));

    if (state.storage == SpulseMomentumStorage::Fp32) {
        const int threads = 256;
        const int blocks = (std::min)(1024, elementwiseBlocks(elementCount, threads));
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
            keep,
            momCorrection);
        throwIfFailed(cudaGetLastError(), "spulseFusedStep");
    } else if (state.storage == SpulseMomentumStorage::Fp16) {
        const int threads = 256;
        const int blocks = (std::min)(1024, elementwiseBlocks(elementCount, threads));
        spulseFusedStepHalf<<<blocks, threads, 0, stream>>>(
            parameter.buffer.deviceData,
            reinterpret_cast<__half*>(state.momentumHalf.deviceData),
            gradient.buffer.deviceData,
            state.energy.deviceData,
            sumSquares,
            elementCount,
            this->momentumBeta,
            oneMinusMom,
            gradientScale,
            this->learningRate,
            keep,
            momCorrection);
        throwIfFailed(cudaGetLastError(), "spulseFusedStepHalf");
    } else {
        const int blockSize = state.int8BlockSize;
        const int threads = (std::min)(256, blockSize);
        const int blocks = state.scaleCount;
        const size_t sharedBytes = static_cast<size_t>(blockSize + 2 * threads) * sizeof(float);
        spulseStepInt8<<<blocks, threads, sharedBytes, stream>>>(
            parameter.buffer.deviceData,
            state.momentumQ.deviceData,
            state.momentumScales.deviceData,
            gradient.buffer.deviceData,
            state.energy.deviceData,
            sumSquares,
            elementCount,
            blockSize,
            this->momentumBeta,
            oneMinusMom,
            gradientScale,
            this->learningRate,
            keep,
            momCorrection,
            0);
        throwIfFailed(cudaGetLastError(), "spulseStepInt8");
    }

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

void CudaSpulse::prepareHostDeltaInPlace(
    float* gradientOrDelta,
    size_t rows,
    size_t cols,
    CudaSpulseState& state,
    float gradientScale
) {
    if (gradientOrDelta == nullptr) throw std::invalid_argument("CudaSpulse::prepareHostDeltaInPlace null buffer");
    if (rows == 0 || cols == 0) throw std::invalid_argument("CudaSpulse::prepareHostDeltaInPlace empty shape");
    state.ensure(rows, cols, this->momentumStorage, this->int8BlockSize);

    const int elementCount = static_cast<int>(rows * cols);
    const float oneMinusMom = 1.0f - this->momentumBeta;
    const float momCorrection = this->momentumBiasCorrection();
    cudaStream_t stream = CudaMatmul::activeStream();

    this->sumSquaresScratch.ensureCapacity(sizeof(float));
    float* sumSquares = this->sumSquaresScratch.deviceData;
    CudaMatmul::memsetDevice(sumSquares, 0, sizeof(float));

    if (state.storage == SpulseMomentumStorage::Fp32) {
        const int threads = 256;
        const int blocks = (std::min)(1024, elementwiseBlocks(elementCount, threads));
        spulsePrepareHostDelta<<<blocks, threads, 0, stream>>>(
            gradientOrDelta,
            state.momentum.buffer.deviceData,
            state.energy.deviceData,
            sumSquares,
            elementCount,
            this->momentumBeta,
            oneMinusMom,
            gradientScale,
            this->learningRate,
            momCorrection);
        throwIfFailed(cudaGetLastError(), "spulsePrepareHostDelta");
    } else if (state.storage == SpulseMomentumStorage::Fp16) {
        const int threads = 256;
        const int blocks = (std::min)(1024, elementwiseBlocks(elementCount, threads));
        spulsePrepareHostDeltaHalf<<<blocks, threads, 0, stream>>>(
            gradientOrDelta,
            reinterpret_cast<__half*>(state.momentumHalf.deviceData),
            state.energy.deviceData,
            sumSquares,
            elementCount,
            this->momentumBeta,
            oneMinusMom,
            gradientScale,
            this->learningRate,
            momCorrection);
        throwIfFailed(cudaGetLastError(), "spulsePrepareHostDeltaHalf");
    } else {
        const int blockSize = state.int8BlockSize;
        const int threads = (std::min)(256, blockSize);
        const int blocks = state.scaleCount;
        const size_t sharedBytes = static_cast<size_t>(blockSize + 2 * threads) * sizeof(float);
        spulseStepInt8<<<blocks, threads, sharedBytes, stream>>>(
            gradientOrDelta,
            state.momentumQ.deviceData,
            state.momentumScales.deviceData,
            nullptr,
            state.energy.deviceData,
            sumSquares,
            elementCount,
            blockSize,
            this->momentumBeta,
            oneMinusMom,
            gradientScale,
            this->learningRate,
            1.0f,
            momCorrection,
            1);
        throwIfFailed(cudaGetLastError(), "spulseStepInt8 (host delta)");
    }

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
    throwIfFailed(cudaGetLastError(), "spulseCommitEnergyAndScale (host delta)");
}

void CudaSpulse::prepareHybridBlockHostDeltas(
    CudaTransformerBlock& block,
    CudaTransformerBlockSpulseStates& spulseStates,
    float gradientScale
) {
    if (!this->ownsHybridBlockWeights()) return;

    // Same source buffers as CudaTransformerBlock::enqueueDeferredHostWeightGradDownloads.
    if (block.feedForwardDownWeightGradient.empty())
        throw std::logic_error("CudaSpulse::prepareHybridBlockHostDeltas empty down grad");
    this->prepareHostDeltaInPlace(
        block.feedForwardDownWeightGradient.buffer.deviceData,
        block.feedForwardDownWeightGradient.rows,
        block.feedForwardDownWeightGradient.cols,
        spulseStates.feedForwardDownWeight,
        gradientScale);

    if (!block.feedForward.gateUpWeightGradient.empty()) {
        const size_t gRows = block.feedForward.gateWeight.rows;
        const size_t gCols = block.feedForward.gateWeight.cols;
        const size_t gElems = block.feedForward.gateWeight.elementCount();
        const size_t uRows = block.feedForward.upWeight.rows;
        const size_t uCols = block.feedForward.upWeight.cols;
        this->prepareHostDeltaInPlace(
            block.feedForward.gateUpWeightGradient.buffer.deviceData,
            gRows,
            gCols,
            spulseStates.feedForwardGateWeight,
            gradientScale);
        this->prepareHostDeltaInPlace(
            block.feedForward.gateUpWeightGradient.buffer.deviceData + gElems,
            uRows,
            uCols,
            spulseStates.feedForwardUpWeight,
            gradientScale);
    } else {
        this->prepareHostDeltaInPlace(
            block.feedForwardGateWeightGradient.buffer.deviceData,
            block.feedForwardGateWeightGradient.rows,
            block.feedForwardGateWeightGradient.cols,
            spulseStates.feedForwardGateWeight,
            gradientScale);
        this->prepareHostDeltaInPlace(
            block.feedForwardUpWeightGradient.buffer.deviceData,
            block.feedForwardUpWeightGradient.rows,
            block.feedForwardUpWeightGradient.cols,
            spulseStates.feedForwardUpWeight,
            gradientScale);
    }

    this->prepareHostDeltaInPlace(
        block.attentionOutputWeightGradient.buffer.deviceData,
        block.attentionOutputWeightGradient.rows,
        block.attentionOutputWeightGradient.cols,
        spulseStates.attentionOutputWeight,
        gradientScale);

    if (!block.attention.qkvWeightGradient.empty()) {
        const size_t qRows = block.attention.queryWeight.rows;
        const size_t qCols = block.attention.queryWeight.cols;
        const size_t qElems = block.attention.queryWeight.elementCount();
        this->prepareHostDeltaInPlace(
            block.attention.qkvWeightGradient.buffer.deviceData,
            qRows,
            qCols,
            spulseStates.queryWeight,
            gradientScale);
        this->prepareHostDeltaInPlace(
            block.attention.qkvWeightGradient.buffer.deviceData + qElems,
            qRows,
            qCols,
            spulseStates.keyWeight,
            gradientScale);
        this->prepareHostDeltaInPlace(
            block.attention.qkvWeightGradient.buffer.deviceData + 2ull * qElems,
            qRows,
            qCols,
            spulseStates.valueWeight,
            gradientScale);
    } else {
        this->prepareHostDeltaInPlace(
            block.queryWeightGradient.buffer.deviceData,
            block.queryWeightGradient.rows,
            block.queryWeightGradient.cols,
            spulseStates.queryWeight,
            gradientScale);
        this->prepareHostDeltaInPlace(
            block.keyWeightGradient.buffer.deviceData,
            block.keyWeightGradient.rows,
            block.keyWeightGradient.cols,
            spulseStates.keyWeight,
            gradientScale);
        this->prepareHostDeltaInPlace(
            block.valueWeightGradient.buffer.deviceData,
            block.valueWeightGradient.rows,
            block.valueWeightGradient.cols,
            spulseStates.valueWeight,
            gradientScale);
    }
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
        gradientScale,
        this->momentumBiasCorrection());
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
        gradientScale,
        this->momentumBiasCorrection());
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

    auto runOnce = [&](SpulseMomentumStorage storage, float maxAllowedDiff, const char* label) {
        CudaSpulse opt(
            1e-3f, 0.9f, 0.9f, 0.999f, 1e-8f, 0.25f, 4.0f, 0.0f,
            SpulseCoverage::Hybrid, false, storage, 256);
        Matrix hostParamRef = hostParam;
        SpulseState hostState;
        // Host reference always uses FP32 u.
        CudaSpulse hostOpt(1e-3f, 0.9f, 0.9f, 0.999f, 1e-8f, 0.25f, 4.0f, 0.0f, SpulseCoverage::Hybrid);
        hostOpt.step();
        hostOpt.updateHost(hostParamRef, hostState, hostGrad, 1.0f);

        CudaMatrix deviceParam;
        deviceParam.upload(hostParam);
        CudaMatrix deviceGrad;
        deviceGrad.upload(hostGrad);
        CudaSpulseState deviceState;
        opt.step();
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
            label,
            "storage=%s  shape=%dx%d  maxDiff=%.3e  finite=%s  energyOk=%s",
            CudaSpulse::momentumStorageName(storage),
            parameterRows,
            parameterCols,
            maxDiff,
            nonFinite ? "no" : "yes",
            energyOk ? "yes" : "no");

        if (nonFinite || !energyOk || maxDiff > maxAllowedDiff)
            throw std::runtime_error(std::string("CudaSpulse::runSmokeDemo failed: ") + label);
    };

    runOnce(SpulseMomentumStorage::Fp32, 5e-4f, "SPULSE host↔GPU");
    runOnce(SpulseMomentumStorage::Fp16, 2e-3f, "SPULSE Fp16-u");
    runOnce(SpulseMomentumStorage::Int8, 5e-2f, "SPULSE Int8-u");
}
