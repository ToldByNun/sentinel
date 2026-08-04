#include "CudaMuon.hpp"

#include "CudaAmp.hpp"
#include "CudaOps.hpp"
#include "../Utils/SmokeLog.hpp"

#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr float kNsA = 3.4445f;
constexpr float kNsB = -4.7750f;
constexpr float kNsC = 2.0315f;
constexpr float kNsEps = 1e-7f;

void throwIfFailed(cudaError_t status, const char* what) {
    if (status == cudaSuccess) return;
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
}

int elementwiseBlocks(int elementCount, int threads = 256) {
    return (elementCount + threads - 1) / threads;
}

__global__ void muonLerpMomentum(
    float* momentum,
    const float* gradient,
    int elementCount,
    float beta,
    float oneMinusBeta,
    float gradientScale
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elementCount) return;
    const float g = gradient[index] * gradientScale;
    momentum[index] = beta * momentum[index] + oneMinusBeta * g;
}

__global__ void muonNesterovUpdate(
    float* update,
    const float* momentum,
    const float* gradient,
    int elementCount,
    float beta,
    float oneMinusBeta,
    float gradientScale
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elementCount) return;
    const float g = gradient[index] * gradientScale;
    update[index] = oneMinusBeta * g + beta * momentum[index];
}

__global__ void muonCopy(float* destination, const float* source, int elementCount) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elementCount) return;
    destination[index] = source[index];
}

__global__ void muonSumSquares(const float* data, int elementCount, float* outSum) {
    __shared__ float shared[256];
    float local = 0.0f;
    for (int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
         index < elementCount;
         index += static_cast<int>(blockDim.x * gridDim.x)) {
        const float value = data[index];
        local += value * value;
    }
    shared[threadIdx.x] = local;
    __syncthreads();
    for (int stride = static_cast<int>(blockDim.x) / 2; stride > 0; stride >>= 1) {
        if (static_cast<int>(threadIdx.x) < stride)
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        atomicAdd(outSum, shared[0]);
}

__global__ void muonScale(float* data, int elementCount, float scalar) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elementCount) return;
    data[index] *= scalar;
}

/// <summary>invScale = 1 / (sqrt(sumSquares) + eps); runs as 1 thread</summary>
__global__ void muonInvFrobeniusFromSumSquares(const float* sumSquares, float* invScale) {
    *invScale = 1.0f / (sqrtf(*sumSquares) + kNsEps);
}

__global__ void muonScaleByDeviceScalar(float* data, int elementCount, const float* scalar) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elementCount) return;
    data[index] *= *scalar;
}

__global__ void muonPolyCombine(float* out, const float* a, const float* aa, int elementCount, float scaleA, float scaleAA) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elementCount) return;
    out[index] = scaleA * a[index] + scaleAA * aa[index];
}

__global__ void muonScaleAddInPlace(float* total, const float* delta, int elementCount, float scaleTotal) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elementCount) return;
    total[index] = scaleTotal * total[index] + delta[index];
}

__global__ void muonAxpy(float* total, const float* delta, int elementCount, float alpha) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elementCount) return;
    total[index] += alpha * delta[index];
}

__global__ void muonDecay(float* parameter, int elementCount, float keep) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elementCount) return;
    parameter[index] *= keep;
}

__global__ void muonTranspose(const float* source, float* destination, int sourceRows, int sourceCols) {
    const int col = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int row = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (row >= sourceRows || col >= sourceCols) return;
    destination[col * sourceRows + row] = source[row * sourceCols + col];
}

void transposeInto(const CudaMatrix& source, CudaMatrix& destination) {
    destination.ensureSize(source.cols, source.rows);
    const dim3 block(16, 16);
    const dim3 grid(
        static_cast<unsigned>((source.cols + block.x - 1) / block.x),
        static_cast<unsigned>((source.rows + block.y - 1) / block.y));
    muonTranspose<<<grid, block, 0, CudaMatmul::activeStream()>>>(
        source.buffer.deviceData,
        destination.buffer.deviceData,
        static_cast<int>(source.rows),
        static_cast<int>(source.cols));
    throwIfFailed(cudaGetLastError(), "muonTranspose");
}

/// <summary>
/// Frobenius-normalize on device without host sync:
/// sumSquares → invScale → scale. Same CUDA stream keeps ordering.
/// frobeniusScratch layout: [0]=sumSquares, [1]=invScale.
/// </summary>
void normalizeFrobeniusInPlace(CudaMatrix& matrix, CudaDeviceBuffer& frobeniusScratch) {
    frobeniusScratch.ensureCapacity(2 * sizeof(float));
    float* sumSquares = frobeniusScratch.deviceData;
    float* invScale = frobeniusScratch.deviceData + 1;
    CudaMatmul::memsetDevice(sumSquares, 0, sizeof(float));

    const int elementCount = static_cast<int>(matrix.elementCount());
    const int threads = 256;
    const int blocks = (std::min)(1024, elementwiseBlocks(elementCount, threads));
    muonSumSquares<<<blocks, threads, 0, CudaMatmul::activeStream()>>>(
        matrix.buffer.deviceData, elementCount, sumSquares);
    throwIfFailed(cudaGetLastError(), "muonSumSquares");

    muonInvFrobeniusFromSumSquares<<<1, 1, 0, CudaMatmul::activeStream()>>>(sumSquares, invScale);
    throwIfFailed(cudaGetLastError(), "muonInvFrobeniusFromSumSquares");

    muonScaleByDeviceScalar<<<elementwiseBlocks(elementCount, threads), threads, 0, CudaMatmul::activeStream()>>>(
        matrix.buffer.deviceData, elementCount, invScale);
    throwIfFailed(cudaGetLastError(), "muonScaleByDeviceScalar");
}

Matrix hostNewtonSchulz5(Matrix matrix, int steps) {
    const bool transposeBack = matrix.rows > matrix.cols;
    if (transposeBack)
        matrix = Matrix::transpose(matrix);

    float sumSquares = 0.0f;
    for (float value : matrix.data)
        sumSquares += value * value;
    Matrix::scaleInPlace(matrix, 1.0f / (std::sqrt(sumSquares) + kNsEps));

    for (int step = 0; step < steps; ++step) {
        Matrix a = Matrix::multiply(matrix, matrix, false, true);
        Matrix aa = Matrix::multiply(a, a);
        Matrix b = Matrix::add(Matrix::scale(a, kNsB), Matrix::scale(aa, kNsC));
        Matrix bx = Matrix::multiply(b, matrix);
        matrix = Matrix::add(Matrix::scale(matrix, kNsA), bx);
    }

    if (transposeBack)
        matrix = Matrix::transpose(matrix);
    return matrix;
}

} // namespace

bool CudaMuonState::empty() const {
    return this->momentum.empty();
}

void CudaMuonState::ensure(const CudaMatrix& parameter) {
    if (parameter.empty()) throw std::invalid_argument("CudaMuonState::ensure empty parameter");
    if (!this->momentum.empty()
        && this->momentum.rows == parameter.rows
        && this->momentum.cols == parameter.cols)
        return;
    this->momentum.ensureSize(parameter.rows, parameter.cols);
    CudaOps::zeroInPlace(this->momentum);
}

void CudaMuonState::free() {
    this->momentum.free();
}

void CudaMuonState::downloadInto(MuonState& host) const {
    if (this->momentum.empty()) {
        host.momentum = Matrix();
        return;
    }
    this->momentum.downloadInto(host.momentum);
}

void CudaMuonState::uploadFrom(const MuonState& host) {
    if (host.momentum.empty()) {
        this->free();
        return;
    }
    this->momentum.upload(host.momentum);
}

CudaMuon::CudaMuon(
    float learningRate,
    float momentumBeta,
    float weightDecay,
    int nsSteps,
    bool nesterov,
    bool adjustLrMatchRmsAdamw
)
    : learningRate(learningRate),
      momentumBeta(momentumBeta),
      weightDecay(weightDecay),
      nsSteps(nsSteps),
      nesterov(nesterov),
      adjustLrMatchRmsAdamw(adjustLrMatchRmsAdamw),
      profileEnabled(false) {
    if (learningRate <= 0.0f) throw std::invalid_argument("CudaMuon learningRate must be > 0");
    if (momentumBeta < 0.0f || momentumBeta >= 1.0f) throw std::invalid_argument("CudaMuon momentumBeta must be in [0, 1)");
    if (weightDecay < 0.0f) throw std::invalid_argument("CudaMuon weightDecay must be >= 0");
    if (nsSteps <= 0) throw std::invalid_argument("CudaMuon nsSteps must be > 0");
}

namespace {

struct ScopedCudaEventPair {
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    bool active = false;

    void begin(bool enabled) {
        active = enabled;
        if (!active) return;
        throwIfFailed(cudaEventCreate(&start), "cudaEventCreate muon start");
        throwIfFailed(cudaEventCreate(&stop), "cudaEventCreate muon stop");
        throwIfFailed(cudaEventRecord(start, CudaMatmul::activeStream()), "cudaEventRecord muon start");
    }

    double endMs() {
        if (!active) return 0.0;
        throwIfFailed(cudaEventRecord(stop, CudaMatmul::activeStream()), "cudaEventRecord muon stop");
        throwIfFailed(cudaEventSynchronize(stop), "cudaEventSynchronize muon stop");
        float ms = 0.0f;
        throwIfFailed(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime muon");
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        start = nullptr;
        stop = nullptr;
        active = false;
        return static_cast<double>(ms);
    }
};

} // namespace

void CudaMuon::newtonSchulz5InPlace(CudaMatrix& matrix) {
    if (matrix.empty()) throw std::invalid_argument("CudaMuon::newtonSchulz5InPlace empty matrix");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaMuon::newtonSchulz5InPlace no CUDA device");

    // NS intermediates are not AMP master weights — FP16 path would re-cast every GEMM.
    // Reference Muon orthogonalizes in FP32/TF32; keep that here for speed and stability.
    const bool previousAmp = CudaAmp::preferMixedPrecision;
    CudaAmp::preferMixedPrecision = false;

    ScopedCudaEventPair normalizeTimer;
    ScopedCudaEventPair nsGemmTimer;
    ScopedCudaEventPair nsElemwiseTimer;

    const bool transposeBack = matrix.rows > matrix.cols;
    CudaMatrix* working = &matrix;
    if (transposeBack) {
        this->nsX.ensureSize(matrix.cols, matrix.rows);
        transposeInto(matrix, this->nsX);
        working = &this->nsX;
    }

    normalizeTimer.begin(this->profileEnabled);
    normalizeFrobeniusInPlace(*working, this->frobeniusScratch);
    this->profile.normalizeMs += normalizeTimer.endMs();

    for (int step = 0; step < this->nsSteps; ++step) {
        this->nsA.ensureSize(working->rows, working->rows);
        nsGemmTimer.begin(this->profileEnabled);
        CudaMatrix::multiplyInto(*working, *working, this->nsA, false, true);

        this->nsAA.ensureSize(this->nsA.rows, this->nsA.cols);
        CudaMatrix::multiplyInto(this->nsA, this->nsA, this->nsAA, false, false);
        this->profile.nsGemmMs += nsGemmTimer.endMs();

        this->nsB.ensureSize(this->nsA.rows, this->nsA.cols);
        const int bCount = static_cast<int>(this->nsB.elementCount());
        nsElemwiseTimer.begin(this->profileEnabled);
        muonPolyCombine<<<elementwiseBlocks(bCount), 256, 0, CudaMatmul::activeStream()>>>(
            this->nsB.buffer.deviceData,
            this->nsA.buffer.deviceData,
            this->nsAA.buffer.deviceData,
            bCount,
            kNsB,
            kNsC);
        throwIfFailed(cudaGetLastError(), "muonPolyCombine B");

        this->nsBX.ensureSize(working->rows, working->cols);
        nsGemmTimer.begin(this->profileEnabled);
        CudaMatrix::multiplyInto(this->nsB, *working, this->nsBX, false, false);
        this->profile.nsGemmMs += nsGemmTimer.endMs();

        const int xCount = static_cast<int>(working->elementCount());
        muonScaleAddInPlace<<<elementwiseBlocks(xCount), 256, 0, CudaMatmul::activeStream()>>>(
            working->buffer.deviceData,
            this->nsBX.buffer.deviceData,
            xCount,
            kNsA);
        throwIfFailed(cudaGetLastError(), "muonScaleAddInPlace X");
        this->profile.nsElemwiseMs += nsElemwiseTimer.endMs();
    }

    if (transposeBack)
        transposeInto(*working, matrix);

    CudaAmp::preferMixedPrecision = previousAmp;
}

void CudaMuon::update(CudaMatrix& parameter, CudaMuonState& state, const CudaMatrix& gradient, float gradientScale) {
    if (parameter.empty()) throw std::invalid_argument("CudaMuon::update empty parameter");
    if (gradient.elementCount() != parameter.elementCount())
        throw std::invalid_argument("CudaMuon::update gradient/parameter size mismatch");
    if (parameter.rows < 2)
        throw std::invalid_argument("CudaMuon::update requires rows >= 2");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaMuon::update no CUDA device");

    state.ensure(parameter);
    const int elementCount = static_cast<int>(parameter.elementCount());
    const float beta = this->momentumBeta;
    const float oneMinusBeta = 1.0f - beta;
    const int blocks = elementwiseBlocks(elementCount);

    {
        ScopedCudaEventPair timer;
        timer.begin(this->profileEnabled);
        muonLerpMomentum<<<blocks, 256, 0, CudaMatmul::activeStream()>>>(
            state.momentum.buffer.deviceData,
            gradient.buffer.deviceData,
            elementCount,
            beta,
            oneMinusBeta,
            gradientScale);
        throwIfFailed(cudaGetLastError(), "muonLerpMomentum");

        this->updateScratch.ensureSize(parameter.rows, parameter.cols);
        if (this->nesterov) {
            muonNesterovUpdate<<<blocks, 256, 0, CudaMatmul::activeStream()>>>(
                this->updateScratch.buffer.deviceData,
                state.momentum.buffer.deviceData,
                gradient.buffer.deviceData,
                elementCount,
                beta,
                oneMinusBeta,
                gradientScale);
            throwIfFailed(cudaGetLastError(), "muonNesterovUpdate");
        } else {
            muonCopy<<<blocks, 256, 0, CudaMatmul::activeStream()>>>(
                this->updateScratch.buffer.deviceData,
                state.momentum.buffer.deviceData,
                elementCount);
            throwIfFailed(cudaGetLastError(), "muonCopy");
        }
        this->profile.momentumMs += timer.endMs();
    }

    this->newtonSchulz5InPlace(this->updateScratch);

    {
        ScopedCudaEventPair timer;
        timer.begin(this->profileEnabled);
        float stepLr = this->learningRate;
        if (this->adjustLrMatchRmsAdamw) {
            const float shape = static_cast<float>((std::max)(parameter.rows, parameter.cols));
            stepLr *= 0.2f * std::sqrt(shape);
        } else {
            const float aspect = static_cast<float>(parameter.rows) / static_cast<float>((std::max)(size_t{1}, parameter.cols));
            muonScale<<<blocks, 256, 0, CudaMatmul::activeStream()>>>(
                this->updateScratch.buffer.deviceData, elementCount, std::sqrt((std::max)(1.0f, aspect)));
            throwIfFailed(cudaGetLastError(), "muonScale spectral");
        }

        if (this->weightDecay != 0.0f) {
            muonDecay<<<blocks, 256, 0, CudaMatmul::activeStream()>>>(
                parameter.buffer.deviceData, elementCount, 1.0f - stepLr * this->weightDecay);
            throwIfFailed(cudaGetLastError(), "muonDecay");
        }

        muonAxpy<<<blocks, 256, 0, CudaMatmul::activeStream()>>>(
            parameter.buffer.deviceData,
            this->updateScratch.buffer.deviceData,
            elementCount,
            -stepLr);
        throwIfFailed(cudaGetLastError(), "muonAxpy param");
        this->profile.applyMs += timer.endMs();
    }

    if (this->profileEnabled)
        ++this->profile.updateCount;
}

void CudaMuon::runSmokeDemo(int parameterRows, int parameterCols) {
    if (!CudaMatmul::isAvailable()) {
        SmokeLog::skip("Muon NS");
        return;
    }
    if (parameterRows < 2 || parameterCols < 1)
        throw std::invalid_argument("CudaMuon::runSmokeDemo invalid dims");

    Matrix host(static_cast<size_t>(parameterRows), static_cast<size_t>(parameterCols), 0.0f);
    unsigned rng = 271u;
    for (size_t index = 0; index < host.data.size(); ++index) {
        rng = rng * 1664525u + 1013904223u;
        host.data[index] = (static_cast<float>(rng >> 8) / 16777216.0f) * 2.0f - 1.0f;
    }

    Matrix hostNs = hostNewtonSchulz5(host, 5);

    CudaMatrix device;
    device.upload(host);
    CudaMuon muon(0.02f, 0.95f, 0.0f, 5, true, false);
    muon.newtonSchulz5InPlace(device);
    throwIfFailed(cudaDeviceSynchronize(), "CudaMuon NS synchronize");
    Matrix deviceNs = device.download();

    float maxNsDiff = 0.0f;
    bool nsNonFinite = false;
    for (size_t index = 0; index < hostNs.data.size(); ++index) {
        if (!std::isfinite(hostNs.data[index]) || !std::isfinite(deviceNs.data[index]))
            nsNonFinite = true;
        maxNsDiff = (std::max)(maxNsDiff, std::fabs(hostNs.data[index] - deviceNs.data[index]));
    }

    Matrix hostGrad = host;
    for (float& value : hostGrad.data)
        value *= 0.01f;

    CudaMatrix param;
    param.upload(host);
    CudaMatrix grad;
    grad.upload(hostGrad);
    CudaMuonState state;
    muon.adjustLrMatchRmsAdamw = true;
    muon.learningRate = 0.001f;
    muon.update(param, state, grad, 1.0f);
    throwIfFailed(cudaDeviceSynchronize(), "CudaMuon update synchronize");

    Matrix updated = param.download();
    bool stepNonFinite = false;
    float maxAbs = 0.0f;
    for (float value : updated.data) {
        if (!std::isfinite(value)) stepNonFinite = true;
        maxAbs = (std::max)(maxAbs, std::fabs(value));
    }

    SmokeLog::result(
        "Muon NS",
        "rows=%d cols=%d  hostVsGpu=%.2e  nsFinite=%s  stepFinite=%s maxAbs=%.2e",
        parameterRows,
        parameterCols,
        maxNsDiff,
        nsNonFinite ? "no" : "yes",
        stepNonFinite ? "no" : "yes",
        maxAbs);
}
