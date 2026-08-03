#include "CudaMuon.hpp"

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

float frobeniusNorm(const CudaMatrix& matrix, CudaDeviceBuffer& scratch) {
    scratch.ensureCapacity(sizeof(float));
    CudaMatmul::memsetDevice(scratch.deviceData, 0, sizeof(float));
    const int elementCount = static_cast<int>(matrix.elementCount());
    const int threads = 256;
    const int blocks = (std::min)(1024, elementwiseBlocks(elementCount, threads));
    muonSumSquares<<<blocks, threads, 0, CudaMatmul::activeStream()>>>(
        matrix.buffer.deviceData, elementCount, scratch.deviceData);
    throwIfFailed(cudaGetLastError(), "muonSumSquares");
    if (CudaMatmul::activeStream() != nullptr)
        throwIfFailed(cudaStreamSynchronize(CudaMatmul::activeStream()), "muon frobenius sync");
    else
        throwIfFailed(cudaDeviceSynchronize(), "muon frobenius sync");
    float sumSquares = 0.0f;
    throwIfFailed(
        cudaMemcpy(&sumSquares, scratch.deviceData, sizeof(float), cudaMemcpyDeviceToHost),
        "muon frobenius D2H");
    return std::sqrt(sumSquares);
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
      adjustLrMatchRmsAdamw(adjustLrMatchRmsAdamw) {
    if (learningRate <= 0.0f) throw std::invalid_argument("CudaMuon learningRate must be > 0");
    if (momentumBeta < 0.0f || momentumBeta >= 1.0f) throw std::invalid_argument("CudaMuon momentumBeta must be in [0, 1)");
    if (weightDecay < 0.0f) throw std::invalid_argument("CudaMuon weightDecay must be >= 0");
    if (nsSteps <= 0) throw std::invalid_argument("CudaMuon nsSteps must be > 0");
}

void CudaMuon::newtonSchulz5InPlace(CudaMatrix& matrix) {
    if (matrix.empty()) throw std::invalid_argument("CudaMuon::newtonSchulz5InPlace empty matrix");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaMuon::newtonSchulz5InPlace no CUDA device");

    const bool transposeBack = matrix.rows > matrix.cols;
    CudaMatrix* working = &matrix;
    if (transposeBack) {
        transposeInto(matrix, this->nsWork);
        working = &this->nsWork;
    }

    const float norm = frobeniusNorm(*working, this->frobeniusScratch);
    const int elementCount = static_cast<int>(working->elementCount());
    muonScale<<<elementwiseBlocks(elementCount), 256, 0, CudaMatmul::activeStream()>>>(
        working->buffer.deviceData, elementCount, 1.0f / (norm + kNsEps));
    throwIfFailed(cudaGetLastError(), "muonScale normalize");

    this->nsX.ensureSize(working->rows, working->cols);
    CudaOps::copyInto(*working, this->nsX);

    for (int step = 0; step < this->nsSteps; ++step) {
        this->nsA.ensureSize(this->nsX.rows, this->nsX.rows);
        CudaMatrix::multiplyInto(this->nsX, this->nsX, this->nsA, false, true);

        this->nsAA.ensureSize(this->nsA.rows, this->nsA.cols);
        CudaMatrix::multiplyInto(this->nsA, this->nsA, this->nsAA, false, false);

        this->nsB.ensureSize(this->nsA.rows, this->nsA.cols);
        CudaOps::copyInto(this->nsA, this->nsB);
        CudaOps::scaleInPlace(this->nsB, kNsB);
        const int bCount = static_cast<int>(this->nsB.elementCount());
        muonAxpy<<<elementwiseBlocks(bCount), 256, 0, CudaMatmul::activeStream()>>>(
            this->nsB.buffer.deviceData, this->nsAA.buffer.deviceData, bCount, kNsC);
        throwIfFailed(cudaGetLastError(), "muonAxpy B");

        this->nsBX.ensureSize(this->nsX.rows, this->nsX.cols);
        CudaMatrix::multiplyInto(this->nsB, this->nsX, this->nsBX, false, false);

        CudaOps::scaleInPlace(this->nsX, kNsA);
        CudaOps::addInPlace(this->nsX, this->nsBX);
    }

    if (transposeBack)
        transposeInto(this->nsX, matrix);
    else
        CudaOps::copyInto(this->nsX, matrix);
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

    this->newtonSchulz5InPlace(this->updateScratch);

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
