#include "CudaAdam.hpp"

#include "CudaAmp.hpp"
#include "CudaOps.hpp"
#include "../Utils/SmokeLog.hpp"

#include "../Optimizers/Adam.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

bool CudaAdam::preferInt8Moments = true;
bool CudaAdam::preferCpuOffload = false;
bool CudaAdam::preferFp16GpuWeights = false;
bool CudaAdam::preferHostGradients = false;
bool CudaAdam::preferHostSgd = false;
int CudaAdam::int8BlockSize = 256;

namespace {
void throwIfCudaFailedPinned(cudaError_t status, const char* what) {
    if (status == cudaSuccess) return;
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
}

/// <summary>pinned host arena + dedicated copy stream for CPU Adam offload</summary>
struct CudaAdamOffloadArena {
    float* parameters = nullptr;
    float* gradients = nullptr;
    size_t capacityElements = 0;
    cudaStream_t stream = nullptr;

    ~CudaAdamOffloadArena() {
        if (this->stream != nullptr) {
            cudaStreamSynchronize(this->stream);
            cudaStreamDestroy(this->stream);
            this->stream = nullptr;
        }
        if (this->parameters != nullptr) cudaFreeHost(this->parameters);
        if (this->gradients != nullptr) cudaFreeHost(this->gradients);
        this->parameters = nullptr;
        this->gradients = nullptr;
        this->capacityElements = 0;
    }

    void ensureStream() {
        if (this->stream != nullptr) return;
        throwIfCudaFailedPinned(cudaStreamCreateWithFlags(&this->stream, cudaStreamNonBlocking), "CudaAdamOffloadArena create stream");
    }

    void ensure(size_t elementCount) {
        this->ensureStream();
        if (elementCount <= this->capacityElements) return;
        if (this->parameters != nullptr) {
            throwIfCudaFailedPinned(cudaFreeHost(this->parameters), "CudaAdamOffloadArena free parameters");
            this->parameters = nullptr;
        }
        if (this->gradients != nullptr) {
            throwIfCudaFailedPinned(cudaFreeHost(this->gradients), "CudaAdamOffloadArena free gradients");
            this->gradients = nullptr;
        }
        this->capacityElements = 0;
        throwIfCudaFailedPinned(
            cudaMallocHost(reinterpret_cast<void**>(&this->parameters), elementCount * sizeof(float)),
            "CudaAdamOffloadArena malloc parameters");
        throwIfCudaFailedPinned(
            cudaMallocHost(reinterpret_cast<void**>(&this->gradients), elementCount * sizeof(float)),
            "CudaAdamOffloadArena malloc gradients");
        this->capacityElements = elementCount;
    }
};

thread_local CudaAdamOffloadArena gCudaAdamOffloadArena;
}

CudaAdamState::CudaAdamState() : elementCount(0), scaleCount(0) {}

bool CudaAdamState::empty() const {
    return this->firstMoment.empty() && this->firstMomentQ.deviceData == nullptr;
}

void CudaAdamState::ensureFp32(const CudaMatrix& parameter) {
    if (!this->firstMoment.empty()) return;
    this->firstMomentQ.free();
    this->secondMomentQ.free();
    this->firstMomentScales.free();
    this->secondMomentScales.free();
    this->scaleCount = 0;
    this->elementCount = static_cast<int>(parameter.elementCount());
    this->firstMoment.ensureSize(parameter.rows, parameter.cols);
    this->secondMoment.ensureSize(parameter.rows, parameter.cols);
    CudaOps::zeroInPlace(this->firstMoment);
    CudaOps::zeroInPlace(this->secondMoment);
}

void CudaAdamState::ensureInt8(const CudaMatrix& parameter, int blockSize) {
    if (this->firstMomentQ.deviceData != nullptr) return;
    if (blockSize <= 0) throw std::invalid_argument("CudaAdamState::ensureInt8 blockSize must be > 0");

    this->firstMoment.free();
    this->secondMoment.free();
    this->elementCount = static_cast<int>(parameter.elementCount());
    this->scaleCount = (this->elementCount + blockSize - 1) / blockSize;
    this->firstMomentQ.ensureCapacity(static_cast<size_t>(this->elementCount));
    this->secondMomentQ.ensureCapacity(static_cast<size_t>(this->elementCount));
    this->firstMomentScales.ensureCapacity(static_cast<size_t>(this->scaleCount) * sizeof(float));
    this->secondMomentScales.ensureCapacity(static_cast<size_t>(this->scaleCount) * sizeof(float));
    this->firstMomentQ.zeroInPlace();
    this->secondMomentQ.zeroInPlace();
    CudaMatmul::throwIfCudaFailed(cudaMemset(this->firstMomentScales.deviceData, 0, static_cast<size_t>(this->scaleCount) * sizeof(float)), "CudaAdamState::ensureInt8 clear m scales");
    CudaMatmul::throwIfCudaFailed(cudaMemset(this->secondMomentScales.deviceData, 0, static_cast<size_t>(this->scaleCount) * sizeof(float)), "CudaAdamState::ensureInt8 clear v scales");
}

CudaAdamState CudaAdamState::zerosLike(const CudaMatrix& parameter) {
    CudaAdamState state;
    if (CudaAdam::preferInt8Moments)
        state.ensureInt8(parameter, CudaAdam::int8BlockSize);
    else
        state.ensureFp32(parameter);
    return state;
}

void CudaAdamState::free() {
    this->firstMoment.free();
    this->secondMoment.free();
    this->firstMomentQ.free();
    this->secondMomentQ.free();
    this->firstMomentScales.free();
    this->secondMomentScales.free();
    this->elementCount = 0;
    this->scaleCount = 0;
}

void CudaAdamState::downloadInto(AdamState& host, size_t rows, size_t cols) const {
    if (rows == 0 || cols == 0) throw std::invalid_argument("CudaAdamState::downloadInto empty shape");
    host.firstMoment.ensureSize(rows, cols);
    host.secondMoment.ensureSize(rows, cols);

    if (!this->firstMoment.empty()) {
        if (this->firstMoment.rows != rows || this->firstMoment.cols != cols)
            throw std::invalid_argument("CudaAdamState::downloadInto FP32 shape mismatch");
        this->firstMoment.downloadInto(host.firstMoment);
        this->secondMoment.downloadInto(host.secondMoment);
        return;
    }

    if (this->firstMomentQ.deviceData == nullptr)
        throw std::logic_error("CudaAdamState::downloadInto empty moments");
    if (this->elementCount != static_cast<int>(rows * cols))
        throw std::invalid_argument("CudaAdamState::downloadInto int8 elementCount mismatch");
    if (this->scaleCount <= 0) throw std::logic_error("CudaAdamState::downloadInto empty scales");

    const int blockSize = CudaAdam::int8BlockSize;
    std::vector<signed char> firstQ(static_cast<size_t>(this->elementCount));
    std::vector<signed char> secondQ(static_cast<size_t>(this->elementCount));
    std::vector<float> firstScales(static_cast<size_t>(this->scaleCount));
    std::vector<float> secondScales(static_cast<size_t>(this->scaleCount));
    this->firstMomentQ.copyToHost(firstQ.data(), static_cast<size_t>(this->elementCount));
    this->secondMomentQ.copyToHost(secondQ.data(), static_cast<size_t>(this->elementCount));
    CudaMatmul::throwIfCudaFailed(
        cudaMemcpy(firstScales.data(), this->firstMomentScales.deviceData, static_cast<size_t>(this->scaleCount) * sizeof(float), cudaMemcpyDeviceToHost),
        "CudaAdamState::downloadInto first scales");
    CudaMatmul::throwIfCudaFailed(
        cudaMemcpy(secondScales.data(), this->secondMomentScales.deviceData, static_cast<size_t>(this->scaleCount) * sizeof(float), cudaMemcpyDeviceToHost),
        "CudaAdamState::downloadInto second scales");

    for (int index = 0; index < this->elementCount; ++index) {
        const int blockIndex = index / blockSize;
        host.firstMoment.data[static_cast<size_t>(index)] = static_cast<float>(firstQ[static_cast<size_t>(index)]) * firstScales[static_cast<size_t>(blockIndex)];
        host.secondMoment.data[static_cast<size_t>(index)] = static_cast<float>(secondQ[static_cast<size_t>(index)]) * secondScales[static_cast<size_t>(blockIndex)];
    }
}

void CudaAdamState::uploadFrom(const AdamState& host) {
    if (host.firstMoment.empty() || host.secondMoment.empty())
        throw std::invalid_argument("CudaAdamState::uploadFrom empty host moments");
    if (host.firstMoment.rows != host.secondMoment.rows || host.firstMoment.cols != host.secondMoment.cols)
        throw std::invalid_argument("CudaAdamState::uploadFrom host moment shape mismatch");

    CudaMatrix shape;
    shape.rows = host.firstMoment.rows;
    shape.cols = host.firstMoment.cols;

    if (CudaAdam::preferInt8Moments) {
        this->firstMoment.free();
        this->secondMoment.free();
        this->firstMomentQ.free();
        this->secondMomentQ.free();
        this->firstMomentScales.free();
        this->secondMomentScales.free();
        this->ensureInt8(shape, CudaAdam::int8BlockSize);

        const int blockSize = CudaAdam::int8BlockSize;
        std::vector<signed char> firstQ(static_cast<size_t>(this->elementCount));
        std::vector<signed char> secondQ(static_cast<size_t>(this->elementCount));
        std::vector<float> firstScales(static_cast<size_t>(this->scaleCount), 0.0f);
        std::vector<float> secondScales(static_cast<size_t>(this->scaleCount), 0.0f);

        for (int blockIndex = 0; blockIndex < this->scaleCount; ++blockIndex) {
            const int start = blockIndex * blockSize;
            const int count = (std::min)(blockSize, this->elementCount - start);
            float maxFirst = 0.0f;
            float maxSecond = 0.0f;
            for (int local = 0; local < count; ++local) {
                maxFirst = (std::max)(maxFirst, std::fabs(host.firstMoment.data[static_cast<size_t>(start + local)]));
                maxSecond = (std::max)(maxSecond, std::fabs(host.secondMoment.data[static_cast<size_t>(start + local)]));
            }
            maxFirst = (std::max)(maxFirst, 1e-8f);
            maxSecond = (std::max)(maxSecond, 1e-8f);
            firstScales[static_cast<size_t>(blockIndex)] = maxFirst / 127.0f;
            secondScales[static_cast<size_t>(blockIndex)] = maxSecond / 127.0f;
            for (int local = 0; local < count; ++local) {
                float qFirst = std::round(host.firstMoment.data[static_cast<size_t>(start + local)] / firstScales[static_cast<size_t>(blockIndex)]);
                qFirst = (std::min)(127.0f, (std::max)(-127.0f, qFirst));
                firstQ[static_cast<size_t>(start + local)] = static_cast<signed char>(qFirst);

                float qSecond = std::round(host.secondMoment.data[static_cast<size_t>(start + local)] / secondScales[static_cast<size_t>(blockIndex)]);
                qSecond = (std::min)(127.0f, (std::max)(0.0f, qSecond));
                secondQ[static_cast<size_t>(start + local)] = static_cast<signed char>(qSecond);
            }
        }

        this->firstMomentQ.copyFromHost(firstQ.data(), static_cast<size_t>(this->elementCount));
        this->secondMomentQ.copyFromHost(secondQ.data(), static_cast<size_t>(this->elementCount));
        CudaMatmul::throwIfCudaFailed(
            cudaMemcpy(this->firstMomentScales.deviceData, firstScales.data(), static_cast<size_t>(this->scaleCount) * sizeof(float), cudaMemcpyHostToDevice),
            "CudaAdamState::uploadFrom first scales");
        CudaMatmul::throwIfCudaFailed(
            cudaMemcpy(this->secondMomentScales.deviceData, secondScales.data(), static_cast<size_t>(this->scaleCount) * sizeof(float), cudaMemcpyHostToDevice),
            "CudaAdamState::uploadFrom second scales");
        return;
    }

    this->firstMomentQ.free();
    this->secondMomentQ.free();
    this->firstMomentScales.free();
    this->secondMomentScales.free();
    this->scaleCount = 0;
    this->elementCount = static_cast<int>(host.firstMoment.data.size());
    this->firstMoment.upload(host.firstMoment);
    this->secondMoment.upload(host.secondMoment);
}

CudaAdam::CudaAdam(float learningRate, float beta1, float beta2, float epsilon)
    : learningRate(learningRate), beta1(beta1), beta2(beta2), epsilon(epsilon), timeStep(0) {
    if (learningRate <= 0.0f) throw std::invalid_argument("CudaAdam learningRate must be > 0");
    if (beta1 < 0.0f || beta1 >= 1.0f) throw std::invalid_argument("CudaAdam beta1 must be in [0, 1)");
    if (beta2 < 0.0f || beta2 >= 1.0f) throw std::invalid_argument("CudaAdam beta2 must be in [0, 1)");
    if (epsilon <= 0.0f) throw std::invalid_argument("CudaAdam epsilon must be > 0");
}

void CudaAdam::step() {
    ++this->timeStep;
}

__device__ void CudaAdam::runUpdate(float* parameter, float* firstMoment, float* secondMoment, const float* gradient, int elementCount, float learningRate, float beta1, float beta2, float epsilon, float inverseFirstCorrection, float inverseSecondCorrection, float gradientScale) {
    const int index = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (index >= elementCount) return;

    const float gradientValue = gradient[index] * gradientScale;
    if (!isfinite(gradientValue)) return;

    float updatedFirstMoment = beta1 * firstMoment[index] + (1.0f - beta1) * gradientValue;
    float updatedSecondMoment = beta2 * secondMoment[index] + (1.0f - beta2) * gradientValue * gradientValue;
    if (!isfinite(updatedFirstMoment) || !isfinite(updatedSecondMoment)) return;

    firstMoment[index] = updatedFirstMoment;
    secondMoment[index] = updatedSecondMoment;

    const float correctedFirst = updatedFirstMoment * inverseFirstCorrection;
    const float correctedSecond = updatedSecondMoment * inverseSecondCorrection;
    const float delta = learningRate * correctedFirst / (sqrtf(correctedSecond) + epsilon);
    if (!isfinite(delta)) return;
    parameter[index] -= delta;
}

__device__ void CudaAdam::runUpdateMany(const CudaAdamUpdateItem* items, int itemCount, float learningRate, float beta1, float beta2, float epsilon, float inverseFirstCorrection, float inverseSecondCorrection, float gradientScale) {
    const int tensorIndex = static_cast<int>(blockIdx.x);
    if (tensorIndex >= itemCount) return;

    const CudaAdamUpdateItem item = items[tensorIndex];
    for (int index = static_cast<int>(threadIdx.x); index < item.elementCount; index += static_cast<int>(blockDim.x)) {
        const float gradientValue = item.gradient[index] * gradientScale;
        if (!isfinite(gradientValue)) continue;

        float updatedFirstMoment = beta1 * item.firstMoment[index] + (1.0f - beta1) * gradientValue;
        float updatedSecondMoment = beta2 * item.secondMoment[index] + (1.0f - beta2) * gradientValue * gradientValue;
        if (!isfinite(updatedFirstMoment) || !isfinite(updatedSecondMoment)) continue;

        item.firstMoment[index] = updatedFirstMoment;
        item.secondMoment[index] = updatedSecondMoment;

        const float correctedFirst = updatedFirstMoment * inverseFirstCorrection;
        const float correctedSecond = updatedSecondMoment * inverseSecondCorrection;
        const float delta = learningRate * correctedFirst / (sqrtf(correctedSecond) + epsilon);
        if (!isfinite(delta)) continue;
        item.parameter[index] -= delta;
    }
}

__global__ void CudaAdamUpdateEntry(float* parameter, float* firstMoment, float* secondMoment, const float* gradient, int elementCount, float learningRate, float beta1, float beta2, float epsilon, float inverseFirstCorrection, float inverseSecondCorrection, float gradientScale) {
    CudaAdam::runUpdate(parameter, firstMoment, secondMoment, gradient, elementCount, learningRate, beta1, beta2, epsilon, inverseFirstCorrection, inverseSecondCorrection, gradientScale);
}

__global__ void CudaAdamUpdateManyEntry(const CudaAdamUpdateItem* items, int itemCount, float learningRate, float beta1, float beta2, float epsilon, float inverseFirstCorrection, float inverseSecondCorrection, float gradientScale) {
    CudaAdam::runUpdateMany(items, itemCount, learningRate, beta1, beta2, epsilon, inverseFirstCorrection, inverseSecondCorrection, gradientScale);
}

__global__ void CudaAdamUpdateInt8TensorEntry(
    float* parameter,
    signed char* firstMomentQ,
    signed char* secondMomentQ,
    float* firstMomentScales,
    float* secondMomentScales,
    const float* gradient,
    int elementCount,
    int blockSize,
    float learningRate,
    float beta1,
    float beta2,
    float epsilon,
    float inverseFirstCorrection,
    float inverseSecondCorrection,
    float gradientScale) {
    const int quantBlock = static_cast<int>(blockIdx.x);
    const int start = quantBlock * blockSize;
    if (start >= elementCount) return;
    const int count = (start + blockSize <= elementCount) ? blockSize : (elementCount - start);
    const int threadCount = static_cast<int>(blockDim.x);

    // dynamic shared only: first[blockSize] second[blockSize] reduceFirst[threads] reduceSecond[threads]
    extern __shared__ float shared[];
    float* firstFp = shared;
    float* secondFp = shared + blockSize;
    float* reduceFirst = shared + 2 * blockSize;
    float* reduceSecond = reduceFirst + threadCount;

    const float oldFirstScale = firstMomentScales[quantBlock];
    const float oldSecondScale = secondMomentScales[quantBlock];

    for (int local = static_cast<int>(threadIdx.x); local < count; local += threadCount) {
        const int index = start + local;
        float first = (oldFirstScale == 0.0f) ? 0.0f : static_cast<float>(firstMomentQ[index]) * oldFirstScale;
        float second = (oldSecondScale == 0.0f) ? 0.0f : static_cast<float>(secondMomentQ[index]) * oldSecondScale;
        const float gradientValue = gradient[index] * gradientScale;
        if (!isfinite(gradientValue)) {
            firstFp[local] = isfinite(first) ? first : 0.0f;
            secondFp[local] = (isfinite(second) && second > 0.0f) ? second : 0.0f;
            continue;
        }
        if (!isfinite(first)) first = 0.0f;
        if (!isfinite(second) || second < 0.0f) second = 0.0f;

        first = beta1 * first + (1.0f - beta1) * gradientValue;
        second = beta2 * second + (1.0f - beta2) * gradientValue * gradientValue;
        if (!isfinite(first)) first = 0.0f;
        if (!isfinite(second) || second < 0.0f) second = 0.0f;
        firstFp[local] = first;
        secondFp[local] = second;

        const float correctedFirst = first * inverseFirstCorrection;
        // floor v-hat so int8 underflow cannot explode updates via 1/sqrt(v)
        const float correctedSecond = fmaxf(second * inverseSecondCorrection, epsilon * epsilon);
        float delta = learningRate * correctedFirst / (sqrtf(correctedSecond) + epsilon);
        if (!isfinite(delta)) continue;
        // keep bad blocks from taking SGD-sized leaps (lr=1e-3 → max |delta|=1e-2)
        const float maxDelta = learningRate * 10.0f;
        if (delta > maxDelta) delta = maxDelta;
        if (delta < -maxDelta) delta = -maxDelta;
        parameter[index] -= delta;
    }
    __syncthreads();

    float localMaxFirst = 0.0f;
    float localMaxSecond = 0.0f;
    for (int local = static_cast<int>(threadIdx.x); local < count; local += threadCount) {
        localMaxFirst = fmaxf(localMaxFirst, fabsf(firstFp[local]));
        localMaxSecond = fmaxf(localMaxSecond, fabsf(secondFp[local]));
    }
    reduceFirst[threadIdx.x] = localMaxFirst;
    reduceSecond[threadIdx.x] = localMaxSecond;
    __syncthreads();

    for (int stride = threadCount / 2; stride > 0; stride >>= 1) {
        if (static_cast<int>(threadIdx.x) < stride) {
            reduceFirst[threadIdx.x] = fmaxf(reduceFirst[threadIdx.x], reduceFirst[threadIdx.x + stride]);
            reduceSecond[threadIdx.x] = fmaxf(reduceSecond[threadIdx.x], reduceSecond[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float maxFirst = reduceFirst[0];
        float maxSecond = reduceSecond[0];
        if (!isfinite(maxFirst) || maxFirst < 0.0f) maxFirst = 0.0f;
        if (!isfinite(maxSecond) || maxSecond < 0.0f) maxSecond = 0.0f;
        // keep a usable quantum so tiny EMA moments (common in LM grads) do not flush to zero
        const float minAbsMax = 1e-8f;
        maxFirst = fmaxf(maxFirst, minAbsMax);
        maxSecond = fmaxf(maxSecond, minAbsMax);
        if (maxFirst > 1e8f) maxFirst = 1e8f;
        if (maxSecond > 1e8f) maxSecond = 1e8f;
        firstMomentScales[quantBlock] = maxFirst / 127.0f;
        secondMomentScales[quantBlock] = maxSecond / 127.0f;
    }
    __syncthreads();

    const float newFirstScale = firstMomentScales[quantBlock];
    const float newSecondScale = secondMomentScales[quantBlock];
    for (int local = static_cast<int>(threadIdx.x); local < count; local += threadCount) {
        float quantizedFirst = rintf(firstFp[local] / newFirstScale);
        quantizedFirst = fminf(127.0f, fmaxf(-127.0f, quantizedFirst));
        if (!isfinite(quantizedFirst)) quantizedFirst = 0.0f;
        firstMomentQ[start + local] = static_cast<signed char>(quantizedFirst);

        float quantizedSecond = rintf(secondFp[local] / newSecondScale);
        quantizedSecond = fminf(127.0f, fmaxf(0.0f, quantizedSecond));
        if (!isfinite(quantizedSecond)) quantizedSecond = 0.0f;
        secondMomentQ[start + local] = static_cast<signed char>(quantizedSecond);
    }
}

void CudaAdam::update(CudaMatrix& parameter, CudaAdamState& state, const CudaMatrix& gradient, float gradientScale) const {
    if (CudaAdam::preferInt8Moments)
        state.ensureInt8(parameter, CudaAdam::int8BlockSize);
    else
        state.ensureFp32(parameter);

    CudaAdamUpdateItem item;
    item.parameter = parameter.buffer.deviceData;
    item.gradient = gradient.buffer.deviceData;
    item.elementCount = static_cast<int>(parameter.elementCount());
    item.useInt8 = CudaAdam::preferInt8Moments;
    if (item.useInt8) {
        item.firstMoment = nullptr;
        item.secondMoment = nullptr;
        item.firstMomentQ = state.firstMomentQ.deviceData;
        item.secondMomentQ = state.secondMomentQ.deviceData;
        item.firstMomentScales = state.firstMomentScales.deviceData;
        item.secondMomentScales = state.secondMomentScales.deviceData;
        item.scaleCount = state.scaleCount;
    } else {
        item.firstMoment = state.firstMoment.buffer.deviceData;
        item.secondMoment = state.secondMoment.buffer.deviceData;
        item.firstMomentQ = nullptr;
        item.secondMomentQ = nullptr;
        item.firstMomentScales = nullptr;
        item.secondMomentScales = nullptr;
        item.scaleCount = 0;
    }
    this->updateMany(&item, 1, gradientScale);
}

void CudaAdam::updateCpuOffloaded(CudaMatrix& parameter, AdamState& hostState, const CudaMatrix& gradient, float gradientScale) const {
    CudaAdamCpuOffloadItem item;
    item.parameter = &parameter;
    item.gradient = &gradient;
    item.hostState = &hostState;
    this->updateCpuOffloadedMany(&item, 1, gradientScale);
}

void CudaAdam::updateCpuOffloadedMany(const CudaAdamCpuOffloadItem* items, int itemCount, float gradientScale) const {
    if (this->timeStep <= 0) throw std::invalid_argument("CudaAdam::updateCpuOffloadedMany requires step() before update");
    if (items == nullptr || itemCount <= 0) throw std::invalid_argument("CudaAdam::updateCpuOffloadedMany empty items");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaAdam::updateCpuOffloadedMany no CUDA device");

    const bool fp16Working = CudaAdam::preferFp16GpuWeights;

    // FP16 working weights + host masters: update in-place on host, no pinned arena
    // (arena would be ~2x param bytes and OOMs at ~4B on typical machines).
    if (fp16Working) {
        Adam hostAdam(this->learningRate, this->beta1, this->beta2, this->epsilon);
        hostAdam.timeStep = this->timeStep;
        gCudaAdamOffloadArena.ensureStream();
        cudaStream_t stream = gCudaAdamOffloadArena.stream;
        const cudaStream_t previousStream = CudaMatmul::setActiveStream(stream);

        auto restoreStream = [&]() {
            CudaMatmul::setActiveStream(previousStream);
        };

        try {
            // Validate + optional D2H for device grads first (keeps OpenMP SGD simple).
            std::vector<Matrix> deviceGradHosts(static_cast<size_t>(itemCount));
            for (int itemIndex = 0; itemIndex < itemCount; ++itemIndex) {
                const CudaAdamCpuOffloadItem& item = items[itemIndex];
                if (item.parameter == nullptr || item.hostState == nullptr)
                    throw std::invalid_argument("CudaAdam::updateCpuOffloadedMany null item pointer");
                if (item.parameter->empty())
                    throw std::invalid_argument("CudaAdam::updateCpuOffloadedMany empty parameter");
                if (item.hostMaster == nullptr)
                    throw std::invalid_argument("CudaAdam::updateCpuOffloadedMany fp16 mode requires hostMaster");
                if (item.hostMaster->empty() || item.hostMaster->data.size() != item.parameter->elementCount())
                    throw std::invalid_argument("CudaAdam::updateCpuOffloadedMany hostMaster size mismatch");

                if (item.hostGradient != nullptr) {
                    if (item.hostGradient->empty() || item.hostGradient->data.size() != item.parameter->elementCount())
                        throw std::invalid_argument("CudaAdam::updateCpuOffloadedMany hostGradient size mismatch");
                    if (item.hostGradient->rows != item.parameter->rows || item.hostGradient->cols != item.parameter->cols)
                        throw std::invalid_argument("CudaAdam::updateCpuOffloadedMany hostGradient shape mismatch");
                } else {
                    if (item.gradient == nullptr || item.gradient->empty() || item.gradient->buffer.deviceData == nullptr)
                        throw std::invalid_argument("CudaAdam::updateCpuOffloadedMany empty device gradient");
                    if (item.parameter->rows != item.gradient->rows || item.parameter->cols != item.gradient->cols)
                        throw std::invalid_argument("CudaAdam::updateCpuOffloadedMany parameter/gradient shape mismatch");
                    deviceGradHosts[static_cast<size_t>(itemIndex)].ensureSize(item.parameter->rows, item.parameter->cols);
                    throwIfCudaFailedPinned(
                        cudaMemcpyAsync(
                            deviceGradHosts[static_cast<size_t>(itemIndex)].data.data(),
                            item.gradient->buffer.deviceData,
                            item.parameter->byteCount(),
                            cudaMemcpyDeviceToHost,
                            stream),
                        "CudaAdam::updateCpuOffloadedMany D2H gradient (fp16)");
                }
            }
            throwIfCudaFailedPinned(cudaStreamSynchronize(stream), "CudaAdam::updateCpuOffloadedMany D2H sync");

            const float stepScale = this->learningRate * gradientScale;
            if (CudaAdam::preferHostSgd) {
                for (int itemIndex = 0; itemIndex < itemCount; ++itemIndex) {
                    const CudaAdamCpuOffloadItem& item = items[itemIndex];
                    float* masterData = item.hostMaster->data.data();
                    const float* gradData = item.hostGradient != nullptr
                        ? item.hostGradient->data.data()
                        : deviceGradHosts[static_cast<size_t>(itemIndex)].data.data();
                    const ptrdiff_t n = static_cast<ptrdiff_t>(item.hostMaster->data.size());
#if defined(_OPENMP)
                    #pragma omp parallel for schedule(static)
#endif
                    for (ptrdiff_t index = 0; index < n; ++index)
                        masterData[index] -= stepScale * gradData[index];
                }
            } else {
                for (int itemIndex = 0; itemIndex < itemCount; ++itemIndex) {
                    const CudaAdamCpuOffloadItem& item = items[itemIndex];
                    Matrix* gradientForUpdate = item.hostGradient;
                    if (gradientForUpdate == nullptr)
                        gradientForUpdate = &deviceGradHosts[static_cast<size_t>(itemIndex)];
                    if (gradientScale != 1.0f && item.hostGradient != nullptr) {
                        Matrix scaled = *item.hostGradient;
                        Matrix::scaleInPlace(scaled, gradientScale);
                        hostAdam.update(*item.hostMaster, *item.hostState, scaled);
                    } else if (gradientScale != 1.0f) {
                        Matrix::scaleInPlace(*gradientForUpdate, gradientScale);
                        hostAdam.update(*item.hostMaster, *item.hostState, *gradientForUpdate);
                    } else {
                        hostAdam.update(*item.hostMaster, *item.hostState, *gradientForUpdate);
                    }
                }
            }

            for (int itemIndex = 0; itemIndex < itemCount; ++itemIndex) {
                const CudaAdamCpuOffloadItem& item = items[itemIndex];
                if (item.parameter->ampWeightSlot >= 0)
                    CudaAmp::uploadHostMasterToFp16Working(*item.parameter, item.hostMaster->data.data());
                else if (item.parameter->hasDeviceStorage()) {
                    throwIfCudaFailedPinned(
                        cudaMemcpyAsync(
                            item.parameter->buffer.deviceData,
                            item.hostMaster->data.data(),
                            item.parameter->byteCount(),
                            cudaMemcpyHostToDevice,
                            stream),
                        "CudaAdam::updateCpuOffloadedMany H2D fp32 parameter");
                }
            }
            throwIfCudaFailedPinned(cudaStreamSynchronize(stream), "CudaAdam::updateCpuOffloadedMany H2D sync");
        } catch (...) {
            restoreStream();
            throw;
        }
        restoreStream();
        return;
    }

    std::vector<size_t> offsets(static_cast<size_t>(itemCount));
    std::vector<size_t> elementCounts(static_cast<size_t>(itemCount));
    size_t totalElements = 0;
    for (int itemIndex = 0; itemIndex < itemCount; ++itemIndex) {
        const CudaAdamCpuOffloadItem& item = items[itemIndex];
        if (item.parameter == nullptr || item.hostState == nullptr)
            throw std::invalid_argument("CudaAdam::updateCpuOffloadedMany null item pointer");
        if (item.parameter->empty())
            throw std::invalid_argument("CudaAdam::updateCpuOffloadedMany empty parameter");

        const bool useHostGrad = item.hostGradient != nullptr;
        if (useHostGrad) {
            if (item.hostGradient->empty() || item.hostGradient->data.size() != item.parameter->elementCount())
                throw std::invalid_argument("CudaAdam::updateCpuOffloadedMany hostGradient size mismatch");
            if (item.hostGradient->rows != item.parameter->rows || item.hostGradient->cols != item.parameter->cols)
                throw std::invalid_argument("CudaAdam::updateCpuOffloadedMany hostGradient shape mismatch");
        } else {
            if (item.gradient == nullptr || item.gradient->empty())
                throw std::invalid_argument("CudaAdam::updateCpuOffloadedMany empty parameter/gradient");
            if (item.parameter->rows != item.gradient->rows || item.parameter->cols != item.gradient->cols)
                throw std::invalid_argument("CudaAdam::updateCpuOffloadedMany parameter/gradient shape mismatch");
            if (item.gradient->buffer.deviceData == nullptr)
                throw std::invalid_argument("CudaAdam::updateCpuOffloadedMany null gradient device pointer");
        }
        if (item.parameter->buffer.deviceData == nullptr)
            throw std::invalid_argument("CudaAdam::updateCpuOffloadedMany null parameter device pointer");

        const size_t elementCount = item.parameter->elementCount();
        offsets[static_cast<size_t>(itemIndex)] = totalElements;
        elementCounts[static_cast<size_t>(itemIndex)] = elementCount;
        totalElements += elementCount;
    }

    gCudaAdamOffloadArena.ensure(totalElements);
    cudaStream_t stream = gCudaAdamOffloadArena.stream;

    for (int itemIndex = 0; itemIndex < itemCount; ++itemIndex) {
        const CudaAdamCpuOffloadItem& item = items[itemIndex];
        const size_t offset = offsets[static_cast<size_t>(itemIndex)];
        const size_t bytes = elementCounts[static_cast<size_t>(itemIndex)] * sizeof(float);
        throwIfCudaFailedPinned(
            cudaMemcpyAsync(
                gCudaAdamOffloadArena.parameters + offset,
                item.parameter->buffer.deviceData,
                bytes,
                cudaMemcpyDeviceToHost,
                stream),
            "CudaAdam::updateCpuOffloadedMany D2H parameter");
        if (item.hostGradient != nullptr) {
            std::memcpy(gCudaAdamOffloadArena.gradients + offset, item.hostGradient->data.data(), bytes);
        } else {
            throwIfCudaFailedPinned(
                cudaMemcpyAsync(
                    gCudaAdamOffloadArena.gradients + offset,
                    item.gradient->buffer.deviceData,
                    bytes,
                    cudaMemcpyDeviceToHost,
                    stream),
                "CudaAdam::updateCpuOffloadedMany D2H gradient");
        }
    }
    throwIfCudaFailedPinned(cudaStreamSynchronize(stream), "CudaAdam::updateCpuOffloadedMany D2H sync");

    Adam hostAdam(this->learningRate, this->beta1, this->beta2, this->epsilon);
    hostAdam.timeStep = this->timeStep;

#if defined(_OPENMP)
    #pragma omp parallel for schedule(dynamic, 1)
#endif
    for (int itemIndex = 0; itemIndex < itemCount; ++itemIndex) {
        const CudaAdamCpuOffloadItem& item = items[itemIndex];
        const size_t offset = offsets[static_cast<size_t>(itemIndex)];
        const size_t elementCount = elementCounts[static_cast<size_t>(itemIndex)];
        const size_t bytes = elementCount * sizeof(float);

        Matrix hostGradient(item.parameter->rows, item.parameter->cols);
        std::memcpy(hostGradient.data.data(), gCudaAdamOffloadArena.gradients + offset, bytes);
        if (gradientScale != 1.0f)
            Matrix::scaleInPlace(hostGradient, gradientScale);

        Matrix hostParameter(item.parameter->rows, item.parameter->cols);
        std::memcpy(hostParameter.data.data(), gCudaAdamOffloadArena.parameters + offset, bytes);
        hostAdam.update(hostParameter, *item.hostState, hostGradient);
        std::memcpy(gCudaAdamOffloadArena.parameters + offset, hostParameter.data.data(), bytes);
    }

    for (int itemIndex = 0; itemIndex < itemCount; ++itemIndex) {
        const CudaAdamCpuOffloadItem& item = items[itemIndex];
        const size_t offset = offsets[static_cast<size_t>(itemIndex)];
        const size_t bytes = elementCounts[static_cast<size_t>(itemIndex)] * sizeof(float);
        throwIfCudaFailedPinned(
            cudaMemcpyAsync(
                item.parameter->buffer.deviceData,
                gCudaAdamOffloadArena.parameters + offset,
                bytes,
                cudaMemcpyHostToDevice,
                stream),
            "CudaAdam::updateCpuOffloadedMany H2D parameter");
    }
    throwIfCudaFailedPinned(cudaStreamSynchronize(stream), "CudaAdam::updateCpuOffloadedMany H2D sync");
}

void CudaAdam::updateMany(const CudaAdamUpdateItem* items, int itemCount, float gradientScale) const {
    if (this->timeStep <= 0) throw std::invalid_argument("CudaAdam::updateMany requires step() before update");
    if (items == nullptr || itemCount <= 0) throw std::invalid_argument("CudaAdam::updateMany empty items");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaAdam::updateMany no CUDA device");

    bool anyInt8 = false;
    bool anyFp32 = false;
    for (int itemIndex = 0; itemIndex < itemCount; ++itemIndex) {
        const CudaAdamUpdateItem& item = items[itemIndex];
        if (item.parameter == nullptr || item.gradient == nullptr)
            throw std::invalid_argument("CudaAdam::updateMany null pointer");
        if (item.elementCount <= 0) throw std::invalid_argument("CudaAdam::updateMany invalid elementCount");
        if (item.useInt8) {
            anyInt8 = true;
            if (item.firstMomentQ == nullptr || item.secondMomentQ == nullptr || item.firstMomentScales == nullptr || item.secondMomentScales == nullptr)
                throw std::invalid_argument("CudaAdam::updateMany null int8 moments");
            if (item.scaleCount <= 0) throw std::invalid_argument("CudaAdam::updateMany invalid scaleCount");
        } else {
            anyFp32 = true;
            if (item.firstMoment == nullptr || item.secondMoment == nullptr)
                throw std::invalid_argument("CudaAdam::updateMany null fp32 moments");
        }
    }
    if (anyInt8 && anyFp32)
        throw std::invalid_argument("CudaAdam::updateMany mixed int8/fp32 items not supported");

    const float firstMomentCorrection = 1.0f - std::pow(this->beta1, static_cast<float>(this->timeStep));
    const float secondMomentCorrection = 1.0f - std::pow(this->beta2, static_cast<float>(this->timeStep));
    const float inverseFirstCorrection = 1.0f / firstMomentCorrection;
    const float inverseSecondCorrection = 1.0f / secondMomentCorrection;

    if (anyInt8) {
        constexpr int threadCount = 256;
        const int blockSize = CudaAdam::int8BlockSize;
        if (blockSize > 1024)
            throw std::invalid_argument("CudaAdam::updateMany int8BlockSize too large for shared memory");
        const size_t sharedBytes = (static_cast<size_t>(blockSize) * 2u + static_cast<size_t>(threadCount) * 2u) * sizeof(float);
        for (int itemIndex = 0; itemIndex < itemCount; ++itemIndex) {
            const CudaAdamUpdateItem& item = items[itemIndex];
            if (item.scaleCount != (item.elementCount + blockSize - 1) / blockSize)
                throw std::invalid_argument("CudaAdam::updateMany int8 scaleCount mismatch with int8BlockSize");
            CudaAdamUpdateInt8TensorEntry<<<item.scaleCount, threadCount, sharedBytes, CudaMatmul::activeStream()>>>(
                item.parameter,
                item.firstMomentQ,
                item.secondMomentQ,
                item.firstMomentScales,
                item.secondMomentScales,
                item.gradient,
                item.elementCount,
                blockSize,
                this->learningRate,
                this->beta1,
                this->beta2,
                this->epsilon,
                inverseFirstCorrection,
                inverseSecondCorrection,
                gradientScale);
            CudaMatmul::throwIfCudaFailed(cudaGetLastError(), "CudaAdamUpdateInt8TensorEntry launch");
        }
        return;
    }

    const size_t byteCount = static_cast<size_t>(itemCount) * sizeof(CudaAdamUpdateItem);
    CudaDeviceBuffer& buffer = const_cast<CudaAdam*>(this)->itemBuffer;
    buffer.ensureCapacity(byteCount);
    buffer.copyBytesFromHost(items, byteCount);

    constexpr int threadCount = 256;
    CudaAdamUpdateManyEntry<<<itemCount, threadCount, 0, CudaMatmul::activeStream()>>>(
        reinterpret_cast<const CudaAdamUpdateItem*>(buffer.deviceData),
        itemCount,
        this->learningRate,
        this->beta1,
        this->beta2,
        this->epsilon,
        inverseFirstCorrection,
        inverseSecondCorrection,
        gradientScale);
    CudaMatmul::throwIfCudaFailed(cudaGetLastError(), "CudaAdamUpdateManyEntry launch");
}

void CudaAdam::runSmokeDemo(int parameterRows, int parameterCols) {
    if (!CudaMatmul::isAvailable()) {
        SmokeLog::skip("Adam");
        return;
    }
    if (parameterRows <= 0 || parameterCols <= 0) throw std::invalid_argument("CudaAdam::runSmokeDemo invalid dims");

    Matrix hostParameter(static_cast<size_t>(parameterRows), static_cast<size_t>(parameterCols), 0.0f);
    Matrix hostGradient(static_cast<size_t>(parameterRows), static_cast<size_t>(parameterCols), 0.0f);
    unsigned state = 149u;
    for (size_t index = 0; index < hostParameter.data.size(); ++index) {
        state = state * 1664525u + 1013904223u;
        hostParameter.data[index] = (static_cast<float>(state >> 8) / 16777216.0f) * 2.0f - 1.0f;
        state = state * 1664525u + 1013904223u;
        hostGradient.data[index] = (static_cast<float>(state >> 8) / 16777216.0f) * 2.0f - 1.0f;
    }

    Matrix hostParameterCopy = hostParameter;
    Adam hostAdam(0.001f);
    hostAdam.step();
    AdamState hostState = AdamState::zerosLike(hostParameterCopy);
    hostAdam.update(hostParameterCopy, hostState, hostGradient);

    const bool previousInt8 = CudaAdam::preferInt8Moments;
    CudaAdam::preferInt8Moments = false;

    CudaAdam deviceAdam(0.001f);
    deviceAdam.step();
    CudaMatrix deviceParameter;
    deviceParameter.upload(hostParameter);
    CudaMatrix deviceGradient;
    deviceGradient.upload(hostGradient);
    CudaAdamState deviceState;
    deviceState.ensureFp32(deviceParameter);
    deviceAdam.update(deviceParameter, deviceState, deviceGradient);
    CudaMatmul::throwIfCudaFailed(cudaDeviceSynchronize(), "CudaAdam smoke synchronize");

    Matrix deviceHostParameter = deviceParameter.download();
    float maximumDifference = 0.0f;
    for (size_t index = 0; index < hostParameterCopy.data.size(); ++index)
        maximumDifference = (std::max)(maximumDifference, std::fabs(hostParameterCopy.data[index] - deviceHostParameter.data[index]));

    SmokeLog::result("Adam", "rows=%d cols=%d  diff=%.2e  int8Moments=off", parameterRows, parameterCols, maximumDifference);

    CudaMatrix fp32RefParameter;
    fp32RefParameter.upload(hostParameter);
    CudaAdamState fp32RefState;
    fp32RefState.ensureFp32(fp32RefParameter);
    CudaAdam fp32RefAdam(0.001f);

    CudaMatrix int8Parameter;
    int8Parameter.upload(hostParameter);
    CudaAdamState int8State;
    int8State.ensureInt8(int8Parameter, CudaAdam::int8BlockSize);
    CudaAdam int8Adam(0.001f);

    CudaMatrix stepGradient;
    stepGradient.upload(hostGradient);
    for (int step = 0; step < 32; ++step) {
        fp32RefAdam.step();
        int8Adam.step();
        CudaAdam::preferInt8Moments = false;
        fp32RefAdam.update(fp32RefParameter, fp32RefState, stepGradient);
        CudaAdam::preferInt8Moments = true;
        int8Adam.update(int8Parameter, int8State, stepGradient);
    }
    CudaMatmul::throwIfCudaFailed(cudaDeviceSynchronize(), "CudaAdam int8 multi-step synchronize");

    Matrix fp32RefHost = fp32RefParameter.download();
    Matrix int8HostParameter = int8Parameter.download();
    float int8Difference = 0.0f;
    for (size_t index = 0; index < fp32RefHost.data.size(); ++index)
        int8Difference = (std::max)(int8Difference, std::fabs(fp32RefHost.data[index] - int8HostParameter.data[index]));
    SmokeLog::result("Adam int8", "rows=%d cols=%d  steps=32  diff=%.2e  block=%d", parameterRows, parameterCols, int8Difference, CudaAdam::int8BlockSize);

    // multi-tensor + tiny grads (LM-like) — catches bias-correction / quant flush bugs
    struct TensorPair {
        int rows;
        int cols;
        CudaMatrix fp32Parameter;
        CudaMatrix int8Parameter;
        CudaMatrix gradient;
        CudaAdamState fp32State;
        CudaAdamState int8State;
    };

    const int shapes[][2] = {
        {64, 1},
        {1000, 1},
        {64, 64},
        {256, 64},
        {1000, 64},
        {32, 32},
    };
    const int tensorCount = static_cast<int>(sizeof(shapes) / sizeof(shapes[0]));
    std::vector<TensorPair> tensors(static_cast<size_t>(tensorCount));
    unsigned multiState = 991u;
    for (int tensorIndex = 0; tensorIndex < tensorCount; ++tensorIndex) {
        TensorPair& tensor = tensors[static_cast<size_t>(tensorIndex)];
        tensor.rows = shapes[tensorIndex][0];
        tensor.cols = shapes[tensorIndex][1];
        Matrix hostParameter(static_cast<size_t>(tensor.rows), static_cast<size_t>(tensor.cols), 0.0f);
        Matrix hostGradient(static_cast<size_t>(tensor.rows), static_cast<size_t>(tensor.cols), 0.0f);
        for (size_t index = 0; index < hostParameter.data.size(); ++index) {
            multiState = multiState * 1664525u + 1013904223u;
            hostParameter.data[index] = (static_cast<float>(multiState >> 8) / 16777216.0f) * 0.02f - 0.01f;
            multiState = multiState * 1664525u + 1013904223u;
            // small grads like mean-CE / long segments
            hostGradient.data[index] = (static_cast<float>(multiState >> 8) / 16777216.0f) * 2.0e-4f - 1.0e-4f;
        }
        tensor.fp32Parameter.upload(hostParameter);
        tensor.int8Parameter.upload(hostParameter);
        tensor.gradient.upload(hostGradient);
        tensor.fp32State.ensureFp32(tensor.fp32Parameter);
        tensor.int8State.ensureInt8(tensor.int8Parameter, CudaAdam::int8BlockSize);
    }

    CudaAdam multiFp32(0.001f);
    CudaAdam multiInt8(0.001f);
    const float gradientScale = 1.0f / 32.0f;
    for (int step = 0; step < 64; ++step) {
        multiFp32.step();
        multiInt8.step();

        std::vector<CudaAdamUpdateItem> fp32Items;
        std::vector<CudaAdamUpdateItem> int8Items;
        fp32Items.reserve(static_cast<size_t>(tensorCount));
        int8Items.reserve(static_cast<size_t>(tensorCount));
        for (int tensorIndex = 0; tensorIndex < tensorCount; ++tensorIndex) {
            TensorPair& tensor = tensors[static_cast<size_t>(tensorIndex)];
            CudaAdamUpdateItem fp32Item;
            fp32Item.parameter = tensor.fp32Parameter.buffer.deviceData;
            fp32Item.gradient = tensor.gradient.buffer.deviceData;
            fp32Item.elementCount = static_cast<int>(tensor.fp32Parameter.elementCount());
            fp32Item.useInt8 = false;
            fp32Item.firstMoment = tensor.fp32State.firstMoment.buffer.deviceData;
            fp32Item.secondMoment = tensor.fp32State.secondMoment.buffer.deviceData;
            fp32Item.firstMomentQ = nullptr;
            fp32Item.secondMomentQ = nullptr;
            fp32Item.firstMomentScales = nullptr;
            fp32Item.secondMomentScales = nullptr;
            fp32Item.scaleCount = 0;
            fp32Items.push_back(fp32Item);

            CudaAdamUpdateItem int8Item;
            int8Item.parameter = tensor.int8Parameter.buffer.deviceData;
            int8Item.gradient = tensor.gradient.buffer.deviceData;
            int8Item.elementCount = static_cast<int>(tensor.int8Parameter.elementCount());
            int8Item.useInt8 = true;
            int8Item.firstMoment = nullptr;
            int8Item.secondMoment = nullptr;
            int8Item.firstMomentQ = tensor.int8State.firstMomentQ.deviceData;
            int8Item.secondMomentQ = tensor.int8State.secondMomentQ.deviceData;
            int8Item.firstMomentScales = tensor.int8State.firstMomentScales.deviceData;
            int8Item.secondMomentScales = tensor.int8State.secondMomentScales.deviceData;
            int8Item.scaleCount = tensor.int8State.scaleCount;
            int8Items.push_back(int8Item);
        }

        CudaAdam::preferInt8Moments = false;
        multiFp32.updateMany(fp32Items.data(), tensorCount, gradientScale);
        CudaAdam::preferInt8Moments = true;
        multiInt8.updateMany(int8Items.data(), tensorCount, gradientScale);
    }
    CudaMatmul::throwIfCudaFailed(cudaDeviceSynchronize(), "CudaAdam multi-tensor int8 synchronize");

    float multiDifference = 0.0f;
    bool anyNonFinite = false;
    for (int tensorIndex = 0; tensorIndex < tensorCount; ++tensorIndex) {
        Matrix fp32Host = tensors[static_cast<size_t>(tensorIndex)].fp32Parameter.download();
        Matrix int8Host = tensors[static_cast<size_t>(tensorIndex)].int8Parameter.download();
        for (size_t index = 0; index < fp32Host.data.size(); ++index) {
            if (!std::isfinite(fp32Host.data[index]) || !std::isfinite(int8Host.data[index]))
                anyNonFinite = true;
            multiDifference = (std::max)(multiDifference, std::fabs(fp32Host.data[index] - int8Host.data[index]));
        }
    }
    SmokeLog::result("Adam int8 multi", "tensors=%d steps=64  diff=%.2e  nonFinite=%s  block=%d",
        tensorCount, multiDifference, anyNonFinite ? "yes" : "no", CudaAdam::int8BlockSize);

    CudaAdam::preferInt8Moments = previousInt8;
}
