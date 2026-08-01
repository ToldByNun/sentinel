#include "CudaAdam.hpp"

#include "CudaOps.hpp"
#include "../Utils/SmokeLog.hpp"

#include "../Optimizers/Adam.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>

CudaAdamState CudaAdamState::zerosLike(const CudaMatrix& parameter) {
    CudaAdamState state;
    state.firstMoment.ensureSize(parameter.rows, parameter.cols);
    state.secondMoment.ensureSize(parameter.rows, parameter.cols);
    CudaOps::zeroInPlace(state.firstMoment);
    CudaOps::zeroInPlace(state.secondMoment);
    return state;
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
    float updatedFirstMoment = beta1 * firstMoment[index] + (1.0f - beta1) * gradientValue;
    float updatedSecondMoment = beta2 * secondMoment[index] + (1.0f - beta2) * gradientValue * gradientValue;
    firstMoment[index] = updatedFirstMoment;
    secondMoment[index] = updatedSecondMoment;

    const float correctedFirst = updatedFirstMoment * inverseFirstCorrection;
    const float correctedSecond = updatedSecondMoment * inverseSecondCorrection;
    parameter[index] -= learningRate * correctedFirst / (sqrtf(correctedSecond) + epsilon);
}

__device__ void CudaAdam::runUpdateMany(const CudaAdamUpdateItem* items, int itemCount, float learningRate, float beta1, float beta2, float epsilon, float inverseFirstCorrection, float inverseSecondCorrection, float gradientScale) {
    const int tensorIndex = static_cast<int>(blockIdx.x);
    if (tensorIndex >= itemCount) return;

    const CudaAdamUpdateItem item = items[tensorIndex];
    for (int index = static_cast<int>(threadIdx.x); index < item.elementCount; index += static_cast<int>(blockDim.x)) {
        const float gradientValue = item.gradient[index] * gradientScale;
        float updatedFirstMoment = beta1 * item.firstMoment[index] + (1.0f - beta1) * gradientValue;
        float updatedSecondMoment = beta2 * item.secondMoment[index] + (1.0f - beta2) * gradientValue * gradientValue;
        item.firstMoment[index] = updatedFirstMoment;
        item.secondMoment[index] = updatedSecondMoment;

        const float correctedFirst = updatedFirstMoment * inverseFirstCorrection;
        const float correctedSecond = updatedSecondMoment * inverseSecondCorrection;
        item.parameter[index] -= learningRate * correctedFirst / (sqrtf(correctedSecond) + epsilon);
    }
}

__global__ void CudaAdamUpdateEntry(float* parameter, float* firstMoment, float* secondMoment, const float* gradient, int elementCount, float learningRate, float beta1, float beta2, float epsilon, float inverseFirstCorrection, float inverseSecondCorrection, float gradientScale) {
    CudaAdam::runUpdate(parameter, firstMoment, secondMoment, gradient, elementCount, learningRate, beta1, beta2, epsilon, inverseFirstCorrection, inverseSecondCorrection, gradientScale);
}

__global__ void CudaAdamUpdateManyEntry(const CudaAdamUpdateItem* items, int itemCount, float learningRate, float beta1, float beta2, float epsilon, float inverseFirstCorrection, float inverseSecondCorrection, float gradientScale) {
    CudaAdam::runUpdateMany(items, itemCount, learningRate, beta1, beta2, epsilon, inverseFirstCorrection, inverseSecondCorrection, gradientScale);
}

void CudaAdam::update(CudaMatrix& parameter, CudaAdamState& state, const CudaMatrix& gradient, float gradientScale) const {
    CudaAdamUpdateItem item;
    item.parameter = parameter.buffer.deviceData;
    if (state.firstMoment.empty()) {
        state.firstMoment.ensureSize(parameter.rows, parameter.cols);
        state.secondMoment.ensureSize(parameter.rows, parameter.cols);
        CudaOps::zeroInPlace(state.firstMoment);
        CudaOps::zeroInPlace(state.secondMoment);
    }
    item.firstMoment = state.firstMoment.buffer.deviceData;
    item.secondMoment = state.secondMoment.buffer.deviceData;
    item.gradient = gradient.buffer.deviceData;
    item.elementCount = static_cast<int>(parameter.elementCount());
    this->updateMany(&item, 1, gradientScale);
}

void CudaAdam::updateMany(const CudaAdamUpdateItem* items, int itemCount, float gradientScale) const {
    if (this->timeStep <= 0) throw std::invalid_argument("CudaAdam::updateMany requires step() before update");
    if (items == nullptr || itemCount <= 0) throw std::invalid_argument("CudaAdam::updateMany empty items");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaAdam::updateMany no CUDA device");

    for (int itemIndex = 0; itemIndex < itemCount; ++itemIndex) {
        if (items[itemIndex].parameter == nullptr || items[itemIndex].firstMoment == nullptr || items[itemIndex].secondMoment == nullptr || items[itemIndex].gradient == nullptr)
            throw std::invalid_argument("CudaAdam::updateMany null pointer");
        if (items[itemIndex].elementCount <= 0) throw std::invalid_argument("CudaAdam::updateMany invalid elementCount");
    }

    const float firstMomentCorrection = 1.0f - std::pow(this->beta1, static_cast<float>(this->timeStep));
    const float secondMomentCorrection = 1.0f - std::pow(this->beta2, static_cast<float>(this->timeStep));
    const float inverseFirstCorrection = 1.0f / firstMomentCorrection;
    const float inverseSecondCorrection = 1.0f / secondMomentCorrection;

    const size_t byteCount = static_cast<size_t>(itemCount) * sizeof(CudaAdamUpdateItem);
    // const method but buffer is mutable workspace; cast away for pooled reuse
    CudaDeviceBuffer& buffer = const_cast<CudaAdam*>(this)->itemBuffer;
    buffer.ensureCapacity(byteCount);
    buffer.copyBytesFromHost(items, byteCount);

    constexpr int threadCount = 256;
    CudaAdamUpdateManyEntry<<<itemCount, threadCount>>>(
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

    CudaAdam deviceAdam(0.001f);
    deviceAdam.step();
    CudaMatrix deviceParameter;
    deviceParameter.upload(hostParameter);
    CudaMatrix deviceGradient;
    deviceGradient.upload(hostGradient);
    CudaAdamState deviceState = CudaAdamState::zerosLike(deviceParameter);
    deviceAdam.update(deviceParameter, deviceState, deviceGradient);
    CudaMatmul::throwIfCudaFailed(cudaDeviceSynchronize(), "CudaAdam smoke synchronize");

    Matrix deviceHostParameter = deviceParameter.download();
    float maximumDifference = 0.0f;
    for (size_t index = 0; index < hostParameterCopy.data.size(); ++index)
        maximumDifference = (std::max)(maximumDifference, std::fabs(hostParameterCopy.data[index] - deviceHostParameter.data[index]));

    SmokeLog::result("Adam", "rows=%d cols=%d  diff=%.2e", parameterRows, parameterCols, maximumDifference);
}
