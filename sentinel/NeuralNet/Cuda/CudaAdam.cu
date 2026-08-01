#include "CudaAdam.hpp"

#include "CudaOps.hpp"

#include "../Optimizers/Adam.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <stdexcept>

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

__device__ void CudaAdam::runUpdate(float* parameter, float* firstMoment, float* secondMoment, const float* gradient, int elementCount, float learningRate, float beta1, float beta2, float epsilon, float inverseFirstCorrection, float inverseSecondCorrection) {
    const int index = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (index >= elementCount) return;

    const float gradientValue = gradient[index];
    float updatedFirstMoment = beta1 * firstMoment[index] + (1.0f - beta1) * gradientValue;
    float updatedSecondMoment = beta2 * secondMoment[index] + (1.0f - beta2) * gradientValue * gradientValue;
    firstMoment[index] = updatedFirstMoment;
    secondMoment[index] = updatedSecondMoment;

    const float correctedFirst = updatedFirstMoment * inverseFirstCorrection;
    const float correctedSecond = updatedSecondMoment * inverseSecondCorrection;
    parameter[index] -= learningRate * correctedFirst / (sqrtf(correctedSecond) + epsilon);
}

__global__ void CudaAdamUpdateEntry(float* parameter, float* firstMoment, float* secondMoment, const float* gradient, int elementCount, float learningRate, float beta1, float beta2, float epsilon, float inverseFirstCorrection, float inverseSecondCorrection) {
    CudaAdam::runUpdate(parameter, firstMoment, secondMoment, gradient, elementCount, learningRate, beta1, beta2, epsilon, inverseFirstCorrection, inverseSecondCorrection);
}

void CudaAdam::update(CudaMatrix& parameter, CudaAdamState& state, const CudaMatrix& gradient) const {
    if (this->timeStep <= 0) throw std::invalid_argument("CudaAdam::update requires step() before update");
    if (parameter.rows != gradient.rows || parameter.cols != gradient.cols)
        throw std::invalid_argument("CudaAdam::update parameter/gradient shape mismatch");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaAdam::update no CUDA device");

    if (state.firstMoment.empty()) {
        state.firstMoment.ensureSize(parameter.rows, parameter.cols);
        state.secondMoment.ensureSize(parameter.rows, parameter.cols);
        CudaOps::zeroInPlace(state.firstMoment);
        CudaOps::zeroInPlace(state.secondMoment);
    }
    if (state.firstMoment.rows != parameter.rows || state.firstMoment.cols != parameter.cols)
        throw std::invalid_argument("CudaAdam::update moment shape mismatch");

    const float firstMomentCorrection = 1.0f - std::pow(this->beta1, static_cast<float>(this->timeStep));
    const float secondMomentCorrection = 1.0f - std::pow(this->beta2, static_cast<float>(this->timeStep));
    const float inverseFirstCorrection = 1.0f / firstMomentCorrection;
    const float inverseSecondCorrection = 1.0f / secondMomentCorrection;

    const int elementCount = static_cast<int>(parameter.elementCount());
    constexpr int threadCount = 256;
    const int blockCount = (elementCount + threadCount - 1) / threadCount;
    CudaAdamUpdateEntry<<<blockCount, threadCount>>>(parameter.buffer.deviceData, state.firstMoment.buffer.deviceData, state.secondMoment.buffer.deviceData, gradient.buffer.deviceData, elementCount, this->learningRate, this->beta1, this->beta2, this->epsilon, inverseFirstCorrection, inverseSecondCorrection);
    CudaMatmul::throwIfCudaFailed(cudaGetLastError(), "CudaAdamUpdateEntry launch");
}

void CudaAdam::runSmokeDemo(int parameterRows, int parameterCols) {
    if (!CudaMatmul::isAvailable()) {
        std::printf("CUDA Adam smoke: no device\n");
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

    std::printf("CUDA Adam smoke: rows=%d cols=%d  maxAbsDiff=%.6g\n", parameterRows, parameterCols, maximumDifference);
}
