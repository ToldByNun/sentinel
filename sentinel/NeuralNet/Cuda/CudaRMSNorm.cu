#include "CudaRMSNorm.hpp"

#include "CudaOps.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <stdexcept>

CudaRMSNorm::CudaRMSNorm() : epsilon(1e-5f) {}

void CudaRMSNorm::uploadFrom(const RMSNorm& host) {
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaRMSNorm::uploadFrom no CUDA device");

    this->gamma.upload(host.gamma);
    this->epsilon = host.epsilon;
}

CudaRMSNorm CudaRMSNorm::createFrom(const RMSNorm& host) {
    CudaRMSNorm device;
    device.uploadFrom(host);
    return device;
}

__device__ void CudaRMSNorm::runComputeInverseRms(const float* input, float* inverseRms, int embeddingDim, int sequenceLength, float epsilon) {
    const int column = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (column >= sequenceLength) return;

    float squareSum = 0.0f;
    for (int row = 0; row < embeddingDim; ++row) {
        const float value = input[row * sequenceLength + column];
        squareSum += value * value;
    }
    inverseRms[column] = rsqrtf(squareSum / static_cast<float>(embeddingDim) + epsilon);
}

__device__ void CudaRMSNorm::runApply(const float* input, float* out, float* normalizedOut, const float* gamma, const float* inverseRms, int embeddingDim, int sequenceLength) {
    const int elementCount = embeddingDim * sequenceLength;
    const int index = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (index >= elementCount) return;

    const int row = index / sequenceLength;
    const int column = index - row * sequenceLength;
    const float normalized = input[index] * inverseRms[column];
    if (normalizedOut != nullptr) normalizedOut[index] = normalized;
    out[index] = gamma[row] * normalized;
}

__device__ void CudaRMSNorm::runBackwardInputGradColumn(const float* outputGrad, const float* normalized, const float* gamma, float* inputGrad, float inverseRmsValue, int embeddingDim, int sequenceLength, int column, float dimension) {
    float sumGradientNormalized = 0.0f;
    for (int row = 0; row < embeddingDim; ++row) {
        const int index = row * sequenceLength + column;
        const float normalizedGradient = outputGrad[index] * gamma[row];
        sumGradientNormalized += normalizedGradient * normalized[index];
    }

    const float scale = (1.0f / dimension) * inverseRmsValue;
    for (int row = 0; row < embeddingDim; ++row) {
        const int index = row * sequenceLength + column;
        const float normalizedGradient = outputGrad[index] * gamma[row];
        inputGrad[index] = scale * (dimension * normalizedGradient - normalized[index] * sumGradientNormalized);
    }
}

__global__ void CudaRMSNormComputeInverseRmsEntry(const float* input, float* inverseRms, int embeddingDim, int sequenceLength, float epsilon) {
    CudaRMSNorm::runComputeInverseRms(input, inverseRms, embeddingDim, sequenceLength, epsilon);
}

__global__ void CudaRMSNormApplyEntry(const float* input, float* out, float* normalizedOut, const float* gamma, const float* inverseRms, int embeddingDim, int sequenceLength) {
    CudaRMSNorm::runApply(input, out, normalizedOut, gamma, inverseRms, embeddingDim, sequenceLength);
}

__global__ void CudaRMSNormBackwardInputGradEntry(const float* outputGrad, const float* normalized, const float* gamma, float* inputGrad, const float* inverseRms, int embeddingDim, int sequenceLength, float dimension) {
    const int column = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (column >= sequenceLength) return;

    CudaRMSNorm::runBackwardInputGradColumn(outputGrad, normalized, gamma, inputGrad, inverseRms[column], embeddingDim, sequenceLength, column, dimension);
}

void CudaRMSNorm::forward(const CudaMatrix& input, CudaMatrix& out) const {
    if (input.empty()) throw std::invalid_argument("CudaRMSNorm::forward empty input");
    if (this->gamma.empty()) throw std::logic_error("CudaRMSNorm::forward weights not uploaded");
    if (input.rows != this->gamma.rows) throw std::invalid_argument("CudaRMSNorm::forward embedding dim mismatch");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaRMSNorm::forward no CUDA device");

    const int embeddingDim = static_cast<int>(input.rows);
    const int sequenceLength = static_cast<int>(input.cols);
    out.ensureSize(input.rows, input.cols);
    this->inverseRms.ensureSize(1, input.cols);
    this->lastInput.ensureSize(input.rows, input.cols);
    this->lastNormalized.ensureSize(input.rows, input.cols);
    CudaOps::copyInto(input, this->lastInput);

    constexpr int threadCount = 256;
    const int inverseBlockCount = (sequenceLength + threadCount - 1) / threadCount;
    CudaRMSNormComputeInverseRmsEntry<<<inverseBlockCount, threadCount>>>(input.buffer.deviceData, this->inverseRms.buffer.deviceData, embeddingDim, sequenceLength, this->epsilon);
    CudaMatmul::throwIfCudaFailed(cudaGetLastError(), "CudaRMSNormComputeInverseRmsEntry launch");

    const int elementCount = embeddingDim * sequenceLength;
    const int applyBlockCount = (elementCount + threadCount - 1) / threadCount;
    CudaRMSNormApplyEntry<<<applyBlockCount, threadCount>>>(input.buffer.deviceData, out.buffer.deviceData, this->lastNormalized.buffer.deviceData, this->gamma.buffer.deviceData, this->inverseRms.buffer.deviceData, embeddingDim, sequenceLength);
    CudaMatmul::throwIfCudaFailed(cudaGetLastError(), "CudaRMSNormApplyEntry launch");
}

void CudaRMSNorm::backward(const CudaMatrix& outputGradient, CudaMatrix& inputGradient, CudaMatrix& gammaGradient) const {
    if (this->lastInput.empty() || this->lastNormalized.empty()) throw std::logic_error("CudaRMSNorm::backward called before forward");
    if (outputGradient.rows != this->lastInput.rows || outputGradient.cols != this->lastInput.cols)
        throw std::invalid_argument("CudaRMSNorm::backward shape mismatch");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaRMSNorm::backward no CUDA device");

    const int embeddingDim = static_cast<int>(this->lastInput.rows);
    const int sequenceLength = static_cast<int>(this->lastInput.cols);
    const float dimension = static_cast<float>(embeddingDim);

    inputGradient.ensureSize(this->lastInput.rows, this->lastInput.cols);
    gammaGradient.ensureSize(this->gamma.rows, 1);

    this->backwardScratch.ensureSize(outputGradient.rows, outputGradient.cols);
    CudaOps::multiplyElementwiseInto(outputGradient, this->lastNormalized, this->backwardScratch);
    CudaOps::sumColumnsInto(this->backwardScratch, gammaGradient);

    constexpr int threadCount = 256;
    const int backwardBlockCount = (sequenceLength + threadCount - 1) / threadCount;
    CudaRMSNormBackwardInputGradEntry<<<backwardBlockCount, threadCount>>>(outputGradient.buffer.deviceData, this->lastNormalized.buffer.deviceData, this->gamma.buffer.deviceData, inputGradient.buffer.deviceData, this->inverseRms.buffer.deviceData, embeddingDim, sequenceLength, dimension);
    CudaMatmul::throwIfCudaFailed(cudaGetLastError(), "CudaRMSNormBackwardInputGradEntry launch");
}

void CudaRMSNorm::runSmokeDemo(int embeddingDim, int sequenceLength) {
    if (!CudaMatmul::isAvailable()) {
        std::printf("CUDA RMSNorm smoke: no device\n");
        return;
    }
    if (embeddingDim <= 0 || sequenceLength <= 0) throw std::invalid_argument("CudaRMSNorm::runSmokeDemo invalid dims");

    RMSNorm host(embeddingDim);
    for (size_t row = 0; row < host.gamma.rows; ++row)
        host.gamma.at(row, 0) = 0.5f + 0.01f * static_cast<float>(row);

    Matrix hostInput(static_cast<size_t>(embeddingDim), static_cast<size_t>(sequenceLength), 0.0f);
    unsigned state = 123u;
    for (size_t index = 0; index < hostInput.data.size(); ++index) {
        state = state * 1664525u + 1013904223u;
        hostInput.data[index] = (static_cast<float>(state >> 8) / 16777216.0f) * 2.0f - 1.0f;
    }

    RMSNormCache hostCache;
    const auto cpuStart = std::chrono::steady_clock::now();
    Matrix hostOutput = host.forward(hostInput, hostCache);
    const double cpuMilliseconds = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - cpuStart).count();

    CudaRMSNorm device = CudaRMSNorm::createFrom(host);
    CudaMatrix deviceInput;
    deviceInput.upload(hostInput);
    CudaMatrix deviceOutput;

    device.forward(deviceInput, deviceOutput);
    CudaMatmul::throwIfCudaFailed(cudaDeviceSynchronize(), "CudaRMSNorm warm synchronize");

    cudaEvent_t kernelStartEvent = nullptr;
    cudaEvent_t kernelStopEvent = nullptr;
    CudaMatmul::throwIfCudaFailed(cudaEventCreate(&kernelStartEvent), "cudaEventCreate start");
    CudaMatmul::throwIfCudaFailed(cudaEventCreate(&kernelStopEvent), "cudaEventCreate stop");
    CudaMatmul::throwIfCudaFailed(cudaEventRecord(kernelStartEvent), "cudaEventRecord start");
    device.forward(deviceInput, deviceOutput);
    CudaMatmul::throwIfCudaFailed(cudaEventRecord(kernelStopEvent), "cudaEventRecord stop");
    CudaMatmul::throwIfCudaFailed(cudaEventSynchronize(kernelStopEvent), "cudaEventSynchronize stop");
    float deviceMilliseconds = 0.0f;
    CudaMatmul::throwIfCudaFailed(cudaEventElapsedTime(&deviceMilliseconds, kernelStartEvent, kernelStopEvent), "cudaEventElapsedTime");
    cudaEventDestroy(kernelStartEvent);
    cudaEventDestroy(kernelStopEvent);

    Matrix deviceHostOutput = deviceOutput.download();
    float maximumDifference = 0.0f;
    for (size_t index = 0; index < hostOutput.data.size(); ++index)
        maximumDifference = (std::max)(maximumDifference, std::fabs(hostOutput.data[index] - deviceHostOutput.data[index]));

    std::printf("CUDA RMSNorm smoke: embed=%d seq=%d  cpu=%.2fms  device=%.2fms  maxAbsDiff=%.6g\n", embeddingDim, sequenceLength, cpuMilliseconds, static_cast<double>(deviceMilliseconds), maximumDifference);
}

void CudaRMSNorm::runBackwardSmokeDemo(int embeddingDim, int sequenceLength) {
    if (!CudaMatmul::isAvailable()) {
        std::printf("CUDA RMSNorm backward smoke: no device\n");
        return;
    }
    if (embeddingDim <= 0 || sequenceLength <= 0) throw std::invalid_argument("CudaRMSNorm::runBackwardSmokeDemo invalid dims");

    RMSNorm host(embeddingDim);
    for (size_t row = 0; row < host.gamma.rows; ++row)
        host.gamma.at(row, 0) = 0.5f + 0.01f * static_cast<float>(row);

    Matrix hostInput(static_cast<size_t>(embeddingDim), static_cast<size_t>(sequenceLength), 0.0f);
    unsigned state = 131u;
    for (size_t index = 0; index < hostInput.data.size(); ++index) {
        state = state * 1664525u + 1013904223u;
        hostInput.data[index] = (static_cast<float>(state >> 8) / 16777216.0f) * 2.0f - 1.0f;
    }

    Matrix outputGradient(static_cast<size_t>(embeddingDim), static_cast<size_t>(sequenceLength), 0.0f);
    state = 137u;
    for (size_t index = 0; index < outputGradient.data.size(); ++index) {
        state = state * 1664525u + 1013904223u;
        outputGradient.data[index] = (static_cast<float>(state >> 8) / 16777216.0f) * 2.0f - 1.0f;
    }

    RMSNormCache hostCache;
    host.forward(hostInput, hostCache);
    Matrix hostGammaGradient;
    Matrix hostInputGradient = host.backward(outputGradient, hostCache, hostGammaGradient);

    CudaRMSNorm device = CudaRMSNorm::createFrom(host);
    CudaMatrix deviceInput;
    deviceInput.upload(hostInput);
    CudaMatrix deviceOutput;
    device.forward(deviceInput, deviceOutput);

    CudaMatrix deviceOutputGradient;
    deviceOutputGradient.upload(outputGradient);
    CudaMatrix deviceInputGradient;
    CudaMatrix deviceGammaGradient;
    device.backward(deviceOutputGradient, deviceInputGradient, deviceGammaGradient);
    CudaMatmul::throwIfCudaFailed(cudaDeviceSynchronize(), "CudaRMSNorm backward synchronize");

    Matrix deviceHostInputGradient = deviceInputGradient.download();
    Matrix deviceHostGammaGradient = deviceGammaGradient.download();

    float inputGradientDifference = 0.0f;
    for (size_t index = 0; index < hostInputGradient.data.size(); ++index)
        inputGradientDifference = (std::max)(inputGradientDifference, std::fabs(hostInputGradient.data[index] - deviceHostInputGradient.data[index]));

    float gammaGradientDifference = 0.0f;
    for (size_t index = 0; index < hostGammaGradient.data.size(); ++index)
        gammaGradientDifference = (std::max)(gammaGradientDifference, std::fabs(hostGammaGradient.data[index] - deviceHostGammaGradient.data[index]));

    std::printf("CUDA RMSNorm backward smoke: embed=%d seq=%d  inputGrad maxAbsDiff=%.6g  gammaGrad maxAbsDiff=%.6g\n", embeddingDim, sequenceLength, inputGradientDifference, gammaGradientDifference);
}
