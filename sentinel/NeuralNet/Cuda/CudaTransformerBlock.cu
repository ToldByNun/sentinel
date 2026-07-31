#include "CudaTransformerBlock.hpp"

#include "CudaOps.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <stdexcept>

CudaTransformerBlock::CudaTransformerBlock() {}

void CudaTransformerBlock::uploadFrom(const TransformerBlock& host) {
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaTransformerBlock::uploadFrom no CUDA device");

    this->attentionNorm.uploadFrom(host.attentionNorm);
    this->attention.uploadFrom(host.attention);
    this->feedForwardNorm.uploadFrom(host.feedForwardNorm);
    this->feedForward.uploadFrom(host.feedForward);
}

CudaTransformerBlock CudaTransformerBlock::createFrom(const TransformerBlock& host) {
    CudaTransformerBlock device;
    device.uploadFrom(host);
    return device;
}

void CudaTransformerBlock::forward(const CudaMatrix& input, CudaMatrix& out) {
    if (input.empty()) throw std::invalid_argument("CudaTransformerBlock::forward empty input");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaTransformerBlock::forward no CUDA device");

    this->attentionNorm.forward(input, this->attentionInput);
    this->attention.forward(this->attentionInput, this->attended);
    CudaOps::addInto(input, this->attended, this->afterAttention);

    this->feedForwardNorm.forward(this->afterAttention, this->feedForwardInput);
    this->feedForward.forward(this->feedForwardInput, this->feedForwardOutput);
    CudaOps::addInto(this->afterAttention, this->feedForwardOutput, out);
}

void CudaTransformerBlock::runSmokeDemo(int embeddingDim, int headCount, int sequenceLength, int maximumPositionCount) {
    if (!CudaMatmul::isAvailable()) {
        std::printf("CUDA TransformerBlock smoke: no device\n");
        return;
    }
    if (embeddingDim <= 0 || headCount <= 0 || sequenceLength <= 0 || maximumPositionCount < sequenceLength)
        throw std::invalid_argument("CudaTransformerBlock::runSmokeDemo invalid dims");

    TransformerBlock host(embeddingDim, headCount, maximumPositionCount, 21u);
    Matrix hostInput(static_cast<size_t>(embeddingDim), static_cast<size_t>(sequenceLength), 0.0f);
    unsigned state = 77u;
    for (size_t index = 0; index < hostInput.data.size(); ++index) {
        state = state * 1664525u + 1013904223u;
        hostInput.data[index] = (static_cast<float>(state >> 8) / 16777216.0f) * 2.0f - 1.0f;
    }

    TransformerBlockCache hostCache;
    const auto cpuStart = std::chrono::steady_clock::now();
    Matrix hostOutput = host.forward(hostInput, hostCache);
    const double cpuMilliseconds = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - cpuStart).count();

    CudaTransformerBlock device = CudaTransformerBlock::createFrom(host);
    CudaMatrix deviceInput;
    deviceInput.upload(hostInput);
    CudaMatrix deviceOutput;
    device.forward(deviceInput, deviceOutput);
    CudaMatmul::throwIfCudaFailed(cudaDeviceSynchronize(), "CudaTransformerBlock warm synchronize");

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

    std::printf("CUDA TransformerBlock smoke: embed=%d heads=%d seq=%d  cpu=%.2fms  device=%.2fms  maxAbsDiff=%.6g\n", embeddingDim, headCount, sequenceLength, cpuMilliseconds, static_cast<double>(deviceMilliseconds), maximumDifference);
}
