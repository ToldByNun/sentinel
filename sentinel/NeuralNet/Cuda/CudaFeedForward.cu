#include "CudaFeedForward.hpp"

#include "CudaOps.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <stdexcept>

CudaFeedForward::CudaFeedForward() {}

void CudaFeedForward::uploadFrom(const FeedForward& host) {
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaFeedForward::uploadFrom no CUDA device");

    this->gateWeight.upload(host.gateWeight);
    this->gateBias.upload(host.gateBias);
    this->upWeight.upload(host.upWeight);
    this->upBias.upload(host.upBias);
    this->downWeight.upload(host.downWeight);
    this->downBias.upload(host.downBias);
}

CudaFeedForward CudaFeedForward::createFrom(const FeedForward& host) {
    CudaFeedForward device;
    device.uploadFrom(host);
    return device;
}

void CudaFeedForward::forward(const CudaMatrix& input, CudaMatrix& out) {
    if (input.empty()) throw std::invalid_argument("CudaFeedForward::forward empty input");
    if (this->gateWeight.empty()) throw std::logic_error("CudaFeedForward::forward weights not uploaded");
    if (this->gateWeight.cols != input.rows) throw std::invalid_argument("CudaFeedForward::forward embedding dim mismatch");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaFeedForward::forward no CUDA device");

    CudaMatrix::multiplyInto(this->gateWeight, input, this->gatePreActivation);
    CudaOps::broadcastBiasAddInPlace(this->gatePreActivation, this->gateBias);
    CudaOps::siluInto(this->gatePreActivation, this->gateActivated);

    CudaMatrix::multiplyInto(this->upWeight, input, this->up);
    CudaOps::broadcastBiasAddInPlace(this->up, this->upBias);
    CudaOps::multiplyElementwiseInto(this->gateActivated, this->up, this->hidden);

    CudaMatrix::multiplyInto(this->downWeight, this->hidden, out);
    CudaOps::broadcastBiasAddInPlace(out, this->downBias);
}

void CudaFeedForward::runSmokeDemo(int embeddingDim, int sequenceLength) {
    if (!CudaMatmul::isAvailable()) {
        std::printf("CUDA FeedForward smoke: no device\n");
        return;
    }
    if (embeddingDim <= 0) throw std::invalid_argument("CudaFeedForward::runSmokeDemo embeddingDim must be > 0");
    if (sequenceLength <= 0) throw std::invalid_argument("CudaFeedForward::runSmokeDemo sequenceLength must be > 0");

    FeedForward host = FeedForward::create(embeddingDim, 4, 41u);
    Matrix hostInput(static_cast<size_t>(embeddingDim), static_cast<size_t>(sequenceLength), 0.0f);
    unsigned state = 99u;
    for (size_t index = 0; index < hostInput.data.size(); ++index) {
        state = state * 1664525u + 1013904223u;
        hostInput.data[index] = (static_cast<float>(state >> 8) / 16777216.0f) * 2.0f - 1.0f;
    }

    FeedForwardCache hostCache;
    const auto cpuStart = std::chrono::steady_clock::now();
    Matrix hostOutput = host.forward(hostInput, hostCache);
    const double cpuMilliseconds = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - cpuStart).count();

    const auto uploadStart = std::chrono::steady_clock::now();
    CudaFeedForward device = CudaFeedForward::createFrom(host);
    CudaMatrix deviceInput;
    deviceInput.upload(hostInput);
    const double uploadMilliseconds = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - uploadStart).count();

    CudaMatrix deviceOutput;
    device.forward(deviceInput, deviceOutput);
    CudaMatmul::throwIfCudaFailed(cudaDeviceSynchronize(), "CudaFeedForward warm synchronize");

    cudaEvent_t kernelStartEvent = nullptr;
    cudaEvent_t kernelStopEvent = nullptr;
    CudaMatmul::throwIfCudaFailed(cudaEventCreate(&kernelStartEvent), "cudaEventCreate start");
    CudaMatmul::throwIfCudaFailed(cudaEventCreate(&kernelStopEvent), "cudaEventCreate stop");
    CudaMatmul::throwIfCudaFailed(cudaEventRecord(kernelStartEvent), "cudaEventRecord start");
    device.forward(deviceInput, deviceOutput);
    CudaMatmul::throwIfCudaFailed(cudaEventRecord(kernelStopEvent), "cudaEventRecord stop");
    CudaMatmul::throwIfCudaFailed(cudaEventSynchronize(kernelStopEvent), "cudaEventSynchronize stop");
    float deviceForwardMilliseconds = 0.0f;
    CudaMatmul::throwIfCudaFailed(cudaEventElapsedTime(&deviceForwardMilliseconds, kernelStartEvent, kernelStopEvent), "cudaEventElapsedTime");
    cudaEventDestroy(kernelStartEvent);
    cudaEventDestroy(kernelStopEvent);

    const auto downloadStart = std::chrono::steady_clock::now();
    Matrix deviceHostOutput = deviceOutput.download();
    const double downloadMilliseconds = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - downloadStart).count();

    float maximumDifference = 0.0f;
    for (size_t index = 0; index < hostOutput.data.size(); ++index)
        maximumDifference = (std::max)(maximumDifference, std::fabs(hostOutput.data[index] - deviceHostOutput.data[index]));

    std::printf("CUDA FeedForward smoke: embed=%d seq=%d hidden=%zu\n", embeddingDim, sequenceLength, host.gateWeight.rows);
    std::printf("  cpu=%.2fms  upload=%.2fms  device-forward=%.2fms  download=%.2fms  maxAbsDiff=%.6g\n", cpuMilliseconds, uploadMilliseconds, static_cast<double>(deviceForwardMilliseconds), downloadMilliseconds, maximumDifference);
}
