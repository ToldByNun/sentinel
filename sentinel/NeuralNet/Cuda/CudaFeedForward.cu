#include "CudaFeedForward.hpp"

#include "CudaOps.hpp"
#include "../Utils/SmokeLog.hpp"

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
    CudaOps::copyInto(input, this->inputCache);
}

void CudaFeedForward::backward(const CudaMatrix& outputGradient, CudaMatrix& inputGradient, CudaMatrix& gateWeightGradient, CudaMatrix& gateBiasGradient, CudaMatrix& upWeightGradient, CudaMatrix& upBiasGradient, CudaMatrix& downWeightGradient, CudaMatrix& downBiasGradient) {
    if (this->inputCache.empty()) throw std::logic_error("CudaFeedForward::backward called before forward");
    if (outputGradient.rows != this->downWeight.rows || outputGradient.cols != this->inputCache.cols)
        throw std::invalid_argument("CudaFeedForward::backward shape mismatch");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaFeedForward::backward no CUDA device");

    CudaMatrix::multiplyInto(outputGradient, this->hidden, downWeightGradient, false, true);
    CudaOps::sumColumnsInto(outputGradient, downBiasGradient);

    CudaMatrix::multiplyInto(this->downWeight, outputGradient, this->hiddenGradient, true, false);
    CudaOps::multiplyElementwiseInto(this->hiddenGradient, this->gateActivated, this->upGradient);
    CudaOps::multiplyElementwiseInto(this->hiddenGradient, this->up, this->gateGradient);

    CudaOps::siluDerivativeInto(this->gatePreActivation, this->siluDerivative);
    CudaOps::multiplyElementwiseInPlace(this->gateGradient, this->siluDerivative);

    CudaMatrix::multiplyInto(this->gateGradient, this->inputCache, gateWeightGradient, false, true);
    CudaOps::sumColumnsInto(this->gateGradient, gateBiasGradient);
    CudaMatrix::multiplyInto(this->upGradient, this->inputCache, upWeightGradient, false, true);
    CudaOps::sumColumnsInto(this->upGradient, upBiasGradient);

    CudaMatrix::multiplyInto(this->gateWeight, this->gateGradient, inputGradient, true, false);
    CudaMatrix::multiplyInto(this->upWeight, this->upGradient, this->temp, true, false);
    CudaOps::addInPlace(inputGradient, this->temp);
}

static float maximumAbsoluteDifference(const Matrix& left, const Matrix& right) {
    if (left.rows != right.rows || left.cols != right.cols) throw std::invalid_argument("maximumAbsoluteDifference shape mismatch");
    float maximumDifference = 0.0f;
    for (size_t index = 0; index < left.data.size(); ++index)
        maximumDifference = (std::max)(maximumDifference, std::fabs(left.data[index] - right.data[index]));
    return maximumDifference;
}

void CudaFeedForward::runSmokeDemo(int embeddingDim, int sequenceLength) {
    if (!CudaMatmul::isAvailable()) {
        SmokeLog::skip("FeedForward fwd");
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

    SmokeLog::result("FeedForward fwd", "embed=%d seq=%d  cpu=%.2fms  gpu=%.2fms  diff=%.2e",
        embeddingDim, sequenceLength, cpuMilliseconds, static_cast<double>(deviceForwardMilliseconds), maximumDifference);
    (void)uploadMilliseconds;
    (void)downloadMilliseconds;
}

void CudaFeedForward::runBackwardSmokeDemo(int embeddingDim, int sequenceLength) {
    if (!CudaMatmul::isAvailable()) {
        SmokeLog::skip("FeedForward bwd");
        return;
    }
    if (embeddingDim <= 0) throw std::invalid_argument("CudaFeedForward::runBackwardSmokeDemo embeddingDim must be > 0");
    if (sequenceLength <= 0) throw std::invalid_argument("CudaFeedForward::runBackwardSmokeDemo sequenceLength must be > 0");

    FeedForward host = FeedForward::create(embeddingDim, 4, 41u);
    Matrix hostInput(static_cast<size_t>(embeddingDim), static_cast<size_t>(sequenceLength), 0.0f);
    unsigned state = 99u;
    for (size_t index = 0; index < hostInput.data.size(); ++index) {
        state = state * 1664525u + 1013904223u;
        hostInput.data[index] = (static_cast<float>(state >> 8) / 16777216.0f) * 2.0f - 1.0f;
    }

    FeedForwardCache hostCache;
    host.forward(hostInput, hostCache);

    Matrix outputGradient(host.downWeight.rows, hostInput.cols, 1.0f);
    Matrix hostGateWeightGradient;
    Matrix hostGateBiasGradient;
    Matrix hostUpWeightGradient;
    Matrix hostUpBiasGradient;
    Matrix hostDownWeightGradient;
    Matrix hostDownBiasGradient;
    Matrix hostInputGradient = host.backward(outputGradient, hostCache, hostGateWeightGradient, hostGateBiasGradient, hostUpWeightGradient, hostUpBiasGradient, hostDownWeightGradient, hostDownBiasGradient);

    CudaFeedForward device = CudaFeedForward::createFrom(host);
    CudaMatrix deviceInput;
    deviceInput.upload(hostInput);
    CudaMatrix deviceOutput;
    device.forward(deviceInput, deviceOutput);

    CudaMatrix deviceOutputGradient;
    deviceOutputGradient.upload(outputGradient);

    CudaMatrix deviceInputGradient;
    CudaMatrix deviceGateWeightGradient;
    CudaMatrix deviceGateBiasGradient;
    CudaMatrix deviceUpWeightGradient;
    CudaMatrix deviceUpBiasGradient;
    CudaMatrix deviceDownWeightGradient;
    CudaMatrix deviceDownBiasGradient;
    device.backward(deviceOutputGradient, deviceInputGradient, deviceGateWeightGradient, deviceGateBiasGradient, deviceUpWeightGradient, deviceUpBiasGradient, deviceDownWeightGradient, deviceDownBiasGradient);
    CudaMatmul::throwIfCudaFailed(cudaDeviceSynchronize(), "CudaFeedForward backward synchronize");

    const float inputGradientDiff = maximumAbsoluteDifference(hostInputGradient, deviceInputGradient.download());
    const float gateWeightGradientDiff = maximumAbsoluteDifference(hostGateWeightGradient, deviceGateWeightGradient.download());
    const float gateBiasGradientDiff = maximumAbsoluteDifference(hostGateBiasGradient, deviceGateBiasGradient.download());
    const float upWeightGradientDiff = maximumAbsoluteDifference(hostUpWeightGradient, deviceUpWeightGradient.download());
    const float upBiasGradientDiff = maximumAbsoluteDifference(hostUpBiasGradient, deviceUpBiasGradient.download());
    const float downWeightGradientDiff = maximumAbsoluteDifference(hostDownWeightGradient, deviceDownWeightGradient.download());
    const float downBiasGradientDiff = maximumAbsoluteDifference(hostDownBiasGradient, deviceDownBiasGradient.download());

    const float maximumDifference = (std::max)({ inputGradientDiff, gateWeightGradientDiff, gateBiasGradientDiff, upWeightGradientDiff, upBiasGradientDiff, downWeightGradientDiff, downBiasGradientDiff });
    SmokeLog::result("FeedForward bwd", "embed=%d seq=%d  diff=%.2e", embeddingDim, sequenceLength, maximumDifference);
}
