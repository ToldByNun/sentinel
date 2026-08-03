#include "CudaTransformerBlock.hpp"

#include "CudaOps.hpp"
#include "../Utils/SmokeLog.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <stdexcept>

void CudaTransformerBlockGradients::ensureFrom(const CudaTransformerBlock& block) {
    this->queryWeight.ensureSize(block.attention.queryWeight.rows, block.attention.queryWeight.cols);
    this->keyWeight.ensureSize(block.attention.keyWeight.rows, block.attention.keyWeight.cols);
    this->valueWeight.ensureSize(block.attention.valueWeight.rows, block.attention.valueWeight.cols);
    this->attentionOutputWeight.ensureSize(block.attention.outputWeight.rows, block.attention.outputWeight.cols);
    this->attentionNormGamma.ensureSize(block.attentionNorm.gamma.rows, block.attentionNorm.gamma.cols);
    this->feedForwardNormGamma.ensureSize(block.feedForwardNorm.gamma.rows, block.feedForwardNorm.gamma.cols);
    this->feedForwardGateWeight.ensureSize(block.feedForward.gateWeight.rows, block.feedForward.gateWeight.cols);
    this->feedForwardGateBias.ensureSize(block.feedForward.gateBias.rows, block.feedForward.gateBias.cols);
    this->feedForwardUpWeight.ensureSize(block.feedForward.upWeight.rows, block.feedForward.upWeight.cols);
    this->feedForwardUpBias.ensureSize(block.feedForward.upBias.rows, block.feedForward.upBias.cols);
    this->feedForwardDownWeight.ensureSize(block.feedForward.downWeight.rows, block.feedForward.downWeight.cols);
    this->feedForwardDownBias.ensureSize(block.feedForward.downBias.rows, block.feedForward.downBias.cols);
}

void CudaTransformerBlockGradients::zeroInPlace() {
    CudaOps::zeroInPlace(this->queryWeight);
    CudaOps::zeroInPlace(this->keyWeight);
    CudaOps::zeroInPlace(this->valueWeight);
    CudaOps::zeroInPlace(this->attentionOutputWeight);
    CudaOps::zeroInPlace(this->attentionNormGamma);
    CudaOps::zeroInPlace(this->feedForwardNormGamma);
    CudaOps::zeroInPlace(this->feedForwardGateWeight);
    CudaOps::zeroInPlace(this->feedForwardGateBias);
    CudaOps::zeroInPlace(this->feedForwardUpWeight);
    CudaOps::zeroInPlace(this->feedForwardUpBias);
    CudaOps::zeroInPlace(this->feedForwardDownWeight);
    CudaOps::zeroInPlace(this->feedForwardDownBias);
}

void CudaTransformerBlockGradients::addInPlace(const CudaTransformerBlockGradients& other) {
    CudaOps::addInPlace(this->queryWeight, other.queryWeight);
    CudaOps::addInPlace(this->keyWeight, other.keyWeight);
    CudaOps::addInPlace(this->valueWeight, other.valueWeight);
    CudaOps::addInPlace(this->attentionOutputWeight, other.attentionOutputWeight);
    CudaOps::addInPlace(this->attentionNormGamma, other.attentionNormGamma);
    CudaOps::addInPlace(this->feedForwardNormGamma, other.feedForwardNormGamma);
    CudaOps::addInPlace(this->feedForwardGateWeight, other.feedForwardGateWeight);
    CudaOps::addInPlace(this->feedForwardGateBias, other.feedForwardGateBias);
    CudaOps::addInPlace(this->feedForwardUpWeight, other.feedForwardUpWeight);
    CudaOps::addInPlace(this->feedForwardUpBias, other.feedForwardUpBias);
    CudaOps::addInPlace(this->feedForwardDownWeight, other.feedForwardDownWeight);
    CudaOps::addInPlace(this->feedForwardDownBias, other.feedForwardDownBias);
}

void CudaTransformerBlockGradients::scaleInPlace(float scalar) {
    CudaOps::scaleInPlace(this->queryWeight, scalar);
    CudaOps::scaleInPlace(this->keyWeight, scalar);
    CudaOps::scaleInPlace(this->valueWeight, scalar);
    CudaOps::scaleInPlace(this->attentionOutputWeight, scalar);
    CudaOps::scaleInPlace(this->attentionNormGamma, scalar);
    CudaOps::scaleInPlace(this->feedForwardNormGamma, scalar);
    CudaOps::scaleInPlace(this->feedForwardGateWeight, scalar);
    CudaOps::scaleInPlace(this->feedForwardGateBias, scalar);
    CudaOps::scaleInPlace(this->feedForwardUpWeight, scalar);
    CudaOps::scaleInPlace(this->feedForwardUpBias, scalar);
    CudaOps::scaleInPlace(this->feedForwardDownWeight, scalar);
    CudaOps::scaleInPlace(this->feedForwardDownBias, scalar);
}

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

void CudaTransformerBlock::forward(const CudaMatrix& input, CudaMatrix& out, int segmentLength) {
    if (input.empty()) throw std::invalid_argument("CudaTransformerBlock::forward empty input");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaTransformerBlock::forward no CUDA device");

    this->attentionNorm.forward(input, this->attentionInput);
    this->attention.forward(this->attentionInput, this->attended, segmentLength);
    // Epilogue fuse: afterAttention = input + attended, feedForwardInput = RMSNorm(afterAttention)
    this->feedForwardNorm.forwardFromResidual(input, this->attended, this->afterAttention, this->feedForwardInput);
    this->feedForward.forward(this->feedForwardInput, this->feedForwardOutput);
    CudaOps::addInto(this->afterAttention, this->feedForwardOutput, out);
}

void CudaTransformerBlock::forwardSelectiveTrain(const CudaMatrix& input, CudaMatrix& out, int segmentLength) {
    this->forward(input, out, segmentLength);
    // Drop Attn activations; FFN + afterAttention + feedForwardNorm caches stay for bwd.
    this->attention.releaseActivationScratch();
    this->attended.free();
    this->attentionInput.free();
}

void CudaTransformerBlock::recomputeAttention(const CudaMatrix& blockInput, int segmentLength) {
    if (blockInput.empty()) throw std::invalid_argument("CudaTransformerBlock::recomputeAttention empty blockInput");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaTransformerBlock::recomputeAttention no CUDA device");

    this->attentionNorm.forward(blockInput, this->attentionInput);
    this->attention.forward(this->attentionInput, this->attended, segmentLength);
    CudaOps::addInto(blockInput, this->attended, this->afterAttention);
}

void CudaTransformerBlock::prefill(const CudaMatrix& input, CudaKvCache& cache, CudaMatrix& out) {
    if (input.empty()) throw std::invalid_argument("CudaTransformerBlock::prefill empty input");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaTransformerBlock::prefill no CUDA device");

    this->attentionNorm.forward(input, this->attentionInput);
    this->attention.prefill(this->attentionInput, cache, this->attended);
    this->feedForwardNorm.forwardFromResidual(input, this->attended, this->afterAttention, this->feedForwardInput);
    this->feedForward.forward(this->feedForwardInput, this->feedForwardOutput);
    CudaOps::addInto(this->afterAttention, this->feedForwardOutput, out);
}

void CudaTransformerBlock::decode(const CudaMatrix& input, CudaKvCache& cache, CudaMatrix& out) {
    if (input.empty()) throw std::invalid_argument("CudaTransformerBlock::decode empty input");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaTransformerBlock::decode no CUDA device");

    this->attentionNorm.forward(input, this->attentionInput);
    this->attention.decode(this->attentionInput, cache, this->attended);
    this->feedForwardNorm.forwardFromResidual(input, this->attended, this->afterAttention, this->feedForwardInput);
    this->feedForward.forward(this->feedForwardInput, this->feedForwardOutput);
    CudaOps::addInto(this->afterAttention, this->feedForwardOutput, out);
}

void CudaTransformerBlock::backward(const CudaMatrix& outputGradient, CudaMatrix& inputGradient, CudaTransformerBlockGradients& gradients) {
    if (outputGradient.empty()) throw std::invalid_argument("CudaTransformerBlock::backward empty outputGradient");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaTransformerBlock::backward no CUDA device");

    this->feedForward.backward(outputGradient, this->feedForwardInputGradient, this->feedForwardGateWeightGradient, this->feedForwardGateBiasGradient, this->feedForwardUpWeightGradient, this->feedForwardUpBiasGradient, this->feedForwardDownWeightGradient, this->feedForwardDownBiasGradient);
    // afterAttentionGrad = outputGrad + d(FFN-RMSNorm)/d(afterAttention)
    this->feedForwardNorm.backwardThroughResidual(
        this->feedForwardInputGradient,
        outputGradient,
        this->afterAttentionGradient,
        this->feedForwardNormGammaGradient);

    this->attention.backward(this->afterAttentionGradient, this->attentionInputGradient, this->queryWeightGradient, this->keyWeightGradient, this->valueWeightGradient, this->attentionOutputWeightGradient);
    // inputGrad = afterAttentionGrad + d(Attn-RMSNorm)/d(input)
    this->attentionNorm.backwardThroughResidual(
        this->attentionInputGradient,
        this->afterAttentionGradient,
        inputGradient,
        this->attentionNormGammaGradient);

    CudaOps::addInPlace(gradients.feedForwardDownWeight, this->feedForwardDownWeightGradient);
    CudaOps::addInPlace(gradients.feedForwardDownBias, this->feedForwardDownBiasGradient);
    CudaOps::addInPlace(gradients.feedForwardUpWeight, this->feedForwardUpWeightGradient);
    CudaOps::addInPlace(gradients.feedForwardUpBias, this->feedForwardUpBiasGradient);
    CudaOps::addInPlace(gradients.feedForwardGateWeight, this->feedForwardGateWeightGradient);
    CudaOps::addInPlace(gradients.feedForwardGateBias, this->feedForwardGateBiasGradient);
    CudaOps::addInPlace(gradients.feedForwardNormGamma, this->feedForwardNormGammaGradient);
    CudaOps::addInPlace(gradients.attentionOutputWeight, this->attentionOutputWeightGradient);
    CudaOps::addInPlace(gradients.valueWeight, this->valueWeightGradient);
    CudaOps::addInPlace(gradients.keyWeight, this->keyWeightGradient);
    CudaOps::addInPlace(gradients.queryWeight, this->queryWeightGradient);
    CudaOps::addInPlace(gradients.attentionNormGamma, this->attentionNormGammaGradient);
}

void CudaTransformerBlock::backwardSelective(
    const CudaMatrix& outputGradient,
    CudaMatrix& inputGradient,
    CudaTransformerBlockGradients& gradients,
    const CudaMatrix& blockInput,
    int segmentLength
) {
    if (outputGradient.empty()) throw std::invalid_argument("CudaTransformerBlock::backwardSelective empty outputGradient");
    if (blockInput.empty()) throw std::invalid_argument("CudaTransformerBlock::backwardSelective empty blockInput");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaTransformerBlock::backwardSelective no CUDA device");

    // FFN path from kept activations (no SwiGLU recompute).
    this->feedForward.backward(outputGradient, this->feedForwardInputGradient, this->feedForwardGateWeightGradient, this->feedForwardGateBiasGradient, this->feedForwardUpWeightGradient, this->feedForwardUpBiasGradient, this->feedForwardDownWeightGradient, this->feedForwardDownBiasGradient);
    this->feedForwardNorm.backwardThroughResidual(
        this->feedForwardInputGradient,
        outputGradient,
        this->afterAttentionGradient,
        this->feedForwardNormGammaGradient);

    // Refresh Attn activations from saved block input, then Attn bwd.
    this->recomputeAttention(blockInput, segmentLength);
    this->attention.backward(this->afterAttentionGradient, this->attentionInputGradient, this->queryWeightGradient, this->keyWeightGradient, this->valueWeightGradient, this->attentionOutputWeightGradient);
    this->attentionNorm.backwardThroughResidual(
        this->attentionInputGradient,
        this->afterAttentionGradient,
        inputGradient,
        this->attentionNormGammaGradient);

    CudaOps::addInPlace(gradients.feedForwardDownWeight, this->feedForwardDownWeightGradient);
    CudaOps::addInPlace(gradients.feedForwardDownBias, this->feedForwardDownBiasGradient);
    CudaOps::addInPlace(gradients.feedForwardUpWeight, this->feedForwardUpWeightGradient);
    CudaOps::addInPlace(gradients.feedForwardUpBias, this->feedForwardUpBiasGradient);
    CudaOps::addInPlace(gradients.feedForwardGateWeight, this->feedForwardGateWeightGradient);
    CudaOps::addInPlace(gradients.feedForwardGateBias, this->feedForwardGateBiasGradient);
    CudaOps::addInPlace(gradients.feedForwardNormGamma, this->feedForwardNormGammaGradient);
    CudaOps::addInPlace(gradients.attentionOutputWeight, this->attentionOutputWeightGradient);
    CudaOps::addInPlace(gradients.valueWeight, this->valueWeightGradient);
    CudaOps::addInPlace(gradients.keyWeight, this->keyWeightGradient);
    CudaOps::addInPlace(gradients.queryWeight, this->queryWeightGradient);
    CudaOps::addInPlace(gradients.attentionNormGamma, this->attentionNormGammaGradient);
}
void CudaTransformerBlock::runSmokeDemo(int embeddingDim, int headCount, int sequenceLength, int maximumPositionCount) {
    if (!CudaMatmul::isAvailable()) {
        SmokeLog::skip("TransformerBlock fwd");
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

    SmokeLog::result("TransformerBlock fwd", "embed=%d heads=%d seq=%d  cpu=%.2fms  gpu=%.2fms  diff=%.2e",
        embeddingDim, headCount, sequenceLength, cpuMilliseconds, static_cast<double>(deviceMilliseconds), maximumDifference);
}

void CudaTransformerBlock::runBackwardSmokeDemo(int embeddingDim, int headCount, int sequenceLength, int maximumPositionCount) {
    if (!CudaMatmul::isAvailable()) {
        SmokeLog::skip("TransformerBlock bwd");
        return;
    }
    if (embeddingDim <= 0 || headCount <= 0 || sequenceLength <= 0 || maximumPositionCount < sequenceLength)
        throw std::invalid_argument("CudaTransformerBlock::runBackwardSmokeDemo invalid dims");

    TransformerBlock host(embeddingDim, headCount, maximumPositionCount, 21u);
    Matrix hostInput(static_cast<size_t>(embeddingDim), static_cast<size_t>(sequenceLength), 0.0f);
    Matrix hostOutputGradient(static_cast<size_t>(embeddingDim), static_cast<size_t>(sequenceLength), 0.0f);
    unsigned state = 79u;
    for (size_t index = 0; index < hostInput.data.size(); ++index) {
        state = state * 1664525u + 1013904223u;
        hostInput.data[index] = (static_cast<float>(state >> 8) / 16777216.0f) * 2.0f - 1.0f;
        state = state * 1664525u + 1013904223u;
        hostOutputGradient.data[index] = (static_cast<float>(state >> 8) / 16777216.0f) * 2.0f - 1.0f;
    }

    TransformerBlockCache hostCache;
    host.forward(hostInput, hostCache);
    TransformerBlockGradients hostGradients = TransformerBlockGradients::zerosFrom(host);
    host.backward(hostOutputGradient, hostCache, hostGradients);

    CudaTransformerBlock device = CudaTransformerBlock::createFrom(host);
    CudaMatrix deviceInput;
    deviceInput.upload(hostInput);
    CudaMatrix deviceOutput;
    device.forward(deviceInput, deviceOutput);

    CudaMatrix deviceOutputGradient;
    deviceOutputGradient.upload(hostOutputGradient);
    CudaTransformerBlockGradients deviceGradients;
    deviceGradients.ensureFrom(device);
    deviceGradients.zeroInPlace();
    CudaMatrix deviceInputGradient;
    device.backward(deviceOutputGradient, deviceInputGradient, deviceGradients);
    CudaMatmul::throwIfCudaFailed(cudaDeviceSynchronize(), "CudaTransformerBlock backward smoke synchronize");

    Matrix deviceQueryWeightGrad = deviceGradients.queryWeight.download();
    float maximumDifference = 0.0f;
    for (size_t index = 0; index < hostGradients.queryWeight.data.size(); ++index)
        maximumDifference = (std::max)(maximumDifference, std::fabs(hostGradients.queryWeight.data[index] - deviceQueryWeightGrad.data[index]));

    SmokeLog::result("TransformerBlock bwd", "embed=%d heads=%d seq=%d  diff=%.2e", embeddingDim, headCount, sequenceLength, maximumDifference);
}
