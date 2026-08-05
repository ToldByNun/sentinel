#include "CudaTransformerBlock.hpp"

#include "CudaOps.hpp"
#include "../Utils/SmokeLog.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <stdexcept>

void CudaTransformerBlockGradients::ensureFrom(const CudaTransformerBlock& block, bool largeWeightsOnHost) {
    if (largeWeightsOnHost) {
        this->queryWeight.free();
        this->keyWeight.free();
        this->valueWeight.free();
        this->attentionOutputWeight.free();
        this->feedForwardGateWeight.free();
        this->feedForwardUpWeight.free();
        this->feedForwardDownWeight.free();
    } else {
        this->queryWeight.ensureSize(block.attention.queryWeight.rows, block.attention.queryWeight.cols);
        this->keyWeight.ensureSize(block.attention.keyWeight.rows, block.attention.keyWeight.cols);
        this->valueWeight.ensureSize(block.attention.valueWeight.rows, block.attention.valueWeight.cols);
        this->attentionOutputWeight.ensureSize(block.attention.outputWeight.rows, block.attention.outputWeight.cols);
        this->feedForwardGateWeight.ensureSize(block.feedForward.gateWeight.rows, block.feedForward.gateWeight.cols);
        this->feedForwardUpWeight.ensureSize(block.feedForward.upWeight.rows, block.feedForward.upWeight.cols);
        this->feedForwardDownWeight.ensureSize(block.feedForward.downWeight.rows, block.feedForward.downWeight.cols);
    }
    this->attentionNormGamma.ensureSize(block.attentionNorm.gamma.rows, block.attentionNorm.gamma.cols);
    this->feedForwardNormGamma.ensureSize(block.feedForwardNorm.gamma.rows, block.feedForwardNorm.gamma.cols);
    this->feedForwardGateBias.ensureSize(block.feedForward.gateBias.rows, block.feedForward.gateBias.cols);
    this->feedForwardUpBias.ensureSize(block.feedForward.upBias.rows, block.feedForward.upBias.cols);
    this->feedForwardDownBias.ensureSize(block.feedForward.downBias.rows, block.feedForward.downBias.cols);
}

void CudaTransformerBlockGradients::zeroInPlace() {
    if (!this->queryWeight.empty()) CudaOps::zeroInPlace(this->queryWeight);
    if (!this->keyWeight.empty()) CudaOps::zeroInPlace(this->keyWeight);
    if (!this->valueWeight.empty()) CudaOps::zeroInPlace(this->valueWeight);
    if (!this->attentionOutputWeight.empty()) CudaOps::zeroInPlace(this->attentionOutputWeight);
    CudaOps::zeroInPlace(this->attentionNormGamma);
    CudaOps::zeroInPlace(this->feedForwardNormGamma);
    if (!this->feedForwardGateWeight.empty()) CudaOps::zeroInPlace(this->feedForwardGateWeight);
    CudaOps::zeroInPlace(this->feedForwardGateBias);
    if (!this->feedForwardUpWeight.empty()) CudaOps::zeroInPlace(this->feedForwardUpWeight);
    CudaOps::zeroInPlace(this->feedForwardUpBias);
    if (!this->feedForwardDownWeight.empty()) CudaOps::zeroInPlace(this->feedForwardDownWeight);
    CudaOps::zeroInPlace(this->feedForwardDownBias);
}

void CudaTransformerBlockGradients::addInPlace(const CudaTransformerBlockGradients& other) {
    if (!this->queryWeight.empty()) CudaOps::addInPlace(this->queryWeight, other.queryWeight);
    if (!this->keyWeight.empty()) CudaOps::addInPlace(this->keyWeight, other.keyWeight);
    if (!this->valueWeight.empty()) CudaOps::addInPlace(this->valueWeight, other.valueWeight);
    if (!this->attentionOutputWeight.empty()) CudaOps::addInPlace(this->attentionOutputWeight, other.attentionOutputWeight);
    CudaOps::addInPlace(this->attentionNormGamma, other.attentionNormGamma);
    CudaOps::addInPlace(this->feedForwardNormGamma, other.feedForwardNormGamma);
    if (!this->feedForwardGateWeight.empty()) CudaOps::addInPlace(this->feedForwardGateWeight, other.feedForwardGateWeight);
    CudaOps::addInPlace(this->feedForwardGateBias, other.feedForwardGateBias);
    if (!this->feedForwardUpWeight.empty()) CudaOps::addInPlace(this->feedForwardUpWeight, other.feedForwardUpWeight);
    CudaOps::addInPlace(this->feedForwardUpBias, other.feedForwardUpBias);
    if (!this->feedForwardDownWeight.empty()) CudaOps::addInPlace(this->feedForwardDownWeight, other.feedForwardDownWeight);
    CudaOps::addInPlace(this->feedForwardDownBias, other.feedForwardDownBias);
}

void CudaTransformerBlockGradients::scaleInPlace(float scalar) {
    if (!this->queryWeight.empty()) CudaOps::scaleInPlace(this->queryWeight, scalar);
    if (!this->keyWeight.empty()) CudaOps::scaleInPlace(this->keyWeight, scalar);
    if (!this->valueWeight.empty()) CudaOps::scaleInPlace(this->valueWeight, scalar);
    if (!this->attentionOutputWeight.empty()) CudaOps::scaleInPlace(this->attentionOutputWeight, scalar);
    CudaOps::scaleInPlace(this->attentionNormGamma, scalar);
    CudaOps::scaleInPlace(this->feedForwardNormGamma, scalar);
    if (!this->feedForwardGateWeight.empty()) CudaOps::scaleInPlace(this->feedForwardGateWeight, scalar);
    CudaOps::scaleInPlace(this->feedForwardGateBias, scalar);
    if (!this->feedForwardUpWeight.empty()) CudaOps::scaleInPlace(this->feedForwardUpWeight, scalar);
    CudaOps::scaleInPlace(this->feedForwardUpBias, scalar);
    if (!this->feedForwardDownWeight.empty()) CudaOps::scaleInPlace(this->feedForwardDownWeight, scalar);
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
    // Drop Attn activations; keep compact FP16 FFN stash + FFN-RMSNorm caches.
    this->attention.releaseActivationScratch();
    this->attended.free();
    this->attentionInput.free();
    this->attentionNorm.releaseActivationScratch();
    // afterAttention values are unused by FFN-norm bwd (lastNormalized + invRms suffice).
    this->afterAttention.free();
    this->feedForwardInput.free();
    this->feedForwardOutput.free();
    this->feedForward.stashSelectiveHalfActivations();
}

void CudaTransformerBlock::releaseTrainActivationScratch() {
    this->releaseTrainActivationScratch(false);
}

void CudaTransformerBlock::releaseTrainActivationScratch(bool keepDeferredHostWeightGrads) {
    this->attention.releaseActivationScratch();
    this->attention.qkvWeightGradient.free();
    this->feedForward.releaseActivationScratch();
    this->attentionNorm.releaseActivationScratch();
    this->feedForwardNorm.releaseActivationScratch();

    this->attentionInput.releaseDeviceToPool();
    this->attended.releaseDeviceToPool();
    this->afterAttention.releaseDeviceToPool();
    this->feedForwardInput.releaseDeviceToPool();
    this->feedForwardOutput.releaseDeviceToPool();

    this->feedForwardInputGradient.releaseDeviceToPool();
    this->afterAttentionFromFeedForward.releaseDeviceToPool();
    this->afterAttentionGradient.releaseDeviceToPool();
    this->attentionInputGradient.releaseDeviceToPool();
    this->inputFromAttention.releaseDeviceToPool();

    if (!keepDeferredHostWeightGrads) {
        this->feedForwardGateWeightGradient.free();
        this->feedForwardUpWeightGradient.free();
        this->feedForwardDownWeightGradient.free();
        this->queryWeightGradient.free();
        this->keyWeightGradient.free();
        this->valueWeightGradient.free();
        this->attentionOutputWeightGradient.free();
    }
    this->feedForwardGateBiasGradient.free();
    this->feedForwardUpBiasGradient.free();
    this->feedForwardDownBiasGradient.free();
    this->feedForwardNormGammaGradient.free();
    this->attentionNormGammaGradient.free();
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

void CudaTransformerBlock::backward(
    const CudaMatrix& outputGradient,
    CudaMatrix& inputGradient,
    CudaTransformerBlockGradients& gradients,
    CudaTransformerBlockHostWeightGrads* hostWeightGrads,
    bool deferHostWeightDownload,
    bool retainActivationScratch
) {
    if (outputGradient.empty()) throw std::invalid_argument("CudaTransformerBlock::backward empty outputGradient");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaTransformerBlock::backward no CUDA device");
    if (deferHostWeightDownload && hostWeightGrads == nullptr)
        throw std::invalid_argument("CudaTransformerBlock::backward deferHostWeightDownload requires hostWeightGrads");

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

    auto accumulateWeight = [](Matrix* host, CudaMatrix& deviceWorkspace, CudaMatrix& deviceAccum) {
        if (host != nullptr) {
            CudaOps::downloadAddIntoHost(*host, deviceWorkspace);
            deviceWorkspace.free();
        } else {
            CudaOps::addInPlace(deviceAccum, deviceWorkspace);
        }
    };

    if (hostWeightGrads != nullptr && deferHostWeightDownload) {
        // Large weight grads stay on device for async D2H; fused QKV/gateUp copies already materialized.
        this->attention.qkvWeightGradient.free();
        this->feedForward.gateUpWeightGradient.free();
    } else if (hostWeightGrads != nullptr) {
        accumulateWeight(hostWeightGrads->feedForwardDownWeight, this->feedForwardDownWeightGradient, gradients.feedForwardDownWeight);
        accumulateWeight(hostWeightGrads->feedForwardUpWeight, this->feedForwardUpWeightGradient, gradients.feedForwardUpWeight);
        accumulateWeight(hostWeightGrads->feedForwardGateWeight, this->feedForwardGateWeightGradient, gradients.feedForwardGateWeight);
        accumulateWeight(hostWeightGrads->attentionOutputWeight, this->attentionOutputWeightGradient, gradients.attentionOutputWeight);
        accumulateWeight(hostWeightGrads->valueWeight, this->valueWeightGradient, gradients.valueWeight);
        accumulateWeight(hostWeightGrads->keyWeight, this->keyWeightGradient, gradients.keyWeight);
        accumulateWeight(hostWeightGrads->queryWeight, this->queryWeightGradient, gradients.queryWeight);
    } else {
        CudaOps::addInPlace(gradients.feedForwardDownWeight, this->feedForwardDownWeightGradient);
        CudaOps::addInPlace(gradients.feedForwardUpWeight, this->feedForwardUpWeightGradient);
        CudaOps::addInPlace(gradients.feedForwardGateWeight, this->feedForwardGateWeightGradient);
        CudaOps::addInPlace(gradients.attentionOutputWeight, this->attentionOutputWeightGradient);
        CudaOps::addInPlace(gradients.valueWeight, this->valueWeightGradient);
        CudaOps::addInPlace(gradients.keyWeight, this->keyWeightGradient);
        CudaOps::addInPlace(gradients.queryWeight, this->queryWeightGradient);
    }
    CudaOps::addInPlace(gradients.feedForwardDownBias, this->feedForwardDownBiasGradient);
    CudaOps::addInPlace(gradients.feedForwardUpBias, this->feedForwardUpBiasGradient);
    CudaOps::addInPlace(gradients.feedForwardGateBias, this->feedForwardGateBiasGradient);
    CudaOps::addInPlace(gradients.feedForwardNormGamma, this->feedForwardNormGammaGradient);
    CudaOps::addInPlace(gradients.attentionNormGamma, this->attentionNormGammaGradient);

    if (retainActivationScratch)
        return;

    if (hostWeightGrads != nullptr) {
        if (!deferHostWeightDownload) {
            this->attention.qkvWeightGradient.free();
            this->feedForward.gateUpWeightGradient.free();
        }
        this->feedForwardDownBiasGradient.free();
        this->feedForwardUpBiasGradient.free();
        this->feedForwardGateBiasGradient.free();
        this->feedForwardNormGammaGradient.free();
        this->attentionNormGammaGradient.free();
    }

    this->releaseTrainActivationScratch(deferHostWeightDownload);
}

void CudaTransformerBlock::enqueueDeferredHostWeightGradDownloads(
    CudaTransformerBlockHostWeightGrads& hostWeightGrads,
    cudaStream_t copyStream,
    cudaEvent_t gradsReadyOnCompute
) {
    if (copyStream == nullptr) throw std::invalid_argument("enqueueDeferredHostWeightGradDownloads null copyStream");
    if (gradsReadyOnCompute == nullptr) throw std::invalid_argument("enqueueDeferredHostWeightGradDownloads null event");
    if (hostWeightGrads.queryWeight == nullptr || hostWeightGrads.keyWeight == nullptr || hostWeightGrads.valueWeight == nullptr
        || hostWeightGrads.attentionOutputWeight == nullptr || hostWeightGrads.feedForwardGateWeight == nullptr
        || hostWeightGrads.feedForwardUpWeight == nullptr || hostWeightGrads.feedForwardDownWeight == nullptr)
        throw std::invalid_argument("enqueueDeferredHostWeightGradDownloads missing host grads");

    CudaMatmul::throwIfCudaFailed(
        cudaStreamWaitEvent(copyStream, gradsReadyOnCompute, 0),
        "enqueueDeferredHostWeightGradDownloads wait compute");

    auto enqueueOne = [&](Matrix* host, CudaMatrix& device) {
        if (device.empty()) throw std::logic_error("enqueueDeferredHostWeightGradDownloads empty device grad");
        CudaOps::downloadIntoHostAsync(*host, device, copyStream);
    };
    enqueueOne(hostWeightGrads.feedForwardDownWeight, this->feedForwardDownWeightGradient);
    enqueueOne(hostWeightGrads.feedForwardUpWeight, this->feedForwardUpWeightGradient);
    enqueueOne(hostWeightGrads.feedForwardGateWeight, this->feedForwardGateWeightGradient);
    enqueueOne(hostWeightGrads.attentionOutputWeight, this->attentionOutputWeightGradient);
    enqueueOne(hostWeightGrads.valueWeight, this->valueWeightGradient);
    enqueueOne(hostWeightGrads.keyWeight, this->keyWeightGradient);
    enqueueOne(hostWeightGrads.queryWeight, this->queryWeightGradient);
}

void CudaTransformerBlock::releaseDeferredHostWeightGradDevice() {
    this->feedForwardDownWeightGradient.free();
    this->feedForwardUpWeightGradient.free();
    this->feedForwardGateWeightGradient.free();
    this->attentionOutputWeightGradient.free();
    this->valueWeightGradient.free();
    this->keyWeightGradient.free();
    this->queryWeightGradient.free();
}

void CudaTransformerBlock::backwardSelective(
    const CudaMatrix& outputGradient,
    CudaMatrix& inputGradient,
    CudaTransformerBlockGradients& gradients,
    const CudaMatrix& blockInput,
    int segmentLength,
    CudaTransformerBlockHostWeightGrads* hostWeightGrads,
    bool deferHostWeightDownload
) {
    if (outputGradient.empty()) throw std::invalid_argument("CudaTransformerBlock::backwardSelective empty outputGradient");
    if (blockInput.empty()) throw std::invalid_argument("CudaTransformerBlock::backwardSelective empty blockInput");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaTransformerBlock::backwardSelective no CUDA device");
    if (deferHostWeightDownload && hostWeightGrads == nullptr)
        throw std::invalid_argument("CudaTransformerBlock::backwardSelective deferHostWeightDownload requires hostWeightGrads");

    // FFN path from kept half activations — silu/hidden/inputCache restored, no SwiGLU GEMM recompute.
    if (this->feedForward.selectiveHalfStashed())
        this->feedForward.restoreSelectiveHalfActivations(this->feedForwardNorm);
    this->feedForward.backward(outputGradient, this->feedForwardInputGradient, this->feedForwardGateWeightGradient, this->feedForwardGateBiasGradient, this->feedForwardUpWeightGradient, this->feedForwardUpBiasGradient, this->feedForwardDownWeightGradient, this->feedForwardDownBiasGradient);
    this->feedForwardNorm.backwardThroughResidual(
        this->feedForwardInputGradient,
        outputGradient,
        this->afterAttentionGradient,
        this->feedForwardNormGammaGradient);
    // Drop restored FP32 FFN acts before Attn recompute to cut peak VRAM.
    this->feedForward.releaseRestoredSelectiveFp32Activations();

    // Refresh Attn activations from saved block input, then Attn bwd.
    this->recomputeAttention(blockInput, segmentLength);
    this->attention.backward(this->afterAttentionGradient, this->attentionInputGradient, this->queryWeightGradient, this->keyWeightGradient, this->valueWeightGradient, this->attentionOutputWeightGradient);
    this->attentionNorm.backwardThroughResidual(
        this->attentionInputGradient,
        this->afterAttentionGradient,
        inputGradient,
        this->attentionNormGammaGradient);

    auto accumulateWeight = [](Matrix* host, CudaMatrix& deviceWorkspace, CudaMatrix& deviceAccum) {
        if (host != nullptr) {
            CudaOps::downloadAddIntoHost(*host, deviceWorkspace);
            deviceWorkspace.free();
        } else {
            CudaOps::addInPlace(deviceAccum, deviceWorkspace);
        }
    };

    if (hostWeightGrads != nullptr && deferHostWeightDownload) {
        this->attention.qkvWeightGradient.free();
        this->feedForward.gateUpWeightGradient.free();
    } else if (hostWeightGrads != nullptr) {
        accumulateWeight(hostWeightGrads->feedForwardDownWeight, this->feedForwardDownWeightGradient, gradients.feedForwardDownWeight);
        accumulateWeight(hostWeightGrads->feedForwardUpWeight, this->feedForwardUpWeightGradient, gradients.feedForwardUpWeight);
        accumulateWeight(hostWeightGrads->feedForwardGateWeight, this->feedForwardGateWeightGradient, gradients.feedForwardGateWeight);
        accumulateWeight(hostWeightGrads->attentionOutputWeight, this->attentionOutputWeightGradient, gradients.attentionOutputWeight);
        accumulateWeight(hostWeightGrads->valueWeight, this->valueWeightGradient, gradients.valueWeight);
        accumulateWeight(hostWeightGrads->keyWeight, this->keyWeightGradient, gradients.keyWeight);
        accumulateWeight(hostWeightGrads->queryWeight, this->queryWeightGradient, gradients.queryWeight);
    } else {
        CudaOps::addInPlace(gradients.feedForwardDownWeight, this->feedForwardDownWeightGradient);
        CudaOps::addInPlace(gradients.feedForwardUpWeight, this->feedForwardUpWeightGradient);
        CudaOps::addInPlace(gradients.feedForwardGateWeight, this->feedForwardGateWeightGradient);
        CudaOps::addInPlace(gradients.attentionOutputWeight, this->attentionOutputWeightGradient);
        CudaOps::addInPlace(gradients.valueWeight, this->valueWeightGradient);
        CudaOps::addInPlace(gradients.keyWeight, this->keyWeightGradient);
        CudaOps::addInPlace(gradients.queryWeight, this->queryWeightGradient);
    }
    CudaOps::addInPlace(gradients.feedForwardDownBias, this->feedForwardDownBiasGradient);
    CudaOps::addInPlace(gradients.feedForwardUpBias, this->feedForwardUpBiasGradient);
    CudaOps::addInPlace(gradients.feedForwardGateBias, this->feedForwardGateBiasGradient);
    CudaOps::addInPlace(gradients.feedForwardNormGamma, this->feedForwardNormGammaGradient);
    CudaOps::addInPlace(gradients.attentionNormGamma, this->attentionNormGammaGradient);

    if (hostWeightGrads != nullptr) {
        if (!deferHostWeightDownload) {
            this->attention.qkvWeightGradient.free();
            this->feedForward.gateUpWeightGradient.free();
        }
        this->feedForwardDownBiasGradient.free();
        this->feedForwardUpBiasGradient.free();
        this->feedForwardGateBiasGradient.free();
        this->feedForwardNormGammaGradient.free();
        this->attentionNormGammaGradient.free();
    }

    this->releaseTrainActivationScratch(deferHostWeightDownload);
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
