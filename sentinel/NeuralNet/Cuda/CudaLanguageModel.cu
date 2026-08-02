#include "CudaLanguageModel.hpp"

#include "CudaAmp.hpp"
#include "CudaOps.hpp"
#include "../Utils/SmokeLog.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <stdexcept>
#include <utility>
#include <vector>

#include "../Optimizers/Adam.hpp"

CudaLanguageModel::CudaLanguageModel()
    : maximumPositionCount(0), maxPackedColumns(1024), logitChunkRows(2048), gradientAccumulationSteps(4), activationCheckpointing(true), adam(0.001f), trainStateReady(false) {}

void CudaTransformerBlockAdamStates::ensureFrom(const CudaTransformerBlock& block) {
    // moments stay empty until first Adam update (lazy allocation for low VRAM before train)
    (void)block;
}

void CudaLanguageModelGradients::ensureFrom(const CudaLanguageModel& model) {
    this->tokenEmbedding.ensureSize(model.tokenEmbeddingWeight.rows, model.tokenEmbeddingWeight.cols);
    this->blocks.resize(model.blocks.size());
    for (size_t blockIndex = 0; blockIndex < model.blocks.size(); ++blockIndex)
        this->blocks[blockIndex].ensureFrom(model.blocks[blockIndex]);
    this->finalNormGamma.ensureSize(model.finalNorm.gamma.rows, model.finalNorm.gamma.cols);
    this->projectionWeight.ensureSize(model.projectionWeight.rows, model.projectionWeight.cols);
    this->projectionBias.ensureSize(model.projectionBias.rows, model.projectionBias.cols);
}

void CudaLanguageModelGradients::zeroInPlace() {
    CudaOps::zeroInPlace(this->tokenEmbedding);
    for (CudaTransformerBlockGradients& block : this->blocks)
        block.zeroInPlace();
    CudaOps::zeroInPlace(this->finalNormGamma);
    CudaOps::zeroInPlace(this->projectionWeight);
    CudaOps::zeroInPlace(this->projectionBias);
}

void CudaLanguageModelGradients::addInPlace(const CudaLanguageModelGradients& other) {
    CudaOps::addInPlace(this->tokenEmbedding, other.tokenEmbedding);
    for (size_t blockIndex = 0; blockIndex < this->blocks.size(); ++blockIndex)
        this->blocks[blockIndex].addInPlace(other.blocks[blockIndex]);
    CudaOps::addInPlace(this->finalNormGamma, other.finalNormGamma);
    CudaOps::addInPlace(this->projectionWeight, other.projectionWeight);
    CudaOps::addInPlace(this->projectionBias, other.projectionBias);
}

void CudaLanguageModelGradients::scaleInPlace(float scalar) {
    CudaOps::scaleInPlace(this->tokenEmbedding, scalar);
    for (CudaTransformerBlockGradients& block : this->blocks)
        block.scaleInPlace(scalar);
    CudaOps::scaleInPlace(this->finalNormGamma, scalar);
    CudaOps::scaleInPlace(this->projectionWeight, scalar);
    CudaOps::scaleInPlace(this->projectionBias, scalar);
}

void CudaLanguageModel::ensureTrainState() {
    if (this->trainStateReady) return;
    if (this->tokenEmbeddingWeight.empty()) throw std::logic_error("CudaLanguageModel::ensureTrainState weights not uploaded");

    this->trainGradients.ensureFrom(*this);
    this->trainGradients.zeroInPlace();

    // Adam moments allocated lazily on first applyGradients
    this->blockAdamStates.resize(this->blocks.size());
    for (size_t blockIndex = 0; blockIndex < this->blocks.size(); ++blockIndex)
        this->blockAdamStates[blockIndex].ensureFrom(this->blocks[blockIndex]);

    this->finalNormGammaGradient.ensureSize(this->finalNorm.gamma.rows, this->finalNorm.gamma.cols);

    this->ensureTrainWorkspaces();
    this->trainStateReady = true;
}

void CudaLanguageModel::ensureTrainWorkspaces() {
    if (this->tokenEmbeddingWeight.empty()) throw std::logic_error("CudaLanguageModel::ensureTrainWorkspaces weights not uploaded");
    if (this->maxPackedColumns <= 0) throw std::invalid_argument("CudaLanguageModel::ensureTrainWorkspaces maxPackedColumns must be > 0");
    if (this->logitChunkRows <= 0) throw std::invalid_argument("CudaLanguageModel::ensureTrainWorkspaces logitChunkRows must be > 0");

    const size_t embeddingDim = this->tokenEmbeddingWeight.cols;
    const size_t vocabularySize = this->tokenEmbeddingWeight.rows;
    const size_t maxColumns = static_cast<size_t>(this->maxPackedColumns);
    const size_t chunkRows = static_cast<size_t>((std::min)(this->logitChunkRows, static_cast<int>(vocabularySize)));

    this->hidden.ensureSize(embeddingDim, maxColumns);
    this->normalized.ensureSize(embeddingDim, maxColumns);
    this->hiddenGradient.ensureSize(embeddingDim, maxColumns);
    this->blockInputGradientScratch.ensureSize(embeddingDim, maxColumns);
    this->normInputGradientScratch.ensureSize(embeddingDim, maxColumns);
    this->epochLossSum.ensureSize(1, 1);
    this->tokenIdsBuffer.ensureCapacity(maxColumns);
    this->targetTokenIdsBuffer.ensureCapacity(maxColumns);
    this->meanDivisorBuffer.ensureCapacity(maxColumns);

    // full vocab x seq tensors are not needed for chunked train head
    this->logits.free();
    this->probabilities.free();
    this->logitGradient.free();
    this->projectionWeightGradient.free();
    this->projectionBiasGradient.free();

    this->logitChunk.ensureSize(chunkRows, maxColumns);
    this->logitGradientChunk.ensureSize(chunkRows, maxColumns);
    this->projectionWeightGradientChunk.ensureSize(chunkRows, embeddingDim);
    this->hiddenGradientChunk.ensureSize(embeddingDim, maxColumns);
    this->onlineSoftmaxMax.ensureSize(1, maxColumns);
    this->onlineSoftmaxSumExp.ensureSize(1, maxColumns);
    this->targetLogits.ensureSize(1, maxColumns);

    // FP32 checkpoints only — FP16 saves VRAM but overflows on small/consumer runs and skips Adam forever
    this->checkpointRestoreScratch.free();
    for (CudaHalfMatrix& checkpoint : this->blockInputCheckpointsHalf)
        checkpoint.free();
    this->blockInputCheckpointsHalf.clear();
    if (this->activationCheckpointing) {
        if (this->blockInputCheckpoints.size() != this->blocks.size())
            this->blockInputCheckpoints.resize(this->blocks.size());
        for (CudaMatrix& checkpoint : this->blockInputCheckpoints)
            checkpoint.ensureSize(embeddingDim, maxColumns);
    }
}

void CudaLanguageModel::releaseActivationCheckpoints() {
    for (CudaMatrix& checkpoint : this->blockInputCheckpoints)
        checkpoint.free();
    this->blockInputCheckpoints.clear();
    for (CudaHalfMatrix& checkpoint : this->blockInputCheckpointsHalf)
        checkpoint.free();
    this->blockInputCheckpointsHalf.clear();
    this->checkpointRestoreScratch.free();
}

void CudaLanguageModel::downloadTo(LanguageModel& host) {
    if (this->tokenEmbeddingWeight.empty()) throw std::logic_error("CudaLanguageModel::downloadTo weights not uploaded");

    this->tokenEmbeddingWeight.downloadInto(host.tokenEmbedding.weight);
    for (size_t blockIndex = 0; blockIndex < this->blocks.size(); ++blockIndex) {
        const CudaTransformerBlock& block = this->blocks[blockIndex];
        TransformerBlock& hostBlock = host.blocks[blockIndex];
        block.attention.queryWeight.downloadInto(hostBlock.attention.queryWeight);
        block.attention.keyWeight.downloadInto(hostBlock.attention.keyWeight);
        block.attention.valueWeight.downloadInto(hostBlock.attention.valueWeight);
        block.attention.outputWeight.downloadInto(hostBlock.attention.outputWeight);
        block.attentionNorm.gamma.downloadInto(hostBlock.attentionNorm.gamma);
        block.feedForwardNorm.gamma.downloadInto(hostBlock.feedForwardNorm.gamma);
        block.feedForward.gateWeight.downloadInto(hostBlock.feedForward.gateWeight);
        block.feedForward.gateBias.downloadInto(hostBlock.feedForward.gateBias);
        block.feedForward.upWeight.downloadInto(hostBlock.feedForward.upWeight);
        block.feedForward.upBias.downloadInto(hostBlock.feedForward.upBias);
        block.feedForward.downWeight.downloadInto(hostBlock.feedForward.downWeight);
        block.feedForward.downBias.downloadInto(hostBlock.feedForward.downBias);
    }
    this->finalNorm.gamma.downloadInto(host.finalNorm.gamma);
    this->projectionWeight.downloadInto(host.outputProjection.weight);
    this->projectionBias.downloadInto(host.outputProjection.bias);
}

void CudaLanguageModel::downloadOptimizerTo(LanguageModel& host) {
    if (this->tokenEmbeddingWeight.empty()) throw std::logic_error("CudaLanguageModel::downloadOptimizerTo weights not uploaded");
    if (host.blocks.size() != this->blocks.size())
        throw std::invalid_argument("CudaLanguageModel::downloadOptimizerTo block count mismatch");

    host.optimizer.learningRate = this->adam.learningRate;
    host.optimizer.beta1 = this->adam.beta1;
    host.optimizer.beta2 = this->adam.beta2;
    host.optimizer.epsilon = this->adam.epsilon;
    host.optimizer.timeStep = this->adam.timeStep;

    auto downloadOrZero = [](const CudaAdamState& state, AdamState& hostState, size_t rows, size_t cols) {
        if (state.empty()) {
            hostState = AdamState::zerosLike(Matrix(rows, cols, 0.0f));
            return;
        }
        state.downloadInto(hostState, rows, cols);
    };

    downloadOrZero(this->tokenEmbeddingState, host.tokenEmbeddingState, this->tokenEmbeddingWeight.rows, this->tokenEmbeddingWeight.cols);
    downloadOrZero(this->finalNormGammaState, host.finalNormGammaState, this->finalNorm.gamma.rows, this->finalNorm.gamma.cols);
    downloadOrZero(this->projectionWeightState, host.projectionWeightState, this->projectionWeight.rows, this->projectionWeight.cols);
    downloadOrZero(this->projectionBiasState, host.projectionBiasState, this->projectionBias.rows, this->projectionBias.cols);

    if (this->blockAdamStates.size() != this->blocks.size())
        this->blockAdamStates.resize(this->blocks.size());

    for (size_t blockIndex = 0; blockIndex < this->blocks.size(); ++blockIndex) {
        const CudaTransformerBlock& block = this->blocks[blockIndex];
        TransformerBlock& hostBlock = host.blocks[blockIndex];
        const CudaTransformerBlockAdamStates& states = this->blockAdamStates[blockIndex];

        downloadOrZero(states.queryWeight, hostBlock.queryWeightState, block.attention.queryWeight.rows, block.attention.queryWeight.cols);
        downloadOrZero(states.keyWeight, hostBlock.keyWeightState, block.attention.keyWeight.rows, block.attention.keyWeight.cols);
        downloadOrZero(states.valueWeight, hostBlock.valueWeightState, block.attention.valueWeight.rows, block.attention.valueWeight.cols);
        downloadOrZero(states.attentionOutputWeight, hostBlock.attentionOutputWeightState, block.attention.outputWeight.rows, block.attention.outputWeight.cols);
        downloadOrZero(states.attentionNormGamma, hostBlock.attentionNormGammaState, block.attentionNorm.gamma.rows, block.attentionNorm.gamma.cols);
        downloadOrZero(states.feedForwardNormGamma, hostBlock.feedForwardNormGammaState, block.feedForwardNorm.gamma.rows, block.feedForwardNorm.gamma.cols);
        downloadOrZero(states.feedForwardGateWeight, hostBlock.feedForwardGateWeightState, block.feedForward.gateWeight.rows, block.feedForward.gateWeight.cols);
        downloadOrZero(states.feedForwardGateBias, hostBlock.feedForwardGateBiasState, block.feedForward.gateBias.rows, block.feedForward.gateBias.cols);
        downloadOrZero(states.feedForwardUpWeight, hostBlock.feedForwardUpWeightState, block.feedForward.upWeight.rows, block.feedForward.upWeight.cols);
        downloadOrZero(states.feedForwardUpBias, hostBlock.feedForwardUpBiasState, block.feedForward.upBias.rows, block.feedForward.upBias.cols);
        downloadOrZero(states.feedForwardDownWeight, hostBlock.feedForwardDownWeightState, block.feedForward.downWeight.rows, block.feedForward.downWeight.cols);
        downloadOrZero(states.feedForwardDownBias, hostBlock.feedForwardDownBiasState, block.feedForward.downBias.rows, block.feedForward.downBias.cols);
    }
}

void CudaLanguageModel::uploadOptimizerFrom(const LanguageModel& host) {
    if (this->tokenEmbeddingWeight.empty()) throw std::logic_error("CudaLanguageModel::uploadOptimizerFrom weights not uploaded");
    if (host.blocks.size() != this->blocks.size())
        throw std::invalid_argument("CudaLanguageModel::uploadOptimizerFrom block count mismatch");

    this->ensureTrainState();
    this->adam.learningRate = host.optimizer.learningRate;
    this->adam.beta1 = host.optimizer.beta1;
    this->adam.beta2 = host.optimizer.beta2;
    this->adam.epsilon = host.optimizer.epsilon;
    this->adam.timeStep = host.optimizer.timeStep;

    this->tokenEmbeddingState.uploadFrom(host.tokenEmbeddingState);
    this->finalNormGammaState.uploadFrom(host.finalNormGammaState);
    this->projectionWeightState.uploadFrom(host.projectionWeightState);
    this->projectionBiasState.uploadFrom(host.projectionBiasState);

    if (this->blockAdamStates.size() != this->blocks.size())
        this->blockAdamStates.resize(this->blocks.size());

    for (size_t blockIndex = 0; blockIndex < this->blocks.size(); ++blockIndex) {
        const TransformerBlock& hostBlock = host.blocks[blockIndex];
        CudaTransformerBlockAdamStates& states = this->blockAdamStates[blockIndex];
        states.queryWeight.uploadFrom(hostBlock.queryWeightState);
        states.keyWeight.uploadFrom(hostBlock.keyWeightState);
        states.valueWeight.uploadFrom(hostBlock.valueWeightState);
        states.attentionOutputWeight.uploadFrom(hostBlock.attentionOutputWeightState);
        states.attentionNormGamma.uploadFrom(hostBlock.attentionNormGammaState);
        states.feedForwardNormGamma.uploadFrom(hostBlock.feedForwardNormGammaState);
        states.feedForwardGateWeight.uploadFrom(hostBlock.feedForwardGateWeightState);
        states.feedForwardGateBias.uploadFrom(hostBlock.feedForwardGateBiasState);
        states.feedForwardUpWeight.uploadFrom(hostBlock.feedForwardUpWeightState);
        states.feedForwardUpBias.uploadFrom(hostBlock.feedForwardUpBiasState);
        states.feedForwardDownWeight.uploadFrom(hostBlock.feedForwardDownWeightState);
        states.feedForwardDownBias.uploadFrom(hostBlock.feedForwardDownBiasState);
    }
}

float CudaLanguageModel::accumulateExample(const LanguageModelExample& example, CudaLanguageModelGradients& gradients) {
    const LanguageModelExample* pointer = &example;
    return this->accumulatePackedExamples(&pointer, 1, gradients);
}

float CudaLanguageModel::flushPackedHostBuffers(int segmentLength, int exampleCount, CudaLanguageModelGradients& gradients) {
    if (segmentLength <= 0) throw std::invalid_argument("CudaLanguageModel::flushPackedHostBuffers segmentLength must be > 0");
    if (exampleCount <= 0) throw std::invalid_argument("CudaLanguageModel::flushPackedHostBuffers exampleCount must be > 0");
    if (gradients.blocks.size() != this->blocks.size())
        throw std::invalid_argument("CudaLanguageModel::flushPackedHostBuffers gradients not initialized");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaLanguageModel::flushPackedHostBuffers no CUDA device");

    const size_t tokenCount = this->packedInputTokenIds.size();
    if (tokenCount == 0) throw std::invalid_argument("CudaLanguageModel::flushPackedHostBuffers empty pack");
    if (tokenCount != this->packedTargetTokenIds.size() || tokenCount != this->packedMeanDivisors.size())
        throw std::invalid_argument("CudaLanguageModel::flushPackedHostBuffers pack buffer size mismatch");
    if (static_cast<size_t>(segmentLength) * static_cast<size_t>(exampleCount) != tokenCount)
        throw std::invalid_argument("CudaLanguageModel::flushPackedHostBuffers tokenCount mismatch");

    this->forwardTrunkInto(this->packedInputTokenIds, segmentLength);

    this->targetTokenIdsBuffer.ensureCapacity(tokenCount);
    this->targetTokenIdsBuffer.copyFromHost(this->packedTargetTokenIds.data(), tokenCount);
    this->meanDivisorBuffer.ensureCapacity(tokenCount);
    this->meanDivisorBuffer.copyFromHost(this->packedMeanDivisors.data(), tokenCount);

    if (this->epochLossSum.rows != 1 || this->epochLossSum.cols != 1)
        this->epochLossSum.ensureSize(1, 1);
    this->accumulateChunkedProjection(tokenCount, segmentLength, exampleCount, gradients);

    this->finalNorm.backward(this->hiddenGradient, this->normInputGradientScratch, this->finalNormGammaGradient);
    CudaOps::addInPlace(gradients.finalNormGamma, this->finalNormGammaGradient);
    std::swap(this->hiddenGradient, this->normInputGradientScratch);

    for (int blockIndex = static_cast<int>(this->blocks.size()) - 1; blockIndex >= 0; --blockIndex) {
        if (this->activationCheckpointing) {
            this->blocks[static_cast<size_t>(blockIndex)].forward(
                this->blockInputCheckpoints[static_cast<size_t>(blockIndex)],
                this->normalized,
                segmentLength);
        }
        this->blocks[static_cast<size_t>(blockIndex)].backward(this->hiddenGradient, this->blockInputGradientScratch, gradients.blocks[static_cast<size_t>(blockIndex)]);
        std::swap(this->hiddenGradient, this->blockInputGradientScratch);
    }

    CudaOps::embeddingScatterAddInto(gradients.tokenEmbedding, this->tokenIdsBuffer, tokenCount, this->hiddenGradient);
    return 0.0f;
}

float CudaLanguageModel::accumulatePackedExamples(const LanguageModelExample* const* examples, int exampleCount, CudaLanguageModelGradients& gradients) {
    if (examples == nullptr || exampleCount <= 0) throw std::invalid_argument("CudaLanguageModel::accumulatePackedExamples empty examples");

    const size_t segmentTokenCount = examples[0]->inputTokenIds.size();
    if (segmentTokenCount == 0) throw std::invalid_argument("CudaLanguageModel::accumulatePackedExamples empty inputTokenIds");

    this->packedInputTokenIds.clear();
    this->packedTargetTokenIds.clear();
    this->packedMeanDivisors.clear();
    this->packedInputTokenIds.reserve(segmentTokenCount * static_cast<size_t>(exampleCount));
    this->packedTargetTokenIds.reserve(segmentTokenCount * static_cast<size_t>(exampleCount));
    this->packedMeanDivisors.reserve(segmentTokenCount * static_cast<size_t>(exampleCount));

    for (int exampleIndex = 0; exampleIndex < exampleCount; ++exampleIndex) {
        const LanguageModelExample& example = *examples[exampleIndex];
        if (example.inputTokenIds.size() != segmentTokenCount)
            throw std::invalid_argument("CudaLanguageModel::accumulatePackedExamples unequal input lengths");
        if (example.targetTokenIds.size() != segmentTokenCount)
            throw std::invalid_argument("CudaLanguageModel::accumulatePackedExamples target length mismatch");
        this->packedInputTokenIds.insert(this->packedInputTokenIds.end(), example.inputTokenIds.begin(), example.inputTokenIds.end());
        this->packedTargetTokenIds.insert(this->packedTargetTokenIds.end(), example.targetTokenIds.begin(), example.targetTokenIds.end());
        for (size_t position = 0; position < segmentTokenCount; ++position)
            this->packedMeanDivisors.push_back(static_cast<int>(segmentTokenCount));
    }

    return this->flushPackedHostBuffers(static_cast<int>(segmentTokenCount), exampleCount, gradients);
}

float CudaLanguageModel::accumulateBucketPackedExamples(const LanguageModelExample* const* examples, int exampleCount, int bucketLength, CudaLanguageModelGradients& gradients) {
    if (examples == nullptr || exampleCount <= 0) throw std::invalid_argument("CudaLanguageModel::accumulateBucketPackedExamples empty examples");
    if (bucketLength <= 0) throw std::invalid_argument("CudaLanguageModel::accumulateBucketPackedExamples bucketLength must be > 0");
    if (bucketLength > this->maximumPositionCount)
        throw std::invalid_argument("CudaLanguageModel::accumulateBucketPackedExamples bucket exceeds maximumPositionCount");

    this->packedInputTokenIds.clear();
    this->packedTargetTokenIds.clear();
    this->packedMeanDivisors.clear();
    this->packedInputTokenIds.reserve(static_cast<size_t>(bucketLength) * static_cast<size_t>(exampleCount));
    this->packedTargetTokenIds.reserve(static_cast<size_t>(bucketLength) * static_cast<size_t>(exampleCount));
    this->packedMeanDivisors.reserve(static_cast<size_t>(bucketLength) * static_cast<size_t>(exampleCount));

    for (int exampleIndex = 0; exampleIndex < exampleCount; ++exampleIndex) {
        const LanguageModelExample& example = *examples[exampleIndex];
        const int trueLength = static_cast<int>(example.inputTokenIds.size());
        if (trueLength <= 0) throw std::invalid_argument("CudaLanguageModel::accumulateBucketPackedExamples empty example");
        if (static_cast<int>(example.targetTokenIds.size()) != trueLength)
            throw std::invalid_argument("CudaLanguageModel::accumulateBucketPackedExamples target length mismatch");
        if (trueLength > bucketLength)
            throw std::invalid_argument("CudaLanguageModel::accumulateBucketPackedExamples example longer than bucket");
        if (CudaLanguageModel::lengthBucket(trueLength, this->maximumPositionCount) != bucketLength)
            throw std::invalid_argument("CudaLanguageModel::accumulateBucketPackedExamples bucket mismatch");

        this->packedInputTokenIds.insert(this->packedInputTokenIds.end(), example.inputTokenIds.begin(), example.inputTokenIds.end());
        this->packedTargetTokenIds.insert(this->packedTargetTokenIds.end(), example.targetTokenIds.begin(), example.targetTokenIds.end());
        for (int position = trueLength; position < bucketLength; ++position) {
            this->packedInputTokenIds.push_back(CudaLanguageModel::padInputId);
            this->packedTargetTokenIds.push_back(CudaLanguageModel::padTargetId);
        }
        for (int position = 0; position < bucketLength; ++position)
            this->packedMeanDivisors.push_back(trueLength);
    }

    return this->flushPackedHostBuffers(bucketLength, exampleCount, gradients);
}

int CudaLanguageModel::lengthBucket(int trueLength, int maximumPositionCount) {
    if (maximumPositionCount <= 0) throw std::invalid_argument("CudaLanguageModel::lengthBucket maximumPositionCount must be > 0");
    if (trueLength <= 0) return (std::min)(CudaLanguageModel::lengthBucketStep, maximumPositionCount);
    if (trueLength > maximumPositionCount) trueLength = maximumPositionCount;

    int bucket = ((trueLength + CudaLanguageModel::lengthBucketStep - 1) / CudaLanguageModel::lengthBucketStep) * CudaLanguageModel::lengthBucketStep;
    if (bucket > maximumPositionCount) bucket = maximumPositionCount;
    if (bucket < trueLength) bucket = trueLength;
    return bucket;
}

void CudaLanguageModel::forwardTrunkInto(const std::vector<int>& tokenIds, int segmentLength) {
    if (tokenIds.empty()) throw std::invalid_argument("CudaLanguageModel::forwardTrunkInto empty tokenIds");
    if (this->tokenEmbeddingWeight.empty()) throw std::logic_error("CudaLanguageModel::forwardTrunkInto weights not uploaded");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaLanguageModel::forwardTrunkInto no CUDA device");

    if (segmentLength > 0) {
        if (static_cast<int>(tokenIds.size()) % segmentLength != 0)
            throw std::invalid_argument("CudaLanguageModel::forwardTrunkInto tokenCount not divisible by segmentLength");
        if (segmentLength > this->maximumPositionCount)
            throw std::invalid_argument("CudaLanguageModel::forwardTrunkInto segmentLength exceeds maximumPositionCount");
    } else if (static_cast<int>(tokenIds.size()) > this->maximumPositionCount) {
        throw std::invalid_argument("CudaLanguageModel::forwardTrunkInto sequence longer than maximumPositionCount");
    }

    this->tokenIdsBuffer.ensureCapacity(tokenIds.size());
    this->tokenIdsBuffer.copyFromHost(tokenIds.data(), tokenIds.size());
    CudaOps::embeddingGatherInto(this->tokenEmbeddingWeight, this->tokenIdsBuffer, tokenIds.size(), this->hidden);

    if (this->activationCheckpointing) {
        if (this->blockInputCheckpoints.size() != this->blocks.size())
            this->blockInputCheckpoints.resize(this->blocks.size());
        for (CudaMatrix& checkpoint : this->blockInputCheckpoints)
            checkpoint.ensureSize(this->tokenEmbeddingWeight.cols, tokenIds.size());
    }

    for (size_t blockIndex = 0; blockIndex < this->blocks.size(); ++blockIndex) {
        if (this->activationCheckpointing)
            CudaOps::copyInto(this->hidden, this->blockInputCheckpoints[blockIndex]);
        this->blocks[blockIndex].forward(this->hidden, this->normalized, segmentLength);
        CudaMatrix swapBuffer = std::move(this->hidden);
        this->hidden = std::move(this->normalized);
        this->normalized = std::move(swapBuffer);
    }

    this->finalNorm.forward(this->hidden, this->normalized);
}

void CudaLanguageModel::accumulateChunkedProjection(size_t tokenCount, int segmentLength, int exampleCountInPack, CudaLanguageModelGradients& gradients) {
    if (tokenCount == 0) throw std::invalid_argument("CudaLanguageModel::accumulateChunkedProjection empty tokenCount");
    if (segmentLength <= 0) throw std::invalid_argument("CudaLanguageModel::accumulateChunkedProjection segmentLength must be > 0");
    if (exampleCountInPack <= 0) throw std::invalid_argument("CudaLanguageModel::accumulateChunkedProjection exampleCountInPack must be > 0");
    if (static_cast<size_t>(segmentLength) * static_cast<size_t>(exampleCountInPack) != tokenCount)
        throw std::invalid_argument("CudaLanguageModel::accumulateChunkedProjection tokenCount mismatch");
    if (this->projectionWeight.empty()) throw std::logic_error("CudaLanguageModel::accumulateChunkedProjection missing projection");

    const int vocabularySize = static_cast<int>(this->projectionWeight.rows);
    const int embeddingDim = static_cast<int>(this->projectionWeight.cols);
    const int chunkCap = (std::min)(this->logitChunkRows, vocabularySize);
    const float gradScale = CudaAmp::lossScalingActive() ? CudaAmp::lossScaler.scale : 1.0f;

    this->targetLogits.ensureSize(1, tokenCount);
    this->hiddenGradient.ensureSize(static_cast<size_t>(embeddingDim), tokenCount);
    this->hiddenGradientChunk.ensureSize(static_cast<size_t>(embeddingDim), tokenCount);
    CudaOps::onlineSoftmaxReset(this->onlineSoftmaxMax, this->onlineSoftmaxSumExp, tokenCount);
    CudaOps::zeroInPlace(this->targetLogits);
    CudaOps::zeroInPlace(this->hiddenGradient);

    auto projectChunk = [&](int rowStart, int chunkRows) {
        this->logitChunk.ensureSize(static_cast<size_t>(chunkRows), tokenCount);
        const float* weightRows = this->projectionWeight.buffer.deviceData + static_cast<size_t>(rowStart) * static_cast<size_t>(embeddingDim);
        // LM-head stays FP32 even when amp is on — CE is numerically fragile in FP16
        const bool previousAmp = CudaAmp::preferMixedPrecision;
        CudaAmp::preferMixedPrecision = false;
        CudaMatrix::multiplyPointersInto(
            weightRows, static_cast<size_t>(chunkRows), static_cast<size_t>(embeddingDim),
            this->normalized.buffer.deviceData, static_cast<size_t>(embeddingDim), tokenCount,
            this->logitChunk.buffer.deviceData,
            false, false);
        CudaAmp::preferMixedPrecision = previousAmp;
        CudaOps::broadcastBiasRowsAddInPlace(this->logitChunk, this->projectionBias, rowStart, chunkRows);
    };

    for (int rowStart = 0; rowStart < vocabularySize; rowStart += chunkCap) {
        const int chunkRows = (std::min)(chunkCap, vocabularySize - rowStart);
        projectChunk(rowStart, chunkRows);
        CudaOps::onlineSoftmaxUpdateFromChunk(this->logitChunk, chunkRows, tokenCount, this->onlineSoftmaxMax, this->onlineSoftmaxSumExp);
        CudaOps::captureTargetLogitFromChunk(this->logitChunk, this->targetTokenIdsBuffer, rowStart, chunkRows, tokenCount, this->targetLogits);
    }

    // mean over each example's true length; pad targets (padTargetId) are ignored
    CudaOps::onlineSoftmaxAddMeanCrossEntropy(
        this->targetLogits, this->onlineSoftmaxMax, this->onlineSoftmaxSumExp, tokenCount,
        this->epochLossSum, 1.0f, segmentLength,
        &this->targetTokenIdsBuffer, CudaLanguageModel::padTargetId, &this->meanDivisorBuffer);

    for (int rowStart = 0; rowStart < vocabularySize; rowStart += chunkCap) {
        const int chunkRows = (std::min)(chunkCap, vocabularySize - rowStart);
        projectChunk(rowStart, chunkRows);
        CudaOps::onlineSoftmaxLogitGradientChunkInto(
            this->logitChunk, this->targetTokenIdsBuffer, rowStart, chunkRows, tokenCount,
            this->onlineSoftmaxMax, this->onlineSoftmaxSumExp, this->logitGradientChunk, gradScale, segmentLength,
            CudaLanguageModel::padTargetId, &this->meanDivisorBuffer);

        this->projectionWeightGradientChunk.ensureSize(static_cast<size_t>(chunkRows), static_cast<size_t>(embeddingDim));
        const bool previousAmp = CudaAmp::preferMixedPrecision;
        CudaAmp::preferMixedPrecision = false;
        CudaMatrix::multiplyInto(this->logitGradientChunk, this->normalized, this->projectionWeightGradientChunk, false, true);
        CudaOps::addRowsInPlace(gradients.projectionWeight, rowStart, this->projectionWeightGradientChunk);
        CudaOps::sumColumnsAddIntoRows(this->logitGradientChunk, gradients.projectionBias, rowStart);

        const float* weightRows = this->projectionWeight.buffer.deviceData + static_cast<size_t>(rowStart) * static_cast<size_t>(embeddingDim);
        CudaMatrix::multiplyPointersInto(
            weightRows, static_cast<size_t>(chunkRows), static_cast<size_t>(embeddingDim),
            this->logitGradientChunk.buffer.deviceData, static_cast<size_t>(chunkRows), tokenCount,
            this->hiddenGradientChunk.buffer.deviceData,
            true, false);
        CudaAmp::preferMixedPrecision = previousAmp;
        this->hiddenGradientChunk.rows = static_cast<size_t>(embeddingDim);
        this->hiddenGradientChunk.cols = tokenCount;
        CudaOps::addInPlace(this->hiddenGradient, this->hiddenGradientChunk);
    }
}

void CudaLanguageModel::applyGradients(CudaLanguageModelGradients& gradients, float gradientScale) {
    if (!this->trainStateReady) throw std::logic_error("CudaLanguageModel::applyGradients train state not ready");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaLanguageModel::applyGradients no CUDA device");

    float effectiveGradientScale = gradientScale;
    if (CudaAmp::lossScalingActive()) {
        // unscale first so overflow checks and Adam see true grad magnitudes
        const float inverseLossScale = 1.0f / CudaAmp::lossScaler.scale;
        gradients.scaleInPlace(inverseLossScale);
        if (CudaAmp::gradientsHaveNonFinite(gradients)) {
            CudaAmp::lossScaler.updateOnOverflow();
            return;
        }
        effectiveGradientScale = gradientScale;
    }

    this->adam.step();

    auto ensureMoments = [](CudaAdamState& state, const CudaMatrix& parameter) {
        if (!state.empty()) return;
        if (CudaAdam::preferInt8Moments)
            state.ensureInt8(parameter, CudaAdam::int8BlockSize);
        else
            state.ensureFp32(parameter);
    };

    std::vector<CudaAdamUpdateItem> items;
    items.reserve(16 + this->blocks.size() * 12);

    auto pushItem = [&items, &ensureMoments](CudaMatrix& parameter, CudaAdamState& state, const CudaMatrix& gradient) {
        if (parameter.elementCount() == 0) throw std::invalid_argument("CudaLanguageModel::applyGradients empty parameter");
        if (gradient.elementCount() != parameter.elementCount())
            throw std::invalid_argument("CudaLanguageModel::applyGradients gradient/parameter size mismatch");
        ensureMoments(state, parameter);
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
        items.push_back(item);
    };

    pushItem(this->projectionWeight, this->projectionWeightState, gradients.projectionWeight);
    pushItem(this->projectionBias, this->projectionBiasState, gradients.projectionBias);
    pushItem(this->finalNorm.gamma, this->finalNormGammaState, gradients.finalNormGamma);

    for (size_t blockIndex = 0; blockIndex < this->blocks.size(); ++blockIndex) {
        CudaTransformerBlock& block = this->blocks[blockIndex];
        CudaTransformerBlockGradients& blockGradients = gradients.blocks[blockIndex];
        CudaTransformerBlockAdamStates& blockStates = this->blockAdamStates[blockIndex];

        pushItem(block.attention.queryWeight, blockStates.queryWeight, blockGradients.queryWeight);
        pushItem(block.attention.keyWeight, blockStates.keyWeight, blockGradients.keyWeight);
        pushItem(block.attention.valueWeight, blockStates.valueWeight, blockGradients.valueWeight);
        pushItem(block.attention.outputWeight, blockStates.attentionOutputWeight, blockGradients.attentionOutputWeight);
        pushItem(block.attentionNorm.gamma, blockStates.attentionNormGamma, blockGradients.attentionNormGamma);
        pushItem(block.feedForwardNorm.gamma, blockStates.feedForwardNormGamma, blockGradients.feedForwardNormGamma);
        pushItem(block.feedForward.gateWeight, blockStates.feedForwardGateWeight, blockGradients.feedForwardGateWeight);
        pushItem(block.feedForward.gateBias, blockStates.feedForwardGateBias, blockGradients.feedForwardGateBias);
        pushItem(block.feedForward.upWeight, blockStates.feedForwardUpWeight, blockGradients.feedForwardUpWeight);
        pushItem(block.feedForward.upBias, blockStates.feedForwardUpBias, blockGradients.feedForwardUpBias);
        pushItem(block.feedForward.downWeight, blockStates.feedForwardDownWeight, blockGradients.feedForwardDownWeight);
        pushItem(block.feedForward.downBias, blockStates.feedForwardDownBias, blockGradients.feedForwardDownBias);
    }

    pushItem(this->tokenEmbeddingWeight, this->tokenEmbeddingState, gradients.tokenEmbedding);
    this->adam.updateMany(items.data(), static_cast<int>(items.size()), effectiveGradientScale);

    if (CudaAmp::lossScalingActive())
        CudaAmp::lossScaler.updateOnSuccess();
}

float CudaLanguageModel::averageLoss(const LanguageModelDataset& dataset) {
    if (dataset.examples.empty()) return 0.0f;
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaLanguageModel::averageLoss no CUDA device");

    this->epochLossSum.ensureSize(1, 1);
    CudaOps::zeroInPlace(this->epochLossSum);

    for (const LanguageModelExample& example : dataset.examples) {
        this->forwardInto(example.inputTokenIds, this->logits);
        CudaOps::softmaxInto(this->logits, this->probabilities);
        this->targetTokenIdsBuffer.ensureCapacity(example.targetTokenIds.size());
        this->targetTokenIdsBuffer.copyFromHost(example.targetTokenIds.data(), example.targetTokenIds.size());
        CudaOps::crossEntropyAddMeanLossFromIds(this->probabilities, this->targetTokenIdsBuffer, example.targetTokenIds.size(), this->epochLossSum);
    }

    CudaMatmul::throwIfCudaFailed(cudaDeviceSynchronize(), "CudaLanguageModel::averageLoss synchronize");
    Matrix lossHost = this->epochLossSum.download();
    return lossHost.at(0, 0) / static_cast<float>(dataset.size());
}

void CudaLanguageModel::trainOnExamples(
    const LanguageModelDataset& dataset,
    int batchSize,
    bool flushRemainder,
    int& accumulatedExampleCount,
    int& microbatchesSinceStep,
    int& processedExampleCount,
    int& processedPredictionCount,
    int& packCount,
    int& singleExamplePackCount,
    long long& packedExampleSum,
    long long& packedTokenSum
) {
    if (batchSize <= 0) batchSize = 32;
    if (this->gradientAccumulationSteps <= 0) this->gradientAccumulationSteps = 1;

    const int exampleCount = static_cast<int>(dataset.examples.size());
    if (exampleCount > 0) {
        processedExampleCount += exampleCount;
        processedPredictionCount += dataset.totalPredictionCount();
    }

    std::vector<const LanguageModelExample*> packPointers;
    packPointers.reserve(static_cast<size_t>(batchSize));

    for (int batchStart = 0; batchStart < exampleCount; batchStart += batchSize) {
        int batchEnd = batchStart + batchSize;
        if (batchEnd > exampleCount) batchEnd = exampleCount;
        const int batchCount = batchEnd - batchStart;

        std::vector<int> batchIndices;
        batchIndices.reserve(static_cast<size_t>(batchCount));
        for (int index = batchStart; index < batchEnd; ++index)
            batchIndices.push_back(index);
        std::stable_sort(batchIndices.begin(), batchIndices.end(), [&dataset](int left, int right) {
            return dataset.examples[static_cast<size_t>(left)].inputTokenIds.size()
                < dataset.examples[static_cast<size_t>(right)].inputTokenIds.size();
        });

        int packStart = 0;
        while (packStart < static_cast<int>(batchIndices.size())) {
            const int trueLength = static_cast<int>(
                dataset.examples[static_cast<size_t>(batchIndices[static_cast<size_t>(packStart)])].inputTokenIds.size()
            );
            const int bucketLength = CudaLanguageModel::lengthBucket(trueLength, this->maximumPositionCount);
            int maxExamplesInPack = 1;
            if (bucketLength > 0 && this->maxPackedColumns > 0)
                maxExamplesInPack = (std::max)(1, this->maxPackedColumns / bucketLength);

            packPointers.clear();
            int packEnd = packStart;
            while (packEnd < static_cast<int>(batchIndices.size())
                && static_cast<int>(packPointers.size()) < maxExamplesInPack) {
                const int candidateLength = static_cast<int>(
                    dataset.examples[static_cast<size_t>(batchIndices[static_cast<size_t>(packEnd)])].inputTokenIds.size()
                );
                if (CudaLanguageModel::lengthBucket(candidateLength, this->maximumPositionCount) != bucketLength)
                    break;
                packPointers.push_back(&dataset.examples[static_cast<size_t>(batchIndices[static_cast<size_t>(packEnd)])]);
                ++packEnd;
            }

            const int packExampleCount = static_cast<int>(packPointers.size());
            ++packCount;
            if (packExampleCount <= 1) ++singleExamplePackCount;
            packedExampleSum += packExampleCount;
            packedTokenSum += static_cast<long long>(packExampleCount) * static_cast<long long>(bucketLength);

            this->accumulateBucketPackedExamples(packPointers.data(), packExampleCount, bucketLength, this->trainGradients);
            packStart = packEnd;
        }

        accumulatedExampleCount += batchCount;
        ++microbatchesSinceStep;

        const bool isLastBatch = flushRemainder && batchEnd >= exampleCount;
        if (microbatchesSinceStep < this->gradientAccumulationSteps && !isLastBatch)
            continue;

        this->applyGradients(this->trainGradients, 1.0f / static_cast<float>(accumulatedExampleCount));
        this->trainGradients.zeroInPlace();
        accumulatedExampleCount = 0;
        microbatchesSinceStep = 0;
    }

    if (flushRemainder && accumulatedExampleCount > 0) {
        this->applyGradients(this->trainGradients, 1.0f / static_cast<float>(accumulatedExampleCount));
        this->trainGradients.zeroInPlace();
        accumulatedExampleCount = 0;
        microbatchesSinceStep = 0;
    }
}

void CudaLanguageModel::train(const LanguageModelDataset& trainDataset, const LanguageModelDataset& testDataset, int epochs, int logEveryEpochs, int batchSize, int gradientAccumulationSteps) {
    if (trainDataset.examples.empty()) return;
    if (logEveryEpochs <= 0) logEveryEpochs = 1;
    if (batchSize <= 0) batchSize = 32;
    if (gradientAccumulationSteps <= 0) gradientAccumulationSteps = 1;
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaLanguageModel::train no CUDA device");

    this->gradientAccumulationSteps = gradientAccumulationSteps;
    this->ensureTrainState();
    this->ensureTrainWorkspaces();

    for (int epoch = 0; epoch < epochs; ++epoch) {
        CudaOps::zeroInPlace(this->epochLossSum);
        const auto epochStart = std::chrono::steady_clock::now();

        this->trainGradients.zeroInPlace();
        int accumulatedExampleCount = 0;
        int microbatchesSinceStep = 0;
        int processedExampleCount = 0;
        int processedPredictionCount = 0;
        int packCount = 0;
        int singleExamplePackCount = 0;
        long long packedExampleSum = 0;
        long long packedTokenSum = 0;
        this->trainOnExamples(
            trainDataset,
            batchSize,
            true,
            accumulatedExampleCount,
            microbatchesSinceStep,
            processedExampleCount,
            processedPredictionCount,
            packCount,
            singleExamplePackCount,
            packedExampleSum,
            packedTokenSum
        );

        CudaMatmul::throwIfCudaFailed(cudaDeviceSynchronize(), "CudaLanguageModel::train epoch synchronize");
        const auto epochEnd = std::chrono::steady_clock::now();
        const double epochSeconds = std::chrono::duration<double>(epochEnd - epochStart).count();

        if (epoch % logEveryEpochs != 0) continue;

        Matrix lossHost = this->epochLossSum.download();
        const float averageTrainLoss = processedExampleCount > 0
            ? lossHost.at(0, 0) / static_cast<float>(processedExampleCount)
            : 0.0f;
        const double tokensPerSecond = epochSeconds > 0.0 ? static_cast<double>(processedPredictionCount) / epochSeconds : 0.0;
        std::printf("  Epoch %-3d  trainLoss=%.6f  sec=%.2f  tokens/s=%.0f  backend=cuda", epoch, averageTrainLoss, epochSeconds, tokensPerSecond);

        if (CudaAmp::lossScalingActive())
            std::printf("  ampScale=%.0f", CudaAmp::lossScaler.scale);

        if (!testDataset.examples.empty()) {
            const float testLoss = this->averageLoss(testDataset);
            std::printf("  testLoss=%.6f", testLoss);
        }

        if (packCount > 0) {
            const double avgPackExamples = static_cast<double>(packedExampleSum) / static_cast<double>(packCount);
            const double avgPackTokens = static_cast<double>(packedTokenSum) / static_cast<double>(packCount);
            const double size1Percent = 100.0 * static_cast<double>(singleExamplePackCount) / static_cast<double>(packCount);
            std::printf("\n  pack  packs=%d  avgEx=%.2f  avgTok=%.0f  size1=%.1f%%",
                packCount, avgPackExamples, avgPackTokens, size1Percent);
        }

        std::printf("\n");
    }
}

void CudaLanguageModel::train(LanguageModelChunkSource& source, int epochs, int logEveryEpochs, int batchSize, int gradientAccumulationSteps) {
    if (logEveryEpochs <= 0) logEveryEpochs = 1;
    if (batchSize <= 0) batchSize = 32;
    if (gradientAccumulationSteps <= 0) gradientAccumulationSteps = 1;
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaLanguageModel::train(chunk) no CUDA device");

    this->gradientAccumulationSteps = gradientAccumulationSteps;
    this->ensureTrainState();
    this->ensureTrainWorkspaces();

    LanguageModelDataset epochDataset;
    const LanguageModelDataset& testDataset = source.testDataset();

    for (int epoch = 0; epoch < epochs; ++epoch) {
        CudaOps::zeroInPlace(this->epochLossSum);
        const auto epochStart = std::chrono::steady_clock::now();

        this->trainGradients.zeroInPlace();
        int accumulatedExampleCount = 0;
        int microbatchesSinceStep = 0;
        int processedExampleCount = 0;
        int processedPredictionCount = 0;
        int packCount = 0;
        int singleExamplePackCount = 0;
        long long packedExampleSum = 0;
        long long packedTokenSum = 0;

        source.rewindTrain();
        LanguageModelDataset epochDataset;
        source.fillTrainDataset(epochDataset);
        if (!epochDataset.examples.empty()) {
            this->trainOnExamples(
                epochDataset,
                batchSize,
                true,
                accumulatedExampleCount,
                microbatchesSinceStep,
                processedExampleCount,
                processedPredictionCount,
                packCount,
                singleExamplePackCount,
                packedExampleSum,
                packedTokenSum
            );
        }
        if (accumulatedExampleCount > 0) {
            this->applyGradients(this->trainGradients, 1.0f / static_cast<float>(accumulatedExampleCount));
            this->trainGradients.zeroInPlace();
            accumulatedExampleCount = 0;
            microbatchesSinceStep = 0;
        }

        CudaMatmul::throwIfCudaFailed(cudaDeviceSynchronize(), "CudaLanguageModel::train(chunk) epoch synchronize");
        const auto epochEnd = std::chrono::steady_clock::now();
        const double epochSeconds = std::chrono::duration<double>(epochEnd - epochStart).count();

        if (epoch % logEveryEpochs != 0) continue;

        Matrix lossHost = this->epochLossSum.download();
        const float averageTrainLoss = processedExampleCount > 0
            ? lossHost.at(0, 0) / static_cast<float>(processedExampleCount)
            : 0.0f;
        const double tokensPerSecond = epochSeconds > 0.0 ? static_cast<double>(processedPredictionCount) / epochSeconds : 0.0;
        std::printf("  Epoch %-3d  trainLoss=%.6f  sec=%.2f  tokens/s=%.0f  backend=cuda-stream", epoch, averageTrainLoss, epochSeconds, tokensPerSecond);

        if (CudaAmp::lossScalingActive())
            std::printf("  ampScale=%.0f", CudaAmp::lossScaler.scale);

        if (!testDataset.examples.empty()) {
            const float testLoss = this->averageLoss(testDataset);
            std::printf("  testLoss=%.6f", testLoss);
        }

        if (packCount > 0) {
            const double avgPackExamples = static_cast<double>(packedExampleSum) / static_cast<double>(packCount);
            const double avgPackTokens = static_cast<double>(packedTokenSum) / static_cast<double>(packCount);
            const double size1Percent = 100.0 * static_cast<double>(singleExamplePackCount) / static_cast<double>(packCount);
            std::printf("\n  pack  packs=%d  avgEx=%.2f  avgTok=%.0f  size1=%.1f%%",
                packCount, avgPackExamples, avgPackTokens, size1Percent);
        }

        std::printf("\n");
    }
}

void CudaLanguageModel::runTrainSmokeDemo(int vocabularySize, int embeddingDim, int sequenceLength, int blockCount, int headCount) {
    if (!CudaMatmul::isAvailable()) {
        SmokeLog::skip("LanguageModel train");
        return;
    }
    if (vocabularySize <= 0 || embeddingDim <= 0 || sequenceLength <= 0 || blockCount <= 0 || headCount <= 0)
        throw std::invalid_argument("CudaLanguageModel::runTrainSmokeDemo invalid dims");

    LanguageModel host(vocabularySize, embeddingDim, sequenceLength, Adam(0.001f), blockCount, headCount);
    CudaLanguageModel device = CudaLanguageModel::createFrom(host);
    device.adam = CudaAdam(host.optimizer.learningRate, host.optimizer.beta1, host.optimizer.beta2, host.optimizer.epsilon);
    const bool previousAmp = CudaAmp::preferMixedPrecision;
    const bool previousInt8 = CudaAdam::preferInt8Moments;
    CudaAmp::preferMixedPrecision = false;
    CudaAdam::preferInt8Moments = false;
    device.ensureTrainState();

    const int packBatchSize = (std::max)(1, (std::min)(16, device.maxPackedColumns / sequenceLength));
    std::vector<LanguageModelExample> examples(static_cast<size_t>(packBatchSize));
    unsigned state = 103u;
    for (int exampleIndex = 0; exampleIndex < packBatchSize; ++exampleIndex) {
        examples[static_cast<size_t>(exampleIndex)].inputTokenIds.resize(static_cast<size_t>(sequenceLength));
        examples[static_cast<size_t>(exampleIndex)].targetTokenIds.resize(static_cast<size_t>(sequenceLength));
        for (size_t index = 0; index < static_cast<size_t>(sequenceLength); ++index) {
            state = state * 1664525u + 1013904223u;
            examples[static_cast<size_t>(exampleIndex)].inputTokenIds[index] = static_cast<int>(state % static_cast<unsigned>(vocabularySize));
            state = state * 1664525u + 1013904223u;
            examples[static_cast<size_t>(exampleIndex)].targetTokenIds[index] = static_cast<int>(state % static_cast<unsigned>(vocabularySize));
        }
    }

    LanguageModelGradients hostGradients = LanguageModelGradients::zerosFrom(host);
    LanguageModelCache hostCache;
    const float hostLoss = host.accumulateExample(examples[0], hostGradients, hostCache);

    CudaLanguageModelGradients deviceGradients;
    deviceGradients.ensureFrom(device);
    deviceGradients.zeroInPlace();
    device.epochLossSum.ensureSize(1, 1);
    CudaOps::zeroInPlace(device.epochLossSum);
    device.accumulateExample(examples[0], deviceGradients);
    CudaMatmul::throwIfCudaFailed(cudaDeviceSynchronize(), "CudaLanguageModel train smoke synchronize");
    const float deviceLoss = device.epochLossSum.download().at(0, 0);
    Matrix deviceProjectionWeightGrad = deviceGradients.projectionWeight.download();
    float maximumDifference = 0.0f;
    for (size_t index = 0; index < hostGradients.projectionWeight.data.size(); ++index)
        maximumDifference = (std::max)(maximumDifference, std::fabs(hostGradients.projectionWeight.data[index] - deviceProjectionWeightGrad.data[index]));

    CudaLanguageModelGradients packedGradients;
    packedGradients.ensureFrom(device);
    packedGradients.zeroInPlace();
    CudaLanguageModelGradients sequentialGradients;
    sequentialGradients.ensureFrom(device);
    sequentialGradients.zeroInPlace();
    device.epochLossSum.ensureSize(1, 1);
    CudaOps::zeroInPlace(device.epochLossSum);
    std::vector<const LanguageModelExample*> packPointers(static_cast<size_t>(packBatchSize));
    for (int exampleIndex = 0; exampleIndex < packBatchSize; ++exampleIndex)
        packPointers[static_cast<size_t>(exampleIndex)] = &examples[static_cast<size_t>(exampleIndex)];
    device.accumulatePackedExamples(packPointers.data(), packBatchSize, packedGradients);
    const float packedLoss = device.epochLossSum.download().at(0, 0);

    CudaOps::zeroInPlace(device.epochLossSum);
    for (int exampleIndex = 0; exampleIndex < packBatchSize; ++exampleIndex)
        device.accumulateExample(examples[static_cast<size_t>(exampleIndex)], sequentialGradients);
    const float sequentialLoss = device.epochLossSum.download().at(0, 0);
    Matrix packedProjectionWeightGrad = packedGradients.projectionWeight.download();
    Matrix sequentialProjectionWeightGrad = sequentialGradients.projectionWeight.download();
    float packedDifference = 0.0f;
    for (size_t index = 0; index < packedProjectionWeightGrad.data.size(); ++index)
        packedDifference = (std::max)(packedDifference, std::fabs(packedProjectionWeightGrad.data[index] - sequentialProjectionWeightGrad.data[index]));

    const int warmupStepCount = 3;
    const int timedStepCount = 40;
    for (int step = 0; step < warmupStepCount; ++step) {
        device.trainGradients.zeroInPlace();
        device.accumulatePackedExamples(packPointers.data(), packBatchSize, device.trainGradients);
        device.applyGradients(device.trainGradients, 1.0f / static_cast<float>(packBatchSize));
    }
    CudaMatmul::throwIfCudaFailed(cudaDeviceSynchronize(), "CudaLanguageModel train smoke warmup synchronize");

    const auto trainStart = std::chrono::steady_clock::now();
    int totalTokens = 0;
    for (int step = 0; step < timedStepCount; ++step) {
        device.trainGradients.zeroInPlace();
        device.accumulatePackedExamples(packPointers.data(), packBatchSize, device.trainGradients);
        device.applyGradients(device.trainGradients, 1.0f / static_cast<float>(packBatchSize));
        totalTokens += sequenceLength * packBatchSize;
    }
    CudaMatmul::throwIfCudaFailed(cudaDeviceSynchronize(), "CudaLanguageModel train smoke steps synchronize");
    const double trainSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - trainStart).count();
    const double tokensPerSecond = trainSeconds > 0.0 ? static_cast<double>(totalTokens) / trainSeconds : 0.0;

    SmokeLog::result("LanguageModel train", "vocab=%d embed=%d seq=%d pack=%d  loss cpu=%.4f gpu=%.4f  gradDiff=%.2e  packLossDiff=%.2e packGradDiff=%.2e  tokens/s=%.0f",
        vocabularySize, embeddingDim, sequenceLength, packBatchSize, hostLoss, deviceLoss, maximumDifference, std::fabs(packedLoss - sequentialLoss), packedDifference, tokensPerSecond);
    CudaAmp::preferMixedPrecision = previousAmp;
    CudaAdam::preferInt8Moments = previousInt8;
}

void CudaLanguageModel::runTrainInt8AdamSmokeDemo(int vocabularySize, int embeddingDim, int sequenceLength, int blockCount, int headCount) {
    if (!CudaMatmul::isAvailable()) {
        SmokeLog::skip("LanguageModel train int8 Adam");
        return;
    }
    if (vocabularySize <= 0 || embeddingDim <= 0 || sequenceLength <= 0 || blockCount <= 0 || headCount <= 0)
        throw std::invalid_argument("CudaLanguageModel::runTrainInt8AdamSmokeDemo invalid dims");

    LanguageModel host(vocabularySize, embeddingDim, sequenceLength, Adam(0.001f), blockCount, headCount);
    CudaLanguageModel fp32Device = CudaLanguageModel::createFrom(host);
    CudaLanguageModel int8Device = CudaLanguageModel::createFrom(host);
    fp32Device.adam = CudaAdam(host.optimizer.learningRate, host.optimizer.beta1, host.optimizer.beta2, host.optimizer.epsilon);
    int8Device.adam = CudaAdam(host.optimizer.learningRate, host.optimizer.beta1, host.optimizer.beta2, host.optimizer.epsilon);

    const bool previousAmp = CudaAmp::preferMixedPrecision;
    const bool previousInt8 = CudaAdam::preferInt8Moments;
    CudaAmp::preferMixedPrecision = false;

    const int packBatchSize = (std::max)(1, (std::min)(8, fp32Device.maxPackedColumns / sequenceLength));
    std::vector<LanguageModelExample> examples(static_cast<size_t>(packBatchSize));
    unsigned state = 211u;
    for (int exampleIndex = 0; exampleIndex < packBatchSize; ++exampleIndex) {
        examples[static_cast<size_t>(exampleIndex)].inputTokenIds.resize(static_cast<size_t>(sequenceLength));
        examples[static_cast<size_t>(exampleIndex)].targetTokenIds.resize(static_cast<size_t>(sequenceLength));
        for (size_t index = 0; index < static_cast<size_t>(sequenceLength); ++index) {
            state = state * 1664525u + 1013904223u;
            examples[static_cast<size_t>(exampleIndex)].inputTokenIds[index] = static_cast<int>(state % static_cast<unsigned>(vocabularySize));
            state = state * 1664525u + 1013904223u;
            examples[static_cast<size_t>(exampleIndex)].targetTokenIds[index] = static_cast<int>(state % static_cast<unsigned>(vocabularySize));
        }
    }
    std::vector<const LanguageModelExample*> packPointers(static_cast<size_t>(packBatchSize));
    for (int exampleIndex = 0; exampleIndex < packBatchSize; ++exampleIndex)
        packPointers[static_cast<size_t>(exampleIndex)] = &examples[static_cast<size_t>(exampleIndex)];

    CudaAdam::preferInt8Moments = false;
    fp32Device.ensureTrainState();
    CudaAdam::preferInt8Moments = true;
    int8Device.ensureTrainState();

    const int stepCount = 24;
    for (int step = 0; step < stepCount; ++step) {
        CudaAdam::preferInt8Moments = false;
        fp32Device.trainGradients.zeroInPlace();
        fp32Device.accumulatePackedExamples(packPointers.data(), packBatchSize, fp32Device.trainGradients);
        fp32Device.applyGradients(fp32Device.trainGradients, 1.0f / static_cast<float>(packBatchSize));

        CudaAdam::preferInt8Moments = true;
        int8Device.trainGradients.zeroInPlace();
        int8Device.accumulatePackedExamples(packPointers.data(), packBatchSize, int8Device.trainGradients);
        int8Device.applyGradients(int8Device.trainGradients, 1.0f / static_cast<float>(packBatchSize));
    }
    CudaMatmul::throwIfCudaFailed(cudaDeviceSynchronize(), "CudaLanguageModel train int8 Adam synchronize");

    CudaAdam::preferInt8Moments = false;
    fp32Device.epochLossSum.ensureSize(1, 1);
    CudaOps::zeroInPlace(fp32Device.epochLossSum);
    fp32Device.accumulatePackedExamples(packPointers.data(), packBatchSize, fp32Device.trainGradients);
    const float fp32Loss = fp32Device.epochLossSum.download().at(0, 0) / static_cast<float>(packBatchSize);

    CudaAdam::preferInt8Moments = true;
    int8Device.epochLossSum.ensureSize(1, 1);
    CudaOps::zeroInPlace(int8Device.epochLossSum);
    int8Device.accumulatePackedExamples(packPointers.data(), packBatchSize, int8Device.trainGradients);
    const float int8Loss = int8Device.epochLossSum.download().at(0, 0) / static_cast<float>(packBatchSize);

    Matrix fp32Weight = fp32Device.projectionWeight.download();
    Matrix int8Weight = int8Device.projectionWeight.download();
    float weightDifference = 0.0f;
    bool anyNonFinite = false;
    for (size_t index = 0; index < fp32Weight.data.size(); ++index) {
        if (!std::isfinite(fp32Weight.data[index]) || !std::isfinite(int8Weight.data[index]))
            anyNonFinite = true;
        weightDifference = (std::max)(weightDifference, std::fabs(fp32Weight.data[index] - int8Weight.data[index]));
    }

    SmokeLog::result("LanguageModel train int8 Adam", "vocab=%d embed=%d seq=%d steps=%d  loss fp32=%.4f int8=%.4f  weightDiff=%.2e  nonFinite=%s",
        vocabularySize, embeddingDim, sequenceLength, stepCount, fp32Loss, int8Loss, weightDifference, anyNonFinite ? "yes" : "no");

    CudaAmp::preferMixedPrecision = previousAmp;
    CudaAdam::preferInt8Moments = previousInt8;
}

void CudaLanguageModel::runTrainProfileDemo(int vocabularySize, int embeddingDim, int sequenceLength, int blockCount, int headCount, bool preferFlash, int maxPackedColumns) {
    if (!CudaMatmul::isAvailable()) {
        SmokeLog::skip("LanguageModel profile");
        return;
    }
    if (vocabularySize <= 0 || embeddingDim <= 0 || sequenceLength <= 0 || blockCount <= 0 || headCount <= 0)
        throw std::invalid_argument("CudaLanguageModel::runTrainProfileDemo invalid dims");

    LanguageModel host(vocabularySize, embeddingDim, sequenceLength, Adam(0.001f), blockCount, headCount);
    CudaLanguageModel device = CudaLanguageModel::createFrom(host);
    device.adam = CudaAdam(host.optimizer.learningRate, host.optimizer.beta1, host.optimizer.beta2, host.optimizer.epsilon);
    if (maxPackedColumns > 0)
        device.maxPackedColumns = maxPackedColumns;
    for (CudaTransformerBlock& block : device.blocks) {
        block.attention.preferFlashAttention = preferFlash;
        if (preferFlash)
            block.attention.releaseDenseAttentionScratch();
        else
            block.attention.releaseFlashAttentionScratch();
    }
    const bool previousAmp = CudaAmp::preferMixedPrecision;
    const bool previousLossScaling = CudaAmp::useLossScaling;
    CudaAmp::preferMixedPrecision = true;
    CudaAmp::useLossScaling = embeddingDim >= 256;
    if (CudaAmp::useLossScaling)
        CudaAmp::resetLossScaler();
    device.activationCheckpointing = true;
    device.ensureTrainState();
    device.ensureTrainWorkspaces();
    device.epochLossSum.ensureSize(1, 1);

    const int packBatchSize = (std::max)(1, (std::min)(16, device.maxPackedColumns / sequenceLength));
    std::vector<LanguageModelExample> examples(static_cast<size_t>(packBatchSize));
    unsigned state = 211u;
    for (int exampleIndex = 0; exampleIndex < packBatchSize; ++exampleIndex) {
        examples[static_cast<size_t>(exampleIndex)].inputTokenIds.resize(static_cast<size_t>(sequenceLength));
        examples[static_cast<size_t>(exampleIndex)].targetTokenIds.resize(static_cast<size_t>(sequenceLength));
        for (size_t index = 0; index < static_cast<size_t>(sequenceLength); ++index) {
            state = state * 1664525u + 1013904223u;
            examples[static_cast<size_t>(exampleIndex)].inputTokenIds[index] = static_cast<int>(state % static_cast<unsigned>(vocabularySize));
            state = state * 1664525u + 1013904223u;
            examples[static_cast<size_t>(exampleIndex)].targetTokenIds[index] = static_cast<int>(state % static_cast<unsigned>(vocabularySize));
        }
    }

    std::vector<const LanguageModelExample*> packPointers(static_cast<size_t>(packBatchSize));
    for (int exampleIndex = 0; exampleIndex < packBatchSize; ++exampleIndex)
        packPointers[static_cast<size_t>(exampleIndex)] = &examples[static_cast<size_t>(exampleIndex)];

    device.packedInputTokenIds.clear();
    device.packedTargetTokenIds.clear();
    for (int exampleIndex = 0; exampleIndex < packBatchSize; ++exampleIndex) {
        device.packedInputTokenIds.insert(device.packedInputTokenIds.end(), examples[static_cast<size_t>(exampleIndex)].inputTokenIds.begin(), examples[static_cast<size_t>(exampleIndex)].inputTokenIds.end());
        device.packedTargetTokenIds.insert(device.packedTargetTokenIds.end(), examples[static_cast<size_t>(exampleIndex)].targetTokenIds.begin(), examples[static_cast<size_t>(exampleIndex)].targetTokenIds.end());
    }
    const size_t tokenCount = device.packedInputTokenIds.size();
    const int meanDivisor = sequenceLength;

    cudaEvent_t startEvent = nullptr;
    cudaEvent_t stopEvent = nullptr;
    CudaMatmul::throwIfCudaFailed(cudaEventCreate(&startEvent), "profile cudaEventCreate start");
    CudaMatmul::throwIfCudaFailed(cudaEventCreate(&stopEvent), "profile cudaEventCreate stop");

    auto gpuMs = [&](auto&& work) -> float {
        CudaMatmul::throwIfCudaFailed(cudaEventRecord(startEvent), "profile cudaEventRecord start");
        work();
        CudaMatmul::throwIfCudaFailed(cudaEventRecord(stopEvent), "profile cudaEventRecord stop");
        CudaMatmul::throwIfCudaFailed(cudaEventSynchronize(stopEvent), "profile cudaEventSynchronize");
        float milliseconds = 0.0f;
        CudaMatmul::throwIfCudaFailed(cudaEventElapsedTime(&milliseconds, startEvent, stopEvent), "profile cudaEventElapsedTime");
        return milliseconds;
    };

    auto runTimedStep = [&](float& hostPackMs, float& h2dMs, float& embedMs, float& attnMs, float& ffnMs, float& headFwdMs, float& ceMs, float& headBwdMs, float& attnBwdMs, float& ffnBwdMs, float& scatterMs, float& adamMs) {
        hostPackMs = 0.0f;
        h2dMs = 0.0f;
        embedMs = 0.0f;
        attnMs = 0.0f;
        ffnMs = 0.0f;
        headFwdMs = 0.0f;
        ceMs = 0.0f;
        headBwdMs = 0.0f;
        attnBwdMs = 0.0f;
        ffnBwdMs = 0.0f;
        scatterMs = 0.0f;
        adamMs = 0.0f;

        const auto hostPackStart = std::chrono::steady_clock::now();
        device.packedInputTokenIds.clear();
        device.packedTargetTokenIds.clear();
        for (int exampleIndex = 0; exampleIndex < packBatchSize; ++exampleIndex) {
            device.packedInputTokenIds.insert(device.packedInputTokenIds.end(), examples[static_cast<size_t>(exampleIndex)].inputTokenIds.begin(), examples[static_cast<size_t>(exampleIndex)].inputTokenIds.end());
            device.packedTargetTokenIds.insert(device.packedTargetTokenIds.end(), examples[static_cast<size_t>(exampleIndex)].targetTokenIds.begin(), examples[static_cast<size_t>(exampleIndex)].targetTokenIds.end());
        }
        hostPackMs = static_cast<float>(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - hostPackStart).count());

        device.trainGradients.zeroInPlace();

        h2dMs += gpuMs([&]() {
            device.tokenIdsBuffer.ensureCapacity(tokenCount);
            device.tokenIdsBuffer.copyFromHost(device.packedInputTokenIds.data(), tokenCount);
        });

        embedMs += gpuMs([&]() {
            CudaOps::embeddingGatherInto(device.tokenEmbeddingWeight, device.tokenIdsBuffer, tokenCount, device.hidden);
        });

        for (size_t blockIndex = 0; blockIndex < device.blocks.size(); ++blockIndex) {
            CudaTransformerBlock& block = device.blocks[blockIndex];
            attnMs += gpuMs([&]() {
                block.attentionNorm.forward(device.hidden, block.attentionInput);
                block.attention.forward(block.attentionInput, block.attended, meanDivisor);
                CudaOps::addInto(device.hidden, block.attended, block.afterAttention);
            });
            ffnMs += gpuMs([&]() {
                block.feedForwardNorm.forward(block.afterAttention, block.feedForwardInput);
                block.feedForward.forward(block.feedForwardInput, block.feedForwardOutput);
                CudaOps::addInto(block.afterAttention, block.feedForwardOutput, device.normalized);
            });
            CudaMatrix swapBuffer = std::move(device.hidden);
            device.hidden = std::move(device.normalized);
            device.normalized = std::move(swapBuffer);
        }

        headFwdMs += gpuMs([&]() {
            device.finalNorm.forward(device.hidden, device.normalized);
            CudaMatrix::multiplyInto(device.projectionWeight, device.normalized, device.logits);
            CudaOps::broadcastBiasAddInPlace(device.logits, device.projectionBias);
        });

        h2dMs += gpuMs([&]() {
            device.targetTokenIdsBuffer.ensureCapacity(tokenCount);
            device.targetTokenIdsBuffer.copyFromHost(device.packedTargetTokenIds.data(), tokenCount);
        });

        ceMs += gpuMs([&]() {
            CudaOps::softmaxCrossEntropyFromLogitsInto(device.logits, device.targetTokenIdsBuffer, tokenCount, device.probabilities, device.logitGradient, device.epochLossSum, 1.0f, meanDivisor);
        });

        headBwdMs += gpuMs([&]() {
            CudaMatrix::multiplyInto(device.logitGradient, device.normalized, device.projectionWeightGradient, false, true);
            CudaOps::addInPlace(device.trainGradients.projectionWeight, device.projectionWeightGradient);
            CudaOps::sumColumnsInto(device.logitGradient, device.projectionBiasGradient);
            CudaOps::addInPlace(device.trainGradients.projectionBias, device.projectionBiasGradient);
            CudaMatrix::multiplyInto(device.projectionWeight, device.logitGradient, device.hiddenGradient, true, false);
            device.finalNorm.backward(device.hiddenGradient, device.normInputGradientScratch, device.finalNormGammaGradient);
            CudaOps::addInPlace(device.trainGradients.finalNormGamma, device.finalNormGammaGradient);
            std::swap(device.hiddenGradient, device.normInputGradientScratch);
        });

        for (int blockIndex = static_cast<int>(device.blocks.size()) - 1; blockIndex >= 0; --blockIndex) {
            CudaTransformerBlock& block = device.blocks[static_cast<size_t>(blockIndex)];
            CudaTransformerBlockGradients& blockGradients = device.trainGradients.blocks[static_cast<size_t>(blockIndex)];

            ffnBwdMs += gpuMs([&]() {
                block.feedForward.backward(device.hiddenGradient, block.feedForwardInputGradient, block.feedForwardGateWeightGradient, block.feedForwardGateBiasGradient, block.feedForwardUpWeightGradient, block.feedForwardUpBiasGradient, block.feedForwardDownWeightGradient, block.feedForwardDownBiasGradient);
                block.feedForwardNorm.backward(block.feedForwardInputGradient, block.afterAttentionFromFeedForward, block.feedForwardNormGammaGradient);
                CudaOps::addInto(device.hiddenGradient, block.afterAttentionFromFeedForward, block.afterAttentionGradient);
                CudaOps::addInPlace(blockGradients.feedForwardDownWeight, block.feedForwardDownWeightGradient);
                CudaOps::addInPlace(blockGradients.feedForwardDownBias, block.feedForwardDownBiasGradient);
                CudaOps::addInPlace(blockGradients.feedForwardUpWeight, block.feedForwardUpWeightGradient);
                CudaOps::addInPlace(blockGradients.feedForwardUpBias, block.feedForwardUpBiasGradient);
                CudaOps::addInPlace(blockGradients.feedForwardGateWeight, block.feedForwardGateWeightGradient);
                CudaOps::addInPlace(blockGradients.feedForwardGateBias, block.feedForwardGateBiasGradient);
                CudaOps::addInPlace(blockGradients.feedForwardNormGamma, block.feedForwardNormGammaGradient);
            });

            attnBwdMs += gpuMs([&]() {
                block.attention.backward(block.afterAttentionGradient, block.attentionInputGradient, block.queryWeightGradient, block.keyWeightGradient, block.valueWeightGradient, block.attentionOutputWeightGradient);
                block.attentionNorm.backward(block.attentionInputGradient, block.inputFromAttention, block.attentionNormGammaGradient);
                CudaOps::addInPlace(blockGradients.attentionOutputWeight, block.attentionOutputWeightGradient);
                CudaOps::addInPlace(blockGradients.valueWeight, block.valueWeightGradient);
                CudaOps::addInPlace(blockGradients.keyWeight, block.keyWeightGradient);
                CudaOps::addInPlace(blockGradients.queryWeight, block.queryWeightGradient);
                CudaOps::addInPlace(blockGradients.attentionNormGamma, block.attentionNormGammaGradient);
                CudaOps::addInto(block.afterAttentionGradient, block.inputFromAttention, device.blockInputGradientScratch);
            });
            std::swap(device.hiddenGradient, device.blockInputGradientScratch);
        }

        scatterMs += gpuMs([&]() {
            CudaOps::embeddingScatterAddInto(device.trainGradients.tokenEmbedding, device.tokenIdsBuffer, tokenCount, device.hiddenGradient);
        });

        adamMs += gpuMs([&]() {
            device.applyGradients(device.trainGradients, 1.0f / static_cast<float>(packBatchSize));
        });
    };

    float discard[12];
    for (int warm = 0; warm < 3; ++warm)
        runTimedStep(discard[0], discard[1], discard[2], discard[3], discard[4], discard[5], discard[6], discard[7], discard[8], discard[9], discard[10], discard[11]);

    float sumHostPack = 0.0f, sumH2d = 0.0f, sumEmbed = 0.0f, sumAttn = 0.0f, sumFfn = 0.0f, sumHeadFwd = 0.0f;
    float sumCe = 0.0f, sumHeadBwd = 0.0f, sumAttnBwd = 0.0f, sumFfnBwd = 0.0f, sumScatter = 0.0f, sumAdam = 0.0f;
    const int timedStepCount = 20;
    for (int step = 0; step < timedStepCount; ++step) {
        float hostPackMs, h2dMs, embedMs, attnMs, ffnMs, headFwdMs, ceMs, headBwdMs, attnBwdMs, ffnBwdMs, scatterMs, adamMs;
        CudaOps::zeroInPlace(device.epochLossSum);
        runTimedStep(hostPackMs, h2dMs, embedMs, attnMs, ffnMs, headFwdMs, ceMs, headBwdMs, attnBwdMs, ffnBwdMs, scatterMs, adamMs);
        sumHostPack += hostPackMs;
        sumH2d += h2dMs;
        sumEmbed += embedMs;
        sumAttn += attnMs;
        sumFfn += ffnMs;
        sumHeadFwd += headFwdMs;
        sumCe += ceMs;
        sumHeadBwd += headBwdMs;
        sumAttnBwd += attnBwdMs;
        sumFfnBwd += ffnBwdMs;
        sumScatter += scatterMs;
        sumAdam += adamMs;
    }

    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);

    const float inv = 1.0f / static_cast<float>(timedStepCount);
    sumHostPack *= inv; sumH2d *= inv; sumEmbed *= inv; sumAttn *= inv; sumFfn *= inv; sumHeadFwd *= inv;
    sumCe *= inv; sumHeadBwd *= inv; sumAttnBwd *= inv; sumFfnBwd *= inv; sumScatter *= inv; sumAdam *= inv;

    const float attnTotal = sumAttn + sumAttnBwd;
    const float ffnTotal = sumFfn + sumFfnBwd;
    const float headTotal = sumHeadFwd + sumHeadBwd + sumCe;
    const float otherTotal = sumHostPack + sumH2d + sumEmbed + sumScatter + sumAdam;
    const float stepTotal = attnTotal + ffnTotal + headTotal + otherTotal;
    const float percent = stepTotal > 0.0f ? (100.0f / stepTotal) : 0.0f;
    const int tokensPerStep = sequenceLength * packBatchSize;
    const float tokensPerSecond = stepTotal > 0.0f ? (1000.0f * static_cast<float>(tokensPerStep) / stepTotal) : 0.0f;

    SmokeLog::section("train profile");
    SmokeLog::result("profile config", "vocab=%d embed=%d seq=%d blocks=%d pack=%d tokens/step=%d flash=%s amp=%s maxPackCols=%d lossScale=%.0f",
        vocabularySize, embeddingDim, sequenceLength, blockCount, packBatchSize, tokensPerStep, preferFlash ? "on" : "off",
        CudaAmp::preferMixedPrecision ? "on" : "off", device.maxPackedColumns, CudaAmp::lossScaler.scale);
    SmokeLog::result("profile step", "avg=%.2fms  ~tokens/s=%.0f", stepTotal, tokensPerSecond);
    SmokeLog::result("  attention", "fwd=%.2fms bwd=%.2fms  total=%.2fms (%.0f%%)", sumAttn, sumAttnBwd, attnTotal, attnTotal * percent);
    SmokeLog::result("  ffn", "fwd=%.2fms bwd=%.2fms  total=%.2fms (%.0f%%)", sumFfn, sumFfnBwd, ffnTotal, ffnTotal * percent);
    SmokeLog::result("  head+ce", "fwd=%.2fms ce=%.2fms bwd=%.2fms  total=%.2fms (%.0f%%)", sumHeadFwd, sumCe, sumHeadBwd, headTotal, headTotal * percent);
    SmokeLog::result("  embed+h2d", "embed=%.2fms scatter=%.2fms h2d=%.2fms hostPack=%.2fms", sumEmbed, sumScatter, sumH2d, sumHostPack);
    SmokeLog::result("  adam", "%.2fms (%.0f%%)", sumAdam, sumAdam * percent);
    SmokeLog::result("  other sum", "%.2fms (%.0f%%)", otherTotal, otherTotal * percent);
    CudaAmp::preferMixedPrecision = previousAmp;
    CudaAmp::useLossScaling = previousLossScaling;
}

void CudaLanguageModel::uploadFrom(const LanguageModel& host) {
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaLanguageModel::uploadFrom no CUDA device");

    this->tokenEmbeddingWeight.upload(host.tokenEmbedding.weight);
    this->blocks.clear();
    this->blocks.reserve(host.blocks.size());
    for (const TransformerBlock& block : host.blocks)
        this->blocks.push_back(CudaTransformerBlock::createFrom(block));

    this->finalNorm.uploadFrom(host.finalNorm);
    this->projectionWeight.upload(host.outputProjection.weight);
    this->projectionBias.upload(host.outputProjection.bias);
    this->maximumPositionCount = host.maximumPositionCount;
    this->kvCaches.clear();
    this->kvCaches.resize(this->blocks.size());
    this->trainStateReady = false;
}

CudaLanguageModel CudaLanguageModel::createFrom(const LanguageModel& host) {
    CudaLanguageModel device;
    device.uploadFrom(host);
    return device;
}

void CudaLanguageModel::forwardInto(const std::vector<int>& tokenIds, CudaMatrix& outLogits, int segmentLength) {
    if (tokenIds.empty()) throw std::invalid_argument("CudaLanguageModel::forwardInto empty tokenIds");
    if (this->tokenEmbeddingWeight.empty()) throw std::logic_error("CudaLanguageModel::forwardInto weights not uploaded");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaLanguageModel::forwardInto no CUDA device");

    if (segmentLength > 0) {
        if (static_cast<int>(tokenIds.size()) % segmentLength != 0)
            throw std::invalid_argument("CudaLanguageModel::forwardInto tokenCount not divisible by segmentLength");
        if (segmentLength > this->maximumPositionCount)
            throw std::invalid_argument("CudaLanguageModel::forwardInto segmentLength exceeds maximumPositionCount");
    } else if (static_cast<int>(tokenIds.size()) > this->maximumPositionCount) {
        throw std::invalid_argument("CudaLanguageModel::forwardInto sequence longer than maximumPositionCount");
    }

    this->tokenIdsBuffer.ensureCapacity(tokenIds.size());
    this->tokenIdsBuffer.copyFromHost(tokenIds.data(), tokenIds.size());

    CudaOps::embeddingGatherInto(this->tokenEmbeddingWeight, this->tokenIdsBuffer, tokenIds.size(), this->hidden);

    if (this->activationCheckpointing) {
        if (this->blockInputCheckpoints.size() != this->blocks.size())
            this->blockInputCheckpoints.resize(this->blocks.size());
        for (CudaMatrix& checkpoint : this->blockInputCheckpoints)
            checkpoint.ensureSize(this->tokenEmbeddingWeight.cols, tokenIds.size());
    }

    for (size_t blockIndex = 0; blockIndex < this->blocks.size(); ++blockIndex) {
        if (this->activationCheckpointing)
            CudaOps::copyInto(this->hidden, this->blockInputCheckpoints[blockIndex]);
        this->blocks[blockIndex].forward(this->hidden, this->normalized, segmentLength);
        CudaMatrix swapBuffer = std::move(this->hidden);
        this->hidden = std::move(this->normalized);
        this->normalized = std::move(swapBuffer);
    }

    this->finalNorm.forward(this->hidden, this->normalized);
    CudaMatrix::multiplyInto(this->projectionWeight, this->normalized, outLogits);
    CudaOps::broadcastBiasAddInPlace(outLogits, this->projectionBias);
}

Matrix CudaLanguageModel::forward(const std::vector<int>& tokenIds) {
    this->forwardInto(tokenIds, this->logits);
    return this->logits.download();
}

void CudaLanguageModel::resetKvCaches() {
    if (this->blocks.empty()) throw std::logic_error("CudaLanguageModel::resetKvCaches no blocks");
    if (this->tokenEmbeddingWeight.empty()) throw std::logic_error("CudaLanguageModel::resetKvCaches weights not uploaded");

    const int embeddingDim = static_cast<int>(this->tokenEmbeddingWeight.cols);
    this->kvCaches.resize(this->blocks.size());
    for (size_t blockIndex = 0; blockIndex < this->kvCaches.size(); ++blockIndex)
        this->kvCaches[blockIndex].ensureCapacity(embeddingDim, this->maximumPositionCount);
}

void CudaLanguageModel::prefillInto(const std::vector<int>& tokenIds, CudaMatrix& outLogits) {
    if (tokenIds.empty()) throw std::invalid_argument("CudaLanguageModel::prefillInto empty tokenIds");
    if (this->tokenEmbeddingWeight.empty()) throw std::logic_error("CudaLanguageModel::prefillInto weights not uploaded");
    if (static_cast<int>(tokenIds.size()) > this->maximumPositionCount)
        throw std::invalid_argument("CudaLanguageModel::prefillInto sequence longer than maximumPositionCount");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaLanguageModel::prefillInto no CUDA device");

    this->resetKvCaches();
    this->tokenIdsBuffer.ensureCapacity(tokenIds.size());
    this->tokenIdsBuffer.copyFromHost(tokenIds.data(), tokenIds.size());
    CudaOps::embeddingGatherInto(this->tokenEmbeddingWeight, this->tokenIdsBuffer, tokenIds.size(), this->hidden);

    for (size_t blockIndex = 0; blockIndex < this->blocks.size(); ++blockIndex) {
        this->blocks[blockIndex].prefill(this->hidden, this->kvCaches[blockIndex], this->normalized);
        CudaMatrix swapBuffer = std::move(this->hidden);
        this->hidden = std::move(this->normalized);
        this->normalized = std::move(swapBuffer);
    }

    this->finalNorm.forward(this->hidden, this->normalized);
    CudaMatrix::multiplyInto(this->projectionWeight, this->normalized, outLogits);
    CudaOps::broadcastBiasAddInPlace(outLogits, this->projectionBias);
}

void CudaLanguageModel::decodeInto(int tokenId, CudaMatrix& outLogits) {
    if (this->tokenEmbeddingWeight.empty()) throw std::logic_error("CudaLanguageModel::decodeInto weights not uploaded");
    if (this->kvCaches.empty()) throw std::logic_error("CudaLanguageModel::decodeInto caches not allocated");
    if (this->kvCaches[0].length >= this->maximumPositionCount)
        throw std::invalid_argument("CudaLanguageModel::decodeInto cache full");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaLanguageModel::decodeInto no CUDA device");

    const int tokenIdsHost[1] = { tokenId };
    this->tokenIdsBuffer.ensureCapacity(1);
    this->tokenIdsBuffer.copyFromHost(tokenIdsHost, 1);
    CudaOps::embeddingGatherInto(this->tokenEmbeddingWeight, this->tokenIdsBuffer, 1, this->hidden);

    for (size_t blockIndex = 0; blockIndex < this->blocks.size(); ++blockIndex) {
        this->blocks[blockIndex].decode(this->hidden, this->kvCaches[blockIndex], this->normalized);
        CudaMatrix swapBuffer = std::move(this->hidden);
        this->hidden = std::move(this->normalized);
        this->normalized = std::move(swapBuffer);
    }

    this->finalNorm.forward(this->hidden, this->normalized);
    CudaMatrix::multiplyInto(this->projectionWeight, this->normalized, outLogits);
    CudaOps::broadcastBiasAddInPlace(outLogits, this->projectionBias);
}

void CudaLanguageModel::runSmokeDemo(int vocabularySize, int embeddingDim, int sequenceLength, int blockCount, int headCount) {
    if (!CudaMatmul::isAvailable()) {
        SmokeLog::skip("LanguageModel fwd");
        return;
    }
    if (vocabularySize <= 0 || embeddingDim <= 0 || sequenceLength <= 0 || blockCount <= 0 || headCount <= 0)
        throw std::invalid_argument("CudaLanguageModel::runSmokeDemo invalid dims");

    LanguageModel host(vocabularySize, embeddingDim, sequenceLength, Adam(0.001f), blockCount, headCount);

    std::vector<int> tokenIds(static_cast<size_t>(sequenceLength), 0);
    unsigned state = 91u;
    for (size_t index = 0; index < tokenIds.size(); ++index) {
        state = state * 1664525u + 1013904223u;
        tokenIds[index] = static_cast<int>(state % static_cast<unsigned>(vocabularySize));
    }

    const auto cpuStart = std::chrono::steady_clock::now();
    Matrix hostLogits = host.forward(tokenIds);
    const double cpuMilliseconds = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - cpuStart).count();

    const auto uploadStart = std::chrono::steady_clock::now();
    CudaLanguageModel device = CudaLanguageModel::createFrom(host);
    const double uploadMilliseconds = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - uploadStart).count();

    Matrix warmLogits = device.forward(tokenIds);
    CudaMatmul::throwIfCudaFailed(cudaDeviceSynchronize(), "CudaLanguageModel warm synchronize");

    cudaEvent_t kernelStartEvent = nullptr;
    cudaEvent_t kernelStopEvent = nullptr;
    CudaMatmul::throwIfCudaFailed(cudaEventCreate(&kernelStartEvent), "cudaEventCreate start");
    CudaMatmul::throwIfCudaFailed(cudaEventCreate(&kernelStopEvent), "cudaEventCreate stop");
    CudaMatmul::throwIfCudaFailed(cudaEventRecord(kernelStartEvent), "cudaEventRecord start");
    device.forwardInto(tokenIds, device.logits);
    CudaMatmul::throwIfCudaFailed(cudaEventRecord(kernelStopEvent), "cudaEventRecord stop");
    CudaMatmul::throwIfCudaFailed(cudaEventSynchronize(kernelStopEvent), "cudaEventSynchronize stop");
    float deviceMilliseconds = 0.0f;
    CudaMatmul::throwIfCudaFailed(cudaEventElapsedTime(&deviceMilliseconds, kernelStartEvent, kernelStopEvent), "cudaEventElapsedTime");
    cudaEventDestroy(kernelStartEvent);
    cudaEventDestroy(kernelStopEvent);

    const auto downloadStart = std::chrono::steady_clock::now();
    Matrix deviceLogits = device.logits.download();
    const double downloadMilliseconds = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - downloadStart).count();

    float maximumDifference = 0.0f;
    for (size_t index = 0; index < hostLogits.data.size(); ++index)
        maximumDifference = (std::max)(maximumDifference, std::fabs(hostLogits.data[index] - deviceLogits.data[index]));

    SmokeLog::result("LanguageModel fwd", "vocab=%d embed=%d seq=%d  cpu=%.2fms  gpu=%.2fms  diff=%.2e",
        vocabularySize, embeddingDim, sequenceLength, cpuMilliseconds, static_cast<double>(deviceMilliseconds), maximumDifference);
    (void)warmLogits;
    (void)uploadMilliseconds;
    (void)downloadMilliseconds;
    (void)blockCount;
    (void)headCount;
}

void CudaLanguageModel::runKvCacheSmokeDemo(int vocabularySize, int embeddingDim, int sequenceLength, int blockCount, int headCount) {
    if (!CudaMatmul::isAvailable()) {
        SmokeLog::skip("LanguageModel KV");
        return;
    }
    if (vocabularySize <= 0 || embeddingDim <= 0 || sequenceLength < 2 || blockCount <= 0 || headCount <= 0)
        throw std::invalid_argument("CudaLanguageModel::runKvCacheSmokeDemo invalid dims");

    LanguageModel host(vocabularySize, embeddingDim, sequenceLength, Adam(0.001f), blockCount, headCount);

    std::vector<int> tokenIds(static_cast<size_t>(sequenceLength), 0);
    unsigned state = 97u;
    for (size_t index = 0; index < tokenIds.size(); ++index) {
        state = state * 1664525u + 1013904223u;
        tokenIds[index] = static_cast<int>(state % static_cast<unsigned>(vocabularySize));
    }

    CudaLanguageModel device = CudaLanguageModel::createFrom(host);

    CudaMatrix fullLogits;
    device.forwardInto(tokenIds, fullLogits);

    std::vector<int> prefixIds(tokenIds.begin(), tokenIds.end() - 1);
    const int lastTokenId = tokenIds.back();

    CudaMatrix prefillLogits;
    CudaMatrix decodeLogits;
    device.prefillInto(prefixIds, prefillLogits);
    device.decodeInto(lastTokenId, decodeLogits);

    Matrix fullHost = fullLogits.download();
    Matrix decodeHost = decodeLogits.download();
    const int vocabularyRows = static_cast<int>(fullHost.rows);
    float maximumDifference = 0.0f;
    for (int row = 0; row < vocabularyRows; ++row) {
        const float fullValue = fullHost.at(static_cast<size_t>(row), static_cast<size_t>(sequenceLength - 1));
        const float decodeValue = decodeHost.at(static_cast<size_t>(row), 0);
        maximumDifference = (std::max)(maximumDifference, std::fabs(fullValue - decodeValue));
    }

    SmokeLog::result("LanguageModel KV", "vocab=%d embed=%d seq=%d  cache=%d  diff=%.2e",
        vocabularySize, embeddingDim, sequenceLength, device.kvCaches.empty() ? 0 : device.kvCaches[0].length, maximumDifference);
    (void)blockCount;
    (void)headCount;
}
