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
    : maximumPositionCount(0), tieEmbeddingProjection(true), maxPackedColumns(8192), maxPackedColumnsManual(false), logitChunkRows(2048), gradientAccumulationSteps(4), activationCheckpointing(true), adam(0.001f), trainStateReady(false) {}

void CudaTransformerBlockAdamStates::ensureFrom(const CudaTransformerBlock& block) {
    // moments stay empty until first Adam update (lazy allocation for low VRAM before train)
    (void)block;
}

void CudaTransformerBlockAdamStates::free() {
    this->queryWeight.free();
    this->keyWeight.free();
    this->valueWeight.free();
    this->attentionOutputWeight.free();
    this->attentionNormGamma.free();
    this->feedForwardNormGamma.free();
    this->feedForwardGateWeight.free();
    this->feedForwardGateBias.free();
    this->feedForwardUpWeight.free();
    this->feedForwardUpBias.free();
    this->feedForwardDownWeight.free();
    this->feedForwardDownBias.free();
}

void CudaLanguageModelGradients::ensureFrom(const CudaLanguageModel& model) {
    this->tokenEmbedding.ensureSize(model.tokenEmbeddingWeight.rows, model.tokenEmbeddingWeight.cols);
    this->blocks.resize(model.blocks.size());
    for (size_t blockIndex = 0; blockIndex < model.blocks.size(); ++blockIndex)
        this->blocks[blockIndex].ensureFrom(model.blocks[blockIndex]);
    this->finalNormGamma.ensureSize(model.finalNorm.gamma.rows, model.finalNorm.gamma.cols);
    if (model.tieEmbeddingProjection)
        this->projectionWeight.free();
    else
        this->projectionWeight.ensureSize(model.projectionWeight.rows, model.projectionWeight.cols);
    this->projectionBias.ensureSize(model.projectionBias.rows, model.projectionBias.cols);
}

void CudaLanguageModelGradients::zeroInPlace() {
    CudaOps::zeroInPlace(this->tokenEmbedding);
    for (CudaTransformerBlockGradients& block : this->blocks)
        block.zeroInPlace();
    CudaOps::zeroInPlace(this->finalNormGamma);
    if (!this->projectionWeight.empty())
        CudaOps::zeroInPlace(this->projectionWeight);
    CudaOps::zeroInPlace(this->projectionBias);
}

void CudaLanguageModelGradients::zeroInPlaceExceptEmbedding() {
    for (CudaTransformerBlockGradients& block : this->blocks)
        block.zeroInPlace();
    CudaOps::zeroInPlace(this->finalNormGamma);
    if (!this->projectionWeight.empty())
        CudaOps::zeroInPlace(this->projectionWeight);
    CudaOps::zeroInPlace(this->projectionBias);
}

void CudaLanguageModelGradients::addInPlace(const CudaLanguageModelGradients& other) {
    CudaOps::addInPlace(this->tokenEmbedding, other.tokenEmbedding);
    for (size_t blockIndex = 0; blockIndex < this->blocks.size(); ++blockIndex)
        this->blocks[blockIndex].addInPlace(other.blocks[blockIndex]);
    CudaOps::addInPlace(this->finalNormGamma, other.finalNormGamma);
    if (!this->projectionWeight.empty() && !other.projectionWeight.empty())
        CudaOps::addInPlace(this->projectionWeight, other.projectionWeight);
    CudaOps::addInPlace(this->projectionBias, other.projectionBias);
}

void CudaLanguageModelGradients::scaleInPlace(float scalar) {
    CudaOps::scaleInPlace(this->tokenEmbedding, scalar);
    for (CudaTransformerBlockGradients& block : this->blocks)
        block.scaleInPlace(scalar);
    CudaOps::scaleInPlace(this->finalNormGamma, scalar);
    if (!this->projectionWeight.empty())
        CudaOps::scaleInPlace(this->projectionWeight, scalar);
    CudaOps::scaleInPlace(this->projectionBias, scalar);
}

bool CudaLanguageModel::useHalfActivationCheckpoints() const {
    return this->activationCheckpointing && CudaAmp::preferMixedPrecision;
}

CudaMatrix& CudaLanguageModel::lmHeadWeight() {
    return this->tieEmbeddingProjection ? this->tokenEmbeddingWeight : this->projectionWeight;
}

const CudaMatrix& CudaLanguageModel::lmHeadWeight() const {
    return this->tieEmbeddingProjection ? this->tokenEmbeddingWeight : this->projectionWeight;
}

void CudaLanguageModel::ensureTrainState() {
    if (this->trainStateReady) return;
    if (this->tokenEmbeddingWeight.empty()) throw std::logic_error("CudaLanguageModel::ensureTrainState weights not uploaded");

    this->trainGradients.ensureFrom(*this);
    this->trainGradients.zeroInPlace();

    if (CudaAdam::preferCpuOffload) {
        // moments live on host; free any leftover device Adam buffers
        this->tokenEmbeddingState.free();
        this->finalNormGammaState.free();
        this->projectionWeightState.free();
        this->projectionBiasState.free();
        for (CudaTransformerBlockAdamStates& blockStates : this->blockAdamStates)
            blockStates.free();
        this->blockAdamStates.clear();
        this->hostBlockAdamStates.resize(this->blocks.size());
    } else {
        this->hostBlockAdamStates.clear();
        this->hostTokenEmbeddingState = AdamState{};
        this->hostFinalNormGammaState = AdamState{};
        this->hostProjectionWeightState = AdamState{};
        this->hostProjectionBiasState = AdamState{};
        // Adam moments allocated lazily on first applyGradients
        this->blockAdamStates.resize(this->blocks.size());
        for (size_t blockIndex = 0; blockIndex < this->blocks.size(); ++blockIndex)
            this->blockAdamStates[blockIndex].ensureFrom(this->blocks[blockIndex]);
    }

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
    this->packH2dDevice.ensureCapacity(3 * maxColumns);
    this->adamWindowTokenIdsBuffer.ensureCapacity(maxColumns);

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

    // FP16 activation checkpoints when AMP is on (saturated cast); else FP32.
    // Restore scratch holds one block input in FP32 during backward recompute.
    if (!this->activationCheckpointing) {
        this->releaseActivationCheckpoints();
    } else if (this->useHalfActivationCheckpoints()) {
        for (CudaMatrix& checkpoint : this->blockInputCheckpoints)
            checkpoint.free();
        this->blockInputCheckpoints.clear();
        if (this->blockInputCheckpointsHalf.size() != this->blocks.size())
            this->blockInputCheckpointsHalf.resize(this->blocks.size());
        for (CudaHalfMatrix& checkpoint : this->blockInputCheckpointsHalf)
            checkpoint.ensureSize(embeddingDim, maxColumns);
        this->checkpointRestoreScratch.ensureSize(embeddingDim, maxColumns);
    } else {
        this->checkpointRestoreScratch.free();
        for (CudaHalfMatrix& checkpoint : this->blockInputCheckpointsHalf)
            checkpoint.free();
        this->blockInputCheckpointsHalf.clear();
        if (this->blockInputCheckpoints.size() != this->blocks.size())
            this->blockInputCheckpoints.resize(this->blocks.size());
        for (CudaMatrix& checkpoint : this->blockInputCheckpoints)
            checkpoint.ensureSize(embeddingDim, maxColumns);
    }
}

size_t CudaLanguageModel::bytesPerPackedColumn() const {
    if (this->tokenEmbeddingWeight.empty())
        throw std::logic_error("CudaLanguageModel::bytesPerPackedColumn weights not uploaded");
    if (this->logitChunkRows <= 0)
        throw std::invalid_argument("CudaLanguageModel::bytesPerPackedColumn logitChunkRows must be > 0");

    const size_t embeddingDim = this->tokenEmbeddingWeight.cols;
    const size_t vocabularySize = this->tokenEmbeddingWeight.rows;
    const size_t chunkRows = static_cast<size_t>((std::min)(this->logitChunkRows, static_cast<int>(vocabularySize)));
    const size_t floatBytes = sizeof(float);
    const size_t intBytes = sizeof(int);

    // Mirrors ensureTrainWorkspaces column-scaling tensors.
    size_t bytes = 0;
    bytes += 5 * embeddingDim * floatBytes; // hidden, normalized, hiddenGradient, blockInputGrad, normInputGrad
    bytes += 2 * chunkRows * floatBytes;    // logitChunk, logitGradientChunk
    bytes += embeddingDim * floatBytes;     // hiddenGradientChunk
    bytes += 3 * floatBytes;                // onlineSoftmaxMax, onlineSoftmaxSumExp, targetLogits
    bytes += 3 * intBytes;                  // tokenIds, targets, meanDivisors
    bytes += 3 * intBytes;                  // packH2d fused staging
    bytes += intBytes;                      // adamWindowTokenIdsBuffer capacity unit

    // Full LM-head logits cache (vocab x cols) for the single-pass CE path.
    bytes += vocabularySize * floatBytes;

    if (this->activationCheckpointing) {
        if (this->useHalfActivationCheckpoints())
            bytes += this->blocks.size() * embeddingDim * 2u
                + embeddingDim * floatBytes; // restore scratch
        else
            bytes += this->blocks.size() * embeddingDim * floatBytes;
    }

    // Transient block/flash peak: QKV / attn / FFN scratch scales with depth.
    bytes += (48ull + 24ull * this->blocks.size()) * embeddingDim * floatBytes;
    return bytes;
}

void CudaLanguageModel::applyVramPackBudget(float freeFraction) {
    if (this->maxPackedColumnsManual) return;
    if (this->tokenEmbeddingWeight.empty())
        throw std::logic_error("CudaLanguageModel::applyVramPackBudget weights not uploaded");
    if (!(freeFraction > 0.0f && freeFraction <= 1.0f))
        throw std::invalid_argument("CudaLanguageModel::applyVramPackBudget freeFraction must be in (0, 1]");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaLanguageModel::applyVramPackBudget no CUDA device");

    size_t freeBytes = 0;
    size_t totalBytes = 0;
    CudaMatmul::throwIfCudaFailed(cudaMemGetInfo(&freeBytes, &totalBytes), "CudaLanguageModel::applyVramPackBudget memGetInfo");

    const size_t perColumn = this->bytesPerPackedColumn();
    if (perColumn == 0) throw std::logic_error("CudaLanguageModel::applyVramPackBudget zero bytesPerPackedColumn");

    const size_t budgetBytes = static_cast<size_t>(static_cast<double>(freeBytes) * static_cast<double>(freeFraction));
    size_t cols = budgetBytes / perColumn;

    int minCols = CudaLanguageModel::lengthBucketStep;
    if (this->maximumPositionCount > minCols) minCols = this->maximumPositionCount;
    constexpr int maxCols = 16384;
    if (cols < static_cast<size_t>(minCols)) cols = static_cast<size_t>(minCols);
    if (cols > static_cast<size_t>(maxCols)) cols = static_cast<size_t>(maxCols);

    cols = (cols / 64u) * 64u;
    if (cols < static_cast<size_t>(minCols)) cols = static_cast<size_t>(minCols);

    this->maxPackedColumns = static_cast<int>(cols);

    const double freeMiB = static_cast<double>(freeBytes) / (1024.0 * 1024.0);
    const double totalMiB = static_cast<double>(totalBytes) / (1024.0 * 1024.0);
    const double workspaceMiB = static_cast<double>(perColumn) * static_cast<double>(this->maxPackedColumns) / (1024.0 * 1024.0);
    std::printf(
        "CudaLanguageModel::applyVramPackBudget: free=%.0f/%.0f MiB  fraction=%.2f  maxPackCols=%d  ~workspace=%.0f MiB\n",
        freeMiB, totalMiB, freeFraction, this->maxPackedColumns, workspaceMiB);
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
    host.tieEmbeddingProjection = this->tieEmbeddingProjection;
    if (this->tieEmbeddingProjection) {
        host.outputProjection.weight = Matrix();
        host.projectionWeightState = AdamState{};
    } else {
        this->projectionWeight.downloadInto(host.outputProjection.weight);
    }
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

    if (CudaAdam::preferCpuOffload) {
        host.tokenEmbeddingState = this->hostTokenEmbeddingState;
        host.finalNormGammaState = this->hostFinalNormGammaState;
        if (this->tieEmbeddingProjection)
            host.projectionWeightState = AdamState{};
        else
            host.projectionWeightState = this->hostProjectionWeightState;
        host.projectionBiasState = this->hostProjectionBiasState;

        if (this->hostBlockAdamStates.size() != this->blocks.size())
            this->hostBlockAdamStates.resize(this->blocks.size());

        for (size_t blockIndex = 0; blockIndex < this->blocks.size(); ++blockIndex) {
            TransformerBlock& hostBlock = host.blocks[blockIndex];
            const CudaTransformerBlockHostAdamStates& states = this->hostBlockAdamStates[blockIndex];
            hostBlock.queryWeightState = states.queryWeight;
            hostBlock.keyWeightState = states.keyWeight;
            hostBlock.valueWeightState = states.valueWeight;
            hostBlock.attentionOutputWeightState = states.attentionOutputWeight;
            hostBlock.attentionNormGammaState = states.attentionNormGamma;
            hostBlock.feedForwardNormGammaState = states.feedForwardNormGamma;
            hostBlock.feedForwardGateWeightState = states.feedForwardGateWeight;
            hostBlock.feedForwardGateBiasState = states.feedForwardGateBias;
            hostBlock.feedForwardUpWeightState = states.feedForwardUpWeight;
            hostBlock.feedForwardUpBiasState = states.feedForwardUpBias;
            hostBlock.feedForwardDownWeightState = states.feedForwardDownWeight;
            hostBlock.feedForwardDownBiasState = states.feedForwardDownBias;
        }
        return;
    }

    auto downloadOrZero = [](const CudaAdamState& state, AdamState& hostState, size_t rows, size_t cols) {
        if (state.empty()) {
            hostState = AdamState::zerosLike(Matrix(rows, cols, 0.0f));
            return;
        }
        state.downloadInto(hostState, rows, cols);
    };

    downloadOrZero(this->tokenEmbeddingState, host.tokenEmbeddingState, this->tokenEmbeddingWeight.rows, this->tokenEmbeddingWeight.cols);
    downloadOrZero(this->finalNormGammaState, host.finalNormGammaState, this->finalNorm.gamma.rows, this->finalNorm.gamma.cols);
    if (this->tieEmbeddingProjection)
        host.projectionWeightState = AdamState{};
    else
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

    if (CudaAdam::preferCpuOffload) {
        this->hostTokenEmbeddingState = host.tokenEmbeddingState;
        this->hostFinalNormGammaState = host.finalNormGammaState;
        if (this->tieEmbeddingProjection)
            this->hostProjectionWeightState = AdamState{};
        else
            this->hostProjectionWeightState = host.projectionWeightState;
        this->hostProjectionBiasState = host.projectionBiasState;

        if (this->hostBlockAdamStates.size() != this->blocks.size())
            this->hostBlockAdamStates.resize(this->blocks.size());

        for (size_t blockIndex = 0; blockIndex < this->blocks.size(); ++blockIndex) {
            const TransformerBlock& hostBlock = host.blocks[blockIndex];
            CudaTransformerBlockHostAdamStates& states = this->hostBlockAdamStates[blockIndex];
            states.queryWeight = hostBlock.queryWeightState;
            states.keyWeight = hostBlock.keyWeightState;
            states.valueWeight = hostBlock.valueWeightState;
            states.attentionOutputWeight = hostBlock.attentionOutputWeightState;
            states.attentionNormGamma = hostBlock.attentionNormGammaState;
            states.feedForwardNormGamma = hostBlock.feedForwardNormGammaState;
            states.feedForwardGateWeight = hostBlock.feedForwardGateWeightState;
            states.feedForwardGateBias = hostBlock.feedForwardGateBiasState;
            states.feedForwardUpWeight = hostBlock.feedForwardUpWeightState;
            states.feedForwardUpBias = hostBlock.feedForwardUpBiasState;
            states.feedForwardDownWeight = hostBlock.feedForwardDownWeightState;
            states.feedForwardDownBias = hostBlock.feedForwardDownBiasState;
        }
        return;
    }

    this->tokenEmbeddingState.uploadFrom(host.tokenEmbeddingState);
    this->finalNormGammaState.uploadFrom(host.finalNormGammaState);
    if (!this->tieEmbeddingProjection)
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

    // One H2D for [inputs | targets | meanDivisors], then cheap D2D splits.
    this->packH2dHost.resize(3 * tokenCount);
    std::copy(this->packedInputTokenIds.begin(), this->packedInputTokenIds.end(), this->packH2dHost.begin());
    std::copy(this->packedTargetTokenIds.begin(), this->packedTargetTokenIds.end(), this->packH2dHost.begin() + tokenCount);
    std::copy(this->packedMeanDivisors.begin(), this->packedMeanDivisors.end(), this->packH2dHost.begin() + 2 * tokenCount);

    this->packH2dDevice.ensureCapacity(3 * tokenCount);
    this->packH2dDevice.copyFromHost(this->packH2dHost.data(), 3 * tokenCount);

    this->tokenIdsBuffer.ensureCapacity(tokenCount);
    this->targetTokenIdsBuffer.ensureCapacity(tokenCount);
    this->meanDivisorBuffer.ensureCapacity(tokenCount);
    const size_t bytes = tokenCount * sizeof(int);
    CudaMatmul::throwIfCudaFailed(
        cudaMemcpy(this->tokenIdsBuffer.deviceData, this->packH2dDevice.deviceData, bytes, cudaMemcpyDeviceToDevice),
        "CudaLanguageModel::flushPackedHostBuffers D2D inputs");
    CudaMatmul::throwIfCudaFailed(
        cudaMemcpy(this->targetTokenIdsBuffer.deviceData, this->packH2dDevice.deviceData + tokenCount, bytes, cudaMemcpyDeviceToDevice),
        "CudaLanguageModel::flushPackedHostBuffers D2D targets");
    CudaMatmul::throwIfCudaFailed(
        cudaMemcpy(this->meanDivisorBuffer.deviceData, this->packH2dDevice.deviceData + 2 * tokenCount, bytes, cudaMemcpyDeviceToDevice),
        "CudaLanguageModel::flushPackedHostBuffers D2D meanDivisors");

    this->adamWindowTokenIds.insert(
        this->adamWindowTokenIds.end(),
        this->packedInputTokenIds.begin(),
        this->packedInputTokenIds.end());

    this->forwardTrunkFromDevice(tokenCount, segmentLength);

    if (this->epochLossSum.rows != 1 || this->epochLossSum.cols != 1)
        this->epochLossSum.ensureSize(1, 1);
    this->accumulateChunkedProjection(tokenCount, segmentLength, exampleCount, gradients);

    this->finalNorm.backward(this->hiddenGradient, this->normInputGradientScratch, this->finalNormGammaGradient);
    CudaOps::addInPlace(gradients.finalNormGamma, this->finalNormGammaGradient);
    std::swap(this->hiddenGradient, this->normInputGradientScratch);

    for (int blockIndex = static_cast<int>(this->blocks.size()) - 1; blockIndex >= 0; --blockIndex) {
        if (this->activationCheckpointing) {
            if (this->useHalfActivationCheckpoints()) {
                CudaAmp::castToFloat(
                    this->blockInputCheckpointsHalf[static_cast<size_t>(blockIndex)],
                    this->checkpointRestoreScratch);
                this->checkpointRestoreScratch.cols = this->hiddenGradient.cols;
                this->blocks[static_cast<size_t>(blockIndex)].forward(
                    this->checkpointRestoreScratch,
                    this->normalized,
                    segmentLength);
            } else {
                this->blocks[static_cast<size_t>(blockIndex)].forward(
                    this->blockInputCheckpoints[static_cast<size_t>(blockIndex)],
                    this->normalized,
                    segmentLength);
            }
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
    this->forwardTrunkFromDevice(tokenIds.size(), segmentLength);
}

void CudaLanguageModel::forwardTrunkFromDevice(size_t tokenCount, int segmentLength) {
    if (tokenCount == 0) throw std::invalid_argument("CudaLanguageModel::forwardTrunkFromDevice empty tokenCount");
    if (this->tokenEmbeddingWeight.empty()) throw std::logic_error("CudaLanguageModel::forwardTrunkFromDevice weights not uploaded");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaLanguageModel::forwardTrunkFromDevice no CUDA device");
    if (this->tokenIdsBuffer.deviceData == nullptr || tokenCount > this->tokenIdsBuffer.capacityCount)
        throw std::invalid_argument("CudaLanguageModel::forwardTrunkFromDevice tokenIdsBuffer not ready");

    if (segmentLength > 0) {
        if (static_cast<int>(tokenCount) % segmentLength != 0)
            throw std::invalid_argument("CudaLanguageModel::forwardTrunkFromDevice tokenCount not divisible by segmentLength");
        if (segmentLength > this->maximumPositionCount)
            throw std::invalid_argument("CudaLanguageModel::forwardTrunkFromDevice segmentLength exceeds maximumPositionCount");
    } else if (static_cast<int>(tokenCount) > this->maximumPositionCount) {
        throw std::invalid_argument("CudaLanguageModel::forwardTrunkFromDevice sequence longer than maximumPositionCount");
    }

    CudaOps::embeddingGatherInto(this->tokenEmbeddingWeight, this->tokenIdsBuffer, tokenCount, this->hidden);

    if (this->activationCheckpointing) {
        if (this->useHalfActivationCheckpoints()) {
            if (this->blockInputCheckpointsHalf.size() != this->blocks.size())
                this->blockInputCheckpointsHalf.resize(this->blocks.size());
            for (CudaHalfMatrix& checkpoint : this->blockInputCheckpointsHalf)
                checkpoint.ensureSize(this->tokenEmbeddingWeight.cols, tokenCount);
            this->checkpointRestoreScratch.ensureSize(this->tokenEmbeddingWeight.cols, tokenCount);
        } else {
            if (this->blockInputCheckpoints.size() != this->blocks.size())
                this->blockInputCheckpoints.resize(this->blocks.size());
            for (CudaMatrix& checkpoint : this->blockInputCheckpoints)
                checkpoint.ensureSize(this->tokenEmbeddingWeight.cols, tokenCount);
        }
    }

    for (size_t blockIndex = 0; blockIndex < this->blocks.size(); ++blockIndex) {
        if (this->activationCheckpointing) {
            if (this->useHalfActivationCheckpoints())
                CudaAmp::castToHalfSaturated(this->hidden, this->blockInputCheckpointsHalf[blockIndex]);
            else
                CudaOps::copyInto(this->hidden, this->blockInputCheckpoints[blockIndex]);
        }
        this->blocks[blockIndex].forward(this->hidden, this->normalized, segmentLength);
        CudaMatrix swapBuffer = std::move(this->hidden);
        this->hidden = std::move(this->normalized);
        this->normalized = std::move(swapBuffer);
    }

    this->finalNorm.forward(this->hidden, this->normalized);
}

void CudaLanguageModel::zeroTrainGradientsAfterAdam() {
    this->trainGradients.zeroInPlaceExceptEmbedding();
    if (this->adamWindowTokenIds.empty()) {
        CudaOps::zeroInPlace(this->trainGradients.tokenEmbedding);
        return;
    }

    const size_t vocabularySize = this->trainGradients.tokenEmbedding.rows;
    std::sort(this->adamWindowTokenIds.begin(), this->adamWindowTokenIds.end());
    this->adamWindowTokenIds.erase(
        std::unique(this->adamWindowTokenIds.begin(), this->adamWindowTokenIds.end()),
        this->adamWindowTokenIds.end());

    // Dense memset is cheaper once unique rows approach vocab size.
    if (this->adamWindowTokenIds.size() * 2 >= vocabularySize) {
        CudaOps::zeroInPlace(this->trainGradients.tokenEmbedding);
        this->adamWindowTokenIds.clear();
        return;
    }

    this->adamWindowTokenIdsBuffer.ensureCapacity(this->adamWindowTokenIds.size());
    this->adamWindowTokenIdsBuffer.copyFromHost(this->adamWindowTokenIds.data(), this->adamWindowTokenIds.size());
    CudaOps::embeddingZeroRows(
        this->trainGradients.tokenEmbedding,
        this->adamWindowTokenIdsBuffer.deviceData,
        this->adamWindowTokenIds.size());
    this->adamWindowTokenIds.clear();
}

void CudaLanguageModel::accumulateChunkedProjection(size_t tokenCount, int segmentLength, int exampleCountInPack, CudaLanguageModelGradients& gradients) {
    if (tokenCount == 0) throw std::invalid_argument("CudaLanguageModel::accumulateChunkedProjection empty tokenCount");
    if (segmentLength <= 0) throw std::invalid_argument("CudaLanguageModel::accumulateChunkedProjection segmentLength must be > 0");
    if (exampleCountInPack <= 0) throw std::invalid_argument("CudaLanguageModel::accumulateChunkedProjection exampleCountInPack must be > 0");
    if (static_cast<size_t>(segmentLength) * static_cast<size_t>(exampleCountInPack) != tokenCount)
        throw std::invalid_argument("CudaLanguageModel::accumulateChunkedProjection tokenCount mismatch");

    const CudaMatrix& headWeight = this->lmHeadWeight();
    if (headWeight.empty()) throw std::logic_error("CudaLanguageModel::accumulateChunkedProjection missing LM head weight");
    if (this->projectionBias.empty()) throw std::logic_error("CudaLanguageModel::accumulateChunkedProjection missing projection bias");

    CudaMatrix& headWeightGradient = this->tieEmbeddingProjection
        ? gradients.tokenEmbedding
        : gradients.projectionWeight;
    if (headWeightGradient.empty())
        throw std::logic_error("CudaLanguageModel::accumulateChunkedProjection missing LM head gradient");

    const int vocabularySize = static_cast<int>(headWeight.rows);
    const int embeddingDim = static_cast<int>(headWeight.cols);
    const float gradScale = CudaAmp::lossScalingActive() ? CudaAmp::lossScaler.scale : 1.0f;

    this->targetLogits.ensureSize(1, tokenCount);
    this->hiddenGradient.ensureSize(static_cast<size_t>(embeddingDim), tokenCount);
    this->hiddenGradientChunk.ensureSize(static_cast<size_t>(embeddingDim), tokenCount);
    CudaOps::onlineSoftmaxReset(this->onlineSoftmaxMax, this->onlineSoftmaxSumExp, tokenCount);
    CudaOps::zeroInPlace(this->targetLogits);
    CudaOps::zeroInPlace(this->hiddenGradient);

    // Cache full logits when they fit: one LM-head GEMM instead of project-twice-per-chunk.
    constexpr size_t maxCachedLogitsBytes = 512ull * 1024ull * 1024ull;
    const size_t logitsBytes = static_cast<size_t>(vocabularySize) * tokenCount * sizeof(float);
    const bool cacheFullLogits = logitsBytes <= maxCachedLogitsBytes;

    auto projectChunk = [&](int rowStart, int chunkRows, float* destination) {
        const float* weightRows = headWeight.buffer.deviceData + static_cast<size_t>(rowStart) * static_cast<size_t>(embeddingDim);
        // LM-head stays FP32 even when amp is on — CE is numerically fragile in FP16
        const bool previousAmp = CudaAmp::preferMixedPrecision;
        CudaAmp::preferMixedPrecision = false;
        CudaMatrix::multiplyPointersInto(
            weightRows, static_cast<size_t>(chunkRows), static_cast<size_t>(embeddingDim),
            this->normalized.buffer.deviceData, static_cast<size_t>(embeddingDim), tokenCount,
            destination,
            false, false);
        CudaAmp::preferMixedPrecision = previousAmp;
    };

    if (cacheFullLogits) {
        this->logits.ensureSize(static_cast<size_t>(vocabularySize), tokenCount);
        const bool previousAmp = CudaAmp::preferMixedPrecision;
        CudaAmp::preferMixedPrecision = false;
        CudaMatrix::multiplyInto(headWeight, this->normalized, this->logits);
        CudaAmp::preferMixedPrecision = previousAmp;
        CudaOps::broadcastBiasAddInPlace(this->logits, this->projectionBias);

        const int chunkCap = (std::min)(this->logitChunkRows, vocabularySize);
        for (int rowStart = 0; rowStart < vocabularySize; rowStart += chunkCap) {
            const int chunkRows = (std::min)(chunkCap, vocabularySize - rowStart);
            const float* chunkPtr = this->logits.buffer.deviceData + static_cast<size_t>(rowStart) * tokenCount;
            CudaOps::onlineSoftmaxUpdateFromChunk(chunkPtr, chunkRows, tokenCount, this->onlineSoftmaxMax, this->onlineSoftmaxSumExp);
            CudaOps::captureTargetLogitFromChunk(chunkPtr, this->targetTokenIdsBuffer, rowStart, chunkRows, tokenCount, this->targetLogits);
        }

        CudaOps::onlineSoftmaxAddMeanCrossEntropy(
            this->targetLogits, this->onlineSoftmaxMax, this->onlineSoftmaxSumExp, tokenCount,
            this->epochLossSum, 1.0f, segmentLength,
            &this->targetTokenIdsBuffer, CudaLanguageModel::padTargetId, &this->meanDivisorBuffer);

        for (int rowStart = 0; rowStart < vocabularySize; rowStart += chunkCap) {
            const int chunkRows = (std::min)(chunkCap, vocabularySize - rowStart);
            const float* chunkPtr = this->logits.buffer.deviceData + static_cast<size_t>(rowStart) * tokenCount;
            CudaOps::onlineSoftmaxLogitGradientChunkInto(
                chunkPtr, this->targetTokenIdsBuffer, rowStart, chunkRows, tokenCount,
                this->onlineSoftmaxMax, this->onlineSoftmaxSumExp, this->logitGradientChunk, gradScale, segmentLength,
                CudaLanguageModel::padTargetId, &this->meanDivisorBuffer);

            this->projectionWeightGradientChunk.ensureSize(static_cast<size_t>(chunkRows), static_cast<size_t>(embeddingDim));
            const bool previousAmpGrad = CudaAmp::preferMixedPrecision;
            CudaAmp::preferMixedPrecision = false;
            CudaMatrix::multiplyInto(this->logitGradientChunk, this->normalized, this->projectionWeightGradientChunk, false, true);
            CudaOps::addRowsInPlace(headWeightGradient, rowStart, this->projectionWeightGradientChunk);
            CudaOps::sumColumnsAddIntoRows(this->logitGradientChunk, gradients.projectionBias, rowStart);

            const float* weightRows = headWeight.buffer.deviceData + static_cast<size_t>(rowStart) * static_cast<size_t>(embeddingDim);
            CudaMatrix::multiplyPointersInto(
                weightRows, static_cast<size_t>(chunkRows), static_cast<size_t>(embeddingDim),
                this->logitGradientChunk.buffer.deviceData, static_cast<size_t>(chunkRows), tokenCount,
                this->hiddenGradientChunk.buffer.deviceData,
                true, false);
            CudaAmp::preferMixedPrecision = previousAmpGrad;
            this->hiddenGradientChunk.rows = static_cast<size_t>(embeddingDim);
            this->hiddenGradientChunk.cols = tokenCount;
            CudaOps::addInPlace(this->hiddenGradient, this->hiddenGradientChunk);
        }
        return;
    }

    const int chunkCap = (std::min)(this->logitChunkRows, vocabularySize);
    for (int rowStart = 0; rowStart < vocabularySize; rowStart += chunkCap) {
        const int chunkRows = (std::min)(chunkCap, vocabularySize - rowStart);
        this->logitChunk.ensureSize(static_cast<size_t>(chunkRows), tokenCount);
        projectChunk(rowStart, chunkRows, this->logitChunk.buffer.deviceData);
        CudaOps::broadcastBiasRowsAddInPlace(this->logitChunk, this->projectionBias, rowStart, chunkRows);
        CudaOps::onlineSoftmaxUpdateFromChunk(this->logitChunk, chunkRows, tokenCount, this->onlineSoftmaxMax, this->onlineSoftmaxSumExp);
        CudaOps::captureTargetLogitFromChunk(this->logitChunk, this->targetTokenIdsBuffer, rowStart, chunkRows, tokenCount, this->targetLogits);
    }

    CudaOps::onlineSoftmaxAddMeanCrossEntropy(
        this->targetLogits, this->onlineSoftmaxMax, this->onlineSoftmaxSumExp, tokenCount,
        this->epochLossSum, 1.0f, segmentLength,
        &this->targetTokenIdsBuffer, CudaLanguageModel::padTargetId, &this->meanDivisorBuffer);

    for (int rowStart = 0; rowStart < vocabularySize; rowStart += chunkCap) {
        const int chunkRows = (std::min)(chunkCap, vocabularySize - rowStart);
        this->logitChunk.ensureSize(static_cast<size_t>(chunkRows), tokenCount);
        projectChunk(rowStart, chunkRows, this->logitChunk.buffer.deviceData);
        CudaOps::broadcastBiasRowsAddInPlace(this->logitChunk, this->projectionBias, rowStart, chunkRows);
        CudaOps::onlineSoftmaxLogitGradientChunkInto(
            this->logitChunk, this->targetTokenIdsBuffer, rowStart, chunkRows, tokenCount,
            this->onlineSoftmaxMax, this->onlineSoftmaxSumExp, this->logitGradientChunk, gradScale, segmentLength,
            CudaLanguageModel::padTargetId, &this->meanDivisorBuffer);

        this->projectionWeightGradientChunk.ensureSize(static_cast<size_t>(chunkRows), static_cast<size_t>(embeddingDim));
        const bool previousAmp = CudaAmp::preferMixedPrecision;
        CudaAmp::preferMixedPrecision = false;
        CudaMatrix::multiplyInto(this->logitGradientChunk, this->normalized, this->projectionWeightGradientChunk, false, true);
        CudaOps::addRowsInPlace(headWeightGradient, rowStart, this->projectionWeightGradientChunk);
        CudaOps::sumColumnsAddIntoRows(this->logitGradientChunk, gradients.projectionBias, rowStart);

        const float* weightRows = headWeight.buffer.deviceData + static_cast<size_t>(rowStart) * static_cast<size_t>(embeddingDim);
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

    if (CudaAdam::preferCpuOffload) {
        if (this->hostBlockAdamStates.size() != this->blocks.size())
            this->hostBlockAdamStates.resize(this->blocks.size());

        std::vector<CudaAdamCpuOffloadItem> offloadItems;
        offloadItems.reserve(16 + this->blocks.size() * 12);

        auto pushOffload = [&offloadItems](CudaMatrix& parameter, AdamState& state, CudaMatrix& gradient) {
            if (parameter.elementCount() == 0) throw std::invalid_argument("CudaLanguageModel::applyGradients empty parameter");
            if (gradient.elementCount() != parameter.elementCount())
                throw std::invalid_argument("CudaLanguageModel::applyGradients gradient/parameter size mismatch");
            CudaAdamCpuOffloadItem item;
            item.parameter = &parameter;
            item.gradient = &gradient;
            item.hostState = &state;
            offloadItems.push_back(item);
        };

        if (!this->tieEmbeddingProjection)
            pushOffload(this->projectionWeight, this->hostProjectionWeightState, gradients.projectionWeight);
        pushOffload(this->projectionBias, this->hostProjectionBiasState, gradients.projectionBias);
        pushOffload(this->finalNorm.gamma, this->hostFinalNormGammaState, gradients.finalNormGamma);

        for (size_t blockIndex = 0; blockIndex < this->blocks.size(); ++blockIndex) {
            CudaTransformerBlock& block = this->blocks[blockIndex];
            CudaTransformerBlockGradients& blockGradients = gradients.blocks[blockIndex];
            CudaTransformerBlockHostAdamStates& blockStates = this->hostBlockAdamStates[blockIndex];

            pushOffload(block.attention.queryWeight, blockStates.queryWeight, blockGradients.queryWeight);
            pushOffload(block.attention.keyWeight, blockStates.keyWeight, blockGradients.keyWeight);
            pushOffload(block.attention.valueWeight, blockStates.valueWeight, blockGradients.valueWeight);
            pushOffload(block.attention.outputWeight, blockStates.attentionOutputWeight, blockGradients.attentionOutputWeight);
            pushOffload(block.attentionNorm.gamma, blockStates.attentionNormGamma, blockGradients.attentionNormGamma);
            pushOffload(block.feedForwardNorm.gamma, blockStates.feedForwardNormGamma, blockGradients.feedForwardNormGamma);
            pushOffload(block.feedForward.gateWeight, blockStates.feedForwardGateWeight, blockGradients.feedForwardGateWeight);
            pushOffload(block.feedForward.gateBias, blockStates.feedForwardGateBias, blockGradients.feedForwardGateBias);
            pushOffload(block.feedForward.upWeight, blockStates.feedForwardUpWeight, blockGradients.feedForwardUpWeight);
            pushOffload(block.feedForward.upBias, blockStates.feedForwardUpBias, blockGradients.feedForwardUpBias);
            pushOffload(block.feedForward.downWeight, blockStates.feedForwardDownWeight, blockGradients.feedForwardDownWeight);
            pushOffload(block.feedForward.downBias, blockStates.feedForwardDownBias, blockGradients.feedForwardDownBias);
        }

        pushOffload(this->tokenEmbeddingWeight, this->hostTokenEmbeddingState, gradients.tokenEmbedding);
        this->adam.updateCpuOffloadedMany(offloadItems.data(), static_cast<int>(offloadItems.size()), effectiveGradientScale);
        for (CudaTransformerBlock& block : this->blocks)
            block.feedForward.syncFusedGateUpWeight();
        CudaAmp::invalidateMasterWeightHalves();

        if (CudaAmp::lossScalingActive())
            CudaAmp::lossScaler.updateOnSuccess();
        return;
    }

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

    if (!this->tieEmbeddingProjection)
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
    for (CudaTransformerBlock& block : this->blocks)
        block.feedForward.syncFusedGateUpWeight();
    CudaAmp::invalidateMasterWeightHalves();

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
    if (exampleCount <= 0) {
        if (flushRemainder && accumulatedExampleCount > 0) {
            this->applyGradients(this->trainGradients, 1.0f / static_cast<float>(accumulatedExampleCount));
            this->zeroTrainGradientsAfterAdam();
            accumulatedExampleCount = 0;
            microbatchesSinceStep = 0;
        }
        return;
    }

    // Packing window is independent of Adam batchSize: wide enough to fill maxPackedColumns
    // at the smallest length bucket (lengthBucketStep).
    const int packWindow = (std::max)(
        batchSize,
        (std::max)(1, this->maxPackedColumns / CudaLanguageModel::lengthBucketStep)
    );
    const int examplesPerAdamStep = batchSize * this->gradientAccumulationSteps;

    std::vector<int> order(static_cast<size_t>(exampleCount));
    for (int index = 0; index < exampleCount; ++index)
        order[static_cast<size_t>(index)] = index;
    std::stable_sort(order.begin(), order.end(), [&dataset](int left, int right) {
        return dataset.examples[static_cast<size_t>(left)].inputTokenIds.size()
            < dataset.examples[static_cast<size_t>(right)].inputTokenIds.size();
    });

    std::vector<const LanguageModelExample*> packPointers;
    packPointers.reserve(static_cast<size_t>((std::max)(1, this->maxPackedColumns / CudaLanguageModel::lengthBucketStep)));

    for (int windowStart = 0; windowStart < exampleCount; windowStart += packWindow) {
        int windowEnd = windowStart + packWindow;
        if (windowEnd > exampleCount) windowEnd = exampleCount;

        int packStart = windowStart;
        while (packStart < windowEnd) {
            const int trueLength = static_cast<int>(
                dataset.examples[static_cast<size_t>(order[static_cast<size_t>(packStart)])].inputTokenIds.size()
            );
            const int bucketLength = CudaLanguageModel::lengthBucket(trueLength, this->maximumPositionCount);
            int maxExamplesInPack = 1;
            if (bucketLength > 0 && this->maxPackedColumns > 0)
                maxExamplesInPack = (std::max)(1, this->maxPackedColumns / bucketLength);

            packPointers.clear();
            int packEnd = packStart;
            while (packEnd < windowEnd
                && static_cast<int>(packPointers.size()) < maxExamplesInPack) {
                const int candidateLength = static_cast<int>(
                    dataset.examples[static_cast<size_t>(order[static_cast<size_t>(packEnd)])].inputTokenIds.size()
                );
                if (CudaLanguageModel::lengthBucket(candidateLength, this->maximumPositionCount) != bucketLength)
                    break;
                packPointers.push_back(&dataset.examples[static_cast<size_t>(order[static_cast<size_t>(packEnd)])]);
                ++packEnd;
            }

            const int packExampleCount = static_cast<int>(packPointers.size());
            ++packCount;
            if (packExampleCount <= 1) ++singleExamplePackCount;
            packedExampleSum += packExampleCount;
            packedTokenSum += static_cast<long long>(packExampleCount) * static_cast<long long>(bucketLength);

            this->accumulateBucketPackedExamples(packPointers.data(), packExampleCount, bucketLength, this->trainGradients);

            accumulatedExampleCount += packExampleCount;
            const bool isLastPack = flushRemainder && packEnd >= exampleCount;
            if (accumulatedExampleCount >= examplesPerAdamStep || isLastPack) {
                this->applyGradients(this->trainGradients, 1.0f / static_cast<float>(accumulatedExampleCount));
                this->zeroTrainGradientsAfterAdam();
                accumulatedExampleCount = 0;
                microbatchesSinceStep = 0;
            }

            packStart = packEnd;
        }
    }

    if (flushRemainder && accumulatedExampleCount > 0) {
        this->applyGradients(this->trainGradients, 1.0f / static_cast<float>(accumulatedExampleCount));
        this->zeroTrainGradientsAfterAdam();
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
        this->adamWindowTokenIds.clear();
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
        this->adamWindowTokenIds.clear();
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
            this->zeroTrainGradientsAfterAdam();
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
    Matrix deviceHeadWeightGrad = device.tieEmbeddingProjection
        ? deviceGradients.tokenEmbedding.download()
        : deviceGradients.projectionWeight.download();
    const Matrix& hostHeadWeightGrad = host.tieEmbeddingProjection
        ? hostGradients.tokenEmbedding
        : hostGradients.projectionWeight;
    float maximumDifference = 0.0f;
    for (size_t index = 0; index < hostHeadWeightGrad.data.size(); ++index)
        maximumDifference = (std::max)(maximumDifference, std::fabs(hostHeadWeightGrad.data[index] - deviceHeadWeightGrad.data[index]));

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
    Matrix packedHeadWeightGrad = device.tieEmbeddingProjection
        ? packedGradients.tokenEmbedding.download()
        : packedGradients.projectionWeight.download();
    Matrix sequentialHeadWeightGrad = device.tieEmbeddingProjection
        ? sequentialGradients.tokenEmbedding.download()
        : sequentialGradients.projectionWeight.download();
    float packedDifference = 0.0f;
    for (size_t index = 0; index < packedHeadWeightGrad.data.size(); ++index)
        packedDifference = (std::max)(packedDifference, std::fabs(packedHeadWeightGrad.data[index] - sequentialHeadWeightGrad.data[index]));

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

    Matrix fp32Weight = fp32Device.lmHeadWeight().download();
    Matrix int8Weight = int8Device.lmHeadWeight().download();
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

void CudaLanguageModel::runTrainCpuAdamOffloadSmokeDemo(int vocabularySize, int embeddingDim, int sequenceLength, int blockCount, int headCount) {
    if (!CudaMatmul::isAvailable()) {
        SmokeLog::skip("LanguageModel train CPU Adam offload");
        return;
    }
    if (vocabularySize <= 0 || embeddingDim <= 0 || sequenceLength <= 0 || blockCount <= 0 || headCount <= 0)
        throw std::invalid_argument("CudaLanguageModel::runTrainCpuAdamOffloadSmokeDemo invalid dims");

    const bool previousAmp = CudaAmp::preferMixedPrecision;
    const bool previousInt8 = CudaAdam::preferInt8Moments;
    const bool previousCpuOffload = CudaAdam::preferCpuOffload;
    CudaAmp::preferMixedPrecision = false;
    CudaAdam::preferInt8Moments = false;

    LanguageModel host(vocabularySize, embeddingDim, sequenceLength, Adam(0.001f), blockCount, headCount);

    const int packBatchSize = (std::max)(1, (std::min)(8, 8192 / sequenceLength));
    std::vector<LanguageModelExample> examples(static_cast<size_t>(packBatchSize));
    unsigned state = 307u;
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

    const int stepCount = 16;
    size_t totalBytes = 0;
    double gpuUsedMiB = 0.0;
    double cpuUsedMiB = 0.0;

    {
        CudaAdam::preferCpuOffload = false;
        CudaLanguageModel gpuDevice = CudaLanguageModel::createFrom(host);
        gpuDevice.adam = CudaAdam(host.optimizer.learningRate, host.optimizer.beta1, host.optimizer.beta2, host.optimizer.epsilon);
        size_t freeBefore = 0;
        CudaMatmul::throwIfCudaFailed(cudaMemGetInfo(&freeBefore, &totalBytes), "cpu Adam offload memGetInfo before gpu");
        gpuDevice.ensureTrainState();
        for (int step = 0; step < stepCount; ++step) {
            gpuDevice.trainGradients.zeroInPlace();
            gpuDevice.accumulatePackedExamples(packPointers.data(), packBatchSize, gpuDevice.trainGradients);
            gpuDevice.applyGradients(gpuDevice.trainGradients, 1.0f / static_cast<float>(packBatchSize));
        }
        CudaMatmul::throwIfCudaFailed(cudaDeviceSynchronize(), "cpu Adam offload gpu steps synchronize");
        size_t freeAfter = 0;
        CudaMatmul::throwIfCudaFailed(cudaMemGetInfo(&freeAfter, &totalBytes), "cpu Adam offload memGetInfo after gpu");
        gpuUsedMiB = static_cast<double>(freeBefore - freeAfter) / (1024.0 * 1024.0);
    }

    {
        CudaAdam::preferCpuOffload = true;
        CudaLanguageModel cpuDevice = CudaLanguageModel::createFrom(host);
        cpuDevice.adam = CudaAdam(host.optimizer.learningRate, host.optimizer.beta1, host.optimizer.beta2, host.optimizer.epsilon);
        size_t freeBefore = 0;
        CudaMatmul::throwIfCudaFailed(cudaMemGetInfo(&freeBefore, &totalBytes), "cpu Adam offload memGetInfo before cpu");
        cpuDevice.ensureTrainState();
        for (int step = 0; step < stepCount; ++step) {
            cpuDevice.trainGradients.zeroInPlace();
            cpuDevice.accumulatePackedExamples(packPointers.data(), packBatchSize, cpuDevice.trainGradients);
            cpuDevice.applyGradients(cpuDevice.trainGradients, 1.0f / static_cast<float>(packBatchSize));
        }
        CudaMatmul::throwIfCudaFailed(cudaDeviceSynchronize(), "cpu Adam offload cpu steps synchronize");
        size_t freeAfter = 0;
        CudaMatmul::throwIfCudaFailed(cudaMemGetInfo(&freeAfter, &totalBytes), "cpu Adam offload memGetInfo after cpu");
        cpuUsedMiB = static_cast<double>(freeBefore - freeAfter) / (1024.0 * 1024.0);
    }

    LanguageModel parityHost(vocabularySize, embeddingDim, sequenceLength, Adam(0.001f), blockCount, headCount);
    CudaLanguageModel parityGpu = CudaLanguageModel::createFrom(parityHost);
    CudaLanguageModel parityCpu = CudaLanguageModel::createFrom(parityHost);
    parityGpu.adam = CudaAdam(0.001f);
    parityCpu.adam = CudaAdam(0.001f);

    CudaAdam::preferCpuOffload = false;
    parityGpu.ensureTrainState();
    CudaAdam::preferCpuOffload = true;
    parityCpu.ensureTrainState();

    for (int step = 0; step < stepCount; ++step) {
        CudaAdam::preferCpuOffload = false;
        parityGpu.trainGradients.zeroInPlace();
        parityGpu.accumulatePackedExamples(packPointers.data(), packBatchSize, parityGpu.trainGradients);
        parityGpu.applyGradients(parityGpu.trainGradients, 1.0f / static_cast<float>(packBatchSize));

        CudaAdam::preferCpuOffload = true;
        parityCpu.trainGradients.zeroInPlace();
        parityCpu.accumulatePackedExamples(packPointers.data(), packBatchSize, parityCpu.trainGradients);
        parityCpu.applyGradients(parityCpu.trainGradients, 1.0f / static_cast<float>(packBatchSize));
    }
    CudaMatmul::throwIfCudaFailed(cudaDeviceSynchronize(), "cpu Adam offload parity synchronize");

    CudaAdam::preferCpuOffload = false;
    parityGpu.epochLossSum.ensureSize(1, 1);
    CudaOps::zeroInPlace(parityGpu.epochLossSum);
    parityGpu.accumulatePackedExamples(packPointers.data(), packBatchSize, parityGpu.trainGradients);
    const float gpuLoss = parityGpu.epochLossSum.download().at(0, 0) / static_cast<float>(packBatchSize);

    CudaAdam::preferCpuOffload = true;
    parityCpu.epochLossSum.ensureSize(1, 1);
    CudaOps::zeroInPlace(parityCpu.epochLossSum);
    parityCpu.accumulatePackedExamples(packPointers.data(), packBatchSize, parityCpu.trainGradients);
    const float cpuLoss = parityCpu.epochLossSum.download().at(0, 0) / static_cast<float>(packBatchSize);

    Matrix gpuWeight = parityGpu.lmHeadWeight().download();
    Matrix cpuWeight = parityCpu.lmHeadWeight().download();
    float weightDifference = 0.0f;
    bool anyNonFinite = false;
    for (size_t index = 0; index < gpuWeight.data.size(); ++index) {
        if (!std::isfinite(gpuWeight.data[index]) || !std::isfinite(cpuWeight.data[index]))
            anyNonFinite = true;
        weightDifference = (std::max)(weightDifference, std::fabs(gpuWeight.data[index] - cpuWeight.data[index]));
    }

    SmokeLog::result(
        "LanguageModel train CPU Adam offload",
        "vocab=%d embed=%d seq=%d steps=%d  loss gpu=%.4f cpu=%.4f  weightDiff=%.2e  nonFinite=%s  usedMiB gpu=%.1f cpu=%.1f saved=%.1f",
        vocabularySize, embeddingDim, sequenceLength, stepCount,
        gpuLoss, cpuLoss, weightDifference, anyNonFinite ? "yes" : "no",
        gpuUsedMiB, cpuUsedMiB, gpuUsedMiB - cpuUsedMiB);

    CudaAmp::preferMixedPrecision = previousAmp;
    CudaAdam::preferInt8Moments = previousInt8;
    CudaAdam::preferCpuOffload = previousCpuOffload;
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
            CudaMatrix::multiplyInto(device.lmHeadWeight(), device.normalized, device.logits);
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
            if (device.tieEmbeddingProjection)
                CudaOps::addInPlace(device.trainGradients.tokenEmbedding, device.projectionWeightGradient);
            else
                CudaOps::addInPlace(device.trainGradients.projectionWeight, device.projectionWeightGradient);
            CudaOps::sumColumnsInto(device.logitGradient, device.projectionBiasGradient);
            CudaOps::addInPlace(device.trainGradients.projectionBias, device.projectionBiasGradient);
            CudaMatrix::multiplyInto(device.lmHeadWeight(), device.logitGradient, device.hiddenGradient, true, false);
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

    this->tieEmbeddingProjection = host.tieEmbeddingProjection;
    this->tokenEmbeddingWeight.upload(host.tokenEmbedding.weight);
    this->blocks.clear();
    this->blocks.reserve(host.blocks.size());
    for (const TransformerBlock& block : host.blocks)
        this->blocks.push_back(CudaTransformerBlock::createFrom(block));

    this->finalNorm.uploadFrom(host.finalNorm);
    if (this->tieEmbeddingProjection)
        this->projectionWeight.free();
    else
        this->projectionWeight.upload(host.outputProjection.weight);
    this->projectionBias.upload(host.outputProjection.bias);
    this->maximumPositionCount = host.maximumPositionCount;
    this->kvCaches.clear();
    this->kvCaches.resize(this->blocks.size());
    this->trainStateReady = false;
    CudaAmp::registerMasterWeight(this->tokenEmbeddingWeight.buffer.deviceData, this->tokenEmbeddingWeight.elementCount());
    if (!this->tieEmbeddingProjection)
        CudaAmp::registerMasterWeight(this->projectionWeight.buffer.deviceData, this->projectionWeight.elementCount());
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
        if (this->useHalfActivationCheckpoints()) {
            if (this->blockInputCheckpointsHalf.size() != this->blocks.size())
                this->blockInputCheckpointsHalf.resize(this->blocks.size());
            for (CudaHalfMatrix& checkpoint : this->blockInputCheckpointsHalf)
                checkpoint.ensureSize(this->tokenEmbeddingWeight.cols, tokenIds.size());
        } else {
            if (this->blockInputCheckpoints.size() != this->blocks.size())
                this->blockInputCheckpoints.resize(this->blocks.size());
            for (CudaMatrix& checkpoint : this->blockInputCheckpoints)
                checkpoint.ensureSize(this->tokenEmbeddingWeight.cols, tokenIds.size());
        }
    }

    for (size_t blockIndex = 0; blockIndex < this->blocks.size(); ++blockIndex) {
        if (this->activationCheckpointing) {
            if (this->useHalfActivationCheckpoints())
                CudaAmp::castToHalfSaturated(this->hidden, this->blockInputCheckpointsHalf[blockIndex]);
            else
                CudaOps::copyInto(this->hidden, this->blockInputCheckpoints[blockIndex]);
        }
        this->blocks[blockIndex].forward(this->hidden, this->normalized, segmentLength);
        CudaMatrix swapBuffer = std::move(this->hidden);
        this->hidden = std::move(this->normalized);
        this->normalized = std::move(swapBuffer);
    }

    this->finalNorm.forward(this->hidden, this->normalized);
    CudaMatrix::multiplyInto(this->lmHeadWeight(), this->normalized, outLogits);
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
    CudaMatrix::multiplyInto(this->lmHeadWeight(), this->normalized, outLogits);
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
    CudaMatrix::multiplyInto(this->lmHeadWeight(), this->normalized, outLogits);
    CudaOps::broadcastBiasAddInPlace(outLogits, this->projectionBias);
}

void CudaLanguageModel::runConsumerVramDemo(
    int vocabularySize,
    int embeddingDim,
    int maximumPositionCount,
    int blockCount,
    int headCount,
    int exampleCount,
    int epochs,
    int batchSize,
    int gradientAccumulationSteps
) {
    if (!CudaMatmul::isAvailable()) {
        SmokeLog::skip("LanguageModel consumer VRAM");
        return;
    }
    if (vocabularySize <= 0 || embeddingDim <= 0 || maximumPositionCount <= 0
        || blockCount <= 0 || headCount <= 0 || exampleCount <= 0 || epochs <= 0)
        throw std::invalid_argument("CudaLanguageModel::runConsumerVramDemo invalid dims");
    if (embeddingDim % headCount != 0)
        throw std::invalid_argument("CudaLanguageModel::runConsumerVramDemo embeddingDim must divide headCount");

    SmokeLog::section("consumer VRAM");

    size_t freeBefore = 0;
    size_t totalBytes = 0;
    CudaMatmul::throwIfCudaFailed(cudaMemGetInfo(&freeBefore, &totalBytes), "consumer VRAM memGetInfo before");

    LanguageModel host(vocabularySize, embeddingDim, maximumPositionCount, Adam(0.001f), blockCount, headCount);
    host.enableCuda();
    host.enableCudaTrain();
    host.setCudaPreferFlashAttention(true);

    size_t freeAfterSetup = 0;
    CudaMatmul::throwIfCudaFailed(cudaMemGetInfo(&freeAfterSetup, &totalBytes), "consumer VRAM memGetInfo after setup");

    LanguageModelDataset trainDataset;
    trainDataset.vocabularySize = vocabularySize;
    trainDataset.examples.reserve(static_cast<size_t>(exampleCount));
    unsigned state = 42u;
    auto nextU32 = [&state]() -> unsigned {
        state = state * 1664525u + 1013904223u;
        return state;
    };
    for (int exampleIndex = 0; exampleIndex < exampleCount; ++exampleIndex) {
        const float u1 = (static_cast<float>(nextU32() % 10000u) + 0.5f) / 10000.0f;
        const float u2 = (static_cast<float>(nextU32() % 10000u) + 0.5f) / 10000.0f;
        const float radius = std::sqrt(-2.0f * std::log(u1));
        const float angle = 6.2831853f * u2;
        int length = static_cast<int>(std::lround(128.0f + 70.0f * radius * std::cos(angle)));
        if (length < 2) length = 2;
        if (length > maximumPositionCount) length = maximumPositionCount;

        std::vector<int> tokenIds(static_cast<size_t>(length + 1));
        for (int index = 0; index <= length; ++index)
            tokenIds[static_cast<size_t>(index)] = static_cast<int>(nextU32() % static_cast<unsigned>(vocabularySize));
        trainDataset.examples.push_back(LanguageModelDataset::fromTokenIds(tokenIds, vocabularySize, false));
    }

    LanguageModelDataset emptyTest;
    emptyTest.vocabularySize = vocabularySize;

    const int predictions = trainDataset.totalPredictionCount();
    SmokeLog::result(
        "consumer VRAM config",
        "vocab=%d embed=%d pos=%d blocks=%d heads=%d examples=%d preds=%d batch=%d accum=%d maxPackCols=%d lossScale=%s",
        vocabularySize,
        embeddingDim,
        maximumPositionCount,
        blockCount,
        headCount,
        exampleCount,
        predictions,
        batchSize,
        gradientAccumulationSteps,
        host.cudaMaxPackedColumns(),
        (CudaAmp::useLossScaling ? "on" : "off")
    );
    SmokeLog::result(
        "consumer VRAM mem",
        "total=%.0f MiB  freeBefore=%.0f  freeAfterSetup=%.0f  setupUsed=%.0f  maxPackCols=%d",
        static_cast<double>(totalBytes) / (1024.0 * 1024.0),
        static_cast<double>(freeBefore) / (1024.0 * 1024.0),
        static_cast<double>(freeAfterSetup) / (1024.0 * 1024.0),
        static_cast<double>(freeBefore - freeAfterSetup) / (1024.0 * 1024.0),
        host.cudaMaxPackedColumns()
    );

    host.train(trainDataset, emptyTest, epochs, 1, batchSize, gradientAccumulationSteps);

    size_t freeAfterTrain = 0;
    CudaMatmul::throwIfCudaFailed(cudaMemGetInfo(&freeAfterTrain, &totalBytes), "consumer VRAM memGetInfo after train");
    SmokeLog::result(
        "consumer VRAM after",
        "free=%.0f MiB  setupUsed=%.0f MiB",
        static_cast<double>(freeAfterTrain) / (1024.0 * 1024.0),
        static_cast<double>(freeBefore - freeAfterSetup) / (1024.0 * 1024.0)
    );
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
