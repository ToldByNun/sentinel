#include "CudaLanguageModel.hpp"

#include "CudaOps.hpp"
#include "../Utils/SmokeLog.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <stdexcept>
#include <utility>

#include "../Optimizers/Adam.hpp"

CudaLanguageModel::CudaLanguageModel() : maximumPositionCount(0), adam(0.001f), trainStateReady(false) {}

void CudaTransformerBlockAdamStates::ensureFrom(const CudaTransformerBlock& block) {
    this->queryWeight = CudaAdamState::zerosLike(block.attention.queryWeight);
    this->keyWeight = CudaAdamState::zerosLike(block.attention.keyWeight);
    this->valueWeight = CudaAdamState::zerosLike(block.attention.valueWeight);
    this->attentionOutputWeight = CudaAdamState::zerosLike(block.attention.outputWeight);
    this->attentionNormGamma = CudaAdamState::zerosLike(block.attentionNorm.gamma);
    this->feedForwardNormGamma = CudaAdamState::zerosLike(block.feedForwardNorm.gamma);
    this->feedForwardGateWeight = CudaAdamState::zerosLike(block.feedForward.gateWeight);
    this->feedForwardGateBias = CudaAdamState::zerosLike(block.feedForward.gateBias);
    this->feedForwardUpWeight = CudaAdamState::zerosLike(block.feedForward.upWeight);
    this->feedForwardUpBias = CudaAdamState::zerosLike(block.feedForward.upBias);
    this->feedForwardDownWeight = CudaAdamState::zerosLike(block.feedForward.downWeight);
    this->feedForwardDownBias = CudaAdamState::zerosLike(block.feedForward.downBias);
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
    if (this->tokenEmbeddingWeight.empty()) throw std::logic_error("CudaLanguageModel::ensureTrainState weights not uploaded");

    this->trainGradients.ensureFrom(*this);
    this->trainGradients.zeroInPlace();

    this->tokenEmbeddingState = CudaAdamState::zerosLike(this->tokenEmbeddingWeight);
    this->finalNormGammaState = CudaAdamState::zerosLike(this->finalNorm.gamma);
    this->projectionWeightState = CudaAdamState::zerosLike(this->projectionWeight);
    this->projectionBiasState = CudaAdamState::zerosLike(this->projectionBias);

    this->blockAdamStates.resize(this->blocks.size());
    for (size_t blockIndex = 0; blockIndex < this->blocks.size(); ++blockIndex)
        this->blockAdamStates[blockIndex].ensureFrom(this->blocks[blockIndex]);

    this->projectionWeightGradient.ensureSize(this->projectionWeight.rows, this->projectionWeight.cols);
    this->projectionBiasGradient.ensureSize(this->projectionBias.rows, this->projectionBias.cols);
    this->finalNormGammaGradient.ensureSize(this->finalNorm.gamma.rows, this->finalNorm.gamma.cols);

    this->trainStateReady = true;
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

float CudaLanguageModel::accumulateExample(const LanguageModelExample& example, CudaLanguageModelGradients& gradients) {
    if (example.inputTokenIds.empty()) throw std::invalid_argument("CudaLanguageModel::accumulateExample empty inputTokenIds");
    if (example.targetTokenIds.size() != example.inputTokenIds.size())
        throw std::invalid_argument("CudaLanguageModel::accumulateExample target length mismatch");
    if (gradients.blocks.size() != this->blocks.size())
        throw std::invalid_argument("CudaLanguageModel::accumulateExample gradients not initialized");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaLanguageModel::accumulateExample no CUDA device");

    const size_t tokenCount = example.inputTokenIds.size();
    this->forwardInto(example.inputTokenIds, this->logits);
    CudaOps::softmaxInto(this->logits, this->probabilities);

    this->targetTokenIdsBuffer.ensureCapacity(tokenCount);
    this->targetTokenIdsBuffer.copyFromHost(example.targetTokenIds.data(), tokenCount);

    if (this->epochLossSum.rows != 1 || this->epochLossSum.cols != 1)
        this->epochLossSum.ensureSize(1, 1);
    CudaOps::crossEntropyAddMeanLossFromIds(this->probabilities, this->targetTokenIdsBuffer, tokenCount, this->epochLossSum);

    CudaOps::crossEntropyLogitGradientFromIdsInto(this->probabilities, this->targetTokenIdsBuffer, tokenCount, this->logitGradient);

    CudaMatrix::multiplyInto(this->logitGradient, this->normalized, this->projectionWeightGradient, false, true);
    CudaOps::addInPlace(gradients.projectionWeight, this->projectionWeightGradient);
    CudaOps::sumColumnsInto(this->logitGradient, this->projectionBiasGradient);
    CudaOps::addInPlace(gradients.projectionBias, this->projectionBiasGradient);

    CudaMatrix::multiplyInto(this->projectionWeight, this->logitGradient, this->hiddenGradient, true, false);
    this->finalNorm.backward(this->hiddenGradient, this->normInputGradientScratch, this->finalNormGammaGradient);
    CudaOps::addInPlace(gradients.finalNormGamma, this->finalNormGammaGradient);
    std::swap(this->hiddenGradient, this->normInputGradientScratch);

    for (int blockIndex = static_cast<int>(this->blocks.size()) - 1; blockIndex >= 0; --blockIndex) {
        this->blocks[static_cast<size_t>(blockIndex)].backward(this->hiddenGradient, this->blockInputGradientScratch, gradients.blocks[static_cast<size_t>(blockIndex)]);
        std::swap(this->hiddenGradient, this->blockInputGradientScratch);
    }

    CudaOps::embeddingScatterAddInto(gradients.tokenEmbedding, this->tokenIdsBuffer, tokenCount, this->hiddenGradient);
    return 0.0f;
}

void CudaLanguageModel::applyGradients(CudaLanguageModelGradients& gradients) {
    if (!this->trainStateReady) throw std::logic_error("CudaLanguageModel::applyGradients train state not ready");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaLanguageModel::applyGradients no CUDA device");

    this->adam.step();
    this->adam.update(this->projectionWeight, this->projectionWeightState, gradients.projectionWeight);
    this->adam.update(this->projectionBias, this->projectionBiasState, gradients.projectionBias);
    this->adam.update(this->finalNorm.gamma, this->finalNormGammaState, gradients.finalNormGamma);

    for (size_t blockIndex = 0; blockIndex < this->blocks.size(); ++blockIndex) {
        CudaTransformerBlock& block = this->blocks[blockIndex];
        CudaTransformerBlockGradients& blockGradients = gradients.blocks[blockIndex];
        CudaTransformerBlockAdamStates& blockStates = this->blockAdamStates[blockIndex];

        this->adam.update(block.attention.queryWeight, blockStates.queryWeight, blockGradients.queryWeight);
        this->adam.update(block.attention.keyWeight, blockStates.keyWeight, blockGradients.keyWeight);
        this->adam.update(block.attention.valueWeight, blockStates.valueWeight, blockGradients.valueWeight);
        this->adam.update(block.attention.outputWeight, blockStates.attentionOutputWeight, blockGradients.attentionOutputWeight);
        this->adam.update(block.attentionNorm.gamma, blockStates.attentionNormGamma, blockGradients.attentionNormGamma);
        this->adam.update(block.feedForwardNorm.gamma, blockStates.feedForwardNormGamma, blockGradients.feedForwardNormGamma);
        this->adam.update(block.feedForward.gateWeight, blockStates.feedForwardGateWeight, blockGradients.feedForwardGateWeight);
        this->adam.update(block.feedForward.gateBias, blockStates.feedForwardGateBias, blockGradients.feedForwardGateBias);
        this->adam.update(block.feedForward.upWeight, blockStates.feedForwardUpWeight, blockGradients.feedForwardUpWeight);
        this->adam.update(block.feedForward.upBias, blockStates.feedForwardUpBias, blockGradients.feedForwardUpBias);
        this->adam.update(block.feedForward.downWeight, blockStates.feedForwardDownWeight, blockGradients.feedForwardDownWeight);
        this->adam.update(block.feedForward.downBias, blockStates.feedForwardDownBias, blockGradients.feedForwardDownBias);
    }

    this->adam.update(this->tokenEmbeddingWeight, this->tokenEmbeddingState, gradients.tokenEmbedding);
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

void CudaLanguageModel::train(const LanguageModelDataset& trainDataset, const LanguageModelDataset& testDataset, int epochs, int logEveryEpochs, int batchSize) {
    if (trainDataset.examples.empty()) return;
    if (logEveryEpochs <= 0) logEveryEpochs = 1;
    if (batchSize <= 0) batchSize = 32;
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaLanguageModel::train no CUDA device");

    this->ensureTrainState();
    this->epochLossSum.ensureSize(1, 1);
    this->blockInputGradientScratch.ensureSize(this->tokenEmbeddingWeight.cols, 1);
    this->normInputGradientScratch.ensureSize(this->tokenEmbeddingWeight.cols, 1);

    const int exampleCount = static_cast<int>(trainDataset.examples.size());
    const int predictionCount = trainDataset.totalPredictionCount();

    for (int epoch = 0; epoch < epochs; ++epoch) {
        CudaOps::zeroInPlace(this->epochLossSum);
        const auto epochStart = std::chrono::steady_clock::now();

        for (int batchStart = 0; batchStart < exampleCount; batchStart += batchSize) {
            int batchEnd = batchStart + batchSize;
            if (batchEnd > exampleCount) batchEnd = exampleCount;
            const float batchCount = static_cast<float>(batchEnd - batchStart);

            this->trainGradients.zeroInPlace();

            for (int index = batchStart; index < batchEnd; ++index)
                this->accumulateExample(trainDataset.examples[static_cast<size_t>(index)], this->trainGradients);

            this->trainGradients.scaleInPlace(1.0f / batchCount);
            this->applyGradients(this->trainGradients);
        }

        CudaMatmul::throwIfCudaFailed(cudaDeviceSynchronize(), "CudaLanguageModel::train epoch synchronize");
        const auto epochEnd = std::chrono::steady_clock::now();
        const double epochSeconds = std::chrono::duration<double>(epochEnd - epochStart).count();

        if (epoch % logEveryEpochs != 0) continue;

        Matrix lossHost = this->epochLossSum.download();
        const float averageTrainLoss = lossHost.at(0, 0) / static_cast<float>(trainDataset.size());
        const double tokensPerSecond = epochSeconds > 0.0 ? static_cast<double>(predictionCount) / epochSeconds : 0.0;
        std::printf("  Epoch %-3d  trainLoss=%.6f  sec=%.2f  tokens/s=%.0f  backend=cuda", epoch, averageTrainLoss, epochSeconds, tokensPerSecond);

        if (!testDataset.examples.empty()) {
            const float testLoss = this->averageLoss(testDataset);
            std::printf("  testLoss=%.6f", testLoss);
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
    device.ensureTrainState();

    std::vector<int> inputTokenIds(static_cast<size_t>(sequenceLength), 0);
    std::vector<int> targetTokenIds(static_cast<size_t>(sequenceLength), 0);
    unsigned state = 103u;
    for (size_t index = 0; index < inputTokenIds.size(); ++index) {
        state = state * 1664525u + 1013904223u;
        inputTokenIds[index] = static_cast<int>(state % static_cast<unsigned>(vocabularySize));
        state = state * 1664525u + 1013904223u;
        targetTokenIds[index] = static_cast<int>(state % static_cast<unsigned>(vocabularySize));
    }

    LanguageModelExample example;
    example.inputTokenIds = inputTokenIds;
    example.targetTokenIds = targetTokenIds;

    LanguageModelGradients hostGradients = LanguageModelGradients::zerosFrom(host);
    LanguageModelCache hostCache;
    const float hostLoss = host.accumulateExample(example, hostGradients, hostCache);

    CudaLanguageModelGradients deviceGradients;
    deviceGradients.ensureFrom(device);
    deviceGradients.zeroInPlace();
    device.epochLossSum.ensureSize(1, 1);
    CudaOps::zeroInPlace(device.epochLossSum);
    device.accumulateExample(example, deviceGradients);
    CudaMatmul::throwIfCudaFailed(cudaDeviceSynchronize(), "CudaLanguageModel train smoke synchronize");
    const float deviceLoss = device.epochLossSum.download().at(0, 0);
    Matrix deviceProjectionWeightGrad = deviceGradients.projectionWeight.download();
    float maximumDifference = 0.0f;
    for (size_t index = 0; index < hostGradients.projectionWeight.data.size(); ++index)
        maximumDifference = (std::max)(maximumDifference, std::fabs(hostGradients.projectionWeight.data[index] - deviceProjectionWeightGrad.data[index]));

    const int stepCount = 5;
    const auto trainStart = std::chrono::steady_clock::now();
    int totalTokens = 0;
    for (int step = 0; step < stepCount; ++step) {
        device.trainGradients.zeroInPlace();
        const float stepLoss = device.accumulateExample(example, device.trainGradients);
        device.applyGradients(device.trainGradients);
        totalTokens += sequenceLength;
        (void)stepLoss;
    }
    CudaMatmul::throwIfCudaFailed(cudaDeviceSynchronize(), "CudaLanguageModel train smoke steps synchronize");
    const double trainSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - trainStart).count();
    const double tokensPerSecond = trainSeconds > 0.0 ? static_cast<double>(totalTokens) / trainSeconds : 0.0;

    SmokeLog::result("LanguageModel train", "vocab=%d embed=%d seq=%d  loss cpu=%.4f gpu=%.4f  gradDiff=%.2e  tokens/s=%.0f",
        vocabularySize, embeddingDim, sequenceLength, hostLoss, deviceLoss, maximumDifference, tokensPerSecond);
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

void CudaLanguageModel::forwardInto(const std::vector<int>& tokenIds, CudaMatrix& outLogits) {
    if (tokenIds.empty()) throw std::invalid_argument("CudaLanguageModel::forwardInto empty tokenIds");
    if (this->tokenEmbeddingWeight.empty()) throw std::logic_error("CudaLanguageModel::forwardInto weights not uploaded");
    if (static_cast<int>(tokenIds.size()) > this->maximumPositionCount)
        throw std::invalid_argument("CudaLanguageModel::forwardInto sequence longer than maximumPositionCount");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaLanguageModel::forwardInto no CUDA device");

    this->tokenIdsBuffer.ensureCapacity(tokenIds.size());
    this->tokenIdsBuffer.copyFromHost(tokenIds.data(), tokenIds.size());

    CudaOps::embeddingGatherInto(this->tokenEmbeddingWeight, this->tokenIdsBuffer, tokenIds.size(), this->hidden);

    for (size_t blockIndex = 0; blockIndex < this->blocks.size(); ++blockIndex) {
        this->blocks[blockIndex].forward(this->hidden, this->normalized);
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
