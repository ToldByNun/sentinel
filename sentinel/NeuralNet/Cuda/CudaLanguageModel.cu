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
#include <vector>

#include "../Optimizers/Adam.hpp"

CudaLanguageModel::CudaLanguageModel()
    : maximumPositionCount(0), maxPackedColumns(1024), gradientAccumulationSteps(1), activationCheckpointing(false), adam(0.001f), trainStateReady(false) {}

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
    const LanguageModelExample* pointer = &example;
    return this->accumulatePackedExamples(&pointer, 1, gradients);
}

float CudaLanguageModel::accumulatePackedExamples(const LanguageModelExample* const* examples, int exampleCount, CudaLanguageModelGradients& gradients) {
    if (examples == nullptr || exampleCount <= 0) throw std::invalid_argument("CudaLanguageModel::accumulatePackedExamples empty examples");
    if (gradients.blocks.size() != this->blocks.size())
        throw std::invalid_argument("CudaLanguageModel::accumulatePackedExamples gradients not initialized");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaLanguageModel::accumulatePackedExamples no CUDA device");

    const size_t segmentLength = examples[0]->inputTokenIds.size();
    if (segmentLength == 0) throw std::invalid_argument("CudaLanguageModel::accumulatePackedExamples empty inputTokenIds");

    this->packedInputTokenIds.clear();
    this->packedTargetTokenIds.clear();
    this->packedInputTokenIds.reserve(segmentLength * static_cast<size_t>(exampleCount));
    this->packedTargetTokenIds.reserve(segmentLength * static_cast<size_t>(exampleCount));

    for (int exampleIndex = 0; exampleIndex < exampleCount; ++exampleIndex) {
        const LanguageModelExample& example = *examples[exampleIndex];
        if (example.inputTokenIds.size() != segmentLength)
            throw std::invalid_argument("CudaLanguageModel::accumulatePackedExamples unequal input lengths");
        if (example.targetTokenIds.size() != segmentLength)
            throw std::invalid_argument("CudaLanguageModel::accumulatePackedExamples target length mismatch");
        this->packedInputTokenIds.insert(this->packedInputTokenIds.end(), example.inputTokenIds.begin(), example.inputTokenIds.end());
        this->packedTargetTokenIds.insert(this->packedTargetTokenIds.end(), example.targetTokenIds.begin(), example.targetTokenIds.end());
    }

    const size_t tokenCount = this->packedInputTokenIds.size();
    const int meanDivisor = static_cast<int>(segmentLength);
    this->forwardInto(this->packedInputTokenIds, this->logits, meanDivisor);

    this->targetTokenIdsBuffer.ensureCapacity(tokenCount);
    this->targetTokenIdsBuffer.copyFromHost(this->packedTargetTokenIds.data(), tokenCount);

    if (this->epochLossSum.rows != 1 || this->epochLossSum.cols != 1)
        this->epochLossSum.ensureSize(1, 1);
    CudaOps::softmaxCrossEntropyFromLogitsInto(this->logits, this->targetTokenIdsBuffer, tokenCount, this->probabilities, this->logitGradient, this->epochLossSum, 1.0f, meanDivisor);

    CudaMatrix::multiplyInto(this->logitGradient, this->normalized, this->projectionWeightGradient, false, true);
    CudaOps::addInPlace(gradients.projectionWeight, this->projectionWeightGradient);
    CudaOps::sumColumnsInto(this->logitGradient, this->projectionBiasGradient);
    CudaOps::addInPlace(gradients.projectionBias, this->projectionBiasGradient);

    CudaMatrix::multiplyInto(this->projectionWeight, this->logitGradient, this->hiddenGradient, true, false);
    this->finalNorm.backward(this->hiddenGradient, this->normInputGradientScratch, this->finalNormGammaGradient);
    CudaOps::addInPlace(gradients.finalNormGamma, this->finalNormGammaGradient);
    std::swap(this->hiddenGradient, this->normInputGradientScratch);

    for (int blockIndex = static_cast<int>(this->blocks.size()) - 1; blockIndex >= 0; --blockIndex) {
        if (this->activationCheckpointing) {
            this->blocks[static_cast<size_t>(blockIndex)].forward(
                this->blockInputCheckpoints[static_cast<size_t>(blockIndex)],
                this->normalized,
                meanDivisor);
        }
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

    auto ensureMoments = [](CudaAdamState& state, const CudaMatrix& parameter) {
        if (!state.firstMoment.empty()) return;
        state.firstMoment.ensureSize(parameter.rows, parameter.cols);
        state.secondMoment.ensureSize(parameter.rows, parameter.cols);
        CudaOps::zeroInPlace(state.firstMoment);
        CudaOps::zeroInPlace(state.secondMoment);
    };

    std::vector<CudaAdamUpdateItem> items;
    items.reserve(16 + this->blocks.size() * 12);

    auto pushItem = [&items, &ensureMoments](CudaMatrix& parameter, CudaAdamState& state, const CudaMatrix& gradient) {
        ensureMoments(state, parameter);
        CudaAdamUpdateItem item;
        item.parameter = parameter.buffer.deviceData;
        item.firstMoment = state.firstMoment.buffer.deviceData;
        item.secondMoment = state.secondMoment.buffer.deviceData;
        item.gradient = gradient.buffer.deviceData;
        item.elementCount = static_cast<int>(parameter.elementCount());
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
    this->adam.updateMany(items.data(), static_cast<int>(items.size()));
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

void CudaLanguageModel::train(const LanguageModelDataset& trainDataset, const LanguageModelDataset& testDataset, int epochs, int logEveryEpochs, int batchSize, int gradientAccumulationSteps) {
    if (trainDataset.examples.empty()) return;
    if (logEveryEpochs <= 0) logEveryEpochs = 1;
    if (batchSize <= 0) batchSize = 32;
    if (gradientAccumulationSteps <= 0) gradientAccumulationSteps = 1;
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaLanguageModel::train no CUDA device");

    this->gradientAccumulationSteps = gradientAccumulationSteps;
    this->ensureTrainState();
    this->epochLossSum.ensureSize(1, 1);
    this->blockInputGradientScratch.ensureSize(this->tokenEmbeddingWeight.cols, 1);
    this->normInputGradientScratch.ensureSize(this->tokenEmbeddingWeight.cols, 1);

    const int exampleCount = static_cast<int>(trainDataset.examples.size());
    const int predictionCount = trainDataset.totalPredictionCount();
    std::vector<const LanguageModelExample*> packPointers;
    packPointers.reserve(static_cast<size_t>(batchSize));

    for (int epoch = 0; epoch < epochs; ++epoch) {
        CudaOps::zeroInPlace(this->epochLossSum);
        const auto epochStart = std::chrono::steady_clock::now();

        this->trainGradients.zeroInPlace();
        int accumulatedExampleCount = 0;
        int microbatchesSinceStep = 0;

        for (int batchStart = 0; batchStart < exampleCount; batchStart += batchSize) {
            int batchEnd = batchStart + batchSize;
            if (batchEnd > exampleCount) batchEnd = exampleCount;
            const int batchCount = batchEnd - batchStart;

            std::vector<int> batchIndices;
            batchIndices.reserve(static_cast<size_t>(batchCount));
            for (int index = batchStart; index < batchEnd; ++index)
                batchIndices.push_back(index);
            std::stable_sort(batchIndices.begin(), batchIndices.end(), [&trainDataset](int left, int right) {
                return trainDataset.examples[static_cast<size_t>(left)].inputTokenIds.size()
                    < trainDataset.examples[static_cast<size_t>(right)].inputTokenIds.size();
            });

            int packStart = 0;
            while (packStart < static_cast<int>(batchIndices.size())) {
                const size_t segmentLength = trainDataset.examples[static_cast<size_t>(batchIndices[static_cast<size_t>(packStart)])].inputTokenIds.size();
                int maxExamplesInPack = 1;
                if (segmentLength > 0 && this->maxPackedColumns > 0)
                    maxExamplesInPack = (std::max)(1, this->maxPackedColumns / static_cast<int>(segmentLength));

                packPointers.clear();
                int packEnd = packStart;
                while (packEnd < static_cast<int>(batchIndices.size())
                    && static_cast<int>(packPointers.size()) < maxExamplesInPack
                    && trainDataset.examples[static_cast<size_t>(batchIndices[static_cast<size_t>(packEnd)])].inputTokenIds.size() == segmentLength) {
                    packPointers.push_back(&trainDataset.examples[static_cast<size_t>(batchIndices[static_cast<size_t>(packEnd)])]);
                    ++packEnd;
                }
                this->accumulatePackedExamples(packPointers.data(), static_cast<int>(packPointers.size()), this->trainGradients);
                packStart = packEnd;
            }

            accumulatedExampleCount += batchCount;
            ++microbatchesSinceStep;

            const bool isLastBatch = batchEnd >= exampleCount;
            if (microbatchesSinceStep < this->gradientAccumulationSteps && !isLastBatch)
                continue;

            this->trainGradients.scaleInPlace(1.0f / static_cast<float>(accumulatedExampleCount));
            this->applyGradients(this->trainGradients);
            this->trainGradients.zeroInPlace();
            accumulatedExampleCount = 0;
            microbatchesSinceStep = 0;
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
        device.trainGradients.scaleInPlace(1.0f / static_cast<float>(packBatchSize));
        device.applyGradients(device.trainGradients);
    }
    CudaMatmul::throwIfCudaFailed(cudaDeviceSynchronize(), "CudaLanguageModel train smoke warmup synchronize");

    const auto trainStart = std::chrono::steady_clock::now();
    int totalTokens = 0;
    for (int step = 0; step < timedStepCount; ++step) {
        device.trainGradients.zeroInPlace();
        device.accumulatePackedExamples(packPointers.data(), packBatchSize, device.trainGradients);
        device.trainGradients.scaleInPlace(1.0f / static_cast<float>(packBatchSize));
        device.applyGradients(device.trainGradients);
        totalTokens += sequenceLength * packBatchSize;
    }
    CudaMatmul::throwIfCudaFailed(cudaDeviceSynchronize(), "CudaLanguageModel train smoke steps synchronize");
    const double trainSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - trainStart).count();
    const double tokensPerSecond = trainSeconds > 0.0 ? static_cast<double>(totalTokens) / trainSeconds : 0.0;

    SmokeLog::result("LanguageModel train", "vocab=%d embed=%d seq=%d pack=%d  loss cpu=%.4f gpu=%.4f  gradDiff=%.2e  packLossDiff=%.2e packGradDiff=%.2e  tokens/s=%.0f",
        vocabularySize, embeddingDim, sequenceLength, packBatchSize, hostLoss, deviceLoss, maximumDifference, std::fabs(packedLoss - sequentialLoss), packedDifference, tokensPerSecond);
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
    device.ensureTrainState();
    device.epochLossSum.ensureSize(1, 1);
    device.blockInputGradientScratch.ensureSize(static_cast<size_t>(embeddingDim), 1);
    device.normInputGradientScratch.ensureSize(static_cast<size_t>(embeddingDim), 1);

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
            device.trainGradients.scaleInPlace(1.0f / static_cast<float>(packBatchSize));
            device.applyGradients(device.trainGradients);
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
    SmokeLog::result("profile config", "vocab=%d embed=%d seq=%d blocks=%d pack=%d tokens/step=%d flash=%s maxPackCols=%d",
        vocabularySize, embeddingDim, sequenceLength, blockCount, packBatchSize, tokensPerStep, preferFlash ? "on" : "off", device.maxPackedColumns);
    SmokeLog::result("profile step", "avg=%.2fms  ~tokens/s=%.0f", stepTotal, tokensPerSecond);
    SmokeLog::result("  attention", "fwd=%.2fms bwd=%.2fms  total=%.2fms (%.0f%%)", sumAttn, sumAttnBwd, attnTotal, attnTotal * percent);
    SmokeLog::result("  ffn", "fwd=%.2fms bwd=%.2fms  total=%.2fms (%.0f%%)", sumFfn, sumFfnBwd, ffnTotal, ffnTotal * percent);
    SmokeLog::result("  head+ce", "fwd=%.2fms ce=%.2fms bwd=%.2fms  total=%.2fms (%.0f%%)", sumHeadFwd, sumCe, sumHeadBwd, headTotal, headTotal * percent);
    SmokeLog::result("  embed+h2d", "embed=%.2fms scatter=%.2fms h2d=%.2fms hostPack=%.2fms", sumEmbed, sumScatter, sumH2d, sumHostPack);
    SmokeLog::result("  adam", "%.2fms (%.0f%%)", sumAdam, sumAdam * percent);
    SmokeLog::result("  other sum", "%.2fms (%.0f%%)", otherTotal, otherTotal * percent);
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
