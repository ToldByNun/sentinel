#include "LanguageModel.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

#include "../Activations/Softmax.hpp"
#include "../Cuda/CudaLanguageModel.hpp"
#include "../Cuda/CudaAmp.hpp"
#include "../Cuda/CudaAdam.hpp"
#include "../Cuda/CudaMatmul.hpp"
#include "../Initializers/UniformInit.hpp"
#include "../Losses/CrossEntropy.hpp"
#include "../Tokenizer/BPETokenizer.hpp"
#include "../Utils/SmokeLog.hpp"

#if defined(_OPENMP)
#include <omp.h>
#endif

LanguageModelGradients LanguageModelGradients::zerosFrom(const LanguageModel& model) {
    LanguageModelGradients gradients;
    gradients.tokenEmbedding = Matrix::zerosLike(model.tokenEmbedding.weight);
    gradients.blocks.reserve(model.blocks.size());
    for (const TransformerBlock& block : model.blocks)
        gradients.blocks.push_back(TransformerBlockGradients::zerosFrom(block));
    gradients.finalNormGamma = Matrix::zerosLike(model.finalNorm.gamma);
    gradients.projectionWeight = Matrix::zerosLike(model.outputProjection.weight);
    gradients.projectionBias = Matrix::zerosLike(model.outputProjection.bias);
    return gradients;
}

void LanguageModelGradients::zeroInPlace() {
    Matrix::zeroInPlace(this->tokenEmbedding);
    for (TransformerBlockGradients& block : this->blocks)
        block.zeroInPlace();
    Matrix::zeroInPlace(this->finalNormGamma);
    Matrix::zeroInPlace(this->projectionWeight);
    Matrix::zeroInPlace(this->projectionBias);
}

void LanguageModelGradients::addInPlace(const LanguageModelGradients& other) {
    Matrix::addInPlace(this->tokenEmbedding, other.tokenEmbedding);
    for (size_t blockIndex = 0; blockIndex < this->blocks.size(); ++blockIndex)
        this->blocks[blockIndex].addInPlace(other.blocks[blockIndex]);
    Matrix::addInPlace(this->finalNormGamma, other.finalNormGamma);
    Matrix::addInPlace(this->projectionWeight, other.projectionWeight);
    Matrix::addInPlace(this->projectionBias, other.projectionBias);
}

void LanguageModelGradients::scaleInPlace(float scalar) {
    Matrix::scaleInPlace(this->tokenEmbedding, scalar);
    for (TransformerBlockGradients& block : this->blocks)
        block.scaleInPlace(scalar);
    Matrix::scaleInPlace(this->finalNormGamma, scalar);
    Matrix::scaleInPlace(this->projectionWeight, scalar);
    Matrix::scaleInPlace(this->projectionBias, scalar);
}

LanguageModel::LanguageModel(int vocabularySize, int embeddingDim, int maximumPositionCount, Adam optimizer, int blockCount, int headCount)
    : tokenEmbedding(vocabularySize, embeddingDim), finalNorm(embeddingDim), outputProjection(UniformInit::matrix(vocabularySize, embeddingDim, 0.1f, 31u), UniformInit::matrix(vocabularySize, 1, 0.01f, 32u)), optimizer(optimizer), maximumPositionCount(maximumPositionCount), deviceStale(false), deviceTrainEnabled(false) {
    if (maximumPositionCount <= 0) throw std::invalid_argument("LanguageModel maximumPositionCount must be > 0");
    if (blockCount <= 0) throw std::invalid_argument("LanguageModel blockCount must be > 0");
    if (headCount <= 0) throw std::invalid_argument("LanguageModel headCount must be > 0");

    this->blocks.reserve(static_cast<size_t>(blockCount));
    for (int blockIndex = 0; blockIndex < blockCount; ++blockIndex)
        this->blocks.push_back(TransformerBlock(embeddingDim, headCount, maximumPositionCount, 21u + static_cast<unsigned>(blockIndex) * 100u));

    this->tokenEmbeddingState = AdamState::zerosLike(this->tokenEmbedding.weight);
    this->finalNormGammaState = AdamState::zerosLike(this->finalNorm.gamma);
    this->projectionWeightState = AdamState::zerosLike(this->outputProjection.weight);
    this->projectionBiasState = AdamState::zerosLike(this->outputProjection.bias);
}

LanguageModel::~LanguageModel() = default;

LanguageModel::LanguageModel(LanguageModel&&) noexcept = default;

LanguageModel& LanguageModel::operator=(LanguageModel&&) noexcept = default;

void LanguageModel::enableCuda() {
    if (!CudaMatmul::isAvailable()) {
        std::cout << "LanguageModel::enableCuda: no CUDA device\n";
        this->device.reset();
        this->deviceStale = false;
        this->deviceTrainEnabled = false;
        return;
    }

    this->device = std::make_unique<CudaLanguageModel>();
    this->device->uploadFrom(*this);
    this->deviceStale = false;
    std::cout << "LanguageModel::enableCuda: device mirror active (forward/generate)\n";
}

void LanguageModel::enableCudaTrain() {
    if (this->device == nullptr) this->enableCuda();
    if (this->device == nullptr) {
        this->deviceTrainEnabled = false;
        return;
    }

    this->deviceTrainEnabled = true;
    this->device->activationCheckpointing = true;
    CudaAmp::preferMixedPrecision = true;
    // loss scaling only helps when FP16 GEMMs can run (shared dim gate is 256)
    CudaAmp::useLossScaling = this->tokenEmbedding.embeddingDim() >= 256;
    CudaAmp::resetLossScaler();
    // int8 moments default-on for consumer VRAM; opt-out via setCudaPreferInt8AdamMoments(false)
    CudaAdam::preferInt8Moments = true;
    this->device->ensureTrainState();
    std::cout << "LanguageModel::enableCudaTrain: device training enabled (packed batches, checkpointing on, FP16 amp "
              << (CudaAmp::preferMixedPrecision ? "on" : "off")
              << ", lossScale=" << (CudaAmp::useLossScaling ? "on" : "off")
              << ", int8 adam " << (CudaAdam::preferInt8Moments ? "on" : "off") << ")\n";
}

void LanguageModel::enableActivationCheckpointing(bool enabled) {
    if (this->device == nullptr) this->enableCuda();
    if (this->device == nullptr) return;
    this->device->activationCheckpointing = enabled;
    if (enabled)
        this->device->ensureTrainWorkspaces();
    else
        this->device->releaseActivationCheckpoints();
    std::cout << "LanguageModel::enableActivationCheckpointing: " << (enabled ? "on" : "off") << '\n';
}

void LanguageModel::setCudaPreferMixedPrecision(bool enabled) {
    if (this->device == nullptr) this->enableCuda();
    if (this->device == nullptr) return;
    CudaAmp::preferMixedPrecision = enabled;
    CudaAmp::useLossScaling = enabled && this->tokenEmbedding.embeddingDim() >= 256;
    if (enabled)
        CudaAmp::resetLossScaler();
    if (this->deviceTrainEnabled)
        this->device->ensureTrainWorkspaces();
    std::cout << "LanguageModel::setCudaPreferMixedPrecision: " << (enabled ? "on" : "off")
              << "  lossScale=" << (CudaAmp::useLossScaling ? "on" : "off")
              << "  scale=" << CudaAmp::lossScaler.scale << '\n';
}

void LanguageModel::setCudaMaxPackedColumns(int columns) {
    if (this->device == nullptr) this->enableCuda();
    if (this->device == nullptr) return;
    if (columns <= 0) throw std::invalid_argument("LanguageModel::setCudaMaxPackedColumns columns must be > 0");
    this->device->maxPackedColumns = columns;
    if (this->deviceTrainEnabled)
        this->device->ensureTrainWorkspaces();
}

void LanguageModel::setCudaLogitChunkRows(int rows) {
    if (this->device == nullptr) this->enableCuda();
    if (this->device == nullptr) return;
    if (rows <= 0) throw std::invalid_argument("LanguageModel::setCudaLogitChunkRows rows must be > 0");
    this->device->logitChunkRows = rows;
    if (this->deviceTrainEnabled)
        this->device->ensureTrainWorkspaces();
    std::cout << "LanguageModel::setCudaLogitChunkRows: " << rows << '\n';
}

void LanguageModel::setCudaPreferInt8AdamMoments(bool enabled) {
    if (this->device == nullptr) this->enableCuda();
    if (this->device == nullptr) return;
    CudaAdam::preferInt8Moments = enabled;
    std::cout << "LanguageModel::setCudaPreferInt8AdamMoments: " << (enabled ? "on" : "off")
              << "  blockSize=" << CudaAdam::int8BlockSize << '\n';
}

void LanguageModel::setCudaPreferFlashAttention(bool enabled) {
    if (this->device == nullptr) this->enableCuda();
    if (this->device == nullptr) return;
    for (CudaTransformerBlock& block : this->device->blocks) {
        block.attention.preferFlashAttention = enabled;
        if (enabled)
            block.attention.releaseDenseAttentionScratch();
        else
            block.attention.releaseFlashAttentionScratch();
    }
}

bool LanguageModel::cudaEnabled() const {
    return this->device != nullptr;
}

bool LanguageModel::cudaTrainEnabled() const {
    return this->device != nullptr && this->deviceTrainEnabled;
}

void LanguageModel::syncDevice() {
    if (this->device == nullptr) return;
    this->device->uploadFrom(*this);
    this->deviceStale = false;
}

void LanguageModel::syncDeviceIfStale() {
    if (this->device == nullptr) return;
    if (!this->deviceStale) return;
    this->syncDevice();
}

Matrix LanguageModel::sumColumns(const Matrix& gradient) {
    Matrix biasGradient(gradient.rows, 1, 0.0f);
    for (size_t row = 0; row < gradient.rows; ++row) {
        float total = 0.0f;
        for (size_t column = 0; column < gradient.cols; ++column)
            total += gradient.at(row, column);
        biasGradient.at(row, 0) = total;
    }
    return biasGradient;
}

Matrix LanguageModel::broadcastBiasAdd(const Matrix& product, const Matrix& bias) {
    Matrix result = product;
    for (size_t row = 0; row < result.rows; ++row) {
        const float biasValue = bias.at(row, 0);
        for (size_t column = 0; column < result.cols; ++column)
            result.at(row, column) += biasValue;
    }
    return result;
}

Matrix LanguageModel::forwardLocal(const std::vector<int>& tokenIds, LanguageModelCache& cache) const {
    if (tokenIds.empty()) throw std::invalid_argument("LanguageModel::forwardLocal empty tokenIds");
    if (static_cast<int>(tokenIds.size()) > this->maximumPositionCount)
        throw std::invalid_argument("LanguageModel::forwardLocal sequence longer than maximumPositionCount");

    if (cache.blockCaches.size() != this->blocks.size())
        cache.blockCaches.resize(this->blocks.size());

    Matrix hidden = this->tokenEmbedding.forward(tokenIds);

    for (size_t blockIndex = 0; blockIndex < this->blocks.size(); ++blockIndex)
        hidden = this->blocks[blockIndex].forward(hidden, cache.blockCaches[blockIndex]);

    cache.blockOutput = this->finalNorm.forward(hidden, cache.finalNormCache);
    Matrix product = Matrix::multiply(this->outputProjection.weight, cache.blockOutput);
    return LanguageModel::broadcastBiasAdd(product, this->outputProjection.bias);
}

Matrix LanguageModel::forward(const std::vector<int>& tokenIds) {
    if (this->device != nullptr) {
        this->syncDeviceIfStale();
        return this->device->forward(tokenIds);
    }

    LanguageModelCache cache;
    return this->forwardLocal(tokenIds, cache);
}

float LanguageModel::exampleLoss(const LanguageModelExample& example) {
    Matrix logits = this->forward(example.inputTokenIds);
    Matrix probabilities;
    Softmax::applyInto(logits, probabilities);
    Matrix target = example.targetOneHot.empty()
        ? LanguageModelDataset::makeOneHotSequence(example.targetTokenIds, this->tokenEmbedding.vocabSize())
        : example.targetOneHot;
    return CrossEntropy::loss(probabilities, target);
}

float LanguageModel::averageLoss(const LanguageModelDataset& dataset) {
    if (dataset.examples.empty()) return 0.0f;

    if (this->device != nullptr) {
        this->syncDeviceIfStale();
        return this->device->averageLoss(dataset);
    }

    float total = 0.0f;
    const int exampleCount = static_cast<int>(dataset.examples.size());

#if defined(_OPENMP)
    #pragma omp parallel for reduction(+:total) schedule(dynamic, 4)
#endif
    for (int index = 0; index < exampleCount; ++index)
        total += this->exampleLoss(dataset.examples[static_cast<size_t>(index)]);

    return total / static_cast<float>(dataset.size());
}

float LanguageModel::accumulateExample(const LanguageModelExample& example, LanguageModelGradients& gradients, LanguageModelCache& cache) const {
    Matrix logits = this->forwardLocal(example.inputTokenIds, cache);
    Softmax::applyInto(logits, cache.probabilities);
    Matrix target = example.targetOneHot.empty()
        ? LanguageModelDataset::makeOneHotSequence(example.targetTokenIds, this->tokenEmbedding.vocabSize())
        : example.targetOneHot;
    const float loss = CrossEntropy::loss(cache.probabilities, target);

    Matrix logitGradient = CrossEntropy::gradient(cache.probabilities, target);
    Matrix projectionWeightGradient = Matrix::multiply(logitGradient, cache.blockOutput, false, true);
    Matrix projectionBiasGradient = LanguageModel::sumColumns(logitGradient);
    Matrix hiddenGradient = Matrix::multiply(this->outputProjection.weight, logitGradient, true, false);

    Matrix finalNormGammaGradient;
    hiddenGradient = this->finalNorm.backward(hiddenGradient, cache.finalNormCache, finalNormGammaGradient);

    for (int blockIndex = static_cast<int>(this->blocks.size()) - 1; blockIndex >= 0; --blockIndex)
        hiddenGradient = this->blocks[static_cast<size_t>(blockIndex)].backward(hiddenGradient, cache.blockCaches[static_cast<size_t>(blockIndex)], gradients.blocks[static_cast<size_t>(blockIndex)]);

    Matrix tokenEmbeddingGradient = this->tokenEmbedding.backward(hiddenGradient, example.inputTokenIds);

    Matrix::addInPlace(gradients.projectionWeight, projectionWeightGradient);
    Matrix::addInPlace(gradients.projectionBias, projectionBiasGradient);
    Matrix::addInPlace(gradients.finalNormGamma, finalNormGammaGradient);
    Matrix::addInPlace(gradients.tokenEmbedding, tokenEmbeddingGradient);

    return loss;
}

void LanguageModel::applyGradients(const LanguageModelGradients& gradients) {
    this->optimizer.step();
    this->optimizer.update(this->outputProjection.weight, this->projectionWeightState, gradients.projectionWeight);
    this->optimizer.update(this->outputProjection.bias, this->projectionBiasState, gradients.projectionBias);
    this->optimizer.update(this->finalNorm.gamma, this->finalNormGammaState, gradients.finalNormGamma);
    for (size_t blockIndex = 0; blockIndex < this->blocks.size(); ++blockIndex)
        this->blocks[blockIndex].applyGradients(this->optimizer, gradients.blocks[blockIndex]);
    this->optimizer.update(this->tokenEmbedding.weight, this->tokenEmbeddingState, gradients.tokenEmbedding);
    if (this->device != nullptr)
        this->deviceStale = true;
}

void LanguageModel::train(const LanguageModelDataset& dataset, int epochs, int logEveryEpochs) {
    LanguageModelDataset emptyTest;
    this->train(dataset, emptyTest, epochs, logEveryEpochs, 32);
}

float LanguageModel::trainOnExamples(
    const LanguageModelDataset& dataset,
    int batchSize,
    std::vector<LanguageModelGradients>& threadGradients,
    std::vector<LanguageModelCache>& threadCaches,
    LanguageModelGradients& merged
) {
    const int exampleCount = static_cast<int>(dataset.examples.size());
    if (exampleCount <= 0) return 0.0f;
    if (batchSize <= 0) batchSize = 32;

    const int threadCount = static_cast<int>(threadGradients.size());
    float epochLoss = 0.0f;

    for (int batchStart = 0; batchStart < exampleCount; batchStart += batchSize) {
        int batchEnd = batchStart + batchSize;
        if (batchEnd > exampleCount) batchEnd = exampleCount;
        const float batchCount = static_cast<float>(batchEnd - batchStart);

        for (int threadIndex = 0; threadIndex < threadCount; ++threadIndex)
            threadGradients[static_cast<size_t>(threadIndex)].zeroInPlace();

        float batchLoss = 0.0f;

#if defined(_OPENMP)
        #pragma omp parallel
#endif
        {
            int threadIndex = 0;
#if defined(_OPENMP)
            threadIndex = omp_get_thread_num();
            #pragma omp for reduction(+:batchLoss) schedule(dynamic, 1)
#endif
            for (int index = batchStart; index < batchEnd; ++index) {
                batchLoss += this->accumulateExample(
                    dataset.examples[static_cast<size_t>(index)],
                    threadGradients[static_cast<size_t>(threadIndex)],
                    threadCaches[static_cast<size_t>(threadIndex)]
                );
            }
        }

        merged.zeroInPlace();
        for (int threadIndex = 0; threadIndex < threadCount; ++threadIndex)
            merged.addInPlace(threadGradients[static_cast<size_t>(threadIndex)]);
        merged.scaleInPlace(1.0f / batchCount);
        this->applyGradients(merged);

        epochLoss += batchLoss;
    }

    return epochLoss;
}

void LanguageModel::train(const LanguageModelDataset& trainDataset, const LanguageModelDataset& testDataset, int epochs, int logEveryEpochs, int batchSize, int gradientAccumulationSteps) {
    if (trainDataset.examples.empty()) return;
    if (logEveryEpochs <= 0) logEveryEpochs = 1;
    if (batchSize <= 0) batchSize = 32;
    if (gradientAccumulationSteps <= 0) gradientAccumulationSteps = 1;

    if (this->cudaTrainEnabled()) {
        this->syncDeviceIfStale();
        this->device->adam = CudaAdam(this->optimizer.learningRate, this->optimizer.beta1, this->optimizer.beta2, this->optimizer.epsilon);
        this->device->adam.timeStep = this->optimizer.timeStep;
        this->device->train(trainDataset, testDataset, epochs, logEveryEpochs, batchSize, gradientAccumulationSteps);
        this->device->downloadTo(*this);
        this->optimizer.timeStep = this->device->adam.timeStep;
        this->deviceStale = false;
        return;
    }

    int threadCount = 1;
#if defined(_OPENMP)
    threadCount = omp_get_max_threads();
    if (threadCount < 1) threadCount = 1;
#endif

    std::vector<LanguageModelGradients> threadGradients(static_cast<size_t>(threadCount));
    std::vector<LanguageModelCache> threadCaches(static_cast<size_t>(threadCount));
    for (int threadIndex = 0; threadIndex < threadCount; ++threadIndex)
        threadGradients[static_cast<size_t>(threadIndex)] = LanguageModelGradients::zerosFrom(*this);

    LanguageModelGradients merged = LanguageModelGradients::zerosFrom(*this);

    for (int epoch = 0; epoch < epochs; ++epoch) {
        const auto epochStart = std::chrono::steady_clock::now();
        const float epochLoss = this->trainOnExamples(trainDataset, batchSize, threadGradients, threadCaches, merged);

        if (epoch % logEveryEpochs != 0) continue;

        const float averageTrainLoss = epochLoss / static_cast<float>(trainDataset.size());
        const auto epochEnd = std::chrono::steady_clock::now();
        const double epochSeconds = std::chrono::duration<double>(epochEnd - epochStart).count();
        const double tokensPerSecond = epochSeconds > 0.0
            ? static_cast<double>(trainDataset.totalPredictionCount()) / epochSeconds
            : 0.0;
        std::cout << "  Epoch " << epoch << "  trainLoss=" << averageTrainLoss
                  << "  sec=" << epochSeconds << "  tokens/s=" << tokensPerSecond
                  << "  backend=cpu-openmp";

        if (!testDataset.examples.empty()) {
            const float testLoss = this->averageLoss(testDataset);
            std::cout << "  testLoss=" << testLoss;
        }

        std::cout << '\n';
    }
}

void LanguageModel::train(LanguageModelChunkSource& source, int epochs, int logEveryEpochs, int batchSize, int gradientAccumulationSteps) {
    if (logEveryEpochs <= 0) logEveryEpochs = 1;
    if (batchSize <= 0) batchSize = 32;
    if (gradientAccumulationSteps <= 0) gradientAccumulationSteps = 1;

    if (this->cudaTrainEnabled()) {
        this->syncDeviceIfStale();
        this->device->adam = CudaAdam(this->optimizer.learningRate, this->optimizer.beta1, this->optimizer.beta2, this->optimizer.epsilon);
        this->device->adam.timeStep = this->optimizer.timeStep;
        this->device->train(source, epochs, logEveryEpochs, batchSize, gradientAccumulationSteps);
        this->device->downloadTo(*this);
        this->optimizer.timeStep = this->device->adam.timeStep;
        this->deviceStale = false;
        return;
    }

    int threadCount = 1;
#if defined(_OPENMP)
    threadCount = omp_get_max_threads();
    if (threadCount < 1) threadCount = 1;
#endif

    std::vector<LanguageModelGradients> threadGradients(static_cast<size_t>(threadCount));
    std::vector<LanguageModelCache> threadCaches(static_cast<size_t>(threadCount));
    for (int threadIndex = 0; threadIndex < threadCount; ++threadIndex)
        threadGradients[static_cast<size_t>(threadIndex)] = LanguageModelGradients::zerosFrom(*this);

    LanguageModelGradients merged = LanguageModelGradients::zerosFrom(*this);
    LanguageModelDataset chunk;
    const LanguageModelDataset& testDataset = source.testDataset();

    for (int epoch = 0; epoch < epochs; ++epoch) {
        const auto epochStart = std::chrono::steady_clock::now();
        float epochLoss = 0.0f;
        int processedExampleCount = 0;
        int processedPredictionCount = 0;

        source.rewindTrain();
        for (;;) {
            const bool more = source.nextTrainChunk(chunk);
            if (chunk.examples.empty()) break;
            epochLoss += this->trainOnExamples(chunk, batchSize, threadGradients, threadCaches, merged);
            processedExampleCount += chunk.size();
            processedPredictionCount += chunk.totalPredictionCount();
            if (!more) break;
        }

        if (epoch % logEveryEpochs != 0) continue;

        const float averageTrainLoss = processedExampleCount > 0
            ? epochLoss / static_cast<float>(processedExampleCount)
            : 0.0f;
        const auto epochEnd = std::chrono::steady_clock::now();
        const double epochSeconds = std::chrono::duration<double>(epochEnd - epochStart).count();
        const double tokensPerSecond = epochSeconds > 0.0
            ? static_cast<double>(processedPredictionCount) / epochSeconds
            : 0.0;
        std::cout << "  Epoch " << epoch << "  trainLoss=" << averageTrainLoss
                  << "  sec=" << epochSeconds << "  tokens/s=" << tokensPerSecond
                  << "  backend=cpu-stream";

        if (!testDataset.examples.empty()) {
            const float testLoss = this->averageLoss(testDataset);
            std::cout << "  testLoss=" << testLoss;
        }

        std::cout << '\n';
    }
}

int LanguageModel::argmaxLastColumn(const Matrix& logits) {
    const size_t lastColumn = logits.cols - 1;
    int bestTokenId = 0;
    float bestLogit = logits.at(0, lastColumn);
    for (size_t tokenId = 1; tokenId < logits.rows; ++tokenId) {
        if (logits.at(tokenId, lastColumn) <= bestLogit) continue;
        bestLogit = logits.at(tokenId, lastColumn);
        bestTokenId = static_cast<int>(tokenId);
    }
    return bestTokenId;
}

int LanguageModel::sampleLastColumn(const Matrix& logits, float temperature, int topK, unsigned& seed) {
    if (temperature <= 0.0f) return LanguageModel::argmaxLastColumn(logits);

    const size_t vocabularySize = logits.rows;
    const size_t lastColumn = logits.cols - 1;

    Matrix scaledLogits(vocabularySize, 1, 0.0f);
    for (size_t tokenId = 0; tokenId < vocabularySize; ++tokenId)
        scaledLogits.at(tokenId, 0) = logits.at(tokenId, lastColumn) / temperature;

    Matrix probabilities = Softmax::apply(scaledLogits);

    std::vector<int> candidateTokenIds;
    candidateTokenIds.reserve(vocabularySize);
    for (size_t tokenId = 0; tokenId < vocabularySize; ++tokenId)
        candidateTokenIds.push_back(static_cast<int>(tokenId));

    if (topK > 0 && static_cast<size_t>(topK) < vocabularySize) {
        const size_t keepCount = static_cast<size_t>(topK);
        for (size_t rank = 0; rank < keepCount; ++rank) {
            size_t bestIndex = rank;
            for (size_t index = rank + 1; index < candidateTokenIds.size(); ++index) {
                if (probabilities.at(static_cast<size_t>(candidateTokenIds[index]), 0)
                    <= probabilities.at(static_cast<size_t>(candidateTokenIds[bestIndex]), 0))
                    continue;
                bestIndex = index;
            }
            if (bestIndex != rank)
                std::swap(candidateTokenIds[rank], candidateTokenIds[bestIndex]);
        }
        candidateTokenIds.resize(keepCount);

        float probabilitySum = 0.0f;
        for (int tokenId : candidateTokenIds)
            probabilitySum += probabilities.at(static_cast<size_t>(tokenId), 0);
        if (probabilitySum <= 0.0f) return candidateTokenIds[0];
        for (int tokenId : candidateTokenIds)
            probabilities.at(static_cast<size_t>(tokenId), 0) /= probabilitySum;
    }

    const float unit = UniformInit::unitSample(seed);
    float cumulative = 0.0f;
    for (int tokenId : candidateTokenIds) {
        cumulative += probabilities.at(static_cast<size_t>(tokenId), 0);
        if (unit <= cumulative) return tokenId;
    }
    return candidateTokenIds.back();
}

std::vector<int> LanguageModel::generate(const std::vector<int>& promptTokenIds, int newTokenCount, float temperature, int topK, unsigned seed) {
    if (promptTokenIds.empty()) throw std::invalid_argument("LanguageModel::generate empty prompt");
    if (newTokenCount < 0) throw std::invalid_argument("LanguageModel::generate newTokenCount must be >= 0");

    std::vector<int> tokenIds = promptTokenIds;
    for (int step = 0; step < newTokenCount; ++step) {
        if (static_cast<int>(tokenIds.size()) >= this->maximumPositionCount) break;

        Matrix logits = this->forward(tokenIds);
        const int nextTokenId = LanguageModel::sampleLastColumn(logits, temperature, topK, seed);
        tokenIds.push_back(nextTokenId);
    }

    return tokenIds;
}

namespace {
constexpr char kCheckpointMagic[4] = { 'S', 'N', 'L', 'M' };
constexpr std::int32_t kCheckpointVersion = 1;

void writePod(std::ostream& out, const void* data, size_t byteCount) {
    out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(byteCount));
    if (!out) throw std::runtime_error("LanguageModel checkpoint write failed");
}

void readPod(std::istream& in, void* data, size_t byteCount) {
    in.read(reinterpret_cast<char*>(data), static_cast<std::streamsize>(byteCount));
    if (!in) throw std::runtime_error("LanguageModel checkpoint read failed");
}

void writeI32(std::ostream& out, std::int32_t value) { writePod(out, &value, sizeof(value)); }
void writeF32(std::ostream& out, float value) { writePod(out, &value, sizeof(value)); }
std::int32_t readI32(std::istream& in) {
    std::int32_t value = 0;
    readPod(in, &value, sizeof(value));
    return value;
}
float readF32(std::istream& in) {
    float value = 0.0f;
    readPod(in, &value, sizeof(value));
    return value;
}

void writeMatrix(std::ostream& out, const Matrix& matrix) {
    writeI32(out, static_cast<std::int32_t>(matrix.rows));
    writeI32(out, static_cast<std::int32_t>(matrix.cols));
    if (matrix.data.empty()) return;
    writePod(out, matrix.data.data(), matrix.data.size() * sizeof(float));
}

Matrix readMatrix(std::istream& in) {
    const std::int32_t rows = readI32(in);
    const std::int32_t cols = readI32(in);
    if (rows < 0 || cols < 0) throw std::runtime_error("LanguageModel checkpoint invalid matrix shape");
    Matrix matrix(static_cast<size_t>(rows), static_cast<size_t>(cols), 0.0f);
    if (matrix.data.empty()) return matrix;
    readPod(in, matrix.data.data(), matrix.data.size() * sizeof(float));
    return matrix;
}

void expectMatrixShape(const Matrix& matrix, size_t rows, size_t cols, const char* name) {
    if (matrix.rows != rows || matrix.cols != cols)
        throw std::runtime_error(std::string("LanguageModel checkpoint shape mismatch: ") + name);
}

void writeAdamState(std::ostream& out, const AdamState& state) {
    writeMatrix(out, state.firstMoment);
    writeMatrix(out, state.secondMoment);
}

AdamState readAdamState(std::istream& in) {
    AdamState state;
    state.firstMoment = readMatrix(in);
    state.secondMoment = readMatrix(in);
    return state;
}
}

void LanguageModel::saveCheckpoint(const std::string& path, bool includeOptimizer) {
    if (path.empty()) throw std::invalid_argument("LanguageModel::saveCheckpoint empty path");
    if (this->blocks.empty()) throw std::logic_error("LanguageModel::saveCheckpoint no blocks");

    if (this->cudaTrainEnabled() && this->device != nullptr) {
        this->device->downloadTo(*this);
        if (includeOptimizer)
            this->device->downloadOptimizerTo(*this);
        this->deviceStale = false;
    } else if (this->cudaEnabled() && this->device != nullptr && !this->deviceStale) {
        this->device->downloadTo(*this);
    }

    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("LanguageModel::saveCheckpoint cannot open file");

    writePod(out, kCheckpointMagic, 4);
    writeI32(out, kCheckpointVersion);
    writeI32(out, this->tokenEmbedding.vocabSize());
    writeI32(out, this->tokenEmbedding.embeddingDim());
    writeI32(out, this->maximumPositionCount);
    writeI32(out, static_cast<std::int32_t>(this->blocks.size()));
    writeI32(out, this->blocks[0].attention.headCount);
    writeF32(out, this->optimizer.learningRate);
    writeF32(out, this->optimizer.beta1);
    writeF32(out, this->optimizer.beta2);
    writeF32(out, this->optimizer.epsilon);
    writeI32(out, this->optimizer.timeStep);
    writeI32(out, includeOptimizer ? 1 : 0);

    writeMatrix(out, this->tokenEmbedding.weight);
    for (const TransformerBlock& block : this->blocks) {
        writeMatrix(out, block.attention.queryWeight);
        writeMatrix(out, block.attention.keyWeight);
        writeMatrix(out, block.attention.valueWeight);
        writeMatrix(out, block.attention.outputWeight);
        writeMatrix(out, block.attentionNorm.gamma);
        writeMatrix(out, block.feedForwardNorm.gamma);
        writeMatrix(out, block.feedForward.gateWeight);
        writeMatrix(out, block.feedForward.gateBias);
        writeMatrix(out, block.feedForward.upWeight);
        writeMatrix(out, block.feedForward.upBias);
        writeMatrix(out, block.feedForward.downWeight);
        writeMatrix(out, block.feedForward.downBias);
    }
    writeMatrix(out, this->finalNorm.gamma);
    writeMatrix(out, this->outputProjection.weight);
    writeMatrix(out, this->outputProjection.bias);

    if (includeOptimizer) {
        writeAdamState(out, this->tokenEmbeddingState);
        for (const TransformerBlock& block : this->blocks) {
            writeAdamState(out, block.queryWeightState);
            writeAdamState(out, block.keyWeightState);
            writeAdamState(out, block.valueWeightState);
            writeAdamState(out, block.attentionOutputWeightState);
            writeAdamState(out, block.attentionNormGammaState);
            writeAdamState(out, block.feedForwardNormGammaState);
            writeAdamState(out, block.feedForwardGateWeightState);
            writeAdamState(out, block.feedForwardGateBiasState);
            writeAdamState(out, block.feedForwardUpWeightState);
            writeAdamState(out, block.feedForwardUpBiasState);
            writeAdamState(out, block.feedForwardDownWeightState);
            writeAdamState(out, block.feedForwardDownBiasState);
        }
        writeAdamState(out, this->finalNormGammaState);
        writeAdamState(out, this->projectionWeightState);
        writeAdamState(out, this->projectionBiasState);
    }
}

void LanguageModel::loadCheckpoint(const std::string& path) {
    if (path.empty()) throw std::invalid_argument("LanguageModel::loadCheckpoint empty path");
    if (this->blocks.empty()) throw std::logic_error("LanguageModel::loadCheckpoint no blocks");

    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("LanguageModel::loadCheckpoint cannot open file");

    char magic[4] = {};
    readPod(in, magic, 4);
    if (magic[0] != kCheckpointMagic[0] || magic[1] != kCheckpointMagic[1]
        || magic[2] != kCheckpointMagic[2] || magic[3] != kCheckpointMagic[3])
        throw std::runtime_error("LanguageModel::loadCheckpoint bad magic");

    const std::int32_t version = readI32(in);
    if (version != kCheckpointVersion)
        throw std::runtime_error("LanguageModel::loadCheckpoint unsupported version");

    const std::int32_t vocabularySize = readI32(in);
    const std::int32_t embeddingDim = readI32(in);
    const std::int32_t maximumPositionCount = readI32(in);
    const std::int32_t blockCount = readI32(in);
    const std::int32_t headCount = readI32(in);
    if (vocabularySize != this->tokenEmbedding.vocabSize()
        || embeddingDim != this->tokenEmbedding.embeddingDim()
        || maximumPositionCount != this->maximumPositionCount
        || blockCount != static_cast<std::int32_t>(this->blocks.size())
        || headCount != this->blocks[0].attention.headCount)
        throw std::runtime_error("LanguageModel::loadCheckpoint architecture mismatch");

    this->optimizer.learningRate = readF32(in);
    this->optimizer.beta1 = readF32(in);
    this->optimizer.beta2 = readF32(in);
    this->optimizer.epsilon = readF32(in);
    this->optimizer.timeStep = readI32(in);
    const bool includeOptimizer = readI32(in) != 0;

    this->tokenEmbedding.weight = readMatrix(in);
    expectMatrixShape(this->tokenEmbedding.weight, static_cast<size_t>(vocabularySize), static_cast<size_t>(embeddingDim), "tokenEmbedding");
    for (TransformerBlock& block : this->blocks) {
        block.attention.queryWeight = readMatrix(in);
        block.attention.keyWeight = readMatrix(in);
        block.attention.valueWeight = readMatrix(in);
        block.attention.outputWeight = readMatrix(in);
        block.attentionNorm.gamma = readMatrix(in);
        block.feedForwardNorm.gamma = readMatrix(in);
        block.feedForward.gateWeight = readMatrix(in);
        block.feedForward.gateBias = readMatrix(in);
        block.feedForward.upWeight = readMatrix(in);
        block.feedForward.upBias = readMatrix(in);
        block.feedForward.downWeight = readMatrix(in);
        block.feedForward.downBias = readMatrix(in);
    }
    this->finalNorm.gamma = readMatrix(in);
    this->outputProjection.weight = readMatrix(in);
    this->outputProjection.bias = readMatrix(in);

    if (includeOptimizer) {
        this->tokenEmbeddingState = readAdamState(in);
        for (TransformerBlock& block : this->blocks) {
            block.queryWeightState = readAdamState(in);
            block.keyWeightState = readAdamState(in);
            block.valueWeightState = readAdamState(in);
            block.attentionOutputWeightState = readAdamState(in);
            block.attentionNormGammaState = readAdamState(in);
            block.feedForwardNormGammaState = readAdamState(in);
            block.feedForwardGateWeightState = readAdamState(in);
            block.feedForwardGateBiasState = readAdamState(in);
            block.feedForwardUpWeightState = readAdamState(in);
            block.feedForwardUpBiasState = readAdamState(in);
            block.feedForwardDownWeightState = readAdamState(in);
            block.feedForwardDownBiasState = readAdamState(in);
        }
        this->finalNormGammaState = readAdamState(in);
        this->projectionWeightState = readAdamState(in);
        this->projectionBiasState = readAdamState(in);
    }

    if (this->device != nullptr) {
        this->device->uploadFrom(*this);
        if (this->deviceTrainEnabled && includeOptimizer)
            this->device->uploadOptimizerFrom(*this);
        this->deviceStale = false;
    }
}

void LanguageModel::runCheckpointSmokeDemo() {
    LanguageModel model(64, 32, 16, Adam(0.001f), 1, 2);
    model.enableCuda();
    if (model.cudaEnabled())
        model.enableCudaTrain();

    const std::vector<int> tokenIds = { 1, 2, 3, 4, 5, 6, 7, 8 };
    Matrix before = model.forward(tokenIds);

    // touch optimizer state so moments are non-zero when cuda train is on
    if (model.cudaTrainEnabled()) {
        LanguageModelDataset dataset;
        dataset.vocabularySize = 64;
        LanguageModelExample example;
        example.inputTokenIds = tokenIds;
        example.targetTokenIds = { 2, 3, 4, 5, 6, 7, 8, 9 };
        dataset.examples.push_back(example);
        model.train(dataset, LanguageModelDataset(), 1, 1, 1, 1);
        before = model.forward(tokenIds);
    }

    const std::string path = "checkpoint_smoke.snlm";
    model.saveCheckpoint(path, true);

    LanguageModel restored(64, 32, 16, Adam(0.002f), 1, 2);
    if (model.cudaEnabled()) {
        restored.enableCuda();
        if (model.cudaTrainEnabled())
            restored.enableCudaTrain();
    }
    restored.loadCheckpoint(path);

    Matrix after = restored.forward(tokenIds);
    float maximumDifference = 0.0f;
    for (size_t index = 0; index < before.data.size(); ++index)
        maximumDifference = (std::max)(maximumDifference, std::fabs(before.data[index] - after.data[index]));

    const bool optimizerMatch = restored.optimizer.timeStep == model.optimizer.timeStep
        && restored.optimizer.learningRate == model.optimizer.learningRate;
    SmokeLog::result("LanguageModel checkpoint", "path=%s  logitsDiff=%.2e  timeStep=%d  optOk=%s",
        path.c_str(), maximumDifference, restored.optimizer.timeStep, optimizerMatch ? "yes" : "no");

    std::remove(path.c_str());
}

void LanguageModel::runStreamingSmokeDemo() {
    const std::string path = "streaming_smoke.jsonl";
    {
        std::ofstream out(path, std::ios::binary);
        if (!out) throw std::runtime_error("LanguageModel::runStreamingSmokeDemo cannot write temp jsonl");
        const char* rows[] = {
            "{\"problem_statement\": \"alpha beta gamma delta epsilon\", \"source\": \"Sera-T1\"}",
            "{\"problem_statement\": \"one two three four five six\", \"source\": \"Sera-T2\"}",
            "{\"problem_statement\": \"red blue green yellow orange\", \"source\": \"Sera-T1\"}",
            "{\"problem_statement\": \"cat dog bird fish mouse\", \"source\": \"Sera-T2\"}",
            "{\"problem_statement\": \"train test val split batch\", \"source\": \"Sera-T1\"}",
            "{\"problem_statement\": \"cuda kernel launch grid block\", \"source\": \"Sera-T2\"}",
            "{\"problem_statement\": \"token embed rope attention\", \"source\": \"Sera-T1\"}",
            "{\"problem_statement\": \"loss scale adam moments\", \"source\": \"Sera-T2\"}",
        };
        for (const char* row : rows)
            out << row << '\n';
    }

    LanguageModelChunkSource source(path, 0, 32, 3, 0.75f, 7u, 2);
    std::vector<std::string> sample = source.prepareTokenizerSample(8);
    if (sample.empty()) {
        SmokeLog::result("LanguageModel stream", "skipped empty tokenizer sample");
        std::remove(path.c_str());
        return;
    }

    BPETokenizer tokenizer;
    tokenizer.train(sample, 64);
    source.setTokenizer(&tokenizer);
    source.prepareTestReservoir();

    LanguageModel model(tokenizer.vocabSize(), 32, 32, Adam(0.001f), 1, 2);
    model.enableCuda();
    if (model.cudaEnabled())
        model.enableCudaTrain();

    model.train(source, 1, 1, 2, 1);

    int chunkCount = 0;
    int exampleCount = 0;
    LanguageModelDataset chunk;
    source.rewindTrain();
    for (;;) {
        const bool more = source.nextTrainChunk(chunk);
        if (chunk.examples.empty()) break;
        ++chunkCount;
        exampleCount += chunk.size();
        if (!more) break;
    }

    SmokeLog::result("LanguageModel stream", "chunks=%d  trainExamples=%d  test=%d  vocab=%d  backend=%s",
        chunkCount,
        exampleCount,
        source.testDataset().size(),
        tokenizer.vocabSize(),
        model.cudaTrainEnabled() ? "cuda-stream" : "cpu-stream");

    std::remove(path.c_str());
}
