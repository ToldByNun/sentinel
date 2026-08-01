#include "LanguageModel.hpp"

#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

#include "../Activations/Softmax.hpp"
#include "../Cuda/CudaLanguageModel.hpp"
#include "../Cuda/CudaAmp.hpp"
#include "../Cuda/CudaMatmul.hpp"
#include "../Initializers/UniformInit.hpp"
#include "../Losses/CrossEntropy.hpp"

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
    this->device->ensureTrainState();
    std::cout << "LanguageModel::enableCudaTrain: device training enabled (packed batches, checkpointing on, FP16 amp on)\n";
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
    if (this->deviceTrainEnabled)
        this->device->ensureTrainWorkspaces();
    std::cout << "LanguageModel::setCudaPreferMixedPrecision: " << (enabled ? "on" : "off")
              << "  lossScale=" << CudaAmp::lossScaler.scale << '\n';
}

void LanguageModel::setCudaMaxPackedColumns(int columns) {
    if (this->device == nullptr) this->enableCuda();
    if (this->device == nullptr) return;
    if (columns <= 0) throw std::invalid_argument("LanguageModel::setCudaMaxPackedColumns columns must be > 0");
    this->device->maxPackedColumns = columns;
    if (this->deviceTrainEnabled)
        this->device->ensureTrainWorkspaces();
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

    const int exampleCount = static_cast<int>(trainDataset.examples.size());

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
        float epochLoss = 0.0f;
        const auto epochStart = std::chrono::steady_clock::now();

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
                    batchLoss += this->accumulateExample(trainDataset.examples[static_cast<size_t>(index)], threadGradients[static_cast<size_t>(threadIndex)], threadCaches[static_cast<size_t>(threadIndex)]);
                }
            }

            merged.zeroInPlace();
            for (int threadIndex = 0; threadIndex < threadCount; ++threadIndex)
                merged.addInPlace(threadGradients[static_cast<size_t>(threadIndex)]);
            merged.scaleInPlace(1.0f / batchCount);
            this->applyGradients(merged);

            epochLoss += batchLoss;
        }

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
