#include "LanguageModel.hpp"

#include <iostream>
#include <stdexcept>
#include <vector>

#include "../Activations/Softmax.hpp"
#include "../Initializers/UniformInit.hpp"
#include "../Losses/CrossEntropy.hpp"

#if defined(_OPENMP)
#include <omp.h>
#endif

LanguageModelGradients LanguageModelGradients::zerosFrom(const LanguageModel& model) {
    LanguageModelGradients gradients;
    gradients.tokenEmbedding = Matrix::zerosLike(model.tokenEmbedding.weight);
    gradients.positionEmbedding = Matrix::zerosLike(model.positionEmbedding.weight);
    gradients.queryWeight = Matrix::zerosLike(model.attention.queryWeight);
    gradients.keyWeight = Matrix::zerosLike(model.attention.keyWeight);
    gradients.valueWeight = Matrix::zerosLike(model.attention.valueWeight);
    gradients.attentionOutputWeight = Matrix::zerosLike(model.attention.outputWeight);
    gradients.projectionWeight = Matrix::zerosLike(model.outputProjection.weight);
    gradients.projectionBias = Matrix::zerosLike(model.outputProjection.bias);
    return gradients;
}

void LanguageModelGradients::addInPlace(const LanguageModelGradients& other) {
    Matrix::addInPlace(this->tokenEmbedding, other.tokenEmbedding);
    Matrix::addInPlace(this->positionEmbedding, other.positionEmbedding);
    Matrix::addInPlace(this->queryWeight, other.queryWeight);
    Matrix::addInPlace(this->keyWeight, other.keyWeight);
    Matrix::addInPlace(this->valueWeight, other.valueWeight);
    Matrix::addInPlace(this->attentionOutputWeight, other.attentionOutputWeight);
    Matrix::addInPlace(this->projectionWeight, other.projectionWeight);
    Matrix::addInPlace(this->projectionBias, other.projectionBias);
}

void LanguageModelGradients::scaleInPlace(float scalar) {
    Matrix::scaleInPlace(this->tokenEmbedding, scalar);
    Matrix::scaleInPlace(this->positionEmbedding, scalar);
    Matrix::scaleInPlace(this->queryWeight, scalar);
    Matrix::scaleInPlace(this->keyWeight, scalar);
    Matrix::scaleInPlace(this->valueWeight, scalar);
    Matrix::scaleInPlace(this->attentionOutputWeight, scalar);
    Matrix::scaleInPlace(this->projectionWeight, scalar);
    Matrix::scaleInPlace(this->projectionBias, scalar);
}

LanguageModel::LanguageModel(
    int vocabularySize,
    int embeddingDim,
    int maximumPositionCount,
    Adam optimizer
)
    : tokenEmbedding(vocabularySize, embeddingDim),
      positionEmbedding(maximumPositionCount, embeddingDim),
      attention(CausalSelfAttention::create(embeddingDim, 21u)),
      outputProjection(
          UniformInit::matrix(vocabularySize, embeddingDim, 0.1f, 31u),
          UniformInit::matrix(vocabularySize, 1, 0.01f, 32u)
      ),
      optimizer(optimizer),
      maximumPositionCount(maximumPositionCount) {
    if (maximumPositionCount <= 0) throw std::invalid_argument("LanguageModel maximumPositionCount must be > 0");

    this->tokenEmbeddingState = AdamState::zerosLike(this->tokenEmbedding.weight);
    this->positionEmbeddingState = AdamState::zerosLike(this->positionEmbedding.weight);
    this->queryWeightState = AdamState::zerosLike(this->attention.queryWeight);
    this->keyWeightState = AdamState::zerosLike(this->attention.keyWeight);
    this->valueWeightState = AdamState::zerosLike(this->attention.valueWeight);
    this->outputWeightState = AdamState::zerosLike(this->attention.outputWeight);
    this->projectionWeightState = AdamState::zerosLike(this->outputProjection.weight);
    this->projectionBiasState = AdamState::zerosLike(this->outputProjection.bias);
}

std::vector<int> LanguageModel::positionIds(size_t sequenceLength) {
    std::vector<int> ids(sequenceLength);
    for (size_t index = 0; index < sequenceLength; ++index)
        ids[index] = static_cast<int>(index);
    return ids;
}

Matrix LanguageModel::sumColumns(const Matrix& gradient) {
    Matrix biasGradient;
    biasGradient.data = std::vector<std::vector<float>>(gradient.data.size(), std::vector<float>(1, 0.0f));
    for (size_t row = 0; row < gradient.data.size(); ++row) {
        float total = 0.0f;
        for (size_t column = 0; column < gradient.data[row].size(); ++column)
            total += gradient.data[row][column];
        biasGradient.data[row][0] = total;
    }
    return biasGradient;
}

Matrix LanguageModel::broadcastBiasAdd(const Matrix& product, const Matrix& bias) {
    Matrix result = product;
    for (size_t row = 0; row < result.data.size(); ++row) {
        const float biasValue = bias.data[row][0];
        for (size_t column = 0; column < result.data[row].size(); ++column)
            result.data[row][column] += biasValue;
    }
    return result;
}

Matrix LanguageModel::forwardLocal(
    const std::vector<int>& tokenIds,
    CausalSelfAttentionCache& attentionCache,
    Matrix& projectionInput
) const {
    if (tokenIds.empty()) throw std::invalid_argument("LanguageModel::forwardLocal empty tokenIds");
    if (static_cast<int>(tokenIds.size()) > this->maximumPositionCount)
        throw std::invalid_argument("LanguageModel::forwardLocal sequence longer than maximumPositionCount");

    const std::vector<int> positions = LanguageModel::positionIds(tokenIds.size());
    Matrix tokenEmbedded = this->tokenEmbedding.forward(tokenIds);
    Matrix positionEmbedded = this->positionEmbedding.forward(positions);
    Matrix combined = Matrix::add(tokenEmbedded, positionEmbedded);

    Matrix attended = this->attention.forward(combined, attentionCache);
    projectionInput = Matrix::add(combined, attended);
    Matrix product = Matrix::multiply(this->outputProjection.weight, projectionInput);
    return LanguageModel::broadcastBiasAdd(product, this->outputProjection.bias);
}

Matrix LanguageModel::forward(const std::vector<int>& tokenIds) {
    CausalSelfAttentionCache attentionCache;
    Matrix projectionInput;
    return this->forwardLocal(tokenIds, attentionCache, projectionInput);
}

float LanguageModel::exampleLoss(const LanguageModelExample& example) {
    CausalSelfAttentionCache attentionCache;
    Matrix projectionInput;
    Matrix logits = this->forwardLocal(example.inputTokenIds, attentionCache, projectionInput);
    Matrix probabilities = Softmax::apply(logits);
    Matrix target = example.targetOneHot.data.empty()
        ? LanguageModelDataset::makeOneHotSequence(example.targetTokenIds, this->tokenEmbedding.vocabSize())
        : example.targetOneHot;
    return CrossEntropy::loss(probabilities, target);
}

float LanguageModel::averageLoss(const LanguageModelDataset& dataset) {
    if (dataset.examples.empty()) return 0.0f;

    float total = 0.0f;
    const int exampleCount = static_cast<int>(dataset.examples.size());

#if defined(_OPENMP)
    #pragma omp parallel for reduction(+:total) schedule(dynamic, 4)
#endif
    for (int index = 0; index < exampleCount; ++index)
        total += this->exampleLoss(dataset.examples[static_cast<size_t>(index)]);

    return total / static_cast<float>(dataset.size());
}

float LanguageModel::accumulateExample(const LanguageModelExample& example, LanguageModelGradients& gradients) const {
    CausalSelfAttentionCache attentionCache;
    Matrix projectionInput;
    Matrix logits = this->forwardLocal(example.inputTokenIds, attentionCache, projectionInput);
    Matrix probabilities = Softmax::apply(logits);
    Matrix target = example.targetOneHot.data.empty()
        ? LanguageModelDataset::makeOneHotSequence(example.targetTokenIds, this->tokenEmbedding.vocabSize())
        : example.targetOneHot;
    const float loss = CrossEntropy::loss(probabilities, target);

    Matrix logitGradient = CrossEntropy::gradient(probabilities, target);

    Matrix projectionWeightGradient = Matrix::multiply(logitGradient, Matrix::transpose(projectionInput));
    Matrix projectionBiasGradient = LanguageModel::sumColumns(logitGradient);
    Matrix residualGradient = Matrix::multiply(Matrix::transpose(this->outputProjection.weight), logitGradient);

    Matrix queryWeightGradient;
    Matrix keyWeightGradient;
    Matrix valueWeightGradient;
    Matrix outputWeightGradient;
    Matrix attentionInputGradient = this->attention.backward(
        residualGradient,
        attentionCache,
        queryWeightGradient,
        keyWeightGradient,
        valueWeightGradient,
        outputWeightGradient
    );

    Matrix combinedGradient = Matrix::add(residualGradient, attentionInputGradient);
    const std::vector<int> positions = LanguageModel::positionIds(example.inputTokenIds.size());
    Matrix tokenEmbeddingGradient = this->tokenEmbedding.backward(combinedGradient, example.inputTokenIds);
    Matrix positionEmbeddingGradient = this->positionEmbedding.backward(combinedGradient, positions);

    Matrix::addInPlace(gradients.projectionWeight, projectionWeightGradient);
    Matrix::addInPlace(gradients.projectionBias, projectionBiasGradient);
    Matrix::addInPlace(gradients.attentionOutputWeight, outputWeightGradient);
    Matrix::addInPlace(gradients.valueWeight, valueWeightGradient);
    Matrix::addInPlace(gradients.keyWeight, keyWeightGradient);
    Matrix::addInPlace(gradients.queryWeight, queryWeightGradient);
    Matrix::addInPlace(gradients.positionEmbedding, positionEmbeddingGradient);
    Matrix::addInPlace(gradients.tokenEmbedding, tokenEmbeddingGradient);

    return loss;
}

void LanguageModel::applyGradients(const LanguageModelGradients& gradients) {
    this->optimizer.step();
    this->optimizer.update(this->outputProjection.weight, this->projectionWeightState, gradients.projectionWeight);
    this->optimizer.update(this->outputProjection.bias, this->projectionBiasState, gradients.projectionBias);
    this->optimizer.update(this->attention.outputWeight, this->outputWeightState, gradients.attentionOutputWeight);
    this->optimizer.update(this->attention.valueWeight, this->valueWeightState, gradients.valueWeight);
    this->optimizer.update(this->attention.keyWeight, this->keyWeightState, gradients.keyWeight);
    this->optimizer.update(this->attention.queryWeight, this->queryWeightState, gradients.queryWeight);
    this->optimizer.update(this->positionEmbedding.weight, this->positionEmbeddingState, gradients.positionEmbedding);
    this->optimizer.update(this->tokenEmbedding.weight, this->tokenEmbeddingState, gradients.tokenEmbedding);
}

void LanguageModel::train(const LanguageModelDataset& dataset, int epochs, int logEveryEpochs) {
    LanguageModelDataset emptyTest;
    this->train(dataset, emptyTest, epochs, logEveryEpochs, 32);
}

void LanguageModel::train(
    const LanguageModelDataset& trainDataset,
    const LanguageModelDataset& testDataset,
    int epochs,
    int logEveryEpochs,
    int batchSize
) {
    if (trainDataset.examples.empty()) return;
    if (logEveryEpochs <= 0) logEveryEpochs = 1;
    if (batchSize <= 0) batchSize = 32;

    const int exampleCount = static_cast<int>(trainDataset.examples.size());

    int threadCount = 1;
#if defined(_OPENMP)
    threadCount = omp_get_max_threads();
    if (threadCount < 1) threadCount = 1;
#endif

    for (int epoch = 0; epoch < epochs; ++epoch) {
        float epochLoss = 0.0f;

        for (int batchStart = 0; batchStart < exampleCount; batchStart += batchSize) {
            int batchEnd = batchStart + batchSize;
            if (batchEnd > exampleCount) batchEnd = exampleCount;
            const float batchCount = static_cast<float>(batchEnd - batchStart);

            std::vector<LanguageModelGradients> threadGradients(static_cast<size_t>(threadCount));
            for (int threadIndex = 0; threadIndex < threadCount; ++threadIndex)
                threadGradients[static_cast<size_t>(threadIndex)] = LanguageModelGradients::zerosFrom(*this);

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
                        trainDataset.examples[static_cast<size_t>(index)],
                        threadGradients[static_cast<size_t>(threadIndex)]
                    );
                }
            }

            LanguageModelGradients merged = threadGradients[0];
            for (int threadIndex = 1; threadIndex < threadCount; ++threadIndex)
                merged.addInPlace(threadGradients[static_cast<size_t>(threadIndex)]);
            merged.scaleInPlace(1.0f / batchCount);
            this->applyGradients(merged);

            epochLoss += batchLoss;
        }

        if (epoch % logEveryEpochs != 0) continue;

        const float averageTrainLoss = epochLoss / static_cast<float>(trainDataset.size());
        std::cout << "Epoch " << epoch << " | trainLoss: " << averageTrainLoss;

        if (!testDataset.examples.empty()) {
            const float testLoss = this->averageLoss(testDataset);
            std::cout << " | testLoss: " << testLoss;
        }

        std::cout << '\n';
    }
}

std::vector<int> LanguageModel::generate(const std::vector<int>& promptTokenIds, int newTokenCount) {
    if (promptTokenIds.empty()) throw std::invalid_argument("LanguageModel::generate empty prompt");
    if (newTokenCount < 0) throw std::invalid_argument("LanguageModel::generate newTokenCount must be >= 0");

    std::vector<int> tokenIds = promptTokenIds;
    for (int step = 0; step < newTokenCount; ++step) {
        if (static_cast<int>(tokenIds.size()) >= this->maximumPositionCount) break;

        Matrix logits = this->forward(tokenIds);
        const size_t lastColumn = logits.data[0].size() - 1;

        int bestTokenId = 0;
        float bestLogit = logits.data[0][lastColumn];
        for (size_t tokenId = 1; tokenId < logits.data.size(); ++tokenId) {
            if (logits.data[tokenId][lastColumn] <= bestLogit) continue;
            bestLogit = logits.data[tokenId][lastColumn];
            bestTokenId = static_cast<int>(tokenId);
        }

        tokenIds.push_back(bestTokenId);
    }

    return tokenIds;
}
