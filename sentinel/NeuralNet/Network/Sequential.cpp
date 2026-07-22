#include "Sequential.hpp"

#include <iostream>
#include <utility>

#include "../Activations/ReLU.hpp"
#include "../Activations/Softmax.hpp"
#include "../Losses/CrossEntropy.hpp"

Sequential::Sequential(Dense layer1, Dense layer2, Adam optimizer, float dropRate)
    : layer1(std::move(layer1)),
      layer2(std::move(layer2)),
      dropout(dropRate),
      optimizer(optimizer),
      layer1WeightState(AdamState::zerosLike(this->layer1.weight)),
      layer1BiasState(AdamState::zerosLike(this->layer1.bias)),
      layer2WeightState(AdamState::zerosLike(this->layer2.weight)),
      layer2BiasState(AdamState::zerosLike(this->layer2.bias)) {}

Matrix Sequential::zerosLike(const Matrix& matrix) {
    Matrix result = matrix;
    for (size_t row = 0; row < result.data.size(); ++row) {
        for (size_t column = 0; column < result.data[row].size(); ++column)
            result.data[row][column] = 0.0f;
    }
    return result;
}

unsigned Sequential::advanceSeed(unsigned seed) {
    return seed * 1664525u + 1013904223u;
}

std::vector<size_t> Sequential::shuffledOrder(size_t exampleCount, unsigned& seed) {
    std::vector<size_t> order(exampleCount);
    for (size_t index = 0; index < order.size(); ++index) order[index] = index;

    for (size_t index = order.size(); index > 1; --index) {
        seed = Sequential::advanceSeed(seed);
        const size_t swapIndex = static_cast<size_t>(seed % index);
        const size_t lastIndex = index - 1;
        const size_t temporary = order[lastIndex];
        order[lastIndex] = order[swapIndex];
        order[swapIndex] = temporary;
    }

    return order;
}

void Sequential::accumulateGradient(Matrix& total, const Matrix& gradient) {
    for (size_t row = 0; row < total.data.size(); ++row) {
        for (size_t column = 0; column < total.data[row].size(); ++column)
            total.data[row][column] += gradient.data[row][column];
    }
}

void Sequential::updateDenseParameters(
    const Matrix& weightGradient1,
    const Matrix& biasGradient1,
    const Matrix& weightGradient2,
    const Matrix& biasGradient2
) {
    this->optimizer.step();
    this->optimizer.update(this->layer2.weight, this->layer2WeightState, weightGradient2);
    this->optimizer.update(this->layer2.bias, this->layer2BiasState, biasGradient2);
    this->optimizer.update(this->layer1.weight, this->layer1WeightState, weightGradient1);
    this->optimizer.update(this->layer1.bias, this->layer1BiasState, biasGradient1);
}

Matrix Sequential::forward(const Matrix& input) {
    this->dropout.training = false;
    Matrix hidden = ReLU::apply(this->layer1.forward(input));
    Matrix dropped = this->dropout.forward(hidden);
    Matrix logits = this->layer2.forward(dropped);
    return Softmax::apply(logits);
}

Matrix Sequential::forward(Embedding& embedding, MeanPool& meanPool, const std::vector<int>& tokenIds) {
    Matrix embedded = embedding.forward(tokenIds);
    Matrix pooled = meanPool.forward(embedded);
    return this->forward(pooled);
}

void Sequential::train(const Matrix& input, const Matrix& target, int epochs) {
    this->dropout.training = true;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        Matrix hidden = ReLU::apply(this->layer1.forward(input));
        Matrix dropped = this->dropout.forward(hidden);
        Matrix logits = this->layer2.forward(dropped);
        Matrix probabilities = Softmax::apply(logits);

        Matrix layer2Gradient = CrossEntropy::gradient(probabilities, target);
        Matrix weightGradient2 = Matrix::multiply(
            layer2Gradient,
            Matrix::transpose(this->layer2.lastInput)
        );

        Matrix inputGradient2 = Matrix::multiply(
            Matrix::transpose(this->layer2.weight),
            layer2Gradient
        );
        Matrix hiddenGradient = this->dropout.backward(inputGradient2);
        Matrix layer1Gradient = Matrix::multiplyElementwise(
            hiddenGradient,
            ReLU::derivative(this->layer1.lastZ)
        );
        Matrix weightGradient1 = Matrix::multiply(
            layer1Gradient,
            Matrix::transpose(this->layer1.lastInput)
        );

        this->updateDenseParameters(weightGradient1, layer1Gradient, weightGradient2, layer2Gradient);

        if (epoch % 1000 == 0) {
            const float loss = CrossEntropy::loss(probabilities, target);
            std::cout << "Epoch " << epoch << " | loss: " << loss << '\n';
        }
    }

    this->dropout.training = false;
}

void Sequential::train(Embedding& embedding, MeanPool& meanPool, const std::vector<int>& tokenIds, const Matrix& target, int epochs) {
    ClassificationDataset dataset;
    ClassificationExample example;
    example.tokenIds = tokenIds;
    example.target = target;
    dataset.examples.push_back(std::move(example));
    this->train(embedding, meanPool, dataset, epochs);
}

void Sequential::train(Embedding& embedding, MeanPool& meanPool, const ClassificationDataset& dataset, int epochs) {
    ClassificationDataset emptyTest;
    this->train(embedding, meanPool, dataset, emptyTest, epochs, 500, 3, 16);
}

void Sequential::train(
    Embedding& embedding,
    MeanPool& meanPool,
    const ClassificationDataset& trainDataset,
    const ClassificationDataset& testDataset,
    int epochs,
    int logEveryEpochs,
    int earlyStoppingPatience,
    int batchSize
) {
    if (trainDataset.examples.empty()) return;
    if (logEveryEpochs <= 0) logEveryEpochs = 500;
    if (earlyStoppingPatience <= 0) earlyStoppingPatience = 3;
    if (batchSize <= 0) batchSize = 16;

    const bool useEarlyStopping = !testDataset.examples.empty();
    const size_t exampleCount = trainDataset.examples.size();

    AdamState embeddingWeightState = AdamState::zerosLike(embedding.weight);

    Matrix bestLayer1Weight = this->layer1.weight;
    Matrix bestLayer1Bias = this->layer1.bias;
    Matrix bestLayer2Weight = this->layer2.weight;
    Matrix bestLayer2Bias = this->layer2.bias;
    Matrix bestEmbeddingWeight = embedding.weight;
    float bestTestAccuracy = -1.0f;
    int checksWithoutImprovement = 0;
    unsigned shuffleSeed = 123u;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        float epochLoss = 0.0f;
        int correct = 0;
        this->dropout.training = true;

        const std::vector<size_t> order = Sequential::shuffledOrder(exampleCount, shuffleSeed);

        for (size_t batchStart = 0; batchStart < exampleCount; batchStart += static_cast<size_t>(batchSize)) {
            size_t batchEnd = batchStart + static_cast<size_t>(batchSize);
            if (batchEnd > exampleCount) batchEnd = exampleCount;
            const float batchCount = static_cast<float>(batchEnd - batchStart);

            Matrix accumulatedWeightGradient1 = Sequential::zerosLike(this->layer1.weight);
            Matrix accumulatedBiasGradient1 = Sequential::zerosLike(this->layer1.bias);
            Matrix accumulatedWeightGradient2 = Sequential::zerosLike(this->layer2.weight);
            Matrix accumulatedBiasGradient2 = Sequential::zerosLike(this->layer2.bias);
            Matrix accumulatedEmbeddingGradient = Sequential::zerosLike(embedding.weight);

            for (size_t orderIndex = batchStart; orderIndex < batchEnd; ++orderIndex) {
                const ClassificationExample& example = trainDataset.examples[order[orderIndex]];

                Matrix embedded = embedding.forward(example.tokenIds);
                Matrix pooled = meanPool.forward(embedded);

                Matrix hidden = ReLU::apply(this->layer1.forward(pooled));
                Matrix dropped = this->dropout.forward(hidden);
                Matrix logits = this->layer2.forward(dropped);
                Matrix probabilities = Softmax::apply(logits);

                epochLoss += CrossEntropy::loss(probabilities, example.target);

                int predicted = 0;
                float bestProbability = probabilities.data[0][0];
                for (size_t classIndex = 1; classIndex < probabilities.data.size(); ++classIndex) {
                    if (probabilities.data[classIndex][0] <= bestProbability) continue;
                    bestProbability = probabilities.data[classIndex][0];
                    predicted = static_cast<int>(classIndex);
                }
                if (predicted == example.label) ++correct;

                Matrix layer2Gradient = CrossEntropy::gradient(probabilities, example.target);
                Matrix weightGradient2 = Matrix::multiply(
                    layer2Gradient,
                    Matrix::transpose(this->layer2.lastInput)
                );

                Matrix inputGradient2 = Matrix::multiply(
                    Matrix::transpose(this->layer2.weight),
                    layer2Gradient
                );
                Matrix hiddenGradient = this->dropout.backward(inputGradient2);
                Matrix layer1Gradient = Matrix::multiplyElementwise(
                    hiddenGradient,
                    ReLU::derivative(this->layer1.lastZ)
                );
                Matrix weightGradient1 = Matrix::multiply(
                    layer1Gradient,
                    Matrix::transpose(this->layer1.lastInput)
                );

                Matrix pooledGradient = Matrix::multiply(
                    Matrix::transpose(this->layer1.weight),
                    layer1Gradient
                );

                Matrix embeddingGradient = meanPool.backward(pooledGradient);
                Matrix embeddingWeightGradient = embedding.backward(embeddingGradient, example.tokenIds);

                Sequential::accumulateGradient(accumulatedWeightGradient2, weightGradient2);
                Sequential::accumulateGradient(accumulatedBiasGradient2, layer2Gradient);
                Sequential::accumulateGradient(accumulatedWeightGradient1, weightGradient1);
                Sequential::accumulateGradient(accumulatedBiasGradient1, layer1Gradient);
                Sequential::accumulateGradient(accumulatedEmbeddingGradient, embeddingWeightGradient);
            }

            const float inverseBatchCount = 1.0f / batchCount;
            this->updateDenseParameters(
                Matrix::scale(accumulatedWeightGradient1, inverseBatchCount),
                Matrix::scale(accumulatedBiasGradient1, inverseBatchCount),
                Matrix::scale(accumulatedWeightGradient2, inverseBatchCount),
                Matrix::scale(accumulatedBiasGradient2, inverseBatchCount)
            );
            this->optimizer.update(
                embedding.weight,
                embeddingWeightState,
                Matrix::scale(accumulatedEmbeddingGradient, inverseBatchCount)
            );
        }

        if (epoch % logEveryEpochs != 0) continue;

        this->dropout.training = false;

        const float averageLoss = epochLoss / static_cast<float>(trainDataset.size());
        const float trainAccuracy = static_cast<float>(correct) / static_cast<float>(trainDataset.size());

        std::cout << "Epoch " << epoch
                  << " | loss: " << averageLoss
                  << " | trainAccuracy: " << trainAccuracy;

        if (!useEarlyStopping) {
            std::cout << '\n';
            continue;
        }

        const float testAccuracy = this->accuracy(embedding, meanPool, testDataset);
        std::cout << " | testAccuracy: " << testAccuracy << '\n';

        if (testAccuracy > bestTestAccuracy) {
            bestTestAccuracy = testAccuracy;
            checksWithoutImprovement = 0;
            bestLayer1Weight = this->layer1.weight;
            bestLayer1Bias = this->layer1.bias;
            bestLayer2Weight = this->layer2.weight;
            bestLayer2Bias = this->layer2.bias;
            bestEmbeddingWeight = embedding.weight;
            continue;
        }

        ++checksWithoutImprovement;
        if (checksWithoutImprovement < earlyStoppingPatience) continue;

        this->layer1.weight = bestLayer1Weight;
        this->layer1.bias = bestLayer1Bias;
        this->layer2.weight = bestLayer2Weight;
        this->layer2.bias = bestLayer2Bias;
        embedding.weight = bestEmbeddingWeight;

        std::cout << "early stop at epoch " << epoch
                  << " | bestTestAccuracy: " << bestTestAccuracy
                  << '\n';
        this->dropout.training = false;
        return;
    }

    this->dropout.training = false;
    if (!useEarlyStopping) return;

    this->layer1.weight = bestLayer1Weight;
    this->layer1.bias = bestLayer1Bias;
    this->layer2.weight = bestLayer2Weight;
    this->layer2.bias = bestLayer2Bias;
    embedding.weight = bestEmbeddingWeight;
}

int Sequential::predictClass(Embedding& embedding, MeanPool& meanPool, const std::vector<int>& tokenIds) {
    Matrix probabilities = this->forward(embedding, meanPool, tokenIds);

    int predicted = 0;
    float bestProbability = probabilities.data[0][0];
    for (size_t classIndex = 1; classIndex < probabilities.data.size(); ++classIndex) {
        if (probabilities.data[classIndex][0] <= bestProbability) continue;
        bestProbability = probabilities.data[classIndex][0];
        predicted = static_cast<int>(classIndex);
    }
    return predicted;
}

float Sequential::accuracy(Embedding& embedding, MeanPool& meanPool, const ClassificationDataset& dataset) {
    if (dataset.examples.empty()) return 0.0f;

    int correct = 0;
    for (const ClassificationExample& example : dataset.examples) {
        const int predicted = this->predictClass(embedding, meanPool, example.tokenIds);
        if (predicted == example.label) ++correct;
    }

    return static_cast<float>(correct) / static_cast<float>(dataset.size());
}
