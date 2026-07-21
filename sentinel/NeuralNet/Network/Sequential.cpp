#include "Sequential.hpp"

#include <iostream>
#include <utility>

#include "../Activations/ReLU.hpp"
#include "../Activations/Softmax.hpp"
#include "../Losses/CrossEntropy.hpp"

Sequential::Sequential(Dense layer1, Dense layer2, SGD optimizer)
    : layer1(std::move(layer1)), layer2(std::move(layer2)), optimizer(optimizer) {}

Matrix Sequential::forward(const Matrix& input) {
    Matrix hidden = ReLU::apply(this->layer1.forward(input));
    Matrix logits = this->layer2.forward(hidden);
    return Softmax::apply(logits);
}

Matrix Sequential::forward(Embedding& embedding, MeanPool& meanPool, const std::vector<int>& tokenIds) {
    Matrix embedded = embedding.forward(tokenIds);
    Matrix pooled = meanPool.forward(embedded);
    return this->forward(pooled);
}

void Sequential::train(const Matrix& input, const Matrix& target, int epochs) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        Matrix hidden = ReLU::apply(this->layer1.forward(input));
        Matrix logits = this->layer2.forward(hidden);
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
        Matrix layer1Gradient = Matrix::multiplyElementwise(
            inputGradient2,
            ReLU::derivative(this->layer1.lastZ)
        );
        Matrix weightGradient1 = Matrix::multiply(
            layer1Gradient,
            Matrix::transpose(this->layer1.lastInput)
        );

        this->optimizer.update(this->layer2.weight, weightGradient2);
        this->optimizer.update(this->layer2.bias, layer2Gradient);

        this->optimizer.update(this->layer1.weight, weightGradient1);
        this->optimizer.update(this->layer1.bias, layer1Gradient);

        if (epoch % 1000 == 0) {
            const float loss = CrossEntropy::loss(probabilities, target);
            std::cout << "Epoch " << epoch << " | loss: " << loss << '\n';
        }
    }
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
    if (dataset.examples.empty()) return;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        float epochLoss = 0.0f;
        int correct = 0;

        for (const ClassificationExample& example : dataset.examples) {
            Matrix embedded = embedding.forward(example.tokenIds);
            Matrix pooled = meanPool.forward(embedded);

            Matrix hidden = ReLU::apply(this->layer1.forward(pooled));
            Matrix logits = this->layer2.forward(hidden);
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
            Matrix layer1Gradient = Matrix::multiplyElementwise(
                inputGradient2,
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
            Matrix embeddingWeightGradient = embedding.backward(embeddingGradient);

            this->optimizer.update(this->layer2.weight, weightGradient2);
            this->optimizer.update(this->layer2.bias, layer2Gradient);

            this->optimizer.update(this->layer1.weight, weightGradient1);
            this->optimizer.update(this->layer1.bias, layer1Gradient);

            this->optimizer.update(embedding.weight, embeddingWeightGradient);
        }

        if (epoch % 500 == 0) {
            const float averageLoss = epochLoss / static_cast<float>(dataset.size());
            const float accuracy = static_cast<float>(correct) / static_cast<float>(dataset.size());
            std::cout << "Epoch " << epoch
                      << " | loss: " << averageLoss
                      << " | accuracy: " << accuracy
                      << '\n';
        }
    }
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
