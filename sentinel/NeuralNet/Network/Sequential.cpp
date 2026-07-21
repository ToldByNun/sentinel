#include "Sequential.hpp"

#include <iostream>
#include <utility>

#include "../Activations/ReLU.hpp"
#include "../Losses/MSE.hpp"

Sequential::Sequential(Dense layer1, Dense layer2, SGD optimizer)
    : layer1(std::move(layer1)), layer2(std::move(layer2)), optimizer(optimizer) {}

Matrix Sequential::forward(const Matrix& input) {
    Matrix hidden = ReLU::apply(this->layer1.forward(input));
    return ReLU::apply(this->layer2.forward(hidden));
}

void Sequential::train(const Matrix& input, const Matrix& target, int epochs) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        Matrix hidden = ReLU::apply(this->layer1.forward(input));
        Matrix prediction = ReLU::apply(this->layer2.forward(hidden));

        Matrix lossGradient = MSE::gradient(prediction, target);

        Matrix layer2Gradient = Matrix::multiplyElementwise(
            lossGradient,
            ReLU::derivative(this->layer2.lastZ)
        );
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
            std::cout << "Epoch " << epoch << " | Prediction[0]: " << prediction.data[0][0] << '\n';
        }
    }
}

void Sequential::train(
    Embedding& embedding,
    MeanPool& meanPool,
    const std::vector<int>& tokenIds,
    const Matrix& target,
    int epochs
) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        Matrix embedded = embedding.forward(tokenIds);
        Matrix pooled = meanPool.forward(embedded);

        Matrix hidden = ReLU::apply(this->layer1.forward(pooled));
        Matrix prediction = ReLU::apply(this->layer2.forward(hidden));

        Matrix lossGradient = MSE::gradient(prediction, target);

        Matrix layer2Gradient = Matrix::multiplyElementwise(
            lossGradient,
            ReLU::derivative(this->layer2.lastZ)
        );
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

        if (epoch % 1000 == 0) {
            std::cout << "Epoch " << epoch << " | Prediction[0]: " << prediction.data[0][0] << '\n';
        }
    }
}
