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

        Matrix loss = MSE::compute(prediction, target);

        Matrix delta2 = Matrix::multiplyElementwise(loss, ReLU::derivative(this->layer2.lastZ));
        Matrix dW2 = Matrix::multiply(delta2, Matrix::transpose(this->layer2.lastInput));

        Matrix wt_delta2 = Matrix::multiply(Matrix::transpose(this->layer2.weight), delta2);
        Matrix delta1 = Matrix::multiplyElementwise(wt_delta2, ReLU::derivative(this->layer1.lastZ));
        Matrix dW1 = Matrix::multiply(delta1, Matrix::transpose(this->layer1.lastInput));

        this->optimizer.update(this->layer2.weight, dW2);
        this->optimizer.update(this->layer2.bias, delta2);

        this->optimizer.update(this->layer1.weight, dW1);
        this->optimizer.update(this->layer1.bias, delta1);

        if (epoch % 1000 == 0) {
            std::cout << "Epoch " << epoch << " | Prediction[0]: " << prediction.data[0][0] << '\n';
        }
    }
}
