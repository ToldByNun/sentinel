#include "Dense.hpp"

#include <utility>

Dense::Dense(Matrix weight, Matrix bias)
    : weight(std::move(weight)), bias(std::move(bias)) {}

Matrix Dense::forward(const Matrix& input) {
    this->lastInput = input;
    this->lastZ = Matrix::add(Matrix::multiply(this->weight, input), this->bias);
    return this->lastZ;
}
