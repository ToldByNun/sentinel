#include "Dense.hpp"

#include <stdexcept>
#include <utility>

Dense::Dense(Matrix weight, Matrix bias)
    : weight(std::move(weight)), bias(std::move(bias)) {
    if (this->weight.empty() || this->bias.empty()) throw std::invalid_argument("Dense weight/bias must be non-empty");
    if (this->weight.rows != this->bias.rows) throw std::invalid_argument("Dense bias rows must match weight rows");
    if (this->bias.cols != 1) throw std::invalid_argument("Dense bias must be a column vector");
}

Matrix Dense::forward(const Matrix& input) {
    if (input.empty()) throw std::invalid_argument("Dense::forward empty input");
    if (this->weight.cols != input.rows) throw std::invalid_argument("Dense::forward shape mismatch");

    this->lastInput = input;
    Matrix product = Matrix::multiply(this->weight, input);

    // broadcast bias across sequence columns
    for (size_t row = 0; row < product.rows; ++row) {
        const float biasValue = this->bias.at(row, 0);
        for (size_t column = 0; column < product.cols; ++column)
            product.at(row, column) += biasValue;
    }

    this->lastZ = product;
    return this->lastZ;
}
