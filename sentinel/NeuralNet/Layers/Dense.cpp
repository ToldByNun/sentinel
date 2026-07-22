#include "Dense.hpp"

#include <stdexcept>
#include <utility>

Dense::Dense(Matrix weight, Matrix bias)
    : weight(std::move(weight)), bias(std::move(bias)) {
    if (this->weight.data.empty() || this->bias.data.empty()) throw std::invalid_argument("Dense weight/bias must be non-empty");
    if (this->weight.data.size() != this->bias.data.size()) throw std::invalid_argument("Dense bias rows must match weight rows");
    if (this->bias.data[0].size() != 1) throw std::invalid_argument("Dense bias must be a column vector");
}

Matrix Dense::forward(const Matrix& input) {
    if (input.data.empty() || input.data[0].empty()) throw std::invalid_argument("Dense::forward empty input");
    if (this->weight.data[0].size() != input.data.size()) throw std::invalid_argument("Dense::forward shape mismatch");

    this->lastInput = input;
    Matrix product = Matrix::multiply(this->weight, input);

    // broadcast bias across sequence columns
    for (size_t row = 0; row < product.data.size(); ++row) {
        const float biasValue = this->bias.data[row][0];
        for (size_t column = 0; column < product.data[row].size(); ++column)
            product.data[row][column] += biasValue;
    }

    this->lastZ = product;
    return this->lastZ;
}
