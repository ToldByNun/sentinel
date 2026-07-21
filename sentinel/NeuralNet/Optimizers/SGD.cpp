#include "SGD.hpp"

SGD::SGD(float learningRate) : learningRate(learningRate) {}

void SGD::update(Matrix& parameter, const Matrix& gradient) const {
    parameter = Matrix::subtract(parameter, Matrix::scale(gradient, this->learningRate));
}
