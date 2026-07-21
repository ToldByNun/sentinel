#include "Adam.hpp"

#include <cmath>
#include <stdexcept>

AdamState AdamState::zerosLike(const Matrix& parameter) {
    AdamState state;
    state.firstMoment = parameter;
    state.secondMoment = parameter;
    for (size_t row = 0; row < parameter.data.size(); ++row) {
        for (size_t column = 0; column < parameter.data[row].size(); ++column) {
            state.firstMoment.data[row][column] = 0.0f;
            state.secondMoment.data[row][column] = 0.0f;
        }
    }
    return state;
}

Adam::Adam(float learningRate, float beta1, float beta2, float epsilon)
    : learningRate(learningRate), beta1(beta1), beta2(beta2), epsilon(epsilon), timeStep(0) {
    if (learningRate <= 0.0f) throw std::invalid_argument("Adam learningRate must be > 0");
    if (beta1 < 0.0f || beta1 >= 1.0f) throw std::invalid_argument("Adam beta1 must be in [0, 1)");
    if (beta2 < 0.0f || beta2 >= 1.0f) throw std::invalid_argument("Adam beta2 must be in [0, 1)");
    if (epsilon <= 0.0f) throw std::invalid_argument("Adam epsilon must be > 0");
}

void Adam::step() {
    ++this->timeStep;
}

Matrix Adam::zerosLike(const Matrix& matrix) {
    Matrix result = matrix;
    for (size_t row = 0; row < result.data.size(); ++row) {
        for (size_t column = 0; column < result.data[row].size(); ++column)
            result.data[row][column] = 0.0f;
    }
    return result;
}

Matrix Adam::squareElements(const Matrix& matrix) {
    Matrix result = matrix;
    for (size_t row = 0; row < result.data.size(); ++row) {
        for (size_t column = 0; column < result.data[row].size(); ++column)
            result.data[row][column] = matrix.data[row][column] * matrix.data[row][column];
    }
    return result;
}

Matrix Adam::squareRootElements(const Matrix& matrix) {
    Matrix result = matrix;
    for (size_t row = 0; row < result.data.size(); ++row) {
        for (size_t column = 0; column < result.data[row].size(); ++column)
            result.data[row][column] = std::sqrt(matrix.data[row][column]);
    }
    return result;
}

Matrix Adam::divideElements(const Matrix& numerator, const Matrix& denominator) {
    if (numerator.data.size() != denominator.data.size() || numerator.data[0].size() != denominator.data[0].size())
        throw std::invalid_argument("Adam::divideElements shape mismatch");

    Matrix result = numerator;
    for (size_t row = 0; row < result.data.size(); ++row) {
        for (size_t column = 0; column < result.data[row].size(); ++column)
            result.data[row][column] = numerator.data[row][column] / denominator.data[row][column];
    }
    return result;
}

void Adam::update(Matrix& parameter, AdamState& state, const Matrix& gradient) const {
    if (this->timeStep <= 0) throw std::invalid_argument("Adam::update requires step() before update");
    if (parameter.data.size() != gradient.data.size() || parameter.data[0].size() != gradient.data[0].size())
        throw std::invalid_argument("Adam::update parameter/gradient shape mismatch");

    if (state.firstMoment.data.empty()) {
        state.firstMoment = Adam::zerosLike(parameter);
        state.secondMoment = Adam::zerosLike(parameter);
    }

    if (state.firstMoment.data.size() != parameter.data.size() || state.firstMoment.data[0].size() != parameter.data[0].size())
        throw std::invalid_argument("Adam::update moment shape mismatch");

    state.firstMoment = Matrix::add(
        Matrix::scale(state.firstMoment, this->beta1),
        Matrix::scale(gradient, 1.0f - this->beta1)
    );
    state.secondMoment = Matrix::add(
        Matrix::scale(state.secondMoment, this->beta2),
        Matrix::scale(Adam::squareElements(gradient), 1.0f - this->beta2)
    );

    const float firstMomentCorrection = 1.0f - std::pow(this->beta1, static_cast<float>(this->timeStep));
    const float secondMomentCorrection = 1.0f - std::pow(this->beta2, static_cast<float>(this->timeStep));

    Matrix correctedFirstMoment = Matrix::scale(state.firstMoment, 1.0f / firstMomentCorrection);
    Matrix correctedSecondMoment = Matrix::scale(state.secondMoment, 1.0f / secondMomentCorrection);

    Matrix denominator = Adam::squareRootElements(correctedSecondMoment);
    for (size_t row = 0; row < denominator.data.size(); ++row) {
        for (size_t column = 0; column < denominator.data[row].size(); ++column)
            denominator.data[row][column] += this->epsilon;
    }

    Matrix update = Matrix::scale(
        Adam::divideElements(correctedFirstMoment, denominator),
        this->learningRate
    );
    parameter = Matrix::subtract(parameter, update);
}
