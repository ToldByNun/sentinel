#include "Adam.hpp"

#include <cmath>
#include <stdexcept>

AdamState AdamState::zerosLike(const Matrix& parameter) {
    AdamState state;
    state.firstMoment = Matrix::zerosLike(parameter);
    state.secondMoment = Matrix::zerosLike(parameter);
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
    return Matrix::zerosLike(matrix);
}

Matrix Adam::squareElements(const Matrix& matrix) {
    Matrix result = matrix;
    for (size_t row = 0; row < result.rows; ++row) {
        for (size_t column = 0; column < result.cols; ++column) {
            const float value = matrix.at(row, column);
            result.at(row, column) = value * value;
        }
    }
    return result;
}

Matrix Adam::squareRootElements(const Matrix& matrix) {
    Matrix result = matrix;
    for (size_t row = 0; row < result.rows; ++row) {
        for (size_t column = 0; column < result.cols; ++column)
            result.at(row, column) = std::sqrt(matrix.at(row, column));
    }
    return result;
}

Matrix Adam::divideElements(const Matrix& numerator, const Matrix& denominator) {
    if (numerator.rows != denominator.rows || numerator.cols != denominator.cols)
        throw std::invalid_argument("Adam::divideElements shape mismatch");

    Matrix result = numerator;
    for (size_t row = 0; row < result.rows; ++row) {
        for (size_t column = 0; column < result.cols; ++column)
            result.at(row, column) = numerator.at(row, column) / denominator.at(row, column);
    }
    return result;
}

void Adam::update(Matrix& parameter, AdamState& state, const Matrix& gradient) const {
    if (this->timeStep <= 0) throw std::invalid_argument("Adam::update requires step() before update");
    if (parameter.rows != gradient.rows || parameter.cols != gradient.cols)
        throw std::invalid_argument("Adam::update parameter/gradient shape mismatch");

    if (state.firstMoment.empty()) {
        state.firstMoment = Adam::zerosLike(parameter);
        state.secondMoment = Adam::zerosLike(parameter);
    }

    if (state.firstMoment.rows != parameter.rows || state.firstMoment.cols != parameter.cols)
        throw std::invalid_argument("Adam::update moment shape mismatch");

    const float firstMomentCorrection = 1.0f - std::pow(this->beta1, static_cast<float>(this->timeStep));
    const float secondMomentCorrection = 1.0f - std::pow(this->beta2, static_cast<float>(this->timeStep));
    const float inverseFirstCorrection = 1.0f / firstMomentCorrection;
    const float inverseSecondCorrection = 1.0f / secondMomentCorrection;

    for (size_t row = 0; row < parameter.rows; ++row) {
        for (size_t column = 0; column < parameter.cols; ++column) {
            const float gradientValue = gradient.at(row, column);
            float& firstMoment = state.firstMoment.at(row, column);
            float& secondMoment = state.secondMoment.at(row, column);

            firstMoment = this->beta1 * firstMoment + (1.0f - this->beta1) * gradientValue;
            secondMoment = this->beta2 * secondMoment + (1.0f - this->beta2) * gradientValue * gradientValue;

            const float correctedFirst = firstMoment * inverseFirstCorrection;
            const float correctedSecond = secondMoment * inverseSecondCorrection;
            parameter.at(row, column) -= this->learningRate * correctedFirst / (std::sqrt(correctedSecond) + this->epsilon);
        }
    }
}

void Adam::updateSelectedRows(Matrix& parameter, AdamState& state, const Matrix& gradient, const std::vector<int>& rowIndices) const {
    if (this->timeStep <= 0) throw std::invalid_argument("Adam::updateSelectedRows requires step() before update");
    if (parameter.rows != gradient.rows || parameter.cols != gradient.cols)
        throw std::invalid_argument("Adam::updateSelectedRows parameter/gradient shape mismatch");

    if (state.firstMoment.empty()) {
        state.firstMoment = Adam::zerosLike(parameter);
        state.secondMoment = Adam::zerosLike(parameter);
    }

    const float firstMomentCorrection = 1.0f - std::pow(this->beta1, static_cast<float>(this->timeStep));
    const float secondMomentCorrection = 1.0f - std::pow(this->beta2, static_cast<float>(this->timeStep));
    const float inverseFirstCorrection = 1.0f / firstMomentCorrection;
    const float inverseSecondCorrection = 1.0f / secondMomentCorrection;
    const size_t columnCount = parameter.cols;

    for (int rowIndex : rowIndices) {
        if (rowIndex < 0 || rowIndex >= static_cast<int>(parameter.rows))
            throw std::out_of_range("Adam::updateSelectedRows row out of range");

        const size_t row = static_cast<size_t>(rowIndex);
        for (size_t column = 0; column < columnCount; ++column) {
            const float gradientValue = gradient.at(row, column);
            float& firstMoment = state.firstMoment.at(row, column);
            float& secondMoment = state.secondMoment.at(row, column);

            firstMoment = this->beta1 * firstMoment + (1.0f - this->beta1) * gradientValue;
            secondMoment = this->beta2 * secondMoment + (1.0f - this->beta2) * gradientValue * gradientValue;

            const float correctedFirst = firstMoment * inverseFirstCorrection;
            const float correctedSecond = secondMoment * inverseSecondCorrection;
            parameter.at(row, column) -= this->learningRate * correctedFirst / (std::sqrt(correctedSecond) + this->epsilon);
        }
    }
}
