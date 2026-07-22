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

    const float firstMomentCorrection = 1.0f - std::pow(this->beta1, static_cast<float>(this->timeStep));
    const float secondMomentCorrection = 1.0f - std::pow(this->beta2, static_cast<float>(this->timeStep));
    const float inverseFirstCorrection = 1.0f / firstMomentCorrection;
    const float inverseSecondCorrection = 1.0f / secondMomentCorrection;

    for (size_t row = 0; row < parameter.data.size(); ++row) {
        for (size_t column = 0; column < parameter.data[row].size(); ++column) {
            const float gradientValue = gradient.data[row][column];
            float& firstMoment = state.firstMoment.data[row][column];
            float& secondMoment = state.secondMoment.data[row][column];

            firstMoment = this->beta1 * firstMoment + (1.0f - this->beta1) * gradientValue;
            secondMoment = this->beta2 * secondMoment + (1.0f - this->beta2) * gradientValue * gradientValue;

            const float correctedFirst = firstMoment * inverseFirstCorrection;
            const float correctedSecond = secondMoment * inverseSecondCorrection;
            parameter.data[row][column] -= this->learningRate * correctedFirst / (std::sqrt(correctedSecond) + this->epsilon);
        }
    }
}

void Adam::updateSelectedRows(Matrix& parameter, AdamState& state, const Matrix& gradient, const std::vector<int>& rowIndices) const {
    if (this->timeStep <= 0) throw std::invalid_argument("Adam::updateSelectedRows requires step() before update");
    if (parameter.data.size() != gradient.data.size() || parameter.data[0].size() != gradient.data[0].size())
        throw std::invalid_argument("Adam::updateSelectedRows parameter/gradient shape mismatch");

    if (state.firstMoment.data.empty()) {
        state.firstMoment = Adam::zerosLike(parameter);
        state.secondMoment = Adam::zerosLike(parameter);
    }

    const float firstMomentCorrection = 1.0f - std::pow(this->beta1, static_cast<float>(this->timeStep));
    const float secondMomentCorrection = 1.0f - std::pow(this->beta2, static_cast<float>(this->timeStep));
    const float inverseFirstCorrection = 1.0f / firstMomentCorrection;
    const float inverseSecondCorrection = 1.0f / secondMomentCorrection;
    const size_t columnCount = parameter.data[0].size();

    for (int rowIndex : rowIndices) {
        if (rowIndex < 0 || rowIndex >= static_cast<int>(parameter.data.size()))
            throw std::out_of_range("Adam::updateSelectedRows row out of range");

        const size_t row = static_cast<size_t>(rowIndex);
        for (size_t column = 0; column < columnCount; ++column) {
            const float gradientValue = gradient.data[row][column];
            float& firstMoment = state.firstMoment.data[row][column];
            float& secondMoment = state.secondMoment.data[row][column];

            firstMoment = this->beta1 * firstMoment + (1.0f - this->beta1) * gradientValue;
            secondMoment = this->beta2 * secondMoment + (1.0f - this->beta2) * gradientValue * gradientValue;

            const float correctedFirst = firstMoment * inverseFirstCorrection;
            const float correctedSecond = secondMoment * inverseSecondCorrection;
            parameter.data[row][column] -= this->learningRate * correctedFirst / (std::sqrt(correctedSecond) + this->epsilon);
        }
    }
}
