#include "RMSNorm.hpp"

#include <cmath>
#include <stdexcept>

RMSNorm::RMSNorm(int embeddingDim, float epsilon) : epsilon(epsilon) {
    if (embeddingDim <= 0) throw std::invalid_argument("RMSNorm embeddingDim must be > 0");
    if (epsilon <= 0.0f) throw std::invalid_argument("RMSNorm epsilon must be > 0");

    this->gamma = Matrix(static_cast<size_t>(embeddingDim), 1, 1.0f);
}

Matrix RMSNorm::forward(const Matrix& input, RMSNormCache& cache) const {
    if (input.empty()) throw std::invalid_argument("RMSNorm::forward empty input");
    if (input.rows != this->gamma.rows) throw std::invalid_argument("RMSNorm::forward embedding dim mismatch");

    const size_t embeddingDim = input.rows;
    const size_t sequenceLength = input.cols;
    const float inverseDim = 1.0f / static_cast<float>(embeddingDim);

    cache.input = input;
    cache.inverseRms.assign(sequenceLength, 0.0f);
    cache.normalized.ensureSize(embeddingDim, sequenceLength);

    Matrix output;
    output.ensureSize(embeddingDim, sequenceLength);

    for (size_t column = 0; column < sequenceLength; ++column) {
        float squareSum = 0.0f;
        for (size_t row = 0; row < embeddingDim; ++row) {
            const float value = input.at(row, column);
            squareSum += value * value;
        }
        cache.inverseRms[column] = 1.0f / std::sqrt(squareSum * inverseDim + this->epsilon);

        for (size_t row = 0; row < embeddingDim; ++row) {
            const float normalized = input.at(row, column) * cache.inverseRms[column];
            cache.normalized.at(row, column) = normalized;
            output.at(row, column) = this->gamma.at(row, 0) * normalized;
        }
    }

    return output;
}

Matrix RMSNorm::backward(const Matrix& outputGradient, const RMSNormCache& cache, Matrix& gammaGradient) const {
    if (cache.input.empty()) throw std::logic_error("RMSNorm::backward called before forward");
    if (outputGradient.rows != cache.input.rows || outputGradient.cols != cache.input.cols)
        throw std::invalid_argument("RMSNorm::backward shape mismatch");

    const size_t embeddingDim = cache.input.rows;
    const size_t sequenceLength = cache.input.cols;
    const float dimension = static_cast<float>(embeddingDim);

    gammaGradient = Matrix::zerosLike(this->gamma);

    Matrix inputGradient;
    inputGradient.ensureSize(embeddingDim, sequenceLength);

    for (size_t column = 0; column < sequenceLength; ++column) {
        float sumGradientNormalized = 0.0f;

        for (size_t row = 0; row < embeddingDim; ++row) {
            const float outputGrad = outputGradient.at(row, column);
            const float normalized = cache.normalized.at(row, column);
            gammaGradient.at(row, 0) += outputGrad * normalized;

            const float normalizedGradient = outputGrad * this->gamma.at(row, 0);
            sumGradientNormalized += normalizedGradient * normalized;
        }

        for (size_t row = 0; row < embeddingDim; ++row) {
            const float normalized = cache.normalized.at(row, column);
            const float normalizedGradient = outputGradient.at(row, column) * this->gamma.at(row, 0);
            inputGradient.at(row, column) =
                (1.0f / dimension) * cache.inverseRms[column] *
                (dimension * normalizedGradient - normalized * sumGradientNormalized);
        }
    }

    return inputGradient;
}
