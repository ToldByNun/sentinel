#include "LayerNorm.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

LayerNorm::LayerNorm(int embeddingDim, float epsilon) : epsilon(epsilon) {
    if (embeddingDim <= 0) throw std::invalid_argument("LayerNorm embeddingDim must be > 0");
    if (epsilon <= 0.0f) throw std::invalid_argument("LayerNorm epsilon must be > 0");

    this->gamma = Matrix(static_cast<size_t>(embeddingDim), 1, 1.0f);
    this->beta = Matrix(static_cast<size_t>(embeddingDim), 1, 0.0f);
}

Matrix LayerNorm::forward(const Matrix& input, LayerNormCache& cache) const {
    if (input.empty()) throw std::invalid_argument("LayerNorm::forward empty input");
    if (input.rows != this->gamma.rows) throw std::invalid_argument("LayerNorm::forward embedding dim mismatch");

    const size_t embeddingDim = input.rows;
    const size_t sequenceLength = input.cols;

    cache.input = input;
    cache.mean.assign(sequenceLength, 0.0f);
    cache.inverseStd.assign(sequenceLength, 0.0f);
    cache.normalized = Matrix(embeddingDim, sequenceLength, 0.0f);

    Matrix output(embeddingDim, sequenceLength, 0.0f);

    const float inverseDim = 1.0f / static_cast<float>(embeddingDim);

    for (size_t column = 0; column < sequenceLength; ++column) {
        float sum = 0.0f;
        for (size_t row = 0; row < embeddingDim; ++row)
            sum += input.at(row, column);
        cache.mean[column] = sum * inverseDim;

        float varianceSum = 0.0f;
        for (size_t row = 0; row < embeddingDim; ++row) {
            const float centered = input.at(row, column) - cache.mean[column];
            varianceSum += centered * centered;
        }
        cache.inverseStd[column] = 1.0f / std::sqrt(varianceSum * inverseDim + this->epsilon);

        for (size_t row = 0; row < embeddingDim; ++row) {
            const float normalized = (input.at(row, column) - cache.mean[column]) * cache.inverseStd[column];
            cache.normalized.at(row, column) = normalized;
            output.at(row, column) = this->gamma.at(row, 0) * normalized + this->beta.at(row, 0);
        }
    }

    return output;
}

Matrix LayerNorm::backward(const Matrix& outputGradient, const LayerNormCache& cache, Matrix& gammaGradient, Matrix& betaGradient) const {
    if (cache.input.empty()) throw std::logic_error("LayerNorm::backward called before forward");
    if (outputGradient.rows != cache.input.rows || outputGradient.cols != cache.input.cols)
        throw std::invalid_argument("LayerNorm::backward shape mismatch");

    const size_t embeddingDim = cache.input.rows;
    const size_t sequenceLength = cache.input.cols;
    const float dimension = static_cast<float>(embeddingDim);

    gammaGradient = Matrix::zerosLike(this->gamma);
    betaGradient = Matrix::zerosLike(this->beta);

    Matrix inputGradient(embeddingDim, sequenceLength, 0.0f);

    for (size_t column = 0; column < sequenceLength; ++column) {
        float sumGradient = 0.0f;
        float sumGradientNormalized = 0.0f;

        for (size_t row = 0; row < embeddingDim; ++row) {
            const float outputGrad = outputGradient.at(row, column);
            const float normalized = cache.normalized.at(row, column);
            gammaGradient.at(row, 0) += outputGrad * normalized;
            betaGradient.at(row, 0) += outputGrad;

            const float normalizedGradient = outputGrad * this->gamma.at(row, 0);
            sumGradient += normalizedGradient;
            sumGradientNormalized += normalizedGradient * normalized;
        }

        for (size_t row = 0; row < embeddingDim; ++row) {
            const float normalized = cache.normalized.at(row, column);
            const float normalizedGradient = outputGradient.at(row, column) * this->gamma.at(row, 0);
            inputGradient.at(row, column) =
                (1.0f / dimension) * cache.inverseStd[column] *
                (dimension * normalizedGradient - sumGradient - normalized * sumGradientNormalized);
        }
    }

    return inputGradient;
}
