#include "LayerNorm.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

LayerNorm::LayerNorm(int embeddingDim, float epsilon) : epsilon(epsilon) {
    if (embeddingDim <= 0) throw std::invalid_argument("LayerNorm embeddingDim must be > 0");
    if (epsilon <= 0.0f) throw std::invalid_argument("LayerNorm epsilon must be > 0");

    this->gamma.data = std::vector<std::vector<float>>(static_cast<size_t>(embeddingDim), std::vector<float>(1, 1.0f));
    this->beta.data = std::vector<std::vector<float>>(static_cast<size_t>(embeddingDim), std::vector<float>(1, 0.0f));
}

Matrix LayerNorm::forward(const Matrix& input, LayerNormCache& cache) const {
    if (input.data.empty() || input.data[0].empty()) throw std::invalid_argument("LayerNorm::forward empty input");
    if (input.data.size() != this->gamma.data.size()) throw std::invalid_argument("LayerNorm::forward embedding dim mismatch");

    const size_t embeddingDim = input.data.size();
    const size_t sequenceLength = input.data[0].size();

    cache.input = input;
    cache.mean.assign(sequenceLength, 0.0f);
    cache.inverseStd.assign(sequenceLength, 0.0f);
    cache.normalized.data = std::vector<std::vector<float>>(embeddingDim, std::vector<float>(sequenceLength, 0.0f));

    Matrix output;
    output.data = std::vector<std::vector<float>>(embeddingDim, std::vector<float>(sequenceLength, 0.0f));

    const float inverseDim = 1.0f / static_cast<float>(embeddingDim);

    for (size_t column = 0; column < sequenceLength; ++column) {
        float sum = 0.0f;
        for (size_t row = 0; row < embeddingDim; ++row)
            sum += input.data[row][column];
        cache.mean[column] = sum * inverseDim;

        float varianceSum = 0.0f;
        for (size_t row = 0; row < embeddingDim; ++row) {
            const float centered = input.data[row][column] - cache.mean[column];
            varianceSum += centered * centered;
        }
        cache.inverseStd[column] = 1.0f / std::sqrt(varianceSum * inverseDim + this->epsilon);

        for (size_t row = 0; row < embeddingDim; ++row) {
            const float normalized = (input.data[row][column] - cache.mean[column]) * cache.inverseStd[column];
            cache.normalized.data[row][column] = normalized;
            output.data[row][column] = this->gamma.data[row][0] * normalized + this->beta.data[row][0];
        }
    }

    return output;
}

Matrix LayerNorm::backward(const Matrix& outputGradient, const LayerNormCache& cache, Matrix& gammaGradient, Matrix& betaGradient) const {
    if (cache.input.data.empty()) throw std::logic_error("LayerNorm::backward called before forward");
    if (outputGradient.data.size() != cache.input.data.size() || outputGradient.data[0].size() != cache.input.data[0].size())
        throw std::invalid_argument("LayerNorm::backward shape mismatch");

    const size_t embeddingDim = cache.input.data.size();
    const size_t sequenceLength = cache.input.data[0].size();
    const float dimension = static_cast<float>(embeddingDim);

    gammaGradient = Matrix::zerosLike(this->gamma);
    betaGradient = Matrix::zerosLike(this->beta);

    Matrix inputGradient;
    inputGradient.data = std::vector<std::vector<float>>(embeddingDim, std::vector<float>(sequenceLength, 0.0f));

    for (size_t column = 0; column < sequenceLength; ++column) {
        float sumGradient = 0.0f;
        float sumGradientNormalized = 0.0f;

        for (size_t row = 0; row < embeddingDim; ++row) {
            const float outputGrad = outputGradient.data[row][column];
            const float normalized = cache.normalized.data[row][column];
            gammaGradient.data[row][0] += outputGrad * normalized;
            betaGradient.data[row][0] += outputGrad;

            const float normalizedGradient = outputGrad * this->gamma.data[row][0];
            sumGradient += normalizedGradient;
            sumGradientNormalized += normalizedGradient * normalized;
        }

        for (size_t row = 0; row < embeddingDim; ++row) {
            const float normalized = cache.normalized.data[row][column];
            const float normalizedGradient = outputGradient.data[row][column] * this->gamma.data[row][0];
            inputGradient.data[row][column] =
                (1.0f / dimension) * cache.inverseStd[column] *
                (dimension * normalizedGradient - sumGradient - normalized * sumGradientNormalized);
        }
    }

    return inputGradient;
}
