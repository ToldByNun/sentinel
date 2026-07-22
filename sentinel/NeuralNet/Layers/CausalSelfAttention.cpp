#include "CausalSelfAttention.hpp"

#include "../Activations/Softmax.hpp"
#include "../Initializers/UniformInit.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

CausalSelfAttention::CausalSelfAttention(Matrix queryWeight, Matrix keyWeight, Matrix valueWeight, Matrix outputWeight)
    : queryWeight(std::move(queryWeight)),
      keyWeight(std::move(keyWeight)),
      valueWeight(std::move(valueWeight)),
      outputWeight(std::move(outputWeight)) {}

CausalSelfAttention CausalSelfAttention::create(int embeddingDim, unsigned seed) {
    if (embeddingDim <= 0) throw std::invalid_argument("CausalSelfAttention::create embeddingDim must be > 0");

    return CausalSelfAttention(
        UniformInit::matrix(embeddingDim, embeddingDim, 0.1f, seed),
        UniformInit::matrix(embeddingDim, embeddingDim, 0.1f, seed + 1u),
        UniformInit::matrix(embeddingDim, embeddingDim, 0.1f, seed + 2u),
        UniformInit::matrix(embeddingDim, embeddingDim, 0.1f, seed + 3u)
    );
}

Matrix CausalSelfAttention::zerosLike(const Matrix& matrix) {
    Matrix result = matrix;
    for (size_t row = 0; row < result.data.size(); ++row) {
        for (size_t column = 0; column < result.data[row].size(); ++column)
            result.data[row][column] = 0.0f;
    }
    return result;
}

Matrix CausalSelfAttention::forward(const Matrix& input, CausalSelfAttentionCache& cache) const {
    if (input.data.empty() || input.data[0].empty()) throw std::invalid_argument("CausalSelfAttention::forward empty input");
    if (this->queryWeight.data[0].size() != input.data.size())
        throw std::invalid_argument("CausalSelfAttention::forward embedding dim mismatch");

    cache.input = input;
    cache.query = Matrix::multiply(this->queryWeight, input);
    cache.key = Matrix::multiply(this->keyWeight, input);
    cache.value = Matrix::multiply(this->valueWeight, input);

    const size_t sequenceLength = input.data[0].size();
    const float scale = 1.0f / std::sqrt(static_cast<float>(input.data.size()));

    cache.scores = Matrix::scale(
        Matrix::multiply(Matrix::transpose(cache.key), cache.query),
        scale
    );

    for (size_t keyIndex = 0; keyIndex < sequenceLength; ++keyIndex) {
        for (size_t queryIndex = 0; queryIndex < sequenceLength; ++queryIndex) {
            if (keyIndex <= queryIndex) continue;
            cache.scores.data[keyIndex][queryIndex] = -1e9f;
        }
    }

    cache.probabilities = Softmax::apply(cache.scores);
    cache.attended = Matrix::multiply(cache.value, cache.probabilities);
    return Matrix::multiply(this->outputWeight, cache.attended);
}

Matrix CausalSelfAttention::softmaxBackward(const Matrix& probabilities, const Matrix& probabilityGradient) {
    if (probabilities.data.size() != probabilityGradient.data.size() || probabilities.data[0].size() != probabilityGradient.data[0].size())
        throw std::invalid_argument("CausalSelfAttention::softmaxBackward shape mismatch");

    Matrix scoreGradient = CausalSelfAttention::zerosLike(probabilities);
    const size_t keyCount = probabilities.data.size();
    const size_t queryCount = probabilities.data[0].size();

    for (size_t queryIndex = 0; queryIndex < queryCount; ++queryIndex) {
        float dot = 0.0f;
        for (size_t keyIndex = 0; keyIndex < keyCount; ++keyIndex)
            dot += probabilityGradient.data[keyIndex][queryIndex] * probabilities.data[keyIndex][queryIndex];

        for (size_t keyIndex = 0; keyIndex < keyCount; ++keyIndex) {
            scoreGradient.data[keyIndex][queryIndex] =
                probabilities.data[keyIndex][queryIndex] *
                (probabilityGradient.data[keyIndex][queryIndex] - dot);
        }
    }

    return scoreGradient;
}

Matrix CausalSelfAttention::backward(const Matrix& outputGradient, const CausalSelfAttentionCache& cache, Matrix& queryWeightGradient, Matrix& keyWeightGradient, Matrix& valueWeightGradient, Matrix& outputWeightGradient) const {
    if (cache.input.data.empty()) throw std::logic_error("CausalSelfAttention::backward called before forward");
    if (outputGradient.data.size() != this->outputWeight.data.size() || outputGradient.data[0].size() != cache.attended.data[0].size())
        throw std::invalid_argument("CausalSelfAttention::backward output gradient shape mismatch");

    outputWeightGradient = Matrix::multiply(outputGradient, Matrix::transpose(cache.attended));
    Matrix attendedGradient = Matrix::multiply(Matrix::transpose(this->outputWeight), outputGradient);

    Matrix valueGradient = Matrix::multiply(attendedGradient, Matrix::transpose(cache.probabilities));
    Matrix probabilityGradient = Matrix::multiply(Matrix::transpose(cache.value), attendedGradient);

    Matrix scoreGradient = CausalSelfAttention::softmaxBackward(cache.probabilities, probabilityGradient);

    const size_t sequenceLength = cache.input.data[0].size();
    for (size_t keyIndex = 0; keyIndex < sequenceLength; ++keyIndex) {
        for (size_t queryIndex = 0; queryIndex < sequenceLength; ++queryIndex) {
            if (keyIndex <= queryIndex) continue;
            scoreGradient.data[keyIndex][queryIndex] = 0.0f;
        }
    }

    const float scale = 1.0f / std::sqrt(static_cast<float>(cache.input.data.size()));
    scoreGradient = Matrix::scale(scoreGradient, scale);

    Matrix queryGradient = Matrix::multiply(cache.key, scoreGradient);
    Matrix keyGradient = Matrix::multiply(cache.query, Matrix::transpose(scoreGradient));

    queryWeightGradient = Matrix::multiply(queryGradient, Matrix::transpose(cache.input));
    keyWeightGradient = Matrix::multiply(keyGradient, Matrix::transpose(cache.input));
    valueWeightGradient = Matrix::multiply(valueGradient, Matrix::transpose(cache.input));

    Matrix inputGradient = Matrix::multiply(Matrix::transpose(this->queryWeight), queryGradient);
    inputGradient = Matrix::add(inputGradient, Matrix::multiply(Matrix::transpose(this->keyWeight), keyGradient));
    inputGradient = Matrix::add(inputGradient, Matrix::multiply(Matrix::transpose(this->valueWeight), valueGradient));
    return inputGradient;
}
