#include "CausalSelfAttention.hpp"

#include "../Activations/Softmax.hpp"
#include "../Initializers/UniformInit.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

CausalSelfAttention::CausalSelfAttention(Matrix queryWeight, Matrix keyWeight, Matrix valueWeight, Matrix outputWeight, int headCount)
    : queryWeight(std::move(queryWeight)),
      keyWeight(std::move(keyWeight)),
      valueWeight(std::move(valueWeight)),
      outputWeight(std::move(outputWeight)),
      headCount(headCount),
      headDimension(0) {
    if (this->headCount <= 0) throw std::invalid_argument("CausalSelfAttention headCount must be > 0");
    if (this->queryWeight.data.empty()) throw std::invalid_argument("CausalSelfAttention empty weights");
    if (static_cast<int>(this->queryWeight.data.size()) % this->headCount != 0)
        throw std::invalid_argument("CausalSelfAttention embeddingDim must be divisible by headCount");
    this->headDimension = static_cast<int>(this->queryWeight.data.size()) / this->headCount;
}

CausalSelfAttention CausalSelfAttention::create(int embeddingDim, int headCount, unsigned seed) {
    if (embeddingDim <= 0) throw std::invalid_argument("CausalSelfAttention::create embeddingDim must be > 0");
    if (headCount <= 0) throw std::invalid_argument("CausalSelfAttention::create headCount must be > 0");
    if (embeddingDim % headCount != 0) throw std::invalid_argument("CausalSelfAttention::create embeddingDim must be divisible by headCount");

    return CausalSelfAttention(
        UniformInit::matrix(embeddingDim, embeddingDim, 0.1f, seed),
        UniformInit::matrix(embeddingDim, embeddingDim, 0.1f, seed + 1u),
        UniformInit::matrix(embeddingDim, embeddingDim, 0.1f, seed + 2u),
        UniformInit::matrix(embeddingDim, embeddingDim, 0.1f, seed + 3u),
        headCount
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

Matrix CausalSelfAttention::extractHead(const Matrix& full, int headIndex, int headDimension) {
    Matrix head;
    head.data.resize(static_cast<size_t>(headDimension));
    const size_t rowOffset = static_cast<size_t>(headIndex * headDimension);
    for (int row = 0; row < headDimension; ++row)
        head.data[static_cast<size_t>(row)] = full.data[rowOffset + static_cast<size_t>(row)];
    return head;
}

void CausalSelfAttention::writeHead(Matrix& full, int headIndex, int headDimension, const Matrix& head) {
    const size_t rowOffset = static_cast<size_t>(headIndex * headDimension);
    for (int row = 0; row < headDimension; ++row)
        full.data[rowOffset + static_cast<size_t>(row)] = head.data[static_cast<size_t>(row)];
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
    const float scale = 1.0f / std::sqrt(static_cast<float>(this->headDimension));

    cache.scores.assign(static_cast<size_t>(this->headCount), Matrix());
    cache.probabilities.assign(static_cast<size_t>(this->headCount), Matrix());
    cache.attended.data = std::vector<std::vector<float>>(cache.query.data.size(), std::vector<float>(sequenceLength, 0.0f));

    for (int headIndex = 0; headIndex < this->headCount; ++headIndex) {
        Matrix queryHead = CausalSelfAttention::extractHead(cache.query, headIndex, this->headDimension);
        Matrix keyHead = CausalSelfAttention::extractHead(cache.key, headIndex, this->headDimension);
        Matrix valueHead = CausalSelfAttention::extractHead(cache.value, headIndex, this->headDimension);

        Matrix scores = Matrix::scale(Matrix::multiply(Matrix::transpose(keyHead), queryHead), scale);
        for (size_t keyIndex = 0; keyIndex < sequenceLength; ++keyIndex) {
            for (size_t queryIndex = 0; queryIndex < sequenceLength; ++queryIndex) {
                if (keyIndex <= queryIndex) continue;
                scores.data[keyIndex][queryIndex] = -1e9f;
            }
        }

        Matrix probabilities = Softmax::apply(scores);
        Matrix attendedHead = Matrix::multiply(valueHead, probabilities);
        CausalSelfAttention::writeHead(cache.attended, headIndex, this->headDimension, attendedHead);

        cache.scores[static_cast<size_t>(headIndex)] = std::move(scores);
        cache.probabilities[static_cast<size_t>(headIndex)] = std::move(probabilities);
    }

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
    if (static_cast<int>(cache.scores.size()) != this->headCount || static_cast<int>(cache.probabilities.size()) != this->headCount)
        throw std::invalid_argument("CausalSelfAttention::backward head cache size mismatch");

    outputWeightGradient = Matrix::multiply(outputGradient, Matrix::transpose(cache.attended));
    Matrix attendedGradient = Matrix::multiply(Matrix::transpose(this->outputWeight), outputGradient);

    Matrix queryGradient = CausalSelfAttention::zerosLike(cache.query);
    Matrix keyGradient = CausalSelfAttention::zerosLike(cache.key);
    Matrix valueGradient = CausalSelfAttention::zerosLike(cache.value);

    const size_t sequenceLength = cache.input.data[0].size();
    const float scale = 1.0f / std::sqrt(static_cast<float>(this->headDimension));

    for (int headIndex = 0; headIndex < this->headCount; ++headIndex) {
        Matrix attendedHeadGradient = CausalSelfAttention::extractHead(attendedGradient, headIndex, this->headDimension);
        Matrix queryHead = CausalSelfAttention::extractHead(cache.query, headIndex, this->headDimension);
        Matrix keyHead = CausalSelfAttention::extractHead(cache.key, headIndex, this->headDimension);
        Matrix valueHead = CausalSelfAttention::extractHead(cache.value, headIndex, this->headDimension);
        const Matrix& probabilities = cache.probabilities[static_cast<size_t>(headIndex)];

        Matrix valueHeadGradient = Matrix::multiply(attendedHeadGradient, Matrix::transpose(probabilities));
        Matrix probabilityGradient = Matrix::multiply(Matrix::transpose(valueHead), attendedHeadGradient);
        Matrix scoreGradient = CausalSelfAttention::softmaxBackward(probabilities, probabilityGradient);

        for (size_t keyIndex = 0; keyIndex < sequenceLength; ++keyIndex) {
            for (size_t queryIndex = 0; queryIndex < sequenceLength; ++queryIndex) {
                if (keyIndex <= queryIndex) continue;
                scoreGradient.data[keyIndex][queryIndex] = 0.0f;
            }
        }

        scoreGradient = Matrix::scale(scoreGradient, scale);
        Matrix queryHeadGradient = Matrix::multiply(keyHead, scoreGradient);
        Matrix keyHeadGradient = Matrix::multiply(queryHead, Matrix::transpose(scoreGradient));

        CausalSelfAttention::writeHead(queryGradient, headIndex, this->headDimension, queryHeadGradient);
        CausalSelfAttention::writeHead(keyGradient, headIndex, this->headDimension, keyHeadGradient);
        CausalSelfAttention::writeHead(valueGradient, headIndex, this->headDimension, valueHeadGradient);
    }

    queryWeightGradient = Matrix::multiply(queryGradient, Matrix::transpose(cache.input));
    keyWeightGradient = Matrix::multiply(keyGradient, Matrix::transpose(cache.input));
    valueWeightGradient = Matrix::multiply(valueGradient, Matrix::transpose(cache.input));

    Matrix inputGradient = Matrix::multiply(Matrix::transpose(this->queryWeight), queryGradient);
    inputGradient = Matrix::add(inputGradient, Matrix::multiply(Matrix::transpose(this->keyWeight), keyGradient));
    inputGradient = Matrix::add(inputGradient, Matrix::multiply(Matrix::transpose(this->valueWeight), valueGradient));
    return inputGradient;
}
