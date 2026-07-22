#include "CausalSelfAttention.hpp"

#include "../Activations/Softmax.hpp"
#include "../Initializers/UniformInit.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

CausalSelfAttention::CausalSelfAttention(Matrix queryWeight, Matrix keyWeight, Matrix valueWeight, Matrix outputWeight, int headCount)
    : queryWeight(std::move(queryWeight)), keyWeight(std::move(keyWeight)), valueWeight(std::move(valueWeight)), outputWeight(std::move(outputWeight)), headCount(headCount), headDimension(0) {
    if (this->headCount <= 0) throw std::invalid_argument("CausalSelfAttention headCount must be > 0");
    if (this->queryWeight.empty()) throw std::invalid_argument("CausalSelfAttention empty weights");
    if (static_cast<int>(this->queryWeight.rows) % this->headCount != 0) throw std::invalid_argument("CausalSelfAttention embeddingDim must be divisible by headCount");
    this->headDimension = static_cast<int>(this->queryWeight.rows) / this->headCount;
}

CausalSelfAttention CausalSelfAttention::create(int embeddingDim, int headCount, unsigned seed) {
    if (embeddingDim <= 0) throw std::invalid_argument("CausalSelfAttention::create embeddingDim must be > 0");
    if (headCount <= 0) throw std::invalid_argument("CausalSelfAttention::create headCount must be > 0");
    if (embeddingDim % headCount != 0) throw std::invalid_argument("CausalSelfAttention::create embeddingDim must be divisible by headCount");

    return CausalSelfAttention(UniformInit::matrix(embeddingDim, embeddingDim, 0.1f, seed), UniformInit::matrix(embeddingDim, embeddingDim, 0.1f, seed + 1u), UniformInit::matrix(embeddingDim, embeddingDim, 0.1f, seed + 2u), UniformInit::matrix(embeddingDim, embeddingDim, 0.1f, seed + 3u), headCount);
}

Matrix CausalSelfAttention::zerosLike(const Matrix& matrix) {
    return Matrix::zerosLike(matrix);
}

Matrix CausalSelfAttention::extractHead(const Matrix& full, int headIndex, int headDimension) {
    Matrix head(static_cast<size_t>(headDimension), full.cols, 0.0f);
    const size_t rowOffset = static_cast<size_t>(headIndex * headDimension);
    for (int row = 0; row < headDimension; ++row) {
        for (size_t column = 0; column < full.cols; ++column)
            head.at(static_cast<size_t>(row), column) = full.at(rowOffset + static_cast<size_t>(row), column);
    }
    return head;
}

void CausalSelfAttention::writeHead(Matrix& full, int headIndex, int headDimension, const Matrix& head) {
    const size_t rowOffset = static_cast<size_t>(headIndex * headDimension);
    for (int row = 0; row < headDimension; ++row) {
        for (size_t column = 0; column < head.cols; ++column)
            full.at(rowOffset + static_cast<size_t>(row), column) = head.at(static_cast<size_t>(row), column);
    }
}

Matrix CausalSelfAttention::forward(const Matrix& input, CausalSelfAttentionCache& cache) const {
    if (input.empty()) throw std::invalid_argument("CausalSelfAttention::forward empty input");
    if (this->queryWeight.cols != input.rows)
        throw std::invalid_argument("CausalSelfAttention::forward embedding dim mismatch");

    cache.input = input;
    cache.query = Matrix::multiply(this->queryWeight, input);
    cache.key = Matrix::multiply(this->keyWeight, input);
    cache.value = Matrix::multiply(this->valueWeight, input);

    const size_t sequenceLength = input.cols;
    const float scale = 1.0f / std::sqrt(static_cast<float>(this->headDimension));

    cache.scores.assign(static_cast<size_t>(this->headCount), Matrix());
    cache.probabilities.assign(static_cast<size_t>(this->headCount), Matrix());
    cache.attended = Matrix(cache.query.rows, sequenceLength, 0.0f);

    for (int headIndex = 0; headIndex < this->headCount; ++headIndex) {
        Matrix queryHead = CausalSelfAttention::extractHead(cache.query, headIndex, this->headDimension);
        Matrix keyHead = CausalSelfAttention::extractHead(cache.key, headIndex, this->headDimension);
        Matrix valueHead = CausalSelfAttention::extractHead(cache.value, headIndex, this->headDimension);

        Matrix scores = Matrix::scale(Matrix::multiply(keyHead, queryHead, true, false), scale);
        for (size_t keyIndex = 0; keyIndex < sequenceLength; ++keyIndex) {
            for (size_t queryIndex = 0; queryIndex < sequenceLength; ++queryIndex) {
                if (keyIndex <= queryIndex) continue;
                scores.at(keyIndex, queryIndex) = -1e9f;
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
    if (probabilities.rows != probabilityGradient.rows || probabilities.cols != probabilityGradient.cols)
        throw std::invalid_argument("CausalSelfAttention::softmaxBackward shape mismatch");

    Matrix scoreGradient = CausalSelfAttention::zerosLike(probabilities);
    const size_t keyCount = probabilities.rows;
    const size_t queryCount = probabilities.cols;

    for (size_t queryIndex = 0; queryIndex < queryCount; ++queryIndex) {
        float dot = 0.0f;
        for (size_t keyIndex = 0; keyIndex < keyCount; ++keyIndex)
            dot += probabilityGradient.at(keyIndex, queryIndex) * probabilities.at(keyIndex, queryIndex);

        for (size_t keyIndex = 0; keyIndex < keyCount; ++keyIndex) {
            scoreGradient.at(keyIndex, queryIndex) =
                probabilities.at(keyIndex, queryIndex) *
                (probabilityGradient.at(keyIndex, queryIndex) - dot);
        }
    }

    return scoreGradient;
}

Matrix CausalSelfAttention::backward(const Matrix& outputGradient, const CausalSelfAttentionCache& cache, Matrix& queryWeightGradient, Matrix& keyWeightGradient, Matrix& valueWeightGradient, Matrix& outputWeightGradient) const {
    if (cache.input.empty()) throw std::logic_error("CausalSelfAttention::backward called before forward");
    if (outputGradient.rows != this->outputWeight.rows || outputGradient.cols != cache.attended.cols)
        throw std::invalid_argument("CausalSelfAttention::backward output gradient shape mismatch");
    if (static_cast<int>(cache.scores.size()) != this->headCount || static_cast<int>(cache.probabilities.size()) != this->headCount)
        throw std::invalid_argument("CausalSelfAttention::backward head cache size mismatch");

    outputWeightGradient = Matrix::multiply(outputGradient, cache.attended, false, true);
    Matrix attendedGradient = Matrix::multiply(this->outputWeight, outputGradient, true, false);

    Matrix queryGradient = CausalSelfAttention::zerosLike(cache.query);
    Matrix keyGradient = CausalSelfAttention::zerosLike(cache.key);
    Matrix valueGradient = CausalSelfAttention::zerosLike(cache.value);

    const size_t sequenceLength = cache.input.cols;
    const float scale = 1.0f / std::sqrt(static_cast<float>(this->headDimension));

    for (int headIndex = 0; headIndex < this->headCount; ++headIndex) {
        Matrix attendedHeadGradient = CausalSelfAttention::extractHead(attendedGradient, headIndex, this->headDimension);
        Matrix queryHead = CausalSelfAttention::extractHead(cache.query, headIndex, this->headDimension);
        Matrix keyHead = CausalSelfAttention::extractHead(cache.key, headIndex, this->headDimension);
        Matrix valueHead = CausalSelfAttention::extractHead(cache.value, headIndex, this->headDimension);
        const Matrix& probabilities = cache.probabilities[static_cast<size_t>(headIndex)];

        Matrix valueHeadGradient = Matrix::multiply(attendedHeadGradient, probabilities, false, true);
        Matrix probabilityGradient = Matrix::multiply(valueHead, attendedHeadGradient, true, false);
        Matrix scoreGradient = CausalSelfAttention::softmaxBackward(probabilities, probabilityGradient);

        for (size_t keyIndex = 0; keyIndex < sequenceLength; ++keyIndex) {
            for (size_t queryIndex = 0; queryIndex < sequenceLength; ++queryIndex) {
                if (keyIndex <= queryIndex) continue;
                scoreGradient.at(keyIndex, queryIndex) = 0.0f;
            }
        }

        scoreGradient = Matrix::scale(scoreGradient, scale);
        Matrix queryHeadGradient = Matrix::multiply(keyHead, scoreGradient);
        Matrix keyHeadGradient = Matrix::multiply(queryHead, scoreGradient, false, true);

        CausalSelfAttention::writeHead(queryGradient, headIndex, this->headDimension, queryHeadGradient);
        CausalSelfAttention::writeHead(keyGradient, headIndex, this->headDimension, keyHeadGradient);
        CausalSelfAttention::writeHead(valueGradient, headIndex, this->headDimension, valueHeadGradient);
    }

    queryWeightGradient = Matrix::multiply(queryGradient, cache.input, false, true);
    keyWeightGradient = Matrix::multiply(keyGradient, cache.input, false, true);
    valueWeightGradient = Matrix::multiply(valueGradient, cache.input, false, true);

    Matrix inputGradient = Matrix::multiply(this->queryWeight, queryGradient, true, false);
    inputGradient = Matrix::add(inputGradient, Matrix::multiply(this->keyWeight, keyGradient, true, false));
    inputGradient = Matrix::add(inputGradient, Matrix::multiply(this->valueWeight, valueGradient, true, false));
    return inputGradient;
}
