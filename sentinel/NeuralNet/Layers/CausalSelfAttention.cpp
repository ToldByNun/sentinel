#include "CausalSelfAttention.hpp"

#include "../Activations/Softmax.hpp"
#include "../Initializers/UniformInit.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

CausalSelfAttention::CausalSelfAttention(Matrix queryWeight, Matrix keyWeight, Matrix valueWeight, Matrix outputWeight, RotaryEmbedding rotaryEmbedding, int headCount)
    : queryWeight(std::move(queryWeight)), keyWeight(std::move(keyWeight)), valueWeight(std::move(valueWeight)), outputWeight(std::move(outputWeight)), rotaryEmbedding(std::move(rotaryEmbedding)), headCount(headCount), headDimension(0) {
    if (this->headCount <= 0) throw std::invalid_argument("CausalSelfAttention headCount must be > 0");
    if (this->queryWeight.empty()) throw std::invalid_argument("CausalSelfAttention empty weights");
    if (static_cast<int>(this->queryWeight.rows) % this->headCount != 0) throw std::invalid_argument("CausalSelfAttention embeddingDim must be divisible by headCount");
    this->headDimension = static_cast<int>(this->queryWeight.rows) / this->headCount;
    if (this->headDimension % 2 != 0) throw std::invalid_argument("CausalSelfAttention headDimension must be even for RoPE");
}

CausalSelfAttention CausalSelfAttention::create(int embeddingDim, int headCount, int maximumPositionCount, unsigned seed) {
    if (embeddingDim <= 0) throw std::invalid_argument("CausalSelfAttention::create embeddingDim must be > 0");
    if (headCount <= 0) throw std::invalid_argument("CausalSelfAttention::create headCount must be > 0");
    if (embeddingDim % headCount != 0) throw std::invalid_argument("CausalSelfAttention::create embeddingDim must be divisible by headCount");
    if ((embeddingDim / headCount) % 2 != 0) throw std::invalid_argument("CausalSelfAttention::create headDimension must be even for RoPE");

    RotaryEmbedding rotaryEmbedding(embeddingDim / headCount, maximumPositionCount);
    return CausalSelfAttention(UniformInit::matrix(embeddingDim, embeddingDim, 0.1f, seed), UniformInit::matrix(embeddingDim, embeddingDim, 0.1f, seed + 1u), UniformInit::matrix(embeddingDim, embeddingDim, 0.1f, seed + 2u), UniformInit::matrix(embeddingDim, embeddingDim, 0.1f, seed + 3u), std::move(rotaryEmbedding), headCount);
}

void CausalSelfAttention::extractHeadInto(const Matrix& full, int headIndex, int headDimension, Matrix& head) {
    head.ensureSize(static_cast<size_t>(headDimension), full.cols);
    const size_t rowOffset = static_cast<size_t>(headIndex * headDimension);
    for (int row = 0; row < headDimension; ++row) {
        for (size_t column = 0; column < full.cols; ++column)
            head.at(static_cast<size_t>(row), column) = full.at(rowOffset + static_cast<size_t>(row), column);
    }
}

void CausalSelfAttention::writeHead(Matrix& full, int headIndex, int headDimension, const Matrix& head) {
    const size_t rowOffset = static_cast<size_t>(headIndex * headDimension);
    for (int row = 0; row < headDimension; ++row) {
        for (size_t column = 0; column < head.cols; ++column)
            full.at(rowOffset + static_cast<size_t>(row), column) = head.at(static_cast<size_t>(row), column);
    }
}

void CausalSelfAttention::softmaxBackwardInto(const Matrix& probabilities, const Matrix& probabilityGradient, Matrix& scoreGradient) {
    if (probabilities.rows != probabilityGradient.rows || probabilities.cols != probabilityGradient.cols) throw std::invalid_argument("CausalSelfAttention::softmaxBackwardInto shape mismatch");

    scoreGradient.ensureSize(probabilities.rows, probabilities.cols);
    const size_t keyCount = probabilities.rows;
    const size_t queryCount = probabilities.cols;

    for (size_t queryIndex = 0; queryIndex < queryCount; ++queryIndex) {
        float dot = 0.0f;
        for (size_t keyIndex = 0; keyIndex < keyCount; ++keyIndex)
            dot += probabilityGradient.at(keyIndex, queryIndex) * probabilities.at(keyIndex, queryIndex);

        for (size_t keyIndex = 0; keyIndex < keyCount; ++keyIndex)
            scoreGradient.at(keyIndex, queryIndex) = probabilities.at(keyIndex, queryIndex) * (probabilityGradient.at(keyIndex, queryIndex) - dot);
    }
}

Matrix CausalSelfAttention::forward(const Matrix& input, CausalSelfAttentionCache& cache) const {
    if (input.empty()) throw std::invalid_argument("CausalSelfAttention::forward empty input");
    if (this->queryWeight.cols != input.rows) throw std::invalid_argument("CausalSelfAttention::forward embedding dim mismatch");

    cache.input = input;
    Matrix::gemm(this->queryWeight, input, cache.query);
    Matrix::gemm(this->keyWeight, input, cache.key);
    Matrix::gemm(this->valueWeight, input, cache.value);
    this->rotaryEmbedding.rotateInPlace(cache.query, this->headCount);
    this->rotaryEmbedding.rotateInPlace(cache.key, this->headCount);

    const size_t sequenceLength = input.cols;
    const float scale = 1.0f / std::sqrt(static_cast<float>(this->headDimension));

    if (cache.scores.size() != static_cast<size_t>(this->headCount)) cache.scores.assign(static_cast<size_t>(this->headCount), Matrix());
    if (cache.probabilities.size() != static_cast<size_t>(this->headCount)) cache.probabilities.assign(static_cast<size_t>(this->headCount), Matrix());
    cache.attended.ensureSize(cache.query.rows, sequenceLength);
    Matrix::zeroInPlace(cache.attended);

    for (int headIndex = 0; headIndex < this->headCount; ++headIndex) {
        CausalSelfAttention::extractHeadInto(cache.query, headIndex, this->headDimension, cache.queryHead);
        CausalSelfAttention::extractHeadInto(cache.key, headIndex, this->headDimension, cache.keyHead);
        CausalSelfAttention::extractHeadInto(cache.value, headIndex, this->headDimension, cache.valueHead);

        Matrix& scores = cache.scores[static_cast<size_t>(headIndex)];
        Matrix& probabilities = cache.probabilities[static_cast<size_t>(headIndex)];

        Matrix::gemm(cache.keyHead, cache.queryHead, scores, true, false);
        Matrix::scaleInPlace(scores, scale);
        for (size_t keyIndex = 0; keyIndex < sequenceLength; ++keyIndex) {
            for (size_t queryIndex = 0; queryIndex < sequenceLength; ++queryIndex) {
                if (keyIndex <= queryIndex) continue;
                scores.at(keyIndex, queryIndex) = -1e9f;
            }
        }

        Softmax::applyInto(scores, probabilities);
        Matrix::gemm(cache.valueHead, probabilities, cache.attendedHead);
        CausalSelfAttention::writeHead(cache.attended, headIndex, this->headDimension, cache.attendedHead);
    }

    Matrix::gemm(this->outputWeight, cache.attended, cache.output);
    return cache.output;
}

Matrix CausalSelfAttention::backward(const Matrix& outputGradient, CausalSelfAttentionCache& cache, Matrix& queryWeightGradient, Matrix& keyWeightGradient, Matrix& valueWeightGradient, Matrix& outputWeightGradient) const {
    if (cache.input.empty()) throw std::logic_error("CausalSelfAttention::backward called before forward");
    if (outputGradient.rows != this->outputWeight.rows || outputGradient.cols != cache.attended.cols) throw std::invalid_argument("CausalSelfAttention::backward output gradient shape mismatch");
    if (static_cast<int>(cache.scores.size()) != this->headCount || static_cast<int>(cache.probabilities.size()) != this->headCount) throw std::invalid_argument("CausalSelfAttention::backward head cache size mismatch");

    Matrix::gemm(outputGradient, cache.attended, outputWeightGradient, false, true);
    Matrix::gemm(this->outputWeight, outputGradient, cache.attendedGradient, true, false);

    cache.queryGradient.ensureSize(cache.query.rows, cache.query.cols);
    cache.keyGradient.ensureSize(cache.key.rows, cache.key.cols);
    cache.valueGradient.ensureSize(cache.value.rows, cache.value.cols);
    Matrix::zeroInPlace(cache.queryGradient);
    Matrix::zeroInPlace(cache.keyGradient);
    Matrix::zeroInPlace(cache.valueGradient);

    const size_t sequenceLength = cache.input.cols;
    const float scale = 1.0f / std::sqrt(static_cast<float>(this->headDimension));

    for (int headIndex = 0; headIndex < this->headCount; ++headIndex) {
        CausalSelfAttention::extractHeadInto(cache.attendedGradient, headIndex, this->headDimension, cache.attendedHead);
        CausalSelfAttention::extractHeadInto(cache.query, headIndex, this->headDimension, cache.queryHead);
        CausalSelfAttention::extractHeadInto(cache.key, headIndex, this->headDimension, cache.keyHead);
        CausalSelfAttention::extractHeadInto(cache.value, headIndex, this->headDimension, cache.valueHead);
        const Matrix& probabilities = cache.probabilities[static_cast<size_t>(headIndex)];

        Matrix::gemm(cache.attendedHead, probabilities, cache.valueHeadGradient, false, true);
        Matrix::gemm(cache.valueHead, cache.attendedHead, cache.probabilityGradient, true, false);
        CausalSelfAttention::softmaxBackwardInto(probabilities, cache.probabilityGradient, cache.scoreGradient);

        for (size_t keyIndex = 0; keyIndex < sequenceLength; ++keyIndex) {
            for (size_t queryIndex = 0; queryIndex < sequenceLength; ++queryIndex) {
                if (keyIndex <= queryIndex) continue;
                cache.scoreGradient.at(keyIndex, queryIndex) = 0.0f;
            }
        }

        Matrix::scaleInPlace(cache.scoreGradient, scale);
        Matrix::gemm(cache.keyHead, cache.scoreGradient, cache.queryHeadGradient);
        Matrix::gemm(cache.queryHead, cache.scoreGradient, cache.keyHeadGradient, false, true);

        CausalSelfAttention::writeHead(cache.queryGradient, headIndex, this->headDimension, cache.queryHeadGradient);
        CausalSelfAttention::writeHead(cache.keyGradient, headIndex, this->headDimension, cache.keyHeadGradient);
        CausalSelfAttention::writeHead(cache.valueGradient, headIndex, this->headDimension, cache.valueHeadGradient);
    }

    this->rotaryEmbedding.rotateInverseInPlace(cache.queryGradient, this->headCount);
    this->rotaryEmbedding.rotateInverseInPlace(cache.keyGradient, this->headCount);

    Matrix::gemm(cache.queryGradient, cache.input, queryWeightGradient, false, true);
    Matrix::gemm(cache.keyGradient, cache.input, keyWeightGradient, false, true);
    Matrix::gemm(cache.valueGradient, cache.input, valueWeightGradient, false, true);

    Matrix::gemm(this->queryWeight, cache.queryGradient, cache.inputGradient, true, false);
    Matrix::gemm(this->keyWeight, cache.keyGradient, cache.temp, true, false);
    Matrix::addInPlace(cache.inputGradient, cache.temp);
    Matrix::gemm(this->valueWeight, cache.valueGradient, cache.temp, true, false);
    Matrix::addInPlace(cache.inputGradient, cache.temp);
    return cache.inputGradient;
}
