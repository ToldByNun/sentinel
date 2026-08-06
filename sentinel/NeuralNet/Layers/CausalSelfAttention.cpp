#include "CausalSelfAttention.hpp"

#include "../Activations/Softmax.hpp"
#include "../Initializers/UniformInit.hpp"
#include "../Utils/SmokeLog.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <utility>
#include <vector>

CausalSelfAttention::CausalSelfAttention(
    Matrix queryWeight,
    Matrix keyWeight,
    Matrix valueWeight,
    Matrix outputWeight,
    RotaryEmbedding rotaryEmbedding,
    int headCount,
    int windowSize,
    int globalTokenCount,
    int kvHeadCount)
    : queryWeight(std::move(queryWeight)),
      keyWeight(std::move(keyWeight)),
      valueWeight(std::move(valueWeight)),
      outputWeight(std::move(outputWeight)),
      rotaryEmbedding(std::move(rotaryEmbedding)),
      headCount(headCount),
      kvHeadCount(kvHeadCount <= 0 ? headCount : kvHeadCount),
      headDimension(0),
      windowSize(windowSize),
      globalTokenCount(globalTokenCount),
      preferSparseCompute(true) {
    if (this->headCount <= 0) throw std::invalid_argument("CausalSelfAttention headCount must be > 0");
    if (this->kvHeadCount <= 0) throw std::invalid_argument("CausalSelfAttention kvHeadCount must be > 0");
    if (this->headCount % this->kvHeadCount != 0)
        throw std::invalid_argument("CausalSelfAttention headCount must be divisible by kvHeadCount");
    if (this->queryWeight.empty()) throw std::invalid_argument("CausalSelfAttention empty weights");
    if (static_cast<int>(this->queryWeight.rows) % this->headCount != 0)
        throw std::invalid_argument("CausalSelfAttention query rows must be divisible by headCount");
    if (this->globalTokenCount < 0) throw std::invalid_argument("CausalSelfAttention globalTokenCount must be >= 0");
    this->headDimension = static_cast<int>(this->queryWeight.rows) / this->headCount;
    if (this->headDimension % 2 != 0) throw std::invalid_argument("CausalSelfAttention headDimension must be even for RoPE");
    const int expectedKvRows = this->kvHeadCount * this->headDimension;
    if (static_cast<int>(this->keyWeight.rows) != expectedKvRows || static_cast<int>(this->valueWeight.rows) != expectedKvRows)
        throw std::invalid_argument("CausalSelfAttention key/value rows must equal kvHeadCount * headDimension");
    if (this->keyWeight.cols != this->queryWeight.cols || this->valueWeight.cols != this->queryWeight.cols)
        throw std::invalid_argument("CausalSelfAttention Q/K/V embedding dim mismatch");
}

CausalSelfAttention CausalSelfAttention::create(
    int embeddingDim,
    int headCount,
    int maximumPositionCount,
    unsigned seed,
    int windowSize,
    int globalTokenCount,
    float ropeBase,
    int kvHeadCount) {
    if (embeddingDim <= 0) throw std::invalid_argument("CausalSelfAttention::create embeddingDim must be > 0");
    if (headCount <= 0) throw std::invalid_argument("CausalSelfAttention::create headCount must be > 0");
    if (maximumPositionCount <= 0) throw std::invalid_argument("CausalSelfAttention::create maximumPositionCount must be > 0");
    if (embeddingDim % headCount != 0) throw std::invalid_argument("CausalSelfAttention::create embeddingDim must be divisible by headCount");
    if ((embeddingDim / headCount) % 2 != 0) throw std::invalid_argument("CausalSelfAttention::create headDimension must be even for RoPE");
    if (globalTokenCount < 0) throw std::invalid_argument("CausalSelfAttention::create globalTokenCount must be >= 0");
    if (ropeBase <= 0.0f) throw std::invalid_argument("CausalSelfAttention::create ropeBase must be > 0");

    const int resolvedKvHeadCount = (kvHeadCount <= 0) ? headCount : kvHeadCount;
    if (resolvedKvHeadCount <= 0) throw std::invalid_argument("CausalSelfAttention::create kvHeadCount must be > 0");
    if (headCount % resolvedKvHeadCount != 0)
        throw std::invalid_argument("CausalSelfAttention::create headCount must be divisible by kvHeadCount");

    const int headDimension = embeddingDim / headCount;
    const int kvRows = resolvedKvHeadCount * headDimension;
    const int resolvedWindowSize = (windowSize <= 0) ? maximumPositionCount : windowSize;
    RotaryEmbedding rotaryEmbedding(headDimension, maximumPositionCount, ropeBase);
    return CausalSelfAttention(
        UniformInit::matrix(embeddingDim, embeddingDim, 0.1f, seed),
        UniformInit::matrix(kvRows, embeddingDim, 0.1f, seed + 1u),
        UniformInit::matrix(kvRows, embeddingDim, 0.1f, seed + 2u),
        UniformInit::matrix(embeddingDim, embeddingDim, 0.1f, seed + 3u),
        std::move(rotaryEmbedding),
        headCount,
        resolvedWindowSize,
        globalTokenCount,
        resolvedKvHeadCount);
}

int CausalSelfAttention::kvHeadIndexForQueryHead(int queryHeadIndex) const {
    const int groupSize = this->headCount / this->kvHeadCount;
    return queryHeadIndex / groupSize;
}

bool CausalSelfAttention::allowsKey(int queryIndex, int keyIndex, int windowSize, int globalTokenCount) {
    if (keyIndex > queryIndex) return false;
    if (keyIndex < globalTokenCount) return true;
    if (windowSize <= 0) return true;
    return keyIndex >= queryIndex - windowSize + 1;
}

bool CausalSelfAttention::usesSparseCompute(int sequenceLength) const {
    if (!this->preferSparseCompute) return false;
    if (sequenceLength <= 0) return false;
    if (this->windowSize <= 0) return false;
    return this->windowSize < sequenceLength;
}

void CausalSelfAttention::applyAttentionMaskInPlace(Matrix& scores, int windowSize, int globalTokenCount) {
    if (scores.empty()) throw std::invalid_argument("CausalSelfAttention::applyAttentionMaskInPlace empty scores");
    if (scores.rows != scores.cols) throw std::invalid_argument("CausalSelfAttention::applyAttentionMaskInPlace scores must be square");

    const int sequenceLength = static_cast<int>(scores.rows);
    for (int keyIndex = 0; keyIndex < sequenceLength; ++keyIndex) {
        for (int queryIndex = 0; queryIndex < sequenceLength; ++queryIndex) {
            if (CausalSelfAttention::allowsKey(queryIndex, keyIndex, windowSize, globalTokenCount)) continue;
            scores.at(static_cast<size_t>(keyIndex), static_cast<size_t>(queryIndex)) = -1e9f;
        }
    }
}

void CausalSelfAttention::computeDenseMaskedScoresInto(const Matrix& queryHead, const Matrix& keyHead, Matrix& scores, float scale, int windowSize, int globalTokenCount) {
    Matrix::gemm(keyHead, queryHead, scores, true, false);
    Matrix::scaleInPlace(scores, scale);
    CausalSelfAttention::applyAttentionMaskInPlace(scores, windowSize, globalTokenCount);
}

void CausalSelfAttention::computeSparseScoresInto(const Matrix& queryHead, const Matrix& keyHead, Matrix& scores, float scale, int windowSize, int globalTokenCount) {
    const int sequenceLength = static_cast<int>(queryHead.cols);
    const int headDimension = static_cast<int>(queryHead.rows);
    scores.ensureSize(static_cast<size_t>(sequenceLength), static_cast<size_t>(sequenceLength));

    for (int keyIndex = 0; keyIndex < sequenceLength; ++keyIndex) {
        for (int queryIndex = 0; queryIndex < sequenceLength; ++queryIndex)
            scores.at(static_cast<size_t>(keyIndex), static_cast<size_t>(queryIndex)) = -1e9f;
    }

    for (int queryIndex = 0; queryIndex < sequenceLength; ++queryIndex) {
        for (int keyIndex = 0; keyIndex < sequenceLength; ++keyIndex) {
            if (!CausalSelfAttention::allowsKey(queryIndex, keyIndex, windowSize, globalTokenCount)) continue;

            float dot = 0.0f;
            for (int row = 0; row < headDimension; ++row)
                dot += keyHead.at(static_cast<size_t>(row), static_cast<size_t>(keyIndex)) * queryHead.at(static_cast<size_t>(row), static_cast<size_t>(queryIndex));
            scores.at(static_cast<size_t>(keyIndex), static_cast<size_t>(queryIndex)) = scale * dot;
        }
    }
}

void CausalSelfAttention::attendDenseInto(const Matrix& valueHead, const Matrix& probabilities, Matrix& attendedHead) {
    Matrix::gemm(valueHead, probabilities, attendedHead);
}

void CausalSelfAttention::attendSparseInto(const Matrix& valueHead, const Matrix& probabilities, Matrix& attendedHead, int windowSize, int globalTokenCount) {
    const int sequenceLength = static_cast<int>(probabilities.cols);
    const int headDimension = static_cast<int>(valueHead.rows);
    attendedHead.ensureSize(static_cast<size_t>(headDimension), static_cast<size_t>(sequenceLength));
    Matrix::zeroInPlace(attendedHead);

    for (int queryIndex = 0; queryIndex < sequenceLength; ++queryIndex) {
        for (int keyIndex = 0; keyIndex < sequenceLength; ++keyIndex) {
            if (!CausalSelfAttention::allowsKey(queryIndex, keyIndex, windowSize, globalTokenCount)) continue;
            const float probability = probabilities.at(static_cast<size_t>(keyIndex), static_cast<size_t>(queryIndex));
            for (int row = 0; row < headDimension; ++row)
                attendedHead.at(static_cast<size_t>(row), static_cast<size_t>(queryIndex)) += probability * valueHead.at(static_cast<size_t>(row), static_cast<size_t>(keyIndex));
        }
    }
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

void CausalSelfAttention::addHeadInto(Matrix& full, int headIndex, int headDimension, const Matrix& head) {
    const size_t rowOffset = static_cast<size_t>(headIndex * headDimension);
    for (int row = 0; row < headDimension; ++row) {
        for (size_t column = 0; column < head.cols; ++column)
            full.at(rowOffset + static_cast<size_t>(row), column) += head.at(static_cast<size_t>(row), column);
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
    this->rotaryEmbedding.rotateInPlace(cache.key, this->kvHeadCount);

    const size_t sequenceLength = input.cols;
    const float scale = 1.0f / std::sqrt(static_cast<float>(this->headDimension));
    const bool sparseCompute = this->usesSparseCompute(static_cast<int>(sequenceLength));

    if (cache.scores.size() != static_cast<size_t>(this->headCount)) cache.scores.assign(static_cast<size_t>(this->headCount), Matrix());
    if (cache.probabilities.size() != static_cast<size_t>(this->headCount)) cache.probabilities.assign(static_cast<size_t>(this->headCount), Matrix());
    cache.attended.ensureSize(cache.query.rows, sequenceLength);
    Matrix::zeroInPlace(cache.attended);

    for (int headIndex = 0; headIndex < this->headCount; ++headIndex) {
        const int kvHeadIndex = this->kvHeadIndexForQueryHead(headIndex);
        CausalSelfAttention::extractHeadInto(cache.query, headIndex, this->headDimension, cache.queryHead);
        CausalSelfAttention::extractHeadInto(cache.key, kvHeadIndex, this->headDimension, cache.keyHead);
        CausalSelfAttention::extractHeadInto(cache.value, kvHeadIndex, this->headDimension, cache.valueHead);

        Matrix& scores = cache.scores[static_cast<size_t>(headIndex)];
        Matrix& probabilities = cache.probabilities[static_cast<size_t>(headIndex)];

        if (sparseCompute)
            CausalSelfAttention::computeSparseScoresInto(cache.queryHead, cache.keyHead, scores, scale, this->windowSize, this->globalTokenCount);
        else
            CausalSelfAttention::computeDenseMaskedScoresInto(cache.queryHead, cache.keyHead, scores, scale, this->windowSize, this->globalTokenCount);

        Softmax::applyInto(scores, probabilities);

        if (sparseCompute)
            CausalSelfAttention::attendSparseInto(cache.valueHead, probabilities, cache.attendedHead, this->windowSize, this->globalTokenCount);
        else
            CausalSelfAttention::attendDenseInto(cache.valueHead, probabilities, cache.attendedHead);

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

    const int sequenceLength = static_cast<int>(cache.input.cols);
    const float scale = 1.0f / std::sqrt(static_cast<float>(this->headDimension));
    const bool sparseCompute = this->usesSparseCompute(sequenceLength);

    for (int headIndex = 0; headIndex < this->headCount; ++headIndex) {
        const int kvHeadIndex = this->kvHeadIndexForQueryHead(headIndex);
        CausalSelfAttention::extractHeadInto(cache.attendedGradient, headIndex, this->headDimension, cache.attendedHead);
        CausalSelfAttention::extractHeadInto(cache.query, headIndex, this->headDimension, cache.queryHead);
        CausalSelfAttention::extractHeadInto(cache.key, kvHeadIndex, this->headDimension, cache.keyHead);
        CausalSelfAttention::extractHeadInto(cache.value, kvHeadIndex, this->headDimension, cache.valueHead);
        const Matrix& probabilities = cache.probabilities[static_cast<size_t>(headIndex)];

        if (sparseCompute) {
            cache.valueHeadGradient.ensureSize(static_cast<size_t>(this->headDimension), static_cast<size_t>(sequenceLength));
            cache.probabilityGradient.ensureSize(static_cast<size_t>(sequenceLength), static_cast<size_t>(sequenceLength));
            cache.queryHeadGradient.ensureSize(static_cast<size_t>(this->headDimension), static_cast<size_t>(sequenceLength));
            cache.keyHeadGradient.ensureSize(static_cast<size_t>(this->headDimension), static_cast<size_t>(sequenceLength));
            Matrix::zeroInPlace(cache.valueHeadGradient);
            Matrix::zeroInPlace(cache.probabilityGradient);
            Matrix::zeroInPlace(cache.queryHeadGradient);
            Matrix::zeroInPlace(cache.keyHeadGradient);

            for (int queryIndex = 0; queryIndex < sequenceLength; ++queryIndex) {
                for (int keyIndex = 0; keyIndex < sequenceLength; ++keyIndex) {
                    if (!CausalSelfAttention::allowsKey(queryIndex, keyIndex, this->windowSize, this->globalTokenCount)) continue;

                    const float probability = probabilities.at(static_cast<size_t>(keyIndex), static_cast<size_t>(queryIndex));
                    float probabilityGradientValue = 0.0f;
                    for (int row = 0; row < this->headDimension; ++row) {
                        const float attendedGradientValue = cache.attendedHead.at(static_cast<size_t>(row), static_cast<size_t>(queryIndex));
                        cache.valueHeadGradient.at(static_cast<size_t>(row), static_cast<size_t>(keyIndex)) += probability * attendedGradientValue;
                        probabilityGradientValue += cache.valueHead.at(static_cast<size_t>(row), static_cast<size_t>(keyIndex)) * attendedGradientValue;
                    }
                    cache.probabilityGradient.at(static_cast<size_t>(keyIndex), static_cast<size_t>(queryIndex)) = probabilityGradientValue;
                }
            }

            CausalSelfAttention::softmaxBackwardInto(probabilities, cache.probabilityGradient, cache.scoreGradient);

            for (int keyIndex = 0; keyIndex < sequenceLength; ++keyIndex) {
                for (int queryIndex = 0; queryIndex < sequenceLength; ++queryIndex) {
                    if (CausalSelfAttention::allowsKey(queryIndex, keyIndex, this->windowSize, this->globalTokenCount)) continue;
                    cache.scoreGradient.at(static_cast<size_t>(keyIndex), static_cast<size_t>(queryIndex)) = 0.0f;
                }
            }

            Matrix::scaleInPlace(cache.scoreGradient, scale);

            for (int queryIndex = 0; queryIndex < sequenceLength; ++queryIndex) {
                for (int keyIndex = 0; keyIndex < sequenceLength; ++keyIndex) {
                    if (!CausalSelfAttention::allowsKey(queryIndex, keyIndex, this->windowSize, this->globalTokenCount)) continue;
                    const float scoreGradientValue = cache.scoreGradient.at(static_cast<size_t>(keyIndex), static_cast<size_t>(queryIndex));
                    for (int row = 0; row < this->headDimension; ++row) {
                        cache.queryHeadGradient.at(static_cast<size_t>(row), static_cast<size_t>(queryIndex)) += cache.keyHead.at(static_cast<size_t>(row), static_cast<size_t>(keyIndex)) * scoreGradientValue;
                        cache.keyHeadGradient.at(static_cast<size_t>(row), static_cast<size_t>(keyIndex)) += cache.queryHead.at(static_cast<size_t>(row), static_cast<size_t>(queryIndex)) * scoreGradientValue;
                    }
                }
            }
        } else {
            Matrix::gemm(cache.attendedHead, probabilities, cache.valueHeadGradient, false, true);
            Matrix::gemm(cache.valueHead, cache.attendedHead, cache.probabilityGradient, true, false);
            CausalSelfAttention::softmaxBackwardInto(probabilities, cache.probabilityGradient, cache.scoreGradient);

            for (int keyIndex = 0; keyIndex < sequenceLength; ++keyIndex) {
                for (int queryIndex = 0; queryIndex < sequenceLength; ++queryIndex) {
                    if (CausalSelfAttention::allowsKey(queryIndex, keyIndex, this->windowSize, this->globalTokenCount)) continue;
                    cache.scoreGradient.at(static_cast<size_t>(keyIndex), static_cast<size_t>(queryIndex)) = 0.0f;
                }
            }

            Matrix::scaleInPlace(cache.scoreGradient, scale);
            Matrix::gemm(cache.keyHead, cache.scoreGradient, cache.queryHeadGradient);
            Matrix::gemm(cache.queryHead, cache.scoreGradient, cache.keyHeadGradient, false, true);
        }

        CausalSelfAttention::writeHead(cache.queryGradient, headIndex, this->headDimension, cache.queryHeadGradient);
        CausalSelfAttention::addHeadInto(cache.keyGradient, kvHeadIndex, this->headDimension, cache.keyHeadGradient);
        CausalSelfAttention::addHeadInto(cache.valueGradient, kvHeadIndex, this->headDimension, cache.valueHeadGradient);
    }

    this->rotaryEmbedding.rotateInverseInPlace(cache.queryGradient, this->headCount);
    this->rotaryEmbedding.rotateInverseInPlace(cache.keyGradient, this->kvHeadCount);

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

void CausalSelfAttention::runSparseMaskSmokeDemo(int embeddingDim, int headCount, int sequenceLength, int maximumPositionCount, int windowSize, int globalTokenCount) {
    if (embeddingDim <= 0 || headCount <= 0 || sequenceLength <= 0 || maximumPositionCount < sequenceLength)
        throw std::invalid_argument("CausalSelfAttention::runSparseMaskSmokeDemo invalid dims");
    if (windowSize <= 0 || globalTokenCount < 0)
        throw std::invalid_argument("CausalSelfAttention::runSparseMaskSmokeDemo invalid sparse config");

    Matrix hostInput(static_cast<size_t>(embeddingDim), static_cast<size_t>(sequenceLength), 0.0f);
    unsigned state = 71u;
    for (size_t index = 0; index < hostInput.data.size(); ++index) {
        state = state * 1664525u + 1013904223u;
        hostInput.data[index] = (static_cast<float>(state >> 8) / 16777216.0f) * 2.0f - 1.0f;
    }

    CausalSelfAttention dense = CausalSelfAttention::create(embeddingDim, headCount, maximumPositionCount, 11u);
    CausalSelfAttention denseExplicit = CausalSelfAttention::create(embeddingDim, headCount, maximumPositionCount, 11u, maximumPositionCount, 0);
    CausalSelfAttentionCache denseCache;
    CausalSelfAttentionCache denseExplicitCache;
    Matrix denseOutput = dense.forward(hostInput, denseCache);
    Matrix denseExplicitOutput = denseExplicit.forward(hostInput, denseExplicitCache);

    float denseParity = 0.0f;
    for (size_t index = 0; index < denseOutput.data.size(); ++index)
        denseParity = (std::max)(denseParity, std::fabs(denseOutput.data[index] - denseExplicitOutput.data[index]));

    CausalSelfAttention sparse = CausalSelfAttention::create(embeddingDim, headCount, maximumPositionCount, 11u, windowSize, globalTokenCount);
    CausalSelfAttentionCache sparseCache;
    sparse.forward(hostInput, sparseCache);

    float forbiddenProbabilityMass = 0.0f;
    int forbiddenCount = 0;
    int allowedCount = 0;
    for (int headIndex = 0; headIndex < headCount; ++headIndex) {
        const Matrix& probabilities = sparseCache.probabilities[static_cast<size_t>(headIndex)];
        for (int keyIndex = 0; keyIndex < sequenceLength; ++keyIndex) {
            for (int queryIndex = 0; queryIndex < sequenceLength; ++queryIndex) {
                const float probability = probabilities.at(static_cast<size_t>(keyIndex), static_cast<size_t>(queryIndex));
                if (CausalSelfAttention::allowsKey(queryIndex, keyIndex, windowSize, globalTokenCount)) {
                    ++allowedCount;
                    continue;
                }
                ++forbiddenCount;
                forbiddenProbabilityMass = (std::max)(forbiddenProbabilityMass, std::fabs(probability));
            }
        }
    }

    SmokeLog::result("Sparse Attn S1", "embed=%d heads=%d seq=%d W=%d G=%d  denseDiff=%.2e  forbiddenP=%.2e  allowed=%d forbidden=%d",
        embeddingDim, headCount, sequenceLength, windowSize, globalTokenCount, denseParity, forbiddenProbabilityMass, allowedCount, forbiddenCount);
}

void CausalSelfAttention::runSparseBackwardSmokeDemo(int embeddingDim, int headCount, int sequenceLength, int maximumPositionCount, int windowSize, int globalTokenCount) {
    if (embeddingDim <= 0 || headCount <= 0 || sequenceLength <= 0 || maximumPositionCount < sequenceLength)
        throw std::invalid_argument("CausalSelfAttention::runSparseBackwardSmokeDemo invalid dims");
    if (windowSize <= 0 || globalTokenCount < 0)
        throw std::invalid_argument("CausalSelfAttention::runSparseBackwardSmokeDemo invalid sparse config");

    Matrix hostInput(static_cast<size_t>(embeddingDim), static_cast<size_t>(sequenceLength), 0.0f);
    unsigned state = 83u;
    for (size_t index = 0; index < hostInput.data.size(); ++index) {
        state = state * 1664525u + 1013904223u;
        hostInput.data[index] = (static_cast<float>(state >> 8) / 16777216.0f) * 2.0f - 1.0f;
    }

    Matrix outputGradient(static_cast<size_t>(embeddingDim), static_cast<size_t>(sequenceLength), 1.0f);

    CausalSelfAttention denseA = CausalSelfAttention::create(embeddingDim, headCount, maximumPositionCount, 17u);
    CausalSelfAttention denseB = CausalSelfAttention::create(embeddingDim, headCount, maximumPositionCount, 17u, maximumPositionCount, 0);
    CausalSelfAttentionCache denseCacheA;
    CausalSelfAttentionCache denseCacheB;
    denseA.forward(hostInput, denseCacheA);
    denseB.forward(hostInput, denseCacheB);

    Matrix queryGradA;
    Matrix keyGradA;
    Matrix valueGradA;
    Matrix outputGradA;
    Matrix queryGradB;
    Matrix keyGradB;
    Matrix valueGradB;
    Matrix outputGradB;
    Matrix inputGradA = denseA.backward(outputGradient, denseCacheA, queryGradA, keyGradA, valueGradA, outputGradA);
    Matrix inputGradB = denseB.backward(outputGradient, denseCacheB, queryGradB, keyGradB, valueGradB, outputGradB);

    float denseGradientParity = 0.0f;
    for (size_t index = 0; index < inputGradA.data.size(); ++index)
        denseGradientParity = (std::max)(denseGradientParity, std::fabs(inputGradA.data[index] - inputGradB.data[index]));

    CausalSelfAttention sparse = CausalSelfAttention::create(embeddingDim, headCount, maximumPositionCount, 17u, windowSize, globalTokenCount);
    CausalSelfAttentionCache sparseCache;
    sparse.forward(hostInput, sparseCache);
    Matrix queryGradS;
    Matrix keyGradS;
    Matrix valueGradS;
    Matrix outputGradS;
    Matrix inputGradS = sparse.backward(outputGradient, sparseCache, queryGradS, keyGradS, valueGradS, outputGradS);

    float forbiddenScoreGradient = 0.0f;
    for (int keyIndex = 0; keyIndex < sequenceLength; ++keyIndex) {
        for (int queryIndex = 0; queryIndex < sequenceLength; ++queryIndex) {
            if (CausalSelfAttention::allowsKey(queryIndex, keyIndex, windowSize, globalTokenCount)) continue;
            forbiddenScoreGradient = (std::max)(forbiddenScoreGradient, std::fabs(sparseCache.scoreGradient.at(static_cast<size_t>(keyIndex), static_cast<size_t>(queryIndex))));
        }
    }

    const float epsilon = 1.0e-3f;
    const size_t probeIndex = hostInput.data.size() / 3;
    Matrix inputPlus = hostInput;
    Matrix inputMinus = hostInput;
    inputPlus.data[probeIndex] += epsilon;
    inputMinus.data[probeIndex] -= epsilon;

    CausalSelfAttentionCache plusCache;
    CausalSelfAttentionCache minusCache;
    Matrix plusOutput = sparse.forward(inputPlus, plusCache);
    Matrix minusOutput = sparse.forward(inputMinus, minusCache);

    float plusLoss = 0.0f;
    float minusLoss = 0.0f;
    for (size_t index = 0; index < plusOutput.data.size(); ++index) {
        plusLoss += plusOutput.data[index];
        minusLoss += minusOutput.data[index];
    }
    const float numericalGradient = (plusLoss - minusLoss) / (2.0f * epsilon);
    const float analyticGradient = inputGradS.data[probeIndex];
    const float finiteDifferenceError = std::fabs(numericalGradient - analyticGradient);

    SmokeLog::result("Sparse Attn S2", "embed=%d heads=%d seq=%d W=%d G=%d  gradDiff=%.2e  fdErr=%.2e  forbiddenGrad=%.2e",
        embeddingDim, headCount, sequenceLength, windowSize, globalTokenCount, denseGradientParity, finiteDifferenceError, forbiddenScoreGradient);
    (void)analyticGradient;
    (void)numericalGradient;
}

void CausalSelfAttention::runSparseComputeSmokeDemo(int embeddingDim, int headCount, int sequenceLength, int maximumPositionCount, int windowSize, int globalTokenCount) {
    if (embeddingDim <= 0 || headCount <= 0 || sequenceLength <= 0 || maximumPositionCount < sequenceLength)
        throw std::invalid_argument("CausalSelfAttention::runSparseComputeSmokeDemo invalid dims");
    if (windowSize <= 0 || windowSize >= sequenceLength)
        throw std::invalid_argument("CausalSelfAttention::runSparseComputeSmokeDemo windowSize must be in (0, seq)");

    Matrix hostInput(static_cast<size_t>(embeddingDim), static_cast<size_t>(sequenceLength), 0.0f);
    unsigned state = 97u;
    for (size_t index = 0; index < hostInput.data.size(); ++index) {
        state = state * 1664525u + 1013904223u;
        hostInput.data[index] = (static_cast<float>(state >> 8) / 16777216.0f) * 2.0f - 1.0f;
    }

    CausalSelfAttention sparsePath = CausalSelfAttention::create(embeddingDim, headCount, maximumPositionCount, 23u, windowSize, globalTokenCount);
    CausalSelfAttention denseMaskedPath = CausalSelfAttention::create(embeddingDim, headCount, maximumPositionCount, 23u, windowSize, globalTokenCount);
    sparsePath.preferSparseCompute = true;
    denseMaskedPath.preferSparseCompute = false;

    CausalSelfAttentionCache sparseCache;
    CausalSelfAttentionCache denseCache;
    Matrix sparseOutput = sparsePath.forward(hostInput, sparseCache);
    Matrix denseOutput = denseMaskedPath.forward(hostInput, denseCache);

    float forwardParity = 0.0f;
    for (size_t index = 0; index < sparseOutput.data.size(); ++index)
        forwardParity = (std::max)(forwardParity, std::fabs(sparseOutput.data[index] - denseOutput.data[index]));

    Matrix outputGradient(static_cast<size_t>(embeddingDim), static_cast<size_t>(sequenceLength), 1.0f);
    Matrix queryGradS;
    Matrix keyGradS;
    Matrix valueGradS;
    Matrix outputGradS;
    Matrix queryGradD;
    Matrix keyGradD;
    Matrix valueGradD;
    Matrix outputGradD;
    Matrix inputGradS = sparsePath.backward(outputGradient, sparseCache, queryGradS, keyGradS, valueGradS, outputGradS);
    Matrix inputGradD = denseMaskedPath.backward(outputGradient, denseCache, queryGradD, keyGradD, valueGradD, outputGradD);

    float backwardParity = 0.0f;
    for (size_t index = 0; index < inputGradS.data.size(); ++index)
        backwardParity = (std::max)(backwardParity, std::fabs(inputGradS.data[index] - inputGradD.data[index]));

    SmokeLog::result("Sparse Attn S3", "embed=%d heads=%d seq=%d W=%d G=%d  fwd=%.2e  bwd=%.2e",
        embeddingDim, headCount, sequenceLength, windowSize, globalTokenCount, forwardParity, backwardParity);
}

void CausalSelfAttention::runGqaHostSmokeDemo() {
    const int embed = 64;
    const int heads = 8;
    const int kvHeads = 2;
    const int seq = 16;
    const int maxPos = 32;
    const unsigned seed = 41u;

    auto maxAbsDiff = [](const Matrix& a, const Matrix& b) -> float {
        if (a.data.size() != b.data.size())
            throw std::runtime_error("CausalSelfAttention GQA smoke size mismatch");
        float diff = 0.0f;
        for (size_t i = 0; i < a.data.size(); ++i)
            diff = (std::max)(diff, std::fabs(a.data[i] - b.data[i]));
        return diff;
    };

    auto makeInput = [&]() -> Matrix {
        Matrix input(static_cast<size_t>(embed), static_cast<size_t>(seq), 0.0f);
        unsigned state = 123u;
        for (size_t i = 0; i < input.data.size(); ++i) {
            state = state * 1664525u + 1013904223u;
            input.data[i] = (static_cast<float>(state >> 8) / 16777216.0f) * 2.0f - 1.0f;
        }
        return input;
    };

    // 1) MHA default ≡ explicit kvHeadCount == headCount
    CausalSelfAttention mhaDefault = CausalSelfAttention::create(embed, heads, maxPos, seed);
    CausalSelfAttention mhaExplicit = CausalSelfAttention::create(
        embed, heads, maxPos, seed, -1, 0, RotaryEmbedding::DefaultBase, heads);
    if (mhaDefault.kvHeadCount != heads || mhaExplicit.kvHeadCount != heads)
        throw std::runtime_error("CausalSelfAttention GQA smoke expected MHA kvHeadCount");
    if (static_cast<int>(mhaDefault.keyWeight.rows) != embed)
        throw std::runtime_error("CausalSelfAttention GQA smoke MHA key rows mismatch");

    const Matrix input = makeInput();
    CausalSelfAttentionCache cacheA;
    CausalSelfAttentionCache cacheB;
    Matrix outDefault = mhaDefault.forward(input, cacheA);
    Matrix outExplicit = mhaExplicit.forward(input, cacheB);
    const float mhaParity = maxAbsDiff(outDefault, outExplicit);
    if (mhaParity > 1.0e-6f)
        throw std::runtime_error("CausalSelfAttention GQA smoke MHA parity failed");

    Matrix outGrad(static_cast<size_t>(embed), static_cast<size_t>(seq), 0.01f);
    Matrix qgA, kgA, vgA, ogA, qgB, kgB, vgB, ogB;
    Matrix inGradA = mhaDefault.backward(outGrad, cacheA, qgA, kgA, vgA, ogA);
    Matrix inGradB = mhaExplicit.backward(outGrad, cacheB, qgB, kgB, vgB, ogB);
    const float mhaBwdParity = (std::max)(
        maxAbsDiff(inGradA, inGradB),
        (std::max)(
            maxAbsDiff(qgA, qgB),
            (std::max)(
                maxAbsDiff(kgA, kgB),
                (std::max)(maxAbsDiff(vgA, vgB), maxAbsDiff(ogA, ogB)))));
    if (mhaBwdParity > 1.0e-5f)
        throw std::runtime_error("CausalSelfAttention GQA smoke MHA backward parity failed");

    // 2) True GQA vs MHA with K/V heads repeated (same math)
    CausalSelfAttention gqa = CausalSelfAttention::create(
        embed, heads, maxPos, seed + 7u, -1, 0, RotaryEmbedding::DefaultBase, kvHeads);
    if (gqa.kvHeadCount != kvHeads)
        throw std::runtime_error("CausalSelfAttention GQA smoke kvHeadCount mismatch");
    if (static_cast<int>(gqa.keyWeight.rows) != kvHeads * (embed / heads)
        || static_cast<int>(gqa.valueWeight.rows) != kvHeads * (embed / heads))
        throw std::runtime_error("CausalSelfAttention GQA smoke key/value shape mismatch");

    CausalSelfAttention mhaExpanded = CausalSelfAttention::create(
        embed, heads, maxPos, seed + 7u, -1, 0, RotaryEmbedding::DefaultBase, heads);
    mhaExpanded.queryWeight = gqa.queryWeight;
    mhaExpanded.outputWeight = gqa.outputWeight;
    mhaExpanded.rotaryEmbedding = gqa.rotaryEmbedding;

    const int headDim = embed / heads;
    const int groupSize = heads / kvHeads;
    mhaExpanded.keyWeight = Matrix(static_cast<size_t>(embed), static_cast<size_t>(embed), 0.0f);
    mhaExpanded.valueWeight = Matrix(static_cast<size_t>(embed), static_cast<size_t>(embed), 0.0f);
    for (int qHead = 0; qHead < heads; ++qHead) {
        const int kvHead = qHead / groupSize;
        for (int row = 0; row < headDim; ++row) {
            for (size_t col = 0; col < static_cast<size_t>(embed); ++col) {
                mhaExpanded.keyWeight.at(static_cast<size_t>(qHead * headDim + row), col) =
                    gqa.keyWeight.at(static_cast<size_t>(kvHead * headDim + row), col);
                mhaExpanded.valueWeight.at(static_cast<size_t>(qHead * headDim + row), col) =
                    gqa.valueWeight.at(static_cast<size_t>(kvHead * headDim + row), col);
            }
        }
    }

    CausalSelfAttentionCache gqaCache;
    CausalSelfAttentionCache mhaCache;
    Matrix gqaOut = gqa.forward(input, gqaCache);
    Matrix mhaOut = mhaExpanded.forward(input, mhaCache);
    const float expandFwdParity = maxAbsDiff(gqaOut, mhaOut);
    if (expandFwdParity > 1.0e-5f)
        throw std::runtime_error("CausalSelfAttention GQA smoke expanded-KV forward parity failed");

    Matrix qgG, kgG, vgG, ogG, qgM, kgM, vgM, ogM;
    Matrix inGradG = gqa.backward(outGrad, gqaCache, qgG, kgG, vgG, ogG);
    Matrix inGradM = mhaExpanded.backward(outGrad, mhaCache, qgM, kgM, vgM, ogM);
    const float expandInParity = maxAbsDiff(inGradG, inGradM);
    const float expandQParity = maxAbsDiff(qgG, qgM);
    const float expandOParity = maxAbsDiff(ogG, ogM);
    if (expandInParity > 1.0e-4f || expandQParity > 1.0e-4f || expandOParity > 1.0e-4f)
        throw std::runtime_error("CausalSelfAttention GQA smoke expanded-KV input/Q/O grad parity failed");

    // K/V grads on GQA must equal sum of repeated MHA head grads
    Matrix kgExpandedSum = Matrix::zerosLike(gqa.keyWeight);
    Matrix vgExpandedSum = Matrix::zerosLike(gqa.valueWeight);
    for (int qHead = 0; qHead < heads; ++qHead) {
        const int kvHead = qHead / groupSize;
        for (int row = 0; row < headDim; ++row) {
            for (size_t col = 0; col < static_cast<size_t>(embed); ++col) {
                kgExpandedSum.at(static_cast<size_t>(kvHead * headDim + row), col) +=
                    kgM.at(static_cast<size_t>(qHead * headDim + row), col);
                vgExpandedSum.at(static_cast<size_t>(kvHead * headDim + row), col) +=
                    vgM.at(static_cast<size_t>(qHead * headDim + row), col);
            }
        }
    }
    const float expandKvParity = (std::max)(maxAbsDiff(kgG, kgExpandedSum), maxAbsDiff(vgG, vgExpandedSum));
    if (expandKvParity > 1.0e-4f)
        throw std::runtime_error("CausalSelfAttention GQA smoke expanded-KV K/V grad accumulate parity failed");

    SmokeLog::result(
        "CausalSelfAttention GQA host",
        "q=%d kv=%d  mhaFwd=%.2e  mhaBwd=%.2e  expandFwd=%.2e  expandIn=%.2e  expandKV=%.2e",
        heads,
        kvHeads,
        mhaParity,
        mhaBwdParity,
        expandFwdParity,
        expandInParity,
        expandKvParity);
}
