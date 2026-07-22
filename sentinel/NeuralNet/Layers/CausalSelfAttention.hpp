#ifndef CAUSALSELFATTENTION_HPP
#define CAUSALSELFATTENTION_HPP

#include "RotaryEmbedding.hpp"
#include "../Math/Matrix.hpp"

#include <vector>

/// <summary>thread local intermediates and scratch for one multi head attention pass</summary>
class CausalSelfAttentionCache {
public:
    Matrix input;
    Matrix query;
    Matrix key;
    Matrix value;
    std::vector<Matrix> scores;
    std::vector<Matrix> probabilities;
    Matrix attended;
    Matrix output;

    Matrix queryHead;
    Matrix keyHead;
    Matrix valueHead;
    Matrix attendedHead;

    Matrix attendedGradient;
    Matrix queryGradient;
    Matrix keyGradient;
    Matrix valueGradient;
    Matrix probabilityGradient;
    Matrix scoreGradient;
    Matrix valueHeadGradient;
    Matrix queryHeadGradient;
    Matrix keyHeadGradient;
    Matrix inputGradient;
    Matrix temp;
};

/// <summary>
/// multi head causal self attention with RoPE on Q and K
/// input/output shape: embeddingDim x sequenceLength
/// </summary>
class CausalSelfAttention {
public:
    Matrix queryWeight;
    Matrix keyWeight;
    Matrix valueWeight;
    Matrix outputWeight;
    RotaryEmbedding rotaryEmbedding;
    int headCount;
    int headDimension;

    CausalSelfAttention(Matrix queryWeight, Matrix keyWeight, Matrix valueWeight, Matrix outputWeight, RotaryEmbedding rotaryEmbedding, int headCount);

    /// <summary>create dim x dim weights headCount must divide embeddingDim</summary>
    static CausalSelfAttention create(int embeddingDim, int headCount, int maximumPositionCount, unsigned seed = 11u);

    /// <summary>causal multi head attention writes intermediates into cache</summary>
    Matrix forward(const Matrix& input, CausalSelfAttentionCache& cache) const;

    /// <summary>backprop through multi head attention fills weight gradients via out params</summary>
    Matrix backward(const Matrix& outputGradient, CausalSelfAttentionCache& cache, Matrix& queryWeightGradient, Matrix& keyWeightGradient, Matrix& valueWeightGradient, Matrix& outputWeightGradient) const;

private:
    /// <summary>column wise softmax jacobian writing into scoreGradient</summary>
    static void softmaxBackwardInto(const Matrix& probabilities, const Matrix& probabilityGradient, Matrix& scoreGradient);

    /// <summary>copy one head block of rows from a full projection into head</summary>
    static void extractHeadInto(const Matrix& full, int headIndex, int headDimension, Matrix& head);

    /// <summary>write one head block of rows into a full projection</summary>
    static void writeHead(Matrix& full, int headIndex, int headDimension, const Matrix& head);
};

#endif // CAUSALSELFATTENTION_HPP
