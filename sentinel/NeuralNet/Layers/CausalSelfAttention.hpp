#ifndef CAUSALSELFATTENTION_HPP
#define CAUSALSELFATTENTION_HPP

#include "../Math/Matrix.hpp"

#include <vector>

/// <summary>thread local intermediates for one multi head attention forward</summary>
class CausalSelfAttentionCache {
public:
    Matrix input;
    Matrix query;
    Matrix key;
    Matrix value;
    std::vector<Matrix> scores;
    std::vector<Matrix> probabilities;
    Matrix attended;
};

/// <summary>
/// multi head causal self attention
/// input/output shape: embeddingDim x sequenceLength
/// </summary>
class CausalSelfAttention {
public:
    Matrix queryWeight;
    Matrix keyWeight;
    Matrix valueWeight;
    Matrix outputWeight;
    int headCount;
    int headDimension;

    CausalSelfAttention(Matrix queryWeight, Matrix keyWeight, Matrix valueWeight, Matrix outputWeight, int headCount);

    /// <summary>create dim x dim weights headCount must divide embeddingDim</summary>
    static CausalSelfAttention create(int embeddingDim, int headCount = 4, unsigned seed = 11u);

    /// <summary>causal multi head attention writes intermediates into cache</summary>
    Matrix forward(const Matrix& input, CausalSelfAttentionCache& cache) const;

    /// <summary>backprop through multi head attention fills weight gradients via out params</summary>
    Matrix backward(const Matrix& outputGradient, const CausalSelfAttentionCache& cache, Matrix& queryWeightGradient, Matrix& keyWeightGradient, Matrix& valueWeightGradient, Matrix& outputWeightGradient) const;

private:
    /// <summary>column wise softmax jacobian applied to probability gradients</summary>
    static Matrix softmaxBackward(const Matrix& probabilities, const Matrix& probabilityGradient);

    /// <summary>zero matrix matching shape</summary>
    static Matrix zerosLike(const Matrix& matrix);

    /// <summary>copy one head block of rows from a full projection</summary>
    static Matrix extractHead(const Matrix& full, int headIndex, int headDimension);

    /// <summary>write one head block of rows into a full projection</summary>
    static void writeHead(Matrix& full, int headIndex, int headDimension, const Matrix& head);
};

#endif // CAUSALSELFATTENTION_HPP
