#ifndef CAUSALSELFATTENTION_HPP
#define CAUSALSELFATTENTION_HPP

#include "../Math/Matrix.hpp"

/// <summary>thread local intermediates for one attention forward</summary>
class CausalSelfAttentionCache {
public:
    Matrix input;
    Matrix query;
    Matrix key;
    Matrix value;
    Matrix scores;
    Matrix probabilities;
    Matrix attended;
};

/// <summary>
/// single head causal self attention
/// input/output shape: embeddingDim x sequenceLength
/// </summary>
class CausalSelfAttention {
public:
    Matrix queryWeight;
    Matrix keyWeight;
    Matrix valueWeight;
    Matrix outputWeight;

    CausalSelfAttention(Matrix queryWeight, Matrix keyWeight, Matrix valueWeight, Matrix outputWeight);

    /// <summary>create dim x dim weights with small random values</summary>
    static CausalSelfAttention create(int embeddingDim, unsigned seed = 11u);

    /// <summary>causal attention writes intermediates into cache (thread safe if cache is local)</summary>
    Matrix forward(const Matrix& input, CausalSelfAttentionCache& cache) const;

    /// <summary>
    /// backprop through attention using a cache from forward
    /// returns input gradient and fills weight gradients via out params
    /// </summary>
    Matrix backward(const Matrix& outputGradient, const CausalSelfAttentionCache& cache, Matrix& queryWeightGradient, Matrix& keyWeightGradient, Matrix& valueWeightGradient, Matrix& outputWeightGradient) const;

private:
    /// <summary>column wise softmax jacobian applied to probability gradients</summary>
    static Matrix softmaxBackward(const Matrix& probabilities, const Matrix& probabilityGradient);

    /// <summary>zero matrix matching shape</summary>
    static Matrix zerosLike(const Matrix& matrix);
};

#endif // CAUSALSELFATTENTION_HPP
