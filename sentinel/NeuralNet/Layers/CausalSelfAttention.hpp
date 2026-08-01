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
/// optional block-sparse mask: sliding window plus first globalTokenCount keys
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
    int windowSize;
    int globalTokenCount;
    bool preferSparseCompute;

    CausalSelfAttention(Matrix queryWeight, Matrix keyWeight, Matrix valueWeight, Matrix outputWeight, RotaryEmbedding rotaryEmbedding, int headCount, int windowSize, int globalTokenCount);

    /// <summary>create dim x dim weights headCount must divide embeddingDim windowSize&lt;=0 means full causal</summary>
    static CausalSelfAttention create(int embeddingDim, int headCount, int maximumPositionCount, unsigned seed = 11u, int windowSize = -1, int globalTokenCount = 0);

    /// <summary>causal multi head attention writes intermediates into cache</summary>
    Matrix forward(const Matrix& input, CausalSelfAttentionCache& cache) const;

    /// <summary>backprop through multi head attention fills weight gradients via out params</summary>
    Matrix backward(const Matrix& outputGradient, CausalSelfAttentionCache& cache, Matrix& queryWeightGradient, Matrix& keyWeightGradient, Matrix& valueWeightGradient, Matrix& outputWeightGradient) const;

    /// <summary>smoke dense default vs sparse window+global mask probabilities</summary>
    static void runSparseMaskSmokeDemo(int embeddingDim = 32, int headCount = 2, int sequenceLength = 16, int maximumPositionCount = 32, int windowSize = 4, int globalTokenCount = 2);

    /// <summary>smoke sparse backward mask + finite difference + W=max gradient parity</summary>
    static void runSparseBackwardSmokeDemo(int embeddingDim = 32, int headCount = 2, int sequenceLength = 12, int maximumPositionCount = 32, int windowSize = 4, int globalTokenCount = 2);

    /// <summary>smoke sparse compute path vs dense masked path parity</summary>
    static void runSparseComputeSmokeDemo(int embeddingDim = 32, int headCount = 2, int sequenceLength = 16, int maximumPositionCount = 32, int windowSize = 4, int globalTokenCount = 2);

private:
    /// <summary>true when key may attend for this query under causal window/global rules</summary>
    static bool allowsKey(int queryIndex, int keyIndex, int windowSize, int globalTokenCount);

    /// <summary>true when sparse score loops beat dense gemm for this sequence</summary>
    bool usesSparseCompute(int sequenceLength) const;

    /// <summary>set disallowed scores to large negative</summary>
    static void applyAttentionMaskInPlace(Matrix& scores, int windowSize, int globalTokenCount);

    /// <summary>dense K^T Q scale mask into scores</summary>
    static void computeDenseMaskedScoresInto(const Matrix& queryHead, const Matrix& keyHead, Matrix& scores, float scale, int windowSize, int globalTokenCount);

    /// <summary>only allowed score entries O(seq*(W+G))</summary>
    static void computeSparseScoresInto(const Matrix& queryHead, const Matrix& keyHead, Matrix& scores, float scale, int windowSize, int globalTokenCount);

    /// <summary>dense V probabilities product</summary>
    static void attendDenseInto(const Matrix& valueHead, const Matrix& probabilities, Matrix& attendedHead);

    /// <summary>sparse V probabilities product over allowed keys</summary>
    static void attendSparseInto(const Matrix& valueHead, const Matrix& probabilities, Matrix& attendedHead, int windowSize, int globalTokenCount);

    /// <summary>column wise softmax jacobian writing into scoreGradient</summary>
    static void softmaxBackwardInto(const Matrix& probabilities, const Matrix& probabilityGradient, Matrix& scoreGradient);

    /// <summary>copy one head block of rows from a full projection into head</summary>
    static void extractHeadInto(const Matrix& full, int headIndex, int headDimension, Matrix& head);

    /// <summary>write one head block of rows into a full projection</summary>
    static void writeHead(Matrix& full, int headIndex, int headDimension, const Matrix& head);
};

#endif // CAUSALSELFATTENTION_HPP
