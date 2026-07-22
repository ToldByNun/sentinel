#ifndef TRANSFORMERBLOCK_HPP
#define TRANSFORMERBLOCK_HPP

#include "CausalSelfAttention.hpp"
#include "FeedForward.hpp"
#include "LayerNorm.hpp"
#include "../Math/Matrix.hpp"
#include "../Optimizers/Adam.hpp"

class TransformerBlock;

/// <summary>thread local caches for one transformer block forward</summary>
class TransformerBlockCache {
public:
    LayerNormCache attentionNormCache;
    CausalSelfAttentionCache attentionCache;
    LayerNormCache feedForwardNormCache;
    FeedForwardCache feedForwardCache;
    Matrix input;
    Matrix afterAttention;
    Matrix output;
};

/// <summary>gradients for one transformer block</summary>
class TransformerBlockGradients {
public:
    Matrix queryWeight;
    Matrix keyWeight;
    Matrix valueWeight;
    Matrix attentionOutputWeight;
    Matrix attentionNormGamma;
    Matrix attentionNormBeta;
    Matrix feedForwardNormGamma;
    Matrix feedForwardNormBeta;
    Matrix feedForwardFirstWeight;
    Matrix feedForwardFirstBias;
    Matrix feedForwardSecondWeight;
    Matrix feedForwardSecondBias;

    /// <summary>zero tensors matching block parameter shapes</summary>
    static TransformerBlockGradients zerosFrom(const TransformerBlock& block);

    /// <summary>set all gradient tensors to zero in place</summary>
    void zeroInPlace();

    /// <summary>total += other</summary>
    void addInPlace(const TransformerBlockGradients& other);

    /// <summary>scale every tensor</summary>
    void scaleInPlace(float scalar);
};

/// <summary>
/// pre-norm transformer block
/// LN -> multi head Attn -> residual -> LN -> FFN -> residual
/// </summary>
class TransformerBlock {
public:
    LayerNorm attentionNorm;
    CausalSelfAttention attention;
    LayerNorm feedForwardNorm;
    FeedForward feedForward;

    AdamState queryWeightState;
    AdamState keyWeightState;
    AdamState valueWeightState;
    AdamState attentionOutputWeightState;
    AdamState attentionNormGammaState;
    AdamState attentionNormBetaState;
    AdamState feedForwardNormGammaState;
    AdamState feedForwardNormBetaState;
    AdamState feedForwardFirstWeightState;
    AdamState feedForwardFirstBiasState;
    AdamState feedForwardSecondWeightState;
    AdamState feedForwardSecondBiasState;

    TransformerBlock(int embeddingDim, int headCount, unsigned seed);

    /// <summary>forward through one block writes cache</summary>
    Matrix forward(const Matrix& input, TransformerBlockCache& cache) const;

    /// <summary>backprop through one block accumulates into gradients returns input gradient</summary>
    Matrix backward(const Matrix& outputGradient, const TransformerBlockCache& cache, TransformerBlockGradients& gradients) const;

    /// <summary>apply adam updates for this block</summary>
    void applyGradients(Adam& optimizer, const TransformerBlockGradients& gradients);
};

#endif // TRANSFORMERBLOCK_HPP
