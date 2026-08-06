#ifndef TRANSFORMERBLOCK_HPP
#define TRANSFORMERBLOCK_HPP

#include "CausalSelfAttention.hpp"
#include "FeedForward.hpp"
#include "RMSNorm.hpp"
#include "../Math/Matrix.hpp"
#include "../Optimizers/Adam.hpp"

class TransformerBlock;

/// <summary>thread local caches for one transformer block forward</summary>
class TransformerBlockCache {
public:
    RMSNormCache attentionNormCache;
    CausalSelfAttentionCache attentionCache;
    RMSNormCache feedForwardNormCache;
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
    Matrix feedForwardNormGamma;
    Matrix feedForwardGateWeight;
    Matrix feedForwardGateBias;
    Matrix feedForwardUpWeight;
    Matrix feedForwardUpBias;
    Matrix feedForwardDownWeight;
    Matrix feedForwardDownBias;

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
/// RMSNorm -> multi head Attn -> residual -> RMSNorm -> SwiGLU FFN -> residual
/// </summary>
class TransformerBlock {
public:
    RMSNorm attentionNorm;
    CausalSelfAttention attention;
    RMSNorm feedForwardNorm;
    FeedForward feedForward;

    AdamState queryWeightState;
    AdamState keyWeightState;
    AdamState valueWeightState;
    AdamState attentionOutputWeightState;
    AdamState attentionNormGammaState;
    AdamState feedForwardNormGammaState;
    AdamState feedForwardGateWeightState;
    AdamState feedForwardGateBiasState;
    AdamState feedForwardUpWeightState;
    AdamState feedForwardUpBiasState;
    AdamState feedForwardDownWeightState;
    AdamState feedForwardDownBiasState;

    MuonState queryWeightMuon;
    MuonState keyWeightMuon;
    MuonState valueWeightMuon;
    MuonState attentionOutputWeightMuon;
    MuonState feedForwardGateWeightMuon;
    MuonState feedForwardUpWeightMuon;
    MuonState feedForwardDownWeightMuon;

    /// <summary>
    /// create pre-norm block; intermediateSize &lt;= 0 uses FeedForward::defaultIntermediateSize(embed, 4);
    /// ropeTheta is HF rope_theta (RoPE base), default 10000;
    /// useBias=false → zero FFN biases (HF models without MLP bias);
    /// kvHeadCount&lt;=0 → MHA (same as headCount)
    /// </summary>
    TransformerBlock(
        int embeddingDim,
        int headCount,
        int maximumPositionCount,
        unsigned seed,
        int intermediateSize = 0,
        float ropeTheta = RotaryEmbedding::DefaultBase,
        bool useBias = true,
        int kvHeadCount = -1);

    /// <summary>forward through one block writes cache</summary>
    Matrix forward(const Matrix& input, TransformerBlockCache& cache) const;

    /// <summary>backprop through one block accumulates into gradients returns input gradient</summary>
    Matrix backward(const Matrix& outputGradient, TransformerBlockCache& cache, TransformerBlockGradients& gradients) const;

    /// <summary>apply adam updates for this block</summary>
    void applyGradients(Adam& optimizer, const TransformerBlockGradients& gradients);
};

#endif // TRANSFORMERBLOCK_HPP
