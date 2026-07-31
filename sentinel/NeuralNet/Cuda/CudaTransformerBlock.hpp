#ifndef CUDATRANSFORMERBLOCK_HPP
#define CUDATRANSFORMERBLOCK_HPP

#include "../Layers/TransformerBlock.hpp"
#include "CudaCausalSelfAttention.hpp"
#include "CudaFeedForward.hpp"
#include "CudaRMSNorm.hpp"

/// <summary>device resident pre-norm transformer block forward</summary>
class CudaTransformerBlock {
public:
    CudaRMSNorm attentionNorm;
    CudaCausalSelfAttention attention;
    CudaRMSNorm feedForwardNorm;
    CudaFeedForward feedForward;

    CudaMatrix attentionInput;
    CudaMatrix attended;
    CudaMatrix afterAttention;
    CudaMatrix feedForwardInput;
    CudaMatrix feedForwardOutput;

    CudaTransformerBlock();

    /// <summary>upload all host block parameters once</summary>
    void uploadFrom(const TransformerBlock& host);

    /// <summary>create device block from host block</summary>
    static CudaTransformerBlock createFrom(const TransformerBlock& host);

    /// <summary>RMSNorm Attn residual RMSNorm SwiGLU residual writing into out</summary>
    void forward(const CudaMatrix& input, CudaMatrix& out);

    /// <summary>compare CPU transformer block vs device forward</summary>
    static void runSmokeDemo(int embeddingDim = 64, int headCount = 4, int sequenceLength = 32, int maximumPositionCount = 64);
};

#endif // CUDATRANSFORMERBLOCK_HPP
