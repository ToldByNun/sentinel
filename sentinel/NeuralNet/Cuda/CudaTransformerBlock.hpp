#ifndef CUDATRANSFORMERBLOCK_HPP

#define CUDATRANSFORMERBLOCK_HPP



#include "../Layers/TransformerBlock.hpp"

#include "CudaCausalSelfAttention.hpp"

#include "CudaFeedForward.hpp"

#include "CudaKvCache.hpp"

#include "CudaRMSNorm.hpp"



/// <summary>device resident gradients for one transformer block</summary>

class CudaTransformerBlockGradients {

public:

    CudaMatrix queryWeight;

    CudaMatrix keyWeight;

    CudaMatrix valueWeight;

    CudaMatrix attentionOutputWeight;

    CudaMatrix attentionNormGamma;

    CudaMatrix feedForwardNormGamma;

    CudaMatrix feedForwardGateWeight;

    CudaMatrix feedForwardGateBias;

    CudaMatrix feedForwardUpWeight;

    CudaMatrix feedForwardUpBias;

    CudaMatrix feedForwardDownWeight;

    CudaMatrix feedForwardDownBias;



    /// <summary>ensure gradient tensor shapes match block parameters; largeWeightsOnHost skips 2D weight GPU grads</summary>

    void ensureFrom(const CudaTransformerBlock& block, bool largeWeightsOnHost = false);



    /// <summary>set all gradient tensors to zero in place</summary>

    void zeroInPlace();



    /// <summary>total += other</summary>

    void addInPlace(const CudaTransformerBlockGradients& other);



    /// <summary>scale every tensor</summary>

    void scaleInPlace(float scalar);

};

/// <summary>non-owning host accumulators for large 2D weight grads (preferHostGradients)</summary>
struct CudaTransformerBlockHostWeightGrads {
    Matrix* queryWeight = nullptr;
    Matrix* keyWeight = nullptr;
    Matrix* valueWeight = nullptr;
    Matrix* attentionOutputWeight = nullptr;
    Matrix* feedForwardGateWeight = nullptr;
    Matrix* feedForwardUpWeight = nullptr;
    Matrix* feedForwardDownWeight = nullptr;
};



/// <summary>device resident pre-norm transformer block forward and backward</summary>

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



    CudaMatrix feedForwardInputGradient;

    CudaMatrix afterAttentionFromFeedForward;

    CudaMatrix afterAttentionGradient;

    CudaMatrix attentionInputGradient;

    CudaMatrix inputFromAttention;

    CudaMatrix feedForwardGateWeightGradient;

    CudaMatrix feedForwardGateBiasGradient;

    CudaMatrix feedForwardUpWeightGradient;

    CudaMatrix feedForwardUpBiasGradient;

    CudaMatrix feedForwardDownWeightGradient;

    CudaMatrix feedForwardDownBiasGradient;

    CudaMatrix feedForwardNormGammaGradient;

    CudaMatrix queryWeightGradient;

    CudaMatrix keyWeightGradient;

    CudaMatrix valueWeightGradient;

    CudaMatrix attentionOutputWeightGradient;

    CudaMatrix attentionNormGammaGradient;



    CudaTransformerBlock();



    /// <summary>upload all host block parameters once</summary>

    void uploadFrom(const TransformerBlock& host);



    /// <summary>create device block from host block</summary>

    static CudaTransformerBlock createFrom(const TransformerBlock& host);



    /// <summary>RMSNorm Attn residual RMSNorm SwiGLU residual writing into out</summary>
    void forward(const CudaMatrix& input, CudaMatrix& out, int segmentLength = 0);

    /// <summary>
    /// train forward for Selective ckpt: full block, then drop Attn QKV/Flash scratch; FFN acts kept
    /// </summary>
    void forwardSelectiveTrain(const CudaMatrix& input, CudaMatrix& out, int segmentLength = 0);

    /// <summary>
    /// free Attn/FFN/RMSNorm activation scratch and block intermediates (weights untouched).
    /// Safe after Full-ckpt forward (recomputed on bwd) and after any block backward.
    /// </summary>
    void releaseTrainActivationScratch();

    /// <summary>like releaseTrainActivationScratch; keepDeferredHostWeightGrads leaves large 2D weight grads for async D2H</summary>
    void releaseTrainActivationScratch(bool keepDeferredHostWeightGrads);

    /// <summary>recompute AttnNorm+Attn+residual from saved block input (FFN untouched)</summary>
    void recomputeAttention(const CudaMatrix& blockInput, int segmentLength = 0);

    /// <summary>prefill attention KV cache then residual FFN</summary>
    void prefill(const CudaMatrix& input, CudaKvCache& cache, CudaMatrix& out);

    /// <summary>decode one token against attention KV cache then residual FFN</summary>
    void decode(const CudaMatrix& input, CudaKvCache& cache, CudaMatrix& out);

    /// <summary>backprop through one block accumulates into gradients returns input gradient</summary>
    /// <param name="deferHostWeightDownload">keep large weight grads on device; caller downloads async then releaseDeferredHostWeightGradDevice()</param>
    /// <param name="retainActivationScratch">skip free of act/grad scratch (ckpt=off + CUDA graph friendly)</param>
    void backward(
        const CudaMatrix& outputGradient,
        CudaMatrix& inputGradient,
        CudaTransformerBlockGradients& gradients,
        CudaTransformerBlockHostWeightGrads* hostWeightGrads = nullptr,
        bool deferHostWeightDownload = false,
        bool retainActivationScratch = false);

    /// <summary>
    /// Selective bwd: FFN from kept acts, then recompute Attn from blockInput, then Attn bwd
    /// </summary>
    void backwardSelective(
        const CudaMatrix& outputGradient,
        CudaMatrix& inputGradient,
        CudaTransformerBlockGradients& gradients,
        const CudaMatrix& blockInput,
        int segmentLength = 0,
        CudaTransformerBlockHostWeightGrads* hostWeightGrads = nullptr,
        bool deferHostWeightDownload = false);

    /// <summary>queue D2H of deferred large weight grads into host on copyStream (after compute event)</summary>
    void enqueueDeferredHostWeightGradDownloads(
        CudaTransformerBlockHostWeightGrads& hostWeightGrads,
        cudaStream_t copyStream,
        cudaEvent_t gradsReadyOnCompute);

    /// <summary>free device large-weight grad workspaces after deferred D2H completed</summary>
    void releaseDeferredHostWeightGradDevice();



    /// <summary>compare CPU transformer block vs device forward</summary>

    static void runSmokeDemo(int embeddingDim = 64, int headCount = 4, int sequenceLength = 32, int maximumPositionCount = 64);



    /// <summary>compare CPU vs CUDA block backward maxAbsDiff</summary>

    static void runBackwardSmokeDemo(int embeddingDim = 64, int headCount = 4, int sequenceLength = 32, int maximumPositionCount = 64);

};



#endif // CUDATRANSFORMERBLOCK_HPP

