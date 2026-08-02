#ifndef CUDACAUSALSELFATTENTION_HPP
#define CUDACAUSALSELFATTENTION_HPP

#include "../Layers/CausalSelfAttention.hpp"
#include "CudaKvCache.hpp"
#include "CudaMatmul.hpp"

#include <vector>

/// <summary>device resident causal multi head attention forward with RoPE and optional sparse mask</summary>
class CudaCausalSelfAttention {
public:
    CudaMatrix queryWeight;
    CudaMatrix keyWeight;
    CudaMatrix valueWeight;
    CudaMatrix outputWeight;

    /// <summary>stacked [query; key; value] for one forward projection GEMM</summary>
    CudaMatrix qkvWeight;
    CudaMatrix qkvProjected;
    /// <summary>stacked [dW_q; dW_k; dW_v] scratch for fused backward weight GEMM</summary>
    CudaMatrix qkvWeightGradient;

    CudaMatrix cosTable;
    CudaMatrix sinTable;
    int headCount;
    int headDimension;
    int pairCount;
    int maximumPositionCount;
    int windowSize;
    int globalTokenCount;
    int activeSegmentLength;
    int activePackCount;
    bool preferFlashAttention;
    bool usedFlashAttention;

    CudaMatrix query;
    CudaMatrix key;
    CudaMatrix value;
    CudaMatrix queryHead;
    CudaMatrix keyHead;
    CudaMatrix valueHead;
    CudaMatrix scores;
    CudaMatrix probabilities;
    CudaMatrix attendedHead;
    CudaMatrix attended;
    CudaMatrix output;

    CudaMatrix querySegment;
    CudaMatrix keySegment;
    CudaMatrix valueSegment;
    CudaMatrix attendedSegment;
    CudaMatrix attendedGradientSegment;
    CudaMatrix queryGradientSegment;
    CudaMatrix keyGradientSegment;
    CudaMatrix valueGradientSegment;

    CudaMatrix inputCache;
    CudaMatrix attendedGradient;
    CudaMatrix queryGradient;
    CudaMatrix keyGradient;
    CudaMatrix valueGradient;
    CudaMatrix probabilityGradient;
    CudaMatrix scoreGradient;
    CudaMatrix valueHeadGradient;
    CudaMatrix queryHeadGradient;
    CudaMatrix keyHeadGradient;
    CudaMatrix temp;
    std::vector<CudaMatrix> cachedHeadProbabilities;
    CudaMatrix flashLogSumExp;
    CudaMatrix flashDelta;

    CudaCausalSelfAttention();

    /// <summary>upload weights and RoPE tables from host attention</summary>
    void uploadFrom(const CausalSelfAttention& host);

    /// <summary>rebuild stacked QKV weight after Adam updates the split masters</summary>
    void syncFusedQkvWeight();

    /// <summary>create device attention from host</summary>
    static CudaCausalSelfAttention createFrom(const CausalSelfAttention& host);

    /// <summary>release dense-only scratch (scores, P cache, pack segment copies)</summary>
    void releaseDenseAttentionScratch();

    /// <summary>release flash-only scratch (LSE, delta workspace)</summary>
    void releaseFlashAttentionScratch();

    /// <summary>free QKV/Flash activations after forward (weights kept) for selective checkpointing</summary>
    void releaseActivationScratch();

    /// <summary>causal multi head attention writing into out</summary>
    void forward(const CudaMatrix& input, CudaMatrix& out, int segmentLength = 0);

    /// <summary>backprop through multi head attention fills weight gradients via out params</summary>
    void backward(const CudaMatrix& outputGradient, CudaMatrix& inputGradient, CudaMatrix& queryWeightGradient, CudaMatrix& keyWeightGradient, CudaMatrix& valueWeightGradient, CudaMatrix& outputWeightGradient);

    /// <summary>reset cache then run full sequence attention and append rotated KV</summary>
    void prefill(const CudaMatrix& input, CudaKvCache& cache, CudaMatrix& out);

    /// <summary>single token attention against append only KV cache</summary>
    void decode(const CudaMatrix& input, CudaKvCache& cache, CudaMatrix& out);

    /// <summary>compare CPU attention vs device forward</summary>
    static void runSmokeDemo(int embeddingDim = 64, int headCount = 4, int sequenceLength = 32, int maximumPositionCount = 64);

    /// <summary>compare CPU vs CUDA attention backward with dense W=max mask</summary>
    static void runBackwardSmokeDemo(int embeddingDim = 32, int headCount = 2, int sequenceLength = 16, int maximumPositionCount = 32);

    /// <summary>compare prefill plus decode last token vs full forward last column</summary>
    static void runKvCacheSmokeDemo(int embeddingDim = 64, int headCount = 4, int sequenceLength = 32, int maximumPositionCount = 64);

    /// <summary>compare CPU sparse window+global forward vs CUDA</summary>
    static void runSparseSmokeDemo(int embeddingDim = 32, int headCount = 2, int sequenceLength = 16, int maximumPositionCount = 32, int windowSize = 4, int globalTokenCount = 2);

    /// <summary>compare flash vs materialize dense causal forward and backward</summary>
    static void runFlashParitySmokeDemo(int embeddingDim = 64, int headCount = 4, int sequenceLength = 48, int maximumPositionCount = 64);

private:
    void projectAndRotate(const CudaMatrix& input, int positionOffset, int segmentLength = 0);
    void attendFullSequence(CudaMatrix& out, int segmentLength = -1);
    void attendCachedQuery(const CudaKvCache& cache, CudaMatrix& out);
    bool canUseFlashAttention(int segmentLength) const;
};

#endif // CUDACAUSALSELFATTENTION_HPP
