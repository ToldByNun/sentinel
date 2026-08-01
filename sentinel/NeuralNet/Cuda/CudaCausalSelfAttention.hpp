#ifndef CUDACAUSALSELFATTENTION_HPP
#define CUDACAUSALSELFATTENTION_HPP

#include "../Layers/CausalSelfAttention.hpp"
#include "CudaKvCache.hpp"
#include "CudaMatmul.hpp"

/// <summary>device resident causal multi head attention forward with RoPE</summary>
class CudaCausalSelfAttention {
public:
    CudaMatrix queryWeight;
    CudaMatrix keyWeight;
    CudaMatrix valueWeight;
    CudaMatrix outputWeight;
    CudaMatrix cosTable;
    CudaMatrix sinTable;
    int headCount;
    int headDimension;
    int pairCount;
    int maximumPositionCount;

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

    CudaCausalSelfAttention();

    /// <summary>upload weights and RoPE tables from host attention</summary>
    void uploadFrom(const CausalSelfAttention& host);

    /// <summary>create device attention from host</summary>
    static CudaCausalSelfAttention createFrom(const CausalSelfAttention& host);

    /// <summary>causal multi head attention writing into out</summary>
    void forward(const CudaMatrix& input, CudaMatrix& out);

    /// <summary>reset cache then run full sequence attention and append rotated KV</summary>
    void prefill(const CudaMatrix& input, CudaKvCache& cache, CudaMatrix& out);

    /// <summary>single token attention against append only KV cache</summary>
    void decode(const CudaMatrix& input, CudaKvCache& cache, CudaMatrix& out);

    /// <summary>compare CPU attention vs device forward</summary>
    static void runSmokeDemo(int embeddingDim = 64, int headCount = 4, int sequenceLength = 32, int maximumPositionCount = 64);

    /// <summary>compare prefill plus decode last token vs full forward last column</summary>
    static void runKvCacheSmokeDemo(int embeddingDim = 64, int headCount = 4, int sequenceLength = 32, int maximumPositionCount = 64);

private:
    void projectAndRotate(const CudaMatrix& input, int positionOffset);
    void attendFullSequence(CudaMatrix& out);
    void attendCachedQuery(const CudaKvCache& cache, CudaMatrix& out);
};

#endif // CUDACAUSALSELFATTENTION_HPP
