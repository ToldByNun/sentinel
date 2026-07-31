#ifndef CUDACAUSALSELFATTENTION_HPP
#define CUDACAUSALSELFATTENTION_HPP

#include "../Layers/CausalSelfAttention.hpp"
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

    /// <summary>compare CPU attention vs device forward</summary>
    static void runSmokeDemo(int embeddingDim = 64, int headCount = 4, int sequenceLength = 32, int maximumPositionCount = 64);
};

#endif // CUDACAUSALSELFATTENTION_HPP
