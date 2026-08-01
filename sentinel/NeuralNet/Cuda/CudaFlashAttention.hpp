#ifndef CUDAFLASHATTENTION_HPP
#define CUDAFLASHATTENTION_HPP

#include "CudaMatmul.hpp"

/// <summary>
/// tiled causal attention with online softmax
/// forward stores per-query log-sum-exp for backward
/// single head tensors are headDim x sequenceLength
/// </summary>
class CudaFlashAttention {
public:
    static constexpr int queryTileSize = 64;
    static constexpr int keyTileSize = 64;
    static constexpr int maxHeadDimension = 64;

    /// <summary>
    /// O = softmax(scale * Q K^T) V with optional causal mask
    /// logSumExp gets shape 1 x sequenceLength with m_i + log(l_i)
    /// </summary>
    static void forward(const CudaMatrix& query, const CudaMatrix& key, const CudaMatrix& value, CudaMatrix& out, CudaMatrix& logSumExp, float scale, bool causal = true);

    /// <summary>
    /// gradients dQ dK dV from dO using recomputed P = exp(S - LSE) and D = rowsum(dO odot O)
    /// key-tile outer keeps dK/dV atomic free; dQ uses sparse global atomics
    /// </summary>
    static void backward(const CudaMatrix& query, const CudaMatrix& key, const CudaMatrix& value, const CudaMatrix& out, const CudaMatrix& logSumExp, const CudaMatrix& outGradient, CudaMatrix& queryGradient, CudaMatrix& keyGradient, CudaMatrix& valueGradient, float scale, bool causal = true);
};

#endif // CUDAFLASHATTENTION_HPP
