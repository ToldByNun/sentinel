#ifndef CUDAFLASHATTENTION_HPP
#define CUDAFLASHATTENTION_HPP

#include "CudaMatmul.hpp"

/// <summary>
/// tiled causal attention forward with online softmax
/// stores per-query log-sum-exp for a later backward
/// single head tensors are headDim x sequenceLength
/// </summary>
class CudaFlashAttention {
public:
    static constexpr int queryTileSize = 16;
    static constexpr int keyTileSize = 16;
    static constexpr int maxHeadDimension = 64;

    /// <summary>
    /// O = softmax(scale * Q K^T) V with optional causal mask
    /// logSumExp gets shape 1 x sequenceLength with m_i + log(l_i)
    /// </summary>
    static void forward(const CudaMatrix& query, const CudaMatrix& key, const CudaMatrix& value, CudaMatrix& out, CudaMatrix& logSumExp, float scale, bool causal = true);
};

#endif // CUDAFLASHATTENTION_HPP
