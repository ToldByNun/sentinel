#ifndef CUDAFLASHATTENTION_HPP
#define CUDAFLASHATTENTION_HPP

#include "CudaMatmul.hpp"

/// <summary>
/// tiled causal attention with online softmax
/// forward stores per-query log-sum-exp for backward
/// multi-head tensors are (headCount * headDim) x sequenceLength
/// single-head API is a thin wrapper with headCount = 1
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
    /// multi-head flash over columns [columnStart, columnStart + columnCount)
    /// with packCount > 1 launches grid.z packs of equal columnCount
    /// query/key/value/out are (headCount * headDim) x strideColumns
    /// logSumExp is headCount x strideColumns
    /// </summary>
    static void forwardMultiHead(const CudaMatrix& query, const CudaMatrix& key, const CudaMatrix& value, CudaMatrix& out, CudaMatrix& logSumExp, int headCount, int headDimension, float scale, bool causal = true, int columnStart = 0, int columnCount = 0, int packCount = 1);

    /// <summary>
    /// gradients dQ dK dV from dO using recomputed P = exp(S - LSE) and D = rowsum(dO odot O)
    /// key-tile outer keeps dK/dV atomic free; dQ uses sparse global atomics
    /// </summary>
    static void backward(const CudaMatrix& query, const CudaMatrix& key, const CudaMatrix& value, const CudaMatrix& out, const CudaMatrix& logSumExp, const CudaMatrix& outGradient, CudaMatrix& queryGradient, CudaMatrix& keyGradient, CudaMatrix& valueGradient, float scale, bool causal = true);

    /// <summary>multi-head backward matching forwardMultiHead layout, column window, and packCount</summary>
    /// <param name="deltaWorkspace">reusable headCount x (packCount * columnCount) buffer for D</param>
    static void backwardMultiHead(const CudaMatrix& query, const CudaMatrix& key, const CudaMatrix& value, const CudaMatrix& out, const CudaMatrix& logSumExp, const CudaMatrix& outGradient, CudaMatrix& queryGradient, CudaMatrix& keyGradient, CudaMatrix& valueGradient, CudaMatrix& deltaWorkspace, int headCount, int headDimension, float scale, bool causal = true, int columnStart = 0, int columnCount = 0, int packCount = 1);
};

#endif // CUDAFLASHATTENTION_HPP
