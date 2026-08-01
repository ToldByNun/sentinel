#ifndef CUDAOPS_HPP
#define CUDAOPS_HPP

#include "CudaMatmul.hpp"

/// <summary>elementwise broadcast attention and residual ops on CudaMatrix</summary>
class CudaOps {
public:
    /// <summary>add bias column across every sequence column in place</summary>
    static void broadcastBiasAddInPlace(CudaMatrix& product, const CudaMatrix& bias);

    /// <summary>SiLU writing into out</summary>
    static void siluInto(const CudaMatrix& input, CudaMatrix& out);

    /// <summary>out = left * right element wise</summary>
    static void multiplyElementwiseInto(const CudaMatrix& left, const CudaMatrix& right, CudaMatrix& out);

    /// <summary>out = left + right</summary>
    static void addInto(const CudaMatrix& left, const CudaMatrix& right, CudaMatrix& out);

    /// <summary>matrix *= scalar in place</summary>
    static void scaleInPlace(CudaMatrix& matrix, float scalar);

    /// <summary>set every element to zero</summary>
    static void zeroInPlace(CudaMatrix& matrix);

    /// <summary>copy one head block of rows from full into tightly packed head</summary>
    static void extractHeadInto(const CudaMatrix& full, int headIndex, int headDimension, CudaMatrix& head);

    /// <summary>copy one head block using source stride and used column prefix</summary>
    static void extractHeadInto(const CudaMatrix& full, int headIndex, int headDimension, int usedColumnCount, CudaMatrix& head);

    /// <summary>write one head block of rows into full</summary>
    static void writeHead(CudaMatrix& full, int headIndex, int headDimension, const CudaMatrix& head);

    /// <summary>write packed source columns into dest starting at destStartColumn</summary>
    static void writeColumnsInto(CudaMatrix& destination, int destinationStartColumn, const CudaMatrix& source);

    /// <summary>set scores keyIndex greater than queryIndex to large negative</summary>
    static void applyCausalMaskInPlace(CudaMatrix& scores);

    /// <summary>causal window plus global token mask queryAbsolute = queryPositionStart + queryCol</summary>
    static void applySparseAttentionMaskInPlace(CudaMatrix& scores, int windowSize, int globalTokenCount, int queryPositionStart = 0);

    /// <summary>column wise softmax writing into out</summary>
    static void softmaxInto(const CudaMatrix& logits, CudaMatrix& out);

    /// <summary>RoPE rotate Q or K in place column c uses position positionOffset + c</summary>
    static void rotaryRotateInPlace(CudaMatrix& tensor, int headCount, int headDimension, int pairCount, const CudaMatrix& cosTable, const CudaMatrix& sinTable, int positionOffset = 0);

    /// <summary>gather embedding rows into embedDim x tokenCount matrix</summary>
    static void embeddingGatherInto(const CudaMatrix& weight, const CudaIntBuffer& tokenIds, size_t tokenCount, CudaMatrix& out);

private:
    static constexpr int threadCount = 256;

#ifdef __CUDACC__
    __device__ static void runBroadcastBiasAddInPlace(float* product, const float* bias, int rowCount, int columnCount);
    __device__ static void runSiluInto(const float* input, float* out, int elementCount);
    __device__ static void runMultiplyElementwiseInto(const float* left, const float* right, float* out, int elementCount);
    __device__ static void runAddInto(const float* left, const float* right, float* out, int elementCount);
    __device__ static void runScaleInPlace(float* matrix, float scalar, int elementCount);
    __device__ static void runZeroInPlace(float* matrix, int elementCount);
    __device__ static void runExtractHeadInto(const float* full, float* head, int headIndex, int headDimension, int sourceStrideColumns, int usedColumnCount);
    __device__ static void runWriteHead(float* full, const float* head, int headIndex, int headDimension, int sequenceLength, int embeddingDim);
    __device__ static void runWriteColumnsInto(float* destination, const float* source, int embeddingDim, int destinationStrideColumns, int destinationStartColumn, int sourceColumnCount);
    __device__ static void runApplyCausalMaskInPlace(float* scores, int sequenceLength);
    __device__ static void runApplySparseAttentionMaskInPlace(float* scores, int keyCount, int queryCount, int queryPositionStart, int windowSize, int globalTokenCount);
    __device__ static void runSoftmaxInto(const float* logits, float* out, int rowCount, int columnCount);
    __device__ static void runRotaryRotateInPlace(float* tensor, int headCount, int headDimension, int pairCount, int sequenceLength, int positionOffset, const float* cosTable, const float* sinTable);
    __device__ static void runEmbeddingGatherInto(const float* weight, const int* tokenIds, float* out, int embeddingDim, int tokenCount, int vocabularySize);

    friend __global__ void CudaOpsBroadcastBiasAddEntry(float* product, const float* bias, int rowCount, int columnCount);
    friend __global__ void CudaOpsSiluEntry(const float* input, float* out, int elementCount);
    friend __global__ void CudaOpsMultiplyElementwiseEntry(const float* left, const float* right, float* out, int elementCount);
    friend __global__ void CudaOpsAddEntry(const float* left, const float* right, float* out, int elementCount);
    friend __global__ void CudaOpsScaleEntry(float* matrix, float scalar, int elementCount);
    friend __global__ void CudaOpsZeroEntry(float* matrix, int elementCount);
    friend __global__ void CudaOpsExtractHeadEntry(const float* full, float* head, int headIndex, int headDimension, int sourceStrideColumns, int usedColumnCount);
    friend __global__ void CudaOpsWriteHeadEntry(float* full, const float* head, int headIndex, int headDimension, int sequenceLength, int embeddingDim);
    friend __global__ void CudaOpsWriteColumnsEntry(float* destination, const float* source, int embeddingDim, int destinationStrideColumns, int destinationStartColumn, int sourceColumnCount);
    friend __global__ void CudaOpsCausalMaskEntry(float* scores, int sequenceLength);
    friend __global__ void CudaOpsSparseAttentionMaskEntry(float* scores, int keyCount, int queryCount, int queryPositionStart, int windowSize, int globalTokenCount);
    friend __global__ void CudaOpsSoftmaxEntry(const float* logits, float* out, int rowCount, int columnCount);
    friend __global__ void CudaOpsRotaryRotateEntry(float* tensor, int headCount, int headDimension, int pairCount, int sequenceLength, int positionOffset, const float* cosTable, const float* sinTable);
    friend __global__ void CudaOpsEmbeddingGatherEntry(const float* weight, const int* tokenIds, float* out, int embeddingDim, int tokenCount, int vocabularySize);
#endif
};

#ifdef __CUDACC__
__global__ void CudaOpsBroadcastBiasAddEntry(float* product, const float* bias, int rowCount, int columnCount);
__global__ void CudaOpsSiluEntry(const float* input, float* out, int elementCount);
__global__ void CudaOpsMultiplyElementwiseEntry(const float* left, const float* right, float* out, int elementCount);
__global__ void CudaOpsAddEntry(const float* left, const float* right, float* out, int elementCount);
__global__ void CudaOpsScaleEntry(float* matrix, float scalar, int elementCount);
__global__ void CudaOpsZeroEntry(float* matrix, int elementCount);
__global__ void CudaOpsExtractHeadEntry(const float* full, float* head, int headIndex, int headDimension, int sourceStrideColumns, int usedColumnCount);
__global__ void CudaOpsWriteHeadEntry(float* full, const float* head, int headIndex, int headDimension, int sequenceLength, int embeddingDim);
__global__ void CudaOpsWriteColumnsEntry(float* destination, const float* source, int embeddingDim, int destinationStrideColumns, int destinationStartColumn, int sourceColumnCount);
__global__ void CudaOpsCausalMaskEntry(float* scores, int sequenceLength);
__global__ void CudaOpsSparseAttentionMaskEntry(float* scores, int keyCount, int queryCount, int queryPositionStart, int windowSize, int globalTokenCount);
__global__ void CudaOpsSoftmaxEntry(const float* logits, float* out, int rowCount, int columnCount);
__global__ void CudaOpsRotaryRotateEntry(float* tensor, int headCount, int headDimension, int pairCount, int sequenceLength, int positionOffset, const float* cosTable, const float* sinTable);
__global__ void CudaOpsEmbeddingGatherEntry(const float* weight, const int* tokenIds, float* out, int embeddingDim, int tokenCount, int vocabularySize);
#endif

#endif // CUDAOPS_HPP
