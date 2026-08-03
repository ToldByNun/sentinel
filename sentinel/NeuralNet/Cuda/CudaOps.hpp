#ifndef CUDAOPS_HPP
#define CUDAOPS_HPP

#include "CudaMatmul.hpp"

#ifdef __CUDACC__
#include <cuda_fp16.h>
#endif

/// <summary>elementwise broadcast attention and residual ops on CudaMatrix</summary>
class CudaOps {
public:
    /// <summary>add bias column across every sequence column in place</summary>
    static void broadcastBiasAddInPlace(CudaMatrix& product, const CudaMatrix& bias);

    /// <summary>SiLU writing into out</summary>
    static void siluInto(const CudaMatrix& input, CudaMatrix& out);

    /// <summary>out = silu(gatePre) * up in one pass</summary>
    static void siluMultiplyInto(const CudaMatrix& gatePreActivation, const CudaMatrix& up, CudaMatrix& out);

    /// <summary>
    /// from stacked [gatePreRaw|upRaw] (2H x T): add biases, write gatePre/up caches,
    /// gateActivated=silu(gate), hidden=silu(gate)*up
    /// </summary>
    static void swigluFromStackedPreBias(
        const CudaMatrix& stackedPreBias,
        const CudaMatrix& gateBias,
        const CudaMatrix& upBias,
        CudaMatrix& gatePreActivation,
        CudaMatrix& up,
        CudaMatrix& gateActivated,
        CudaMatrix& hidden);

    /// <summary>
    /// from stacked preactivations that already include bias: write gatePre/up caches,
    /// gateActivated=silu(gate), hidden=silu(gate)*up
    /// </summary>
    static void swigluFromStacked(
        const CudaMatrix& stackedPreActivation,
        CudaMatrix& gatePreActivation,
        CudaMatrix& up,
        CudaMatrix& gateActivated,
        CudaMatrix& hidden);

    /// <summary>
    /// SwiGLU bwd elementwise: write stacked [dGate; dUp] (2H x T) from dHidden
    /// also fills gateGradient/upGradient caches
    /// </summary>
    static void swigluBackwardIntoStacked(
        const CudaMatrix& hiddenGradient,
        const CudaMatrix& gatePreActivation,
        const CudaMatrix& up,
        const CudaMatrix& gateActivated,
        CudaMatrix& gateGradient,
        CudaMatrix& upGradient,
        CudaMatrix& stackedGateUpGradient);

    /// <summary>sum columns of stacked [A;B] halves into two bias gradient columns</summary>
    static void sumColumnsStackedHalvesInto(const CudaMatrix& stacked, CudaMatrix& firstBiasGradient, CudaMatrix& secondBiasGradient);

    /// <summary>SiLU derivative writing into out</summary>
    static void siluDerivativeInto(const CudaMatrix& input, CudaMatrix& out);

    /// <summary>out = left * right element wise</summary>
    static void multiplyElementwiseInto(const CudaMatrix& left, const CudaMatrix& right, CudaMatrix& out);

    /// <summary>total *= other element wise in place</summary>
    static void multiplyElementwiseInPlace(CudaMatrix& total, const CudaMatrix& other);

    /// <summary>out = left + right</summary>
    static void addInto(const CudaMatrix& left, const CudaMatrix& right, CudaMatrix& out);

    /// <summary>total += delta element wise in place</summary>
    static void addInPlace(CudaMatrix& total, const CudaMatrix& delta);

    /// <summary>sum gradient columns into biasGradient rows x 1</summary>
    static void sumColumnsInto(const CudaMatrix& gradient, CudaMatrix& biasGradient);

    /// <summary>device to device copy after ensureSize</summary>
    static void copyInto(const CudaMatrix& source, CudaMatrix& destination);

    /// <summary>matrix *= scalar in place</summary>
    static void scaleInPlace(CudaMatrix& matrix, float scalar);

    /// <summary>set every element to zero</summary>
    static void zeroInPlace(CudaMatrix& matrix);

    /// <summary>D2H device delta and add into host accumulator (synchronizes)</summary>
    static void downloadAddIntoHost(Matrix& hostTotal, const CudaMatrix& deviceDelta);

    /// <summary>if non-null, downloadAddIntoHost adds wall seconds into *sink</summary>
    static double* downloadAddIntoHostSecondsSink;

    /// <summary>copy one head block of rows from full into tightly packed head</summary>
    static void extractHeadInto(const CudaMatrix& full, int headIndex, int headDimension, CudaMatrix& head);

    /// <summary>copy one head block using source stride and used column prefix</summary>
    static void extractHeadInto(const CudaMatrix& full, int headIndex, int headDimension, int usedColumnCount, CudaMatrix& head);

    /// <summary>write one head block of rows into full</summary>
    static void writeHead(CudaMatrix& full, int headIndex, int headDimension, const CudaMatrix& head);

    /// <summary>write packed source columns into dest starting at destStartColumn</summary>
    static void writeColumnsInto(CudaMatrix& destination, int destinationStartColumn, const CudaMatrix& source);

    /// <summary>copy destination columns [start, start+count) into tightly packed out</summary>
    static void extractColumnsInto(const CudaMatrix& source, int sourceStartColumn, int columnCount, CudaMatrix& out);

    /// <summary>destination columns [start, ...) += source</summary>
    static void addColumnsInPlace(CudaMatrix& destination, int destinationStartColumn, const CudaMatrix& source);

    /// <summary>set scores keyIndex greater than queryIndex to large negative</summary>
    static void applyCausalMaskInPlace(CudaMatrix& scores);

    /// <summary>causal window plus global token mask queryAbsolute = queryPositionStart + queryCol</summary>
    static void applySparseAttentionMaskInPlace(CudaMatrix& scores, int windowSize, int globalTokenCount, int queryPositionStart = 0, int segmentLength = 0);

    /// <summary>column wise softmax writing into out</summary>
    static void softmaxInto(const CudaMatrix& logits, CudaMatrix& out);

    /// <summary>column wise softmax backward scores/probs rows=keys cols=queries</summary>
    static void softmaxBackwardInto(const CudaMatrix& probabilities, const CudaMatrix& probabilityGradient, CudaMatrix& scoreGradient);

    /// <summary>zero score gradients at sparse attention forbidden positions</summary>
    static void zeroForbiddenScoreGradientsInPlace(CudaMatrix& scoresGrad, int windowSize, int globalTokenCount, int queryPositionStart = 0, int segmentLength = 0);

    /// <summary>RoPE rotate Q or K in place column c uses position positionOffset + c</summary>
    static void rotaryRotateInPlace(CudaMatrix& tensor, int headCount, int headDimension, int pairCount, const CudaMatrix& cosTable, const CudaMatrix& sinTable, int positionOffset = 0, int segmentLength = 0);

    /// <summary>inverse RoPE rotate Q or K in place</summary>
    static void rotaryRotateInverseInPlace(CudaMatrix& tensor, int headCount, int headDimension, int pairCount, const CudaMatrix& cosTable, const CudaMatrix& sinTable, int positionOffset = 0, int segmentLength = 0);

    /// <summary>gather embedding rows into embedDim x tokenCount matrix</summary>
    static void embeddingGatherInto(const CudaMatrix& weight, const CudaIntBuffer& tokenIds, size_t tokenCount, CudaMatrix& out);

    /// <summary>gather using a raw device token-id pointer</summary>
    static void embeddingGatherInto(const CudaMatrix& weight, const int* tokenIdsDevice, size_t tokenCount, CudaMatrix& out);

    /// <summary>scatter add outputGradient into weightGradient rows by tokenId with atomicAdd</summary>
    static void embeddingScatterAddInto(CudaMatrix& weightGradient, const CudaIntBuffer& tokenIds, size_t tokenCount, const CudaMatrix& outputGradient);

    /// <summary>scatter add using a raw device token-id pointer</summary>
    static void embeddingScatterAddInto(CudaMatrix& weightGradient, const int* tokenIdsDevice, size_t tokenCount, const CudaMatrix& outputGradient);

    /// <summary>zero embedding gradient rows listed in tokenIds (duplicates ok)</summary>
    static void embeddingZeroRows(CudaMatrix& weightGradient, const int* tokenIdsDevice, size_t tokenCount);

    /// <summary>mean -log p[target] over sequence columns</summary>
    static float crossEntropyLossFromIds(const CudaMatrix& probabilities, const CudaIntBuffer& targetTokenIds, size_t tokenCount);

    /// <summary>add mean example loss into lossSum 1x1 without host sync</summary>
    static void crossEntropyAddMeanLossFromIds(const CudaMatrix& probabilities, const CudaIntBuffer& targetTokenIds, size_t tokenCount, CudaMatrix& lossSum);

    /// <summary>logit grad (softmax - onehot) / columns from index targets</summary>
    static void crossEntropyLogitGradientFromIdsInto(const CudaMatrix& probabilities, const CudaIntBuffer& targetTokenIds, size_t tokenCount, CudaMatrix& logitGradient);

    /// <summary>softmax + CE loss accumulate + logit grad meanDivisor defaults to tokenCount</summary>
    static void softmaxCrossEntropyFromLogitsInto(const CudaMatrix& logits, const CudaIntBuffer& targetTokenIds, size_t tokenCount, CudaMatrix& probabilities, CudaMatrix& logitGradient, CudaMatrix& lossSum, float lossScale = 1.0f, int meanDivisor = 0);

    /// <summary>set every element to value</summary>
    static void fillInPlace(CudaMatrix& matrix, float value);

    /// <summary>add bias rows [rowStart, rowStart+rowCount) across columns of product rowCount x cols</summary>
    static void broadcastBiasRowsAddInPlace(CudaMatrix& product, const CudaMatrix& bias, int rowStart, int rowCount);

    /// <summary>destination rows [rowStart, ...) += source</summary>
    static void addRowsInPlace(CudaMatrix& destination, int rowStart, const CudaMatrix& source);

    /// <summary>init online softmax max to -inf and sumExp to 0 for tokenCount columns</summary>
    static void onlineSoftmaxReset(CudaMatrix& maximumLogits, CudaMatrix& sumExp, size_t tokenCount);

    /// <summary>fold logit chunk into running max and sumExp one column per thread</summary>
    static void onlineSoftmaxUpdateFromChunk(const CudaMatrix& logitChunk, int chunkRows, size_t tokenCount, CudaMatrix& maximumLogits, CudaMatrix& sumExp);

    /// <summary>fold logit chunk pointer (chunkRows x tokenCount row-major) into online softmax stats</summary>
    static void onlineSoftmaxUpdateFromChunk(const float* logitChunk, int chunkRows, size_t tokenCount, CudaMatrix& maximumLogits, CudaMatrix& sumExp);

    /// <summary>if target id falls in chunk write that logit into targetLogits</summary>
    static void captureTargetLogitFromChunk(const CudaMatrix& logitChunk, const CudaIntBuffer& targetTokenIds, int rowStart, int chunkRows, size_t tokenCount, CudaMatrix& targetLogits);

    /// <summary>capture target logit from chunk pointer (chunkRows x tokenCount)</summary>
    static void captureTargetLogitFromChunk(const float* logitChunk, const CudaIntBuffer& targetTokenIds, int rowStart, int chunkRows, size_t tokenCount, CudaMatrix& targetLogits);

    /// <summary>add mean CE from online softmax stats into lossSum 1x1; skips ignoreIndex targets</summary>
    static void onlineSoftmaxAddMeanCrossEntropy(const CudaMatrix& targetLogits, const CudaMatrix& maximumLogits, const CudaMatrix& sumExp, size_t tokenCount, CudaMatrix& lossSum, float lossScale, int meanDivisor, const CudaIntBuffer* targetTokenIds = nullptr, int ignoreIndex = -1, const CudaIntBuffer* perColumnMeanDivisor = nullptr);

    /// <summary>write chunk logit grads (p - onehot) / meanDivisor * gradScale; ignored targets get zero grad</summary>
    static void onlineSoftmaxLogitGradientChunkInto(const CudaMatrix& logitChunk, const CudaIntBuffer& targetTokenIds, int rowStart, int chunkRows, size_t tokenCount, const CudaMatrix& maximumLogits, const CudaMatrix& sumExp, CudaMatrix& logitGradientChunk, float gradScale, int meanDivisor, int ignoreIndex = -1, const CudaIntBuffer* perColumnMeanDivisor = nullptr);

    /// <summary>write chunk logit grads from logit chunk pointer (chunkRows x tokenCount)</summary>
    static void onlineSoftmaxLogitGradientChunkInto(const float* logitChunk, const CudaIntBuffer& targetTokenIds, int rowStart, int chunkRows, size_t tokenCount, const CudaMatrix& maximumLogits, const CudaMatrix& sumExp, CudaMatrix& logitGradientChunk, float gradScale, int meanDivisor, int ignoreIndex = -1, const CudaIntBuffer* perColumnMeanDivisor = nullptr);

    /// <summary>sum gradient columns into biasGradient rows starting at rowStart</summary>
    static void sumColumnsAddIntoRows(const CudaMatrix& gradient, CudaMatrix& biasGradient, int rowStart);

private:
    static constexpr int threadCount = 256;

#ifdef __CUDACC__
    __device__ static void runBroadcastBiasAddInPlace(float* product, const float* bias, int rowCount, int columnCount);
    __device__ static void runSiluInto(const float* input, float* out, int elementCount);
    __device__ static void runSiluDerivativeInto(const float* input, float* out, int elementCount);
    __device__ static void runMultiplyElementwiseInto(const float* left, const float* right, float* out, int elementCount);
    __device__ static void runMultiplyElementwiseInPlace(float* total, const float* other, int elementCount);
    __device__ static void runAddInto(const float* left, const float* right, float* out, int elementCount);
    __device__ static void runAddInPlace(float* total, const float* delta, int elementCount);
    __device__ static void runSumColumnsInto(const float* gradient, float* biasGradient, int rowCount, int columnCount);
    __device__ static void runScaleInPlace(float* matrix, float scalar, int elementCount);
    __device__ static void runZeroInPlace(float* matrix, int elementCount);
    __device__ static void runExtractHeadInto(const float* full, float* head, int headIndex, int headDimension, int sourceStrideColumns, int usedColumnCount);
    __device__ static void runWriteHead(float* full, const float* head, int headIndex, int headDimension, int sequenceLength, int embeddingDim);
    __device__ static void runWriteColumnsInto(float* destination, const float* source, int embeddingDim, int destinationStrideColumns, int destinationStartColumn, int sourceColumnCount);
    __device__ static void runExtractColumnsInto(const float* source, float* out, int embeddingDim, int sourceStrideColumns, int sourceStartColumn, int columnCount);
    __device__ static void runAddColumnsInPlace(float* destination, const float* source, int embeddingDim, int destinationStrideColumns, int destinationStartColumn, int sourceColumnCount);
    __device__ static void runApplyCausalMaskInPlace(float* scores, int sequenceLength);
    __device__ static void runApplySparseAttentionMaskInPlace(float* scores, int keyCount, int queryCount, int queryPositionStart, int windowSize, int globalTokenCount, int segmentLength);
    __device__ static void runSoftmaxInto(const float* logits, float* out, int rowCount, int columnCount);
    __device__ static void runSoftmaxBackwardInto(const float* probabilities, const float* probabilityGradient, float* scoreGradient, int rowCount, int columnCount);
    __device__ static void runZeroForbiddenScoreGradientsInPlace(float* scoresGrad, int keyCount, int queryCount, int queryPositionStart, int windowSize, int globalTokenCount, int segmentLength);
    __device__ static void runRotaryRotateInPlace(float* tensor, int headCount, int headDimension, int pairCount, int sequenceLength, int positionOffset, int segmentLength, const float* cosTable, const float* sinTable);
    __device__ static void runRotaryRotateInverseInPlace(float* tensor, int headCount, int headDimension, int pairCount, int sequenceLength, int positionOffset, int segmentLength, const float* cosTable, const float* sinTable);
    __device__ static void runEmbeddingGatherInto(const float* weight, const int* tokenIds, float* out, int embeddingDim, int tokenCount, int vocabularySize);
    __device__ static void runEmbeddingGatherHalfInto(const __half* weight, const int* tokenIds, float* out, int embeddingDim, int tokenCount, int vocabularySize);
    __device__ static void runEmbeddingScatterAddInto(float* weightGradient, const int* tokenIds, const float* outputGradient, int embeddingDim, int tokenCount, int vocabularySize);
    __device__ static void runEmbeddingZeroRows(float* weightGradient, const int* tokenIds, int embeddingDim, int tokenCount, int vocabularySize);
    __device__ static void runCrossEntropyLossFromIds(const float* probabilities, const int* targetTokenIds, float* columnLosses, int vocabularySize, int tokenCount);
    __device__ static void runCrossEntropyAddMeanLossFromIds(const float* probabilities, const int* targetTokenIds, float* lossSum, int vocabularySize, int tokenCount);
    __device__ static void runCrossEntropyLogitGradientFromIds(const float* probabilities, const int* targetTokenIds, float* logitGradient, int vocabularySize, int tokenCount);
    __device__ static void runSoftmaxCrossEntropyFromLogits(const float* logits, const int* targetTokenIds, float* probabilities, float* logitGradient, float* lossSum, int vocabularySize, int tokenCount, float lossScale, int meanDivisor);

    friend __global__ void CudaOpsBroadcastBiasAddEntry(float* product, const float* bias, int rowCount, int columnCount);
    friend __global__ void CudaOpsSiluEntry(const float* input, float* out, int elementCount);
    friend __global__ void CudaOpsSiluDerivativeEntry(const float* input, float* out, int elementCount);
    friend __global__ void CudaOpsMultiplyElementwiseEntry(const float* left, const float* right, float* out, int elementCount);
    friend __global__ void CudaOpsMultiplyElementwiseInPlaceEntry(float* total, const float* other, int elementCount);
    friend __global__ void CudaOpsAddEntry(const float* left, const float* right, float* out, int elementCount);
    friend __global__ void CudaOpsAddInPlaceEntry(float* total, const float* delta, int elementCount);
    friend __global__ void CudaOpsSumColumnsEntry(const float* gradient, float* biasGradient, int rowCount, int columnCount);
    friend __global__ void CudaOpsScaleEntry(float* matrix, float scalar, int elementCount);
    friend __global__ void CudaOpsZeroEntry(float* matrix, int elementCount);
    friend __global__ void CudaOpsExtractHeadEntry(const float* full, float* head, int headIndex, int headDimension, int sourceStrideColumns, int usedColumnCount);
    friend __global__ void CudaOpsWriteHeadEntry(float* full, const float* head, int headIndex, int headDimension, int sequenceLength, int embeddingDim);
    friend __global__ void CudaOpsWriteColumnsEntry(float* destination, const float* source, int embeddingDim, int destinationStrideColumns, int destinationStartColumn, int sourceColumnCount);
    friend __global__ void CudaOpsExtractColumnsEntry(const float* source, float* out, int embeddingDim, int sourceStrideColumns, int sourceStartColumn, int columnCount);
    friend __global__ void CudaOpsAddColumnsEntry(float* destination, const float* source, int embeddingDim, int destinationStrideColumns, int destinationStartColumn, int sourceColumnCount);
    friend __global__ void CudaOpsCausalMaskEntry(float* scores, int sequenceLength);
    friend __global__ void CudaOpsSparseAttentionMaskEntry(float* scores, int keyCount, int queryCount, int queryPositionStart, int windowSize, int globalTokenCount, int segmentLength);
    friend __global__ void CudaOpsSoftmaxEntry(const float* logits, float* out, int rowCount, int columnCount);
    friend __global__ void CudaOpsSoftmaxBackwardEntry(const float* probabilities, const float* probabilityGradient, float* scoreGradient, int rowCount, int columnCount);
    friend __global__ void CudaOpsZeroForbiddenScoreGradientsEntry(float* scoresGrad, int keyCount, int queryCount, int queryPositionStart, int windowSize, int globalTokenCount, int segmentLength);
    friend __global__ void CudaOpsRotaryRotateEntry(float* tensor, int headCount, int headDimension, int pairCount, int sequenceLength, int positionOffset, int segmentLength, const float* cosTable, const float* sinTable);
    friend __global__ void CudaOpsRotaryRotateInverseEntry(float* tensor, int headCount, int headDimension, int pairCount, int sequenceLength, int positionOffset, int segmentLength, const float* cosTable, const float* sinTable);
    friend __global__ void CudaOpsEmbeddingGatherEntry(const float* weight, const int* tokenIds, float* out, int embeddingDim, int tokenCount, int vocabularySize);
    friend __global__ void CudaOpsEmbeddingGatherHalfEntry(const __half* weight, const int* tokenIds, float* out, int embeddingDim, int tokenCount, int vocabularySize);
    friend __global__ void CudaOpsEmbeddingScatterAddEntry(float* weightGradient, const int* tokenIds, const float* outputGradient, int embeddingDim, int tokenCount, int vocabularySize);
    friend __global__ void CudaOpsEmbeddingZeroRowsEntry(float* weightGradient, const int* tokenIds, int embeddingDim, int tokenCount, int vocabularySize);
    friend __global__ void CudaOpsCrossEntropyLossFromIdsEntry(const float* probabilities, const int* targetTokenIds, float* columnLosses, int vocabularySize, int tokenCount);
    friend __global__ void CudaOpsCrossEntropyAddMeanLossFromIdsEntry(const float* probabilities, const int* targetTokenIds, float* lossSum, int vocabularySize, int tokenCount);
    friend __global__ void CudaOpsCrossEntropyLogitGradientFromIdsEntry(const float* probabilities, const int* targetTokenIds, float* logitGradient, int vocabularySize, int tokenCount);
    friend __global__ void CudaOpsSoftmaxCrossEntropyFromLogitsEntry(const float* logits, const int* targetTokenIds, float* probabilities, float* logitGradient, float* lossSum, int vocabularySize, int tokenCount, float lossScale, int meanDivisor);
#endif
};

#ifdef __CUDACC__
__global__ void CudaOpsBroadcastBiasAddEntry(float* product, const float* bias, int rowCount, int columnCount);
__global__ void CudaOpsSiluEntry(const float* input, float* out, int elementCount);
__global__ void CudaOpsSiluDerivativeEntry(const float* input, float* out, int elementCount);
__global__ void CudaOpsMultiplyElementwiseEntry(const float* left, const float* right, float* out, int elementCount);
__global__ void CudaOpsMultiplyElementwiseInPlaceEntry(float* total, const float* other, int elementCount);
__global__ void CudaOpsAddEntry(const float* left, const float* right, float* out, int elementCount);
__global__ void CudaOpsAddInPlaceEntry(float* total, const float* delta, int elementCount);
__global__ void CudaOpsSumColumnsEntry(const float* gradient, float* biasGradient, int rowCount, int columnCount);
__global__ void CudaOpsScaleEntry(float* matrix, float scalar, int elementCount);
__global__ void CudaOpsZeroEntry(float* matrix, int elementCount);
__global__ void CudaOpsExtractHeadEntry(const float* full, float* head, int headIndex, int headDimension, int sourceStrideColumns, int usedColumnCount);
__global__ void CudaOpsWriteHeadEntry(float* full, const float* head, int headIndex, int headDimension, int sequenceLength, int embeddingDim);
__global__ void CudaOpsWriteColumnsEntry(float* destination, const float* source, int embeddingDim, int destinationStrideColumns, int destinationStartColumn, int sourceColumnCount);
__global__ void CudaOpsExtractColumnsEntry(const float* source, float* out, int embeddingDim, int sourceStrideColumns, int sourceStartColumn, int columnCount);
__global__ void CudaOpsAddColumnsEntry(float* destination, const float* source, int embeddingDim, int destinationStrideColumns, int destinationStartColumn, int sourceColumnCount);
__global__ void CudaOpsCausalMaskEntry(float* scores, int sequenceLength);
__global__ void CudaOpsSparseAttentionMaskEntry(float* scores, int keyCount, int queryCount, int queryPositionStart, int windowSize, int globalTokenCount, int segmentLength);
__global__ void CudaOpsSoftmaxEntry(const float* logits, float* out, int rowCount, int columnCount);
__global__ void CudaOpsSoftmaxBackwardEntry(const float* probabilities, const float* probabilityGradient, float* scoreGradient, int rowCount, int columnCount);
__global__ void CudaOpsZeroForbiddenScoreGradientsEntry(float* scoresGrad, int keyCount, int queryCount, int queryPositionStart, int windowSize, int globalTokenCount, int segmentLength);
__global__ void CudaOpsRotaryRotateEntry(float* tensor, int headCount, int headDimension, int pairCount, int sequenceLength, int positionOffset, int segmentLength, const float* cosTable, const float* sinTable);
__global__ void CudaOpsRotaryRotateInverseEntry(float* tensor, int headCount, int headDimension, int pairCount, int sequenceLength, int positionOffset, int segmentLength, const float* cosTable, const float* sinTable);
__global__ void CudaOpsEmbeddingGatherEntry(const float* weight, const int* tokenIds, float* out, int embeddingDim, int tokenCount, int vocabularySize);
__global__ void CudaOpsEmbeddingGatherHalfEntry(const __half* weight, const int* tokenIds, float* out, int embeddingDim, int tokenCount, int vocabularySize);
__global__ void CudaOpsEmbeddingScatterAddEntry(float* weightGradient, const int* tokenIds, const float* outputGradient, int embeddingDim, int tokenCount, int vocabularySize);
__global__ void CudaOpsEmbeddingZeroRowsEntry(float* weightGradient, const int* tokenIds, int embeddingDim, int tokenCount, int vocabularySize);
__global__ void CudaOpsCrossEntropyLossFromIdsEntry(const float* probabilities, const int* targetTokenIds, float* columnLosses, int vocabularySize, int tokenCount);
__global__ void CudaOpsCrossEntropyAddMeanLossFromIdsEntry(const float* probabilities, const int* targetTokenIds, float* lossSum, int vocabularySize, int tokenCount);
__global__ void CudaOpsCrossEntropyLogitGradientFromIdsEntry(const float* probabilities, const int* targetTokenIds, float* logitGradient, int vocabularySize, int tokenCount);
__global__ void CudaOpsSoftmaxCrossEntropyFromLogitsEntry(const float* logits, const int* targetTokenIds, float* probabilities, float* logitGradient, float* lossSum, int vocabularySize, int tokenCount, float lossScale, int meanDivisor);
#endif

#endif // CUDAOPS_HPP
