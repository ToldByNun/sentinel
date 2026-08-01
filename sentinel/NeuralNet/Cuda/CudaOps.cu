#include "CudaOps.hpp"

#include <cuda_runtime.h>
#include <stdexcept>

__device__ void CudaOps::runBroadcastBiasAddInPlace(float* product, const float* bias, int rowCount, int columnCount) {
    const int elementCount = rowCount * columnCount;
    const int index = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (index >= elementCount) return;

    const int row = index / columnCount;
    product[index] += bias[row];
}

__device__ void CudaOps::runSiluInto(const float* input, float* out, int elementCount) {
    const int index = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (index >= elementCount) return;

    const float value = input[index];
    const float sigmoid = 1.0f / (1.0f + expf(-value));
    out[index] = value * sigmoid;
}

__device__ void CudaOps::runSiluDerivativeInto(const float* input, float* out, int elementCount) {
    const int index = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (index >= elementCount) return;

    const float value = input[index];
    const float sigmoid = 1.0f / (1.0f + expf(-value));
    out[index] = sigmoid * (1.0f + value * (1.0f - sigmoid));
}

__device__ void CudaOps::runMultiplyElementwiseInto(const float* left, const float* right, float* out, int elementCount) {
    const int index = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (index >= elementCount) return;

    out[index] = left[index] * right[index];
}

__device__ void CudaOps::runMultiplyElementwiseInPlace(float* total, const float* other, int elementCount) {
    const int index = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (index >= elementCount) return;

    total[index] *= other[index];
}

__device__ void CudaOps::runAddInto(const float* left, const float* right, float* out, int elementCount) {
    const int index = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (index >= elementCount) return;

    out[index] = left[index] + right[index];
}

__device__ void CudaOps::runAddInPlace(float* total, const float* delta, int elementCount) {
    const int index = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (index >= elementCount) return;

    total[index] += delta[index];
}

__device__ void CudaOps::runSumColumnsInto(const float* gradient, float* biasGradient, int rowCount, int columnCount) {
    const int row = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (row >= rowCount) return;

    float total = 0.0f;
    for (int column = 0; column < columnCount; ++column)
        total += gradient[row * columnCount + column];
    biasGradient[row] = total;
}

__device__ void CudaOps::runScaleInPlace(float* matrix, float scalar, int elementCount) {
    const int index = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (index >= elementCount) return;

    matrix[index] *= scalar;
}

__device__ void CudaOps::runZeroInPlace(float* matrix, int elementCount) {
    const int index = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (index >= elementCount) return;

    matrix[index] = 0.0f;
}

__device__ void CudaOps::runExtractHeadInto(const float* full, float* head, int headIndex, int headDimension, int sourceStrideColumns, int usedColumnCount) {
    const int elementCount = headDimension * usedColumnCount;
    const int index = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (index >= elementCount) return;

    const int row = index / usedColumnCount;
    const int column = index - row * usedColumnCount;
    const int fullRow = headIndex * headDimension + row;
    head[index] = full[fullRow * sourceStrideColumns + column];
}

__device__ void CudaOps::runWriteHead(float* full, const float* head, int headIndex, int headDimension, int sequenceLength, int embeddingDim) {
    const int elementCount = headDimension * sequenceLength;
    const int index = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (index >= elementCount) return;

    const int row = index / sequenceLength;
    const int column = index - row * sequenceLength;
    const int fullRow = headIndex * headDimension + row;
    full[fullRow * sequenceLength + column] = head[index];
    (void)embeddingDim;
}

__device__ void CudaOps::runWriteColumnsInto(float* destination, const float* source, int embeddingDim, int destinationStrideColumns, int destinationStartColumn, int sourceColumnCount) {
    const int elementCount = embeddingDim * sourceColumnCount;
    const int index = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (index >= elementCount) return;

    const int row = index / sourceColumnCount;
    const int sourceColumn = index - row * sourceColumnCount;
    const int destinationColumn = destinationStartColumn + sourceColumn;
    destination[row * destinationStrideColumns + destinationColumn] = source[index];
}

__device__ void CudaOps::runApplyCausalMaskInPlace(float* scores, int sequenceLength) {
    CudaOps::runApplySparseAttentionMaskInPlace(scores, sequenceLength, sequenceLength, 0, sequenceLength, 0);
}

__device__ void CudaOps::runApplySparseAttentionMaskInPlace(float* scores, int keyCount, int queryCount, int queryPositionStart, int windowSize, int globalTokenCount) {
    const int elementCount = keyCount * queryCount;
    const int index = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (index >= elementCount) return;

    const int keyIndex = index / queryCount;
    const int queryIndex = index - keyIndex * queryCount;
    const int absoluteQuery = queryPositionStart + queryIndex;
    const int absoluteKey = keyIndex;

    bool allowed = false;
    if (absoluteKey <= absoluteQuery) {
        if (absoluteKey < globalTokenCount)
            allowed = true;
        else if (windowSize <= 0)
            allowed = true;
        else if (absoluteKey >= absoluteQuery - windowSize + 1)
            allowed = true;
    }

    if (allowed) return;
    scores[index] = -1.0e9f;
}

__device__ void CudaOps::runSoftmaxInto(const float* logits, float* out, int rowCount, int columnCount) {
    const int column = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (column >= columnCount) return;

    float maximumLogit = logits[column];
    for (int row = 1; row < rowCount; ++row) {
        const float value = logits[row * columnCount + column];
        if (value > maximumLogit)
            maximumLogit = value;
    }

    float exponentialSum = 0.0f;
    for (int row = 0; row < rowCount; ++row) {
        const float value = expf(logits[row * columnCount + column] - maximumLogit);
        out[row * columnCount + column] = value;
        exponentialSum += value;
    }

    const float inverseSum = 1.0f / exponentialSum;
    for (int row = 0; row < rowCount; ++row)
        out[row * columnCount + column] *= inverseSum;
}

__device__ void CudaOps::runSoftmaxBackwardInto(const float* probabilities, const float* probabilityGradient, float* scoreGradient, int rowCount, int columnCount) {
    const int column = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (column >= columnCount) return;

    float dot = 0.0f;
    for (int row = 0; row < rowCount; ++row)
        dot += probabilityGradient[row * columnCount + column] * probabilities[row * columnCount + column];

    for (int row = 0; row < rowCount; ++row)
        scoreGradient[row * columnCount + column] = probabilities[row * columnCount + column] * (probabilityGradient[row * columnCount + column] - dot);
}

__device__ void CudaOps::runZeroForbiddenScoreGradientsInPlace(float* scoresGrad, int keyCount, int queryCount, int queryPositionStart, int windowSize, int globalTokenCount) {
    const int elementCount = keyCount * queryCount;
    const int index = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (index >= elementCount) return;

    const int keyIndex = index / queryCount;
    const int queryIndex = index - keyIndex * queryCount;
    const int absoluteQuery = queryPositionStart + queryIndex;
    const int absoluteKey = keyIndex;

    bool allowed = false;
    if (absoluteKey <= absoluteQuery) {
        if (absoluteKey < globalTokenCount)
            allowed = true;
        else if (windowSize <= 0)
            allowed = true;
        else if (absoluteKey >= absoluteQuery - windowSize + 1)
            allowed = true;
    }

    if (allowed) return;
    scoresGrad[index] = 0.0f;
}

__device__ void CudaOps::runRotaryRotateInPlace(float* tensor, int headCount, int headDimension, int pairCount, int sequenceLength, int positionOffset, const float* cosTable, const float* sinTable) {
    const int position = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (position >= sequenceLength) return;

    const int absolutePosition = positionOffset + position;
    for (int headIndex = 0; headIndex < headCount; ++headIndex) {
        const int rowOffset = headIndex * headDimension;
        for (int pairIndex = 0; pairIndex < pairCount; ++pairIndex) {
            const int rowEven = rowOffset + 2 * pairIndex;
            const int rowOdd = rowEven + 1;
            const int tableIndex = absolutePosition * pairCount + pairIndex;
            const float cosValue = cosTable[tableIndex];
            const float sinValue = sinTable[tableIndex];
            const float evenValue = tensor[rowEven * sequenceLength + position];
            const float oddValue = tensor[rowOdd * sequenceLength + position];
            tensor[rowEven * sequenceLength + position] = evenValue * cosValue - oddValue * sinValue;
            tensor[rowOdd * sequenceLength + position] = evenValue * sinValue + oddValue * cosValue;
        }
    }
}

__device__ void CudaOps::runRotaryRotateInverseInPlace(float* tensor, int headCount, int headDimension, int pairCount, int sequenceLength, int positionOffset, const float* cosTable, const float* sinTable) {
    const int position = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (position >= sequenceLength) return;

    const int absolutePosition = positionOffset + position;
    for (int headIndex = 0; headIndex < headCount; ++headIndex) {
        const int rowOffset = headIndex * headDimension;
        for (int pairIndex = 0; pairIndex < pairCount; ++pairIndex) {
            const int rowEven = rowOffset + 2 * pairIndex;
            const int rowOdd = rowEven + 1;
            const int tableIndex = absolutePosition * pairCount + pairIndex;
            const float cosValue = cosTable[tableIndex];
            const float sinValue = sinTable[tableIndex];
            const float evenValue = tensor[rowEven * sequenceLength + position];
            const float oddValue = tensor[rowOdd * sequenceLength + position];
            tensor[rowEven * sequenceLength + position] = evenValue * cosValue + oddValue * sinValue;
            tensor[rowOdd * sequenceLength + position] = -evenValue * sinValue + oddValue * cosValue;
        }
    }
}

__device__ void CudaOps::runEmbeddingGatherInto(const float* weight, const int* tokenIds, float* out, int embeddingDim, int tokenCount, int vocabularySize) {
    const int elementCount = embeddingDim * tokenCount;
    const int index = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (index >= elementCount) return;

    const int dimensionIndex = index / tokenCount;
    const int tokenIndex = index - dimensionIndex * tokenCount;
    const int tokenId = tokenIds[tokenIndex];
    if (tokenId < 0 || tokenId >= vocabularySize) {
        out[index] = 0.0f;
        return;
    }

    out[index] = weight[tokenId * embeddingDim + dimensionIndex];
}

__device__ void CudaOps::runEmbeddingScatterAddInto(float* weightGradient, const int* tokenIds, const float* outputGradient, int embeddingDim, int tokenCount, int vocabularySize) {
    const int elementCount = embeddingDim * tokenCount;
    const int index = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (index >= elementCount) return;

    const int dimensionIndex = index / tokenCount;
    const int tokenIndex = index - dimensionIndex * tokenCount;
    const int tokenId = tokenIds[tokenIndex];
    if (tokenId < 0 || tokenId >= vocabularySize) return;

    atomicAdd(&weightGradient[tokenId * embeddingDim + dimensionIndex], outputGradient[index]);
}

__device__ void CudaOps::runCrossEntropyLossFromIds(const float* probabilities, const int* targetTokenIds, float* columnLosses, int vocabularySize, int tokenCount) {
    const int column = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (column >= tokenCount) return;

    const int targetId = targetTokenIds[column];
    if (targetId < 0 || targetId >= vocabularySize) {
        columnLosses[column] = 0.0f;
        return;
    }

    float probability = probabilities[targetId * tokenCount + column];
    if (probability < 1e-7f) probability = 1e-7f;
    columnLosses[column] = -logf(probability);
}

__device__ void CudaOps::runCrossEntropyAddMeanLossFromIds(const float* probabilities, const int* targetTokenIds, float* lossSum, int vocabularySize, int tokenCount) {
    const int column = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (column >= tokenCount) return;

    const int targetId = targetTokenIds[column];
    if (targetId < 0 || targetId >= vocabularySize) return;

    float probability = probabilities[targetId * tokenCount + column];
    if (probability < 1e-7f) probability = 1e-7f;
    atomicAdd(lossSum, (-logf(probability)) / static_cast<float>(tokenCount));
}

__device__ void CudaOps::runCrossEntropyLogitGradientFromIds(const float* probabilities, const int* targetTokenIds, float* logitGradient, int vocabularySize, int tokenCount) {
    const int elementCount = vocabularySize * tokenCount;
    const int index = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (index >= elementCount) return;

    const int column = index % tokenCount;
    const int row = index / tokenCount;
    const float inverseTokenCount = 1.0f / static_cast<float>(tokenCount);
    float gradient = probabilities[index] * inverseTokenCount;
    if (row == targetTokenIds[column])
        gradient -= inverseTokenCount;
    logitGradient[index] = gradient;
}

__global__ void CudaOpsCrossEntropyLossFromIdsEntry(const float* probabilities, const int* targetTokenIds, float* columnLosses, int vocabularySize, int tokenCount) {
    CudaOps::runCrossEntropyLossFromIds(probabilities, targetTokenIds, columnLosses, vocabularySize, tokenCount);
}

__global__ void CudaOpsCrossEntropyAddMeanLossFromIdsEntry(const float* probabilities, const int* targetTokenIds, float* lossSum, int vocabularySize, int tokenCount) {
    CudaOps::runCrossEntropyAddMeanLossFromIds(probabilities, targetTokenIds, lossSum, vocabularySize, tokenCount);
}

__global__ void CudaOpsCrossEntropyLogitGradientFromIdsEntry(const float* probabilities, const int* targetTokenIds, float* logitGradient, int vocabularySize, int tokenCount) {
    CudaOps::runCrossEntropyLogitGradientFromIds(probabilities, targetTokenIds, logitGradient, vocabularySize, tokenCount);
}

__global__ void CudaOpsBroadcastBiasAddEntry(float* product, const float* bias, int rowCount, int columnCount) {
    CudaOps::runBroadcastBiasAddInPlace(product, bias, rowCount, columnCount);
}

__global__ void CudaOpsSiluEntry(const float* input, float* out, int elementCount) {
    CudaOps::runSiluInto(input, out, elementCount);
}

__global__ void CudaOpsSiluDerivativeEntry(const float* input, float* out, int elementCount) {
    CudaOps::runSiluDerivativeInto(input, out, elementCount);
}

__global__ void CudaOpsMultiplyElementwiseEntry(const float* left, const float* right, float* out, int elementCount) {
    CudaOps::runMultiplyElementwiseInto(left, right, out, elementCount);
}

__global__ void CudaOpsMultiplyElementwiseInPlaceEntry(float* total, const float* other, int elementCount) {
    CudaOps::runMultiplyElementwiseInPlace(total, other, elementCount);
}

__global__ void CudaOpsAddEntry(const float* left, const float* right, float* out, int elementCount) {
    CudaOps::runAddInto(left, right, out, elementCount);
}

__global__ void CudaOpsAddInPlaceEntry(float* total, const float* delta, int elementCount) {
    CudaOps::runAddInPlace(total, delta, elementCount);
}

__global__ void CudaOpsSumColumnsEntry(const float* gradient, float* biasGradient, int rowCount, int columnCount) {
    CudaOps::runSumColumnsInto(gradient, biasGradient, rowCount, columnCount);
}

__global__ void CudaOpsScaleEntry(float* matrix, float scalar, int elementCount) {
    CudaOps::runScaleInPlace(matrix, scalar, elementCount);
}

__global__ void CudaOpsZeroEntry(float* matrix, int elementCount) {
    CudaOps::runZeroInPlace(matrix, elementCount);
}

__global__ void CudaOpsExtractHeadEntry(const float* full, float* head, int headIndex, int headDimension, int sourceStrideColumns, int usedColumnCount) {
    CudaOps::runExtractHeadInto(full, head, headIndex, headDimension, sourceStrideColumns, usedColumnCount);
}

__global__ void CudaOpsWriteHeadEntry(float* full, const float* head, int headIndex, int headDimension, int sequenceLength, int embeddingDim) {
    CudaOps::runWriteHead(full, head, headIndex, headDimension, sequenceLength, embeddingDim);
}

__global__ void CudaOpsWriteColumnsEntry(float* destination, const float* source, int embeddingDim, int destinationStrideColumns, int destinationStartColumn, int sourceColumnCount) {
    CudaOps::runWriteColumnsInto(destination, source, embeddingDim, destinationStrideColumns, destinationStartColumn, sourceColumnCount);
}

__global__ void CudaOpsCausalMaskEntry(float* scores, int sequenceLength) {
    CudaOps::runApplyCausalMaskInPlace(scores, sequenceLength);
}

__global__ void CudaOpsSparseAttentionMaskEntry(float* scores, int keyCount, int queryCount, int queryPositionStart, int windowSize, int globalTokenCount) {
    CudaOps::runApplySparseAttentionMaskInPlace(scores, keyCount, queryCount, queryPositionStart, windowSize, globalTokenCount);
}

__global__ void CudaOpsSoftmaxEntry(const float* logits, float* out, int rowCount, int columnCount) {
    CudaOps::runSoftmaxInto(logits, out, rowCount, columnCount);
}

__global__ void CudaOpsSoftmaxBackwardEntry(const float* probabilities, const float* probabilityGradient, float* scoreGradient, int rowCount, int columnCount) {
    CudaOps::runSoftmaxBackwardInto(probabilities, probabilityGradient, scoreGradient, rowCount, columnCount);
}

__global__ void CudaOpsZeroForbiddenScoreGradientsEntry(float* scoresGrad, int keyCount, int queryCount, int queryPositionStart, int windowSize, int globalTokenCount) {
    CudaOps::runZeroForbiddenScoreGradientsInPlace(scoresGrad, keyCount, queryCount, queryPositionStart, windowSize, globalTokenCount);
}

__global__ void CudaOpsRotaryRotateEntry(float* tensor, int headCount, int headDimension, int pairCount, int sequenceLength, int positionOffset, const float* cosTable, const float* sinTable) {
    CudaOps::runRotaryRotateInPlace(tensor, headCount, headDimension, pairCount, sequenceLength, positionOffset, cosTable, sinTable);
}

__global__ void CudaOpsRotaryRotateInverseEntry(float* tensor, int headCount, int headDimension, int pairCount, int sequenceLength, int positionOffset, const float* cosTable, const float* sinTable) {
    CudaOps::runRotaryRotateInverseInPlace(tensor, headCount, headDimension, pairCount, sequenceLength, positionOffset, cosTable, sinTable);
}

__global__ void CudaOpsEmbeddingGatherEntry(const float* weight, const int* tokenIds, float* out, int embeddingDim, int tokenCount, int vocabularySize) {
    CudaOps::runEmbeddingGatherInto(weight, tokenIds, out, embeddingDim, tokenCount, vocabularySize);
}

__global__ void CudaOpsEmbeddingScatterAddEntry(float* weightGradient, const int* tokenIds, const float* outputGradient, int embeddingDim, int tokenCount, int vocabularySize) {
    CudaOps::runEmbeddingScatterAddInto(weightGradient, tokenIds, outputGradient, embeddingDim, tokenCount, vocabularySize);
}

void CudaOps::broadcastBiasAddInPlace(CudaMatrix& product, const CudaMatrix& bias) {
    if (product.empty()) throw std::invalid_argument("CudaOps::broadcastBiasAddInPlace empty product");
    if (bias.empty()) throw std::invalid_argument("CudaOps::broadcastBiasAddInPlace empty bias");
    if (bias.cols != 1) throw std::invalid_argument("CudaOps::broadcastBiasAddInPlace bias must be a column");
    if (bias.rows != product.rows) throw std::invalid_argument("CudaOps::broadcastBiasAddInPlace row mismatch");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaOps::broadcastBiasAddInPlace no CUDA device");

    const int rowCount = static_cast<int>(product.rows);
    const int columnCount = static_cast<int>(product.cols);
    const int elementCount = rowCount * columnCount;
    const int blockCount = (elementCount + CudaOps::threadCount - 1) / CudaOps::threadCount;
    CudaOpsBroadcastBiasAddEntry<<<blockCount, CudaOps::threadCount>>>(product.buffer.deviceData, bias.buffer.deviceData, rowCount, columnCount);
    CudaMatmul::throwIfCudaFailed(cudaGetLastError(), "CudaOpsBroadcastBiasAddEntry launch");
}

void CudaOps::siluInto(const CudaMatrix& input, CudaMatrix& out) {
    if (input.empty()) throw std::invalid_argument("CudaOps::siluInto empty input");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaOps::siluInto no CUDA device");

    out.ensureSize(input.rows, input.cols);
    const int elementCount = static_cast<int>(input.elementCount());
    const int blockCount = (elementCount + CudaOps::threadCount - 1) / CudaOps::threadCount;
    CudaOpsSiluEntry<<<blockCount, CudaOps::threadCount>>>(input.buffer.deviceData, out.buffer.deviceData, elementCount);
    CudaMatmul::throwIfCudaFailed(cudaGetLastError(), "CudaOpsSiluEntry launch");
}

void CudaOps::siluDerivativeInto(const CudaMatrix& input, CudaMatrix& out) {
    if (input.empty()) throw std::invalid_argument("CudaOps::siluDerivativeInto empty input");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaOps::siluDerivativeInto no CUDA device");

    out.ensureSize(input.rows, input.cols);
    const int elementCount = static_cast<int>(input.elementCount());
    const int blockCount = (elementCount + CudaOps::threadCount - 1) / CudaOps::threadCount;
    CudaOpsSiluDerivativeEntry<<<blockCount, CudaOps::threadCount>>>(input.buffer.deviceData, out.buffer.deviceData, elementCount);
    CudaMatmul::throwIfCudaFailed(cudaGetLastError(), "CudaOpsSiluDerivativeEntry launch");
}

void CudaOps::multiplyElementwiseInto(const CudaMatrix& left, const CudaMatrix& right, CudaMatrix& out) {
    if (left.empty() || right.empty()) throw std::invalid_argument("CudaOps::multiplyElementwiseInto empty input");
    if (left.rows != right.rows || left.cols != right.cols) throw std::invalid_argument("CudaOps::multiplyElementwiseInto shape mismatch");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaOps::multiplyElementwiseInto no CUDA device");

    out.ensureSize(left.rows, left.cols);
    const int elementCount = static_cast<int>(left.elementCount());
    const int blockCount = (elementCount + CudaOps::threadCount - 1) / CudaOps::threadCount;
    CudaOpsMultiplyElementwiseEntry<<<blockCount, CudaOps::threadCount>>>(left.buffer.deviceData, right.buffer.deviceData, out.buffer.deviceData, elementCount);
    CudaMatmul::throwIfCudaFailed(cudaGetLastError(), "CudaOpsMultiplyElementwiseEntry launch");
}

void CudaOps::multiplyElementwiseInPlace(CudaMatrix& total, const CudaMatrix& other) {
    if (total.empty() || other.empty()) throw std::invalid_argument("CudaOps::multiplyElementwiseInPlace empty input");
    if (total.rows != other.rows || total.cols != other.cols) throw std::invalid_argument("CudaOps::multiplyElementwiseInPlace shape mismatch");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaOps::multiplyElementwiseInPlace no CUDA device");

    const int elementCount = static_cast<int>(total.elementCount());
    const int blockCount = (elementCount + CudaOps::threadCount - 1) / CudaOps::threadCount;
    CudaOpsMultiplyElementwiseInPlaceEntry<<<blockCount, CudaOps::threadCount>>>(total.buffer.deviceData, other.buffer.deviceData, elementCount);
    CudaMatmul::throwIfCudaFailed(cudaGetLastError(), "CudaOpsMultiplyElementwiseInPlaceEntry launch");
}

void CudaOps::addInto(const CudaMatrix& left, const CudaMatrix& right, CudaMatrix& out) {
    if (left.empty() || right.empty()) throw std::invalid_argument("CudaOps::addInto empty input");
    if (left.rows != right.rows || left.cols != right.cols) throw std::invalid_argument("CudaOps::addInto shape mismatch");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaOps::addInto no CUDA device");

    out.ensureSize(left.rows, left.cols);
    const int elementCount = static_cast<int>(left.elementCount());
    const int blockCount = (elementCount + CudaOps::threadCount - 1) / CudaOps::threadCount;
    CudaOpsAddEntry<<<blockCount, CudaOps::threadCount>>>(left.buffer.deviceData, right.buffer.deviceData, out.buffer.deviceData, elementCount);
    CudaMatmul::throwIfCudaFailed(cudaGetLastError(), "CudaOpsAddEntry launch");
}

void CudaOps::addInPlace(CudaMatrix& total, const CudaMatrix& delta) {
    if (total.empty() || delta.empty()) throw std::invalid_argument("CudaOps::addInPlace empty input");
    if (total.rows != delta.rows || total.cols != delta.cols) throw std::invalid_argument("CudaOps::addInPlace shape mismatch");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaOps::addInPlace no CUDA device");

    const int elementCount = static_cast<int>(total.elementCount());
    const int blockCount = (elementCount + CudaOps::threadCount - 1) / CudaOps::threadCount;
    CudaOpsAddInPlaceEntry<<<blockCount, CudaOps::threadCount>>>(total.buffer.deviceData, delta.buffer.deviceData, elementCount);
    CudaMatmul::throwIfCudaFailed(cudaGetLastError(), "CudaOpsAddInPlaceEntry launch");
}

void CudaOps::sumColumnsInto(const CudaMatrix& gradient, CudaMatrix& biasGradient) {
    if (gradient.empty()) throw std::invalid_argument("CudaOps::sumColumnsInto empty gradient");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaOps::sumColumnsInto no CUDA device");

    biasGradient.ensureSize(gradient.rows, 1);
    const int rowCount = static_cast<int>(gradient.rows);
    const int columnCount = static_cast<int>(gradient.cols);
    const int blockCount = (rowCount + CudaOps::threadCount - 1) / CudaOps::threadCount;
    CudaOpsSumColumnsEntry<<<blockCount, CudaOps::threadCount>>>(gradient.buffer.deviceData, biasGradient.buffer.deviceData, rowCount, columnCount);
    CudaMatmul::throwIfCudaFailed(cudaGetLastError(), "CudaOpsSumColumnsEntry launch");
}

void CudaOps::copyInto(const CudaMatrix& source, CudaMatrix& destination) {
    if (source.empty()) throw std::invalid_argument("CudaOps::copyInto empty source");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaOps::copyInto no CUDA device");

    destination.ensureSize(source.rows, source.cols);
    CudaMatmul::throwIfCudaFailed(cudaMemcpy(destination.buffer.deviceData, source.buffer.deviceData, source.byteCount(), cudaMemcpyDeviceToDevice), "CudaOps::copyInto memcpy");
}

void CudaOps::scaleInPlace(CudaMatrix& matrix, float scalar) {
    if (matrix.empty()) throw std::invalid_argument("CudaOps::scaleInPlace empty matrix");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaOps::scaleInPlace no CUDA device");

    const int elementCount = static_cast<int>(matrix.elementCount());
    const int blockCount = (elementCount + CudaOps::threadCount - 1) / CudaOps::threadCount;
    CudaOpsScaleEntry<<<blockCount, CudaOps::threadCount>>>(matrix.buffer.deviceData, scalar, elementCount);
    CudaMatmul::throwIfCudaFailed(cudaGetLastError(), "CudaOpsScaleEntry launch");
}

void CudaOps::zeroInPlace(CudaMatrix& matrix) {
    if (matrix.empty()) return;
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaOps::zeroInPlace no CUDA device");

    const int elementCount = static_cast<int>(matrix.elementCount());
    const int blockCount = (elementCount + CudaOps::threadCount - 1) / CudaOps::threadCount;
    CudaOpsZeroEntry<<<blockCount, CudaOps::threadCount>>>(matrix.buffer.deviceData, elementCount);
    CudaMatmul::throwIfCudaFailed(cudaGetLastError(), "CudaOpsZeroEntry launch");
}

void CudaOps::extractHeadInto(const CudaMatrix& full, int headIndex, int headDimension, CudaMatrix& head) {
    CudaOps::extractHeadInto(full, headIndex, headDimension, static_cast<int>(full.cols), head);
}

void CudaOps::extractHeadInto(const CudaMatrix& full, int headIndex, int headDimension, int usedColumnCount, CudaMatrix& head) {
    if (full.empty()) throw std::invalid_argument("CudaOps::extractHeadInto empty full");
    if (headIndex < 0 || headDimension <= 0) throw std::invalid_argument("CudaOps::extractHeadInto invalid head");
    if (usedColumnCount <= 0 || usedColumnCount > static_cast<int>(full.cols)) throw std::invalid_argument("CudaOps::extractHeadInto invalid usedColumnCount");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaOps::extractHeadInto no CUDA device");

    head.ensureSize(static_cast<size_t>(headDimension), static_cast<size_t>(usedColumnCount));
    const int sourceStrideColumns = static_cast<int>(full.cols);
    const int elementCount = headDimension * usedColumnCount;
    const int blockCount = (elementCount + CudaOps::threadCount - 1) / CudaOps::threadCount;
    CudaOpsExtractHeadEntry<<<blockCount, CudaOps::threadCount>>>(full.buffer.deviceData, head.buffer.deviceData, headIndex, headDimension, sourceStrideColumns, usedColumnCount);
    CudaMatmul::throwIfCudaFailed(cudaGetLastError(), "CudaOpsExtractHeadEntry launch");
}

void CudaOps::writeHead(CudaMatrix& full, int headIndex, int headDimension, const CudaMatrix& head) {
    if (full.empty() || head.empty()) throw std::invalid_argument("CudaOps::writeHead empty input");
    if (head.rows != static_cast<size_t>(headDimension) || head.cols != full.cols) throw std::invalid_argument("CudaOps::writeHead shape mismatch");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaOps::writeHead no CUDA device");

    const int sequenceLength = static_cast<int>(full.cols);
    const int elementCount = headDimension * sequenceLength;
    const int blockCount = (elementCount + CudaOps::threadCount - 1) / CudaOps::threadCount;
    CudaOpsWriteHeadEntry<<<blockCount, CudaOps::threadCount>>>(full.buffer.deviceData, head.buffer.deviceData, headIndex, headDimension, sequenceLength, static_cast<int>(full.rows));
    CudaMatmul::throwIfCudaFailed(cudaGetLastError(), "CudaOpsWriteHeadEntry launch");
}

void CudaOps::writeColumnsInto(CudaMatrix& destination, int destinationStartColumn, const CudaMatrix& source) {
    if (destination.empty() || source.empty()) throw std::invalid_argument("CudaOps::writeColumnsInto empty input");
    if (destination.rows != source.rows) throw std::invalid_argument("CudaOps::writeColumnsInto row mismatch");
    if (destinationStartColumn < 0) throw std::invalid_argument("CudaOps::writeColumnsInto negative start");
    if (destinationStartColumn + static_cast<int>(source.cols) > static_cast<int>(destination.cols))
        throw std::invalid_argument("CudaOps::writeColumnsInto exceeds destination width");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaOps::writeColumnsInto no CUDA device");

    const int embeddingDim = static_cast<int>(source.rows);
    const int destinationStrideColumns = static_cast<int>(destination.cols);
    const int sourceColumnCount = static_cast<int>(source.cols);
    const int elementCount = embeddingDim * sourceColumnCount;
    const int blockCount = (elementCount + CudaOps::threadCount - 1) / CudaOps::threadCount;
    CudaOpsWriteColumnsEntry<<<blockCount, CudaOps::threadCount>>>(destination.buffer.deviceData, source.buffer.deviceData, embeddingDim, destinationStrideColumns, destinationStartColumn, sourceColumnCount);
    CudaMatmul::throwIfCudaFailed(cudaGetLastError(), "CudaOpsWriteColumnsEntry launch");
}

void CudaOps::applyCausalMaskInPlace(CudaMatrix& scores) {
    if (scores.empty()) throw std::invalid_argument("CudaOps::applyCausalMaskInPlace empty scores");
    if (scores.rows != scores.cols) throw std::invalid_argument("CudaOps::applyCausalMaskInPlace scores must be square");
    CudaOps::applySparseAttentionMaskInPlace(scores, static_cast<int>(scores.rows), 0, 0);
}

void CudaOps::applySparseAttentionMaskInPlace(CudaMatrix& scores, int windowSize, int globalTokenCount, int queryPositionStart) {
    if (scores.empty()) throw std::invalid_argument("CudaOps::applySparseAttentionMaskInPlace empty scores");
    if (globalTokenCount < 0) throw std::invalid_argument("CudaOps::applySparseAttentionMaskInPlace globalTokenCount must be >= 0");
    if (queryPositionStart < 0) throw std::invalid_argument("CudaOps::applySparseAttentionMaskInPlace queryPositionStart must be >= 0");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaOps::applySparseAttentionMaskInPlace no CUDA device");

    const int keyCount = static_cast<int>(scores.rows);
    const int queryCount = static_cast<int>(scores.cols);
    const int elementCount = keyCount * queryCount;
    const int blockCount = (elementCount + CudaOps::threadCount - 1) / CudaOps::threadCount;
    CudaOpsSparseAttentionMaskEntry<<<blockCount, CudaOps::threadCount>>>(scores.buffer.deviceData, keyCount, queryCount, queryPositionStart, windowSize, globalTokenCount);
    CudaMatmul::throwIfCudaFailed(cudaGetLastError(), "CudaOpsSparseAttentionMaskEntry launch");
}

void CudaOps::softmaxInto(const CudaMatrix& logits, CudaMatrix& out) {
    if (logits.empty()) throw std::invalid_argument("CudaOps::softmaxInto empty logits");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaOps::softmaxInto no CUDA device");

    out.ensureSize(logits.rows, logits.cols);
    const int rowCount = static_cast<int>(logits.rows);
    const int columnCount = static_cast<int>(logits.cols);
    const int blockCount = (columnCount + CudaOps::threadCount - 1) / CudaOps::threadCount;
    CudaOpsSoftmaxEntry<<<blockCount, CudaOps::threadCount>>>(logits.buffer.deviceData, out.buffer.deviceData, rowCount, columnCount);
    CudaMatmul::throwIfCudaFailed(cudaGetLastError(), "CudaOpsSoftmaxEntry launch");
}

void CudaOps::softmaxBackwardInto(const CudaMatrix& probabilities, const CudaMatrix& probabilityGradient, CudaMatrix& scoreGradient) {
    if (probabilities.empty() || probabilityGradient.empty()) throw std::invalid_argument("CudaOps::softmaxBackwardInto empty input");
    if (probabilities.rows != probabilityGradient.rows || probabilities.cols != probabilityGradient.cols)
        throw std::invalid_argument("CudaOps::softmaxBackwardInto shape mismatch");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaOps::softmaxBackwardInto no CUDA device");

    scoreGradient.ensureSize(probabilities.rows, probabilities.cols);
    const int rowCount = static_cast<int>(probabilities.rows);
    const int columnCount = static_cast<int>(probabilities.cols);
    const int blockCount = (columnCount + CudaOps::threadCount - 1) / CudaOps::threadCount;
    CudaOpsSoftmaxBackwardEntry<<<blockCount, CudaOps::threadCount>>>(probabilities.buffer.deviceData, probabilityGradient.buffer.deviceData, scoreGradient.buffer.deviceData, rowCount, columnCount);
    CudaMatmul::throwIfCudaFailed(cudaGetLastError(), "CudaOpsSoftmaxBackwardEntry launch");
}

void CudaOps::zeroForbiddenScoreGradientsInPlace(CudaMatrix& scoresGrad, int windowSize, int globalTokenCount, int queryPositionStart) {
    if (scoresGrad.empty()) throw std::invalid_argument("CudaOps::zeroForbiddenScoreGradientsInPlace empty scoresGrad");
    if (globalTokenCount < 0) throw std::invalid_argument("CudaOps::zeroForbiddenScoreGradientsInPlace globalTokenCount must be >= 0");
    if (queryPositionStart < 0) throw std::invalid_argument("CudaOps::zeroForbiddenScoreGradientsInPlace queryPositionStart must be >= 0");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaOps::zeroForbiddenScoreGradientsInPlace no CUDA device");

    const int keyCount = static_cast<int>(scoresGrad.rows);
    const int queryCount = static_cast<int>(scoresGrad.cols);
    const int elementCount = keyCount * queryCount;
    const int blockCount = (elementCount + CudaOps::threadCount - 1) / CudaOps::threadCount;
    CudaOpsZeroForbiddenScoreGradientsEntry<<<blockCount, CudaOps::threadCount>>>(scoresGrad.buffer.deviceData, keyCount, queryCount, queryPositionStart, windowSize, globalTokenCount);
    CudaMatmul::throwIfCudaFailed(cudaGetLastError(), "CudaOpsZeroForbiddenScoreGradientsEntry launch");
}

void CudaOps::rotaryRotateInPlace(CudaMatrix& tensor, int headCount, int headDimension, int pairCount, const CudaMatrix& cosTable, const CudaMatrix& sinTable, int positionOffset) {
    if (tensor.empty()) throw std::invalid_argument("CudaOps::rotaryRotateInPlace empty tensor");
    if (headCount <= 0 || headDimension <= 0 || pairCount <= 0) throw std::invalid_argument("CudaOps::rotaryRotateInPlace invalid rope dims");
    if (static_cast<int>(tensor.rows) != headCount * headDimension) throw std::invalid_argument("CudaOps::rotaryRotateInPlace embedding dim mismatch");
    if (cosTable.empty() || sinTable.empty()) throw std::invalid_argument("CudaOps::rotaryRotateInPlace empty tables");
    if (positionOffset < 0) throw std::invalid_argument("CudaOps::rotaryRotateInPlace negative positionOffset");
    if (positionOffset + static_cast<int>(tensor.cols) > static_cast<int>(cosTable.rows))
        throw std::invalid_argument("CudaOps::rotaryRotateInPlace position exceeds RoPE table");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaOps::rotaryRotateInPlace no CUDA device");

    const int sequenceLength = static_cast<int>(tensor.cols);
    const int blockCount = (sequenceLength + CudaOps::threadCount - 1) / CudaOps::threadCount;
    CudaOpsRotaryRotateEntry<<<blockCount, CudaOps::threadCount>>>(tensor.buffer.deviceData, headCount, headDimension, pairCount, sequenceLength, positionOffset, cosTable.buffer.deviceData, sinTable.buffer.deviceData);
    CudaMatmul::throwIfCudaFailed(cudaGetLastError(), "CudaOpsRotaryRotateEntry launch");
}

void CudaOps::rotaryRotateInverseInPlace(CudaMatrix& tensor, int headCount, int headDimension, int pairCount, const CudaMatrix& cosTable, const CudaMatrix& sinTable, int positionOffset) {
    if (tensor.empty()) throw std::invalid_argument("CudaOps::rotaryRotateInverseInPlace empty tensor");
    if (headCount <= 0 || headDimension <= 0 || pairCount <= 0) throw std::invalid_argument("CudaOps::rotaryRotateInverseInPlace invalid rope dims");
    if (static_cast<int>(tensor.rows) != headCount * headDimension) throw std::invalid_argument("CudaOps::rotaryRotateInverseInPlace embedding dim mismatch");
    if (cosTable.empty() || sinTable.empty()) throw std::invalid_argument("CudaOps::rotaryRotateInverseInPlace empty tables");
    if (positionOffset < 0) throw std::invalid_argument("CudaOps::rotaryRotateInverseInPlace negative positionOffset");
    if (positionOffset + static_cast<int>(tensor.cols) > static_cast<int>(cosTable.rows))
        throw std::invalid_argument("CudaOps::rotaryRotateInverseInPlace position exceeds RoPE table");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaOps::rotaryRotateInverseInPlace no CUDA device");

    const int sequenceLength = static_cast<int>(tensor.cols);
    const int blockCount = (sequenceLength + CudaOps::threadCount - 1) / CudaOps::threadCount;
    CudaOpsRotaryRotateInverseEntry<<<blockCount, CudaOps::threadCount>>>(tensor.buffer.deviceData, headCount, headDimension, pairCount, sequenceLength, positionOffset, cosTable.buffer.deviceData, sinTable.buffer.deviceData);
    CudaMatmul::throwIfCudaFailed(cudaGetLastError(), "CudaOpsRotaryRotateInverseEntry launch");
}

void CudaOps::embeddingGatherInto(const CudaMatrix& weight, const CudaIntBuffer& tokenIds, size_t tokenCount, CudaMatrix& out) {
    if (weight.empty()) throw std::invalid_argument("CudaOps::embeddingGatherInto empty weight");
    if (tokenCount == 0) throw std::invalid_argument("CudaOps::embeddingGatherInto empty tokenCount");
    if (tokenIds.deviceData == nullptr) throw std::invalid_argument("CudaOps::embeddingGatherInto empty tokenIds");
    if (tokenCount > tokenIds.capacityCount) throw std::invalid_argument("CudaOps::embeddingGatherInto tokenCount exceeds capacity");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaOps::embeddingGatherInto no CUDA device");

    const int vocabularySize = static_cast<int>(weight.rows);
    const int embeddingDim = static_cast<int>(weight.cols);
    out.ensureSize(static_cast<size_t>(embeddingDim), tokenCount);

    const int elementCount = embeddingDim * static_cast<int>(tokenCount);
    const int blockCount = (elementCount + CudaOps::threadCount - 1) / CudaOps::threadCount;
    CudaOpsEmbeddingGatherEntry<<<blockCount, CudaOps::threadCount>>>(weight.buffer.deviceData, tokenIds.deviceData, out.buffer.deviceData, embeddingDim, static_cast<int>(tokenCount), vocabularySize);
    CudaMatmul::throwIfCudaFailed(cudaGetLastError(), "CudaOpsEmbeddingGatherEntry launch");
}

void CudaOps::embeddingScatterAddInto(CudaMatrix& weightGradient, const CudaIntBuffer& tokenIds, size_t tokenCount, const CudaMatrix& outputGradient) {
    if (weightGradient.empty()) throw std::invalid_argument("CudaOps::embeddingScatterAddInto empty weightGradient");
    if (outputGradient.empty()) throw std::invalid_argument("CudaOps::embeddingScatterAddInto empty outputGradient");
    if (tokenCount == 0) throw std::invalid_argument("CudaOps::embeddingScatterAddInto empty tokenCount");
    if (tokenIds.deviceData == nullptr) throw std::invalid_argument("CudaOps::embeddingScatterAddInto empty tokenIds");
    if (tokenCount > tokenIds.capacityCount) throw std::invalid_argument("CudaOps::embeddingScatterAddInto tokenCount exceeds capacity");
    if (outputGradient.cols != tokenCount) throw std::invalid_argument("CudaOps::embeddingScatterAddInto tokenCount mismatch");
    if (static_cast<size_t>(outputGradient.rows) != weightGradient.cols) throw std::invalid_argument("CudaOps::embeddingScatterAddInto embedding dim mismatch");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaOps::embeddingScatterAddInto no CUDA device");

    const int vocabularySize = static_cast<int>(weightGradient.rows);
    const int embeddingDim = static_cast<int>(weightGradient.cols);
    const int elementCount = embeddingDim * static_cast<int>(tokenCount);
    const int blockCount = (elementCount + CudaOps::threadCount - 1) / CudaOps::threadCount;
    CudaOpsEmbeddingScatterAddEntry<<<blockCount, CudaOps::threadCount>>>(weightGradient.buffer.deviceData, tokenIds.deviceData, outputGradient.buffer.deviceData, embeddingDim, static_cast<int>(tokenCount), vocabularySize);
    CudaMatmul::throwIfCudaFailed(cudaGetLastError(), "CudaOpsEmbeddingScatterAddEntry launch");
}

float CudaOps::crossEntropyLossFromIds(const CudaMatrix& probabilities, const CudaIntBuffer& targetTokenIds, size_t tokenCount) {
    if (probabilities.empty()) throw std::invalid_argument("CudaOps::crossEntropyLossFromIds empty probabilities");
    if (tokenCount == 0) throw std::invalid_argument("CudaOps::crossEntropyLossFromIds empty tokenCount");
    if (probabilities.cols != tokenCount) throw std::invalid_argument("CudaOps::crossEntropyLossFromIds tokenCount mismatch");
    if (targetTokenIds.deviceData == nullptr) throw std::invalid_argument("CudaOps::crossEntropyLossFromIds empty targetTokenIds");
    if (tokenCount > targetTokenIds.capacityCount) throw std::invalid_argument("CudaOps::crossEntropyLossFromIds tokenCount exceeds capacity");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaOps::crossEntropyLossFromIds no CUDA device");

    static CudaMatrix columnLossScratch;
    columnLossScratch.ensureSize(1, tokenCount);

    const int vocabularySize = static_cast<int>(probabilities.rows);
    const int tokenCountInt = static_cast<int>(tokenCount);
    const int blockCount = (tokenCountInt + CudaOps::threadCount - 1) / CudaOps::threadCount;
    CudaOpsCrossEntropyLossFromIdsEntry<<<blockCount, CudaOps::threadCount>>>(probabilities.buffer.deviceData, targetTokenIds.deviceData, columnLossScratch.buffer.deviceData, vocabularySize, tokenCountInt);
    CudaMatmul::throwIfCudaFailed(cudaGetLastError(), "CudaOpsCrossEntropyLossFromIdsEntry launch");
    CudaMatmul::throwIfCudaFailed(cudaDeviceSynchronize(), "CudaOps::crossEntropyLossFromIds synchronize");

    Matrix columnLossHost = columnLossScratch.download();
    float total = 0.0f;
    for (size_t column = 0; column < tokenCount; ++column)
        total += columnLossHost.at(0, column);
    return total / static_cast<float>(tokenCount);
}

void CudaOps::crossEntropyAddMeanLossFromIds(const CudaMatrix& probabilities, const CudaIntBuffer& targetTokenIds, size_t tokenCount, CudaMatrix& lossSum) {
    if (probabilities.empty()) throw std::invalid_argument("CudaOps::crossEntropyAddMeanLossFromIds empty probabilities");
    if (tokenCount == 0) throw std::invalid_argument("CudaOps::crossEntropyAddMeanLossFromIds empty tokenCount");
    if (probabilities.cols != tokenCount) throw std::invalid_argument("CudaOps::crossEntropyAddMeanLossFromIds tokenCount mismatch");
    if (targetTokenIds.deviceData == nullptr) throw std::invalid_argument("CudaOps::crossEntropyAddMeanLossFromIds empty targetTokenIds");
    if (tokenCount > targetTokenIds.capacityCount) throw std::invalid_argument("CudaOps::crossEntropyAddMeanLossFromIds tokenCount exceeds capacity");
    if (lossSum.rows != 1 || lossSum.cols != 1) throw std::invalid_argument("CudaOps::crossEntropyAddMeanLossFromIds lossSum must be 1x1");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaOps::crossEntropyAddMeanLossFromIds no CUDA device");

    const int vocabularySize = static_cast<int>(probabilities.rows);
    const int tokenCountInt = static_cast<int>(tokenCount);
    const int blockCount = (tokenCountInt + CudaOps::threadCount - 1) / CudaOps::threadCount;
    CudaOpsCrossEntropyAddMeanLossFromIdsEntry<<<blockCount, CudaOps::threadCount>>>(probabilities.buffer.deviceData, targetTokenIds.deviceData, lossSum.buffer.deviceData, vocabularySize, tokenCountInt);
    CudaMatmul::throwIfCudaFailed(cudaGetLastError(), "CudaOpsCrossEntropyAddMeanLossFromIdsEntry launch");
}

void CudaOps::crossEntropyLogitGradientFromIdsInto(const CudaMatrix& probabilities, const CudaIntBuffer& targetTokenIds, size_t tokenCount, CudaMatrix& logitGradient) {
    if (probabilities.empty()) throw std::invalid_argument("CudaOps::crossEntropyLogitGradientFromIdsInto empty probabilities");
    if (tokenCount == 0) throw std::invalid_argument("CudaOps::crossEntropyLogitGradientFromIdsInto empty tokenCount");
    if (probabilities.cols != tokenCount) throw std::invalid_argument("CudaOps::crossEntropyLogitGradientFromIdsInto tokenCount mismatch");
    if (targetTokenIds.deviceData == nullptr) throw std::invalid_argument("CudaOps::crossEntropyLogitGradientFromIdsInto empty targetTokenIds");
    if (tokenCount > targetTokenIds.capacityCount) throw std::invalid_argument("CudaOps::crossEntropyLogitGradientFromIdsInto tokenCount exceeds capacity");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaOps::crossEntropyLogitGradientFromIdsInto no CUDA device");

    logitGradient.ensureSize(probabilities.rows, probabilities.cols);
    const int vocabularySize = static_cast<int>(probabilities.rows);
    const int tokenCountInt = static_cast<int>(tokenCount);
    const int elementCount = vocabularySize * tokenCountInt;
    const int blockCount = (elementCount + CudaOps::threadCount - 1) / CudaOps::threadCount;
    CudaOpsCrossEntropyLogitGradientFromIdsEntry<<<blockCount, CudaOps::threadCount>>>(probabilities.buffer.deviceData, targetTokenIds.deviceData, logitGradient.buffer.deviceData, vocabularySize, tokenCountInt);
    CudaMatmul::throwIfCudaFailed(cudaGetLastError(), "CudaOpsCrossEntropyLogitGradientFromIdsEntry launch");
}
