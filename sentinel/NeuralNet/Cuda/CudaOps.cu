#include "CudaOps.hpp"

#include "CudaAmp.hpp"
#include "CudaAdam.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

double* CudaOps::downloadAddIntoHostSecondsSink = nullptr;

namespace {
/// <summary>double-buffered pinned staging for pageable host D2H (full PCIe rate + CPU memcpy overlap)</summary>
struct PinnedFloatDownloadStaging {
    void* buffers[2] = { nullptr, nullptr };
    size_t capacityBytes[2] = { 0, 0 };
    cudaEvent_t readyEvents[2] = { nullptr, nullptr };
    int turn = 0;

    ~PinnedFloatDownloadStaging() {
        for (int slot = 0; slot < 2; ++slot) {
            if (this->readyEvents[slot] != nullptr) {
                cudaEventDestroy(this->readyEvents[slot]);
                this->readyEvents[slot] = nullptr;
            }
            if (this->buffers[slot] != nullptr) {
                cudaFreeHost(this->buffers[slot]);
                this->buffers[slot] = nullptr;
            }
            this->capacityBytes[slot] = 0;
        }
    }

    void ensureEvents() {
        for (int slot = 0; slot < 2; ++slot) {
            if (this->readyEvents[slot] != nullptr) continue;
            if (cudaEventCreateWithFlags(&this->readyEvents[slot], cudaEventDisableTiming) != cudaSuccess)
                throw std::runtime_error("PinnedFloatDownloadStaging event create failed");
        }
    }

    void* acquire(size_t bytes, int* outSlot) {
        this->ensureEvents();
        const int slot = this->turn;
        this->turn = 1 - this->turn;
        if (this->readyEvents[slot] != nullptr)
            cudaEventSynchronize(this->readyEvents[slot]);
        if (bytes > this->capacityBytes[slot]) {
            if (this->buffers[slot] != nullptr) {
                cudaFreeHost(this->buffers[slot]);
                this->buffers[slot] = nullptr;
            }
            this->capacityBytes[slot] = 0;
            if (cudaMallocHost(&this->buffers[slot], bytes) != cudaSuccess)
                throw std::runtime_error("PinnedFloatDownloadStaging cudaMallocHost failed");
            this->capacityBytes[slot] = bytes;
        }
        if (outSlot != nullptr)
            *outSlot = slot;
        return this->buffers[slot];
    }

    void record(cudaStream_t stream, int slot) {
        this->ensureEvents();
        if (cudaEventRecord(this->readyEvents[slot], stream) != cudaSuccess)
            throw std::runtime_error("PinnedFloatDownloadStaging event record failed");
    }

    void syncSlot(int slot) {
        if (slot < 0 || slot > 1) return;
        if (this->readyEvents[slot] != nullptr)
            cudaEventSynchronize(this->readyEvents[slot]);
    }
};

thread_local PinnedFloatDownloadStaging gPinnedFloatDownloadStaging;
} // namespace

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

__device__ void CudaOps::runAddScaledInPlace(float* total, const float* delta, float scalar, int elementCount) {
    const int index = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (index >= elementCount) return;

    total[index] += scalar * delta[index];
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

__device__ void CudaOps::runExtractColumnsInto(const float* source, float* out, int embeddingDim, int sourceStrideColumns, int sourceStartColumn, int columnCount) {
    const int elementCount = embeddingDim * columnCount;
    const int index = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (index >= elementCount) return;

    const int row = index / columnCount;
    const int outColumn = index - row * columnCount;
    const int sourceColumn = sourceStartColumn + outColumn;
    out[index] = source[row * sourceStrideColumns + sourceColumn];
}

__device__ void CudaOps::runAddColumnsInPlace(float* destination, const float* source, int embeddingDim, int destinationStrideColumns, int destinationStartColumn, int sourceColumnCount) {
    const int elementCount = embeddingDim * sourceColumnCount;
    const int index = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (index >= elementCount) return;

    const int row = index / sourceColumnCount;
    const int sourceColumn = index - row * sourceColumnCount;
    const int destinationColumn = destinationStartColumn + sourceColumn;
    destination[row * destinationStrideColumns + destinationColumn] += source[index];
}

__device__ void CudaOps::runApplyCausalMaskInPlace(float* scores, int sequenceLength) {
    CudaOps::runApplySparseAttentionMaskInPlace(scores, sequenceLength, sequenceLength, 0, sequenceLength, 0, 0);
}

__device__ void CudaOps::runApplySparseAttentionMaskInPlace(float* scores, int keyCount, int queryCount, int queryPositionStart, int windowSize, int globalTokenCount, int segmentLength) {
    const int elementCount = keyCount * queryCount;
    const int index = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (index >= elementCount) return;

    const int keyIndex = index / queryCount;
    const int queryIndex = index - keyIndex * queryCount;

    bool allowed = false;
    if (segmentLength > 0) {
        const int queryPos = queryIndex % segmentLength;
        const int queryBatch = queryIndex / segmentLength;
        const int keyPos = keyIndex % segmentLength;
        const int keyBatch = keyIndex / segmentLength;
        if (queryBatch == keyBatch && keyPos <= queryPos) {
            if (keyPos < globalTokenCount)
                allowed = true;
            else if (windowSize <= 0)
                allowed = true;
            else if (keyPos >= queryPos - windowSize + 1)
                allowed = true;
        }
    } else {
        const int absoluteQuery = queryPositionStart + queryIndex;
        const int absoluteKey = keyIndex;
        if (absoluteKey <= absoluteQuery) {
            if (absoluteKey < globalTokenCount)
                allowed = true;
            else if (windowSize <= 0)
                allowed = true;
            else if (absoluteKey >= absoluteQuery - windowSize + 1)
                allowed = true;
        }
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

__device__ void CudaOps::runZeroForbiddenScoreGradientsInPlace(float* scoresGrad, int keyCount, int queryCount, int queryPositionStart, int windowSize, int globalTokenCount, int segmentLength) {
    const int elementCount = keyCount * queryCount;
    const int index = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (index >= elementCount) return;

    const int keyIndex = index / queryCount;
    const int queryIndex = index - keyIndex * queryCount;

    bool allowed = false;
    if (segmentLength > 0) {
        const int queryPos = queryIndex % segmentLength;
        const int queryBatch = queryIndex / segmentLength;
        const int keyPos = keyIndex % segmentLength;
        const int keyBatch = keyIndex / segmentLength;
        if (queryBatch == keyBatch && keyPos <= queryPos) {
            if (keyPos < globalTokenCount)
                allowed = true;
            else if (windowSize <= 0)
                allowed = true;
            else if (keyPos >= queryPos - windowSize + 1)
                allowed = true;
        }
    } else {
        const int absoluteQuery = queryPositionStart + queryIndex;
        const int absoluteKey = keyIndex;
        if (absoluteKey <= absoluteQuery) {
            if (absoluteKey < globalTokenCount)
                allowed = true;
            else if (windowSize <= 0)
                allowed = true;
            else if (absoluteKey >= absoluteQuery - windowSize + 1)
                allowed = true;
        }
    }

    if (allowed) return;
    scoresGrad[index] = 0.0f;
}

__device__ void CudaOps::runRotaryRotateInPlace(float* tensor, int headCount, int headDimension, int pairCount, int sequenceLength, int positionOffset, int segmentLength, const float* cosTable, const float* sinTable) {
    const int position = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (position >= sequenceLength) return;

    const int absolutePosition = segmentLength > 0 ? (position % segmentLength) : (positionOffset + position);
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

__device__ void CudaOps::runRotaryRotateInverseInPlace(float* tensor, int headCount, int headDimension, int pairCount, int sequenceLength, int positionOffset, int segmentLength, const float* cosTable, const float* sinTable) {
    const int position = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (position >= sequenceLength) return;

    const int absolutePosition = segmentLength > 0 ? (position % segmentLength) : (positionOffset + position);
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

__device__ void CudaOps::runEmbeddingGatherHalfInto(const __half* weight, const int* tokenIds, float* out, int embeddingDim, int tokenCount, int vocabularySize) {
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

    out[index] = __half2float(weight[tokenId * embeddingDim + dimensionIndex]);
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

__device__ void CudaOps::runEmbeddingZeroRows(float* weightGradient, const int* tokenIds, int embeddingDim, int tokenCount, int vocabularySize) {
    const int elementCount = embeddingDim * tokenCount;
    const int index = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (index >= elementCount) return;

    const int dimensionIndex = index / tokenCount;
    const int tokenIndex = index - dimensionIndex * tokenCount;
    const int tokenId = tokenIds[tokenIndex];
    if (tokenId < 0 || tokenId >= vocabularySize) return;

    weightGradient[tokenId * embeddingDim + dimensionIndex] = 0.0f;
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

__device__ void CudaOps::runSoftmaxCrossEntropyFromLogits(const float* logits, const int* targetTokenIds, float* probabilities, float* logitGradient, float* lossSum, int vocabularySize, int tokenCount, float lossScale, int meanDivisor) {
    const int column = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (column >= tokenCount) return;

    float maximumLogit = logits[column];
    for (int row = 1; row < vocabularySize; ++row) {
        const float value = logits[row * tokenCount + column];
        if (value > maximumLogit)
            maximumLogit = value;
    }

    float exponentialSum = 0.0f;
    for (int row = 0; row < vocabularySize; ++row) {
        const float value = expf(logits[row * tokenCount + column] - maximumLogit);
        probabilities[row * tokenCount + column] = value;
        exponentialSum += value;
    }

    const float inverseSum = 1.0f / exponentialSum;
    const float inverseMeanDivisor = 1.0f / static_cast<float>(meanDivisor);
    const int targetId = targetTokenIds[column];

    for (int row = 0; row < vocabularySize; ++row) {
        const float probability = probabilities[row * tokenCount + column] * inverseSum;
        probabilities[row * tokenCount + column] = probability;
        float gradient = probability * inverseMeanDivisor;
        if (row == targetId)
            gradient -= inverseMeanDivisor;
        logitGradient[row * tokenCount + column] = gradient;
    }

    if (lossSum == nullptr) return;
    if (targetId < 0 || targetId >= vocabularySize) return;
    float targetProbability = probabilities[targetId * tokenCount + column];
    if (targetProbability < 1e-7f) targetProbability = 1e-7f;
    atomicAdd(lossSum, (-logf(targetProbability)) * inverseMeanDivisor * lossScale);
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

__global__ void CudaOpsSoftmaxCrossEntropyFromLogitsEntry(const float* logits, const int* targetTokenIds, float* probabilities, float* logitGradient, float* lossSum, int vocabularySize, int tokenCount, float lossScale, int meanDivisor) {
    CudaOps::runSoftmaxCrossEntropyFromLogits(logits, targetTokenIds, probabilities, logitGradient, lossSum, vocabularySize, tokenCount, lossScale, meanDivisor);
}

__global__ void CudaOpsBroadcastBiasAddEntry(float* product, const float* bias, int rowCount, int columnCount) {
    CudaOps::runBroadcastBiasAddInPlace(product, bias, rowCount, columnCount);
}

__global__ void CudaOpsSiluEntry(const float* input, float* out, int elementCount) {
    CudaOps::runSiluInto(input, out, elementCount);
}

__global__ void CudaOpsSiluMultiplyEntry(const float* gatePreActivation, const float* up, float* out, int elementCount) {
    const int index = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (index >= elementCount) return;
    const float gate = gatePreActivation[index];
    const float silu = gate / (1.0f + expf(-gate));
    out[index] = silu * up[index];
}

__global__ void CudaOpsSwigluFromStackedEntry(
    const float* stackedPreBias,
    const float* gateBias,
    const float* upBias,
    float* gatePreActivation,
    float* up,
    float* gateActivated,
    float* hidden,
    int hiddenRows,
    int columnCount
) {
    const int elementCount = hiddenRows * columnCount;
    const int index = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (index >= elementCount) return;
    const int row = index / columnCount;
    const float gate = stackedPreBias[index] + gateBias[row];
    const float upValue = stackedPreBias[index + elementCount] + upBias[row];
    const float silu = gate / (1.0f + expf(-gate));
    gatePreActivation[index] = gate;
    up[index] = upValue;
    gateActivated[index] = silu;
    hidden[index] = silu * upValue;
}

__global__ void CudaOpsSwigluFromStackedBiasedEntry(
    const float* stackedPreActivation,
    float* gatePreActivation,
    float* up,
    float* gateActivated,
    float* hidden,
    int hiddenRows,
    int columnCount
) {
    const int elementCount = hiddenRows * columnCount;
    const int index = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (index >= elementCount) return;
    const float gate = stackedPreActivation[index];
    const float upValue = stackedPreActivation[index + elementCount];
    const float silu = gate / (1.0f + expf(-gate));
    gatePreActivation[index] = gate;
    up[index] = upValue;
    gateActivated[index] = silu;
    hidden[index] = silu * upValue;
}

__global__ void CudaOpsSwigluBackwardStackedEntry(
    const float* hiddenGradient,
    const float* gatePreActivation,
    const float* up,
    const float* gateActivated,
    float* gateGradient,
    float* upGradient,
    float* stackedGateUpGradient,
    int elementCount
) {
    const int index = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (index >= elementCount) return;

    const float dHidden = hiddenGradient[index];
    const float gatePre = gatePreActivation[index];
    const float sigmoid = 1.0f / (1.0f + expf(-gatePre));
    const float siluDerivative = sigmoid * (1.0f + gatePre * (1.0f - sigmoid));

    const float dUp = dHidden * gateActivated[index];
    const float dGate = dHidden * up[index] * siluDerivative;

    gateGradient[index] = dGate;
    upGradient[index] = dUp;
    stackedGateUpGradient[index] = dGate;
    stackedGateUpGradient[index + elementCount] = dUp;
}

__global__ void CudaOpsSumColumnsStackedHalvesEntry(
    const float* stacked,
    float* firstBiasGradient,
    float* secondBiasGradient,
    int halfRows,
    int columnCount
) {
    const int row = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (row >= halfRows) return;
    float firstSum = 0.0f;
    float secondSum = 0.0f;
    for (int column = 0; column < columnCount; ++column) {
        firstSum += stacked[row * columnCount + column];
        secondSum += stacked[(row + halfRows) * columnCount + column];
    }
    firstBiasGradient[row] = firstSum;
    secondBiasGradient[row] = secondSum;
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

__global__ void CudaOpsAddScaledInPlaceEntry(float* total, const float* delta, float scalar, int elementCount) {
    CudaOps::runAddScaledInPlace(total, delta, scalar, elementCount);
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

__global__ void CudaOpsExtractColumnsEntry(const float* source, float* out, int embeddingDim, int sourceStrideColumns, int sourceStartColumn, int columnCount) {
    CudaOps::runExtractColumnsInto(source, out, embeddingDim, sourceStrideColumns, sourceStartColumn, columnCount);
}

__global__ void CudaOpsAddColumnsEntry(float* destination, const float* source, int embeddingDim, int destinationStrideColumns, int destinationStartColumn, int sourceColumnCount) {
    CudaOps::runAddColumnsInPlace(destination, source, embeddingDim, destinationStrideColumns, destinationStartColumn, sourceColumnCount);
}

__global__ void CudaOpsCausalMaskEntry(float* scores, int sequenceLength) {
    CudaOps::runApplyCausalMaskInPlace(scores, sequenceLength);
}

__global__ void CudaOpsSparseAttentionMaskEntry(float* scores, int keyCount, int queryCount, int queryPositionStart, int windowSize, int globalTokenCount, int segmentLength) {
    CudaOps::runApplySparseAttentionMaskInPlace(scores, keyCount, queryCount, queryPositionStart, windowSize, globalTokenCount, segmentLength);
}

__global__ void CudaOpsSoftmaxEntry(const float* logits, float* out, int rowCount, int columnCount) {
    CudaOps::runSoftmaxInto(logits, out, rowCount, columnCount);
}

__global__ void CudaOpsSoftmaxBackwardEntry(const float* probabilities, const float* probabilityGradient, float* scoreGradient, int rowCount, int columnCount) {
    CudaOps::runSoftmaxBackwardInto(probabilities, probabilityGradient, scoreGradient, rowCount, columnCount);
}

__global__ void CudaOpsZeroForbiddenScoreGradientsEntry(float* scoresGrad, int keyCount, int queryCount, int queryPositionStart, int windowSize, int globalTokenCount, int segmentLength) {
    CudaOps::runZeroForbiddenScoreGradientsInPlace(scoresGrad, keyCount, queryCount, queryPositionStart, windowSize, globalTokenCount, segmentLength);
}

__global__ void CudaOpsRotaryRotateEntry(float* tensor, int headCount, int headDimension, int pairCount, int sequenceLength, int positionOffset, int segmentLength, const float* cosTable, const float* sinTable) {
    CudaOps::runRotaryRotateInPlace(tensor, headCount, headDimension, pairCount, sequenceLength, positionOffset, segmentLength, cosTable, sinTable);
}

__global__ void CudaOpsRotaryRotateInverseEntry(float* tensor, int headCount, int headDimension, int pairCount, int sequenceLength, int positionOffset, int segmentLength, const float* cosTable, const float* sinTable) {
    CudaOps::runRotaryRotateInverseInPlace(tensor, headCount, headDimension, pairCount, sequenceLength, positionOffset, segmentLength, cosTable, sinTable);
}

__global__ void CudaOpsEmbeddingGatherEntry(const float* weight, const int* tokenIds, float* out, int embeddingDim, int tokenCount, int vocabularySize) {
    CudaOps::runEmbeddingGatherInto(weight, tokenIds, out, embeddingDim, tokenCount, vocabularySize);
}

__global__ void CudaOpsEmbeddingGatherHalfEntry(const __half* weight, const int* tokenIds, float* out, int embeddingDim, int tokenCount, int vocabularySize) {
    CudaOps::runEmbeddingGatherHalfInto(weight, tokenIds, out, embeddingDim, tokenCount, vocabularySize);
}

__global__ void CudaOpsEmbeddingScatterAddEntry(float* weightGradient, const int* tokenIds, const float* outputGradient, int embeddingDim, int tokenCount, int vocabularySize) {
    CudaOps::runEmbeddingScatterAddInto(weightGradient, tokenIds, outputGradient, embeddingDim, tokenCount, vocabularySize);
}

__global__ void CudaOpsEmbeddingZeroRowsEntry(float* weightGradient, const int* tokenIds, int embeddingDim, int tokenCount, int vocabularySize) {
    CudaOps::runEmbeddingZeroRows(weightGradient, tokenIds, embeddingDim, tokenCount, vocabularySize);
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
    CudaOpsBroadcastBiasAddEntry<<<blockCount, CudaOps::threadCount, 0, CudaMatmul::activeStream()>>>(product.buffer.deviceData, bias.buffer.deviceData, rowCount, columnCount);
    CudaMatmul::throwIfCudaFailed(cudaGetLastError(), "CudaOpsBroadcastBiasAddEntry launch");
}

void CudaOps::siluInto(const CudaMatrix& input, CudaMatrix& out) {
    if (input.empty()) throw std::invalid_argument("CudaOps::siluInto empty input");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaOps::siluInto no CUDA device");

    out.ensureSize(input.rows, input.cols);
    const int elementCount = static_cast<int>(input.elementCount());
    const int blockCount = (elementCount + CudaOps::threadCount - 1) / CudaOps::threadCount;
    CudaOpsSiluEntry<<<blockCount, CudaOps::threadCount, 0, CudaMatmul::activeStream()>>>(input.buffer.deviceData, out.buffer.deviceData, elementCount);
    CudaMatmul::throwIfCudaFailed(cudaGetLastError(), "CudaOpsSiluEntry launch");
}

void CudaOps::siluMultiplyInto(const CudaMatrix& gatePreActivation, const CudaMatrix& up, CudaMatrix& out) {
    if (gatePreActivation.empty() || up.empty()) throw std::invalid_argument("CudaOps::siluMultiplyInto empty input");
    if (gatePreActivation.rows != up.rows || gatePreActivation.cols != up.cols)
        throw std::invalid_argument("CudaOps::siluMultiplyInto shape mismatch");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaOps::siluMultiplyInto no CUDA device");

    out.ensureSize(gatePreActivation.rows, gatePreActivation.cols);
    const int elementCount = static_cast<int>(gatePreActivation.elementCount());
    const int blockCount = (elementCount + CudaOps::threadCount - 1) / CudaOps::threadCount;
    CudaOpsSiluMultiplyEntry<<<blockCount, CudaOps::threadCount, 0, CudaMatmul::activeStream()>>>(
        gatePreActivation.buffer.deviceData, up.buffer.deviceData, out.buffer.deviceData, elementCount);
    CudaMatmul::throwIfCudaFailed(cudaGetLastError(), "CudaOpsSiluMultiplyEntry launch");
}

void CudaOps::swigluFromStackedPreBias(
    const CudaMatrix& stackedPreBias,
    const CudaMatrix& gateBias,
    const CudaMatrix& upBias,
    CudaMatrix& gatePreActivation,
    CudaMatrix& up,
    CudaMatrix& gateActivated,
    CudaMatrix& hidden
) {
    if (stackedPreBias.empty()) throw std::invalid_argument("CudaOps::swigluFromStackedPreBias empty stacked");
    if (gateBias.empty() || upBias.empty()) throw std::invalid_argument("CudaOps::swigluFromStackedPreBias empty bias");
    if (stackedPreBias.rows != gateBias.rows + upBias.rows)
        throw std::invalid_argument("CudaOps::swigluFromStackedPreBias stacked rows mismatch");
    if (gateBias.rows != upBias.rows) throw std::invalid_argument("CudaOps::swigluFromStackedPreBias bias rows mismatch");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaOps::swigluFromStackedPreBias no CUDA device");

    const int hiddenRows = static_cast<int>(gateBias.rows);
    const int columnCount = static_cast<int>(stackedPreBias.cols);
    gatePreActivation.ensureSize(static_cast<size_t>(hiddenRows), static_cast<size_t>(columnCount));
    up.ensureSize(static_cast<size_t>(hiddenRows), static_cast<size_t>(columnCount));
    gateActivated.ensureSize(static_cast<size_t>(hiddenRows), static_cast<size_t>(columnCount));
    hidden.ensureSize(static_cast<size_t>(hiddenRows), static_cast<size_t>(columnCount));

    const int elementCount = hiddenRows * columnCount;
    const int blockCount = (elementCount + CudaOps::threadCount - 1) / CudaOps::threadCount;
    CudaOpsSwigluFromStackedEntry<<<blockCount, CudaOps::threadCount, 0, CudaMatmul::activeStream()>>>(
        stackedPreBias.buffer.deviceData,
        gateBias.buffer.deviceData,
        upBias.buffer.deviceData,
        gatePreActivation.buffer.deviceData,
        up.buffer.deviceData,
        gateActivated.buffer.deviceData,
        hidden.buffer.deviceData,
        hiddenRows,
        columnCount);
    CudaMatmul::throwIfCudaFailed(cudaGetLastError(), "CudaOpsSwigluFromStackedEntry launch");
}

void CudaOps::swigluFromStacked(
    const CudaMatrix& stackedPreActivation,
    CudaMatrix& gatePreActivation,
    CudaMatrix& up,
    CudaMatrix& gateActivated,
    CudaMatrix& hidden
) {
    if (stackedPreActivation.empty()) throw std::invalid_argument("CudaOps::swigluFromStacked empty stacked");
    if (stackedPreActivation.rows % 2ull != 0)
        throw std::invalid_argument("CudaOps::swigluFromStacked stacked rows must be even");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaOps::swigluFromStacked no CUDA device");

    const int hiddenRows = static_cast<int>(stackedPreActivation.rows / 2ull);
    const int columnCount = static_cast<int>(stackedPreActivation.cols);
    gatePreActivation.ensureSize(static_cast<size_t>(hiddenRows), static_cast<size_t>(columnCount));
    up.ensureSize(static_cast<size_t>(hiddenRows), static_cast<size_t>(columnCount));
    gateActivated.ensureSize(static_cast<size_t>(hiddenRows), static_cast<size_t>(columnCount));
    hidden.ensureSize(static_cast<size_t>(hiddenRows), static_cast<size_t>(columnCount));

    const int elementCount = hiddenRows * columnCount;
    const int blockCount = (elementCount + CudaOps::threadCount - 1) / CudaOps::threadCount;
    CudaOpsSwigluFromStackedBiasedEntry<<<blockCount, CudaOps::threadCount, 0, CudaMatmul::activeStream()>>>(
        stackedPreActivation.buffer.deviceData,
        gatePreActivation.buffer.deviceData,
        up.buffer.deviceData,
        gateActivated.buffer.deviceData,
        hidden.buffer.deviceData,
        hiddenRows,
        columnCount);
    CudaMatmul::throwIfCudaFailed(cudaGetLastError(), "CudaOpsSwigluFromStackedBiasedEntry launch");
}

void CudaOps::swigluBackwardIntoStacked(
    const CudaMatrix& hiddenGradient,
    const CudaMatrix& gatePreActivation,
    const CudaMatrix& up,
    const CudaMatrix& gateActivated,
    CudaMatrix& gateGradient,
    CudaMatrix& upGradient,
    CudaMatrix& stackedGateUpGradient
) {
    if (hiddenGradient.empty()) throw std::invalid_argument("CudaOps::swigluBackwardIntoStacked empty hiddenGradient");
    if (gatePreActivation.rows != hiddenGradient.rows || gatePreActivation.cols != hiddenGradient.cols
        || up.rows != hiddenGradient.rows || up.cols != hiddenGradient.cols
        || gateActivated.rows != hiddenGradient.rows || gateActivated.cols != hiddenGradient.cols)
        throw std::invalid_argument("CudaOps::swigluBackwardIntoStacked shape mismatch");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaOps::swigluBackwardIntoStacked no CUDA device");

    const size_t hiddenRows = hiddenGradient.rows;
    const size_t columnCount = hiddenGradient.cols;
    gateGradient.ensureSize(hiddenRows, columnCount);
    upGradient.ensureSize(hiddenRows, columnCount);
    stackedGateUpGradient.ensureSize(hiddenRows * 2ull, columnCount);

    const int elementCount = static_cast<int>(hiddenGradient.elementCount());
    const int blockCount = (elementCount + CudaOps::threadCount - 1) / CudaOps::threadCount;
    CudaOpsSwigluBackwardStackedEntry<<<blockCount, CudaOps::threadCount, 0, CudaMatmul::activeStream()>>>(
        hiddenGradient.buffer.deviceData,
        gatePreActivation.buffer.deviceData,
        up.buffer.deviceData,
        gateActivated.buffer.deviceData,
        gateGradient.buffer.deviceData,
        upGradient.buffer.deviceData,
        stackedGateUpGradient.buffer.deviceData,
        elementCount);
    CudaMatmul::throwIfCudaFailed(cudaGetLastError(), "CudaOpsSwigluBackwardStackedEntry launch");
}

void CudaOps::sumColumnsStackedHalvesInto(const CudaMatrix& stacked, CudaMatrix& firstBiasGradient, CudaMatrix& secondBiasGradient) {
    if (stacked.empty()) throw std::invalid_argument("CudaOps::sumColumnsStackedHalvesInto empty stacked");
    if (stacked.rows % 2ull != 0)
        throw std::invalid_argument("CudaOps::sumColumnsStackedHalvesInto stacked rows must be even");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaOps::sumColumnsStackedHalvesInto no CUDA device");

    const int halfRows = static_cast<int>(stacked.rows / 2ull);
    const int columnCount = static_cast<int>(stacked.cols);
    firstBiasGradient.ensureSize(static_cast<size_t>(halfRows), 1);
    secondBiasGradient.ensureSize(static_cast<size_t>(halfRows), 1);

    const int blockCount = (halfRows + CudaOps::threadCount - 1) / CudaOps::threadCount;
    CudaOpsSumColumnsStackedHalvesEntry<<<blockCount, CudaOps::threadCount, 0, CudaMatmul::activeStream()>>>(
        stacked.buffer.deviceData,
        firstBiasGradient.buffer.deviceData,
        secondBiasGradient.buffer.deviceData,
        halfRows,
        columnCount);
    CudaMatmul::throwIfCudaFailed(cudaGetLastError(), "CudaOpsSumColumnsStackedHalvesEntry launch");
}

void CudaOps::siluDerivativeInto(const CudaMatrix& input, CudaMatrix& out) {
    if (input.empty()) throw std::invalid_argument("CudaOps::siluDerivativeInto empty input");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaOps::siluDerivativeInto no CUDA device");

    out.ensureSize(input.rows, input.cols);
    const int elementCount = static_cast<int>(input.elementCount());
    const int blockCount = (elementCount + CudaOps::threadCount - 1) / CudaOps::threadCount;
    CudaOpsSiluDerivativeEntry<<<blockCount, CudaOps::threadCount, 0, CudaMatmul::activeStream()>>>(input.buffer.deviceData, out.buffer.deviceData, elementCount);
    CudaMatmul::throwIfCudaFailed(cudaGetLastError(), "CudaOpsSiluDerivativeEntry launch");
}

void CudaOps::multiplyElementwiseInto(const CudaMatrix& left, const CudaMatrix& right, CudaMatrix& out) {
    if (left.empty() || right.empty()) throw std::invalid_argument("CudaOps::multiplyElementwiseInto empty input");
    if (left.rows != right.rows || left.cols != right.cols) throw std::invalid_argument("CudaOps::multiplyElementwiseInto shape mismatch");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaOps::multiplyElementwiseInto no CUDA device");

    out.ensureSize(left.rows, left.cols);
    const int elementCount = static_cast<int>(left.elementCount());
    const int blockCount = (elementCount + CudaOps::threadCount - 1) / CudaOps::threadCount;
    CudaOpsMultiplyElementwiseEntry<<<blockCount, CudaOps::threadCount, 0, CudaMatmul::activeStream()>>>(left.buffer.deviceData, right.buffer.deviceData, out.buffer.deviceData, elementCount);
    CudaMatmul::throwIfCudaFailed(cudaGetLastError(), "CudaOpsMultiplyElementwiseEntry launch");
}

void CudaOps::multiplyElementwiseInPlace(CudaMatrix& total, const CudaMatrix& other) {
    if (total.empty() || other.empty()) throw std::invalid_argument("CudaOps::multiplyElementwiseInPlace empty input");
    if (total.rows != other.rows || total.cols != other.cols) throw std::invalid_argument("CudaOps::multiplyElementwiseInPlace shape mismatch");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaOps::multiplyElementwiseInPlace no CUDA device");

    const int elementCount = static_cast<int>(total.elementCount());
    const int blockCount = (elementCount + CudaOps::threadCount - 1) / CudaOps::threadCount;
    CudaOpsMultiplyElementwiseInPlaceEntry<<<blockCount, CudaOps::threadCount, 0, CudaMatmul::activeStream()>>>(total.buffer.deviceData, other.buffer.deviceData, elementCount);
    CudaMatmul::throwIfCudaFailed(cudaGetLastError(), "CudaOpsMultiplyElementwiseInPlaceEntry launch");
}

void CudaOps::addInto(const CudaMatrix& left, const CudaMatrix& right, CudaMatrix& out) {
    if (left.empty() || right.empty()) throw std::invalid_argument("CudaOps::addInto empty input");
    if (left.rows != right.rows || left.cols != right.cols) throw std::invalid_argument("CudaOps::addInto shape mismatch");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaOps::addInto no CUDA device");

    out.ensureSize(left.rows, left.cols);
    const int elementCount = static_cast<int>(left.elementCount());
    const int blockCount = (elementCount + CudaOps::threadCount - 1) / CudaOps::threadCount;
    CudaOpsAddEntry<<<blockCount, CudaOps::threadCount, 0, CudaMatmul::activeStream()>>>(left.buffer.deviceData, right.buffer.deviceData, out.buffer.deviceData, elementCount);
    CudaMatmul::throwIfCudaFailed(cudaGetLastError(), "CudaOpsAddEntry launch");
}

void CudaOps::addInPlace(CudaMatrix& total, const CudaMatrix& delta) {
    if (total.empty() || delta.empty()) throw std::invalid_argument("CudaOps::addInPlace empty input");
    if (total.rows != delta.rows || total.cols != delta.cols) throw std::invalid_argument("CudaOps::addInPlace shape mismatch");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaOps::addInPlace no CUDA device");

    const int elementCount = static_cast<int>(total.elementCount());
    const int blockCount = (elementCount + CudaOps::threadCount - 1) / CudaOps::threadCount;
    CudaOpsAddInPlaceEntry<<<blockCount, CudaOps::threadCount, 0, CudaMatmul::activeStream()>>>(total.buffer.deviceData, delta.buffer.deviceData, elementCount);
    CudaMatmul::throwIfCudaFailed(cudaGetLastError(), "CudaOpsAddInPlaceEntry launch");
}

void CudaOps::addScaledInPlace(CudaMatrix& total, const CudaMatrix& delta, float scalar) {
    if (total.empty() || delta.empty()) throw std::invalid_argument("CudaOps::addScaledInPlace empty input");
    if (total.rows != delta.rows || total.cols != delta.cols) throw std::invalid_argument("CudaOps::addScaledInPlace shape mismatch");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaOps::addScaledInPlace no CUDA device");

    const int elementCount = static_cast<int>(total.elementCount());
    const int blockCount = (elementCount + CudaOps::threadCount - 1) / CudaOps::threadCount;
    CudaOpsAddScaledInPlaceEntry<<<blockCount, CudaOps::threadCount, 0, CudaMatmul::activeStream()>>>(
        total.buffer.deviceData, delta.buffer.deviceData, scalar, elementCount);
    CudaMatmul::throwIfCudaFailed(cudaGetLastError(), "CudaOpsAddScaledInPlaceEntry launch");
}

void CudaOps::sumColumnsInto(const CudaMatrix& gradient, CudaMatrix& biasGradient) {
    if (gradient.empty()) throw std::invalid_argument("CudaOps::sumColumnsInto empty gradient");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaOps::sumColumnsInto no CUDA device");

    biasGradient.ensureSize(gradient.rows, 1);
    const int rowCount = static_cast<int>(gradient.rows);
    const int columnCount = static_cast<int>(gradient.cols);
    const int blockCount = (rowCount + CudaOps::threadCount - 1) / CudaOps::threadCount;
    CudaOpsSumColumnsEntry<<<blockCount, CudaOps::threadCount, 0, CudaMatmul::activeStream()>>>(gradient.buffer.deviceData, biasGradient.buffer.deviceData, rowCount, columnCount);
    CudaMatmul::throwIfCudaFailed(cudaGetLastError(), "CudaOpsSumColumnsEntry launch");
}

void CudaOps::copyInto(const CudaMatrix& source, CudaMatrix& destination) {
    if (source.empty()) throw std::invalid_argument("CudaOps::copyInto empty source");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaOps::copyInto no CUDA device");

    destination.ensureSize(source.rows, source.cols);
    CudaMatmul::memcpyDevice(destination.buffer.deviceData, source.buffer.deviceData, source.byteCount());
}

void CudaOps::scaleInPlace(CudaMatrix& matrix, float scalar) {
    if (matrix.empty()) throw std::invalid_argument("CudaOps::scaleInPlace empty matrix");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaOps::scaleInPlace no CUDA device");

    const int elementCount = static_cast<int>(matrix.elementCount());
    const int blockCount = (elementCount + CudaOps::threadCount - 1) / CudaOps::threadCount;
    CudaOpsScaleEntry<<<blockCount, CudaOps::threadCount, 0, CudaMatmul::activeStream()>>>(matrix.buffer.deviceData, scalar, elementCount);
    CudaMatmul::throwIfCudaFailed(cudaGetLastError(), "CudaOpsScaleEntry launch");
}

void CudaOps::zeroInPlace(CudaMatrix& matrix) {
    if (matrix.empty()) return;
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaOps::zeroInPlace no CUDA device");

    CudaMatmul::memsetDevice(matrix.buffer.deviceData, 0, matrix.byteCount());
}

void CudaOps::downloadAddIntoHost(Matrix& hostTotal, const CudaMatrix& deviceDelta) {
    if (deviceDelta.empty()) throw std::invalid_argument("CudaOps::downloadAddIntoHost empty deviceDelta");
    if (hostTotal.empty())
        hostTotal.ensureSize(deviceDelta.rows, deviceDelta.cols);
    if (hostTotal.rows != deviceDelta.rows || hostTotal.cols != deviceDelta.cols)
        throw std::invalid_argument("CudaOps::downloadAddIntoHost shape mismatch");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaOps::downloadAddIntoHost no CUDA device");

    const bool timeIt = CudaOps::downloadAddIntoHostSecondsSink != nullptr;
    const auto t0 = timeIt ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    // PreferHostSgd microbatches overwrite fresh host grads — avoid alloc+add temp Matrix.
    if (CudaAdam::preferHostSgd) {
        CudaOps::downloadIntoHost(hostTotal, deviceDelta);
    } else {
        Matrix delta = deviceDelta.download();
        Matrix::addInPlace(hostTotal, delta);
    }
    if (timeIt) {
        *CudaOps::downloadAddIntoHostSecondsSink += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
    }
}

void CudaOps::downloadIntoHost(Matrix& hostOut, const CudaMatrix& deviceSource) {
    if (deviceSource.empty()) throw std::invalid_argument("CudaOps::downloadIntoHost empty deviceSource");
    if (!deviceSource.hasDeviceStorage()) throw std::invalid_argument("CudaOps::downloadIntoHost missing device storage");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaOps::downloadIntoHost no CUDA device");

    hostOut.ensureSize(deviceSource.rows, deviceSource.cols);
    const size_t elementCount = deviceSource.elementCount();
    float* hostData = hostOut.data.data();
    const float* deviceData = deviceSource.buffer.deviceData;
    constexpr size_t kDirectThreshold = 1ull << 20; // 1M floats ≈ 4 MiB
    if (elementCount <= kDirectThreshold) {
        CudaMatmul::throwIfCudaFailed(
            cudaMemcpy(hostData, deviceData, elementCount * sizeof(float), cudaMemcpyDeviceToHost),
            "CudaOps::downloadIntoHost D2H");
        return;
    }

    constexpr size_t kChunkElements = 16ull * 1024ull * 1024ull; // 16M floats ≈ 64 MiB
    cudaStream_t stream = CudaMatmul::activeStream();
    float* pendingPinned = nullptr;
    size_t pendingOffset = 0;
    size_t pendingChunk = 0;
    int pendingSlot = -1;

    auto flushPending = [&]() {
        if (pendingPinned == nullptr || pendingSlot < 0) return;
        gPinnedFloatDownloadStaging.syncSlot(pendingSlot);
        std::memcpy(hostData + pendingOffset, pendingPinned, pendingChunk * sizeof(float));
        pendingPinned = nullptr;
        pendingSlot = -1;
    };

    for (size_t offset = 0; offset < elementCount; offset += kChunkElements) {
        const size_t chunk = (std::min)(kChunkElements, elementCount - offset);
        int slot = 0;
        float* pinned = static_cast<float*>(gPinnedFloatDownloadStaging.acquire(chunk * sizeof(float), &slot));
        CudaMatmul::throwIfCudaFailed(
            cudaMemcpyAsync(pinned, deviceData + offset, chunk * sizeof(float), cudaMemcpyDeviceToHost, stream),
            "CudaOps::downloadIntoHost D2H pinned");
        gPinnedFloatDownloadStaging.record(stream, slot);
        flushPending(); // overlap: CPU memcpy of previous chunk while this D2H runs
        pendingPinned = pinned;
        pendingOffset = offset;
        pendingChunk = chunk;
        pendingSlot = slot;
    }
    flushPending();
}

void CudaOps::downloadIntoHostAsync(Matrix& hostOut, const CudaMatrix& deviceSource, cudaStream_t stream) {
    if (deviceSource.empty()) throw std::invalid_argument("CudaOps::downloadIntoHostAsync empty deviceSource");
    CudaOps::downloadIntoHostAsyncSlice(hostOut, deviceSource, 0, deviceSource.rows, deviceSource.cols, stream);
}

namespace {
__global__ void CudaOpsCastFloatToHalfEntry(const float* source, __half* destination, int elementCount) {
    const int index = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (index >= elementCount) return;
    destination[index] = __float2half(source[index]);
}

struct FusedHalfGradPiece {
    Matrix* host = nullptr;
    const float* deviceFloat = nullptr;
    size_t halfOffset = 0;
    size_t elementCount = 0;
    size_t rows = 0;
    size_t cols = 0;
};

// Pipeline depth: in-flight D2H + async expand/SGD + next commit. Shared (not TLS) so the
// SGD worker can pop batches committed on the main thread.
constexpr int kFusedHalfHostSlots = 8;
struct FusedHalfHostSlot {
    uint16_t* data = nullptr;
    size_t capacityBytes = 0;
    bool inUse = false;
};
struct FusedHalfDeviceSlot {
    CudaDeviceBuffer pack;
    bool inUse = false;
};
struct FusedHalfGradBatch {
    std::vector<FusedHalfGradPiece> pieces;
    uint16_t* hostHalf = nullptr;
    FusedHalfDeviceSlot* deviceSlot = nullptr;
    size_t elementCount = 0;
};

FusedHalfHostSlot g_fusedHalfHostSlots[kFusedHalfHostSlots];
FusedHalfDeviceSlot g_fusedHalfDeviceSlots[kFusedHalfHostSlots];
std::mutex g_fusedHalfReadyMutex;
std::mutex g_fusedHalfFreelistMutex;
std::vector<FusedHalfGradBatch> g_fusedHalfReadyBatches;
thread_local std::vector<FusedHalfGradPiece> g_fusedHalfOpenPieces;
thread_local size_t g_fusedHalfOpenElements = 0;

uint16_t* acquireFusedHalfHost(size_t bytes) {
    std::lock_guard<std::mutex> lock(g_fusedHalfFreelistMutex);
    for (int i = 0; i < kFusedHalfHostSlots; ++i) {
        FusedHalfHostSlot& slot = g_fusedHalfHostSlots[i];
        if (slot.inUse) continue;
        if (slot.capacityBytes < bytes) {
            if (slot.data != nullptr) {
                cudaFreeHost(slot.data);
                slot.data = nullptr;
                slot.capacityBytes = 0;
            }
            void* pinned = nullptr;
            const cudaError_t status = cudaMallocHost(&pinned, bytes);
            if (status != cudaSuccess)
                throw std::runtime_error(std::string("cudaMallocHost fused half grads: ") + cudaGetErrorString(status));
            slot.data = static_cast<uint16_t*>(pinned);
            slot.capacityBytes = bytes;
        }
        slot.inUse = true;
        return slot.data;
    }
    throw std::runtime_error("CudaOps fused half host freelist exhausted");
}

FusedHalfDeviceSlot* acquireFusedHalfDevice(size_t halfBytes) {
    std::lock_guard<std::mutex> lock(g_fusedHalfFreelistMutex);
    for (int i = 0; i < kFusedHalfHostSlots; ++i) {
        FusedHalfDeviceSlot& slot = g_fusedHalfDeviceSlots[i];
        if (slot.inUse) continue;
        slot.pack.ensureCapacity(halfBytes);
        slot.inUse = true;
        return &slot;
    }
    throw std::runtime_error("CudaOps fused half device freelist exhausted");
}

void releaseFusedHalfHost(uint16_t* data) {
    if (data == nullptr) return;
    std::lock_guard<std::mutex> lock(g_fusedHalfFreelistMutex);
    for (int i = 0; i < kFusedHalfHostSlots; ++i) {
        if (g_fusedHalfHostSlots[i].data == data) {
            g_fusedHalfHostSlots[i].inUse = false;
            return;
        }
    }
}

void releaseFusedHalfDevice(FusedHalfDeviceSlot* slot) {
    if (slot == nullptr) return;
    std::lock_guard<std::mutex> lock(g_fusedHalfFreelistMutex);
    slot->inUse = false;
}
} // namespace

void CudaOps::beginFusedHalfGradOffload() {
    g_fusedHalfOpenPieces.clear();
    g_fusedHalfOpenElements = 0;
}

void CudaOps::appendFusedHalfGradOffload(Matrix& hostOut, const float* deviceFloat, size_t rows, size_t cols) {
    if (deviceFloat == nullptr) throw std::invalid_argument("appendFusedHalfGradOffload null deviceFloat");
    if (rows == 0 || cols == 0) throw std::invalid_argument("appendFusedHalfGradOffload empty shape");
    const size_t elementCount = rows * cols;
    hostOut.ensureSize(rows, cols);
    FusedHalfGradPiece piece;
    piece.host = &hostOut;
    piece.deviceFloat = deviceFloat;
    piece.halfOffset = g_fusedHalfOpenElements;
    piece.elementCount = elementCount;
    piece.rows = rows;
    piece.cols = cols;
    g_fusedHalfOpenPieces.push_back(piece);
    g_fusedHalfOpenElements += elementCount;
}

void CudaOps::commitFusedHalfGradOffload(cudaStream_t stream) {
    if (g_fusedHalfOpenPieces.empty() || g_fusedHalfOpenElements == 0) return;
    if (stream == nullptr) throw std::invalid_argument("commitFusedHalfGradOffload null stream");

    const size_t halfBytes = g_fusedHalfOpenElements * sizeof(__half);
    FusedHalfDeviceSlot* deviceSlot = acquireFusedHalfDevice(halfBytes);
    uint16_t* hostHalf = acquireFusedHalfHost(halfBytes);
    __half* deviceHalf = reinterpret_cast<__half*>(deviceSlot->pack.deviceData);

    for (const FusedHalfGradPiece& piece : g_fusedHalfOpenPieces) {
        const int n = static_cast<int>(piece.elementCount);
        const int threads = 256;
        const int blocks = (n + threads - 1) / threads;
        CudaOpsCastFloatToHalfEntry<<<blocks, threads, 0, stream>>>(
            piece.deviceFloat,
            deviceHalf + piece.halfOffset,
            n);
        CudaMatmul::throwIfCudaFailed(cudaGetLastError(), "CudaOpsCastFloatToHalfEntry launch");
    }

    CudaMatmul::throwIfCudaFailed(
        cudaMemcpyAsync(hostHalf, deviceHalf, halfBytes, cudaMemcpyDeviceToHost, stream),
        "commitFusedHalfGradOffload D2H");

    FusedHalfGradBatch batch;
    batch.pieces = std::move(g_fusedHalfOpenPieces);
    batch.hostHalf = hostHalf;
    batch.deviceSlot = deviceSlot;
    batch.elementCount = g_fusedHalfOpenElements;
    {
        std::lock_guard<std::mutex> lock(g_fusedHalfReadyMutex);
        g_fusedHalfReadyBatches.push_back(std::move(batch));
    }
    g_fusedHalfOpenPieces.clear();
    g_fusedHalfOpenElements = 0;
}

void CudaOps::commitPinnedD2hBatch() {
    // Fused half path commits inside enqueueDeferred via commitFusedHalfGradOffload.
}

void CudaOps::flushPinnedD2hToHost() {
    FusedHalfGradBatch batch;
    {
        std::lock_guard<std::mutex> lock(g_fusedHalfReadyMutex);
        if (g_fusedHalfReadyBatches.empty()) return;
        batch = std::move(g_fusedHalfReadyBatches.front());
        g_fusedHalfReadyBatches.erase(g_fusedHalfReadyBatches.begin());
    }

    if (batch.deviceSlot != nullptr) {
        releaseFusedHalfDevice(batch.deviceSlot);
        batch.deviceSlot = nullptr;
    }

#if defined(_OPENMP)
    #pragma omp parallel for schedule(dynamic)
#endif
    for (int pieceIndex = 0; pieceIndex < static_cast<int>(batch.pieces.size()); ++pieceIndex) {
        const FusedHalfGradPiece& piece = batch.pieces[static_cast<size_t>(pieceIndex)];
        if (piece.host == nullptr || batch.hostHalf == nullptr) continue;
        piece.host->ensureSize(piece.rows, piece.cols);
        float* dst = piece.host->data.data();
        const __half* src = reinterpret_cast<const __half*>(batch.hostHalf + piece.halfOffset);
        for (size_t i = 0; i < piece.elementCount; ++i)
            dst[i] = __half2float(src[i]);
    }
    releaseFusedHalfHost(batch.hostHalf);
}

void CudaOps::applyFusedHalfHostSgd(float stepScale) {
    FusedHalfGradBatch batch;
    {
        std::lock_guard<std::mutex> lock(g_fusedHalfReadyMutex);
        if (g_fusedHalfReadyBatches.empty()) return;
        batch = std::move(g_fusedHalfReadyBatches.front());
        g_fusedHalfReadyBatches.erase(g_fusedHalfReadyBatches.begin());
    }

    if (batch.deviceSlot != nullptr) {
        releaseFusedHalfDevice(batch.deviceSlot);
        batch.deviceSlot = nullptr;
    }

#if defined(_OPENMP)
    #pragma omp parallel for schedule(dynamic)
#endif
    for (int pieceIndex = 0; pieceIndex < static_cast<int>(batch.pieces.size()); ++pieceIndex) {
        const FusedHalfGradPiece& piece = batch.pieces[static_cast<size_t>(pieceIndex)];
        if (piece.host == nullptr || batch.hostHalf == nullptr) continue;
        if (piece.host->data.size() != piece.elementCount)
            throw std::runtime_error("CudaOps::applyFusedHalfHostSgd master size mismatch");
        float* master = piece.host->data.data();
        const __half* grad = reinterpret_cast<const __half*>(batch.hostHalf + piece.halfOffset);
        for (size_t i = 0; i < piece.elementCount; ++i)
            master[i] -= stepScale * __half2float(grad[i]);
    }
    releaseFusedHalfHost(batch.hostHalf);
}

void CudaOps::releaseCompletedFusedHalfDevices() {
    std::vector<FusedHalfDeviceSlot*> toRelease;
    {
        std::lock_guard<std::mutex> lock(g_fusedHalfReadyMutex);
        for (FusedHalfGradBatch& batch : g_fusedHalfReadyBatches) {
            if (batch.deviceSlot == nullptr) continue;
            toRelease.push_back(batch.deviceSlot);
            batch.deviceSlot = nullptr;
        }
    }
    for (FusedHalfDeviceSlot* slot : toRelease)
        releaseFusedHalfDevice(slot);
}

void CudaOps::downloadIntoHostAsyncSlice(
    Matrix& hostOut,
    const CudaMatrix& deviceSource,
    size_t elementOffset,
    size_t rows,
    size_t cols,
    cudaStream_t stream
) {
    if (deviceSource.empty()) throw std::invalid_argument("CudaOps::downloadIntoHostAsyncSlice empty deviceSource");
    if (!deviceSource.hasDeviceStorage()) throw std::invalid_argument("CudaOps::downloadIntoHostAsyncSlice missing device storage");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaOps::downloadIntoHostAsyncSlice no CUDA device");
    if (stream == nullptr) throw std::invalid_argument("CudaOps::downloadIntoHostAsyncSlice null stream");
    if (rows == 0 || cols == 0) throw std::invalid_argument("CudaOps::downloadIntoHostAsyncSlice empty shape");
    const size_t needElements = elementOffset + rows * cols;
    if (needElements > deviceSource.elementCount())
        throw std::invalid_argument("CudaOps::downloadIntoHostAsyncSlice slice out of range");

    hostOut.ensureSize(rows, cols);
    const size_t bytes = rows * cols * sizeof(float);
    float* hostData = hostOut.data.data();
    const float* deviceData = deviceSource.buffer.deviceData + elementOffset;
    const cudaError_t reg = cudaHostRegister(hostData, bytes, cudaHostRegisterDefault);
    if (reg != cudaSuccess && reg != cudaErrorHostMemoryAlreadyRegistered) {
        cudaGetLastError();
        CudaMatmul::throwIfCudaFailed(
            cudaMemcpyAsync(hostData, deviceData, bytes, cudaMemcpyDeviceToHost, stream),
            "CudaOps::downloadIntoHostAsyncSlice D2H fallback");
        CudaMatmul::throwIfCudaFailed(cudaStreamSynchronize(stream), "CudaOps::downloadIntoHostAsyncSlice fallback sync");
        return;
    }
    if (reg == cudaErrorHostMemoryAlreadyRegistered)
        cudaGetLastError();
    CudaMatmul::throwIfCudaFailed(
        cudaMemcpyAsync(hostData, deviceData, bytes, cudaMemcpyDeviceToHost, stream),
        "CudaOps::downloadIntoHostAsyncSlice D2H");
}

void CudaOps::unregisterHostMatrix(Matrix& host) {
    if (host.empty() || host.data.empty()) return;
    const cudaError_t status = cudaHostUnregister(host.data.data());
    if (status != cudaSuccess) {
        // Already unregistered or never registered (fallback path) — clear sticky error.
        cudaGetLastError();
    }
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
    CudaOpsExtractHeadEntry<<<blockCount, CudaOps::threadCount, 0, CudaMatmul::activeStream()>>>(full.buffer.deviceData, head.buffer.deviceData, headIndex, headDimension, sourceStrideColumns, usedColumnCount);
    CudaMatmul::throwIfCudaFailed(cudaGetLastError(), "CudaOpsExtractHeadEntry launch");
}

void CudaOps::writeHead(CudaMatrix& full, int headIndex, int headDimension, const CudaMatrix& head) {
    if (full.empty() || head.empty()) throw std::invalid_argument("CudaOps::writeHead empty input");
    if (head.rows != static_cast<size_t>(headDimension) || head.cols != full.cols) throw std::invalid_argument("CudaOps::writeHead shape mismatch");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaOps::writeHead no CUDA device");

    const int sequenceLength = static_cast<int>(full.cols);
    const int elementCount = headDimension * sequenceLength;
    const int blockCount = (elementCount + CudaOps::threadCount - 1) / CudaOps::threadCount;
    CudaOpsWriteHeadEntry<<<blockCount, CudaOps::threadCount, 0, CudaMatmul::activeStream()>>>(full.buffer.deviceData, head.buffer.deviceData, headIndex, headDimension, sequenceLength, static_cast<int>(full.rows));
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
    CudaOpsWriteColumnsEntry<<<blockCount, CudaOps::threadCount, 0, CudaMatmul::activeStream()>>>(destination.buffer.deviceData, source.buffer.deviceData, embeddingDim, destinationStrideColumns, destinationStartColumn, sourceColumnCount);
    CudaMatmul::throwIfCudaFailed(cudaGetLastError(), "CudaOpsWriteColumnsEntry launch");
}

void CudaOps::extractColumnsInto(const CudaMatrix& source, int sourceStartColumn, int columnCount, CudaMatrix& out) {
    if (source.empty()) throw std::invalid_argument("CudaOps::extractColumnsInto empty source");
    if (columnCount <= 0) throw std::invalid_argument("CudaOps::extractColumnsInto empty columnCount");
    if (sourceStartColumn < 0) throw std::invalid_argument("CudaOps::extractColumnsInto negative start");
    if (sourceStartColumn + columnCount > static_cast<int>(source.cols))
        throw std::invalid_argument("CudaOps::extractColumnsInto exceeds source width");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaOps::extractColumnsInto no CUDA device");

    out.ensureSize(source.rows, static_cast<size_t>(columnCount));
    const int embeddingDim = static_cast<int>(source.rows);
    const int sourceStrideColumns = static_cast<int>(source.cols);
    const int elementCount = embeddingDim * columnCount;
    const int blockCount = (elementCount + CudaOps::threadCount - 1) / CudaOps::threadCount;
    CudaOpsExtractColumnsEntry<<<blockCount, CudaOps::threadCount, 0, CudaMatmul::activeStream()>>>(source.buffer.deviceData, out.buffer.deviceData, embeddingDim, sourceStrideColumns, sourceStartColumn, columnCount);
    CudaMatmul::throwIfCudaFailed(cudaGetLastError(), "CudaOpsExtractColumnsEntry launch");
}

void CudaOps::addColumnsInPlace(CudaMatrix& destination, int destinationStartColumn, const CudaMatrix& source) {
    if (destination.empty() || source.empty()) throw std::invalid_argument("CudaOps::addColumnsInPlace empty input");
    if (destination.rows != source.rows) throw std::invalid_argument("CudaOps::addColumnsInPlace row mismatch");
    if (destinationStartColumn < 0) throw std::invalid_argument("CudaOps::addColumnsInPlace negative start");
    if (destinationStartColumn + static_cast<int>(source.cols) > static_cast<int>(destination.cols))
        throw std::invalid_argument("CudaOps::addColumnsInPlace exceeds destination width");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaOps::addColumnsInPlace no CUDA device");

    const int embeddingDim = static_cast<int>(source.rows);
    const int destinationStrideColumns = static_cast<int>(destination.cols);
    const int sourceColumnCount = static_cast<int>(source.cols);
    const int elementCount = embeddingDim * sourceColumnCount;
    const int blockCount = (elementCount + CudaOps::threadCount - 1) / CudaOps::threadCount;
    CudaOpsAddColumnsEntry<<<blockCount, CudaOps::threadCount, 0, CudaMatmul::activeStream()>>>(destination.buffer.deviceData, source.buffer.deviceData, embeddingDim, destinationStrideColumns, destinationStartColumn, sourceColumnCount);
    CudaMatmul::throwIfCudaFailed(cudaGetLastError(), "CudaOpsAddColumnsEntry launch");
}

void CudaOps::applyCausalMaskInPlace(CudaMatrix& scores) {
    if (scores.empty()) throw std::invalid_argument("CudaOps::applyCausalMaskInPlace empty scores");
    if (scores.rows != scores.cols) throw std::invalid_argument("CudaOps::applyCausalMaskInPlace scores must be square");
    CudaOps::applySparseAttentionMaskInPlace(scores, static_cast<int>(scores.rows), 0, 0);
}

void CudaOps::applySparseAttentionMaskInPlace(CudaMatrix& scores, int windowSize, int globalTokenCount, int queryPositionStart, int segmentLength) {
    if (scores.empty()) throw std::invalid_argument("CudaOps::applySparseAttentionMaskInPlace empty scores");
    if (globalTokenCount < 0) throw std::invalid_argument("CudaOps::applySparseAttentionMaskInPlace globalTokenCount must be >= 0");
    if (queryPositionStart < 0) throw std::invalid_argument("CudaOps::applySparseAttentionMaskInPlace queryPositionStart must be >= 0");
    if (segmentLength < 0) throw std::invalid_argument("CudaOps::applySparseAttentionMaskInPlace segmentLength must be >= 0");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaOps::applySparseAttentionMaskInPlace no CUDA device");

    const int keyCount = static_cast<int>(scores.rows);
    const int queryCount = static_cast<int>(scores.cols);
    const int elementCount = keyCount * queryCount;
    const int blockCount = (elementCount + CudaOps::threadCount - 1) / CudaOps::threadCount;
    CudaOpsSparseAttentionMaskEntry<<<blockCount, CudaOps::threadCount, 0, CudaMatmul::activeStream()>>>(scores.buffer.deviceData, keyCount, queryCount, queryPositionStart, windowSize, globalTokenCount, segmentLength);
    CudaMatmul::throwIfCudaFailed(cudaGetLastError(), "CudaOpsSparseAttentionMaskEntry launch");
}

void CudaOps::softmaxInto(const CudaMatrix& logits, CudaMatrix& out) {
    if (logits.empty()) throw std::invalid_argument("CudaOps::softmaxInto empty logits");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaOps::softmaxInto no CUDA device");

    out.ensureSize(logits.rows, logits.cols);
    const int rowCount = static_cast<int>(logits.rows);
    const int columnCount = static_cast<int>(logits.cols);
    const int blockCount = (columnCount + CudaOps::threadCount - 1) / CudaOps::threadCount;
    CudaOpsSoftmaxEntry<<<blockCount, CudaOps::threadCount, 0, CudaMatmul::activeStream()>>>(logits.buffer.deviceData, out.buffer.deviceData, rowCount, columnCount);
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
    CudaOpsSoftmaxBackwardEntry<<<blockCount, CudaOps::threadCount, 0, CudaMatmul::activeStream()>>>(probabilities.buffer.deviceData, probabilityGradient.buffer.deviceData, scoreGradient.buffer.deviceData, rowCount, columnCount);
    CudaMatmul::throwIfCudaFailed(cudaGetLastError(), "CudaOpsSoftmaxBackwardEntry launch");
}

void CudaOps::zeroForbiddenScoreGradientsInPlace(CudaMatrix& scoresGrad, int windowSize, int globalTokenCount, int queryPositionStart, int segmentLength) {
    if (scoresGrad.empty()) throw std::invalid_argument("CudaOps::zeroForbiddenScoreGradientsInPlace empty scoresGrad");
    if (globalTokenCount < 0) throw std::invalid_argument("CudaOps::zeroForbiddenScoreGradientsInPlace globalTokenCount must be >= 0");
    if (queryPositionStart < 0) throw std::invalid_argument("CudaOps::zeroForbiddenScoreGradientsInPlace queryPositionStart must be >= 0");
    if (segmentLength < 0) throw std::invalid_argument("CudaOps::zeroForbiddenScoreGradientsInPlace segmentLength must be >= 0");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaOps::zeroForbiddenScoreGradientsInPlace no CUDA device");

    const int keyCount = static_cast<int>(scoresGrad.rows);
    const int queryCount = static_cast<int>(scoresGrad.cols);
    const int elementCount = keyCount * queryCount;
    const int blockCount = (elementCount + CudaOps::threadCount - 1) / CudaOps::threadCount;
    CudaOpsZeroForbiddenScoreGradientsEntry<<<blockCount, CudaOps::threadCount, 0, CudaMatmul::activeStream()>>>(scoresGrad.buffer.deviceData, keyCount, queryCount, queryPositionStart, windowSize, globalTokenCount, segmentLength);
    CudaMatmul::throwIfCudaFailed(cudaGetLastError(), "CudaOpsZeroForbiddenScoreGradientsEntry launch");
}

void CudaOps::rotaryRotateInPlace(CudaMatrix& tensor, int headCount, int headDimension, int pairCount, const CudaMatrix& cosTable, const CudaMatrix& sinTable, int positionOffset, int segmentLength) {
    if (tensor.empty()) throw std::invalid_argument("CudaOps::rotaryRotateInPlace empty tensor");
    if (headCount <= 0 || headDimension <= 0 || pairCount <= 0) throw std::invalid_argument("CudaOps::rotaryRotateInPlace invalid rope dims");
    if (static_cast<int>(tensor.rows) != headCount * headDimension) throw std::invalid_argument("CudaOps::rotaryRotateInPlace embedding dim mismatch");
    if (cosTable.empty() || sinTable.empty()) throw std::invalid_argument("CudaOps::rotaryRotateInPlace empty tables");
    if (positionOffset < 0) throw std::invalid_argument("CudaOps::rotaryRotateInPlace negative positionOffset");
    if (segmentLength < 0) throw std::invalid_argument("CudaOps::rotaryRotateInPlace segmentLength must be >= 0");
    if (segmentLength > 0) {
        if (segmentLength > static_cast<int>(cosTable.rows))
            throw std::invalid_argument("CudaOps::rotaryRotateInPlace segmentLength exceeds RoPE table");
    } else if (positionOffset + static_cast<int>(tensor.cols) > static_cast<int>(cosTable.rows)) {
        throw std::invalid_argument("CudaOps::rotaryRotateInPlace position exceeds RoPE table");
    }
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaOps::rotaryRotateInPlace no CUDA device");

    const int sequenceLength = static_cast<int>(tensor.cols);
    const int blockCount = (sequenceLength + CudaOps::threadCount - 1) / CudaOps::threadCount;
    CudaOpsRotaryRotateEntry<<<blockCount, CudaOps::threadCount, 0, CudaMatmul::activeStream()>>>(tensor.buffer.deviceData, headCount, headDimension, pairCount, sequenceLength, positionOffset, segmentLength, cosTable.buffer.deviceData, sinTable.buffer.deviceData);
    CudaMatmul::throwIfCudaFailed(cudaGetLastError(), "CudaOpsRotaryRotateEntry launch");
}

void CudaOps::rotaryRotateInverseInPlace(CudaMatrix& tensor, int headCount, int headDimension, int pairCount, const CudaMatrix& cosTable, const CudaMatrix& sinTable, int positionOffset, int segmentLength) {
    if (tensor.empty()) throw std::invalid_argument("CudaOps::rotaryRotateInverseInPlace empty tensor");
    if (headCount <= 0 || headDimension <= 0 || pairCount <= 0) throw std::invalid_argument("CudaOps::rotaryRotateInverseInPlace invalid rope dims");
    if (static_cast<int>(tensor.rows) != headCount * headDimension) throw std::invalid_argument("CudaOps::rotaryRotateInverseInPlace embedding dim mismatch");
    if (cosTable.empty() || sinTable.empty()) throw std::invalid_argument("CudaOps::rotaryRotateInverseInPlace empty tables");
    if (positionOffset < 0) throw std::invalid_argument("CudaOps::rotaryRotateInverseInPlace negative positionOffset");
    if (segmentLength < 0) throw std::invalid_argument("CudaOps::rotaryRotateInverseInPlace segmentLength must be >= 0");
    if (segmentLength > 0) {
        if (segmentLength > static_cast<int>(cosTable.rows))
            throw std::invalid_argument("CudaOps::rotaryRotateInverseInPlace segmentLength exceeds RoPE table");
    } else if (positionOffset + static_cast<int>(tensor.cols) > static_cast<int>(cosTable.rows)) {
        throw std::invalid_argument("CudaOps::rotaryRotateInverseInPlace position exceeds RoPE table");
    }
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaOps::rotaryRotateInverseInPlace no CUDA device");

    const int sequenceLength = static_cast<int>(tensor.cols);
    const int blockCount = (sequenceLength + CudaOps::threadCount - 1) / CudaOps::threadCount;
    CudaOpsRotaryRotateInverseEntry<<<blockCount, CudaOps::threadCount, 0, CudaMatmul::activeStream()>>>(tensor.buffer.deviceData, headCount, headDimension, pairCount, sequenceLength, positionOffset, segmentLength, cosTable.buffer.deviceData, sinTable.buffer.deviceData);
    CudaMatmul::throwIfCudaFailed(cudaGetLastError(), "CudaOpsRotaryRotateInverseEntry launch");
}

void CudaOps::embeddingGatherInto(const CudaMatrix& weight, const CudaIntBuffer& tokenIds, size_t tokenCount, CudaMatrix& out) {
    if (tokenIds.deviceData == nullptr) throw std::invalid_argument("CudaOps::embeddingGatherInto empty tokenIds");
    if (tokenCount > tokenIds.capacityCount) throw std::invalid_argument("CudaOps::embeddingGatherInto tokenCount exceeds capacity");
    CudaOps::embeddingGatherInto(weight, tokenIds.deviceData, tokenCount, out);
}

void CudaOps::embeddingGatherInto(const CudaMatrix& weight, const int* tokenIdsDevice, size_t tokenCount, CudaMatrix& out) {
    if (weight.empty()) throw std::invalid_argument("CudaOps::embeddingGatherInto empty weight");
    if (tokenCount == 0) throw std::invalid_argument("CudaOps::embeddingGatherInto empty tokenCount");
    if (tokenIdsDevice == nullptr) throw std::invalid_argument("CudaOps::embeddingGatherInto empty tokenIds");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaOps::embeddingGatherInto no CUDA device");

    const int vocabularySize = static_cast<int>(weight.rows);
    const int embeddingDim = static_cast<int>(weight.cols);
    out.ensureSize(static_cast<size_t>(embeddingDim), tokenCount);

    const int elementCount = embeddingDim * static_cast<int>(tokenCount);
    const int blockCount = (elementCount + CudaOps::threadCount - 1) / CudaOps::threadCount;
    if (const void* halfWeight = CudaAmp::fp16WorkingWeightOrNull(weight)) {
        CudaOpsEmbeddingGatherHalfEntry<<<blockCount, CudaOps::threadCount, 0, CudaMatmul::activeStream()>>>(
            reinterpret_cast<const __half*>(halfWeight), tokenIdsDevice, out.buffer.deviceData, embeddingDim, static_cast<int>(tokenCount), vocabularySize);
        CudaMatmul::throwIfCudaFailed(cudaGetLastError(), "CudaOpsEmbeddingGatherHalfEntry launch");
        return;
    }
    if (!weight.hasDeviceStorage())
        throw std::runtime_error("CudaOps::embeddingGatherInto weight missing FP32/FP16 storage");
    CudaOpsEmbeddingGatherEntry<<<blockCount, CudaOps::threadCount, 0, CudaMatmul::activeStream()>>>(weight.buffer.deviceData, tokenIdsDevice, out.buffer.deviceData, embeddingDim, static_cast<int>(tokenCount), vocabularySize);
    CudaMatmul::throwIfCudaFailed(cudaGetLastError(), "CudaOpsEmbeddingGatherEntry launch");
}

void CudaOps::embeddingScatterAddInto(CudaMatrix& weightGradient, const CudaIntBuffer& tokenIds, size_t tokenCount, const CudaMatrix& outputGradient) {
    if (tokenIds.deviceData == nullptr) throw std::invalid_argument("CudaOps::embeddingScatterAddInto empty tokenIds");
    if (tokenCount > tokenIds.capacityCount) throw std::invalid_argument("CudaOps::embeddingScatterAddInto tokenCount exceeds capacity");
    CudaOps::embeddingScatterAddInto(weightGradient, tokenIds.deviceData, tokenCount, outputGradient);
}

void CudaOps::embeddingScatterAddInto(CudaMatrix& weightGradient, const int* tokenIdsDevice, size_t tokenCount, const CudaMatrix& outputGradient) {
    if (weightGradient.empty()) throw std::invalid_argument("CudaOps::embeddingScatterAddInto empty weightGradient");
    if (outputGradient.empty()) throw std::invalid_argument("CudaOps::embeddingScatterAddInto empty outputGradient");
    if (tokenCount == 0) throw std::invalid_argument("CudaOps::embeddingScatterAddInto empty tokenCount");
    if (tokenIdsDevice == nullptr) throw std::invalid_argument("CudaOps::embeddingScatterAddInto empty tokenIds");
    if (outputGradient.cols != tokenCount) throw std::invalid_argument("CudaOps::embeddingScatterAddInto tokenCount mismatch");
    if (static_cast<size_t>(outputGradient.rows) != weightGradient.cols) throw std::invalid_argument("CudaOps::embeddingScatterAddInto embedding dim mismatch");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaOps::embeddingScatterAddInto no CUDA device");

    const int vocabularySize = static_cast<int>(weightGradient.rows);
    const int embeddingDim = static_cast<int>(weightGradient.cols);
    const int elementCount = embeddingDim * static_cast<int>(tokenCount);
    const int blockCount = (elementCount + CudaOps::threadCount - 1) / CudaOps::threadCount;
    CudaOpsEmbeddingScatterAddEntry<<<blockCount, CudaOps::threadCount, 0, CudaMatmul::activeStream()>>>(weightGradient.buffer.deviceData, tokenIdsDevice, outputGradient.buffer.deviceData, embeddingDim, static_cast<int>(tokenCount), vocabularySize);
    CudaMatmul::throwIfCudaFailed(cudaGetLastError(), "CudaOpsEmbeddingScatterAddEntry launch");
}

void CudaOps::embeddingZeroRows(CudaMatrix& weightGradient, const int* tokenIdsDevice, size_t tokenCount) {
    if (weightGradient.empty()) throw std::invalid_argument("CudaOps::embeddingZeroRows empty weightGradient");
    if (tokenCount == 0) return;
    if (tokenIdsDevice == nullptr) throw std::invalid_argument("CudaOps::embeddingZeroRows empty tokenIds");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaOps::embeddingZeroRows no CUDA device");

    const int vocabularySize = static_cast<int>(weightGradient.rows);
    const int embeddingDim = static_cast<int>(weightGradient.cols);
    const int elementCount = embeddingDim * static_cast<int>(tokenCount);
    const int blockCount = (elementCount + CudaOps::threadCount - 1) / CudaOps::threadCount;
    CudaOpsEmbeddingZeroRowsEntry<<<blockCount, CudaOps::threadCount, 0, CudaMatmul::activeStream()>>>(weightGradient.buffer.deviceData, tokenIdsDevice, embeddingDim, static_cast<int>(tokenCount), vocabularySize);
    CudaMatmul::throwIfCudaFailed(cudaGetLastError(), "CudaOpsEmbeddingZeroRowsEntry launch");
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
    CudaOpsCrossEntropyLossFromIdsEntry<<<blockCount, CudaOps::threadCount, 0, CudaMatmul::activeStream()>>>(probabilities.buffer.deviceData, targetTokenIds.deviceData, columnLossScratch.buffer.deviceData, vocabularySize, tokenCountInt);
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
    CudaOpsCrossEntropyAddMeanLossFromIdsEntry<<<blockCount, CudaOps::threadCount, 0, CudaMatmul::activeStream()>>>(probabilities.buffer.deviceData, targetTokenIds.deviceData, lossSum.buffer.deviceData, vocabularySize, tokenCountInt);
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
    CudaOpsCrossEntropyLogitGradientFromIdsEntry<<<blockCount, CudaOps::threadCount, 0, CudaMatmul::activeStream()>>>(probabilities.buffer.deviceData, targetTokenIds.deviceData, logitGradient.buffer.deviceData, vocabularySize, tokenCountInt);
    CudaMatmul::throwIfCudaFailed(cudaGetLastError(), "CudaOpsCrossEntropyLogitGradientFromIdsEntry launch");
}

void CudaOps::softmaxCrossEntropyFromLogitsInto(const CudaMatrix& logits, const CudaIntBuffer& targetTokenIds, size_t tokenCount, CudaMatrix& probabilities, CudaMatrix& logitGradient, CudaMatrix& lossSum, float lossScale, int meanDivisor) {
    if (logits.empty()) throw std::invalid_argument("CudaOps::softmaxCrossEntropyFromLogitsInto empty logits");
    if (tokenCount == 0) throw std::invalid_argument("CudaOps::softmaxCrossEntropyFromLogitsInto empty tokenCount");
    if (logits.cols != tokenCount) throw std::invalid_argument("CudaOps::softmaxCrossEntropyFromLogitsInto tokenCount mismatch");
    if (targetTokenIds.deviceData == nullptr) throw std::invalid_argument("CudaOps::softmaxCrossEntropyFromLogitsInto empty targetTokenIds");
    if (tokenCount > targetTokenIds.capacityCount) throw std::invalid_argument("CudaOps::softmaxCrossEntropyFromLogitsInto tokenCount exceeds capacity");
    if (lossSum.rows != 1 || lossSum.cols != 1) throw std::invalid_argument("CudaOps::softmaxCrossEntropyFromLogitsInto lossSum must be 1x1");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaOps::softmaxCrossEntropyFromLogitsInto no CUDA device");

    if (meanDivisor <= 0) meanDivisor = static_cast<int>(tokenCount);

    probabilities.ensureSize(logits.rows, logits.cols);
    logitGradient.ensureSize(logits.rows, logits.cols);

    const int vocabularySize = static_cast<int>(logits.rows);
    const int tokenCountInt = static_cast<int>(tokenCount);
    const int blockCount = (tokenCountInt + CudaOps::threadCount - 1) / CudaOps::threadCount;
    CudaOpsSoftmaxCrossEntropyFromLogitsEntry<<<blockCount, CudaOps::threadCount, 0, CudaMatmul::activeStream()>>>(logits.buffer.deviceData, targetTokenIds.deviceData, probabilities.buffer.deviceData, logitGradient.buffer.deviceData, lossSum.buffer.deviceData, vocabularySize, tokenCountInt, lossScale, meanDivisor);
    CudaMatmul::throwIfCudaFailed(cudaGetLastError(), "CudaOpsSoftmaxCrossEntropyFromLogitsEntry launch");
}

__global__ void CudaOpsFillEntry(float* matrix, float value, int elementCount) {
    const int index = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (index >= elementCount) return;
    matrix[index] = value;
}

__global__ void CudaOpsBroadcastBiasRowsAddEntry(float* product, const float* bias, int rowStart, int rowCount, int columnCount) {
    const int elementCount = rowCount * columnCount;
    const int index = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (index >= elementCount) return;
    const int row = index / columnCount;
    product[index] += bias[rowStart + row];
}

__global__ void CudaOpsAddRowsInPlaceEntry(float* destination, const float* source, int rowStart, int rowCount, int columnCount) {
    const int elementCount = rowCount * columnCount;
    const int index = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (index >= elementCount) return;
    const int row = index / columnCount;
    const int column = index - row * columnCount;
    destination[(rowStart + row) * columnCount + column] += source[index];
}

__global__ void CudaOpsOnlineSoftmaxUpdateEntry(const float* logitChunk, int chunkRows, int tokenCount, float* maximumLogits, float* sumExp) {
    const int column = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (column >= tokenCount) return;

    float chunkMax = logitChunk[column];
    for (int row = 1; row < chunkRows; ++row) {
        const float value = logitChunk[row * tokenCount + column];
        if (value > chunkMax)
            chunkMax = value;
    }
    if (!isfinite(chunkMax))
        chunkMax = 0.0f;

    float chunkSum = 0.0f;
    for (int row = 0; row < chunkRows; ++row)
        chunkSum += expf(logitChunk[row * tokenCount + column] - chunkMax);

    const float oldMax = maximumLogits[column];
    const float oldSum = sumExp[column];
    float newMax = chunkMax;
    if (isfinite(oldMax) && oldMax > newMax)
        newMax = oldMax;

    float combined = chunkSum;
    if (isfinite(oldMax) && oldSum > 0.0f)
        combined = oldSum * expf(oldMax - newMax) + chunkSum * expf(chunkMax - newMax);
    else if (chunkMax != newMax)
        combined = chunkSum * expf(chunkMax - newMax);

    if (!isfinite(combined) || combined < 0.0f)
        combined = 0.0f;
    sumExp[column] = combined;
    maximumLogits[column] = newMax;
}

__global__ void CudaOpsCaptureTargetLogitEntry(const float* logitChunk, const int* targetTokenIds, int rowStart, int chunkRows, int tokenCount, float* targetLogits) {
    const int column = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (column >= tokenCount) return;
    const int targetId = targetTokenIds[column];
    if (targetId < 0) return;
    if (targetId < rowStart) return;
    if (targetId >= rowStart + chunkRows) return;
    targetLogits[column] = logitChunk[(targetId - rowStart) * tokenCount + column];
}

__global__ void CudaOpsOnlineSoftmaxBiasedUpdateCaptureEntry(const float* logitChunk, const float* bias, int rowStart, int chunkRows, const int* targetTokenIds, int tokenCount, float* maximumLogits, float* sumExp, float* targetLogits) {
    const int column = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (column >= tokenCount) return;

    float chunkMax = logitChunk[column] + bias[rowStart];
    for (int row = 1; row < chunkRows; ++row) {
        const float value = logitChunk[row * tokenCount + column] + bias[rowStart + row];
        if (value > chunkMax)
            chunkMax = value;
    }
    if (!isfinite(chunkMax))
        chunkMax = 0.0f;

    float chunkSum = 0.0f;
    for (int row = 0; row < chunkRows; ++row)
        chunkSum += expf(logitChunk[row * tokenCount + column] + bias[rowStart + row] - chunkMax);

    const float oldMax = maximumLogits[column];
    const float oldSum = sumExp[column];
    float newMax = chunkMax;
    if (isfinite(oldMax) && oldMax > newMax)
        newMax = oldMax;

    float combined = chunkSum;
    if (isfinite(oldMax) && oldSum > 0.0f)
        combined = oldSum * expf(oldMax - newMax) + chunkSum * expf(chunkMax - newMax);
    else if (chunkMax != newMax)
        combined = chunkSum * expf(chunkMax - newMax);

    if (!isfinite(combined) || combined < 0.0f)
        combined = 0.0f;
    sumExp[column] = combined;
    maximumLogits[column] = newMax;

    const int targetId = targetTokenIds[column];
    if (targetId >= rowStart && targetId < rowStart + chunkRows)
        targetLogits[column] = logitChunk[(targetId - rowStart) * tokenCount + column] + bias[targetId];
}

__global__ void CudaOpsOnlineSoftmaxAddCeEntry(const float* targetLogits, const float* maximumLogits, const float* sumExp, float* lossSum, int tokenCount, float lossScale, int meanDivisor, const int* targetTokenIds, int ignoreIndex, const int* perColumnMeanDivisor) {
    const int column = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (column >= tokenCount) return;
    if (targetTokenIds != nullptr && targetTokenIds[column] == ignoreIndex) return;
    if (!isfinite(targetLogits[column]) || !isfinite(maximumLogits[column]) || !isfinite(sumExp[column])) return;
    int divisor = meanDivisor;
    if (perColumnMeanDivisor != nullptr) divisor = perColumnMeanDivisor[column];
    if (divisor <= 0) return;
    const float inverseMeanDivisor = 1.0f / static_cast<float>(divisor);
    float safeSum = sumExp[column];
    if (safeSum < 1e-20f) safeSum = 1e-20f;
    const float loss = (-targetLogits[column] + maximumLogits[column] + logf(safeSum)) * inverseMeanDivisor * lossScale;
    if (!isfinite(loss)) return;
    atomicAdd(lossSum, loss);
}

__global__ void CudaOpsOnlineSoftmaxLogitGradChunkEntry(const float* logitChunk, const int* targetTokenIds, int rowStart, int chunkRows, int tokenCount, const float* maximumLogits, const float* sumExp, float* logitGradientChunk, float gradScale, int meanDivisor, int ignoreIndex, const int* perColumnMeanDivisor) {
    const int elementCount = chunkRows * tokenCount;
    const int index = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (index >= elementCount) return;

    const int row = index / tokenCount;
    const int column = index - row * tokenCount;
    if (targetTokenIds[column] == ignoreIndex) {
        logitGradientChunk[index] = 0.0f;
        return;
    }

    float safeSum = sumExp[column];
    if (safeSum < 1e-20f) safeSum = 1e-20f;
    const float probability = expf(logitChunk[index] - maximumLogits[column]) / safeSum;
    int divisor = meanDivisor;
    if (perColumnMeanDivisor != nullptr) divisor = perColumnMeanDivisor[column];
    if (divisor <= 0) {
        logitGradientChunk[index] = 0.0f;
        return;
    }
    const float inverseMeanDivisor = 1.0f / static_cast<float>(divisor);
    float gradient = probability * inverseMeanDivisor;
    if (rowStart + row == targetTokenIds[column])
        gradient -= inverseMeanDivisor;
    logitGradientChunk[index] = gradient * gradScale;
}

__global__ void CudaOpsOnlineSoftmaxBiasedLogitGradChunkEntry(const float* logitChunk, const float* bias, const int* targetTokenIds, int rowStart, int chunkRows, int tokenCount, const float* maximumLogits, const float* sumExp, float* logitGradientChunk, float gradScale, int meanDivisor, int ignoreIndex, const int* perColumnMeanDivisor) {
    const int elementCount = chunkRows * tokenCount;
    const int index = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (index >= elementCount) return;

    const int row = index / tokenCount;
    const int column = index - row * tokenCount;
    if (targetTokenIds[column] == ignoreIndex) {
        logitGradientChunk[index] = 0.0f;
        return;
    }

    float safeSum = sumExp[column];
    if (safeSum < 1e-20f) safeSum = 1e-20f;
    const float biasedLogit = logitChunk[index] + bias[rowStart + row];
    const float probability = expf(biasedLogit - maximumLogits[column]) / safeSum;
    int divisor = meanDivisor;
    if (perColumnMeanDivisor != nullptr) divisor = perColumnMeanDivisor[column];
    if (divisor <= 0) {
        logitGradientChunk[index] = 0.0f;
        return;
    }
    const float inverseMeanDivisor = 1.0f / static_cast<float>(divisor);
    float gradient = probability * inverseMeanDivisor;
    if (rowStart + row == targetTokenIds[column])
        gradient -= inverseMeanDivisor;
    logitGradientChunk[index] = gradient * gradScale;
}

__global__ void CudaOpsSumColumnsAddIntoRowsEntry(const float* gradient, float* biasGradient, int rowStart, int rowCount, int columnCount) {
    const int row = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (row >= rowCount) return;
    float sum = 0.0f;
    for (int column = 0; column < columnCount; ++column)
        sum += gradient[row * columnCount + column];
    biasGradient[rowStart + row] += sum;
}

void CudaOps::fillInPlace(CudaMatrix& matrix, float value) {
    if (matrix.empty()) throw std::invalid_argument("CudaOps::fillInPlace empty matrix");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaOps::fillInPlace no CUDA device");
    const int elementCount = static_cast<int>(matrix.elementCount());
    const int blockCount = (elementCount + CudaOps::threadCount - 1) / CudaOps::threadCount;
    CudaOpsFillEntry<<<blockCount, CudaOps::threadCount, 0, CudaMatmul::activeStream()>>>(matrix.buffer.deviceData, value, elementCount);
    CudaMatmul::throwIfCudaFailed(cudaGetLastError(), "CudaOpsFillEntry launch");
}

void CudaOps::broadcastBiasRowsAddInPlace(CudaMatrix& product, const CudaMatrix& bias, int rowStart, int rowCount) {
    if (product.empty()) throw std::invalid_argument("CudaOps::broadcastBiasRowsAddInPlace empty product");
    if (bias.empty()) throw std::invalid_argument("CudaOps::broadcastBiasRowsAddInPlace empty bias");
    if (rowStart < 0 || rowCount <= 0) throw std::invalid_argument("CudaOps::broadcastBiasRowsAddInPlace invalid rows");
    if (rowStart + rowCount > static_cast<int>(bias.rows)) throw std::invalid_argument("CudaOps::broadcastBiasRowsAddInPlace bias range");
    if (static_cast<int>(product.rows) != rowCount) throw std::invalid_argument("CudaOps::broadcastBiasRowsAddInPlace rowCount mismatch");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaOps::broadcastBiasRowsAddInPlace no CUDA device");

    const int columnCount = static_cast<int>(product.cols);
    const int elementCount = rowCount * columnCount;
    const int blockCount = (elementCount + CudaOps::threadCount - 1) / CudaOps::threadCount;
    CudaOpsBroadcastBiasRowsAddEntry<<<blockCount, CudaOps::threadCount, 0, CudaMatmul::activeStream()>>>(product.buffer.deviceData, bias.buffer.deviceData, rowStart, rowCount, columnCount);
    CudaMatmul::throwIfCudaFailed(cudaGetLastError(), "CudaOpsBroadcastBiasRowsAddEntry launch");
}

void CudaOps::addRowsInPlace(CudaMatrix& destination, int rowStart, const CudaMatrix& source) {
    if (source.empty()) throw std::invalid_argument("CudaOps::addRowsInPlace empty source");
    if (rowStart < 0) throw std::invalid_argument("CudaOps::addRowsInPlace negative rowStart");
    if (rowStart + static_cast<int>(source.rows) > static_cast<int>(destination.rows)) throw std::invalid_argument("CudaOps::addRowsInPlace destination range");
    if (destination.cols != source.cols) throw std::invalid_argument("CudaOps::addRowsInPlace cols mismatch");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaOps::addRowsInPlace no CUDA device");

    const int rowCount = static_cast<int>(source.rows);
    const int columnCount = static_cast<int>(source.cols);
    const int elementCount = rowCount * columnCount;
    const int blockCount = (elementCount + CudaOps::threadCount - 1) / CudaOps::threadCount;
    CudaOpsAddRowsInPlaceEntry<<<blockCount, CudaOps::threadCount, 0, CudaMatmul::activeStream()>>>(destination.buffer.deviceData, source.buffer.deviceData, rowStart, rowCount, columnCount);
    CudaMatmul::throwIfCudaFailed(cudaGetLastError(), "CudaOpsAddRowsInPlaceEntry launch");
}

void CudaOps::onlineSoftmaxReset(CudaMatrix& maximumLogits, CudaMatrix& sumExp, size_t tokenCount) {
    if (tokenCount == 0) throw std::invalid_argument("CudaOps::onlineSoftmaxReset empty tokenCount");
    maximumLogits.ensureSize(1, tokenCount);
    sumExp.ensureSize(1, tokenCount);
    CudaOps::fillInPlace(maximumLogits, -std::numeric_limits<float>::infinity());
    CudaOps::zeroInPlace(sumExp);
}

void CudaOps::onlineSoftmaxUpdateFromChunk(const CudaMatrix& logitChunk, int chunkRows, size_t tokenCount, CudaMatrix& maximumLogits, CudaMatrix& sumExp) {
    if (logitChunk.empty()) throw std::invalid_argument("CudaOps::onlineSoftmaxUpdateFromChunk empty logitChunk");
    if (chunkRows <= 0 || static_cast<int>(logitChunk.rows) < chunkRows) throw std::invalid_argument("CudaOps::onlineSoftmaxUpdateFromChunk invalid chunkRows");
    if (logitChunk.cols != tokenCount) throw std::invalid_argument("CudaOps::onlineSoftmaxUpdateFromChunk tokenCount mismatch");
    CudaOps::onlineSoftmaxUpdateFromChunk(logitChunk.buffer.deviceData, chunkRows, tokenCount, maximumLogits, sumExp);
}

void CudaOps::onlineSoftmaxUpdateFromChunk(const float* logitChunk, int chunkRows, size_t tokenCount, CudaMatrix& maximumLogits, CudaMatrix& sumExp) {
    if (logitChunk == nullptr) throw std::invalid_argument("CudaOps::onlineSoftmaxUpdateFromChunk null logitChunk");
    if (chunkRows <= 0) throw std::invalid_argument("CudaOps::onlineSoftmaxUpdateFromChunk invalid chunkRows");
    if (maximumLogits.cols != tokenCount || sumExp.cols != tokenCount) throw std::invalid_argument("CudaOps::onlineSoftmaxUpdateFromChunk stats shape");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaOps::onlineSoftmaxUpdateFromChunk no CUDA device");

    const int tokenCountInt = static_cast<int>(tokenCount);
    const int blockCount = (tokenCountInt + CudaOps::threadCount - 1) / CudaOps::threadCount;
    CudaOpsOnlineSoftmaxUpdateEntry<<<blockCount, CudaOps::threadCount, 0, CudaMatmul::activeStream()>>>(logitChunk, chunkRows, tokenCountInt, maximumLogits.buffer.deviceData, sumExp.buffer.deviceData);
    CudaMatmul::throwIfCudaFailed(cudaGetLastError(), "CudaOpsOnlineSoftmaxUpdateEntry launch");
}

void CudaOps::onlineSoftmaxBiasedUpdateAndCaptureTargetFromChunk(const float* logitChunk, const CudaMatrix& bias, int rowStart, int chunkRows, const CudaIntBuffer& targetTokenIds, size_t tokenCount, CudaMatrix& maximumLogits, CudaMatrix& sumExp, CudaMatrix& targetLogits) {
    if (logitChunk == nullptr) throw std::invalid_argument("CudaOps::onlineSoftmaxBiasedUpdateAndCaptureTargetFromChunk null logitChunk");
    if (bias.empty()) throw std::invalid_argument("CudaOps::onlineSoftmaxBiasedUpdateAndCaptureTargetFromChunk empty bias");
    if (targetTokenIds.deviceData == nullptr) throw std::invalid_argument("CudaOps::onlineSoftmaxBiasedUpdateAndCaptureTargetFromChunk empty targets");
    if (chunkRows <= 0) throw std::invalid_argument("CudaOps::onlineSoftmaxBiasedUpdateAndCaptureTargetFromChunk invalid chunkRows");
    if (rowStart < 0 || rowStart + chunkRows > static_cast<int>(bias.rows)) throw std::invalid_argument("CudaOps::onlineSoftmaxBiasedUpdateAndCaptureTargetFromChunk bias range");
    if (maximumLogits.cols != tokenCount || sumExp.cols != tokenCount) throw std::invalid_argument("CudaOps::onlineSoftmaxBiasedUpdateAndCaptureTargetFromChunk stats shape");
    if (tokenCount > targetTokenIds.capacityCount) throw std::invalid_argument("CudaOps::onlineSoftmaxBiasedUpdateAndCaptureTargetFromChunk target capacity");
    targetLogits.ensureSize(1, tokenCount);
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaOps::onlineSoftmaxBiasedUpdateAndCaptureTargetFromChunk no CUDA device");

    const int tokenCountInt = static_cast<int>(tokenCount);
    const int blockCount = (tokenCountInt + CudaOps::threadCount - 1) / CudaOps::threadCount;
    CudaOpsOnlineSoftmaxBiasedUpdateCaptureEntry<<<blockCount, CudaOps::threadCount, 0, CudaMatmul::activeStream()>>>(
        logitChunk, bias.buffer.deviceData, rowStart, chunkRows, targetTokenIds.deviceData, tokenCountInt,
        maximumLogits.buffer.deviceData, sumExp.buffer.deviceData, targetLogits.buffer.deviceData);
    CudaMatmul::throwIfCudaFailed(cudaGetLastError(), "CudaOpsOnlineSoftmaxBiasedUpdateCaptureEntry launch");
}

void CudaOps::captureTargetLogitFromChunk(const CudaMatrix& logitChunk, const CudaIntBuffer& targetTokenIds, int rowStart, int chunkRows, size_t tokenCount, CudaMatrix& targetLogits) {
    if (logitChunk.empty()) throw std::invalid_argument("CudaOps::captureTargetLogitFromChunk empty logitChunk");
    CudaOps::captureTargetLogitFromChunk(logitChunk.buffer.deviceData, targetTokenIds, rowStart, chunkRows, tokenCount, targetLogits);
}

void CudaOps::captureTargetLogitFromChunk(const float* logitChunk, const CudaIntBuffer& targetTokenIds, int rowStart, int chunkRows, size_t tokenCount, CudaMatrix& targetLogits) {
    if (logitChunk == nullptr) throw std::invalid_argument("CudaOps::captureTargetLogitFromChunk null logitChunk");
    if (targetTokenIds.deviceData == nullptr) throw std::invalid_argument("CudaOps::captureTargetLogitFromChunk empty targets");
    if (tokenCount > targetTokenIds.capacityCount) throw std::invalid_argument("CudaOps::captureTargetLogitFromChunk capacity");
    targetLogits.ensureSize(1, tokenCount);
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaOps::captureTargetLogitFromChunk no CUDA device");

    const int tokenCountInt = static_cast<int>(tokenCount);
    const int blockCount = (tokenCountInt + CudaOps::threadCount - 1) / CudaOps::threadCount;
    CudaOpsCaptureTargetLogitEntry<<<blockCount, CudaOps::threadCount, 0, CudaMatmul::activeStream()>>>(logitChunk, targetTokenIds.deviceData, rowStart, chunkRows, tokenCountInt, targetLogits.buffer.deviceData);
    CudaMatmul::throwIfCudaFailed(cudaGetLastError(), "CudaOpsCaptureTargetLogitEntry launch");
}

void CudaOps::onlineSoftmaxAddMeanCrossEntropy(const CudaMatrix& targetLogits, const CudaMatrix& maximumLogits, const CudaMatrix& sumExp, size_t tokenCount, CudaMatrix& lossSum, float lossScale, int meanDivisor, const CudaIntBuffer* targetTokenIds, int ignoreIndex, const CudaIntBuffer* perColumnMeanDivisor) {
    if (tokenCount == 0) throw std::invalid_argument("CudaOps::onlineSoftmaxAddMeanCrossEntropy empty tokenCount");
    if (lossSum.rows != 1 || lossSum.cols != 1) throw std::invalid_argument("CudaOps::onlineSoftmaxAddMeanCrossEntropy lossSum must be 1x1");
    if (meanDivisor <= 0) meanDivisor = static_cast<int>(tokenCount);
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaOps::onlineSoftmaxAddMeanCrossEntropy no CUDA device");
    if (targetTokenIds != nullptr && tokenCount > targetTokenIds->capacityCount)
        throw std::invalid_argument("CudaOps::onlineSoftmaxAddMeanCrossEntropy target capacity");
    if (perColumnMeanDivisor != nullptr && tokenCount > perColumnMeanDivisor->capacityCount)
        throw std::invalid_argument("CudaOps::onlineSoftmaxAddMeanCrossEntropy meanDivisor capacity");

    const int tokenCountInt = static_cast<int>(tokenCount);
    const int blockCount = (tokenCountInt + CudaOps::threadCount - 1) / CudaOps::threadCount;
    CudaOpsOnlineSoftmaxAddCeEntry<<<blockCount, CudaOps::threadCount, 0, CudaMatmul::activeStream()>>>(
        targetLogits.buffer.deviceData,
        maximumLogits.buffer.deviceData,
        sumExp.buffer.deviceData,
        lossSum.buffer.deviceData,
        tokenCountInt,
        lossScale,
        meanDivisor,
        targetTokenIds != nullptr ? targetTokenIds->deviceData : nullptr,
        ignoreIndex,
        perColumnMeanDivisor != nullptr ? perColumnMeanDivisor->deviceData : nullptr
    );
    CudaMatmul::throwIfCudaFailed(cudaGetLastError(), "CudaOpsOnlineSoftmaxAddCeEntry launch");
}

void CudaOps::onlineSoftmaxLogitGradientChunkInto(const CudaMatrix& logitChunk, const CudaIntBuffer& targetTokenIds, int rowStart, int chunkRows, size_t tokenCount, const CudaMatrix& maximumLogits, const CudaMatrix& sumExp, CudaMatrix& logitGradientChunk, float gradScale, int meanDivisor, int ignoreIndex, const CudaIntBuffer* perColumnMeanDivisor) {
    if (logitChunk.empty()) throw std::invalid_argument("CudaOps::onlineSoftmaxLogitGradientChunkInto empty logitChunk");
    CudaOps::onlineSoftmaxLogitGradientChunkInto(
        logitChunk.buffer.deviceData, targetTokenIds, rowStart, chunkRows, tokenCount,
        maximumLogits, sumExp, logitGradientChunk, gradScale, meanDivisor, ignoreIndex, perColumnMeanDivisor);
}

void CudaOps::onlineSoftmaxLogitGradientChunkInto(const float* logitChunk, const CudaIntBuffer& targetTokenIds, int rowStart, int chunkRows, size_t tokenCount, const CudaMatrix& maximumLogits, const CudaMatrix& sumExp, CudaMatrix& logitGradientChunk, float gradScale, int meanDivisor, int ignoreIndex, const CudaIntBuffer* perColumnMeanDivisor) {
    if (logitChunk == nullptr) throw std::invalid_argument("CudaOps::onlineSoftmaxLogitGradientChunkInto null logitChunk");
    if (chunkRows <= 0) throw std::invalid_argument("CudaOps::onlineSoftmaxLogitGradientChunkInto invalid chunkRows");
    if (meanDivisor <= 0) meanDivisor = static_cast<int>(tokenCount);
    if (tokenCount > targetTokenIds.capacityCount) throw std::invalid_argument("CudaOps::onlineSoftmaxLogitGradientChunkInto target capacity");
    if (perColumnMeanDivisor != nullptr && tokenCount > perColumnMeanDivisor->capacityCount)
        throw std::invalid_argument("CudaOps::onlineSoftmaxLogitGradientChunkInto meanDivisor capacity");
    logitGradientChunk.ensureSize(static_cast<size_t>(chunkRows), tokenCount);
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaOps::onlineSoftmaxLogitGradientChunkInto no CUDA device");

    const int tokenCountInt = static_cast<int>(tokenCount);
    const int elementCount = chunkRows * tokenCountInt;
    const int blockCount = (elementCount + CudaOps::threadCount - 1) / CudaOps::threadCount;
    CudaOpsOnlineSoftmaxLogitGradChunkEntry<<<blockCount, CudaOps::threadCount, 0, CudaMatmul::activeStream()>>>(
        logitChunk,
        targetTokenIds.deviceData,
        rowStart,
        chunkRows,
        tokenCountInt,
        maximumLogits.buffer.deviceData,
        sumExp.buffer.deviceData,
        logitGradientChunk.buffer.deviceData,
        gradScale,
        meanDivisor,
        ignoreIndex,
        perColumnMeanDivisor != nullptr ? perColumnMeanDivisor->deviceData : nullptr
    );
    CudaMatmul::throwIfCudaFailed(cudaGetLastError(), "CudaOpsOnlineSoftmaxLogitGradChunkEntry launch");
}

void CudaOps::onlineSoftmaxBiasedLogitGradientChunkInto(const float* logitChunk, const CudaMatrix& bias, const CudaIntBuffer& targetTokenIds, int rowStart, int chunkRows, size_t tokenCount, const CudaMatrix& maximumLogits, const CudaMatrix& sumExp, CudaMatrix& logitGradientChunk, float gradScale, int meanDivisor, int ignoreIndex, const CudaIntBuffer* perColumnMeanDivisor) {
    if (logitChunk == nullptr) throw std::invalid_argument("CudaOps::onlineSoftmaxBiasedLogitGradientChunkInto null logitChunk");
    if (bias.empty()) throw std::invalid_argument("CudaOps::onlineSoftmaxBiasedLogitGradientChunkInto empty bias");
    if (chunkRows <= 0) throw std::invalid_argument("CudaOps::onlineSoftmaxBiasedLogitGradientChunkInto invalid chunkRows");
    if (meanDivisor <= 0) meanDivisor = static_cast<int>(tokenCount);
    if (rowStart < 0 || rowStart + chunkRows > static_cast<int>(bias.rows)) throw std::invalid_argument("CudaOps::onlineSoftmaxBiasedLogitGradientChunkInto bias range");
    if (tokenCount > targetTokenIds.capacityCount) throw std::invalid_argument("CudaOps::onlineSoftmaxBiasedLogitGradientChunkInto target capacity");
    if (perColumnMeanDivisor != nullptr && tokenCount > perColumnMeanDivisor->capacityCount)
        throw std::invalid_argument("CudaOps::onlineSoftmaxBiasedLogitGradientChunkInto meanDivisor capacity");
    logitGradientChunk.ensureSize(static_cast<size_t>(chunkRows), tokenCount);
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaOps::onlineSoftmaxBiasedLogitGradientChunkInto no CUDA device");

    const int tokenCountInt = static_cast<int>(tokenCount);
    const int elementCount = chunkRows * tokenCountInt;
    const int blockCount = (elementCount + CudaOps::threadCount - 1) / CudaOps::threadCount;
    CudaOpsOnlineSoftmaxBiasedLogitGradChunkEntry<<<blockCount, CudaOps::threadCount, 0, CudaMatmul::activeStream()>>>(
        logitChunk,
        bias.buffer.deviceData,
        targetTokenIds.deviceData,
        rowStart,
        chunkRows,
        tokenCountInt,
        maximumLogits.buffer.deviceData,
        sumExp.buffer.deviceData,
        logitGradientChunk.buffer.deviceData,
        gradScale,
        meanDivisor,
        ignoreIndex,
        perColumnMeanDivisor != nullptr ? perColumnMeanDivisor->deviceData : nullptr
    );
    CudaMatmul::throwIfCudaFailed(cudaGetLastError(), "CudaOpsOnlineSoftmaxBiasedLogitGradChunkEntry launch");
}

void CudaOps::sumColumnsAddIntoRows(const CudaMatrix& gradient, CudaMatrix& biasGradient, int rowStart) {
    if (gradient.empty()) throw std::invalid_argument("CudaOps::sumColumnsAddIntoRows empty gradient");
    if (rowStart < 0) throw std::invalid_argument("CudaOps::sumColumnsAddIntoRows negative rowStart");
    if (rowStart + static_cast<int>(gradient.rows) > static_cast<int>(biasGradient.rows)) throw std::invalid_argument("CudaOps::sumColumnsAddIntoRows bias range");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaOps::sumColumnsAddIntoRows no CUDA device");

    const int rowCount = static_cast<int>(gradient.rows);
    const int columnCount = static_cast<int>(gradient.cols);
    const int blockCount = (rowCount + CudaOps::threadCount - 1) / CudaOps::threadCount;
    CudaOpsSumColumnsAddIntoRowsEntry<<<blockCount, CudaOps::threadCount, 0, CudaMatmul::activeStream()>>>(gradient.buffer.deviceData, biasGradient.buffer.deviceData, rowStart, rowCount, columnCount);
    CudaMatmul::throwIfCudaFailed(cudaGetLastError(), "CudaOpsSumColumnsAddIntoRowsEntry launch");
}
