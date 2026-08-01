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

__device__ void CudaOps::runMultiplyElementwiseInto(const float* left, const float* right, float* out, int elementCount) {
    const int index = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (index >= elementCount) return;

    out[index] = left[index] * right[index];
}

__device__ void CudaOps::runAddInto(const float* left, const float* right, float* out, int elementCount) {
    const int index = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (index >= elementCount) return;

    out[index] = left[index] + right[index];
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

__global__ void CudaOpsBroadcastBiasAddEntry(float* product, const float* bias, int rowCount, int columnCount) {
    CudaOps::runBroadcastBiasAddInPlace(product, bias, rowCount, columnCount);
}

__global__ void CudaOpsSiluEntry(const float* input, float* out, int elementCount) {
    CudaOps::runSiluInto(input, out, elementCount);
}

__global__ void CudaOpsMultiplyElementwiseEntry(const float* left, const float* right, float* out, int elementCount) {
    CudaOps::runMultiplyElementwiseInto(left, right, out, elementCount);
}

__global__ void CudaOpsAddEntry(const float* left, const float* right, float* out, int elementCount) {
    CudaOps::runAddInto(left, right, out, elementCount);
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

__global__ void CudaOpsRotaryRotateEntry(float* tensor, int headCount, int headDimension, int pairCount, int sequenceLength, int positionOffset, const float* cosTable, const float* sinTable) {
    CudaOps::runRotaryRotateInPlace(tensor, headCount, headDimension, pairCount, sequenceLength, positionOffset, cosTable, sinTable);
}

__global__ void CudaOpsEmbeddingGatherEntry(const float* weight, const int* tokenIds, float* out, int embeddingDim, int tokenCount, int vocabularySize) {
    CudaOps::runEmbeddingGatherInto(weight, tokenIds, out, embeddingDim, tokenCount, vocabularySize);
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
