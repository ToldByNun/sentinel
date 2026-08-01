#include "CudaFlashAttention.hpp"

#include "CudaOps.hpp"

#include <cuda_runtime.h>
#include <cmath>
#include <stdexcept>
#include <string>

namespace {

constexpr int kBr = CudaFlashAttention::queryTileSize;
constexpr int kBc = CudaFlashAttention::keyTileSize;
constexpr int kMaxHeadDim = CudaFlashAttention::maxHeadDimension;

__global__ void CudaFlashAttentionForwardEntry(const float* query, const float* key, const float* value, float* out, float* logSumExp, int headDim, int sequenceLength, float scale, int causal) {
    const int queryTile = static_cast<int>(blockIdx.x);
    const int queryStart = queryTile * kBr;
    if (queryStart >= sequenceLength) return;

    const int queryCount = sequenceLength - queryStart < kBr ? sequenceLength - queryStart : kBr;
    const int threadIndex = static_cast<int>(threadIdx.x);

    __shared__ float sharedQuery[kBr * kMaxHeadDim];
    __shared__ float sharedKey[kBc * kMaxHeadDim];
    __shared__ float sharedValue[kBc * kMaxHeadDim];
    __shared__ float sharedScores[kBr * kBc];

    for (int index = threadIndex; index < queryCount * headDim; index += static_cast<int>(blockDim.x)) {
        const int localQuery = index / headDim;
        const int dim = index - localQuery * headDim;
        const int globalQuery = queryStart + localQuery;
        sharedQuery[localQuery * kMaxHeadDim + dim] = query[dim * sequenceLength + globalQuery];
    }
    __syncthreads();

    float rowMax = -INFINITY;
    float rowSum = 0.0f;
    float outputAccum[kMaxHeadDim];
    for (int dim = 0; dim < kMaxHeadDim; ++dim)
        outputAccum[dim] = 0.0f;

    const int keyTileCount = (sequenceLength + kBc - 1) / kBc;
    for (int keyTile = 0; keyTile < keyTileCount; ++keyTile) {
        const int keyStart = keyTile * kBc;
        const int keyCount = sequenceLength - keyStart < kBc ? sequenceLength - keyStart : kBc;

        for (int index = threadIndex; index < keyCount * headDim; index += static_cast<int>(blockDim.x)) {
            const int localKey = index / headDim;
            const int dim = index - localKey * headDim;
            const int globalKey = keyStart + localKey;
            sharedKey[localKey * kMaxHeadDim + dim] = key[dim * sequenceLength + globalKey];
            sharedValue[localKey * kMaxHeadDim + dim] = value[dim * sequenceLength + globalKey];
        }
        __syncthreads();

        if (threadIndex < queryCount) {
            const int globalQuery = queryStart + threadIndex;
            float tileRowMax = -INFINITY;

            for (int localKey = 0; localKey < keyCount; ++localKey) {
                const int globalKey = keyStart + localKey;
                float score = -INFINITY;
                if (!causal || globalKey <= globalQuery) {
                    float dot = 0.0f;
                    for (int dim = 0; dim < headDim; ++dim)
                        dot += sharedQuery[threadIndex * kMaxHeadDim + dim] * sharedKey[localKey * kMaxHeadDim + dim];
                    score = dot * scale;
                }
                sharedScores[threadIndex * kBc + localKey] = score;
                if (score > tileRowMax)
                    tileRowMax = score;
            }

            const float newRowMax = rowMax > tileRowMax ? rowMax : tileRowMax;
            const float rescale = expf(rowMax - newRowMax);
            float tileSum = 0.0f;

            for (int localKey = 0; localKey < keyCount; ++localKey) {
                const float probability = expf(sharedScores[threadIndex * kBc + localKey] - newRowMax);
                sharedScores[threadIndex * kBc + localKey] = probability;
                tileSum += probability;
            }

            for (int dim = 0; dim < headDim; ++dim) {
                float weighted = 0.0f;
                for (int localKey = 0; localKey < keyCount; ++localKey)
                    weighted += sharedScores[threadIndex * kBc + localKey] * sharedValue[localKey * kMaxHeadDim + dim];
                outputAccum[dim] = outputAccum[dim] * rescale + weighted;
            }

            rowSum = rowSum * rescale + tileSum;
            rowMax = newRowMax;
        }
        __syncthreads();
    }

    if (threadIndex < queryCount) {
        const int globalQuery = queryStart + threadIndex;
        const float inverseSum = 1.0f / rowSum;
        for (int dim = 0; dim < headDim; ++dim)
            out[dim * sequenceLength + globalQuery] = outputAccum[dim] * inverseSum;
        logSumExp[globalQuery] = rowMax + logf(rowSum);
    }
}

__global__ void CudaFlashAttentionDeltaEntry(const float* out, const float* outGradient, float* delta, int headDim, int sequenceLength) {
    const int column = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (column >= sequenceLength) return;

    float sum = 0.0f;
    for (int dim = 0; dim < headDim; ++dim)
        sum += out[dim * sequenceLength + column] * outGradient[dim * sequenceLength + column];
    delta[column] = sum;
}

__global__ void CudaFlashAttentionBackwardEntry(const float* query, const float* key, const float* value, const float* outGradient, const float* logSumExp, const float* delta, float* queryGradient, float* keyGradient, float* valueGradient, int headDim, int sequenceLength, float scale, int causal) {
    const int queryTile = static_cast<int>(blockIdx.x);
    const int queryStart = queryTile * kBr;
    if (queryStart >= sequenceLength) return;

    const int queryCount = sequenceLength - queryStart < kBr ? sequenceLength - queryStart : kBr;
    const int threadIndex = static_cast<int>(threadIdx.x);

    __shared__ float sharedQuery[kBr * kMaxHeadDim];
    __shared__ float sharedOutGrad[kBr * kMaxHeadDim];
    __shared__ float sharedKey[kBc * kMaxHeadDim];
    __shared__ float sharedValue[kBc * kMaxHeadDim];
    __shared__ float sharedProb[kBr * kBc];
    __shared__ float sharedLse[kBr];
    __shared__ float sharedDelta[kBr];

    for (int index = threadIndex; index < queryCount * headDim; index += static_cast<int>(blockDim.x)) {
        const int localQuery = index / headDim;
        const int dim = index - localQuery * headDim;
        const int globalQuery = queryStart + localQuery;
        sharedQuery[localQuery * kMaxHeadDim + dim] = query[dim * sequenceLength + globalQuery];
        sharedOutGrad[localQuery * kMaxHeadDim + dim] = outGradient[dim * sequenceLength + globalQuery];
    }
    if (threadIndex < queryCount) {
        const int globalQuery = queryStart + threadIndex;
        sharedLse[threadIndex] = logSumExp[globalQuery];
        sharedDelta[threadIndex] = delta[globalQuery];
    }
    __syncthreads();

    float queryGradAccum[kMaxHeadDim];
    for (int dim = 0; dim < kMaxHeadDim; ++dim)
        queryGradAccum[dim] = 0.0f;

    const int keyTileCount = (sequenceLength + kBc - 1) / kBc;
    for (int keyTile = 0; keyTile < keyTileCount; ++keyTile) {
        const int keyStart = keyTile * kBc;
        const int keyCount = sequenceLength - keyStart < kBc ? sequenceLength - keyStart : kBc;

        for (int index = threadIndex; index < keyCount * headDim; index += static_cast<int>(blockDim.x)) {
            const int localKey = index / headDim;
            const int dim = index - localKey * headDim;
            const int globalKey = keyStart + localKey;
            sharedKey[localKey * kMaxHeadDim + dim] = key[dim * sequenceLength + globalKey];
            sharedValue[localKey * kMaxHeadDim + dim] = value[dim * sequenceLength + globalKey];
        }
        __syncthreads();

        if (threadIndex < queryCount) {
            const int globalQuery = queryStart + threadIndex;
            const float lse = sharedLse[threadIndex];
            const float deltaValue = sharedDelta[threadIndex];

            for (int localKey = 0; localKey < keyCount; ++localKey) {
                const int globalKey = keyStart + localKey;
                float probability = 0.0f;
                if (!causal || globalKey <= globalQuery) {
                    float dot = 0.0f;
                    for (int dim = 0; dim < headDim; ++dim)
                        dot += sharedQuery[threadIndex * kMaxHeadDim + dim] * sharedKey[localKey * kMaxHeadDim + dim];
                    probability = expf(dot * scale - lse);
                }
                sharedProb[threadIndex * kBc + localKey] = probability;
            }

            for (int localKey = 0; localKey < keyCount; ++localKey) {
                const float probability = sharedProb[threadIndex * kBc + localKey];
                if (probability == 0.0f) continue;

                float dP = 0.0f;
                for (int dim = 0; dim < headDim; ++dim)
                    dP += sharedOutGrad[threadIndex * kMaxHeadDim + dim] * sharedValue[localKey * kMaxHeadDim + dim];

                const float dS = probability * (dP - deltaValue);
                const float scaledDs = dS * scale;

                for (int dim = 0; dim < headDim; ++dim) {
                    queryGradAccum[dim] += scaledDs * sharedKey[localKey * kMaxHeadDim + dim];
                    atomicAdd(&keyGradient[dim * sequenceLength + (keyStart + localKey)], scaledDs * sharedQuery[threadIndex * kMaxHeadDim + dim]);
                    atomicAdd(&valueGradient[dim * sequenceLength + (keyStart + localKey)], probability * sharedOutGrad[threadIndex * kMaxHeadDim + dim]);
                }
            }
        }
        __syncthreads();
    }

    if (threadIndex < queryCount) {
        const int globalQuery = queryStart + threadIndex;
        for (int dim = 0; dim < headDim; ++dim)
            queryGradient[dim * sequenceLength + globalQuery] = queryGradAccum[dim];
    }
}

void validateSameHeadShape(const CudaMatrix& query, const CudaMatrix& key, const CudaMatrix& value, const char* operationName) {
    if (query.empty() || key.empty() || value.empty())
        throw std::invalid_argument(std::string(operationName) + " empty input");
    if (query.rows != key.rows || query.rows != value.rows)
        throw std::invalid_argument(std::string(operationName) + " headDim mismatch");
    if (query.cols != key.cols || query.cols != value.cols)
        throw std::invalid_argument(std::string(operationName) + " sequenceLength mismatch");
    if (static_cast<int>(query.rows) > CudaFlashAttention::maxHeadDimension)
        throw std::invalid_argument(std::string(operationName) + " headDim exceeds maxHeadDimension");
}

} // namespace

void CudaFlashAttention::forward(const CudaMatrix& query, const CudaMatrix& key, const CudaMatrix& value, CudaMatrix& out, CudaMatrix& logSumExp, float scale, bool causal) {
    validateSameHeadShape(query, key, value, "CudaFlashAttention::forward");
    if (!CudaMatmul::isAvailable())
        throw std::runtime_error("CudaFlashAttention::forward no CUDA device");

    const int headDim = static_cast<int>(query.rows);
    const int sequenceLength = static_cast<int>(query.cols);
    if (sequenceLength <= 0)
        throw std::invalid_argument("CudaFlashAttention::forward empty sequence");

    out.ensureSize(query.rows, query.cols);
    logSumExp.ensureSize(1, query.cols);

    const int queryTileCount = (sequenceLength + queryTileSize - 1) / queryTileSize;
    const int threadCount = 128;
    CudaFlashAttentionForwardEntry<<<queryTileCount, threadCount>>>(
        query.buffer.deviceData,
        key.buffer.deviceData,
        value.buffer.deviceData,
        out.buffer.deviceData,
        logSumExp.buffer.deviceData,
        headDim,
        sequenceLength,
        scale,
        causal ? 1 : 0);
    CudaMatmul::throwIfCudaFailed(cudaGetLastError(), "CudaFlashAttentionForwardEntry launch");
}

void CudaFlashAttention::backward(const CudaMatrix& query, const CudaMatrix& key, const CudaMatrix& value, const CudaMatrix& out, const CudaMatrix& logSumExp, const CudaMatrix& outGradient, CudaMatrix& queryGradient, CudaMatrix& keyGradient, CudaMatrix& valueGradient, float scale, bool causal) {
    validateSameHeadShape(query, key, value, "CudaFlashAttention::backward");
    if (out.rows != query.rows || out.cols != query.cols)
        throw std::invalid_argument("CudaFlashAttention::backward out shape mismatch");
    if (outGradient.rows != query.rows || outGradient.cols != query.cols)
        throw std::invalid_argument("CudaFlashAttention::backward outGradient shape mismatch");
    if (logSumExp.rows != 1 || logSumExp.cols != query.cols)
        throw std::invalid_argument("CudaFlashAttention::backward logSumExp shape mismatch");
    if (!CudaMatmul::isAvailable())
        throw std::runtime_error("CudaFlashAttention::backward no CUDA device");

    const int headDim = static_cast<int>(query.rows);
    const int sequenceLength = static_cast<int>(query.cols);
    if (sequenceLength <= 0)
        throw std::invalid_argument("CudaFlashAttention::backward empty sequence");

    queryGradient.ensureSize(query.rows, query.cols);
    keyGradient.ensureSize(key.rows, key.cols);
    valueGradient.ensureSize(value.rows, value.cols);
    CudaOps::zeroInPlace(queryGradient);
    CudaOps::zeroInPlace(keyGradient);
    CudaOps::zeroInPlace(valueGradient);

    CudaMatrix delta;
    delta.ensureSize(1, query.cols);

    const int deltaThreads = 128;
    const int deltaBlocks = (sequenceLength + deltaThreads - 1) / deltaThreads;
    CudaFlashAttentionDeltaEntry<<<deltaBlocks, deltaThreads>>>(
        out.buffer.deviceData,
        outGradient.buffer.deviceData,
        delta.buffer.deviceData,
        headDim,
        sequenceLength);
    CudaMatmul::throwIfCudaFailed(cudaGetLastError(), "CudaFlashAttentionDeltaEntry launch");

    const int queryTileCount = (sequenceLength + queryTileSize - 1) / queryTileSize;
    const int threadCount = 128;
    CudaFlashAttentionBackwardEntry<<<queryTileCount, threadCount>>>(
        query.buffer.deviceData,
        key.buffer.deviceData,
        value.buffer.deviceData,
        outGradient.buffer.deviceData,
        logSumExp.buffer.deviceData,
        delta.buffer.deviceData,
        queryGradient.buffer.deviceData,
        keyGradient.buffer.deviceData,
        valueGradient.buffer.deviceData,
        headDim,
        sequenceLength,
        scale,
        causal ? 1 : 0);
    CudaMatmul::throwIfCudaFailed(cudaGetLastError(), "CudaFlashAttentionBackwardEntry launch");
}
