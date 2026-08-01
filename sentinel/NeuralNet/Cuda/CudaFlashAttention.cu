#include "CudaFlashAttention.hpp"

#include <cuda_runtime.h>
#include <cmath>
#include <stdexcept>

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

    // load Q tile: layout headDim x seq -> store as queryLocal x headDim for coalesced dots
    for (int index = threadIndex; index < queryCount * headDim; index += static_cast<int>(blockDim.x)) {
        const int localQuery = index / headDim;
        const int dim = index - localQuery * headDim;
        const int globalQuery = queryStart + localQuery;
        sharedQuery[localQuery * kMaxHeadDim + dim] = query[dim * sequenceLength + globalQuery];
    }
    for (int index = threadIndex; index < kBr * kMaxHeadDim; index += static_cast<int>(blockDim.x)) {
        if (index >= queryCount * kMaxHeadDim)
            sharedQuery[index] = 0.0f;
    }
    __syncthreads();

    // each thread owns one query row in the tile
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

} // namespace

void CudaFlashAttention::forward(const CudaMatrix& query, const CudaMatrix& key, const CudaMatrix& value, CudaMatrix& out, CudaMatrix& logSumExp, float scale, bool causal) {
    if (query.empty() || key.empty() || value.empty())
        throw std::invalid_argument("CudaFlashAttention::forward empty input");
    if (query.rows != key.rows || query.rows != value.rows)
        throw std::invalid_argument("CudaFlashAttention::forward headDim mismatch");
    if (query.cols != key.cols || query.cols != value.cols)
        throw std::invalid_argument("CudaFlashAttention::forward sequenceLength mismatch");
    if (static_cast<int>(query.rows) > maxHeadDimension)
        throw std::invalid_argument("CudaFlashAttention::forward headDim exceeds maxHeadDimension");
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
