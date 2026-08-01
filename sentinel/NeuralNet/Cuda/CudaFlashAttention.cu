#include "CudaFlashAttention.hpp"

#include "CudaOps.hpp"

#include <cuda_runtime.h>
#include <cmath>
#include <stdexcept>
#include <string>

namespace {

constexpr int kBrMax = CudaFlashAttention::queryTileSize;

__device__ __forceinline__ float flashDot(const float* left, const float* right, int headDim) {
    float sum = 0.0f;
#pragma unroll 4
    for (int dim = 0; dim < headDim; ++dim)
        sum += left[dim] * right[dim];
    return sum;
}

template <int HeadDim>
__device__ __forceinline__ float flashDotFixed(const float* left, const float* right) {
    float sum = 0.0f;
#pragma unroll
    for (int dim = 0; dim < HeadDim; ++dim)
        sum += left[dim] * right[dim];
    return sum;
}

__device__ __forceinline__ const float* headRow(const float* tensor, int headIndex, int headDim, int dim, int strideColumns, int absoluteColumn) {
    return tensor + (headIndex * headDim + dim) * strideColumns + absoluteColumn;
}

__device__ __forceinline__ float* headRowMutable(float* tensor, int headIndex, int headDim, int dim, int strideColumns, int absoluteColumn) {
    return tensor + (headIndex * headDim + dim) * strideColumns + absoluteColumn;
}

__device__ __forceinline__ void fillScores(
    float* sharedScores,
    const float* sharedQuery,
    const float* sharedKey,
    int queryCount,
    int keyCount,
    int tileBc,
    int headDim,
    int queryStart,
    int keyStart,
    float scale,
    int causal,
    int threadIndex,
    int threadCount) {
    const int pairCount = queryCount * keyCount;
    for (int pair = threadIndex; pair < pairCount; pair += threadCount) {
        const int localQuery = pair / keyCount;
        const int localKey = pair - localQuery * keyCount;
        const int globalQuery = queryStart + localQuery;
        const int globalKey = keyStart + localKey;
        float score = -INFINITY;
        if (!causal || globalKey <= globalQuery)
            score = flashDot(sharedQuery + localQuery * headDim, sharedKey + localKey * headDim, headDim) * scale;
        sharedScores[localQuery * tileBc + localKey] = score;
    }
}

template <int HeadDim>
__device__ __forceinline__ void fillScoresFixed(
    float* sharedScores,
    const float* sharedQuery,
    const float* sharedKey,
    int queryCount,
    int keyCount,
    int tileBc,
    int queryStart,
    int keyStart,
    float scale,
    int causal,
    int threadIndex,
    int threadCount) {
    const int pairCount = queryCount * keyCount;
    for (int pair = threadIndex; pair < pairCount; pair += threadCount) {
        const int localQuery = pair / keyCount;
        const int localKey = pair - localQuery * keyCount;
        const int globalQuery = queryStart + localQuery;
        const int globalKey = keyStart + localKey;
        float score = -INFINITY;
        if (!causal || globalKey <= globalQuery)
            score = flashDotFixed<HeadDim>(sharedQuery + localQuery * HeadDim, sharedKey + localKey * HeadDim) * scale;
        sharedScores[localQuery * tileBc + localKey] = score;
    }
}

__device__ __forceinline__ void fillProbabilities(
    float* sharedProb,
    const float* sharedQuery,
    const float* sharedKey,
    const float* sharedLse,
    int queryCount,
    int keyCount,
    int tileBc,
    int headDim,
    int queryStart,
    int keyStart,
    float scale,
    int causal,
    int threadIndex,
    int threadCount) {
    const int pairCount = queryCount * keyCount;
    for (int pair = threadIndex; pair < pairCount; pair += threadCount) {
        const int localQuery = pair / keyCount;
        const int localKey = pair - localQuery * keyCount;
        const int globalQuery = queryStart + localQuery;
        const int globalKey = keyStart + localKey;
        float probability = 0.0f;
        if (!causal || globalKey <= globalQuery) {
            const float score = flashDot(sharedQuery + localQuery * headDim, sharedKey + localKey * headDim, headDim) * scale;
            probability = __expf(score - sharedLse[localQuery]);
        }
        sharedProb[localQuery * tileBc + localKey] = probability;
    }
}

template <int HeadDim>
__device__ __forceinline__ void fillProbabilitiesFixed(
    float* sharedProb,
    const float* sharedQuery,
    const float* sharedKey,
    const float* sharedLse,
    int queryCount,
    int keyCount,
    int tileBc,
    int queryStart,
    int keyStart,
    float scale,
    int causal,
    int threadIndex,
    int threadCount) {
    const int pairCount = queryCount * keyCount;
    for (int pair = threadIndex; pair < pairCount; pair += threadCount) {
        const int localQuery = pair / keyCount;
        const int localKey = pair - localQuery * keyCount;
        const int globalQuery = queryStart + localQuery;
        const int globalKey = keyStart + localKey;
        float probability = 0.0f;
        if (!causal || globalKey <= globalQuery) {
            const float score = flashDotFixed<HeadDim>(sharedQuery + localQuery * HeadDim, sharedKey + localKey * HeadDim) * scale;
            probability = __expf(score - sharedLse[localQuery]);
        }
        sharedProb[localQuery * tileBc + localKey] = probability;
    }
}

__global__ void CudaFlashAttentionForwardEntry(
    const float* query,
    const float* key,
    const float* value,
    float* out,
    float* logSumExp,
    int headDim,
    int strideColumns,
    int columnStart,
    int sequenceLength,
    float scale,
    int causal,
    int tileBr,
    int tileBc) {
    extern __shared__ float shared[];
    float* sharedQuery = shared;
    float* sharedKey = sharedQuery + tileBr * headDim;
    float* sharedValue = sharedKey + tileBc * headDim;
    float* sharedScores = sharedValue + tileBc * headDim;

    const int headIndex = static_cast<int>(blockIdx.y);
    const int queryTile = static_cast<int>(blockIdx.x);
    const int queryStart = queryTile * tileBr;
    if (queryStart >= sequenceLength) return;

    const int queryCount = sequenceLength - queryStart < tileBr ? sequenceLength - queryStart : tileBr;
    const int threadIndex = static_cast<int>(threadIdx.x);
    const int threadCount = static_cast<int>(blockDim.x);

    for (int index = threadIndex; index < queryCount * headDim; index += threadCount) {
        const int localQuery = index / headDim;
        const int dim = index - localQuery * headDim;
        const int absoluteColumn = columnStart + queryStart + localQuery;
        sharedQuery[localQuery * headDim + dim] = *headRow(query, headIndex, headDim, dim, strideColumns, absoluteColumn);
    }
    __syncthreads();

    float rowMax = -INFINITY;
    float rowSum = 0.0f;
    float outputAccum[64];
#pragma unroll
    for (int dim = 0; dim < 64; ++dim)
        outputAccum[dim] = 0.0f;

    const int keyTileCount = (sequenceLength + tileBc - 1) / tileBc;
    for (int keyTile = 0; keyTile < keyTileCount; ++keyTile) {
        const int keyStart = keyTile * tileBc;
        const int keyCount = sequenceLength - keyStart < tileBc ? sequenceLength - keyStart : tileBc;
        if (causal && keyStart > (queryStart + queryCount - 1)) break;

        for (int index = threadIndex; index < keyCount * headDim; index += threadCount) {
            const int localKey = index / headDim;
            const int dim = index - localKey * headDim;
            const int absoluteColumn = columnStart + keyStart + localKey;
            sharedKey[localKey * headDim + dim] = *headRow(key, headIndex, headDim, dim, strideColumns, absoluteColumn);
            sharedValue[localKey * headDim + dim] = *headRow(value, headIndex, headDim, dim, strideColumns, absoluteColumn);
        }
        __syncthreads();

        fillScores(sharedScores, sharedQuery, sharedKey, queryCount, keyCount, tileBc, headDim, queryStart, keyStart, scale, causal, threadIndex, threadCount);
        __syncthreads();

        if (threadIndex < queryCount) {
            float tileRowMax = -INFINITY;
            for (int localKey = 0; localKey < keyCount; ++localKey) {
                const float score = sharedScores[threadIndex * tileBc + localKey];
                if (score > tileRowMax)
                    tileRowMax = score;
            }

            const float newRowMax = rowMax > tileRowMax ? rowMax : tileRowMax;
            const float rescale = __expf(rowMax - newRowMax);
            float tileSum = 0.0f;
            for (int localKey = 0; localKey < keyCount; ++localKey) {
                const float probability = __expf(sharedScores[threadIndex * tileBc + localKey] - newRowMax);
                sharedScores[threadIndex * tileBc + localKey] = probability;
                tileSum += probability;
            }

            for (int dim = 0; dim < headDim; ++dim) {
                float weighted = 0.0f;
                for (int localKey = 0; localKey < keyCount; ++localKey)
                    weighted += sharedScores[threadIndex * tileBc + localKey] * sharedValue[localKey * headDim + dim];
                outputAccum[dim] = outputAccum[dim] * rescale + weighted;
            }

            rowSum = rowSum * rescale + tileSum;
            rowMax = newRowMax;
        }
        __syncthreads();
    }

    if (threadIndex < queryCount) {
        const int absoluteColumn = columnStart + queryStart + threadIndex;
        const float inverseSum = 1.0f / rowSum;
        for (int dim = 0; dim < headDim; ++dim)
            *headRowMutable(out, headIndex, headDim, dim, strideColumns, absoluteColumn) = outputAccum[dim] * inverseSum;
        logSumExp[headIndex * strideColumns + absoluteColumn] = rowMax + __logf(rowSum);
    }
}

template <int HeadDim, int TileBr, int TileBc>
__global__ void CudaFlashAttentionForwardFixedEntry(
    const float* query,
    const float* key,
    const float* value,
    float* out,
    float* logSumExp,
    int strideColumns,
    int columnStart,
    int sequenceLength,
    float scale,
    int causal) {
    extern __shared__ float shared[];
    float* sharedQuery = shared;
    float* sharedKey = sharedQuery + TileBr * HeadDim;
    float* sharedValue = sharedKey + TileBc * HeadDim;
    float* sharedScores = sharedValue + TileBc * HeadDim;

    const int headIndex = static_cast<int>(blockIdx.y);
    const int queryTile = static_cast<int>(blockIdx.x);
    const int queryStart = queryTile * TileBr;
    if (queryStart >= sequenceLength) return;

    const int queryCount = sequenceLength - queryStart < TileBr ? sequenceLength - queryStart : TileBr;
    const int threadIndex = static_cast<int>(threadIdx.x);
    const int threadCount = static_cast<int>(blockDim.x);

    for (int index = threadIndex; index < queryCount * HeadDim; index += threadCount) {
        const int localQuery = index / HeadDim;
        const int dim = index - localQuery * HeadDim;
        const int absoluteColumn = columnStart + queryStart + localQuery;
        sharedQuery[localQuery * HeadDim + dim] = *headRow(query, headIndex, HeadDim, dim, strideColumns, absoluteColumn);
    }
    __syncthreads();

    float rowMax = -INFINITY;
    float rowSum = 0.0f;
    float outputAccum[HeadDim];
#pragma unroll
    for (int dim = 0; dim < HeadDim; ++dim)
        outputAccum[dim] = 0.0f;

    const int keyTileCount = (sequenceLength + TileBc - 1) / TileBc;
    for (int keyTile = 0; keyTile < keyTileCount; ++keyTile) {
        const int keyStart = keyTile * TileBc;
        const int keyCount = sequenceLength - keyStart < TileBc ? sequenceLength - keyStart : TileBc;
        if (causal && keyStart > (queryStart + queryCount - 1)) break;

        for (int index = threadIndex; index < keyCount * HeadDim; index += threadCount) {
            const int localKey = index / HeadDim;
            const int dim = index - localKey * HeadDim;
            const int absoluteColumn = columnStart + keyStart + localKey;
            sharedKey[localKey * HeadDim + dim] = *headRow(key, headIndex, HeadDim, dim, strideColumns, absoluteColumn);
            sharedValue[localKey * HeadDim + dim] = *headRow(value, headIndex, HeadDim, dim, strideColumns, absoluteColumn);
        }
        __syncthreads();

        fillScoresFixed<HeadDim>(sharedScores, sharedQuery, sharedKey, queryCount, keyCount, TileBc, queryStart, keyStart, scale, causal, threadIndex, threadCount);
        __syncthreads();

        if (threadIndex < queryCount) {
            float tileRowMax = -INFINITY;
            for (int localKey = 0; localKey < keyCount; ++localKey) {
                const float score = sharedScores[threadIndex * TileBc + localKey];
                if (score > tileRowMax)
                    tileRowMax = score;
            }

            const float newRowMax = rowMax > tileRowMax ? rowMax : tileRowMax;
            const float rescale = __expf(rowMax - newRowMax);
            float tileSum = 0.0f;
            for (int localKey = 0; localKey < keyCount; ++localKey) {
                const float probability = __expf(sharedScores[threadIndex * TileBc + localKey] - newRowMax);
                sharedScores[threadIndex * TileBc + localKey] = probability;
                tileSum += probability;
            }

#pragma unroll
            for (int dim = 0; dim < HeadDim; ++dim) {
                float weighted = 0.0f;
                for (int localKey = 0; localKey < keyCount; ++localKey)
                    weighted += sharedScores[threadIndex * TileBc + localKey] * sharedValue[localKey * HeadDim + dim];
                outputAccum[dim] = outputAccum[dim] * rescale + weighted;
            }

            rowSum = rowSum * rescale + tileSum;
            rowMax = newRowMax;
        }
        __syncthreads();
    }

    if (threadIndex < queryCount) {
        const int absoluteColumn = columnStart + queryStart + threadIndex;
        const float inverseSum = 1.0f / rowSum;
#pragma unroll
        for (int dim = 0; dim < HeadDim; ++dim)
            *headRowMutable(out, headIndex, HeadDim, dim, strideColumns, absoluteColumn) = outputAccum[dim] * inverseSum;
        logSumExp[headIndex * strideColumns + absoluteColumn] = rowMax + __logf(rowSum);
    }
}

__global__ void CudaFlashAttentionDeltaEntry(
    const float* out,
    const float* outGradient,
    float* delta,
    int headDim,
    int strideColumns,
    int columnStart,
    int sequenceLength) {
    const int headIndex = static_cast<int>(blockIdx.y);
    const int localColumn = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (localColumn >= sequenceLength) return;

    const int absoluteColumn = columnStart + localColumn;
    float sum = 0.0f;
    for (int dim = 0; dim < headDim; ++dim)
        sum += *headRow(out, headIndex, headDim, dim, strideColumns, absoluteColumn) * *headRow(outGradient, headIndex, headDim, dim, strideColumns, absoluteColumn);
    delta[headIndex * sequenceLength + localColumn] = sum;
}

/// <summary>key-tile outer: dK/dV in smem; dQ via sparse global atomics</summary>
__global__ void CudaFlashAttentionBackwardKeyEntry(
    const float* query,
    const float* key,
    const float* value,
    const float* outGradient,
    const float* logSumExp,
    const float* delta,
    float* queryGradient,
    float* keyGradient,
    float* valueGradient,
    int headDim,
    int strideColumns,
    int columnStart,
    int sequenceLength,
    float scale,
    int causal,
    int tileBr,
    int tileBc) {
    extern __shared__ float shared[];
    float* sharedKey = shared;
    float* sharedValue = sharedKey + tileBc * headDim;
    float* sharedKeyGrad = sharedValue + tileBc * headDim;
    float* sharedValueGrad = sharedKeyGrad + tileBc * headDim;
    float* sharedQuery = sharedValueGrad + tileBc * headDim;
    float* sharedOutGrad = sharedQuery + tileBr * headDim;
    float* sharedProb = sharedOutGrad + tileBr * headDim;
    float* sharedLse = sharedProb + tileBr * tileBc;
    float* sharedDelta = sharedLse + tileBr;

    const int headIndex = static_cast<int>(blockIdx.y);
    const int keyTile = static_cast<int>(blockIdx.x);
    const int keyStart = keyTile * tileBc;
    if (keyStart >= sequenceLength) return;

    const int keyCount = sequenceLength - keyStart < tileBc ? sequenceLength - keyStart : tileBc;
    const int threadIndex = static_cast<int>(threadIdx.x);
    const int threadCount = static_cast<int>(blockDim.x);

    for (int index = threadIndex; index < keyCount * headDim; index += threadCount) {
        const int localKey = index / headDim;
        const int dim = index - localKey * headDim;
        const int absoluteColumn = columnStart + keyStart + localKey;
        sharedKey[localKey * headDim + dim] = *headRow(key, headIndex, headDim, dim, strideColumns, absoluteColumn);
        sharedValue[localKey * headDim + dim] = *headRow(value, headIndex, headDim, dim, strideColumns, absoluteColumn);
        sharedKeyGrad[localKey * headDim + dim] = 0.0f;
        sharedValueGrad[localKey * headDim + dim] = 0.0f;
    }
    __syncthreads();

    const int queryTileCount = (sequenceLength + tileBr - 1) / tileBr;
    for (int queryTile = 0; queryTile < queryTileCount; ++queryTile) {
        const int queryStart = queryTile * tileBr;
        const int queryCount = sequenceLength - queryStart < tileBr ? sequenceLength - queryStart : tileBr;
        if (causal && (queryStart + queryCount - 1) < keyStart) continue;

        for (int index = threadIndex; index < queryCount * headDim; index += threadCount) {
            const int localQuery = index / headDim;
            const int dim = index - localQuery * headDim;
            const int absoluteColumn = columnStart + queryStart + localQuery;
            sharedQuery[localQuery * headDim + dim] = *headRow(query, headIndex, headDim, dim, strideColumns, absoluteColumn);
            sharedOutGrad[localQuery * headDim + dim] = *headRow(outGradient, headIndex, headDim, dim, strideColumns, absoluteColumn);
        }
        if (threadIndex < queryCount) {
            const int absoluteColumn = columnStart + queryStart + threadIndex;
            sharedLse[threadIndex] = logSumExp[headIndex * strideColumns + absoluteColumn];
            sharedDelta[threadIndex] = delta[headIndex * sequenceLength + queryStart + threadIndex];
        }
        __syncthreads();

        fillProbabilities(sharedProb, sharedQuery, sharedKey, sharedLse, queryCount, keyCount, tileBc, headDim, queryStart, keyStart, scale, causal, threadIndex, threadCount);
        __syncthreads();

        if (threadIndex < keyCount) {
            const int localKey = threadIndex;
            for (int localQuery = 0; localQuery < queryCount; ++localQuery) {
                const float probability = sharedProb[localQuery * tileBc + localKey];
                float scaledDs = 0.0f;
                if (probability != 0.0f) {
                    float dP = 0.0f;
                    for (int dim = 0; dim < headDim; ++dim)
                        dP += sharedOutGrad[localQuery * headDim + dim] * sharedValue[localKey * headDim + dim];
                    scaledDs = probability * (dP - sharedDelta[localQuery]) * scale;
                    for (int dim = 0; dim < headDim; ++dim) {
                        sharedKeyGrad[localKey * headDim + dim] += scaledDs * sharedQuery[localQuery * headDim + dim];
                        sharedValueGrad[localKey * headDim + dim] += probability * sharedOutGrad[localQuery * headDim + dim];
                    }
                }
                sharedProb[localQuery * tileBc + localKey] = scaledDs;
            }
        }
        __syncthreads();

        if (threadIndex < queryCount) {
            float queryGradLocal[64];
#pragma unroll
            for (int dim = 0; dim < 64; ++dim)
                queryGradLocal[dim] = 0.0f;
            for (int localKey = 0; localKey < keyCount; ++localKey) {
                const float scaledDs = sharedProb[threadIndex * tileBc + localKey];
                if (scaledDs == 0.0f) continue;
                for (int dim = 0; dim < headDim; ++dim)
                    queryGradLocal[dim] += scaledDs * sharedKey[localKey * headDim + dim];
            }
            const int absoluteColumn = columnStart + queryStart + threadIndex;
            for (int dim = 0; dim < headDim; ++dim)
                atomicAdd(headRowMutable(queryGradient, headIndex, headDim, dim, strideColumns, absoluteColumn), queryGradLocal[dim]);
        }
        __syncthreads();
    }

    for (int index = threadIndex; index < keyCount * headDim; index += threadCount) {
        const int localKey = index / headDim;
        const int dim = index - localKey * headDim;
        const int absoluteColumn = columnStart + keyStart + localKey;
        *headRowMutable(keyGradient, headIndex, headDim, dim, strideColumns, absoluteColumn) = sharedKeyGrad[localKey * headDim + dim];
        *headRowMutable(valueGradient, headIndex, headDim, dim, strideColumns, absoluteColumn) = sharedValueGrad[localKey * headDim + dim];
    }
}

template <int HeadDim, int TileBr, int TileBc>
__global__ void CudaFlashAttentionBackwardKeyFixedEntry(
    const float* query,
    const float* key,
    const float* value,
    const float* outGradient,
    const float* logSumExp,
    const float* delta,
    float* queryGradient,
    float* keyGradient,
    float* valueGradient,
    int strideColumns,
    int columnStart,
    int sequenceLength,
    float scale,
    int causal) {
    extern __shared__ float shared[];
    float* sharedKey = shared;
    float* sharedValue = sharedKey + TileBc * HeadDim;
    float* sharedKeyGrad = sharedValue + TileBc * HeadDim;
    float* sharedValueGrad = sharedKeyGrad + TileBc * HeadDim;
    float* sharedQuery = sharedValueGrad + TileBc * HeadDim;
    float* sharedOutGrad = sharedQuery + TileBr * HeadDim;
    float* sharedProb = sharedOutGrad + TileBr * HeadDim;
    float* sharedLse = sharedProb + TileBr * TileBc;
    float* sharedDelta = sharedLse + TileBr;

    const int headIndex = static_cast<int>(blockIdx.y);
    const int keyTile = static_cast<int>(blockIdx.x);
    const int keyStart = keyTile * TileBc;
    if (keyStart >= sequenceLength) return;

    const int keyCount = sequenceLength - keyStart < TileBc ? sequenceLength - keyStart : TileBc;
    const int threadIndex = static_cast<int>(threadIdx.x);
    const int threadCount = static_cast<int>(blockDim.x);

    for (int index = threadIndex; index < keyCount * HeadDim; index += threadCount) {
        const int localKey = index / HeadDim;
        const int dim = index - localKey * HeadDim;
        const int absoluteColumn = columnStart + keyStart + localKey;
        sharedKey[localKey * HeadDim + dim] = *headRow(key, headIndex, HeadDim, dim, strideColumns, absoluteColumn);
        sharedValue[localKey * HeadDim + dim] = *headRow(value, headIndex, HeadDim, dim, strideColumns, absoluteColumn);
        sharedKeyGrad[localKey * HeadDim + dim] = 0.0f;
        sharedValueGrad[localKey * HeadDim + dim] = 0.0f;
    }
    __syncthreads();

    const int queryTileCount = (sequenceLength + TileBr - 1) / TileBr;
    for (int queryTile = 0; queryTile < queryTileCount; ++queryTile) {
        const int queryStart = queryTile * TileBr;
        const int queryCount = sequenceLength - queryStart < TileBr ? sequenceLength - queryStart : TileBr;
        if (causal && (queryStart + queryCount - 1) < keyStart) continue;

        for (int index = threadIndex; index < queryCount * HeadDim; index += threadCount) {
            const int localQuery = index / HeadDim;
            const int dim = index - localQuery * HeadDim;
            const int absoluteColumn = columnStart + queryStart + localQuery;
            sharedQuery[localQuery * HeadDim + dim] = *headRow(query, headIndex, HeadDim, dim, strideColumns, absoluteColumn);
            sharedOutGrad[localQuery * HeadDim + dim] = *headRow(outGradient, headIndex, HeadDim, dim, strideColumns, absoluteColumn);
        }
        if (threadIndex < queryCount) {
            const int absoluteColumn = columnStart + queryStart + threadIndex;
            sharedLse[threadIndex] = logSumExp[headIndex * strideColumns + absoluteColumn];
            sharedDelta[threadIndex] = delta[headIndex * sequenceLength + queryStart + threadIndex];
        }
        __syncthreads();

        fillProbabilitiesFixed<HeadDim>(sharedProb, sharedQuery, sharedKey, sharedLse, queryCount, keyCount, TileBc, queryStart, keyStart, scale, causal, threadIndex, threadCount);
        __syncthreads();

        if (threadIndex < keyCount) {
            const int localKey = threadIndex;
            for (int localQuery = 0; localQuery < queryCount; ++localQuery) {
                const float probability = sharedProb[localQuery * TileBc + localKey];
                float scaledDs = 0.0f;
                if (probability != 0.0f) {
                    float dP = 0.0f;
#pragma unroll
                    for (int dim = 0; dim < HeadDim; ++dim)
                        dP += sharedOutGrad[localQuery * HeadDim + dim] * sharedValue[localKey * HeadDim + dim];
                    scaledDs = probability * (dP - sharedDelta[localQuery]) * scale;
#pragma unroll
                    for (int dim = 0; dim < HeadDim; ++dim) {
                        sharedKeyGrad[localKey * HeadDim + dim] += scaledDs * sharedQuery[localQuery * HeadDim + dim];
                        sharedValueGrad[localKey * HeadDim + dim] += probability * sharedOutGrad[localQuery * HeadDim + dim];
                    }
                }
                sharedProb[localQuery * TileBc + localKey] = scaledDs;
            }
        }
        __syncthreads();

        if (threadIndex < queryCount) {
            float queryGradLocal[HeadDim];
#pragma unroll
            for (int dim = 0; dim < HeadDim; ++dim)
                queryGradLocal[dim] = 0.0f;
            for (int localKey = 0; localKey < keyCount; ++localKey) {
                const float scaledDs = sharedProb[threadIndex * TileBc + localKey];
                if (scaledDs == 0.0f) continue;
#pragma unroll
                for (int dim = 0; dim < HeadDim; ++dim)
                    queryGradLocal[dim] += scaledDs * sharedKey[localKey * HeadDim + dim];
            }
            const int absoluteColumn = columnStart + queryStart + threadIndex;
#pragma unroll
            for (int dim = 0; dim < HeadDim; ++dim)
                atomicAdd(headRowMutable(queryGradient, headIndex, HeadDim, dim, strideColumns, absoluteColumn), queryGradLocal[dim]);
        }
        __syncthreads();
    }

    for (int index = threadIndex; index < keyCount * HeadDim; index += threadCount) {
        const int localKey = index / HeadDim;
        const int dim = index - localKey * HeadDim;
        const int absoluteColumn = columnStart + keyStart + localKey;
        *headRowMutable(keyGradient, headIndex, HeadDim, dim, strideColumns, absoluteColumn) = sharedKeyGrad[localKey * HeadDim + dim];
        *headRowMutable(valueGradient, headIndex, HeadDim, dim, strideColumns, absoluteColumn) = sharedValueGrad[localKey * HeadDim + dim];
    }
}

void validateMultiHeadShape(const CudaMatrix& query, const CudaMatrix& key, const CudaMatrix& value, int headCount, int headDimension, const char* operationName) {
    if (query.empty() || key.empty() || value.empty())
        throw std::invalid_argument(std::string(operationName) + " empty input");
    if (headCount <= 0 || headDimension <= 0)
        throw std::invalid_argument(std::string(operationName) + " invalid head geometry");
    if (headDimension > CudaFlashAttention::maxHeadDimension)
        throw std::invalid_argument(std::string(operationName) + " headDim exceeds maxHeadDimension");
    if (query.rows != static_cast<size_t>(headCount) * static_cast<size_t>(headDimension))
        throw std::invalid_argument(std::string(operationName) + " query rows mismatch headCount*headDim");
    if (key.rows != query.rows || value.rows != query.rows)
        throw std::invalid_argument(std::string(operationName) + " key/value rows mismatch");
    if (query.cols != key.cols || query.cols != value.cols)
        throw std::invalid_argument(std::string(operationName) + " sequenceLength mismatch");
}

size_t forwardSharedBytes(int headDim, int tileBr, int tileBc) {
    return static_cast<size_t>((tileBr + 2 * tileBc) * headDim + tileBr * tileBc) * sizeof(float);
}

size_t backwardKeySharedBytes(int headDim, int tileBr, int tileBc) {
    return static_cast<size_t>((2 * tileBr + 4 * tileBc) * headDim + tileBr * tileBc + 2 * tileBr) * sizeof(float);
}

void chooseTileSizes(int headDim, int& tileBr, int& tileBc) {
    tileBr = 8;
    tileBc = 8;
    for (int candidate = kBrMax; candidate >= 8; candidate /= 2) {
        if (backwardKeySharedBytes(headDim, candidate, candidate) <= 48 * 1024) {
            tileBr = candidate;
            tileBc = candidate;
            return;
        }
    }
}

int chooseThreadCount(int tileBr, int tileBc) {
    const int pairs = tileBr * tileBc;
    if (pairs >= 256) return 256;
    if (pairs >= 128) return 128;
    return 64;
}

void resolveColumnWindow(const CudaMatrix& query, int& columnStart, int& columnCount) {
    const int strideColumns = static_cast<int>(query.cols);
    if (columnCount <= 0)
        columnCount = strideColumns - columnStart;
    if (columnStart < 0 || columnCount <= 0 || columnStart + columnCount > strideColumns)
        throw std::invalid_argument("CudaFlashAttention invalid column window");
}

} // namespace

void CudaFlashAttention::forwardMultiHead(const CudaMatrix& query, const CudaMatrix& key, const CudaMatrix& value, CudaMatrix& out, CudaMatrix& logSumExp, int headCount, int headDimension, float scale, bool causal, int columnStart, int columnCount) {
    validateMultiHeadShape(query, key, value, headCount, headDimension, "CudaFlashAttention::forwardMultiHead");
    if (!CudaMatmul::isAvailable())
        throw std::runtime_error("CudaFlashAttention::forwardMultiHead no CUDA device");

    resolveColumnWindow(query, columnStart, columnCount);
    const int strideColumns = static_cast<int>(query.cols);

    out.ensureSize(query.rows, query.cols);
    logSumExp.ensureSize(static_cast<size_t>(headCount), query.cols);

    int tileBr = 0;
    int tileBc = 0;
    chooseTileSizes(headDimension, tileBr, tileBc);

    const int queryTileCount = (columnCount + tileBr - 1) / tileBr;
    const dim3 grid(static_cast<unsigned>(queryTileCount), static_cast<unsigned>(headCount));
    const int threadCount = chooseThreadCount(tileBr, tileBc);
    const size_t sharedBytes = forwardSharedBytes(headDimension, tileBr, tileBc);
    const int causalFlag = causal ? 1 : 0;

    if (headDimension == 16 && tileBr == 64 && tileBc == 64) {
        CudaFlashAttentionForwardFixedEntry<16, 64, 64><<<grid, threadCount, sharedBytes>>>(
            query.buffer.deviceData, key.buffer.deviceData, value.buffer.deviceData, out.buffer.deviceData, logSumExp.buffer.deviceData, strideColumns, columnStart, columnCount, scale, causalFlag);
    } else if (headDimension == 16 && tileBr == 32 && tileBc == 32) {
        CudaFlashAttentionForwardFixedEntry<16, 32, 32><<<grid, threadCount, sharedBytes>>>(
            query.buffer.deviceData, key.buffer.deviceData, value.buffer.deviceData, out.buffer.deviceData, logSumExp.buffer.deviceData, strideColumns, columnStart, columnCount, scale, causalFlag);
    } else if (headDimension == 64 && tileBr == 16 && tileBc == 16) {
        CudaFlashAttentionForwardFixedEntry<64, 16, 16><<<grid, threadCount, sharedBytes>>>(
            query.buffer.deviceData, key.buffer.deviceData, value.buffer.deviceData, out.buffer.deviceData, logSumExp.buffer.deviceData, strideColumns, columnStart, columnCount, scale, causalFlag);
    } else {
        CudaFlashAttentionForwardEntry<<<grid, threadCount, sharedBytes>>>(
            query.buffer.deviceData, key.buffer.deviceData, value.buffer.deviceData, out.buffer.deviceData, logSumExp.buffer.deviceData, headDimension, strideColumns, columnStart, columnCount, scale, causalFlag, tileBr, tileBc);
    }
    CudaMatmul::throwIfCudaFailed(cudaGetLastError(), "CudaFlashAttentionForwardEntry launch");
}

void CudaFlashAttention::forward(const CudaMatrix& query, const CudaMatrix& key, const CudaMatrix& value, CudaMatrix& out, CudaMatrix& logSumExp, float scale, bool causal) {
    forwardMultiHead(query, key, value, out, logSumExp, 1, static_cast<int>(query.rows), scale, causal, 0, static_cast<int>(query.cols));
}

void CudaFlashAttention::backwardMultiHead(const CudaMatrix& query, const CudaMatrix& key, const CudaMatrix& value, const CudaMatrix& out, const CudaMatrix& logSumExp, const CudaMatrix& outGradient, CudaMatrix& queryGradient, CudaMatrix& keyGradient, CudaMatrix& valueGradient, CudaMatrix& deltaWorkspace, int headCount, int headDimension, float scale, bool causal, int columnStart, int columnCount) {
    validateMultiHeadShape(query, key, value, headCount, headDimension, "CudaFlashAttention::backwardMultiHead");
    if (out.rows != query.rows || out.cols != query.cols)
        throw std::invalid_argument("CudaFlashAttention::backwardMultiHead out shape mismatch");
    if (outGradient.rows != query.rows || outGradient.cols != query.cols)
        throw std::invalid_argument("CudaFlashAttention::backwardMultiHead outGradient shape mismatch");
    if (logSumExp.rows != static_cast<size_t>(headCount) || logSumExp.cols != query.cols)
        throw std::invalid_argument("CudaFlashAttention::backwardMultiHead logSumExp shape mismatch");
    if (!CudaMatmul::isAvailable())
        throw std::runtime_error("CudaFlashAttention::backwardMultiHead no CUDA device");

    resolveColumnWindow(query, columnStart, columnCount);
    const int strideColumns = static_cast<int>(query.cols);

    queryGradient.ensureSize(query.rows, query.cols);
    keyGradient.ensureSize(key.rows, key.cols);
    valueGradient.ensureSize(value.rows, value.cols);

    // dK/dV overwrite the active window; dQ accumulates with atomics (caller must zero dQ)
    deltaWorkspace.ensureSize(static_cast<size_t>(headCount), static_cast<size_t>(columnCount));

    const int deltaThreads = 128;
    const int deltaBlocksX = (columnCount + deltaThreads - 1) / deltaThreads;
    const dim3 deltaGrid(static_cast<unsigned>(deltaBlocksX), static_cast<unsigned>(headCount));
    CudaFlashAttentionDeltaEntry<<<deltaGrid, deltaThreads>>>(
        out.buffer.deviceData,
        outGradient.buffer.deviceData,
        deltaWorkspace.buffer.deviceData,
        headDimension,
        strideColumns,
        columnStart,
        columnCount);
    CudaMatmul::throwIfCudaFailed(cudaGetLastError(), "CudaFlashAttentionDeltaEntry launch");

    int tileBr = 0;
    int tileBc = 0;
    chooseTileSizes(headDimension, tileBr, tileBc);

    const int keyTileCount = (columnCount + tileBc - 1) / tileBc;
    const dim3 grid(static_cast<unsigned>(keyTileCount), static_cast<unsigned>(headCount));
    const int threadCount = chooseThreadCount(tileBr, tileBc);
    const size_t keySharedBytes = backwardKeySharedBytes(headDimension, tileBr, tileBc);
    const int causalFlag = causal ? 1 : 0;

    if (headDimension == 16 && tileBr == 64 && tileBc == 64) {
        CudaFlashAttentionBackwardKeyFixedEntry<16, 64, 64><<<grid, threadCount, keySharedBytes>>>(
            query.buffer.deviceData, key.buffer.deviceData, value.buffer.deviceData, outGradient.buffer.deviceData, logSumExp.buffer.deviceData, deltaWorkspace.buffer.deviceData, queryGradient.buffer.deviceData, keyGradient.buffer.deviceData, valueGradient.buffer.deviceData, strideColumns, columnStart, columnCount, scale, causalFlag);
    } else if (headDimension == 16 && tileBr == 32 && tileBc == 32) {
        CudaFlashAttentionBackwardKeyFixedEntry<16, 32, 32><<<grid, threadCount, keySharedBytes>>>(
            query.buffer.deviceData, key.buffer.deviceData, value.buffer.deviceData, outGradient.buffer.deviceData, logSumExp.buffer.deviceData, deltaWorkspace.buffer.deviceData, queryGradient.buffer.deviceData, keyGradient.buffer.deviceData, valueGradient.buffer.deviceData, strideColumns, columnStart, columnCount, scale, causalFlag);
    } else if (headDimension == 64 && tileBr == 16 && tileBc == 16) {
        CudaFlashAttentionBackwardKeyFixedEntry<64, 16, 16><<<grid, threadCount, keySharedBytes>>>(
            query.buffer.deviceData, key.buffer.deviceData, value.buffer.deviceData, outGradient.buffer.deviceData, logSumExp.buffer.deviceData, deltaWorkspace.buffer.deviceData, queryGradient.buffer.deviceData, keyGradient.buffer.deviceData, valueGradient.buffer.deviceData, strideColumns, columnStart, columnCount, scale, causalFlag);
    } else {
        CudaFlashAttentionBackwardKeyEntry<<<grid, threadCount, keySharedBytes>>>(
            query.buffer.deviceData, key.buffer.deviceData, value.buffer.deviceData, outGradient.buffer.deviceData, logSumExp.buffer.deviceData, deltaWorkspace.buffer.deviceData, queryGradient.buffer.deviceData, keyGradient.buffer.deviceData, valueGradient.buffer.deviceData, headDimension, strideColumns, columnStart, columnCount, scale, causalFlag, tileBr, tileBc);
    }
    CudaMatmul::throwIfCudaFailed(cudaGetLastError(), "CudaFlashAttentionBackwardEntry launch");
}

void CudaFlashAttention::backward(const CudaMatrix& query, const CudaMatrix& key, const CudaMatrix& value, const CudaMatrix& out, const CudaMatrix& logSumExp, const CudaMatrix& outGradient, CudaMatrix& queryGradient, CudaMatrix& keyGradient, CudaMatrix& valueGradient, float scale, bool causal) {
    queryGradient.ensureSize(query.rows, query.cols);
    CudaOps::zeroInPlace(queryGradient);
    CudaMatrix deltaWorkspace;
    backwardMultiHead(query, key, value, out, logSumExp, outGradient, queryGradient, keyGradient, valueGradient, deltaWorkspace, 1, static_cast<int>(query.rows), scale, causal, 0, static_cast<int>(query.cols));
}
