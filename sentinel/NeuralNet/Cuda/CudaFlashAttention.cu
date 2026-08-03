#include "CudaFlashAttention.hpp"

#include "CudaOps.hpp"

#include <algorithm>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <stdexcept>
#include <string>

namespace {

constexpr int kBrMax = CudaFlashAttention::queryTileSize;
constexpr size_t kDefaultSharedBytes = 48ull * 1024ull;

size_t deviceMaxDynamicSharedBytes() {
    static size_t cached = 0;
    if (cached != 0) return cached;
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) {
        cached = kDefaultSharedBytes;
        return cached;
    }
    cudaDeviceProp properties{};
    if (cudaGetDeviceProperties(&properties, device) != cudaSuccess) {
        cached = kDefaultSharedBytes;
        return cached;
    }
    cached = properties.sharedMemPerBlockOptin > 0
        ? static_cast<size_t>(properties.sharedMemPerBlockOptin)
        : static_cast<size_t>(properties.sharedMemPerBlock);
    if (cached < kDefaultSharedBytes)
        cached = kDefaultSharedBytes;
    return cached;
}

template <typename Kernel>
void ensureDynamicShared(Kernel* kernel, size_t sharedBytes) {
    const size_t limit = deviceMaxDynamicSharedBytes();
    if (sharedBytes > limit)
        throw std::runtime_error("CudaFlashAttention shared memory exceeds device opt-in limit");
    if (sharedBytes > kDefaultSharedBytes) {
        static bool configured = false;
        if (!configured) {
            const cudaError_t status = cudaFuncSetAttribute(
                kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(sharedBytes));
            if (status != cudaSuccess)
                throw std::runtime_error(std::string("CudaFlashAttention cudaFuncSetAttribute: ") + cudaGetErrorString(status));
            configured = true;
        }
    }
}

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
    if constexpr ((HeadDim % 4) == 0) {
#pragma unroll
        for (int dim = 0; dim < HeadDim; dim += 4) {
            const float4 a = *reinterpret_cast<const float4*>(left + dim);
            const float4 b = *reinterpret_cast<const float4*>(right + dim);
            sum += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
        }
    } else {
#pragma unroll
        for (int dim = 0; dim < HeadDim; ++dim)
            sum += left[dim] * right[dim];
    }
    return sum;
}

template <int HeadDim>
__device__ __forceinline__ float flashDotHalfFixed(const __half* left, const __half* right) {
    float sum = 0.0f;
#pragma unroll
    for (int dim = 0; dim < HeadDim; dim += 2) {
        const float2 a = __half22float2(*reinterpret_cast<const __half2*>(left + dim));
        const float2 b = __half22float2(*reinterpret_cast<const __half2*>(right + dim));
        sum += a.x * b.x + a.y * b.y;
    }
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
    const int packIndex = static_cast<int>(blockIdx.z);
    const int packColumnStart = columnStart + packIndex * sequenceLength;
    const int queryTile = static_cast<int>(blockIdx.x);
    const int queryStart = queryTile * tileBr;
    if (queryStart >= sequenceLength) return;

    const int queryCount = sequenceLength - queryStart < tileBr ? sequenceLength - queryStart : tileBr;
    const int threadIndex = static_cast<int>(threadIdx.x);
    const int threadCount = static_cast<int>(blockDim.x);

    for (int index = threadIndex; index < queryCount * headDim; index += threadCount) {
        const int localQuery = index / headDim;
        const int dim = index - localQuery * headDim;
        const int absoluteColumn = packColumnStart + queryStart + localQuery;
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
            const int absoluteColumn = packColumnStart + keyStart + localKey;
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
        const int absoluteColumn = packColumnStart + queryStart + threadIndex;
        const float inverseSum = 1.0f / rowSum;
        for (int dim = 0; dim < headDim; ++dim)
            *headRowMutable(out, headIndex, headDim, dim, strideColumns, absoluteColumn) = outputAccum[dim] * inverseSum;
        logSumExp[headIndex * strideColumns + absoluteColumn] = rowMax + __logf(rowSum);
    }
}

template <int HeadDim>
__device__ __forceinline__ void fillScoresHalfFixed(
    float* sharedScores,
    const __half* sharedQuery,
    const __half* sharedKey,
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
            score = flashDotHalfFixed<HeadDim>(sharedQuery + localQuery * HeadDim, sharedKey + localKey * HeadDim) * scale;
        sharedScores[localQuery * tileBc + localKey] = score;
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
    extern __shared__ char sharedBytes[];
    __half* sharedQuery = reinterpret_cast<__half*>(sharedBytes);
    __half* sharedKey = sharedQuery + TileBr * HeadDim;
    __half* sharedValue = sharedKey + TileBc * HeadDim;
    float* sharedScores = reinterpret_cast<float*>(sharedValue + TileBc * HeadDim);

    const int headIndex = static_cast<int>(blockIdx.y);
    const int packIndex = static_cast<int>(blockIdx.z);
    const int packColumnStart = columnStart + packIndex * sequenceLength;
    const int queryTile = static_cast<int>(blockIdx.x);
    const int queryStart = queryTile * TileBr;
    if (queryStart >= sequenceLength) return;

    const int queryCount = sequenceLength - queryStart < TileBr ? sequenceLength - queryStart : TileBr;
    const int threadIndex = static_cast<int>(threadIdx.x);
    const int threadCount = static_cast<int>(blockDim.x);

    for (int index = threadIndex; index < queryCount * HeadDim; index += threadCount) {
        const int localQuery = index / HeadDim;
        const int dim = index - localQuery * HeadDim;
        const int absoluteColumn = packColumnStart + queryStart + localQuery;
        sharedQuery[localQuery * HeadDim + dim] = __float2half_rn(*headRow(query, headIndex, HeadDim, dim, strideColumns, absoluteColumn));
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
            const int absoluteColumn = packColumnStart + keyStart + localKey;
            sharedKey[localKey * HeadDim + dim] = __float2half_rn(*headRow(key, headIndex, HeadDim, dim, strideColumns, absoluteColumn));
            sharedValue[localKey * HeadDim + dim] = __float2half_rn(*headRow(value, headIndex, HeadDim, dim, strideColumns, absoluteColumn));
        }
        __syncthreads();

        fillScoresHalfFixed<HeadDim>(sharedScores, sharedQuery, sharedKey, queryCount, keyCount, TileBc, queryStart, keyStart, scale, causal, threadIndex, threadCount);
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
            for (int dim = 0; dim < HeadDim; dim += 2) {
                float weighted0 = 0.0f;
                float weighted1 = 0.0f;
                for (int localKey = 0; localKey < keyCount; ++localKey) {
                    const float probability = sharedScores[threadIndex * TileBc + localKey];
                    const float2 val = __half22float2(*reinterpret_cast<const __half2*>(sharedValue + localKey * HeadDim + dim));
                    weighted0 += probability * val.x;
                    weighted1 += probability * val.y;
                }
                outputAccum[dim] = outputAccum[dim] * rescale + weighted0;
                outputAccum[dim + 1] = outputAccum[dim + 1] * rescale + weighted1;
            }

            rowSum = rowSum * rescale + tileSum;
            rowMax = newRowMax;
        }
        __syncthreads();
    }

    if (threadIndex < queryCount) {
        const int absoluteColumn = packColumnStart + queryStart + threadIndex;
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
    const int packIndex = static_cast<int>(blockIdx.z);
    const int packColumnStart = columnStart + packIndex * sequenceLength;
    const int deltaStride = static_cast<int>(gridDim.z) * sequenceLength;
    const int localColumn = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (localColumn >= sequenceLength) return;

    const int absoluteColumn = packColumnStart + localColumn;
    float sum = 0.0f;
    for (int dim = 0; dim < headDim; ++dim)
        sum += *headRow(out, headIndex, headDim, dim, strideColumns, absoluteColumn) * *headRow(outGradient, headIndex, headDim, dim, strideColumns, absoluteColumn);
    delta[headIndex * deltaStride + packIndex * sequenceLength + localColumn] = sum;
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
    const int packIndex = static_cast<int>(blockIdx.z);
    const int packColumnStart = columnStart + packIndex * sequenceLength;
    const int deltaStride = static_cast<int>(gridDim.z) * sequenceLength;
    const int keyTile = static_cast<int>(blockIdx.x);
    const int keyStart = keyTile * tileBc;
    if (keyStart >= sequenceLength) return;

    const int keyCount = sequenceLength - keyStart < tileBc ? sequenceLength - keyStart : tileBc;
    const int threadIndex = static_cast<int>(threadIdx.x);
    const int threadCount = static_cast<int>(blockDim.x);

    for (int index = threadIndex; index < keyCount * headDim; index += threadCount) {
        const int localKey = index / headDim;
        const int dim = index - localKey * headDim;
        const int absoluteColumn = packColumnStart + keyStart + localKey;
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
            const int absoluteColumn = packColumnStart + queryStart + localQuery;
            sharedQuery[localQuery * headDim + dim] = *headRow(query, headIndex, headDim, dim, strideColumns, absoluteColumn);
            sharedOutGrad[localQuery * headDim + dim] = *headRow(outGradient, headIndex, headDim, dim, strideColumns, absoluteColumn);
        }
        if (threadIndex < queryCount) {
            const int absoluteColumn = packColumnStart + queryStart + threadIndex;
            sharedLse[threadIndex] = logSumExp[headIndex * strideColumns + absoluteColumn];
            sharedDelta[threadIndex] = delta[headIndex * deltaStride + packIndex * sequenceLength + queryStart + threadIndex];
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
            const int absoluteColumn = packColumnStart + queryStart + threadIndex;
            for (int dim = 0; dim < headDim; ++dim)
                atomicAdd(headRowMutable(queryGradient, headIndex, headDim, dim, strideColumns, absoluteColumn), queryGradLocal[dim]);
        }
        __syncthreads();
    }

    for (int index = threadIndex; index < keyCount * headDim; index += threadCount) {
        const int localKey = index / headDim;
        const int dim = index - localKey * headDim;
        const int absoluteColumn = packColumnStart + keyStart + localKey;
        *headRowMutable(keyGradient, headIndex, headDim, dim, strideColumns, absoluteColumn) = sharedKeyGrad[localKey * headDim + dim];
        *headRowMutable(valueGradient, headIndex, headDim, dim, strideColumns, absoluteColumn) = sharedValueGrad[localKey * headDim + dim];
    }
}

/// <summary>key-tile outer: dK/dV only (atomic-free). Pair with BackwardQueryFixedEntry for dQ.</summary>
template <int HeadDim, int TileBr, int TileBc>
__global__ void CudaFlashAttentionBackwardKeyFixedEntry(
    const float* query,
    const float* key,
    const float* value,
    const float* outGradient,
    const float* logSumExp,
    const float* delta,
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
    const int packIndex = static_cast<int>(blockIdx.z);
    const int packColumnStart = columnStart + packIndex * sequenceLength;
    const int deltaStride = static_cast<int>(gridDim.z) * sequenceLength;
    const int keyTile = static_cast<int>(blockIdx.x);
    const int keyStart = keyTile * TileBc;
    if (keyStart >= sequenceLength) return;

    const int keyCount = sequenceLength - keyStart < TileBc ? sequenceLength - keyStart : TileBc;
    const int threadIndex = static_cast<int>(threadIdx.x);
    const int threadCount = static_cast<int>(blockDim.x);

    for (int index = threadIndex; index < keyCount * HeadDim; index += threadCount) {
        const int localKey = index / HeadDim;
        const int dim = index - localKey * HeadDim;
        const int absoluteColumn = packColumnStart + keyStart + localKey;
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
            const int absoluteColumn = packColumnStart + queryStart + localQuery;
            sharedQuery[localQuery * HeadDim + dim] = *headRow(query, headIndex, HeadDim, dim, strideColumns, absoluteColumn);
            sharedOutGrad[localQuery * HeadDim + dim] = *headRow(outGradient, headIndex, HeadDim, dim, strideColumns, absoluteColumn);
        }
        if (threadIndex < queryCount) {
            const int absoluteColumn = packColumnStart + queryStart + threadIndex;
            sharedLse[threadIndex] = logSumExp[headIndex * strideColumns + absoluteColumn];
            sharedDelta[threadIndex] = delta[headIndex * deltaStride + packIndex * sequenceLength + queryStart + threadIndex];
        }
        __syncthreads();

        fillProbabilitiesFixed<HeadDim>(sharedProb, sharedQuery, sharedKey, sharedLse, queryCount, keyCount, TileBc, queryStart, keyStart, scale, causal, threadIndex, threadCount);
        __syncthreads();

        if (threadIndex < keyCount) {
            const int localKey = threadIndex;
            for (int localQuery = 0; localQuery < queryCount; ++localQuery) {
                const float probability = sharedProb[localQuery * TileBc + localKey];
                if (probability == 0.0f) continue;
                float dP = 0.0f;
#pragma unroll
                for (int dim = 0; dim < HeadDim; ++dim)
                    dP += sharedOutGrad[localQuery * HeadDim + dim] * sharedValue[localKey * HeadDim + dim];
                const float scaledDs = probability * (dP - sharedDelta[localQuery]) * scale;
#pragma unroll
                for (int dim = 0; dim < HeadDim; ++dim) {
                    sharedKeyGrad[localKey * HeadDim + dim] += scaledDs * sharedQuery[localQuery * HeadDim + dim];
                    sharedValueGrad[localKey * HeadDim + dim] += probability * sharedOutGrad[localQuery * HeadDim + dim];
                }
            }
        }
        __syncthreads();
    }

    for (int index = threadIndex; index < keyCount * HeadDim; index += threadCount) {
        const int localKey = index / HeadDim;
        const int dim = index - localKey * HeadDim;
        const int absoluteColumn = packColumnStart + keyStart + localKey;
        *headRowMutable(keyGradient, headIndex, HeadDim, dim, strideColumns, absoluteColumn) = sharedKeyGrad[localKey * HeadDim + dim];
        *headRowMutable(valueGradient, headIndex, HeadDim, dim, strideColumns, absoluteColumn) = sharedValueGrad[localKey * HeadDim + dim];
    }
}

template <int HeadDim>
__device__ __forceinline__ void fillProbabilitiesHalfFixed(
    float* sharedProb,
    const __half* sharedQuery,
    const __half* sharedKey,
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
            const float score = flashDotHalfFixed<HeadDim>(sharedQuery + localQuery * HeadDim, sharedKey + localKey * HeadDim) * scale;
            probability = __expf(score - sharedLse[localQuery]);
        }
        sharedProb[localQuery * tileBc + localKey] = probability;
    }
}

/// <summary>
/// query-tile outer: exclusive dQ in registers; dK/dV via atomics (caller zeros dK/dV).
/// Q/K/V/dO tiles live in FP16 shared memory to cut smem (~50KiB @ 64) for better occupancy on sm_120.
/// </summary>
template <int HeadDim, int TileBr, int TileBc>
__global__ void CudaFlashAttentionBackwardQueryFixedEntry(
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
    extern __shared__ char sharedBytes[];
    __half* sharedKey = reinterpret_cast<__half*>(sharedBytes);
    __half* sharedValue = sharedKey + TileBc * HeadDim;
    __half* sharedQuery = sharedValue + TileBc * HeadDim;
    __half* sharedOutGrad = sharedQuery + TileBr * HeadDim;
    float* sharedProb = reinterpret_cast<float*>(sharedOutGrad + TileBr * HeadDim);
    float* sharedLse = sharedProb + TileBr * TileBc;
    float* sharedDelta = sharedLse + TileBr;

    const int headIndex = static_cast<int>(blockIdx.y);
    const int packIndex = static_cast<int>(blockIdx.z);
    const int packColumnStart = columnStart + packIndex * sequenceLength;
    const int deltaStride = static_cast<int>(gridDim.z) * sequenceLength;
    const int queryTile = static_cast<int>(blockIdx.x);
    const int queryStart = queryTile * TileBr;
    if (queryStart >= sequenceLength) return;

    const int queryCount = sequenceLength - queryStart < TileBr ? sequenceLength - queryStart : TileBr;
    const int threadIndex = static_cast<int>(threadIdx.x);
    const int threadCount = static_cast<int>(blockDim.x);

    float queryGradLocal[HeadDim];
#pragma unroll
    for (int dim = 0; dim < HeadDim; ++dim)
        queryGradLocal[dim] = 0.0f;

    for (int index = threadIndex; index < queryCount * HeadDim; index += threadCount) {
        const int localQuery = index / HeadDim;
        const int dim = index - localQuery * HeadDim;
        const int absoluteColumn = packColumnStart + queryStart + localQuery;
        sharedQuery[localQuery * HeadDim + dim] = __float2half_rn(*headRow(query, headIndex, HeadDim, dim, strideColumns, absoluteColumn));
        sharedOutGrad[localQuery * HeadDim + dim] = __float2half_rn(*headRow(outGradient, headIndex, HeadDim, dim, strideColumns, absoluteColumn));
    }
    if (threadIndex < queryCount) {
        const int absoluteColumn = packColumnStart + queryStart + threadIndex;
        sharedLse[threadIndex] = logSumExp[headIndex * strideColumns + absoluteColumn];
        sharedDelta[threadIndex] = delta[headIndex * deltaStride + packIndex * sequenceLength + queryStart + threadIndex];
    }
    __syncthreads();

    const int keyTileCount = (sequenceLength + TileBc - 1) / TileBc;
    for (int keyTile = 0; keyTile < keyTileCount; ++keyTile) {
        const int keyStart = keyTile * TileBc;
        const int keyCount = sequenceLength - keyStart < TileBc ? sequenceLength - keyStart : TileBc;
        if (causal && keyStart > (queryStart + queryCount - 1)) continue;

        for (int index = threadIndex; index < keyCount * HeadDim; index += threadCount) {
            const int localKey = index / HeadDim;
            const int dim = index - localKey * HeadDim;
            const int absoluteColumn = packColumnStart + keyStart + localKey;
            sharedKey[localKey * HeadDim + dim] = __float2half_rn(*headRow(key, headIndex, HeadDim, dim, strideColumns, absoluteColumn));
            sharedValue[localKey * HeadDim + dim] = __float2half_rn(*headRow(value, headIndex, HeadDim, dim, strideColumns, absoluteColumn));
        }
        __syncthreads();

        fillProbabilitiesHalfFixed<HeadDim>(sharedProb, sharedQuery, sharedKey, sharedLse, queryCount, keyCount, TileBc, queryStart, keyStart, scale, causal, threadIndex, threadCount);
        __syncthreads();

        if (threadIndex < keyCount) {
            const int localKey = threadIndex;
            float keyGradLocal[HeadDim];
            float valueGradLocal[HeadDim];
#pragma unroll
            for (int dim = 0; dim < HeadDim; ++dim) {
                keyGradLocal[dim] = 0.0f;
                valueGradLocal[dim] = 0.0f;
            }
            for (int localQuery = 0; localQuery < queryCount; ++localQuery) {
                const float probability = sharedProb[localQuery * TileBc + localKey];
                float scaledDs = 0.0f;
                if (probability != 0.0f) {
                    float dP = 0.0f;
#pragma unroll
                    for (int dim = 0; dim < HeadDim; dim += 2) {
                        const float2 outG = __half22float2(*reinterpret_cast<const __half2*>(sharedOutGrad + localQuery * HeadDim + dim));
                        const float2 val = __half22float2(*reinterpret_cast<const __half2*>(sharedValue + localKey * HeadDim + dim));
                        dP += outG.x * val.x + outG.y * val.y;
                    }
                    scaledDs = probability * (dP - sharedDelta[localQuery]) * scale;
#pragma unroll
                    for (int dim = 0; dim < HeadDim; dim += 2) {
                        const float2 q = __half22float2(*reinterpret_cast<const __half2*>(sharedQuery + localQuery * HeadDim + dim));
                        const float2 outG = __half22float2(*reinterpret_cast<const __half2*>(sharedOutGrad + localQuery * HeadDim + dim));
                        keyGradLocal[dim] += scaledDs * q.x;
                        keyGradLocal[dim + 1] += scaledDs * q.y;
                        valueGradLocal[dim] += probability * outG.x;
                        valueGradLocal[dim + 1] += probability * outG.y;
                    }
                }
                sharedProb[localQuery * TileBc + localKey] = scaledDs;
            }
            const int absoluteColumn = packColumnStart + keyStart + localKey;
#pragma unroll
            for (int dim = 0; dim < HeadDim; ++dim) {
                atomicAdd(headRowMutable(keyGradient, headIndex, HeadDim, dim, strideColumns, absoluteColumn), keyGradLocal[dim]);
                atomicAdd(headRowMutable(valueGradient, headIndex, HeadDim, dim, strideColumns, absoluteColumn), valueGradLocal[dim]);
            }
        }
        __syncthreads();

        if (threadIndex < queryCount) {
            const int localQuery = threadIndex;
            for (int localKey = 0; localKey < keyCount; ++localKey) {
                const float scaledDs = sharedProb[localQuery * TileBc + localKey];
                if (scaledDs == 0.0f) continue;
#pragma unroll
                for (int dim = 0; dim < HeadDim; dim += 2) {
                    const float2 k = __half22float2(*reinterpret_cast<const __half2*>(sharedKey + localKey * HeadDim + dim));
                    queryGradLocal[dim] += scaledDs * k.x;
                    queryGradLocal[dim + 1] += scaledDs * k.y;
                }
            }
        }
        __syncthreads();
    }

    if (threadIndex < queryCount) {
        const int absoluteColumn = packColumnStart + queryStart + threadIndex;
#pragma unroll
        for (int dim = 0; dim < HeadDim; ++dim)
            *headRowMutable(queryGradient, headIndex, HeadDim, dim, strideColumns, absoluteColumn) = queryGradLocal[dim];
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

size_t forwardSharedBytesHalf(int headDim, int tileBr, int tileBc) {
    const size_t halfElements = static_cast<size_t>((tileBr + 2 * tileBc) * headDim);
    const size_t floatElements = static_cast<size_t>(tileBr * tileBc);
    return halfElements * sizeof(__half) + floatElements * sizeof(float);
}

size_t forwardSharedBytesFloat(int headDim, int tileBr, int tileBc) {
    // CudaFlashAttentionForwardEntry keeps Q/K/V tiles in float shared memory.
    return static_cast<size_t>((tileBr + 2 * tileBc) * headDim + tileBr * tileBc) * sizeof(float);
}

size_t backwardKeySharedBytes(int headDim, int tileBr, int tileBc) {
    return static_cast<size_t>((2 * tileBr + 4 * tileBc) * headDim + tileBr * tileBc + 2 * tileBr) * sizeof(float);
}

size_t backwardQuerySharedBytes(int headDim, int tileBr, int tileBc) {
    // FP16 Q/K/V/dO tiles + FP32 P/LSE/Delta (dQ lives in registers)
    const size_t halfElements = static_cast<size_t>((2 * tileBr + 2 * tileBc) * headDim);
    const size_t floatElements = static_cast<size_t>(tileBr * tileBc + 2 * tileBr);
    return halfElements * sizeof(__half) + floatElements * sizeof(float);
}

void chooseTileSizes(int headDim, int& tileBr, int& tileBc) {
    // Hot path is query-outer fixed; size tiles to that kernel's shared footprint.
    const size_t sharedLimit = deviceMaxDynamicSharedBytes();
    tileBr = 8;
    tileBc = 8;
    for (int candidate = kBrMax; candidate >= 8; candidate /= 2) {
        if (backwardQuerySharedBytes(headDim, candidate, candidate) <= sharedLimit) {
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

void CudaFlashAttention::forwardMultiHead(const CudaMatrix& query, const CudaMatrix& key, const CudaMatrix& value, CudaMatrix& out, CudaMatrix& logSumExp, int headCount, int headDimension, float scale, bool causal, int columnStart, int columnCount, int packCount) {
    validateMultiHeadShape(query, key, value, headCount, headDimension, "CudaFlashAttention::forwardMultiHead");
    if (!CudaMatmul::isAvailable())
        throw std::runtime_error("CudaFlashAttention::forwardMultiHead no CUDA device");
    if (packCount <= 0)
        packCount = 1;

    resolveColumnWindow(query, columnStart, columnCount);
    if (columnStart + packCount * columnCount > static_cast<int>(query.cols))
        throw std::invalid_argument("CudaFlashAttention::forwardMultiHead pack window exceeds columns");
    const int strideColumns = static_cast<int>(query.cols);

    out.ensureSize(query.rows, query.cols);
    logSumExp.ensureSize(static_cast<size_t>(headCount), query.cols);

    int tileBr = 0;
    int tileBc = 0;
    chooseTileSizes(headDimension, tileBr, tileBc);

    const int queryTileCount = (columnCount + tileBr - 1) / tileBr;
    const dim3 grid(static_cast<unsigned>(queryTileCount), static_cast<unsigned>(headCount), static_cast<unsigned>(packCount));
    const int threadCount = chooseThreadCount(tileBr, tileBc);
    const int causalFlag = causal ? 1 : 0;

    if (headDimension == 16 && tileBr == 64 && tileBc == 64) {
        const size_t sharedBytes = forwardSharedBytesHalf(headDimension, tileBr, tileBc);
        ensureDynamicShared(CudaFlashAttentionForwardFixedEntry<16, 64, 64>, sharedBytes);
        CudaFlashAttentionForwardFixedEntry<16, 64, 64><<<grid, threadCount, sharedBytes, CudaMatmul::activeStream()>>>(
            query.buffer.deviceData, key.buffer.deviceData, value.buffer.deviceData, out.buffer.deviceData, logSumExp.buffer.deviceData, strideColumns, columnStart, columnCount, scale, causalFlag);
    } else if (headDimension == 16 && tileBr == 32 && tileBc == 32) {
        const size_t sharedBytes = forwardSharedBytesHalf(headDimension, tileBr, tileBc);
        ensureDynamicShared(CudaFlashAttentionForwardFixedEntry<16, 32, 32>, sharedBytes);
        CudaFlashAttentionForwardFixedEntry<16, 32, 32><<<grid, threadCount, sharedBytes, CudaMatmul::activeStream()>>>(
            query.buffer.deviceData, key.buffer.deviceData, value.buffer.deviceData, out.buffer.deviceData, logSumExp.buffer.deviceData, strideColumns, columnStart, columnCount, scale, causalFlag);
    } else if (headDimension == 64 && tileBr == 64 && tileBc == 64) {
        const size_t sharedBytes = forwardSharedBytesHalf(headDimension, tileBr, tileBc);
        ensureDynamicShared(CudaFlashAttentionForwardFixedEntry<64, 64, 64>, sharedBytes);
        CudaFlashAttentionForwardFixedEntry<64, 64, 64><<<grid, threadCount, sharedBytes, CudaMatmul::activeStream()>>>(
            query.buffer.deviceData, key.buffer.deviceData, value.buffer.deviceData, out.buffer.deviceData, logSumExp.buffer.deviceData, strideColumns, columnStart, columnCount, scale, causalFlag);
    } else if (headDimension == 64 && tileBr == 32 && tileBc == 32) {
        const size_t sharedBytes = forwardSharedBytesHalf(headDimension, tileBr, tileBc);
        ensureDynamicShared(CudaFlashAttentionForwardFixedEntry<64, 32, 32>, sharedBytes);
        CudaFlashAttentionForwardFixedEntry<64, 32, 32><<<grid, threadCount, sharedBytes, CudaMatmul::activeStream()>>>(
            query.buffer.deviceData, key.buffer.deviceData, value.buffer.deviceData, out.buffer.deviceData, logSumExp.buffer.deviceData, strideColumns, columnStart, columnCount, scale, causalFlag);
    } else if (headDimension == 64 && tileBr == 16 && tileBc == 16) {
        const size_t sharedBytes = forwardSharedBytesHalf(headDimension, tileBr, tileBc);
        ensureDynamicShared(CudaFlashAttentionForwardFixedEntry<64, 16, 16>, sharedBytes);
        CudaFlashAttentionForwardFixedEntry<64, 16, 16><<<grid, threadCount, sharedBytes, CudaMatmul::activeStream()>>>(
            query.buffer.deviceData, key.buffer.deviceData, value.buffer.deviceData, out.buffer.deviceData, logSumExp.buffer.deviceData, strideColumns, columnStart, columnCount, scale, causalFlag);
    } else {
        // Shrink float shared tiles until they fit device opt-in limit.
        while (tileBr > 8 && forwardSharedBytesFloat(headDimension, tileBr, tileBc) > deviceMaxDynamicSharedBytes()) {
            tileBr /= 2;
            tileBc /= 2;
        }
        const int fallbackTileCount = (columnCount + tileBr - 1) / tileBr;
        const dim3 fallbackGrid(static_cast<unsigned>(fallbackTileCount), static_cast<unsigned>(headCount), static_cast<unsigned>(packCount));
        const int fallbackThreads = chooseThreadCount(tileBr, tileBc);
        const size_t sharedBytes = forwardSharedBytesFloat(headDimension, tileBr, tileBc);
        ensureDynamicShared(CudaFlashAttentionForwardEntry, sharedBytes);
        CudaFlashAttentionForwardEntry<<<fallbackGrid, fallbackThreads, sharedBytes, CudaMatmul::activeStream()>>>(
            query.buffer.deviceData, key.buffer.deviceData, value.buffer.deviceData, out.buffer.deviceData, logSumExp.buffer.deviceData, headDimension, strideColumns, columnStart, columnCount, scale, causalFlag, tileBr, tileBc);
    }
    CudaMatmul::throwIfCudaFailed(cudaGetLastError(), "CudaFlashAttentionForwardEntry launch");
}

void CudaFlashAttention::forward(const CudaMatrix& query, const CudaMatrix& key, const CudaMatrix& value, CudaMatrix& out, CudaMatrix& logSumExp, float scale, bool causal) {
    forwardMultiHead(query, key, value, out, logSumExp, 1, static_cast<int>(query.rows), scale, causal, 0, static_cast<int>(query.cols));
}

void CudaFlashAttention::backwardMultiHead(const CudaMatrix& query, const CudaMatrix& key, const CudaMatrix& value, const CudaMatrix& out, const CudaMatrix& logSumExp, const CudaMatrix& outGradient, CudaMatrix& queryGradient, CudaMatrix& keyGradient, CudaMatrix& valueGradient, CudaMatrix& deltaWorkspace, int headCount, int headDimension, float scale, bool causal, int columnStart, int columnCount, int packCount) {
    validateMultiHeadShape(query, key, value, headCount, headDimension, "CudaFlashAttention::backwardMultiHead");
    if (out.rows != query.rows || out.cols != query.cols)
        throw std::invalid_argument("CudaFlashAttention::backwardMultiHead out shape mismatch");
    if (outGradient.rows != query.rows || outGradient.cols != query.cols)
        throw std::invalid_argument("CudaFlashAttention::backwardMultiHead outGradient shape mismatch");
    if (logSumExp.rows != static_cast<size_t>(headCount) || logSumExp.cols != query.cols)
        throw std::invalid_argument("CudaFlashAttention::backwardMultiHead logSumExp shape mismatch");
    if (!CudaMatmul::isAvailable())
        throw std::runtime_error("CudaFlashAttention::backwardMultiHead no CUDA device");
    if (packCount <= 0)
        packCount = 1;

    resolveColumnWindow(query, columnStart, columnCount);
    if (columnStart + packCount * columnCount > static_cast<int>(query.cols))
        throw std::invalid_argument("CudaFlashAttention::backwardMultiHead pack window exceeds columns");
    const int strideColumns = static_cast<int>(query.cols);

    queryGradient.ensureSize(query.rows, query.cols);
    keyGradient.ensureSize(key.rows, key.cols);
    valueGradient.ensureSize(value.rows, value.cols);

    // Fixed path: query-tile outer (exclusive dQ, atomic dK/dV). Dynamic fallback: key-tile + atomic dQ.
    deltaWorkspace.ensureSize(static_cast<size_t>(headCount), static_cast<size_t>(packCount) * static_cast<size_t>(columnCount));

    const int deltaThreads = 128;
    const int deltaBlocksX = (columnCount + deltaThreads - 1) / deltaThreads;
    const dim3 deltaGrid(static_cast<unsigned>(deltaBlocksX), static_cast<unsigned>(headCount), static_cast<unsigned>(packCount));
    CudaFlashAttentionDeltaEntry<<<deltaGrid, deltaThreads, 0, CudaMatmul::activeStream()>>>(
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

    const int queryTileCount = (columnCount + tileBr - 1) / tileBr;
    const dim3 queryGrid(static_cast<unsigned>(queryTileCount), static_cast<unsigned>(headCount), static_cast<unsigned>(packCount));
    const int threadCount = chooseThreadCount(tileBr, tileBc);
    const size_t querySharedBytes = backwardQuerySharedBytes(headDimension, tileBr, tileBc);
    const int causalFlag = causal ? 1 : 0;

    auto launchQueryFixed = [&](auto queryKernel) {
        ensureDynamicShared(queryKernel, querySharedBytes);
        queryKernel<<<queryGrid, threadCount, querySharedBytes, CudaMatmul::activeStream()>>>(
            query.buffer.deviceData, key.buffer.deviceData, value.buffer.deviceData, outGradient.buffer.deviceData, logSumExp.buffer.deviceData, deltaWorkspace.buffer.deviceData, queryGradient.buffer.deviceData, keyGradient.buffer.deviceData, valueGradient.buffer.deviceData, strideColumns, columnStart, columnCount, scale, causalFlag);
        CudaMatmul::throwIfCudaFailed(cudaGetLastError(), "CudaFlashAttentionBackwardQueryFixedEntry launch");
    };

    if (headDimension == 16 && tileBr == 64 && tileBc == 64) {
        launchQueryFixed(CudaFlashAttentionBackwardQueryFixedEntry<16, 64, 64>);
    } else if (headDimension == 16 && tileBr == 32 && tileBc == 32) {
        launchQueryFixed(CudaFlashAttentionBackwardQueryFixedEntry<16, 32, 32>);
    } else if (headDimension == 64 && tileBr == 64 && tileBc == 64) {
        launchQueryFixed(CudaFlashAttentionBackwardQueryFixedEntry<64, 64, 64>);
    } else if (headDimension == 64 && tileBr == 32 && tileBc == 32) {
        launchQueryFixed(CudaFlashAttentionBackwardQueryFixedEntry<64, 32, 32>);
    } else if (headDimension == 64 && tileBr == 16 && tileBc == 16) {
        launchQueryFixed(CudaFlashAttentionBackwardQueryFixedEntry<64, 16, 16>);
    } else {
        // Shrink until the key-outer fallback fits in 48KiB shared.
        while (tileBr > 8 && backwardKeySharedBytes(headDimension, tileBr, tileBc) > deviceMaxDynamicSharedBytes()) {
            tileBr /= 2;
            tileBc /= 2;
        }
        const int keyTileCount = (columnCount + tileBc - 1) / tileBc;
        const dim3 keyGrid(static_cast<unsigned>(keyTileCount), static_cast<unsigned>(headCount), static_cast<unsigned>(packCount));
        const int fallbackThreads = chooseThreadCount(tileBr, tileBc);
        const size_t keySharedBytes = backwardKeySharedBytes(headDimension, tileBr, tileBc);
        ensureDynamicShared(CudaFlashAttentionBackwardKeyEntry, keySharedBytes);
        CudaFlashAttentionBackwardKeyEntry<<<keyGrid, fallbackThreads, keySharedBytes, CudaMatmul::activeStream()>>>(
            query.buffer.deviceData, key.buffer.deviceData, value.buffer.deviceData, outGradient.buffer.deviceData, logSumExp.buffer.deviceData, deltaWorkspace.buffer.deviceData, queryGradient.buffer.deviceData, keyGradient.buffer.deviceData, valueGradient.buffer.deviceData, headDimension, strideColumns, columnStart, columnCount, scale, causalFlag, tileBr, tileBc);
        CudaMatmul::throwIfCudaFailed(cudaGetLastError(), "CudaFlashAttentionBackwardEntry launch");
    }
}

void CudaFlashAttention::backward(const CudaMatrix& query, const CudaMatrix& key, const CudaMatrix& value, const CudaMatrix& out, const CudaMatrix& logSumExp, const CudaMatrix& outGradient, CudaMatrix& queryGradient, CudaMatrix& keyGradient, CudaMatrix& valueGradient, float scale, bool causal) {
    queryGradient.ensureSize(query.rows, query.cols);
    keyGradient.ensureSize(key.rows, key.cols);
    valueGradient.ensureSize(value.rows, value.cols);
    CudaOps::zeroInPlace(queryGradient);
    CudaOps::zeroInPlace(keyGradient);
    CudaOps::zeroInPlace(valueGradient);
    CudaMatrix deltaWorkspace;
    backwardMultiHead(query, key, value, out, logSumExp, outGradient, queryGradient, keyGradient, valueGradient, deltaWorkspace, 1, static_cast<int>(query.rows), scale, causal, 0, static_cast<int>(query.cols));
}
