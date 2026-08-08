#include "CudaSPULSE.hpp"

#include "CudaMatmul.hpp"
#include "CudaOps.hpp"
#include "CudaTransformerBlock.hpp"
#include "../Utils/SmokeLog.hpp"

#include <algorithm>
#include <cmath>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace {

void throwIfFailed(cudaError_t status, const char* what) {
    if (status == cudaSuccess) return;
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
}

int elementwiseBlocks(int elementCount, int threads = 256) {
    return (elementCount + threads - 1) / threads;
}

float clipScale(float scale, float scaleMin, float scaleMax) {
    if (scale < scaleMin) return scaleMin;
    if (scale > scaleMax) return scaleMax;
    return scale;
}

float computeScale(float energyFast, float energySlow, float epsilon, float scaleMin, float scaleMax) {
    return clipScale(std::sqrt(energySlow / (energyFast + epsilon)), scaleMin, scaleMax);
}

__device__ __forceinline__ void spulseCommitEnergyDevice(
    float* energy,
    float g2,
    float fastBeta,
    float slowBeta,
    float oneMinusFast,
    float oneMinusSlow,
    float epsilon,
    float scaleMin,
    float scaleMax
) {
    float eFast = energy[0];
    float eSlow = energy[1];
    eFast = fastBeta * eFast + oneMinusFast * g2;
    eSlow = slowBeta * eSlow + oneMinusSlow * g2;
    energy[0] = eFast;
    energy[1] = eSlow;
    float scale = sqrtf(eSlow / (eFast + epsilon));
    if (scale < scaleMin) scale = scaleMin;
    if (scale > scaleMax) scale = scaleMax;
    energy[2] = scale;
}

/// <summary>Block-reduce local sum, atomic into sumSquares; last grid block commits energy (no &lt;&lt;&lt;1,1&gt;&gt;&gt;).</summary>
__device__ __forceinline__ void spulseFinishEnergy(
    float* shared,
    float* sumSquares,
    int* blocksDone,
    float* energy,
    float fastBeta,
    float slowBeta,
    float oneMinusFast,
    float oneMinusSlow,
    float epsilon,
    float scaleMin,
    float scaleMax
) {
    __syncthreads();
    for (int stride = static_cast<int>(blockDim.x) / 2; stride > 0; stride >>= 1) {
        if (static_cast<int>(threadIdx.x) < stride)
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(sumSquares, shared[0]);
        __threadfence();
        const int done = atomicAdd(blocksDone, 1);
        if (done == static_cast<int>(gridDim.x) - 1) {
            spulseCommitEnergyDevice(
                energy,
                *sumSquares,
                fastBeta,
                slowBeta,
                oneMinusFast,
                oneMinusSlow,
                epsilon,
                scaleMin,
                scaleMax);
        }
    }
}

__global__ void spulseFusedStep(
    float* parameter,
    float* momentum,
    const float* gradient,
    float* energy,
    float* sumSquares,
    int* blocksDone,
    int elementCount,
    float momentumBeta,
    float oneMinusMom,
    float gradientScale,
    float learningRate,
    float keep,
    float momCorrection,
    float fastBeta,
    float slowBeta,
    float oneMinusFast,
    float oneMinusSlow,
    float epsilon,
    float scaleMin,
    float scaleMax
) {
    __shared__ float shared[256];
    const float stepScale = energy[2] * momCorrection;
    const float lrStep = learningRate * stepScale;
    float local = 0.0f;

    const int vecCount = elementCount >> 2;
    float4* __restrict param4 = reinterpret_cast<float4*>(parameter);
    float4* __restrict mom4 = reinterpret_cast<float4*>(momentum);
    const float4* __restrict grad4 = reinterpret_cast<const float4*>(gradient);

    for (int v = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
         v < vecCount;
         v += static_cast<int>(blockDim.x * gridDim.x)) {
        float4 g4 = grad4[v];
        float4 m4 = mom4[v];
        float4 p4 = param4[v];

        float g = g4.x * gradientScale;
        float m = momentumBeta * m4.x + oneMinusMom * g;
        m4.x = m;
        if (keep != 1.0f) p4.x *= keep;
        p4.x -= lrStep * m;
        local += g * g;

        g = g4.y * gradientScale;
        m = momentumBeta * m4.y + oneMinusMom * g;
        m4.y = m;
        if (keep != 1.0f) p4.y *= keep;
        p4.y -= lrStep * m;
        local += g * g;

        g = g4.z * gradientScale;
        m = momentumBeta * m4.z + oneMinusMom * g;
        m4.z = m;
        if (keep != 1.0f) p4.z *= keep;
        p4.z -= lrStep * m;
        local += g * g;

        g = g4.w * gradientScale;
        m = momentumBeta * m4.w + oneMinusMom * g;
        m4.w = m;
        if (keep != 1.0f) p4.w *= keep;
        p4.w -= lrStep * m;
        local += g * g;

        mom4[v] = m4;
        param4[v] = p4;
    }

    for (int index = (vecCount << 2) + static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
         index < elementCount;
         index += static_cast<int>(blockDim.x * gridDim.x)) {
        const float g = gradient[index] * gradientScale;
        const float m = momentumBeta * momentum[index] + oneMinusMom * g;
        momentum[index] = m;
        float value = parameter[index];
        if (keep != 1.0f)
            value *= keep;
        parameter[index] = value - lrStep * m;
        local += g * g;
    }

    shared[threadIdx.x] = local;
    spulseFinishEnergy(
        shared, sumSquares, blocksDone, energy,
        fastBeta, slowBeta, oneMinusFast, oneMinusSlow, epsilon, scaleMin, scaleMax);
}

__global__ void spulsePrepareHostDelta(
    float* gradientOrDelta,
    float* momentum,
    float* energy,
    float* sumSquares,
    int* blocksDone,
    int elementCount,
    float momentumBeta,
    float oneMinusMom,
    float gradientScale,
    float learningRate,
    float momCorrection,
    float fastBeta,
    float slowBeta,
    float oneMinusFast,
    float oneMinusSlow,
    float epsilon,
    float scaleMin,
    float scaleMax
) {
    __shared__ float shared[256];
    const float stepScale = energy[2] * momCorrection;
    const float lrStep = learningRate * stepScale;
    float local = 0.0f;

    const int vecCount = elementCount >> 2;
    float4* __restrict delta4 = reinterpret_cast<float4*>(gradientOrDelta);
    float4* __restrict mom4 = reinterpret_cast<float4*>(momentum);

    for (int v = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
         v < vecCount;
         v += static_cast<int>(blockDim.x * gridDim.x)) {
        float4 g4 = delta4[v];
        float4 m4 = mom4[v];

        float g = g4.x * gradientScale;
        float m = momentumBeta * m4.x + oneMinusMom * g;
        m4.x = m;
        g4.x = lrStep * m;
        local += g * g;

        g = g4.y * gradientScale;
        m = momentumBeta * m4.y + oneMinusMom * g;
        m4.y = m;
        g4.y = lrStep * m;
        local += g * g;

        g = g4.z * gradientScale;
        m = momentumBeta * m4.z + oneMinusMom * g;
        m4.z = m;
        g4.z = lrStep * m;
        local += g * g;

        g = g4.w * gradientScale;
        m = momentumBeta * m4.w + oneMinusMom * g;
        m4.w = m;
        g4.w = lrStep * m;
        local += g * g;

        mom4[v] = m4;
        delta4[v] = g4;
    }

    for (int index = (vecCount << 2) + static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
         index < elementCount;
         index += static_cast<int>(blockDim.x * gridDim.x)) {
        const float g = gradientOrDelta[index] * gradientScale;
        const float m = momentumBeta * momentum[index] + oneMinusMom * g;
        momentum[index] = m;
        gradientOrDelta[index] = lrStep * m;
        local += g * g;
    }

    shared[threadIdx.x] = local;
    spulseFinishEnergy(
        shared, sumSquares, blocksDone, energy,
        fastBeta, slowBeta, oneMinusFast, oneMinusSlow, epsilon, scaleMin, scaleMax);
}

/// <summary>one Hybrid weight for batched host-delta prepare (FP32 u)</summary>
struct SpulseHostDeltaPiece {
    float* gradOrDelta;
    float* momentum;
    float* energy;
    float* sumSquares;
    int elementCount;
};

/// <summary>
/// All Hybrid pieces in one launch: grid (blocksX, pieceCount).
/// No per-block threadfence — energy committed by a tiny follow-up kernel.
/// </summary>
__global__ void spulsePrepareHostDeltaBatched(
    const SpulseHostDeltaPiece* pieces,
    int pieceCount,
    float momentumBeta,
    float oneMinusMom,
    float gradientScale,
    float learningRate,
    float momCorrection
) {
    const int pieceIndex = static_cast<int>(blockIdx.y);
    if (pieceIndex >= pieceCount) return;
    const SpulseHostDeltaPiece piece = pieces[pieceIndex];
    if (piece.gradOrDelta == nullptr || piece.momentum == nullptr || piece.elementCount <= 0) return;

    __shared__ float shared[256];
    __shared__ float lrStepShared;
    // One energy[2] load per block (lagged scale); energy commit is a separate kernel.
    if (threadIdx.x == 0)
        lrStepShared = learningRate * (piece.energy[2] * momCorrection);
    __syncthreads();
    const float lrStep = lrStepShared;
    float local = 0.0f;

    const int elementCount = piece.elementCount;
    const int vecCount = elementCount >> 2;
    const bool useVec = vecCount > 0
        && (reinterpret_cast<uintptr_t>(piece.gradOrDelta) % 16u) == 0
        && (reinterpret_cast<uintptr_t>(piece.momentum) % 16u) == 0;

    if (useVec) {
        float4* __restrict delta4 = reinterpret_cast<float4*>(piece.gradOrDelta);
        float4* __restrict mom4 = reinterpret_cast<float4*>(piece.momentum);
        for (int v = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
             v < vecCount;
             v += static_cast<int>(blockDim.x * gridDim.x)) {
            float4 g4 = delta4[v];
            float4 m4 = mom4[v];

            float g = g4.x * gradientScale;
            float m = momentumBeta * m4.x + oneMinusMom * g;
            m4.x = m;
            g4.x = lrStep * m;
            local += g * g;

            g = g4.y * gradientScale;
            m = momentumBeta * m4.y + oneMinusMom * g;
            m4.y = m;
            g4.y = lrStep * m;
            local += g * g;

            g = g4.z * gradientScale;
            m = momentumBeta * m4.z + oneMinusMom * g;
            m4.z = m;
            g4.z = lrStep * m;
            local += g * g;

            g = g4.w * gradientScale;
            m = momentumBeta * m4.w + oneMinusMom * g;
            m4.w = m;
            g4.w = lrStep * m;
            local += g * g;

            mom4[v] = m4;
            delta4[v] = g4;
        }
        for (int index = (vecCount << 2) + static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
             index < elementCount;
             index += static_cast<int>(blockDim.x * gridDim.x)) {
            const float g = piece.gradOrDelta[index] * gradientScale;
            const float m = momentumBeta * piece.momentum[index] + oneMinusMom * g;
            piece.momentum[index] = m;
            piece.gradOrDelta[index] = lrStep * m;
            local += g * g;
        }
    } else {
        for (int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
             index < elementCount;
             index += static_cast<int>(blockDim.x * gridDim.x)) {
            const float g = piece.gradOrDelta[index] * gradientScale;
            const float m = momentumBeta * piece.momentum[index] + oneMinusMom * g;
            piece.momentum[index] = m;
            piece.gradOrDelta[index] = lrStep * m;
            local += g * g;
        }
    }

    shared[threadIdx.x] = local;
    __syncthreads();
    for (int stride = static_cast<int>(blockDim.x) / 2; stride > 0; stride >>= 1) {
        if (static_cast<int>(threadIdx.x) < stride)
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        atomicAdd(piece.sumSquares, shared[0]);
}

__global__ void spulseCommitEnergyMany(
    const SpulseHostDeltaPiece* pieces,
    int pieceCount,
    float fastBeta,
    float slowBeta,
    float oneMinusFast,
    float oneMinusSlow,
    float epsilon,
    float scaleMin,
    float scaleMax
) {
    const int pieceIndex = static_cast<int>(threadIdx.x);
    if (pieceIndex >= pieceCount) return;
    const SpulseHostDeltaPiece piece = pieces[pieceIndex];
    if (piece.energy == nullptr || piece.sumSquares == nullptr) return;
    spulseCommitEnergyDevice(
        piece.energy,
        *piece.sumSquares,
        fastBeta,
        slowBeta,
        oneMinusFast,
        oneMinusSlow,
        epsilon,
        scaleMin,
        scaleMax);
}

__global__ void spulseFusedStepHalf(
    float* parameter,
    __half* momentum,
    const float* gradient,
    float* energy,
    float* sumSquares,
    int* blocksDone,
    int elementCount,
    float momentumBeta,
    float oneMinusMom,
    float gradientScale,
    float learningRate,
    float keep,
    float momCorrection,
    float fastBeta,
    float slowBeta,
    float oneMinusFast,
    float oneMinusSlow,
    float epsilon,
    float scaleMin,
    float scaleMax
) {
    __shared__ float shared[256];
    const float stepScale = energy[2] * momCorrection;
    const float lrStep = learningRate * stepScale;
    float local = 0.0f;

    const int vecCount = elementCount >> 2;
    float4* __restrict param4 = reinterpret_cast<float4*>(parameter);
    const float4* __restrict grad4 = reinterpret_cast<const float4*>(gradient);
    __half2* __restrict mom2 = reinterpret_cast<__half2*>(momentum);

    for (int v = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
         v < vecCount;
         v += static_cast<int>(blockDim.x * gridDim.x)) {
        float4 g4 = grad4[v];
        float4 p4 = param4[v];
        __half2 m01 = mom2[v * 2];
        __half2 m23 = mom2[v * 2 + 1];
        float2 m01f = __half22float2(m01);
        float2 m23f = __half22float2(m23);

        float g = g4.x * gradientScale;
        float m = momentumBeta * m01f.x + oneMinusMom * g;
        if (!isfinite(m)) m = 0.0f;
        m01f.x = m;
        if (keep != 1.0f) p4.x *= keep;
        p4.x -= lrStep * m;
        local += g * g;

        g = g4.y * gradientScale;
        m = momentumBeta * m01f.y + oneMinusMom * g;
        if (!isfinite(m)) m = 0.0f;
        m01f.y = m;
        if (keep != 1.0f) p4.y *= keep;
        p4.y -= lrStep * m;
        local += g * g;

        g = g4.z * gradientScale;
        m = momentumBeta * m23f.x + oneMinusMom * g;
        if (!isfinite(m)) m = 0.0f;
        m23f.x = m;
        if (keep != 1.0f) p4.z *= keep;
        p4.z -= lrStep * m;
        local += g * g;

        g = g4.w * gradientScale;
        m = momentumBeta * m23f.y + oneMinusMom * g;
        if (!isfinite(m)) m = 0.0f;
        m23f.y = m;
        if (keep != 1.0f) p4.w *= keep;
        p4.w -= lrStep * m;
        local += g * g;

        mom2[v * 2] = __float22half2_rn(m01f);
        mom2[v * 2 + 1] = __float22half2_rn(m23f);
        param4[v] = p4;
    }

    for (int index = (vecCount << 2) + static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
         index < elementCount;
         index += static_cast<int>(blockDim.x * gridDim.x)) {
        const float g = gradient[index] * gradientScale;
        float m = momentumBeta * __half2float(momentum[index]) + oneMinusMom * g;
        if (!isfinite(m)) m = 0.0f;
        momentum[index] = __float2half_rn(m);
        float value = parameter[index];
        if (keep != 1.0f)
            value *= keep;
        parameter[index] = value - lrStep * m;
        local += g * g;
    }

    shared[threadIdx.x] = local;
    spulseFinishEnergy(
        shared, sumSquares, blocksDone, energy,
        fastBeta, slowBeta, oneMinusFast, oneMinusSlow, epsilon, scaleMin, scaleMax);
}

__global__ void spulsePrepareHostDeltaHalf(
    float* gradientOrDelta,
    __half* momentum,
    float* energy,
    float* sumSquares,
    int* blocksDone,
    int elementCount,
    float momentumBeta,
    float oneMinusMom,
    float gradientScale,
    float learningRate,
    float momCorrection,
    float fastBeta,
    float slowBeta,
    float oneMinusFast,
    float oneMinusSlow,
    float epsilon,
    float scaleMin,
    float scaleMax
) {
    __shared__ float shared[256];
    const float stepScale = energy[2] * momCorrection;
    const float lrStep = learningRate * stepScale;
    float local = 0.0f;

    const int vecCount = elementCount >> 2;
    float4* __restrict delta4 = reinterpret_cast<float4*>(gradientOrDelta);
    __half2* __restrict mom2 = reinterpret_cast<__half2*>(momentum);

    for (int v = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
         v < vecCount;
         v += static_cast<int>(blockDim.x * gridDim.x)) {
        float4 g4 = delta4[v];
        __half2 m01 = mom2[v * 2];
        __half2 m23 = mom2[v * 2 + 1];
        float2 m01f = __half22float2(m01);
        float2 m23f = __half22float2(m23);

        float g = g4.x * gradientScale;
        float m = momentumBeta * m01f.x + oneMinusMom * g;
        if (!isfinite(m)) m = 0.0f;
        m01f.x = m;
        g4.x = lrStep * m;
        local += g * g;

        g = g4.y * gradientScale;
        m = momentumBeta * m01f.y + oneMinusMom * g;
        if (!isfinite(m)) m = 0.0f;
        m01f.y = m;
        g4.y = lrStep * m;
        local += g * g;

        g = g4.z * gradientScale;
        m = momentumBeta * m23f.x + oneMinusMom * g;
        if (!isfinite(m)) m = 0.0f;
        m23f.x = m;
        g4.z = lrStep * m;
        local += g * g;

        g = g4.w * gradientScale;
        m = momentumBeta * m23f.y + oneMinusMom * g;
        if (!isfinite(m)) m = 0.0f;
        m23f.y = m;
        g4.w = lrStep * m;
        local += g * g;

        mom2[v * 2] = __float22half2_rn(m01f);
        mom2[v * 2 + 1] = __float22half2_rn(m23f);
        delta4[v] = g4;
    }

    for (int index = (vecCount << 2) + static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
         index < elementCount;
         index += static_cast<int>(blockDim.x * gridDim.x)) {
        const float g = gradientOrDelta[index] * gradientScale;
        float m = momentumBeta * __half2float(momentum[index]) + oneMinusMom * g;
        if (!isfinite(m)) m = 0.0f;
        momentum[index] = __float2half_rn(m);
        gradientOrDelta[index] = lrStep * m;
        local += g * g;
    }

    shared[threadIdx.x] = local;
    spulseFinishEnergy(
        shared, sumSquares, blocksDone, energy,
        fastBeta, slowBeta, oneMinusFast, oneMinusSlow, epsilon, scaleMin, scaleMax);
}

/// <summary>One CUDA block = one absmax quant block. writeDelta=true → host-delta path (no θ update).</summary>
__global__ void spulseStepInt8(
    float* parameterOrDelta,
    signed char* momentumQ,
    float* momentumScales,
    const float* gradient,
    float* energy,
    float* sumSquares,
    int* blocksDone,
    int elementCount,
    int blockSize,
    float momentumBeta,
    float oneMinusMom,
    float gradientScale,
    float learningRate,
    float keep,
    float momCorrection,
    int writeDelta,
    float fastBeta,
    float slowBeta,
    float oneMinusFast,
    float oneMinusSlow,
    float epsilon,
    float scaleMin,
    float scaleMax
) {
    const int quantBlock = static_cast<int>(blockIdx.x);
    const int start = quantBlock * blockSize;
    if (start >= elementCount) return;
    const int count = (start + blockSize <= elementCount) ? blockSize : (elementCount - start);
    const int threadCount = static_cast<int>(blockDim.x);

    extern __shared__ float shared[];
    float* momFp = shared;
    float* reduceMax = shared + blockSize;
    float* reduceSum = reduceMax + threadCount;

    const float oldScale = momentumScales[quantBlock];
    const float stepScale = energy[2] * momCorrection;
    float localSum = 0.0f;

    for (int local = static_cast<int>(threadIdx.x); local < count; local += threadCount) {
        const int index = start + local;
        float m = (oldScale == 0.0f) ? 0.0f : static_cast<float>(momentumQ[index]) * oldScale;
        if (!isfinite(m)) m = 0.0f;
        const float g = (gradient != nullptr ? gradient[index] : parameterOrDelta[index]) * gradientScale;
        m = momentumBeta * m + oneMinusMom * g;
        if (!isfinite(m)) m = 0.0f;
        momFp[local] = m;
        localSum += g * g;

        if (writeDelta) {
            parameterOrDelta[index] = learningRate * stepScale * m;
        } else {
            float value = parameterOrDelta[index];
            if (keep != 1.0f)
                value *= keep;
            parameterOrDelta[index] = value - learningRate * stepScale * m;
        }
    }
    reduceSum[threadIdx.x] = localSum;

    float localMax = 0.0f;
    for (int local = static_cast<int>(threadIdx.x); local < count; local += threadCount)
        localMax = fmaxf(localMax, fabsf(momFp[local]));
    reduceMax[threadIdx.x] = localMax;
    __syncthreads();

    for (int stride = threadCount / 2; stride > 0; stride >>= 1) {
        if (static_cast<int>(threadIdx.x) < stride) {
            reduceMax[threadIdx.x] = fmaxf(reduceMax[threadIdx.x], reduceMax[threadIdx.x + stride]);
            reduceSum[threadIdx.x] += reduceSum[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(sumSquares, reduceSum[0]);
        const float absmax = reduceMax[0];
        const float newScale = (absmax > 0.0f) ? (absmax / 127.0f) : 0.0f;
        momentumScales[quantBlock] = newScale;
        __threadfence();
        const int done = atomicAdd(blocksDone, 1);
        if (done == static_cast<int>(gridDim.x) - 1) {
            spulseCommitEnergyDevice(
                energy,
                *sumSquares,
                fastBeta,
                slowBeta,
                oneMinusFast,
                oneMinusSlow,
                epsilon,
                scaleMin,
                scaleMax);
        }
    }
    __syncthreads();

    const float newScale = momentumScales[quantBlock];
    for (int local = static_cast<int>(threadIdx.x); local < count; local += threadCount) {
        float quantized = (newScale == 0.0f) ? 0.0f : rintf(momFp[local] / newScale);
        quantized = fminf(127.0f, fmaxf(-127.0f, quantized));
        if (!isfinite(quantized)) quantized = 0.0f;
        momentumQ[start + local] = static_cast<signed char>(quantized);
    }
}

void hostUpdateCore(
    float* parameter,
    float* momentum,
    float& energyFast,
    float& energySlow,
    float& scale,
    const float* gradient,
    size_t elementCount,
    float learningRate,
    float momentumBeta,
    float fastBeta,
    float slowBeta,
    float epsilon,
    float scaleMin,
    float scaleMax,
    float weightDecay,
    float gradientScale,
    float momCorrection
) {
    const float oneMinusMom = 1.0f - momentumBeta;
    const float oneMinusFast = 1.0f - fastBeta;
    const float oneMinusSlow = 1.0f - slowBeta;
    const float stepScale = scale * momCorrection;
    float sumSquares = 0.0f;
    const float keep = 1.0f - learningRate * weightDecay;
    const float step = learningRate * stepScale;
    for (size_t i = 0; i < elementCount; ++i) {
        const float g = gradient[i] * gradientScale;
        momentum[i] = momentumBeta * momentum[i] + oneMinusMom * g;
        sumSquares += g * g;
        float value = parameter[i];
        if (keep != 1.0f)
            value *= keep;
        parameter[i] = value - step * momentum[i];
    }
    energyFast = fastBeta * energyFast + oneMinusFast * sumSquares;
    energySlow = slowBeta * energySlow + oneMinusSlow * sumSquares;
    scale = computeScale(energyFast, energySlow, epsilon, scaleMin, scaleMax);
}

void hostUpdateFromHalfCore(
    float* parameter,
    float* momentum,
    float& energyFast,
    float& energySlow,
    float& scale,
    const __half* gradHalf,
    size_t elementCount,
    float learningRate,
    float momentumBeta,
    float fastBeta,
    float slowBeta,
    float epsilon,
    float scaleMin,
    float scaleMax,
    float weightDecay,
    float gradientScale,
    float momCorrection
) {
    const float oneMinusMom = 1.0f - momentumBeta;
    const float oneMinusFast = 1.0f - fastBeta;
    const float oneMinusSlow = 1.0f - slowBeta;
    const float stepScale = scale * momCorrection;
    float sumSquares = 0.0f;
    const float keep = 1.0f - learningRate * weightDecay;
    const float step = learningRate * stepScale;
    for (size_t i = 0; i < elementCount; ++i) {
        const float g = gradientScale * __half2float(gradHalf[i]);
        momentum[i] = momentumBeta * momentum[i] + oneMinusMom * g;
        sumSquares += g * g;
        float value = parameter[i];
        if (keep != 1.0f)
            value *= keep;
        parameter[i] = value - step * momentum[i];
    }
    energyFast = fastBeta * energyFast + oneMinusFast * sumSquares;
    energySlow = slowBeta * energySlow + oneMinusSlow * sumSquares;
    scale = computeScale(energyFast, energySlow, epsilon, scaleMin, scaleMax);
}

/// <summary>
/// HostSGD-shaped loop: master -= lr * scale * g. No momentum buffer.
/// Extra work vs HostSGD: accumulate ||g||² + update dual-horizon scale for next step.
/// </summary>
void hostUpdateFromHalfLite(
    float* parameter,
    float& energyFast,
    float& energySlow,
    float& scale,
    const __half* gradHalf,
    size_t elementCount,
    float learningRate,
    float fastBeta,
    float slowBeta,
    float epsilon,
    float scaleMin,
    float scaleMax,
    float weightDecay,
    float gradientScale
) {
    const float oneMinusFast = 1.0f - fastBeta;
    const float oneMinusSlow = 1.0f - slowBeta;
    const float step = learningRate * scale;
    const float keep = 1.0f - learningRate * weightDecay;
    float sumSquares = 0.0f;
    for (size_t i = 0; i < elementCount; ++i) {
        const float g = gradientScale * __half2float(gradHalf[i]);
        sumSquares += g * g;
        float value = parameter[i];
        if (keep != 1.0f)
            value *= keep;
        parameter[i] = value - step * g;
    }
    energyFast = fastBeta * energyFast + oneMinusFast * sumSquares;
    energySlow = slowBeta * energySlow + oneMinusSlow * sumSquares;
    scale = computeScale(energyFast, energySlow, epsilon, scaleMin, scaleMax);
}

} // namespace

bool CudaSpulseState::empty() const {
    return this->rows == 0 || this->cols == 0;
}

void CudaSpulseState::ensure(
    size_t rows,
    size_t cols,
    SpulseMomentumStorage storage,
    int int8BlockSize
) {
    if (rows == 0 || cols == 0) throw std::invalid_argument("CudaSpulseState::ensure empty shape");
    if (int8BlockSize <= 0) throw std::invalid_argument("CudaSpulseState::ensure int8BlockSize must be > 0");

    const size_t elementCount = rows * cols;
    const bool sameShape = this->rows == rows && this->cols == cols && this->energy.deviceData != nullptr
        && this->storage == storage
        && (storage != SpulseMomentumStorage::Int8 || this->int8BlockSize == int8BlockSize);
    if (sameShape) {
        if (storage == SpulseMomentumStorage::Fp32 && this->momentum.hasDeviceStorage()) return;
        if (storage == SpulseMomentumStorage::Fp16 && this->momentumHalf.deviceData != nullptr) return;
        if (storage == SpulseMomentumStorage::Int8 && this->momentumQ.deviceData != nullptr) return;
    }

    this->free();
    this->rows = rows;
    this->cols = cols;
    this->storage = storage;
    this->int8BlockSize = int8BlockSize;
    this->scaleCount = 0;

    if (storage == SpulseMomentumStorage::Fp32) {
        this->momentum.ensureSize(rows, cols);
        CudaOps::zeroInPlace(this->momentum);
    } else if (storage == SpulseMomentumStorage::Fp16) {
        this->momentumHalf.ensureCapacity(elementCount * sizeof(__half));
        throwIfFailed(
            cudaMemset(this->momentumHalf.deviceData, 0, elementCount * sizeof(__half)),
            "CudaSpulseState::ensure momentumHalf zero");
    } else {
        this->scaleCount = static_cast<int>((elementCount + static_cast<size_t>(int8BlockSize) - 1)
            / static_cast<size_t>(int8BlockSize));
        this->momentumQ.ensureCapacity(elementCount);
        this->momentumQ.zeroInPlace();
        this->momentumScales.ensureCapacity(static_cast<size_t>(this->scaleCount) * sizeof(float));
        throwIfFailed(
            cudaMemset(this->momentumScales.deviceData, 0, static_cast<size_t>(this->scaleCount) * sizeof(float)),
            "CudaSpulseState::ensure momentumScales zero");
    }

    this->energy.ensureCapacity(3 * sizeof(float));
    const float init[3] = {0.0f, 0.0f, 1.0f};
    throwIfFailed(
        cudaMemcpy(this->energy.deviceData, init, 3 * sizeof(float), cudaMemcpyHostToDevice),
        "CudaSpulseState::ensure energy init");
}

void CudaSpulseState::ensure(
    const CudaMatrix& parameter,
    SpulseMomentumStorage storage,
    int int8BlockSize
) {
    if (parameter.empty()) throw std::invalid_argument("CudaSpulseState::ensure empty parameter");
    this->ensure(parameter.rows, parameter.cols, storage, int8BlockSize);
}

void CudaSpulseState::free() {
    this->momentum.free();
    this->momentumHalf.free();
    this->momentumQ.free();
    this->momentumScales.free();
    this->energy.free();
    this->rows = 0;
    this->cols = 0;
    this->scaleCount = 0;
}

void CudaSpulseState::downloadInto(SpulseState& host) const {
    if (this->empty()) {
        host.clear();
        return;
    }
    host.momentum.ensureSize(this->rows, this->cols);
    const size_t elementCount = this->rows * this->cols;

    if (this->storage == SpulseMomentumStorage::Fp32) {
        this->momentum.downloadInto(host.momentum);
    } else if (this->storage == SpulseMomentumStorage::Fp16) {
        std::vector<__half> halfHost(elementCount);
        throwIfFailed(
            cudaMemcpy(halfHost.data(), this->momentumHalf.deviceData, elementCount * sizeof(__half), cudaMemcpyDeviceToHost),
            "CudaSpulseState::downloadInto momentumHalf");
        for (size_t i = 0; i < elementCount; ++i)
            host.momentum.data[i] = __half2float(halfHost[i]);
    } else {
        std::vector<signed char> qHost(elementCount);
        std::vector<float> scales(static_cast<size_t>(this->scaleCount), 0.0f);
        this->momentumQ.copyToHost(qHost.data(), elementCount);
        throwIfFailed(
            cudaMemcpy(scales.data(), this->momentumScales.deviceData, scales.size() * sizeof(float), cudaMemcpyDeviceToHost),
            "CudaSpulseState::downloadInto momentumScales");
        for (size_t i = 0; i < elementCount; ++i) {
            const int block = static_cast<int>(i / static_cast<size_t>(this->int8BlockSize));
            const float scale = scales[static_cast<size_t>(block)];
            host.momentum.data[i] = scale == 0.0f ? 0.0f : static_cast<float>(qHost[i]) * scale;
        }
    }

    float energyHost[3] = {0.0f, 0.0f, 1.0f};
    if (this->energy.deviceData != nullptr) {
        throwIfFailed(
            cudaMemcpy(energyHost, this->energy.deviceData, 3 * sizeof(float), cudaMemcpyDeviceToHost),
            "CudaSpulseState::downloadInto energy");
    }
    host.energyFast = energyHost[0];
    host.energySlow = energyHost[1];
    host.scale = energyHost[2];
}

void CudaSpulseState::uploadFrom(const SpulseState& host) {
    if (host.momentum.empty()) {
        this->free();
        return;
    }
    this->ensure(host.momentum.rows, host.momentum.cols, this->storage, this->int8BlockSize);
    const size_t elementCount = host.momentum.data.size();

    if (this->storage == SpulseMomentumStorage::Fp32) {
        this->momentum.upload(host.momentum);
    } else if (this->storage == SpulseMomentumStorage::Fp16) {
        std::vector<__half> halfHost(elementCount);
        for (size_t i = 0; i < elementCount; ++i)
            halfHost[i] = __float2half_rn(host.momentum.data[i]);
        throwIfFailed(
            cudaMemcpy(this->momentumHalf.deviceData, halfHost.data(), elementCount * sizeof(__half), cudaMemcpyHostToDevice),
            "CudaSpulseState::uploadFrom momentumHalf");
    } else {
        std::vector<signed char> qHost(elementCount, 0);
        std::vector<float> scales(static_cast<size_t>(this->scaleCount), 0.0f);
        for (int block = 0; block < this->scaleCount; ++block) {
            const size_t start = static_cast<size_t>(block) * static_cast<size_t>(this->int8BlockSize);
            const size_t end = (std::min)(elementCount, start + static_cast<size_t>(this->int8BlockSize));
            float absmax = 0.0f;
            for (size_t i = start; i < end; ++i)
                absmax = (std::max)(absmax, std::fabs(host.momentum.data[i]));
            const float scale = absmax > 0.0f ? absmax / 127.0f : 0.0f;
            scales[static_cast<size_t>(block)] = scale;
            for (size_t i = start; i < end; ++i) {
                float q = scale == 0.0f ? 0.0f : std::round(host.momentum.data[i] / scale);
                q = (std::min)(127.0f, (std::max)(-127.0f, q));
                qHost[i] = static_cast<signed char>(q);
            }
        }
        this->momentumQ.copyFromHost(qHost.data(), elementCount);
        throwIfFailed(
            cudaMemcpy(this->momentumScales.deviceData, scales.data(), scales.size() * sizeof(float), cudaMemcpyHostToDevice),
            "CudaSpulseState::uploadFrom momentumScales");
    }

    this->energy.ensureCapacity(3 * sizeof(float));
    const float energyHost[3] = {host.energyFast, host.energySlow, host.scale};
    throwIfFailed(
        cudaMemcpy(this->energy.deviceData, energyHost, 3 * sizeof(float), cudaMemcpyHostToDevice),
        "CudaSpulseState::uploadFrom energy");
}

void CudaTransformerBlockSpulseStates::ensureFrom(
    const CudaTransformerBlock& block,
    SpulseMomentumStorage storage,
    int int8BlockSize,
    bool includeAux
) {
    this->queryWeight.ensure(block.attention.queryWeight, storage, int8BlockSize);
    this->keyWeight.ensure(block.attention.keyWeight, storage, int8BlockSize);
    this->valueWeight.ensure(block.attention.valueWeight, storage, int8BlockSize);
    this->attentionOutputWeight.ensure(block.attention.outputWeight, storage, int8BlockSize);
    this->feedForwardGateWeight.ensure(block.feedForward.gateWeight, storage, int8BlockSize);
    this->feedForwardUpWeight.ensure(block.feedForward.upWeight, storage, int8BlockSize);
    this->feedForwardDownWeight.ensure(block.feedForward.downWeight, storage, int8BlockSize);

    if (includeAux) {
        this->attentionNormGamma.ensure(block.attentionNorm.gamma, storage, int8BlockSize);
        this->feedForwardNormGamma.ensure(block.feedForwardNorm.gamma, storage, int8BlockSize);
        if (!block.feedForward.gateBias.empty())
            this->feedForwardGateBias.ensure(block.feedForward.gateBias, storage, int8BlockSize);
        else
            this->feedForwardGateBias.free();
        if (!block.feedForward.upBias.empty())
            this->feedForwardUpBias.ensure(block.feedForward.upBias, storage, int8BlockSize);
        else
            this->feedForwardUpBias.free();
        if (!block.feedForward.downBias.empty())
            this->feedForwardDownBias.ensure(block.feedForward.downBias, storage, int8BlockSize);
        else
            this->feedForwardDownBias.free();
    } else {
        this->attentionNormGamma.free();
        this->feedForwardNormGamma.free();
        this->feedForwardGateBias.free();
        this->feedForwardUpBias.free();
        this->feedForwardDownBias.free();
    }
}

void CudaTransformerBlockSpulseStates::free() {
    this->queryWeight.free();
    this->keyWeight.free();
    this->valueWeight.free();
    this->attentionOutputWeight.free();
    this->feedForwardGateWeight.free();
    this->feedForwardUpWeight.free();
    this->feedForwardDownWeight.free();
    this->attentionNormGamma.free();
    this->feedForwardNormGamma.free();
    this->feedForwardGateBias.free();
    this->feedForwardUpBias.free();
    this->feedForwardDownBias.free();
}

void CudaTransformerBlockHostSpulseStates::clear() {
    this->queryWeight.clear();
    this->keyWeight.clear();
    this->valueWeight.clear();
    this->attentionOutputWeight.clear();
    this->feedForwardGateWeight.clear();
    this->feedForwardUpWeight.clear();
    this->feedForwardDownWeight.clear();
}

CudaSpulse::CudaSpulse(
    float learningRate,
    float momentumBeta,
    float fastBeta,
    float slowBeta,
    float epsilon,
    float scaleMin,
    float scaleMax,
    float weightDecay,
    SpulseCoverage coverage,
    bool hostLightweight,
    SpulseMomentumStorage momentumStorage,
    int int8BlockSize
)
    : learningRate(learningRate),
      momentumBeta(momentumBeta),
      fastBeta(fastBeta),
      slowBeta(slowBeta),
      epsilon(epsilon),
      scaleMin(scaleMin),
      scaleMax(scaleMax),
      weightDecay(weightDecay),
      coverage(coverage),
      hostLightweight(hostLightweight),
      momentumStorage(momentumStorage),
      int8BlockSize(int8BlockSize),
      timeStep(0) {
    if (learningRate <= 0.0f) throw std::invalid_argument("CudaSpulse learningRate must be > 0");
    if (momentumBeta < 0.0f || momentumBeta >= 1.0f) throw std::invalid_argument("CudaSpulse momentumBeta must be in [0, 1)");
    if (fastBeta < 0.0f || fastBeta >= 1.0f) throw std::invalid_argument("CudaSpulse fastBeta must be in [0, 1)");
    if (slowBeta < 0.0f || slowBeta >= 1.0f) throw std::invalid_argument("CudaSpulse slowBeta must be in [0, 1)");
    if (slowBeta < fastBeta) throw std::invalid_argument("CudaSpulse slowBeta must be >= fastBeta");
    if (epsilon <= 0.0f) throw std::invalid_argument("CudaSpulse epsilon must be > 0");
    if (scaleMin <= 0.0f || scaleMax < scaleMin) throw std::invalid_argument("CudaSpulse invalid scale clip");
    if (weightDecay < 0.0f) throw std::invalid_argument("CudaSpulse weightDecay must be >= 0");
    if (int8BlockSize <= 0) throw std::invalid_argument("CudaSpulse int8BlockSize must be > 0");
}

void CudaSpulse::step() {
    ++this->timeStep;
}

float CudaSpulse::momentumBiasCorrection() const {
    if (this->timeStep <= 0) return 1.0f;
    const float oneMinusPow = 1.0f - std::pow(this->momentumBeta, static_cast<float>(this->timeStep));
    if (!(oneMinusPow > 0.0f)) return 1.0f;
    return 1.0f / oneMinusPow;
}

const char* CudaSpulse::coverageName(SpulseCoverage coverage) {
    switch (coverage) {
    case SpulseCoverage::Hybrid: return "Hybrid";
    case SpulseCoverage::Full: return "Full";
    }
    return "Unknown";
}

const char* CudaSpulse::momentumStorageName(SpulseMomentumStorage storage) {
    switch (storage) {
    case SpulseMomentumStorage::Fp32: return "Fp32";
    case SpulseMomentumStorage::Fp16: return "Fp16";
    case SpulseMomentumStorage::Int8: return "Int8";
    }
    return "Unknown";
}

bool CudaSpulse::ownsHybridBlockWeights() const {
    return this->coverage == SpulseCoverage::Hybrid || this->coverage == SpulseCoverage::Full;
}

bool CudaSpulse::ownsFullModelWeights() const {
    return this->coverage == SpulseCoverage::Full;
}

void CudaSpulse::update(CudaMatrix& parameter, CudaSpulseState& state, const CudaMatrix& gradient, float gradientScale) {
    if (parameter.empty()) throw std::invalid_argument("CudaSpulse::update empty parameter");
    if (gradient.elementCount() != parameter.elementCount())
        throw std::invalid_argument("CudaSpulse::update gradient/parameter size mismatch");
    state.ensure(parameter, this->momentumStorage, this->int8BlockSize);

    const int elementCount = static_cast<int>(parameter.elementCount());
    const float oneMinusMom = 1.0f - this->momentumBeta;
    const float keep = 1.0f - this->learningRate * this->weightDecay;
    const float momCorrection = this->momentumBiasCorrection();
    const float oneMinusFast = 1.0f - this->fastBeta;
    const float oneMinusSlow = 1.0f - this->slowBeta;
    cudaStream_t stream = CudaMatmul::activeStream();

    // [0]=sumSquares (float), [1]=blocksDone (int) — zero both each launch.
    this->sumSquaresScratch.ensureCapacity(sizeof(float) + sizeof(int));
    float* sumSquares = this->sumSquaresScratch.deviceData;
    int* blocksDone = reinterpret_cast<int*>(sumSquares + 1);
    CudaMatmul::memsetDevice(sumSquares, 0, sizeof(float) + sizeof(int));

    if (state.storage == SpulseMomentumStorage::Fp32) {
        const int threads = 256;
        const int blocks = (std::min)(1024, elementwiseBlocks((std::max)(1, elementCount >> 2), threads));
        spulseFusedStep<<<blocks, threads, 0, stream>>>(
            parameter.buffer.deviceData,
            state.momentum.buffer.deviceData,
            gradient.buffer.deviceData,
            state.energy.deviceData,
            sumSquares,
            blocksDone,
            elementCount,
            this->momentumBeta,
            oneMinusMom,
            gradientScale,
            this->learningRate,
            keep,
            momCorrection,
            this->fastBeta,
            this->slowBeta,
            oneMinusFast,
            oneMinusSlow,
            this->epsilon,
            this->scaleMin,
            this->scaleMax);
        throwIfFailed(cudaGetLastError(), "spulseFusedStep");
    } else if (state.storage == SpulseMomentumStorage::Fp16) {
        const int threads = 256;
        const int blocks = (std::min)(1024, elementwiseBlocks((std::max)(1, elementCount >> 2), threads));
        spulseFusedStepHalf<<<blocks, threads, 0, stream>>>(
            parameter.buffer.deviceData,
            reinterpret_cast<__half*>(state.momentumHalf.deviceData),
            gradient.buffer.deviceData,
            state.energy.deviceData,
            sumSquares,
            blocksDone,
            elementCount,
            this->momentumBeta,
            oneMinusMom,
            gradientScale,
            this->learningRate,
            keep,
            momCorrection,
            this->fastBeta,
            this->slowBeta,
            oneMinusFast,
            oneMinusSlow,
            this->epsilon,
            this->scaleMin,
            this->scaleMax);
        throwIfFailed(cudaGetLastError(), "spulseFusedStepHalf");
    } else {
        const int blockSize = state.int8BlockSize;
        const int threads = (std::min)(256, blockSize);
        const int blocks = state.scaleCount;
        const size_t sharedBytes = static_cast<size_t>(blockSize + 2 * threads) * sizeof(float);
        spulseStepInt8<<<blocks, threads, sharedBytes, stream>>>(
            parameter.buffer.deviceData,
            state.momentumQ.deviceData,
            state.momentumScales.deviceData,
            gradient.buffer.deviceData,
            state.energy.deviceData,
            sumSquares,
            blocksDone,
            elementCount,
            blockSize,
            this->momentumBeta,
            oneMinusMom,
            gradientScale,
            this->learningRate,
            keep,
            momCorrection,
            0,
            this->fastBeta,
            this->slowBeta,
            oneMinusFast,
            oneMinusSlow,
            this->epsilon,
            this->scaleMin,
            this->scaleMax);
        throwIfFailed(cudaGetLastError(), "spulseStepInt8");
    }
}

void CudaSpulse::prepareHostDeltaInPlace(
    float* gradientOrDelta,
    size_t rows,
    size_t cols,
    CudaSpulseState& state,
    float gradientScale
) {
    if (gradientOrDelta == nullptr) throw std::invalid_argument("CudaSpulse::prepareHostDeltaInPlace null buffer");
    if (rows == 0 || cols == 0) throw std::invalid_argument("CudaSpulse::prepareHostDeltaInPlace empty shape");
    state.ensure(rows, cols, this->momentumStorage, this->int8BlockSize);

    const int elementCount = static_cast<int>(rows * cols);
    const float oneMinusMom = 1.0f - this->momentumBeta;
    const float momCorrection = this->momentumBiasCorrection();
    const float oneMinusFast = 1.0f - this->fastBeta;
    const float oneMinusSlow = 1.0f - this->slowBeta;
    cudaStream_t stream = CudaMatmul::activeStream();

    this->sumSquaresScratch.ensureCapacity(sizeof(float) + sizeof(int));
    float* sumSquares = this->sumSquaresScratch.deviceData;
    int* blocksDone = reinterpret_cast<int*>(sumSquares + 1);
    CudaMatmul::memsetDevice(sumSquares, 0, sizeof(float) + sizeof(int));

    if (state.storage == SpulseMomentumStorage::Fp32) {
        const int threads = 256;
        const int blocks = (std::min)(1024, elementwiseBlocks((std::max)(1, elementCount >> 2), threads));
        spulsePrepareHostDelta<<<blocks, threads, 0, stream>>>(
            gradientOrDelta,
            state.momentum.buffer.deviceData,
            state.energy.deviceData,
            sumSquares,
            blocksDone,
            elementCount,
            this->momentumBeta,
            oneMinusMom,
            gradientScale,
            this->learningRate,
            momCorrection,
            this->fastBeta,
            this->slowBeta,
            oneMinusFast,
            oneMinusSlow,
            this->epsilon,
            this->scaleMin,
            this->scaleMax);
        throwIfFailed(cudaGetLastError(), "spulsePrepareHostDelta");
    } else if (state.storage == SpulseMomentumStorage::Fp16) {
        const int threads = 256;
        const int blocks = (std::min)(1024, elementwiseBlocks((std::max)(1, elementCount >> 2), threads));
        spulsePrepareHostDeltaHalf<<<blocks, threads, 0, stream>>>(
            gradientOrDelta,
            reinterpret_cast<__half*>(state.momentumHalf.deviceData),
            state.energy.deviceData,
            sumSquares,
            blocksDone,
            elementCount,
            this->momentumBeta,
            oneMinusMom,
            gradientScale,
            this->learningRate,
            momCorrection,
            this->fastBeta,
            this->slowBeta,
            oneMinusFast,
            oneMinusSlow,
            this->epsilon,
            this->scaleMin,
            this->scaleMax);
        throwIfFailed(cudaGetLastError(), "spulsePrepareHostDeltaHalf");
    } else {
        const int blockSize = state.int8BlockSize;
        const int threads = (std::min)(256, blockSize);
        const int blocks = state.scaleCount;
        const size_t sharedBytes = static_cast<size_t>(blockSize + 2 * threads) * sizeof(float);
        spulseStepInt8<<<blocks, threads, sharedBytes, stream>>>(
            gradientOrDelta,
            state.momentumQ.deviceData,
            state.momentumScales.deviceData,
            nullptr,
            state.energy.deviceData,
            sumSquares,
            blocksDone,
            elementCount,
            blockSize,
            this->momentumBeta,
            oneMinusMom,
            gradientScale,
            this->learningRate,
            1.0f,
            momCorrection,
            1,
            this->fastBeta,
            this->slowBeta,
            oneMinusFast,
            oneMinusSlow,
            this->epsilon,
            this->scaleMin,
            this->scaleMax);
        throwIfFailed(cudaGetLastError(), "spulseStepInt8 (host delta)");
    }
}

void CudaSpulse::prepareHybridBlockHostDeltas(
    CudaTransformerBlock& block,
    CudaTransformerBlockSpulseStates& spulseStates,
    float gradientScale
) {
    if (!this->ownsHybridBlockWeights()) return;
    if (block.feedForwardDownWeightGradient.empty())
        throw std::logic_error("CudaSpulse::prepareHybridBlockHostDeltas empty down grad");

    // Any prior deferred energy commit must finish before we reuse scratch.
    this->commitPendingHybridHostEnergies();

    // Ensure states first, then decide batching from actual storage after ensure.
    auto ensurePiece = [this](CudaSpulseState& state, size_t rows, size_t cols) {
        state.ensure(rows, cols, this->momentumStorage, this->int8BlockSize);
    };

    ensurePiece(
        spulseStates.feedForwardDownWeight,
        block.feedForwardDownWeightGradient.rows,
        block.feedForwardDownWeightGradient.cols);
    ensurePiece(
        spulseStates.attentionOutputWeight,
        block.attentionOutputWeightGradient.rows,
        block.attentionOutputWeightGradient.cols);

    if (!block.feedForward.gateUpWeightGradient.empty()) {
        ensurePiece(spulseStates.feedForwardGateWeight, block.feedForward.gateWeight.rows, block.feedForward.gateWeight.cols);
        ensurePiece(spulseStates.feedForwardUpWeight, block.feedForward.upWeight.rows, block.feedForward.upWeight.cols);
    } else {
        ensurePiece(
            spulseStates.feedForwardGateWeight,
            block.feedForwardGateWeightGradient.rows,
            block.feedForwardGateWeightGradient.cols);
        ensurePiece(
            spulseStates.feedForwardUpWeight,
            block.feedForwardUpWeightGradient.rows,
            block.feedForwardUpWeightGradient.cols);
    }
    if (!block.attention.qkvWeightGradient.empty()) {
        ensurePiece(spulseStates.queryWeight, block.attention.queryWeight.rows, block.attention.queryWeight.cols);
        ensurePiece(spulseStates.keyWeight, block.attention.keyWeight.rows, block.attention.keyWeight.cols);
        ensurePiece(spulseStates.valueWeight, block.attention.valueWeight.rows, block.attention.valueWeight.cols);
    } else {
        ensurePiece(spulseStates.queryWeight, block.queryWeightGradient.rows, block.queryWeightGradient.cols);
        ensurePiece(spulseStates.keyWeight, block.keyWeightGradient.rows, block.keyWeightGradient.cols);
        ensurePiece(spulseStates.valueWeight, block.valueWeightGradient.rows, block.valueWeightGradient.cols);
    }

    if (this->momentumStorage != SpulseMomentumStorage::Fp32
        || spulseStates.feedForwardDownWeight.storage != SpulseMomentumStorage::Fp32) {
        this->prepareHostDeltaInPlace(
            block.feedForwardDownWeightGradient.buffer.deviceData,
            block.feedForwardDownWeightGradient.rows,
            block.feedForwardDownWeightGradient.cols,
            spulseStates.feedForwardDownWeight,
            gradientScale);
        if (!block.feedForward.gateUpWeightGradient.empty()) {
            const size_t gElems = block.feedForward.gateWeight.elementCount();
            this->prepareHostDeltaInPlace(
                block.feedForward.gateUpWeightGradient.buffer.deviceData,
                block.feedForward.gateWeight.rows,
                block.feedForward.gateWeight.cols,
                spulseStates.feedForwardGateWeight,
                gradientScale);
            this->prepareHostDeltaInPlace(
                block.feedForward.gateUpWeightGradient.buffer.deviceData + gElems,
                block.feedForward.upWeight.rows,
                block.feedForward.upWeight.cols,
                spulseStates.feedForwardUpWeight,
                gradientScale);
        } else {
            this->prepareHostDeltaInPlace(
                block.feedForwardGateWeightGradient.buffer.deviceData,
                block.feedForwardGateWeightGradient.rows,
                block.feedForwardGateWeightGradient.cols,
                spulseStates.feedForwardGateWeight,
                gradientScale);
            this->prepareHostDeltaInPlace(
                block.feedForwardUpWeightGradient.buffer.deviceData,
                block.feedForwardUpWeightGradient.rows,
                block.feedForwardUpWeightGradient.cols,
                spulseStates.feedForwardUpWeight,
                gradientScale);
        }
        this->prepareHostDeltaInPlace(
            block.attentionOutputWeightGradient.buffer.deviceData,
            block.attentionOutputWeightGradient.rows,
            block.attentionOutputWeightGradient.cols,
            spulseStates.attentionOutputWeight,
            gradientScale);
        if (!block.attention.qkvWeightGradient.empty()) {
            const size_t qElems = block.attention.queryWeight.elementCount();
            this->prepareHostDeltaInPlace(
                block.attention.qkvWeightGradient.buffer.deviceData,
                block.attention.queryWeight.rows,
                block.attention.queryWeight.cols,
                spulseStates.queryWeight,
                gradientScale);
            this->prepareHostDeltaInPlace(
                block.attention.qkvWeightGradient.buffer.deviceData + qElems,
                block.attention.keyWeight.rows,
                block.attention.keyWeight.cols,
                spulseStates.keyWeight,
                gradientScale);
            this->prepareHostDeltaInPlace(
                block.attention.qkvWeightGradient.buffer.deviceData + 2ull * qElems,
                block.attention.valueWeight.rows,
                block.attention.valueWeight.cols,
                spulseStates.valueWeight,
                gradientScale);
        } else {
            this->prepareHostDeltaInPlace(
                block.queryWeightGradient.buffer.deviceData,
                block.queryWeightGradient.rows,
                block.queryWeightGradient.cols,
                spulseStates.queryWeight,
                gradientScale);
            this->prepareHostDeltaInPlace(
                block.keyWeightGradient.buffer.deviceData,
                block.keyWeightGradient.rows,
                block.keyWeightGradient.cols,
                spulseStates.keyWeight,
                gradientScale);
            this->prepareHostDeltaInPlace(
                block.valueWeightGradient.buffer.deviceData,
                block.valueWeightGradient.rows,
                block.valueWeightGradient.cols,
                spulseStates.valueWeight,
                gradientScale);
        }
        return;
    }

    SpulseHostDeltaPiece hostPieces[7];
    int pieceCount = 0;
    auto push = [&](float* grad, CudaSpulseState& state) {
        SpulseHostDeltaPiece& piece = hostPieces[pieceCount++];
        piece.gradOrDelta = grad;
        piece.momentum = state.momentum.buffer.deviceData;
        piece.energy = state.energy.deviceData;
        piece.sumSquares = nullptr; // filled after scratch alloc
        piece.elementCount = static_cast<int>(state.rows * state.cols);
    };

    push(block.feedForwardDownWeightGradient.buffer.deviceData, spulseStates.feedForwardDownWeight);
    if (!block.feedForward.gateUpWeightGradient.empty()) {
        const size_t gElems = block.feedForward.gateWeight.elementCount();
        push(block.feedForward.gateUpWeightGradient.buffer.deviceData, spulseStates.feedForwardGateWeight);
        push(block.feedForward.gateUpWeightGradient.buffer.deviceData + gElems, spulseStates.feedForwardUpWeight);
    } else {
        push(block.feedForwardGateWeightGradient.buffer.deviceData, spulseStates.feedForwardGateWeight);
        push(block.feedForwardUpWeightGradient.buffer.deviceData, spulseStates.feedForwardUpWeight);
    }
    push(block.attentionOutputWeightGradient.buffer.deviceData, spulseStates.attentionOutputWeight);
    if (!block.attention.qkvWeightGradient.empty()) {
        const size_t qElems = block.attention.queryWeight.elementCount();
        push(block.attention.qkvWeightGradient.buffer.deviceData, spulseStates.queryWeight);
        push(block.attention.qkvWeightGradient.buffer.deviceData + qElems, spulseStates.keyWeight);
        push(block.attention.qkvWeightGradient.buffer.deviceData + 2ull * qElems, spulseStates.valueWeight);
    } else {
        push(block.queryWeightGradient.buffer.deviceData, spulseStates.queryWeight);
        push(block.keyWeightGradient.buffer.deviceData, spulseStates.keyWeight);
        push(block.valueWeightGradient.buffer.deviceData, spulseStates.valueWeight);
    }

    cudaStream_t stream = CudaMatmul::activeStream();
    this->sumSquaresScratch.ensureCapacity(static_cast<size_t>(pieceCount) * sizeof(float));
    float* sumSquaresBase = this->sumSquaresScratch.deviceData;
    CudaMatmul::memsetDevice(sumSquaresBase, 0, static_cast<size_t>(pieceCount) * sizeof(float));

    int maxElements = 1;
    for (int i = 0; i < pieceCount; ++i) {
        hostPieces[i].sumSquares = sumSquaresBase + i;
        maxElements = (std::max)(maxElements, hostPieces[i].elementCount);
    }

    this->deltaPieceScratch.ensureCapacity(static_cast<size_t>(pieceCount) * sizeof(SpulseHostDeltaPiece));
    throwIfFailed(
        cudaMemcpyAsync(
            this->deltaPieceScratch.deviceData,
            hostPieces,
            static_cast<size_t>(pieceCount) * sizeof(SpulseHostDeltaPiece),
            cudaMemcpyHostToDevice,
            stream),
        "prepareHybridBlockHostDeltas piece upload");

    const int threads = 256;
    const int blocksX = (std::min)(1024, elementwiseBlocks((std::max)(1, maxElements >> 2), threads));
    const dim3 grid(static_cast<unsigned>(blocksX), static_cast<unsigned>(pieceCount));
    const float momCorrection = this->momentumBiasCorrection();
    spulsePrepareHostDeltaBatched<<<grid, threads, 0, stream>>>(
        reinterpret_cast<const SpulseHostDeltaPiece*>(this->deltaPieceScratch.deviceData),
        pieceCount,
        this->momentumBeta,
        1.0f - this->momentumBeta,
        gradientScale,
        this->learningRate,
        momCorrection);
    throwIfFailed(cudaGetLastError(), "spulsePrepareHostDeltaBatched");

    // Defer energy commit so the caller can record the host-grad event and start D2H first.
    this->pendingHostEnergyPieceCount = pieceCount;
}

void CudaSpulse::commitPendingHybridHostEnergies() {
    if (this->pendingHostEnergyPieceCount <= 0) return;
    const int pieceCount = this->pendingHostEnergyPieceCount;
    this->pendingHostEnergyPieceCount = 0;

    cudaStream_t stream = CudaMatmul::activeStream();
    spulseCommitEnergyMany<<<1, pieceCount, 0, stream>>>(
        reinterpret_cast<const SpulseHostDeltaPiece*>(this->deltaPieceScratch.deviceData),
        pieceCount,
        this->fastBeta,
        this->slowBeta,
        1.0f - this->fastBeta,
        1.0f - this->slowBeta,
        this->epsilon,
        this->scaleMin,
        this->scaleMax);
    throwIfFailed(cudaGetLastError(), "spulseCommitEnergyMany");
}

void CudaSpulse::updateHost(Matrix& parameter, SpulseState& state, const Matrix& gradient, float gradientScale) const {
    if (parameter.empty()) throw std::invalid_argument("CudaSpulse::updateHost empty parameter");
    if (gradient.data.size() != parameter.data.size())
        throw std::invalid_argument("CudaSpulse::updateHost gradient/parameter size mismatch");
    state.ensure(parameter);
    hostUpdateCore(
        parameter.data.data(),
        state.momentum.data.data(),
        state.energyFast,
        state.energySlow,
        state.scale,
        gradient.data.data(),
        parameter.data.size(),
        this->learningRate,
        this->momentumBeta,
        this->fastBeta,
        this->slowBeta,
        this->epsilon,
        this->scaleMin,
        this->scaleMax,
        this->weightDecay,
        gradientScale,
        this->momentumBiasCorrection());
}

void CudaSpulse::updateHostFromHalf(
    Matrix& parameter,
    SpulseState& state,
    const std::uint16_t* gradHalf,
    size_t elementCount,
    float gradientScale
) const {
    if (parameter.empty()) throw std::invalid_argument("CudaSpulse::updateHostFromHalf empty parameter");
    if (gradHalf == nullptr) throw std::invalid_argument("CudaSpulse::updateHostFromHalf null gradHalf");
    if (parameter.data.size() != elementCount)
        throw std::invalid_argument("CudaSpulse::updateHostFromHalf size mismatch");

    if (this->hostLightweight) {
        // Energy scalars only — do not allocate per-element momentum (HostSGD traffic).
        if (state.scale <= 0.0f)
            state.scale = 1.0f;
        hostUpdateFromHalfLite(
            parameter.data.data(),
            state.energyFast,
            state.energySlow,
            state.scale,
            reinterpret_cast<const __half*>(gradHalf),
            elementCount,
            this->learningRate,
            this->fastBeta,
            this->slowBeta,
            this->epsilon,
            this->scaleMin,
            this->scaleMax,
            this->weightDecay,
            gradientScale);
        return;
    }

    state.ensure(parameter);
    hostUpdateFromHalfCore(
        parameter.data.data(),
        state.momentum.data.data(),
        state.energyFast,
        state.energySlow,
        state.scale,
        reinterpret_cast<const __half*>(gradHalf),
        elementCount,
        this->learningRate,
        this->momentumBeta,
        this->fastBeta,
        this->slowBeta,
        this->epsilon,
        this->scaleMin,
        this->scaleMax,
        this->weightDecay,
        gradientScale,
        this->momentumBiasCorrection());
}

void CudaSpulse::applyFusedHalfHostPieces(const std::vector<SpulseFusedHalfHostPiece>& pieces, float gradientScale) const {
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
#endif
    for (int pieceIndex = 0; pieceIndex < static_cast<int>(pieces.size()); ++pieceIndex) {
        const SpulseFusedHalfHostPiece& piece = pieces[static_cast<size_t>(pieceIndex)];
        if (piece.host == nullptr || piece.state == nullptr || piece.gradHalf == nullptr) continue;
        this->updateHostFromHalf(*piece.host, *piece.state, piece.gradHalf, piece.elementCount, gradientScale);
    }
}

void CudaSpulse::applyHybridBlockWeights(
    CudaTransformerBlock& block,
    CudaTransformerBlockGradients& blockGradients,
    CudaTransformerBlockSpulseStates& spulseStates,
    float gradientScale
) {
    if (!this->ownsHybridBlockWeights()) return;

    // Per-tensor updates (no QKV/gateUp fuse memcpy — that was Muon-only overhead).
    // Fused mirrors are refreshed by CudaLanguageModel::applyGradients → syncFusedMirrors.
    this->update(block.attention.queryWeight, spulseStates.queryWeight, blockGradients.queryWeight, gradientScale);
    this->update(block.attention.keyWeight, spulseStates.keyWeight, blockGradients.keyWeight, gradientScale);
    this->update(block.attention.valueWeight, spulseStates.valueWeight, blockGradients.valueWeight, gradientScale);
    this->update(block.attention.outputWeight, spulseStates.attentionOutputWeight, blockGradients.attentionOutputWeight, gradientScale);
    this->update(block.feedForward.gateWeight, spulseStates.feedForwardGateWeight, blockGradients.feedForwardGateWeight, gradientScale);
    this->update(block.feedForward.upWeight, spulseStates.feedForwardUpWeight, blockGradients.feedForwardUpWeight, gradientScale);
    this->update(block.feedForward.downWeight, spulseStates.feedForwardDownWeight, blockGradients.feedForwardDownWeight, gradientScale);
}

void CudaSpulse::applyFullBlockAuxWeights(
    CudaTransformerBlock& block,
    CudaTransformerBlockGradients& blockGradients,
    CudaTransformerBlockSpulseStates& spulseStates,
    float gradientScale,
    bool useBias
) {
    if (!this->ownsFullModelWeights()) return;

    this->update(
        block.attentionNorm.gamma,
        spulseStates.attentionNormGamma,
        blockGradients.attentionNormGamma,
        gradientScale);
    this->update(
        block.feedForwardNorm.gamma,
        spulseStates.feedForwardNormGamma,
        blockGradients.feedForwardNormGamma,
        gradientScale);
    if (useBias) {
        if (!block.feedForward.gateBias.empty())
            this->update(
                block.feedForward.gateBias,
                spulseStates.feedForwardGateBias,
                blockGradients.feedForwardGateBias,
                gradientScale);
        if (!block.feedForward.upBias.empty())
            this->update(
                block.feedForward.upBias,
                spulseStates.feedForwardUpBias,
                blockGradients.feedForwardUpBias,
                gradientScale);
        if (!block.feedForward.downBias.empty())
            this->update(
                block.feedForward.downBias,
                spulseStates.feedForwardDownBias,
                blockGradients.feedForwardDownBias,
                gradientScale);
    }
}

void CudaSpulse::runSmokeDemo(int parameterRows, int parameterCols) {
    if (!CudaMatmul::isAvailable()) {
        SmokeLog::skip("SPULSE");
        return;
    }
    if (parameterRows < 2 || parameterCols < 1)
        throw std::invalid_argument("CudaSpulse::runSmokeDemo invalid dims");

    Matrix hostParam(static_cast<size_t>(parameterRows), static_cast<size_t>(parameterCols), 0.0f);
    Matrix hostGrad(static_cast<size_t>(parameterRows), static_cast<size_t>(parameterCols), 0.0f);
    unsigned rng = 424242u;
    for (size_t index = 0; index < hostParam.data.size(); ++index) {
        rng = rng * 1664525u + 1013904223u;
        hostParam.data[index] = (static_cast<float>(rng >> 8) / 16777216.0f) * 2.0f - 1.0f;
        rng = rng * 1664525u + 1013904223u;
        hostGrad.data[index] = (static_cast<float>(rng >> 8) / 16777216.0f) * 0.02f - 0.01f;
    }

    auto runOnce = [&](SpulseMomentumStorage storage, float maxAllowedDiff, const char* label) {
        CudaSpulse opt(
            1e-3f, 0.9f, 0.9f, 0.999f, 1e-8f, 0.25f, 4.0f, 0.0f,
            SpulseCoverage::Hybrid, false, storage, 256);
        Matrix hostParamRef = hostParam;
        SpulseState hostState;
        // Host reference always uses FP32 u.
        CudaSpulse hostOpt(1e-3f, 0.9f, 0.9f, 0.999f, 1e-8f, 0.25f, 4.0f, 0.0f, SpulseCoverage::Hybrid);
        hostOpt.step();
        hostOpt.updateHost(hostParamRef, hostState, hostGrad, 1.0f);

        CudaMatrix deviceParam;
        deviceParam.upload(hostParam);
        CudaMatrix deviceGrad;
        deviceGrad.upload(hostGrad);
        CudaSpulseState deviceState;
        opt.step();
        opt.update(deviceParam, deviceState, deviceGrad, 1.0f);
        throwIfFailed(cudaDeviceSynchronize(), "CudaSpulse update synchronize");
        Matrix deviceDownloaded = deviceParam.download();

        float maxDiff = 0.0f;
        bool nonFinite = false;
        for (size_t index = 0; index < hostParamRef.data.size(); ++index) {
            if (!std::isfinite(hostParamRef.data[index]) || !std::isfinite(deviceDownloaded.data[index]))
                nonFinite = true;
            maxDiff = (std::max)(maxDiff, std::fabs(hostParamRef.data[index] - deviceDownloaded.data[index]));
        }

        SpulseState energyCheck;
        deviceState.downloadInto(energyCheck);
        const bool energyOk = std::isfinite(energyCheck.energyFast) && std::isfinite(energyCheck.energySlow)
            && energyCheck.energyFast >= 0.0f && energyCheck.energySlow >= 0.0f;

        SmokeLog::result(
            label,
            "storage=%s  shape=%dx%d  maxDiff=%.3e  finite=%s  energyOk=%s",
            CudaSpulse::momentumStorageName(storage),
            parameterRows,
            parameterCols,
            maxDiff,
            nonFinite ? "no" : "yes",
            energyOk ? "yes" : "no");

        if (nonFinite || !energyOk || maxDiff > maxAllowedDiff)
            throw std::runtime_error(std::string("CudaSpulse::runSmokeDemo failed: ") + label);
    };

    runOnce(SpulseMomentumStorage::Fp32, 5e-4f, "SPULSE host↔GPU");
    runOnce(SpulseMomentumStorage::Fp16, 2e-3f, "SPULSE Fp16-u");
    runOnce(SpulseMomentumStorage::Int8, 5e-2f, "SPULSE Int8-u");
}
