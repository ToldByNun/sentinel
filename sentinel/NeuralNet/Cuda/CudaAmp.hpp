#ifndef CUDAAMP_HPP
#define CUDAAMP_HPP

#include "CudaMatmul.hpp"

#include <cstddef>

class CudaLanguageModelGradients;

/// <summary>dynamic loss scaler for FP16 mixed precision training</summary>
class CudaLossScaler {
public:
    float scale;
    float growthFactor;
    float backoffFactor;
    float minScale;
    float maxScale;
    int growthInterval;
    int successfulSteps;

    CudaLossScaler();

    /// <summary>halve scale after non finite grads</summary>
    void updateOnOverflow();

    /// <summary>grow scale after enough clean Adam steps</summary>
    void updateOnSuccess();
};

/// <summary>row major FP16 matrix living on device for amp checkpoints</summary>
class CudaHalfMatrix {
public:
    size_t rows;
    size_t cols;
    CudaDeviceBuffer buffer;

    CudaHalfMatrix();
    CudaHalfMatrix(const CudaHalfMatrix&) = delete;
    CudaHalfMatrix& operator=(const CudaHalfMatrix&) = delete;
    CudaHalfMatrix(CudaHalfMatrix&& other) noexcept;
    CudaHalfMatrix& operator=(CudaHalfMatrix&& other) noexcept;

    bool empty() const;
    size_t elementCount() const;
    size_t byteCount() const;
    void ensureSize(size_t rowCount, size_t columnCount);
    void free();
};

/// <summary>
/// FP16 mixed precision helpers for consumer GPU training
/// master weights stay FP32 GEMMs cast to FP16 with FP32 accumulate
/// </summary>
class CudaAmp {
public:
    /// <summary>when true multiplyInto prefers FP16 tensor core GEMM</summary>
    static bool preferMixedPrecision;

    /// <summary>when true scale CE grads and skip Adam on nonfinite (only useful if FP16 GEMMs can run)</summary>
    static bool useLossScaling;

    static CudaLossScaler lossScaler;

    /// <summary>preferMixedPrecision and useLossScaling</summary>
    static bool lossScalingActive();

    /// <summary>reset dynamic loss scaler to a consumer-safe start</summary>
    static void resetLossScaler();

    /// <summary>cast float matrix to half matrix</summary>
    static void castToHalf(const CudaMatrix& source, CudaHalfMatrix& destination);

    /// <summary>cast to half clamping to FP16 range; non-finite becomes 0 (safe for activation checkpoints)</summary>
    static void castToHalfSaturated(const CudaMatrix& source, CudaHalfMatrix& destination);

    /// <summary>cast half matrix to float matrix</summary>
    static void castToFloat(const CudaHalfMatrix& source, CudaMatrix& destination);

    /// <summary>cast float buffer to half into reusable scratch</summary>
    static void castFloatBufferToHalf(const float* source, size_t elementCount, CudaDeviceBuffer& destinationHalf);

    /// <summary>true if any element is NaN or Inf</summary>
    static bool hasNonFinite(const CudaMatrix& matrix);

    /// <summary>true if any gradient tensor is non finite</summary>
    static bool gradientsHaveNonFinite(const CudaLanguageModelGradients& gradients);

    /// <summary>cuBLASLt FP16xFP16 GEMM with FP32 accumulate into float out</summary>
    static bool launchCublasLtMatmulFp16(const float* deviceLeft, const float* deviceRight, float* deviceOut, int rowCount, int columnCount, int sharedCount, bool transposeLeft, bool transposeRight, double* kernelMilliseconds);

    /// <summary>FP16 GEMM using a pre-cast right operand (avoids re-casting shared activations)</summary>
    static bool launchCublasLtMatmulFp16PreCastRight(const float* deviceLeft, const void* rightHalf, float* deviceOut, int rowCount, int columnCount, int sharedCount, bool transposeLeft, bool transposeRight, double* kernelMilliseconds);

    /// <summary>cast activations to half into reusable scratch; returns device half pointer</summary>
    static const void* castActivationToHalfScratch(const float* source, size_t elementCount);

    /// <summary>register FP32 master weight for sticky FP16 GEMM casts (same pointer until free)</summary>
    static void registerMasterWeight(const float* deviceData, size_t elementCount);

    /// <summary>mark all registered master-weight FP16 mirrors stale (call after Adam)</summary>
    static void invalidateMasterWeightHalves();

    /// <summary>drop all master-weight registrations</summary>
    static void clearMasterWeights();

private:
    static CudaDeviceBuffer halfScratchLeft;
    static CudaDeviceBuffer halfScratchRight;
    static CudaDeviceBuffer nonFiniteFlag;

    struct MasterWeightHalf {
        const float* deviceData;
        size_t elementCount;
        CudaDeviceBuffer half;
        bool valid;
    };

    static constexpr int maxMasterWeights = 128;
    static MasterWeightHalf masterWeights[maxMasterWeights];
    static int masterWeightCount;

    static const void* masterWeightHalfOrNull(const float* deviceData, size_t elementCount);
};

#endif // CUDAAMP_HPP
