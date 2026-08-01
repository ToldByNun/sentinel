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

    static CudaLossScaler lossScaler;

    /// <summary>cast float matrix to half matrix</summary>
    static void castToHalf(const CudaMatrix& source, CudaHalfMatrix& destination);

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

private:
    static CudaDeviceBuffer halfScratchLeft;
    static CudaDeviceBuffer halfScratchRight;
    static CudaDeviceBuffer nonFiniteFlag;
};

#endif // CUDAAMP_HPP
