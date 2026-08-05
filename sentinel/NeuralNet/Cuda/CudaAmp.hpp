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
    int overflowCount;

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
    /// <summary>move donor device half buffer into this; clears shapes</summary>
    void takeDeviceStorageFrom(CudaHalfMatrix& donor);
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

    /// <summary>cuBLASLt FP16xFP16 GEMM with FP32 accumulate into float out; optional bias epilogue (length=rowCount)</summary>
    static bool launchCublasLtMatmulFp16(const float* deviceLeft, const float* deviceRight, float* deviceOut, int rowCount, int columnCount, int sharedCount, bool transposeLeft, bool transposeRight, double* kernelMilliseconds, const float* deviceBiasOrNull = nullptr);

    /// <summary>FP16 GEMM using a pre-cast right operand (avoids re-casting shared activations)</summary>
    static bool launchCublasLtMatmulFp16PreCastRight(const float* deviceLeft, const void* rightHalf, float* deviceOut, int rowCount, int columnCount, int sharedCount, bool transposeLeft, bool transposeRight, double* kernelMilliseconds, const float* deviceBiasOrNull = nullptr);

    /// <summary>cast activations to half into reusable scratch; returns device half pointer</summary>
    static const void* castActivationToHalfScratch(const float* source, size_t elementCount);

    /// <summary>
    /// persist FP16 view of an FP32 activation (reuses sticky scratch if present).
    /// Used so Full-ckpt bwd weight GEMMs skip a second float→half cast of inputCache.
    /// </summary>
    static void persistActivationHalf(const float* source, size_t rows, size_t cols, CudaHalfMatrix& destination);

    /// <summary>register FP32 master weight for sticky FP16 GEMM casts (same pointer until free)</summary>
    static void registerMasterWeight(const float* deviceData, size_t elementCount);

    /// <summary>
    /// bind matrix as FP16-only working weight: cast FP32 device → sticky half, free FP32 buffer
    /// </summary>
    static void bindFp16WorkingWeight(CudaMatrix& matrix);

    /// <summary>upload host FP32 master into an existing FP16 working-weight slot</summary>
    static void uploadHostMasterToFp16Working(CudaMatrix& matrix, const float* hostMaster);

    /// <summary>same as uploadHostMasterToFp16Working but H2D on stream (for pipelined host-SGD)</summary>
    static void uploadHostMasterToFp16Working(CudaMatrix& matrix, const float* hostMaster, cudaStream_t stream);

    /// <summary>
    /// cast+H2D several host FP32 masters into one fused FP16 working weight (no float pack buffer)
    /// </summary>
    static void uploadHostMastersConcatToFp16Working(
        CudaMatrix& matrix,
        const float* const* parts,
        const size_t* partElements,
        int partCount,
        cudaStream_t stream);

    /// <summary>device half pointer for a bound working weight; nullptr if unbound</summary>
    static const void* fp16WorkingWeightOrNull(const CudaMatrix& matrix);

    /// <summary>
    /// FP32 device pointer for GEMM: resident buffer, or cast of FP16 working weight into scratch
    /// </summary>
    static const float* resolveFp32Operand(const CudaMatrix& matrix, CudaDeviceBuffer& floatScratch);

    /// <summary>FP16 GEMM using CudaMatrix operands (resolves FP16 working weights)</summary>
    static bool launchCublasLtMatmulFp16Matrices(const CudaMatrix& left, const CudaMatrix& right, CudaMatrix& out, bool transposeLeft, bool transposeRight, double* kernelMilliseconds, const float* deviceBiasOrNull = nullptr);

    /// <summary>
    /// FP16 GEMM with an already-half left weight pointer (e.g. up-slice of fused gateUp) and FP32 right.
    /// </summary>
    static bool launchCublasLtMatmulFp16HalfLeft(
        const void* leftHalf,
        const float* deviceRight,
        float* deviceOut,
        int rowCount,
        int columnCount,
        int sharedCount,
        bool transposeLeft,
        bool transposeRight,
        const float* deviceBiasOrNull = nullptr);

    /// <summary>mark all registered master-weight FP16 mirrors stale (call after Adam)</summary>
    static void invalidateMasterWeightHalves();

    /// <summary>
    /// drop sticky activation FP16 identity for any slot whose FP32 source is devicePtr
    /// (call when that buffer is freed; other slots stay warm)
    /// </summary>
    static void invalidateActivationHalfCachesFor(const void* devicePtr);

    /// <summary>drop all sticky activation FP16 identities</summary>
    static void invalidateActivationHalfCaches();

    /// <summary>drop all master-weight registrations</summary>
    static void clearMasterWeights();

    static CudaDeviceBuffer floatScratchLeft;
    static CudaDeviceBuffer floatScratchRight;

private:
    static CudaDeviceBuffer halfScratchLeft;
    static CudaDeviceBuffer halfScratchRight;
    static CudaDeviceBuffer nonFiniteFlag;

    struct MasterWeightHalf {
        const float* deviceData;
        size_t elementCount;
        CudaDeviceBuffer half;
        bool valid;
        bool fp16Working;
    };

    /// <summary>4B @ 34L needs ~embed+proj+9/layer fused+2d; leave headroom</summary>
    static constexpr int maxMasterWeights = 1024;
    static MasterWeightHalf masterWeights[maxMasterWeights];
    static int masterWeightCount;

    static const void* masterWeightHalfOrNull(const float* deviceData, size_t elementCount);
};

#endif // CUDAAMP_HPP
