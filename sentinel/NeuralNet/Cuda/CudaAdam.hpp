#ifndef CUDAADAM_HPP
#define CUDAADAM_HPP

#include "CudaMatmul.hpp"

class AdamState;

/// <summary>device resident first and second moment buffers FP32 or int8</summary>
class CudaAdamState {
public:
    CudaMatrix firstMoment;
    CudaMatrix secondMoment;
    CudaByteBuffer firstMomentQ;
    CudaByteBuffer secondMomentQ;
    CudaDeviceBuffer firstMomentScales;
    CudaDeviceBuffer secondMomentScales;
    int elementCount;
    int scaleCount;

    CudaAdamState();

    /// <summary>true when no moment storage allocated yet</summary>
    bool empty() const;

    /// <summary>allocate zero FP32 moments matching parameter shape</summary>
    void ensureFp32(const CudaMatrix& parameter);

    /// <summary>allocate zero int8 moments with per-block absmax scales</summary>
    void ensureInt8(const CudaMatrix& parameter, int blockSize);

    /// <summary>download moments to host FP32 (dequantizes int8 if needed)</summary>
    void downloadInto(AdamState& host, size_t rows, size_t cols) const;

    /// <summary>upload host FP32 moments (respects preferInt8Moments)</summary>
    void uploadFrom(const AdamState& host);

    /// <summary>allocate zero moments matching parameter shape FP32</summary>
    static CudaAdamState zerosLike(const CudaMatrix& parameter);

    /// <summary>release all device moment storage</summary>
    void free();
};

/// <summary>one parameter tensor for a packed multi-tensor Adam launch</summary>
class CudaAdamUpdateItem {
public:
    float* parameter;
    float* firstMoment;
    float* secondMoment;
    signed char* firstMomentQ;
    signed char* secondMomentQ;
    float* firstMomentScales;
    float* secondMomentScales;
    const float* gradient;
    int elementCount;
    int scaleCount;
    bool useInt8;
};

/// <summary>one tensor for CPU-offloaded Adam (device weights + host moments)</summary>
class CudaAdamCpuOffloadItem {
public:
    CudaMatrix* parameter;
    const CudaMatrix* gradient;
    AdamState* hostState;
};

/// <summary>device resident Adam optimizer with bias corrected moments</summary>
class CudaAdam {
public:
    float learningRate;
    float beta1;
    float beta2;
    float epsilon;
    int timeStep;
    CudaDeviceBuffer itemBuffer;

    /// <summary>store Adam moments as int8 + scales — default on for consumer VRAM</summary>
    static bool preferInt8Moments;

    /// <summary>
    /// ZeRO-Offload Stage-1 style: keep Adam m/v on host RAM (mutually exclusive with preferInt8Moments for train wiring)
    /// </summary>
    static bool preferCpuOffload;

    /// <summary>absmax quantization block size for int8 moments</summary>
    static int int8BlockSize;

    explicit CudaAdam(float learningRate, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f);

    /// <summary>advance shared time step used for bias correction</summary>
    void step();

    /// <summary>in place Adam update for one parameter tensor</summary>
    void update(CudaMatrix& parameter, CudaAdamState& state, const CudaMatrix& gradient, float gradientScale = 1.0f) const;

    /// <summary>one launch updating many parameter tensors; gradientScale multiplies grads in-kernel</summary>
    void updateMany(const CudaAdamUpdateItem* items, int itemCount, float gradientScale = 1.0f) const;

    /// <summary>
    /// CPU-offloaded Adam for one tensor (async copy stream); prefer updateCpuOffloadedMany for batches
    /// </summary>
    void updateCpuOffloaded(CudaMatrix& parameter, AdamState& hostState, const CudaMatrix& gradient, float gradientScale = 1.0f) const;

    /// <summary>
    /// async bulk D2H of all params+grads, OpenMP host Adam, async bulk H2D — one sync each side
    /// </summary>
    void updateCpuOffloadedMany(const CudaAdamCpuOffloadItem* items, int itemCount, float gradientScale = 1.0f) const;

    /// <summary>compare host Adam vs device Adam one update step</summary>
    static void runSmokeDemo(int parameterRows = 128, int parameterCols = 64);

private:
#ifdef __CUDACC__
    __device__ static void runUpdate(float* parameter, float* firstMoment, float* secondMoment, const float* gradient, int elementCount, float learningRate, float beta1, float beta2, float epsilon, float inverseFirstCorrection, float inverseSecondCorrection, float gradientScale);
    __device__ static void runUpdateMany(const CudaAdamUpdateItem* items, int itemCount, float learningRate, float beta1, float beta2, float epsilon, float inverseFirstCorrection, float inverseSecondCorrection, float gradientScale);

    friend __global__ void CudaAdamUpdateEntry(float* parameter, float* firstMoment, float* secondMoment, const float* gradient, int elementCount, float learningRate, float beta1, float beta2, float epsilon, float inverseFirstCorrection, float inverseSecondCorrection, float gradientScale);
    friend __global__ void CudaAdamUpdateManyEntry(const CudaAdamUpdateItem* items, int itemCount, float learningRate, float beta1, float beta2, float epsilon, float inverseFirstCorrection, float inverseSecondCorrection, float gradientScale);
    friend __global__ void CudaAdamUpdateInt8TensorEntry(float* parameter, signed char* firstMomentQ, signed char* secondMomentQ, float* firstMomentScales, float* secondMomentScales, const float* gradient, int elementCount, int blockSize, float learningRate, float beta1, float beta2, float epsilon, float inverseFirstCorrection, float inverseSecondCorrection, float gradientScale);
#endif
};

#ifdef __CUDACC__
__global__ void CudaAdamUpdateEntry(float* parameter, float* firstMoment, float* secondMoment, const float* gradient, int elementCount, float learningRate, float beta1, float beta2, float epsilon, float inverseFirstCorrection, float inverseSecondCorrection, float gradientScale);
__global__ void CudaAdamUpdateManyEntry(const CudaAdamUpdateItem* items, int itemCount, float learningRate, float beta1, float beta2, float epsilon, float inverseFirstCorrection, float inverseSecondCorrection, float gradientScale);
__global__ void CudaAdamUpdateInt8TensorEntry(float* parameter, signed char* firstMomentQ, signed char* secondMomentQ, float* firstMomentScales, float* secondMomentScales, const float* gradient, int elementCount, int blockSize, float learningRate, float beta1, float beta2, float epsilon, float inverseFirstCorrection, float inverseSecondCorrection, float gradientScale);
#endif

#endif // CUDAADAM_HPP
