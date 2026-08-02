#ifndef CUDAADAM_HPP
#define CUDAADAM_HPP

#include "CudaMatmul.hpp"

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

    /// <summary>allocate zero moments matching parameter shape FP32</summary>
    static CudaAdamState zerosLike(const CudaMatrix& parameter);
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

    /// <summary>absmax quantization block size for int8 moments</summary>
    static int int8BlockSize;

    explicit CudaAdam(float learningRate, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f);

    /// <summary>advance shared time step used for bias correction</summary>
    void step();

    /// <summary>in place Adam update for one parameter tensor</summary>
    void update(CudaMatrix& parameter, CudaAdamState& state, const CudaMatrix& gradient, float gradientScale = 1.0f) const;

    /// <summary>one launch updating many parameter tensors; gradientScale multiplies grads in-kernel</summary>
    void updateMany(const CudaAdamUpdateItem* items, int itemCount, float gradientScale = 1.0f) const;

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
