#ifndef CUDAADAM_HPP
#define CUDAADAM_HPP

#include "CudaMatmul.hpp"

/// <summary>device resident first and second moment buffers</summary>
class CudaAdamState {
public:
    CudaMatrix firstMoment;
    CudaMatrix secondMoment;

    /// <summary>allocate zero moments matching parameter shape</summary>
    static CudaAdamState zerosLike(const CudaMatrix& parameter);
};

/// <summary>one parameter tensor for a packed multi-tensor Adam launch</summary>
class CudaAdamUpdateItem {
public:
    float* parameter;
    float* firstMoment;
    float* secondMoment;
    const float* gradient;
    int elementCount;
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
#endif
};

#ifdef __CUDACC__
__global__ void CudaAdamUpdateEntry(float* parameter, float* firstMoment, float* secondMoment, const float* gradient, int elementCount, float learningRate, float beta1, float beta2, float epsilon, float inverseFirstCorrection, float inverseSecondCorrection, float gradientScale);
__global__ void CudaAdamUpdateManyEntry(const CudaAdamUpdateItem* items, int itemCount, float learningRate, float beta1, float beta2, float epsilon, float inverseFirstCorrection, float inverseSecondCorrection, float gradientScale);
#endif

#endif // CUDAADAM_HPP
