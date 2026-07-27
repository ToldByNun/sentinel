#ifndef CUDAOPS_HPP
#define CUDAOPS_HPP

#include "CudaMatmul.hpp"

/// <summary>elementwise and broadcast ops on CudaMatrix</summary>
class CudaOps {
public:
    /// <summary>add bias column across every sequence column in place</summary>
    static void broadcastBiasAddInPlace(CudaMatrix& product, const CudaMatrix& bias);

    /// <summary>SiLU writing into out</summary>
    static void siluInto(const CudaMatrix& input, CudaMatrix& out);

    /// <summary>out = left * right element wise</summary>
    static void multiplyElementwiseInto(const CudaMatrix& left, const CudaMatrix& right, CudaMatrix& out);

private:
    static constexpr int threadCount = 256;

#ifdef __CUDACC__
    /// <summary>device body for bias broadcast CUDA forbids __global__ members</summary>
    __device__ static void runBroadcastBiasAddInPlace(float* product, const float* bias, int rowCount, int columnCount);

    /// <summary>device body for SiLU</summary>
    __device__ static void runSiluInto(const float* input, float* out, int elementCount);

    /// <summary>device body for element wise multiply</summary>
    __device__ static void runMultiplyElementwiseInto(const float* left, const float* right, float* out, int elementCount);

    friend __global__ void CudaOpsBroadcastBiasAddEntry(float* product, const float* bias, int rowCount, int columnCount);
    friend __global__ void CudaOpsSiluEntry(const float* input, float* out, int elementCount);
    friend __global__ void CudaOpsMultiplyElementwiseEntry(const float* left, const float* right, float* out, int elementCount);
#endif
};

#ifdef __CUDACC__
/// <summary>CUDA language requires free __global__ entry trampolines into CudaOps</summary>
__global__ void CudaOpsBroadcastBiasAddEntry(float* product, const float* bias, int rowCount, int columnCount);
__global__ void CudaOpsSiluEntry(const float* input, float* out, int elementCount);
__global__ void CudaOpsMultiplyElementwiseEntry(const float* left, const float* right, float* out, int elementCount);
#endif

#endif // CUDAOPS_HPP
