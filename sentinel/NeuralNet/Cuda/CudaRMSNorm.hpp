#ifndef CUDARMSNORM_HPP
#define CUDARMSNORM_HPP

#include "../Layers/RMSNorm.hpp"
#include "CudaMatmul.hpp"

/// <summary>device resident RMSNorm forward</summary>
class CudaRMSNorm {
public:
    CudaMatrix gamma;
    float epsilon;

    CudaRMSNorm();

    /// <summary>upload gamma and epsilon from host RMSNorm</summary>
    void uploadFrom(const RMSNorm& host);

    /// <summary>create device RMSNorm from host</summary>
    static CudaRMSNorm createFrom(const RMSNorm& host);

    /// <summary>normalize by rms then scale writing into out</summary>
    void forward(const CudaMatrix& input, CudaMatrix& out) const;

    /// <summary>compare CPU RMSNorm vs device forward</summary>
    static void runSmokeDemo(int embeddingDim = 128, int sequenceLength = 64);

private:
    /// <summary>scratch inverse rms per sequence column reused across calls</summary>
    mutable CudaMatrix inverseRms;

#ifdef __CUDACC__
    __device__ static void runComputeInverseRms(const float* input, float* inverseRms, int embeddingDim, int sequenceLength, float epsilon);
    __device__ static void runApply(const float* input, float* out, const float* gamma, const float* inverseRms, int embeddingDim, int sequenceLength);

    friend __global__ void CudaRMSNormComputeInverseRmsEntry(const float* input, float* inverseRms, int embeddingDim, int sequenceLength, float epsilon);
    friend __global__ void CudaRMSNormApplyEntry(const float* input, float* out, const float* gamma, const float* inverseRms, int embeddingDim, int sequenceLength);
#endif
};

#ifdef __CUDACC__
__global__ void CudaRMSNormComputeInverseRmsEntry(const float* input, float* inverseRms, int embeddingDim, int sequenceLength, float epsilon);
__global__ void CudaRMSNormApplyEntry(const float* input, float* out, const float* gamma, const float* inverseRms, int embeddingDim, int sequenceLength);
#endif

#endif // CUDARMSNORM_HPP
