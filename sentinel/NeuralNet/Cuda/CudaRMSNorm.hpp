#ifndef CUDARMSNORM_HPP
#define CUDARMSNORM_HPP

#include "../Layers/RMSNorm.hpp"
#include "CudaMatmul.hpp"

/// <summary>device resident RMSNorm forward and backward</summary>
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

    /// <summary>free lastInput / lastNormalized / rms scratch (gamma untouched)</summary>
    void releaseActivationScratch() const;

    /// <summary>pre-size Selective-kept norm caches (lastNormalized + invRms + shape sentinel)</summary>
    void ensureSelectiveCacheCapacity(size_t embeddingDim, size_t columns) const;

    /// <summary>out = lastNormalized * gamma (reconstructs FFN input after Selective stash)</summary>
    void reconstructNormalizedOutput(CudaMatrix& out) const;

    /// <summary>
    /// block epilogue: residualOut = left + right, then normOut = RMSNorm(residualOut)
    /// one launch; caches lastNormalized/inverseRms for backward (lastInput shape only)
    /// </summary>
    void forwardFromResidual(const CudaMatrix& left, const CudaMatrix& right, CudaMatrix& residualOut, CudaMatrix& normOut) const;

    /// <summary>backprop through RMSNorm fills gamma gradient and input gradient</summary>
    void backward(const CudaMatrix& outputGradient, CudaMatrix& inputGradient, CudaMatrix& gammaGradient) const;

    /// <summary>
    /// fused residual bwd: residualInputGrad = residualOutputGrad + d(RMSNorm)/d(residual)
    /// (covers Attn residual→input and FFN residual→afterAttention)
    /// </summary>
    void backwardThroughResidual(
        const CudaMatrix& normOutputGradient,
        const CudaMatrix& residualOutputGradient,
        CudaMatrix& residualInputGradient,
        CudaMatrix& gammaGradient) const;

    /// <summary>compare CPU RMSNorm vs device forward</summary>
    static void runSmokeDemo(int embeddingDim = 128, int sequenceLength = 64);

    /// <summary>compare CPU vs CUDA RMSNorm backward gradients</summary>
    static void runBackwardSmokeDemo(int embeddingDim = 64, int sequenceLength = 32);

    /// <summary>parity: forwardFromResidual + backwardThroughResidual vs add+norm path</summary>
    static void runResidualEpilogueSmokeDemo(int embeddingDim = 64, int sequenceLength = 48);

private:
    /// <summary>scratch inverse rms per sequence column reused across calls</summary>
    mutable CudaMatrix inverseRms;

    /// <summary>forward input cache for backward</summary>
    mutable CudaMatrix lastInput;

    /// <summary>forward normalized values before gamma scale</summary>
    mutable CudaMatrix lastNormalized;

    /// <summary>scratch for elementwise products during backward</summary>
    mutable CudaMatrix backwardScratch;

#ifdef __CUDACC__
    __device__ static void runComputeInverseRms(const float* input, float* inverseRms, int embeddingDim, int sequenceLength, float epsilon);
    __device__ static void runApply(const float* input, float* out, float* normalizedOut, const float* gamma, const float* inverseRms, int embeddingDim, int sequenceLength);
    __device__ static void runBackwardInputGradColumn(const float* outputGrad, const float* normalized, const float* gamma, float* inputGrad, float inverseRmsValue, int embeddingDim, int sequenceLength, int column, float dimension);
    __device__ static void runBackwardThroughResidualColumn(
        const float* normOutputGrad,
        const float* residualOutputGrad,
        const float* normalized,
        const float* gamma,
        float* residualInputGrad,
        float inverseRmsValue,
        int embeddingDim,
        int sequenceLength,
        int column,
        float dimension);

    friend __global__ void CudaRMSNormComputeInverseRmsEntry(const float* input, float* inverseRms, int embeddingDim, int sequenceLength, float epsilon);
    friend __global__ void CudaRMSNormApplyEntry(const float* input, float* out, float* normalizedOut, const float* gamma, const float* inverseRms, int embeddingDim, int sequenceLength);
    friend __global__ void CudaRMSNormBackwardInputGradEntry(const float* outputGrad, const float* normalized, const float* gamma, float* inputGrad, const float* inverseRms, int embeddingDim, int sequenceLength, float dimension);
    friend __global__ void CudaRMSNormBackwardThroughResidualEntry(
        const float* normOutputGrad,
        const float* residualOutputGrad,
        const float* normalized,
        const float* gamma,
        float* residualInputGrad,
        const float* inverseRms,
        int embeddingDim,
        int sequenceLength,
        float dimension);
#endif
};

#ifdef __CUDACC__
__global__ void CudaRMSNormComputeInverseRmsEntry(const float* input, float* inverseRms, int embeddingDim, int sequenceLength, float epsilon);
__global__ void CudaRMSNormApplyEntry(const float* input, float* out, float* normalizedOut, const float* gamma, const float* inverseRms, int embeddingDim, int sequenceLength);
__global__ void CudaRMSNormBackwardInputGradEntry(const float* outputGrad, const float* normalized, const float* gamma, float* inputGrad, const float* inverseRms, int embeddingDim, int sequenceLength, float dimension);
__global__ void CudaRMSNormBackwardThroughResidualEntry(
    const float* normOutputGrad,
    const float* residualOutputGrad,
    const float* normalized,
    const float* gamma,
    float* residualInputGrad,
    const float* inverseRms,
    int embeddingDim,
    int sequenceLength,
    float dimension);
#endif

#endif // CUDARMSNORM_HPP
