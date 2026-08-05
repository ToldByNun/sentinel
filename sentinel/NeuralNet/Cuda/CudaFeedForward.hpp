#ifndef CUDAFEEDFORWARD_HPP
#define CUDAFEEDFORWARD_HPP

#include "../Layers/FeedForward.hpp"
#include "CudaAmp.hpp"
#include "CudaMatmul.hpp"

class CudaRMSNorm;

/// <summary>
/// device resident SwiGLU FeedForward forward
/// weights stay on GPU between calls
/// </summary>
class CudaFeedForward {
public:
    CudaMatrix gateWeight;
    CudaMatrix gateBias;
    CudaMatrix upWeight;
    CudaMatrix upBias;
    CudaMatrix downWeight;
    CudaMatrix downBias;

    /// <summary>stacked [gateWeight; upWeight] for one forward GEMM</summary>
    CudaMatrix gateUpWeight;
    /// <summary>stacked [gateBias; upBias] for GEMM bias epilogue</summary>
    CudaMatrix gateUpBias;
    CudaMatrix gateUpPreActivation;
    CudaMatrix gateUpHiddenGradient;
    CudaMatrix gateUpWeightGradient;

    CudaMatrix gatePreActivation;
    CudaMatrix gateActivated;
    CudaMatrix up;
    CudaMatrix hidden;
    CudaMatrix output;

    CudaMatrix inputCache;
    CudaMatrix hiddenGradient;
    CudaMatrix upGradient;
    CudaMatrix gateGradient;
    CudaMatrix siluDerivative;
    CudaMatrix temp;

    CudaFeedForward();

    /// <summary>upload all host FeedForward weights once</summary>
    void uploadFrom(const FeedForward& host);

    /// <summary>rebuild stacked gate|up weight after Adam updates the split masters</summary>
    void syncFusedGateUpWeight();

    /// <summary>create device FFN from host FFN</summary>
    static CudaFeedForward createFrom(const FeedForward& host);

    /// <summary>SwiGLU forward writing into out without host copies</summary>
    void forward(const CudaMatrix& input, CudaMatrix& out);

    /// <summary>free forward/backward activation scratch (weights untouched)</summary>
    void releaseActivationScratch();

    /// <summary>SwiGLU backward using cached forward activations</summary>
    void backward(const CudaMatrix& outputGradient, CudaMatrix& inputGradient, CudaMatrix& gateWeightGradient, CudaMatrix& gateBiasGradient, CudaMatrix& upWeightGradient, CudaMatrix& upBiasGradient, CudaMatrix& downWeightGradient, CudaMatrix& downBiasGradient);

    /// <summary>
    /// Selective ckpt: cast gatePre to FP16 and free heavy FP32 FFN acts.
    /// up is recomputed via GEMM on restore; inputCache from RMSNorm(lastNormalized * gamma).
    /// </summary>
    void stashSelectiveHalfActivations();

    /// <summary>restore FP32 FFN acts: gatePre from half, up via GEMM, silu/hidden/inputCache</summary>
    void restoreSelectiveHalfActivations(const CudaRMSNorm& feedForwardNorm);

    /// <summary>free restored FP32 FFN acts after FFN bwd (half capacity kept for next fwd stash)</summary>
    void releaseRestoredSelectiveFp32Activations();

    /// <summary>pre-size FP16 Selective gatePre stash so VRAM tune sees real footprint</summary>
    void ensureSelectiveHalfCapacity(size_t hiddenRows, size_t columns);

    /// <summary>true after stash until restore/release</summary>
    bool selectiveHalfStashed() const { return this->selectiveStashed; }

    /// <summary>compare CPU FeedForward vs device resident forward</summary>
    static void runSmokeDemo(int embeddingDim = 128, int sequenceLength = 64);

    /// <summary>compare CPU vs device FeedForward backward gradients</summary>
    static void runBackwardSmokeDemo(int embeddingDim = 64, int sequenceLength = 32);

private:
    CudaHalfMatrix selectiveGatePreHalf;
    bool selectiveStashed = false;
};

#endif // CUDAFEEDFORWARD_HPP
