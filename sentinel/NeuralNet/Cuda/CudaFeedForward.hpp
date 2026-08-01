#ifndef CUDAFEEDFORWARD_HPP
#define CUDAFEEDFORWARD_HPP

#include "../Layers/FeedForward.hpp"
#include "CudaMatmul.hpp"

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

    /// <summary>create device FFN from host FFN</summary>
    static CudaFeedForward createFrom(const FeedForward& host);

    /// <summary>SwiGLU forward writing into out without host copies</summary>
    void forward(const CudaMatrix& input, CudaMatrix& out);

    /// <summary>SwiGLU backward using cached forward activations</summary>
    void backward(const CudaMatrix& outputGradient, CudaMatrix& inputGradient, CudaMatrix& gateWeightGradient, CudaMatrix& gateBiasGradient, CudaMatrix& upWeightGradient, CudaMatrix& upBiasGradient, CudaMatrix& downWeightGradient, CudaMatrix& downBiasGradient);

    /// <summary>compare CPU FeedForward vs device resident forward</summary>
    static void runSmokeDemo(int embeddingDim = 128, int sequenceLength = 64);

    /// <summary>compare CPU vs device FeedForward backward gradients</summary>
    static void runBackwardSmokeDemo(int embeddingDim = 64, int sequenceLength = 32);
};

#endif // CUDAFEEDFORWARD_HPP
