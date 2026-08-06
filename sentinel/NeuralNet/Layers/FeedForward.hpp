#ifndef FEEDFORWARD_HPP
#define FEEDFORWARD_HPP

#include "../Math/Matrix.hpp"

/// <summary>thread local intermediates and scratch for one SwiGLU FeedForward pass</summary>
class FeedForwardCache {
public:
    Matrix input;
    Matrix gatePreActivation;
    Matrix gateActivated;
    Matrix up;
    Matrix hidden;
    Matrix output;
    Matrix hiddenGradient;
    Matrix gateGradient;
    Matrix upGradient;
    Matrix siluDerivative;
    Matrix inputGradient;
    Matrix temp;
};

/// <summary>
/// position wise SwiGLU MLP: silu(gate) * up then down project
/// input/output shape: embeddingDim x sequenceLength
/// </summary>
class FeedForward {
public:
    Matrix gateWeight;
    Matrix gateBias;
    Matrix upWeight;
    Matrix upBias;
    Matrix downWeight;
    Matrix downBias;

    FeedForward(Matrix gateWeight, Matrix gateBias, Matrix upWeight, Matrix upBias, Matrix downWeight, Matrix downBias);

    /// <summary>SwiGLU hidden = (2 * embeddingDim * expandRatio) / 3 (legacy Sentinel default expandRatio=4)</summary>
    static int defaultIntermediateSize(int embeddingDim, int expandRatio = 4);

    /// <summary>create SwiGLU weights using defaultIntermediateSize(embeddingDim, expandRatio)</summary>
    static FeedForward create(int embeddingDim, int expandRatio = 4, unsigned seed = 41u);

    /// <summary>create SwiGLU weights with an explicit MLP intermediate width (e.g. HF intermediate_size)</summary>
    static FeedForward createWithIntermediateSize(int embeddingDim, int intermediateSize, unsigned seed = 41u);

    /// <summary>gate/up row count (MLP hidden / intermediate size)</summary>
    int intermediateSize() const;

    /// <summary>assert legacy expand-4 width vs createWithIntermediateSize</summary>
    static void runIntermediateSizeSmokeDemo();

    /// <summary>forward writes cache for backprop</summary>
    Matrix forward(const Matrix& input, FeedForwardCache& cache) const;

    /// <summary>backprop through SwiGLU returns input gradient and fills weight/bias gradients</summary>
    Matrix backward(const Matrix& outputGradient, FeedForwardCache& cache, Matrix& gateWeightGradient, Matrix& gateBiasGradient, Matrix& upWeightGradient, Matrix& upBiasGradient, Matrix& downWeightGradient, Matrix& downBiasGradient) const;

private:
    /// <summary>add bias column into product in place</summary>
    static void broadcastBiasAddInPlace(Matrix& product, const Matrix& bias);

    /// <summary>sum gradient columns into a bias column vector</summary>
    static void sumColumnsInto(const Matrix& gradient, Matrix& biasGradient);

    /// <summary>out = left * right element wise writing into out</summary>
    static void multiplyElementwiseInto(const Matrix& left, const Matrix& right, Matrix& out);
};

#endif // FEEDFORWARD_HPP
