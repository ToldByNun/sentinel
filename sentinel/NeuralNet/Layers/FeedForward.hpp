#ifndef FEEDFORWARD_HPP
#define FEEDFORWARD_HPP

#include "../Math/Matrix.hpp"

/// <summary>thread local intermediates for one FeedForward forward</summary>
class FeedForwardCache {
public:
    Matrix input;
    Matrix hiddenPreActivation;
    Matrix hiddenActivated;
};

/// <summary>
/// position wise MLP: Dense -> ReLU -> Dense
/// input/output shape: embeddingDim x sequenceLength
/// </summary>
class FeedForward {
public:
    Matrix firstWeight;
    Matrix firstBias;
    Matrix secondWeight;
    Matrix secondBias;

    FeedForward(Matrix firstWeight, Matrix firstBias, Matrix secondWeight, Matrix secondBias);

    /// <summary>create expand then project weights (hidden = embeddingDim * expandRatio)</summary>
    static FeedForward create(int embeddingDim, int expandRatio = 4, unsigned seed = 41u);

    /// <summary>forward writes cache for backprop</summary>
    Matrix forward(const Matrix& input, FeedForwardCache& cache) const;

    /// <summary>
    /// backprop through the MLP
    /// returns input gradient and fills weight/bias gradients
    /// </summary>
    Matrix backward(const Matrix& outputGradient, const FeedForwardCache& cache, Matrix& firstWeightGradient, Matrix& firstBiasGradient, Matrix& secondWeightGradient, Matrix& secondBiasGradient) const;

private:
    /// <summary>broadcast bias column across sequence length</summary>
    static Matrix broadcastBiasAdd(const Matrix& product, const Matrix& bias);

    /// <summary>sum gradient columns into a bias column vector</summary>
    static Matrix sumColumns(const Matrix& gradient);
};

#endif // FEEDFORWARD_HPP
