#ifndef LAYERNORM_HPP
#define LAYERNORM_HPP

#include "../Math/Matrix.hpp"

#include <vector>

/// <summary>thread local intermediates for one LayerNorm forward</summary>
class LayerNormCache {
public:
    Matrix input;
    Matrix normalized;
    std::vector<float> mean;
    std::vector<float> inverseStd;
};

/// <summary>
/// LayerNorm over embedding rows for each sequence column
/// input/output shape: embeddingDim x sequenceLength
/// </summary>
class LayerNorm {
public:
    Matrix gamma;
    Matrix beta;
    float epsilon;

    LayerNorm(int embeddingDim, float epsilon = 1e-5f);

    /// <summary>normalize then scale/shift writes cache for backprop</summary>
    Matrix forward(const Matrix& input, LayerNormCache& cache) const;

    /// <summary>
    /// backprop through LayerNorm
    /// fills gamma/beta gradients and returns input gradient
    /// </summary>
    Matrix backward(const Matrix& outputGradient, const LayerNormCache& cache, Matrix& gammaGradient, Matrix& betaGradient) const;
};

#endif // LAYERNORM_HPP
