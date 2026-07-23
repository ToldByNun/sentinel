#ifndef RMSNORM_HPP
#define RMSNORM_HPP

#include "../Math/Matrix.hpp"

#include <vector>

/// <summary>thread local intermediates for one RMSNorm forward</summary>
class RMSNormCache {
public:
    Matrix input;
    Matrix normalized;
    std::vector<float> inverseRms;
};

/// <summary>
/// RMSNorm over embedding rows for each sequence column
/// input/output shape: embeddingDim x sequenceLength
/// </summary>
class RMSNorm {
public:
    Matrix gamma;
    float epsilon;

    RMSNorm(int embeddingDim, float epsilon = 1e-5f);

    /// <summary>normalize by rms then scale writes cache for backprop</summary>
    Matrix forward(const Matrix& input, RMSNormCache& cache) const;

    /// <summary>backprop through RMSNorm fills gamma gradient and returns input gradient</summary>
    Matrix backward(const Matrix& outputGradient, const RMSNormCache& cache, Matrix& gammaGradient) const;
};

#endif // RMSNORM_HPP
