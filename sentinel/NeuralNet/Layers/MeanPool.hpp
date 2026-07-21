#ifndef MEANPOOL_HPP
#define MEANPOOL_HPP

#include "../Math/Matrix.hpp"

/// <summary>mean over sequence: (dim x seqLen) -> (dim x 1)</summary>
class MeanPool {
public:
    size_t lastSequenceLength = 0;

    /// <summary>average all token columns into one vector</summary>
    /// <param name="embeddings">embeddingDim x seqLen</param>
    /// <returns>embeddingDim x 1</returns>
    Matrix forward(const Matrix& embeddings);

    /// <summary>give each token column the same share of the pooled grad ( / seqLen)</summary>
    /// <param name="pooledGradient">embeddingDim x 1</param>
    /// <returns>embeddingDim x seqLen</returns>
    Matrix backward(const Matrix& pooledGradient) const;
};

#endif // MEANPOOL_HPP
