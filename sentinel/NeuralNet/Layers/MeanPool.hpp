#ifndef MEANPOOL_HPP
#define MEANPOOL_HPP

#include "../Math/Matrix.hpp"

/// <summary>mean over sequence: (embeddingDim x sequenceLength) -> (embeddingDim x 1)</summary>
class MeanPool {
public:
    size_t lastSequenceLength = 0;

    /// <summary>average all token columns into one vector</summary>
    /// <param name="embeddings">embeddingDim x sequenceLength</param>
    /// <returns>embeddingDim x 1</returns>
    Matrix forward(const Matrix& embeddings);

    /// <summary>give each token column the same share of the pooled gradient (/ sequenceLength)</summary>
    /// <param name="pooledGradient">embeddingDim x 1</param>
    /// <returns>embeddingDim x sequenceLength</returns>
    Matrix backward(const Matrix& pooledGradient) const;
};

#endif // MEANPOOL_HPP
