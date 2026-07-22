#include "MeanPool.hpp"

#include <stdexcept>

Matrix MeanPool::forward(const Matrix& embeddings) {
    if (embeddings.empty()) throw std::invalid_argument("MeanPool::forward expects a non-empty embedding matrix");

    const size_t embeddingDim = embeddings.rows;
    this->lastSequenceLength = embeddings.cols;

    Matrix pooled(embeddingDim, 1, 0.0f);

    for (size_t dimension = 0; dimension < embeddingDim; ++dimension) {
        float sum = 0.0f;
        for (size_t tokenIndex = 0; tokenIndex < this->lastSequenceLength; ++tokenIndex) sum += embeddings.at(dimension, tokenIndex);
        pooled.at(dimension, 0) = sum / static_cast<float>(this->lastSequenceLength);
    }

    return pooled;
}

Matrix MeanPool::backward(const Matrix& pooledGradient) const {
    if (this->lastSequenceLength == 0) throw std::logic_error("MeanPool::backward called before forward");
    if (pooledGradient.empty() || pooledGradient.cols != 1) throw std::invalid_argument("MeanPool::backward expects gradient shape embeddingDim x 1");

    const size_t embeddingDim = pooledGradient.rows;
    const float scale = 1.0f / static_cast<float>(this->lastSequenceLength);

    Matrix embeddingGradient(embeddingDim, this->lastSequenceLength, 0.0f);

    for (size_t dimension = 0; dimension < embeddingDim; ++dimension) {
        const float sharedGradient = pooledGradient.at(dimension, 0) * scale;
        for (size_t tokenIndex = 0; tokenIndex < this->lastSequenceLength; ++tokenIndex)
            embeddingGradient.at(dimension, tokenIndex) = sharedGradient;
    }

    return embeddingGradient;
}
