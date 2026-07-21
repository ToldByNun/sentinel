#include "MeanPool.hpp"

#include <stdexcept>
#include <vector>

Matrix MeanPool::forward(const Matrix& embeddings) {
    if (embeddings.data.empty() || embeddings.data[0].empty())
        throw std::invalid_argument("MeanPool::forward expects a non-empty embedding matrix");

    const size_t embeddingDim = embeddings.data.size();
    this->lastSequenceLength = embeddings.data[0].size();

    Matrix pooled;
    pooled.data = std::vector<std::vector<float>>(embeddingDim, std::vector<float>(1, 0.0f));

    for (size_t dimension = 0; dimension < embeddingDim; ++dimension) {
        float sum = 0.0f;
        for (size_t tokenIndex = 0; tokenIndex < this->lastSequenceLength; ++tokenIndex)
            sum += embeddings.data[dimension][tokenIndex];
        pooled.data[dimension][0] = sum / static_cast<float>(this->lastSequenceLength);
    }

    return pooled;
}

Matrix MeanPool::backward(const Matrix& pooledGradient) const {
    if (this->lastSequenceLength == 0) throw std::logic_error("MeanPool::backward called before forward");
    if (pooledGradient.data.empty() || pooledGradient.data[0].size() != 1)
        throw std::invalid_argument("MeanPool::backward expects gradient shape embeddingDim x 1");

    const size_t embeddingDim = pooledGradient.data.size();
    const float scale = 1.0f / static_cast<float>(this->lastSequenceLength);

    Matrix embeddingGradient;
    embeddingGradient.data = std::vector<std::vector<float>>(
        embeddingDim,
        std::vector<float>(this->lastSequenceLength, 0.0f)
    );

    for (size_t dimension = 0; dimension < embeddingDim; ++dimension) {
        const float sharedGradient = pooledGradient.data[dimension][0] * scale;
        for (size_t tokenIndex = 0; tokenIndex < this->lastSequenceLength; ++tokenIndex)
            embeddingGradient.data[dimension][tokenIndex] = sharedGradient;
    }

    return embeddingGradient;
}
