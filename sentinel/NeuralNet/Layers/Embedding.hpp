#ifndef EMBEDDING_HPP
#define EMBEDDING_HPP

#include "../Math/Matrix.hpp"

#include <vector>

/// <summary>token lookup table weight = vocabSize x embeddingDim</summary>
class Embedding {
public:
    Matrix weight;

    Embedding(int vocabSize, int embeddingDim);
    explicit Embedding(Matrix weight);

    int vocabSize() const;
    int embeddingDim() const;

    /// <summary>lookup embeddings for each token id (does not mutate shared state)</summary>
    Matrix forward(const std::vector<int>& tokenIds) const;

    /// <summary>push gradients back into the embedding rows that were used</summary>
    Matrix backward(const Matrix& outputGradient, const std::vector<int>& tokenIds) const;
};

#endif // EMBEDDING_HPP
