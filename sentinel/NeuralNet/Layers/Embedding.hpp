#ifndef EMBEDDING_HPP
#define EMBEDDING_HPP

#include "../Math/Matrix.hpp"

#include <vector>

/// <summary>token lookup table weight = vocabSize x embeddingDim</summary>
class Embedding {
public:
    Matrix weight;
    std::vector<int> lastTokenIds;

    Embedding(int vocabSize, int embeddingDim);
    explicit Embedding(Matrix weight);

    int vocabSize() const;
    int embeddingDim() const;

    /// <summary>lookup embeddings for each token id</summary>
    /// <param name="tokenIds">BPE token ids</param>
    /// <returns>embeddingDim x sequenceLength (one column per token)</returns>
    Matrix forward(const std::vector<int>& tokenIds);

    /// <summary>push gradients back into the embedding rows that were used</summary>
    /// <param name="outputGradient">gradient wrt forward output (embeddingDim x sequenceLength)</param>
    /// <returns>gradient for weight (vocabSize x embeddingDim)</returns>
    Matrix backward(const Matrix& outputGradient) const;
};

#endif // EMBEDDING_HPP
