#ifndef EMBEDDING_HPP
#define EMBEDDING_HPP

#include "../Math/Matrix.hpp"

#include <vector>

/// <summary>token lookup table / weight = vocabSize x embeddingDim</summary>
class Embedding {
public:
    Matrix weight;
    std::vector<int> lastTokenIds;

    Embedding(int vocabSize, int embeddingDim);
    explicit Embedding(Matrix weight);

    int vocabSize() const;
    int embeddingDim() const;

    /// <summary>lookup embeddings for each token id</summary>
    /// <param name="tokenIds">BPE ids</param>
    /// <returns>embeddingDim x seqLen (one column per token)</returns>
    Matrix forward(const std::vector<int>& tokenIds);

    /// <summary>push grads back into the embedding rows that were used</summary>
    /// <param name="outputGradient">grad w.r.t. forward output (embeddingDim x seqLen)</param>
    /// <returns>grad for weight (vocabSize x embeddingDim)</returns>
    Matrix backward(const Matrix& outputGradient) const;
};

#endif // EMBEDDING_HPP
