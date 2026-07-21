#ifndef EMBEDDING_HPP
#define EMBEDDING_HPP

#include "../Math/Matrix.hpp"

#include <vector>

class Embedding {
public:
    Matrix weight; // rows = vocabSize, cols = embeddingDim
    std::vector<int> lastTokenIds;

    Embedding(int vocabSize, int embeddingDim);
    explicit Embedding(Matrix weight);

    int vocabSize() const;
    int embeddingDim() const;

    // Returns embeddingDim x sequenceLength (one column per token)
    Matrix forward(const std::vector<int>& tokenIds);
};

#endif // EMBEDDING_HPP
