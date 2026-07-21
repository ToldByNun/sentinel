#ifndef EMBEDDING_HPP
#define EMBEDDING_HPP

#include "../Math/Matrix.hpp"

#include <vector>

class Embedding {
public:
    Matrix weight;
    std::vector<int> lastTokenIds;

    Embedding(int vocabSize, int embeddingDim);
    explicit Embedding(Matrix weight);

    int vocabSize() const;
    int embeddingDim() const;

    Matrix forward(const std::vector<int>& tokenIds);
};

#endif // EMBEDDING_HPP
