#include "Embedding.hpp"

#include <stdexcept>
#include <utility>

Embedding::Embedding(int vocabSize, int embeddingDim) {
    if (vocabSize <= 0 || embeddingDim <= 0) throw std::invalid_argument("vocabSize and embeddingDim must be > 0");

    this->weight.data = std::vector<std::vector<float>>(
        static_cast<size_t>(vocabSize),
        std::vector<float>(static_cast<size_t>(embeddingDim), 0.0f)
    );
}

Embedding::Embedding(Matrix weight) : weight(std::move(weight)) {
    if (this->weight.data.empty() || this->weight.data[0].empty()) throw std::invalid_argument("embedding weight must be non-empty");
}

int Embedding::vocabSize() const {
    return static_cast<int>(this->weight.data.size());
}

int Embedding::embeddingDim() const {
    return this->weight.data.empty() ? 0 : static_cast<int>(this->weight.data[0].size());
}

Matrix Embedding::forward(const std::vector<int>& tokenIds) {
    if (tokenIds.empty()) throw std::invalid_argument("tokenIds must not be empty");

    this->lastTokenIds = tokenIds;

    const int dim = this->embeddingDim();
    Matrix result;
    result.data = std::vector<std::vector<float>>(
        static_cast<size_t>(dim),
        std::vector<float>(tokenIds.size(), 0.0f)
    );

    for (size_t t = 0; t < tokenIds.size(); ++t) {
        const int id = tokenIds[t];
        if (id < 0 || id >= this->vocabSize()) throw std::out_of_range("token id out of range in Embedding::forward");

        for (int d = 0; d < dim; ++d) {
            result.data[d][t] = this->weight.data[id][d];
        }
    }

    return result;
}
