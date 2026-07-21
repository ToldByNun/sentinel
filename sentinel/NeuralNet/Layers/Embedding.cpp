#include "Embedding.hpp"
#include "../Initializers/UniformInit.hpp"

#include <stdexcept>
#include <utility>
#include <vector>

Embedding::Embedding(int vocabSize, int embeddingDim) {
    if (vocabSize <= 0 || embeddingDim <= 0) throw std::invalid_argument("vocabSize and embeddingDim must be > 0");

    this->weight.data = std::vector<std::vector<float>>(
        static_cast<size_t>(vocabSize),
        std::vector<float>(static_cast<size_t>(embeddingDim), 0.0f)
    );
    UniformInit::fill(this->weight, 0.05f, 42u);
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

    const int dimension = this->embeddingDim();
    Matrix result;
    result.data = std::vector<std::vector<float>>(static_cast<size_t>(dimension), std::vector<float>(tokenIds.size(), 0.0f));

    for (size_t tokenIndex = 0; tokenIndex < tokenIds.size(); ++tokenIndex) {
        const int tokenId = tokenIds[tokenIndex];
        if (tokenId < 0 || tokenId >= this->vocabSize()) throw std::out_of_range("token id out of range in Embedding::forward");

        for (int dimensionIndex = 0; dimensionIndex < dimension; ++dimensionIndex) result.data[dimensionIndex][tokenIndex] = this->weight.data[tokenId][dimensionIndex];
    }

    return result;
}

Matrix Embedding::backward(const Matrix& outputGradient) const {
    if (this->lastTokenIds.empty()) throw std::logic_error("Embedding::backward called before forward");
    if (outputGradient.data.empty()) throw std::invalid_argument("Embedding::backward expects a non-empty gradient");
    if (outputGradient.data[0].size() != this->lastTokenIds.size()) throw std::invalid_argument("Embedding::backward sequence length mismatch");
    if (static_cast<int>(outputGradient.data.size()) != this->embeddingDim()) throw std::invalid_argument("Embedding::backward embedding dim mismatch");

    Matrix weightGradient;
    weightGradient.data = std::vector<std::vector<float>>(static_cast<size_t>(this->vocabSize()), std::vector<float>(static_cast<size_t>(this->embeddingDim()), 0.0f));

    const int dimension = this->embeddingDim();

    for (size_t tokenIndex = 0; tokenIndex < this->lastTokenIds.size(); ++tokenIndex) {
        const int tokenId = this->lastTokenIds[tokenIndex];
        for (int dimensionIndex = 0; dimensionIndex < dimension; ++dimensionIndex) weightGradient.data[tokenId][dimensionIndex] += outputGradient.data[dimensionIndex][tokenIndex];
    }

    return weightGradient;
}
