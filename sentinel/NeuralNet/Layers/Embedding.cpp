#include "Embedding.hpp"
#include "../Initializers/UniformInit.hpp"

#include <stdexcept>
#include <utility>
#include <vector>

Embedding::Embedding(int vocabSize, int embeddingDim) {
    if (vocabSize <= 0 || embeddingDim <= 0) throw std::invalid_argument("vocabSize and embeddingDim must be > 0");

    this->weight.resize(static_cast<size_t>(vocabSize), static_cast<size_t>(embeddingDim), 0.0f);
    UniformInit::fill(this->weight, 0.05f, 42u);
}

Embedding::Embedding(Matrix weight) : weight(std::move(weight)) {
    if (this->weight.empty()) throw std::invalid_argument("embedding weight must be non-empty");
}

int Embedding::vocabSize() const {
    return static_cast<int>(this->weight.rows);
}

int Embedding::embeddingDim() const {
    return this->weight.empty() ? 0 : static_cast<int>(this->weight.cols);
}

Matrix Embedding::forward(const std::vector<int>& tokenIds) const {
    if (tokenIds.empty()) throw std::invalid_argument("tokenIds must not be empty");

    const int dimension = this->embeddingDim();
    Matrix result(static_cast<size_t>(dimension), tokenIds.size(), 0.0f);

    for (size_t tokenIndex = 0; tokenIndex < tokenIds.size(); ++tokenIndex) {
        const int tokenId = tokenIds[tokenIndex];
        if (tokenId < 0 || tokenId >= this->vocabSize()) throw std::out_of_range("token id out of range in Embedding::forward");

        for (int dimensionIndex = 0; dimensionIndex < dimension; ++dimensionIndex)
            result.at(static_cast<size_t>(dimensionIndex), tokenIndex) = this->weight.at(static_cast<size_t>(tokenId), static_cast<size_t>(dimensionIndex));
    }

    return result;
}

Matrix Embedding::backward(const Matrix& outputGradient, const std::vector<int>& tokenIds) const {
    if (tokenIds.empty()) throw std::invalid_argument("Embedding::backward empty tokenIds");
    if (outputGradient.empty()) throw std::invalid_argument("Embedding::backward expects a non-empty gradient");
    if (outputGradient.cols != tokenIds.size()) throw std::invalid_argument("Embedding::backward sequence length mismatch");
    if (static_cast<int>(outputGradient.rows) != this->embeddingDim()) throw std::invalid_argument("Embedding::backward embedding dim mismatch");

    Matrix weightGradient(static_cast<size_t>(this->vocabSize()), static_cast<size_t>(this->embeddingDim()), 0.0f);

    const int dimension = this->embeddingDim();
    for (size_t tokenIndex = 0; tokenIndex < tokenIds.size(); ++tokenIndex) {
        const int tokenId = tokenIds[tokenIndex];
        for (int dimensionIndex = 0; dimensionIndex < dimension; ++dimensionIndex)
            weightGradient.at(static_cast<size_t>(tokenId), static_cast<size_t>(dimensionIndex)) += outputGradient.at(static_cast<size_t>(dimensionIndex), tokenIndex);
    }

    return weightGradient;
}
