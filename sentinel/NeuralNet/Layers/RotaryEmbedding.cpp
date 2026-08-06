#include "RotaryEmbedding.hpp"

#include <cmath>
#include <stdexcept>

RotaryEmbedding::RotaryEmbedding() : headDimension(0), maximumPositionCount(0), pairCount(0), base(0.0f) {}

RotaryEmbedding::RotaryEmbedding(int headDimension, int maximumPositionCount, float base)
    : headDimension(headDimension), maximumPositionCount(maximumPositionCount), pairCount(0), base(base) {
    if (headDimension <= 0) throw std::invalid_argument("RotaryEmbedding headDimension must be > 0");
    if (headDimension % 2 != 0) throw std::invalid_argument("RotaryEmbedding headDimension must be even");
    if (maximumPositionCount <= 0) throw std::invalid_argument("RotaryEmbedding maximumPositionCount must be > 0");
    if (base <= 0.0f) throw std::invalid_argument("RotaryEmbedding base must be > 0");

    this->pairCount = headDimension / 2;
    this->cosTable.assign(static_cast<size_t>(maximumPositionCount * this->pairCount), 0.0f);
    this->sinTable.assign(static_cast<size_t>(maximumPositionCount * this->pairCount), 0.0f);

    for (int position = 0; position < maximumPositionCount; ++position) {
        for (int pairIndex = 0; pairIndex < this->pairCount; ++pairIndex) {
            const float exponent = -2.0f * static_cast<float>(pairIndex) / static_cast<float>(headDimension);
            const float angle = static_cast<float>(position) * std::pow(base, exponent);
            const size_t tableIndex = static_cast<size_t>(position * this->pairCount + pairIndex);
            this->cosTable[tableIndex] = std::cos(angle);
            this->sinTable[tableIndex] = std::sin(angle);
        }
    }
}

float RotaryEmbedding::cosAt(size_t position, size_t pairIndex) const {
    return this->cosTable[position * static_cast<size_t>(this->pairCount) + pairIndex];
}

float RotaryEmbedding::sinAt(size_t position, size_t pairIndex) const {
    return this->sinTable[position * static_cast<size_t>(this->pairCount) + pairIndex];
}

void RotaryEmbedding::rotateInPlace(Matrix& tensor, int headCount) const {
    if (this->pairCount <= 0) throw std::logic_error("RotaryEmbedding::rotateInPlace empty rope");
    if (headCount <= 0) throw std::invalid_argument("RotaryEmbedding::rotateInPlace headCount must be > 0");
    if (static_cast<int>(tensor.rows) != headCount * this->headDimension) throw std::invalid_argument("RotaryEmbedding::rotateInPlace embedding dim mismatch");
    if (static_cast<int>(tensor.cols) > this->maximumPositionCount) throw std::invalid_argument("RotaryEmbedding::rotateInPlace sequence longer than maximumPositionCount");

    for (int headIndex = 0; headIndex < headCount; ++headIndex) {
        const size_t rowOffset = static_cast<size_t>(headIndex * this->headDimension);
        for (size_t position = 0; position < tensor.cols; ++position) {
            for (int pairIndex = 0; pairIndex < this->pairCount; ++pairIndex) {
                const size_t rowEven = rowOffset + static_cast<size_t>(2 * pairIndex);
                const size_t rowOdd = rowEven + 1;
                const float cosValue = this->cosAt(position, static_cast<size_t>(pairIndex));
                const float sinValue = this->sinAt(position, static_cast<size_t>(pairIndex));
                const float evenValue = tensor.at(rowEven, position);
                const float oddValue = tensor.at(rowOdd, position);
                tensor.at(rowEven, position) = evenValue * cosValue - oddValue * sinValue;
                tensor.at(rowOdd, position) = evenValue * sinValue + oddValue * cosValue;
            }
        }
    }
}

void RotaryEmbedding::rotateInverseInPlace(Matrix& tensor, int headCount) const {
    if (this->pairCount <= 0) throw std::logic_error("RotaryEmbedding::rotateInverseInPlace empty rope");
    if (headCount <= 0) throw std::invalid_argument("RotaryEmbedding::rotateInverseInPlace headCount must be > 0");
    if (static_cast<int>(tensor.rows) != headCount * this->headDimension) throw std::invalid_argument("RotaryEmbedding::rotateInverseInPlace embedding dim mismatch");
    if (static_cast<int>(tensor.cols) > this->maximumPositionCount) throw std::invalid_argument("RotaryEmbedding::rotateInverseInPlace sequence longer than maximumPositionCount");

    for (int headIndex = 0; headIndex < headCount; ++headIndex) {
        const size_t rowOffset = static_cast<size_t>(headIndex * this->headDimension);
        for (size_t position = 0; position < tensor.cols; ++position) {
            for (int pairIndex = 0; pairIndex < this->pairCount; ++pairIndex) {
                const size_t rowEven = rowOffset + static_cast<size_t>(2 * pairIndex);
                const size_t rowOdd = rowEven + 1;
                const float cosValue = this->cosAt(position, static_cast<size_t>(pairIndex));
                const float sinValue = this->sinAt(position, static_cast<size_t>(pairIndex));
                const float evenValue = tensor.at(rowEven, position);
                const float oddValue = tensor.at(rowOdd, position);
                tensor.at(rowEven, position) = evenValue * cosValue + oddValue * sinValue;
                tensor.at(rowOdd, position) = -evenValue * sinValue + oddValue * cosValue;
            }
        }
    }
}
