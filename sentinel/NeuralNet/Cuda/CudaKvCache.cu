#include "CudaKvCache.hpp"

#include "CudaOps.hpp"

#include <stdexcept>

CudaKvCache::CudaKvCache() : length(0), maximumPositionCount(0), embeddingDim(0) {}

void CudaKvCache::ensureCapacity(int embeddingDim, int maximumPositionCount) {
    if (embeddingDim <= 0) throw std::invalid_argument("CudaKvCache::ensureCapacity embeddingDim must be > 0");
    if (maximumPositionCount <= 0) throw std::invalid_argument("CudaKvCache::ensureCapacity maximumPositionCount must be > 0");

    if (this->embeddingDim != embeddingDim || this->maximumPositionCount != maximumPositionCount || this->key.empty() || this->value.empty()) {
        this->key.ensureSize(static_cast<size_t>(embeddingDim), static_cast<size_t>(maximumPositionCount));
        this->value.ensureSize(static_cast<size_t>(embeddingDim), static_cast<size_t>(maximumPositionCount));
        CudaOps::zeroInPlace(this->key);
        CudaOps::zeroInPlace(this->value);
    }

    this->embeddingDim = embeddingDim;
    this->maximumPositionCount = maximumPositionCount;
    this->length = 0;
}

void CudaKvCache::reset() {
    this->length = 0;
}

void CudaKvCache::append(const CudaMatrix& keyStep, const CudaMatrix& valueStep) {
    if (this->key.empty() || this->value.empty()) throw std::logic_error("CudaKvCache::append capacity not allocated");
    if (keyStep.empty() || valueStep.empty()) throw std::invalid_argument("CudaKvCache::append empty step");
    if (keyStep.rows != static_cast<size_t>(this->embeddingDim) || valueStep.rows != static_cast<size_t>(this->embeddingDim))
        throw std::invalid_argument("CudaKvCache::append embedding dim mismatch");
    if (keyStep.cols != valueStep.cols) throw std::invalid_argument("CudaKvCache::append step length mismatch");

    const int stepCount = static_cast<int>(keyStep.cols);
    if (this->length + stepCount > this->maximumPositionCount)
        throw std::invalid_argument("CudaKvCache::append exceeds maximumPositionCount");

    CudaOps::writeColumnsInto(this->key, this->length, keyStep);
    CudaOps::writeColumnsInto(this->value, this->length, valueStep);
    this->length += stepCount;
}
