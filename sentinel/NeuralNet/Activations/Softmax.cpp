#include "Softmax.hpp"

#include <cmath>
#include <stdexcept>

#if defined(_OPENMP)
#include <omp.h>
#endif

void Softmax::applyInto(const Matrix& logits, Matrix& out) {
    if (logits.empty()) throw std::invalid_argument("Softmax::applyInto expects a non-empty matrix");

    const size_t classCount = logits.rows;
    const size_t columnCount = logits.cols;
    out.ensureSize(classCount, columnCount);

    const bool useParallel =
#if defined(_OPENMP)
        columnCount >= 32 && !omp_in_parallel();
#else
        false;
#endif

#if defined(_OPENMP)
    #pragma omp parallel for schedule(static) if(useParallel)
#endif
    for (int column = 0; column < static_cast<int>(columnCount); ++column) {
        float maxLogit = logits.at(0, static_cast<size_t>(column));
        for (size_t classIndex = 1; classIndex < classCount; ++classIndex) {
            if (logits.at(classIndex, static_cast<size_t>(column)) > maxLogit)
                maxLogit = logits.at(classIndex, static_cast<size_t>(column));
        }

        float exponentialSum = 0.0f;
        for (size_t classIndex = 0; classIndex < classCount; ++classIndex) {
            const float value = std::exp(logits.at(classIndex, static_cast<size_t>(column)) - maxLogit);
            out.at(classIndex, static_cast<size_t>(column)) = value;
            exponentialSum += value;
        }

        for (size_t classIndex = 0; classIndex < classCount; ++classIndex)
            out.at(classIndex, static_cast<size_t>(column)) /= exponentialSum;
    }
}

Matrix Softmax::apply(const Matrix& logits) {
    Matrix probabilities;
    Softmax::applyInto(logits, probabilities);
    return probabilities;
}
