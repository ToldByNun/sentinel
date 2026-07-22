#include "Softmax.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

Matrix Softmax::apply(const Matrix& logits) {
    if (logits.data.empty() || logits.data[0].empty()) throw std::invalid_argument("Softmax::apply expects a non-empty matrix");

    const size_t classCount = logits.data.size();
    const size_t columnCount = logits.data[0].size();

    Matrix probabilities;
    probabilities.data = std::vector<std::vector<float>>(
        classCount,
        std::vector<float>(columnCount, 0.0f)
    );

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
        float maxLogit = logits.data[0][static_cast<size_t>(column)];
        for (size_t classIndex = 1; classIndex < classCount; ++classIndex) {
            if (logits.data[classIndex][static_cast<size_t>(column)] > maxLogit)
                maxLogit = logits.data[classIndex][static_cast<size_t>(column)];
        }

        float exponentialSum = 0.0f;
        for (size_t classIndex = 0; classIndex < classCount; ++classIndex) {
            const float value = std::exp(logits.data[classIndex][static_cast<size_t>(column)] - maxLogit);
            probabilities.data[classIndex][static_cast<size_t>(column)] = value;
            exponentialSum += value;
        }

        for (size_t classIndex = 0; classIndex < classCount; ++classIndex)
            probabilities.data[classIndex][static_cast<size_t>(column)] /= exponentialSum;
    }

    return probabilities;
}
