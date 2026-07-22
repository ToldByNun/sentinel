#include "UniformInit.hpp"

#include <stdexcept>
#include <vector>

unsigned UniformInit::advanceSeed(unsigned seed) {
    return seed * 1664525u + 1013904223u;
}

float UniformInit::sampleValue(unsigned& seed, float scale) {
    seed = UniformInit::advanceSeed(seed);
    const float unit = static_cast<float>(seed % 10001u) / 10000.0f;
    return (unit * 2.0f - 1.0f) * scale;
}

float UniformInit::unitSample(unsigned& seed) {
    return (UniformInit::sampleValue(seed, 1.0f) + 1.0f) * 0.5f;
}

Matrix UniformInit::matrix(int rows, int columns, float scale, unsigned seed) {
    if (rows <= 0 || columns <= 0) throw std::invalid_argument("UniformInit::matrix invalid shape");

    Matrix result;
    result.data = std::vector<std::vector<float>>(
        static_cast<size_t>(rows),
        std::vector<float>(static_cast<size_t>(columns), 0.0f)
    );
    UniformInit::fill(result, scale, seed);
    return result;
}

void UniformInit::fill(Matrix& matrix, float scale, unsigned seed) {
    if (matrix.data.empty() || matrix.data[0].empty())
        throw std::invalid_argument("UniformInit::fill expects a non-empty matrix");

    for (size_t row = 0; row < matrix.data.size(); ++row) {
        for (size_t column = 0; column < matrix.data[row].size(); ++column) matrix.data[row][column] = UniformInit::sampleValue(seed, scale);
    }
}
