#ifndef UNIFORMINIT_HPP
#define UNIFORMINIT_HPP

#include "../Math/Matrix.hpp"

/// <summary>fills matrices with small unique values in [-scale, scale] breaks weight symmetry</summary>
class UniformInit {
public:
    /// <summary>create rows x columns matrix with pseudo random values</summary>
    static Matrix matrix(int rows, int columns, float scale, unsigned seed);

    /// <summary>overwrite an existing matrix in place</summary>
    static void fill(Matrix& matrix, float scale, unsigned seed);

    /// <summary>advance seed and return a value in [0, 1]</summary>
    static float unitSample(unsigned& seed);

private:
    static unsigned advanceSeed(unsigned seed);
    static float sampleValue(unsigned& seed, float scale);
};

#endif // UNIFORMINIT_HPP
