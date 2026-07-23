#ifndef SILU_HPP
#define SILU_HPP

#include "../Math/Matrix.hpp"

/// <summary>SiLU / Swish: x * sigmoid(x)</summary>
class SiLU {
public:
    /// <summary>x * sigmoid(x) writing into out</summary>
    static void applyInto(const Matrix& input, Matrix& out);

    /// <summary>derivative of SiLU writing into out</summary>
    static void derivativeInto(const Matrix& input, Matrix& out);
};

#endif // SILU_HPP
