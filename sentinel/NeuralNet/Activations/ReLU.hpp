#ifndef RELU_HPP
#define RELU_HPP

#include "../Math/Matrix.hpp"

class ReLU {
public:
    /// <summary>max(0, x) element wise</summary>
    static Matrix apply(Matrix matrix);

    /// <summary>max(0, x) writing into out</summary>
    static void applyInto(const Matrix& input, Matrix& out);

    /// <summary>1 where x > 0 else 0</summary>
    static Matrix derivative(Matrix matrix);

    /// <summary>1 where x > 0 else 0 writing into out</summary>
    static void derivativeInto(const Matrix& input, Matrix& out);
};

#endif // RELU_HPP
