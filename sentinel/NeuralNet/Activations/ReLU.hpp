#ifndef RELU_HPP
#define RELU_HPP

#include "../Math/Matrix.hpp"

class ReLU {
public:
    /// <summary>max(0, x) element wise.</summary>
    static Matrix apply(Matrix matrix);

    /// <summary>1 where x > 0, else 0. pass lastZ in</summary>
    static Matrix derivative(Matrix matrix);
};

#endif // RELU_HPP
