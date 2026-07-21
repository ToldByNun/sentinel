#ifndef RELU_HPP
#define RELU_HPP

#include "../Math/Matrix.hpp"

class ReLU {
public:
    static Matrix apply(Matrix matrix);
    static Matrix derivative(Matrix matrix);
};

#endif // RELU_HPP