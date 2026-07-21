#ifndef RELU_HPP
#define RELU_HPP

#include "../Math/Matrix.hpp"

class ReLU {
public:
    static Matrix apply(Matrix a);
    static Matrix derivative(Matrix a);
};

#endif // RELU_HPP