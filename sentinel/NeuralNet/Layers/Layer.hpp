#ifndef LAYER_HPP
#define LAYER_HPP

#include "../Math/Matrix.hpp"

class Layer {
public:
    virtual ~Layer() = default;

    virtual Matrix forward(const Matrix& input) = 0;
};

#endif // LAYER_HPP
