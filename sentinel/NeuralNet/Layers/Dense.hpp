#ifndef DENSE_HPP
#define DENSE_HPP

#include "Layer.hpp"
#include "../Math/Matrix.hpp"

class Dense : public Layer {
public:
    Matrix weight;
    Matrix bias;
    Matrix lastInput;
    Matrix lastZ;

    Dense(Matrix weight, Matrix bias);

    Matrix forward(const Matrix& input) override;
};

#endif // DENSE_HPP