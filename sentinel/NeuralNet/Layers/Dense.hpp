#ifndef DENSE_HPP
#define DENSE_HPP

#include "Layer.hpp"
#include "../Math/Matrix.hpp"

/// <summary>fully connected layer: z = W * x + b keeps lastInput / lastZ for backprop</summary>
class Dense : public Layer {
public:
    Matrix weight;
    Matrix bias;
    Matrix lastInput;
    Matrix lastZ;

    Dense(Matrix weight, Matrix bias);

    /// <summary>linear transform does not apply activation</summary>
    /// <param name="input">incoming activations</param>
    /// <returns>z = weight * input + bias</returns>
    Matrix forward(const Matrix& input) override;
};

#endif // DENSE_HPP
