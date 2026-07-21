#ifndef DROPOUT_HPP
#define DROPOUT_HPP

#include "Layer.hpp"
#include "../Math/Matrix.hpp"

// TODO: implement that (i dont want to so if anyone wants to feel free :D)
class Dropout : public Layer {
public:
    Matrix forward(const Matrix& input) override;
};

#endif // DROPOUT_HPP
