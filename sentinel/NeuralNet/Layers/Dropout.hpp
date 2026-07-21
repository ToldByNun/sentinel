#ifndef DROPOUT_HPP
#define DROPOUT_HPP

#include "Layer.hpp"
#include "../Math/Matrix.hpp"

class Dropout : public Layer {
public:
    Matrix forward(const Matrix& input) override;
};

#endif // DROPOUT_HPP