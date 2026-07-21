#ifndef SGD_HPP
#define SGD_HPP

#include "../Math/Matrix.hpp"

class SGD {
public:
    float learningRate;

    explicit SGD(float learningRate);

    void update(Matrix& parameter, const Matrix& gradient) const;
};

#endif // SGD_HPP