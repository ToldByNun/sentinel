#ifndef SGD_HPP
#define SGD_HPP

#include "../Math/Matrix.hpp"

/// <summary>SGD: parameter -= learningRate * gradient</summary>
class SGD {
public:
    float learningRate;

    explicit SGD(float learningRate);

    /// <summary>in place update</summary>
    void update(Matrix& parameter, const Matrix& gradient) const;
};

#endif // SGD_HPP
