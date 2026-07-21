#ifndef MSE_HPP
#define MSE_HPP

#include "../Math/Matrix.hpp"

class MSE {
public:
    /// <summary>mean squared error</summary>
    static float loss(const Matrix& prediction, const Matrix& target);

    /// <summary>dL/d(prediction)</summary>
    static Matrix gradient(const Matrix& prediction, const Matrix& target);
};

#endif // MSE_HPP
