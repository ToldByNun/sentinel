#ifndef MSE_HPP
#define MSE_HPP

#include "../Math/Matrix.hpp"

class MSE {
public:
    static float loss(const Matrix& prediction, const Matrix& target);
    static Matrix gradient(const Matrix& prediction, const Matrix& target);
};

#endif // MSE_HPP