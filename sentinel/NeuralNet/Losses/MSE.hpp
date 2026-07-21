#ifndef MSE_HPP
#define MSE_HPP

#include "../Math/Matrix.hpp"

class MSE {
public:
    static Matrix compute(const Matrix& prediction, const Matrix& target);
};

#endif // MSE_HPP