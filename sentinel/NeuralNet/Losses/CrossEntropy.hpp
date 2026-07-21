#ifndef CROSSENTROPY_HPP
#define CROSSENTROPY_HPP

#include "../Math/Matrix.hpp"

class CrossEntropy {
public:
    static float loss(const Matrix& probabilities, const Matrix& target);

    static Matrix gradient(const Matrix& probabilities, const Matrix& target);
};

#endif // CROSSENTROPY_HPP
