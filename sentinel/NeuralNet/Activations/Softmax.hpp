#ifndef SOFTMAX_HPP
#define SOFTMAX_HPP

#include "../Math/Matrix.hpp"

class Softmax {
public:
    static Matrix apply(const Matrix& logits);
};

#endif // SOFTMAX_HPP
