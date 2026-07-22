#ifndef SOFTMAX_HPP
#define SOFTMAX_HPP

#include "../Math/Matrix.hpp"

/// <summary>logits -> probabilities applied column wise</summary>
class Softmax {
public:
    /// <summary>stable Softmax (subtract max before exp)</summary>
    static Matrix apply(const Matrix& logits);

    /// <summary>stable Softmax writing into out</summary>
    static void applyInto(const Matrix& logits, Matrix& out);
};

#endif // SOFTMAX_HPP
