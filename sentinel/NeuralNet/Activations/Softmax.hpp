#ifndef SOFTMAX_HPP
#define SOFTMAX_HPP

#include "../Math/Matrix.hpp"

/// <summary>logits -> probabilities applied column wise</summary>
class Softmax {
public:
    /// <summary>stable Softmax (subtract max before exp)</summary>
    /// <param name="logits">raw scores</param>
    /// <returns>probabilities same shape sum to 1 per column</returns>
    static Matrix apply(const Matrix& logits);
};

#endif // SOFTMAX_HPP
