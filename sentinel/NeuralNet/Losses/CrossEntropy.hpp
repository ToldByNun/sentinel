#ifndef CROSSENTROPY_HPP
#define CROSSENTROPY_HPP

#include "../Math/Matrix.hpp"

/// <summary>classification loss. pair with softmax</summary>
class CrossEntropy {
public:
    /// <summary>-sum(target * log(probs)) averaged over columns</summary>
    static float loss(const Matrix& probabilities, const Matrix& target);

    /// <summary>grad w.r.t. logits for Softmax+CE: probs - target</summary>
    static Matrix gradient(const Matrix& probabilities, const Matrix& target);
};

#endif // CROSSENTROPY_HPP
