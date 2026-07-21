#ifndef CROSSENTROPY_HPP
#define CROSSENTROPY_HPP

#include "../Math/Matrix.hpp"

/// <summary>Classification loss. Pair with Softmax.</summary>
class CrossEntropy {
public:
    /// <summary>-sum(target * log(probabilities)), averaged over columns.</summary>
    static float loss(const Matrix& probabilities, const Matrix& target);

    /// <summary>Gradient w.r.t. logits for Softmax+CrossEntropy: probabilities - target.</summary>
    static Matrix gradient(const Matrix& probabilities, const Matrix& target);
};

#endif // CROSSENTROPY_HPP
