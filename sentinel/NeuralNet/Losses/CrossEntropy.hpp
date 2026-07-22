#ifndef CROSSENTROPY_HPP
#define CROSSENTROPY_HPP

#include "../Math/Matrix.hpp"

/// <summary>classification loss pair with Softmax</summary>
class CrossEntropy {
public:
    /// <summary>-sum(target * log(probabilities)) averaged over columns</summary>
    static float loss(const Matrix& probabilities, const Matrix& target);

    /// <summary>gradient wrt logits for mean Softmax+CrossEntropy: (probabilities - target) / columns</summary>
    static Matrix gradient(const Matrix& probabilities, const Matrix& target);
};

#endif // CROSSENTROPY_HPP
