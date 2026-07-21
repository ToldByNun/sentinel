#ifndef MEANPOOL_HPP
#define MEANPOOL_HPP

#include "../Math/Matrix.hpp"

class MeanPool {
public:
    size_t lastSequenceLength = 0;

    Matrix forward(const Matrix& embeddings);
    Matrix backward(const Matrix& pooledGradient) const;
};

#endif // MEANPOOL_HPP
