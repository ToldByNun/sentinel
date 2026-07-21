#ifndef DROPOUT_HPP
#define DROPOUT_HPP

#include "Layer.hpp"
#include "../Math/Matrix.hpp"

/// <summary>
/// inverted dropout: train scales kept units by 1 / keepProbability
/// eval is identity
/// </summary>
class Dropout : public Layer {
public:
    float dropRate;
    bool training = true;
    Matrix lastMask;
    unsigned seed;

    /// <summary>dropRate in [0, 1) fraction of units zeroed while training</summary>
    explicit Dropout(float dropRate = 0.3f, unsigned seed = 7u);

    /// <summary>apply mask when training otherwise pass through</summary>
    Matrix forward(const Matrix& input) override;

    /// <summary>route gradient only through units that were kept</summary>
    Matrix backward(const Matrix& outputGradient) const;

private:
    /// <summary>linear congruential step for the mask rng</summary>
    static unsigned advanceSeed(unsigned seed);
};

#endif // DROPOUT_HPP
