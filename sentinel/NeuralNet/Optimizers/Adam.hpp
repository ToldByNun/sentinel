#ifndef ADAM_HPP
#define ADAM_HPP

#include "../Math/Matrix.hpp"

#include <vector>

/// <summary>first and second moment buffers for one parameter matrix</summary>
class AdamState {
public:
    Matrix firstMoment;
    Matrix secondMoment;

    /// <summary>allocate zero moments matching parameter shape</summary>
    static AdamState zerosLike(const Matrix& parameter);
};

/// <summary>
/// Adam optimizer with bias corrected moments
/// call step once per batch then update each parameter
/// </summary>
class Adam {
public:
    float learningRate;
    float beta1;
    float beta2;
    float epsilon;
    int timeStep;

    explicit Adam(
        float learningRate,
        float beta1 = 0.9f,
        float beta2 = 0.999f,
        float epsilon = 1e-8f
    );

    /// <summary>advance the shared time step used for bias correction</summary>
    void step();

    /// <summary>in place Adam update for one parameter tensor</summary>
    void update(Matrix& parameter, AdamState& state, const Matrix& gradient) const;

    /// <summary>Adam update only for selected rows (sparse embedding updates)</summary>
    void updateSelectedRows(
        Matrix& parameter,
        AdamState& state,
        const Matrix& gradient,
        const std::vector<int>& rowIndices
    ) const;

private:
    /// <summary>same shape filled with zeros</summary>
    static Matrix zerosLike(const Matrix& matrix);

    /// <summary>element wise square</summary>
    static Matrix squareElements(const Matrix& matrix);

    /// <summary>element wise sqrt</summary>
    static Matrix squareRootElements(const Matrix& matrix);

    /// <summary>element wise divide</summary>
    static Matrix divideElements(const Matrix& numerator, const Matrix& denominator);
};

#endif // ADAM_HPP
