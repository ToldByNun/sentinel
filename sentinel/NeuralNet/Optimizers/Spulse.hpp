#ifndef SPULSE_HPP
#define SPULSE_HPP

#include "../Math/Matrix.hpp"

#include <cstdint>

/// <summary>
/// SPULSE coverage: which tensors the optimizer owns.
/// Hybrid = hidden 2D block weights (v1). Full = all params (planned).
/// </summary>
enum class SpulseCoverage : std::int32_t {
    Hybrid = 0,
    Full = 1
};

/// <summary>host momentum + dual-horizon energy scalars for one parameter</summary>
class SpulseState {
public:
    Matrix momentum;
    float energyFast = 0.0f;
    float energySlow = 0.0f;

    static SpulseState zerosLike(const Matrix& parameter);
    void ensure(const Matrix& parameter);
    void clear();
};

#endif // SPULSE_HPP
