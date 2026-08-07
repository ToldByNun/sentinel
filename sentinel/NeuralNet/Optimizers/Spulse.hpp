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

/// <summary>
/// Device storage for momentum <c>u</c>. FP16/Int8 cut VRAM for host-offload GPU-<c>u</c>
/// (EMA is usually smooth enough). Default Fp32 for reference / parity.
/// </summary>
enum class SpulseMomentumStorage : std::int32_t {
    Fp32 = 0,
    Fp16 = 1,
    Int8 = 2
};

/// <summary>
/// Host + device momentum and <b>one</b> dual-horizon energy triple per weight tensor
/// (never per-row / per-element). <c>scale</c> is the lagged step multiplier.
/// </summary>
class SpulseState {
public:
    Matrix momentum;
    float energyFast = 0.0f;
    float energySlow = 0.0f;
    /// <summary>lagged step scale (init 1); updated after each energy sample from dual-horizon EMA</summary>
    float scale = 1.0f;

    static SpulseState zerosLike(const Matrix& parameter);
    void ensure(const Matrix& parameter);
    void clear();
};

#endif // SPULSE_HPP
