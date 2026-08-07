#ifndef CUDASPULSE_HPP
#define CUDASPULSE_HPP

#include "../Optimizers/Spulse.hpp"
#include "CudaMatmul.hpp"

#include <cstdint>
#include <vector>

class CudaMatrix;
class CudaTransformerBlock;
class CudaTransformerBlockGradients;

/// <summary>device momentum + energy pair (e_fast, e_slow) for one parameter</summary>
class CudaSpulseState {
public:
    size_t rows = 0;
    size_t cols = 0;
    SpulseMomentumStorage storage = SpulseMomentumStorage::Fp32;
    int int8BlockSize = 256;
    int scaleCount = 0;

    /// <summary>FP32 momentum (storage == Fp32)</summary>
    CudaMatrix momentum;
    /// <summary>FP16 momentum bytes (storage == Fp16); elementCount * sizeof(__half)</summary>
    CudaDeviceBuffer momentumHalf;
    /// <summary>int8 momentum (storage == Int8)</summary>
    CudaByteBuffer momentumQ;
    /// <summary>per-block absmax scales for int8 (storage == Int8)</summary>
    CudaDeviceBuffer momentumScales;
    /// <summary>device buffer: [0]=e_fast, [1]=e_slow, [2]=lagged scale</summary>
    CudaDeviceBuffer energy;

    CudaSpulseState() = default;

    bool empty() const;
    void ensure(
        size_t rows,
        size_t cols,
        SpulseMomentumStorage storage = SpulseMomentumStorage::Fp32,
        int int8BlockSize = 256);
    void ensure(
        const CudaMatrix& parameter,
        SpulseMomentumStorage storage = SpulseMomentumStorage::Fp32,
        int int8BlockSize = 256);
    void free();

    void downloadInto(SpulseState& host) const;
    void uploadFrom(const SpulseState& host);
};

/// <summary>device SPULSE state for hidden 2D weights in one block (per-tensor; no fuse memcpy)</summary>
class CudaTransformerBlockSpulseStates {
public:
    CudaSpulseState queryWeight;
    CudaSpulseState keyWeight;
    CudaSpulseState valueWeight;
    CudaSpulseState attentionOutputWeight;
    CudaSpulseState feedForwardGateWeight;
    CudaSpulseState feedForwardUpWeight;
    CudaSpulseState feedForwardDownWeight;

    void ensureFrom(
        const CudaTransformerBlock& block,
        SpulseMomentumStorage storage = SpulseMomentumStorage::Fp32,
        int int8BlockSize = 256);
    void free();
};

/// <summary>host SPULSE state for large 2D weights (fused-half host path; per Q/K/V/… piece)</summary>
class CudaTransformerBlockHostSpulseStates {
public:
    SpulseState queryWeight;
    SpulseState keyWeight;
    SpulseState valueWeight;
    SpulseState attentionOutputWeight;
    SpulseState feedForwardGateWeight;
    SpulseState feedForwardUpWeight;
    SpulseState feedForwardDownWeight;

    void clear();
};

/// <summary>one fused-half host piece for SPULSE apply (filled via CudaSbao::takeReady…)</summary>
struct SpulseFusedHalfHostPiece {
    Matrix* host = nullptr;
    SpulseState* state = nullptr;
    const std::uint16_t* gradHalf = nullptr;
    size_t rows = 0;
    size_t cols = 0;
    size_t elementCount = 0;
};

/// <summary>
/// SPULSE optimizer: single momentum + dual-horizon energy scale.
/// Keep update math, host/GPU kernels, hybrid block apply, and coverage here so
/// formula changes stay local to CudaSPULSE.*.
/// </summary>
class CudaSpulse {
public:
    float learningRate;
    float momentumBeta;
    float fastBeta;
    float slowBeta;
    float epsilon;
    float scaleMin;
    float scaleMax;
    float weightDecay;
    SpulseCoverage coverage;
    /// <summary>
    /// Host fused-half VRAM fallback: skip device momentum; host apply = dual-horizon scale × grad.
    /// Default false: keep <c>u</c> on GPU, bake <c>delta = lr·s·u</c> into the grad buffer before
    /// fused-half D2H, then HostSGD-shaped host axpy (same PCIe volume as HostSGD).
    /// </summary>
    bool hostLightweight;
    /// <summary>device <c>u</c> precision (Fp32 default; Fp16/Int8 for VRAM experiments)</summary>
    SpulseMomentumStorage momentumStorage;
    /// <summary>absmax block size when momentumStorage == Int8</summary>
    int int8BlockSize;
    /// <summary>optimizer step count for momentum bias correction (Adam-style)</summary>
    int timeStep;

    /// <summary>scratch for ||g||² reduction (device); reused across updates</summary>
    CudaDeviceBuffer sumSquaresScratch;
    /// <summary>device descriptors for batched host-delta prepare (Hybrid block)</summary>
    CudaDeviceBuffer deltaPieceScratch;
    /// <summary>
    /// Pieces waiting for <c>commitPendingHybridHostEnergies</c> after host-delta bake.
    /// Allows energy commit to overlap fused-half D2H on the copy stream.
    /// </summary>
    int pendingHostEnergyPieceCount = 0;

    explicit CudaSpulse(
        float learningRate = 3e-3f,
        float momentumBeta = 0.9f,
        float fastBeta = 0.9f,
        float slowBeta = 0.999f,
        float epsilon = 1e-8f,
        float scaleMin = 0.25f,
        float scaleMax = 4.0f,
        float weightDecay = 0.0f,
        SpulseCoverage coverage = SpulseCoverage::Hybrid,
        bool hostLightweight = false,
        SpulseMomentumStorage momentumStorage = SpulseMomentumStorage::Fp32,
        int int8BlockSize = 256);

    /// <summary>advance timeStep (call once per optimizer step before updates)</summary>
    void step();

    /// <summary>1 / (1 - β^t); 1 when timeStep==0</summary>
    float momentumBiasCorrection() const;

    /// <summary>true when this tensor class is owned by SPULSE under current coverage</summary>
    bool ownsHybridBlockWeights() const;
    bool ownsFullModelWeights() const;

    /// <summary>GPU in-place update for one parameter</summary>
    void update(CudaMatrix& parameter, CudaSpulseState& state, const CudaMatrix& gradient, float gradientScale = 1.0f);

    /// <summary>
    /// Host fused-half (GPU-<c>u</c>): update momentum/energy from device grads and overwrite the
    /// buffer with <c>delta = lr · lagged_scale · u</c> for half D2H + host axpy. Does not touch θ.
    /// </summary>
    void prepareHostDeltaInPlace(
        float* gradientOrDelta,
        size_t rows,
        size_t cols,
        CudaSpulseState& state,
        float gradientScale = 1.0f);

    /// <summary>
    /// Bake <c>delta = lr·s·û</c> for every Hybrid host-offload weight in a block
    /// (same layout as <c>enqueueDeferredHostWeightGradDownloads</c>).
    /// Energy commit is deferred to <c>commitPendingHybridHostEnergies</c> so D2H can overlap.
    /// </summary>
    void prepareHybridBlockHostDeltas(
        CudaTransformerBlock& block,
        CudaTransformerBlockSpulseStates& spulseStates,
        float gradientScale);

    /// <summary>
    /// Commit dual-horizon energy for the last batched host-delta bake (no-op if nothing pending).
    /// Call after recording the host-grad compute event so commit overlaps copy-stream D2H.
    /// </summary>
    void commitPendingHybridHostEnergies();

    /// <summary>host FP32 update (masters / reference)</summary>
    void updateHost(Matrix& parameter, SpulseState& state, const Matrix& gradient, float gradientScale = 1.0f) const;

    /// <summary>host update from FP16 grad slab (fused-half D2H path; lite / legacy host-<c>u</c>)</summary>
    void updateHostFromHalf(
        Matrix& parameter,
        SpulseState& state,
        const std::uint16_t* gradHalf,
        size_t elementCount,
        float gradientScale) const;

    /// <summary>apply SPULSE to every piece in a fused-half ready batch view</summary>
    void applyFusedHalfHostPieces(const std::vector<SpulseFusedHalfHostPiece>& pieces, float gradientScale) const;

    /// <summary>
    /// Hybrid coverage: Q/K/V / attn-out / gate / up / down on device.
    /// Adam still owns embed / norms / biases / head via the caller.
    /// </summary>
    void applyHybridBlockWeights(
        CudaTransformerBlock& block,
        CudaTransformerBlockGradients& blockGradients,
        CudaTransformerBlockSpulseStates& spulseStates,
        float gradientScale);

    /// <summary>host reference vs GPU parity + one finite step (also spot-checks Fp16 u)</summary>
    static void runSmokeDemo(int parameterRows = 64, int parameterCols = 48);

    /// <summary>stable name for logging</summary>
    static const char* coverageName(SpulseCoverage coverage);
    static const char* momentumStorageName(SpulseMomentumStorage storage);
};

#endif // CUDASPULSE_HPP
