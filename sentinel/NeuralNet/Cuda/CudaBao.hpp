#ifndef CUDABAO_HPP
#define CUDABAO_HPP

#include <cstddef>

/// <summary>
/// Bandwidth-Aware Optimizer: pick GPU-resident int8 Adam vs host fused-half Adam/SGD
/// from free VRAM + param footprint + host RAM. Sets CudaAdam residency flags.
/// </summary>
enum class BaoMode {
    Auto = 0,
    GpuInt8Adam = 1,
    HostFusedHalfAdam = 2,
    HostFusedHalfSgd = 3
};

class CudaBao {
public:
    /// <summary>policy active (set by apply / setCudaPreferBao)</summary>
    static bool enabled;

    /// <summary>requested mode; Auto = resolve in select / resolveAndApply</summary>
    static BaoMode request;

    /// <summary>last concrete mode after apply / resolveAndApply</summary>
    static BaoMode resolved;

    /// <summary>
    /// pick residency from free VRAM, FP32 parameter bytes, and optional host-RAM hint
    /// (0 = query OS when possible, else assume generous host → prefer HostFusedHalfAdam).
    /// GpuInt8 if int8 moments + grads + weights fit with pack slack; else HostFusedHalfAdam
    /// if ~2×params fit on host; else HostFusedHalfSgd.
    /// </summary>
    static BaoMode select(size_t freeVramBytes, size_t parameterBytes, size_t hostRamHintBytes = 0);

    /// <summary>set CudaAdam prefer* flags from a concrete mode (not Auto)</summary>
    static void apply(BaoMode mode);

    /// <summary>
    /// if request==Auto, select(...) then apply; else apply(request).
    /// Returns the concrete mode applied.
    /// </summary>
    static BaoMode resolveAndApply(size_t freeVramBytes, size_t parameterBytes, size_t hostRamHintBytes = 0);

    /// <summary>best-effort available physical host RAM (0 if unknown)</summary>
    static size_t queryAvailableHostRamBytes();

    /// <summary>true when large 2D weights use fused-half D2H + async host update during bwd</summary>
    static bool pipelineHostWeightUpdate();

    /// <summary>true when that host update is Adam (vs SGD)</summary>
    static bool pipelineHostAdam();

    /// <summary>stable name for logging</summary>
    static const char* modeName(BaoMode mode);
};

#endif // CUDABAO_HPP
