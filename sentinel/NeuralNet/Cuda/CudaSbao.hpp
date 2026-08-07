#ifndef CUDASBAO_HPP
#define CUDASBAO_HPP

#include <cstddef>
#include <cstdint>
#include <vector>

class AdamState;
class Matrix;
class SpulseState;
typedef struct CUstream_st* cudaStream_t;

/// <summary>view of one fused-half host piece after D2H (optimizer-agnostic transport)</summary>
struct SbaoFusedHalfHostPiece {
    Matrix* host = nullptr;
    AdamState* adamState = nullptr;
    SpulseState* spulseState = nullptr;
    const std::uint16_t* gradHalf = nullptr;
    size_t rows = 0;
    size_t cols = 0;
    size_t elementCount = 0;
};

/// <summary>
/// SBAO — Sentinel Backend Adaptive Optimization: pick GPU-resident int8 Adam vs host
/// fused-half Adam/SGD from free VRAM + param footprint + host RAM. Owns the fused FP16
/// grad D2H freelist and host apply kernels used by the LanguageModel bwd pipeline.
/// </summary>
enum class SbaoMode {
    Auto = 0,
    GpuInt8Adam = 1,
    HostFusedHalfAdam = 2,
    HostFusedHalfSgd = 3
};

class CudaSbao {
public:
    /// <summary>policy active (set by apply / setCudaPreferSbao)</summary>
    static bool enabled;

    /// <summary>requested mode; Auto = resolve in select / resolveAndApply</summary>
    static SbaoMode request;

    /// <summary>last concrete mode after apply / resolveAndApply</summary>
    static SbaoMode resolved;

    /// <summary>
    /// pick residency from free VRAM, FP32 parameter bytes, and optional host-RAM hint
    /// (0 = query OS when possible, else assume generous host → prefer HostFusedHalfAdam).
    /// GpuInt8 if int8 moments + grads + weights fit with pack slack; else HostFusedHalfAdam
    /// if ~2×params fit on host; else HostFusedHalfSgd.
    /// </summary>
    static SbaoMode select(size_t freeVramBytes, size_t parameterBytes, size_t hostRamHintBytes = 0);

    /// <summary>set CudaAdam prefer* flags from a concrete mode (not Auto)</summary>
    static void apply(SbaoMode mode);

    /// <summary>
    /// if request==Auto, select(...) then apply; else apply(request).
    /// Returns the concrete mode applied.
    /// </summary>
    static SbaoMode resolveAndApply(size_t freeVramBytes, size_t parameterBytes, size_t hostRamHintBytes = 0);

    /// <summary>best-effort available physical host RAM (0 if unknown)</summary>
    static size_t queryAvailableHostRamBytes();

    /// <summary>true when large 2D weights use fused-half D2H + async host update during bwd</summary>
    static bool pipelineHostWeightUpdate();

    /// <summary>true when that host update is Adam (vs SGD)</summary>
    static bool pipelineHostAdam();

    /// <summary>stable name for logging</summary>
    static const char* modeName(SbaoMode mode);

    /// <summary>
    /// Fused FP16 gradient slab offload: cast device float grads → one half pack on stream,
    /// queue a single D2H. Cuts Host-SGD PCIe volume ~2× vs FP32 D2H.
    /// </summary>
    static void beginFusedHalfGradOffload();
    static void appendFusedHalfGradOffload(
        Matrix& hostOut,
        const float* deviceFloat,
        size_t rows,
        size_t cols,
        AdamState* hostAdamState = nullptr,
        SpulseState* hostSpulseState = nullptr);
    static void commitFusedHalfGradOffload(cudaStream_t stream);

    /// <summary>
    /// seal the open pinned-D2H batch (call after enqueueing one block's downloads,
    /// before recording the D2H-complete event)
    /// </summary>
    static void commitPinnedD2hBatch();

    /// <summary>
    /// after the matching D2H copy-stream event is complete: expand fused half slab → host floats
    /// </summary>
    static void flushPinnedD2hToHost();

    /// <summary>
    /// like flushPinnedD2hToHost, but apply SGD directly into piece host matrices (masters):
    /// master -= stepScale * half2float(grad).
    /// </summary>
    static void applyFusedHalfHostSgd(float stepScale);

    /// <summary>
    /// HostFusedHalfAdam: Adam update on piece masters from fused half grads (piece.adamState required).
    /// </summary>
    static void applyFusedHalfHostAdam(
        float learningRate,
        float beta1,
        float beta2,
        float epsilon,
        int timeStep,
        float gradScale);

    /// <summary>
    /// after D2H event sync: free fused-half device packs still held in the ready queue
    /// (host pins stay until flushPinnedD2hToHost / applyFusedHalfHostSgd)
    /// </summary>
    static void releaseCompletedFusedHalfDevices();

    /// <summary>
    /// Pop one ready fused-half batch into optimizer-agnostic pieces.
    /// Returns false if the queue is empty. Caller must releaseFusedHalfHostPin(hostPin).
    /// </summary>
    static bool takeReadyFusedHalfHostBatch(std::vector<SbaoFusedHalfHostPiece>& outPieces, std::uint16_t*& hostPin);

    /// <summary>return a host pin acquired via takeReadyFusedHalfHostBatch</summary>
    static void releaseFusedHalfHostPin(std::uint16_t* hostPin);
};

#endif // CUDASBAO_HPP
