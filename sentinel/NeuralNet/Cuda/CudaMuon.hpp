#ifndef CUDAMUON_HPP
#define CUDAMUON_HPP

#include "../Optimizers/Adam.hpp"
#include "CudaMatmul.hpp"

/// <summary>device SGD-momentum buffer for one Muon 2D parameter</summary>
class CudaMuonState {
public:
    CudaMatrix momentum;

    CudaMuonState() = default;

    bool empty() const;
    void ensure(const CudaMatrix& parameter);
    void free();

    /// <summary>download momentum to host</summary>
    void downloadInto(MuonState& host) const;

    /// <summary>upload host momentum (allocates device buffer if needed)</summary>
    void uploadFrom(const MuonState& host);
};

/// <summary>accumulated GPU ms for Muon update sections (profileEnabled)</summary>
struct CudaMuonProfile {
    double momentumMs = 0.0;
    double normalizeMs = 0.0;
    double nsGemmMs = 0.0;
    double nsElemwiseMs = 0.0;
    double applyMs = 0.0;
    int updateCount = 0;

    void reset() {
        momentumMs = 0.0;
        normalizeMs = 0.0;
        nsGemmMs = 0.0;
        nsElemwiseMs = 0.0;
        applyMs = 0.0;
        updateCount = 0;
    }

    double totalMs() const {
        return momentumMs + normalizeMs + nsGemmMs + nsElemwiseMs + applyMs;
    }
};

/// <summary>
/// Muon (Keller Jordan): momentum + Newton-Schulz5 for hidden 2D weights.
/// Use Adam for embeddings, norms, biases, and LM head.
/// </summary>
class CudaMuon {
public:
    float learningRate;
    float momentumBeta;
    float weightDecay;
    int nsSteps;
    bool nesterov;
    /// <summary>if true: lr_eff = lr * 0.2 * sqrt(max(rows,cols)); else Jordan spectral scale on update</summary>
    bool adjustLrMatchRmsAdamw;

    /// <summary>when true, update() accumulates section times into profile (syncs per section)</summary>
    bool profileEnabled;
    CudaMuonProfile profile;

    CudaMatrix updateScratch;
    CudaMatrix nsX;
    CudaMatrix nsA;
    CudaMatrix nsAA;
    CudaMatrix nsB;
    CudaMatrix nsBX;
    CudaMatrix nsWork;
    CudaDeviceBuffer frobeniusScratch;

    explicit CudaMuon(
        float learningRate = 0.02f,
        float momentumBeta = 0.95f,
        float weightDecay = 0.0f,
        int nsSteps = 3,
        bool nesterov = true,
        bool adjustLrMatchRmsAdamw = true);

    void update(CudaMatrix& parameter, CudaMuonState& state, const CudaMatrix& gradient, float gradientScale = 1.0f);
    void newtonSchulz5InPlace(CudaMatrix& matrix);

    /// <summary>host NS5 vs GPU parity + one finite Muon step</summary>
    static void runSmokeDemo(int parameterRows = 64, int parameterCols = 48);
};

#endif // CUDAMUON_HPP
