#ifndef LANGUAGEMODEL_HPP
#define LANGUAGEMODEL_HPP

#include "../Data/LanguageModelChunkSource.hpp"
#include "../Data/LanguageModelDataset.hpp"
#include "../Layers/Dense.hpp"
#include "../Layers/Embedding.hpp"
#include "../Layers/RMSNorm.hpp"
#include "../Layers/TransformerBlock.hpp"
#include "../Math/Matrix.hpp"
#include "../Cuda/CudaSbao.hpp"
#include "../Optimizers/Adam.hpp"

#include <memory>
#include <string>
#include <vector>

class CudaLanguageModel;
class LanguageModel;

/// <summary>device train activation checkpoint policy</summary>
enum class ActivationCheckpointMode {
    Off = 0,
    Full = 1,
    Selective = 2
};

/// <summary>thread local caches for one LM forward/backward</summary>
class LanguageModelCache {
public:
    std::vector<TransformerBlockCache> blockCaches;
    RMSNormCache finalNormCache;
    Matrix blockOutput;
    Matrix probabilities;
};

/// <summary>accumulated gradients for one parallel training batch</summary>
class LanguageModelGradients {
public:
    Matrix tokenEmbedding;
    std::vector<TransformerBlockGradients> blocks;
    Matrix finalNormGamma;
    Matrix projectionWeight;
    Matrix projectionBias;

    /// <summary>zero tensors matching model parameter shapes</summary>
    static LanguageModelGradients zerosFrom(const LanguageModel& model);

    /// <summary>set all gradient tensors to zero in place</summary>
    void zeroInPlace();

    /// <summary>total += other</summary>
    void addInPlace(const LanguageModelGradients& other);

    /// <summary>scale every tensor</summary>
    void scaleInPlace(float scalar);
};

/// <summary>
/// causal LM with stacked pre-norm transformer blocks and RoPE
/// token embed -> (RMSNorm Attn SwiGLU) x N -> final RMSNorm -> vocab
/// optional CudaLanguageModel device mirror for inference forward
/// </summary>
class LanguageModel {
    friend class CudaLanguageModel;

public:
    Embedding tokenEmbedding;
    std::vector<TransformerBlock> blocks;
    RMSNorm finalNorm;
    Dense outputProjection;
    Adam optimizer;

    AdamState tokenEmbeddingState;
    AdamState finalNormGammaState;
    AdamState projectionWeightState;
    AdamState projectionBiasState;

    int maximumPositionCount;

    /// <summary>share token embedding with LM-head weight (default on; saves params + Adam VRAM)</summary>
    bool tieEmbeddingProjection;

    /// <summary>
    /// build LM; intermediateSize &lt;= 0 → legacy expand-4 SwiGLU width per block;
    /// ropeTheta is HF rope_theta (default 10000; some models use much larger bases);
    /// useBias=false → zero FFN + lm_head biases (HF models without those biases; tensors kept for CUDA);
    /// kvHeadCount&lt;=0 → MHA (same as headCount); else GQA with headCount % kvHeadCount == 0
    /// </summary>
    LanguageModel(
        int vocabularySize,
        int embeddingDim,
        int maximumPositionCount,
        Adam optimizer,
        int blockCount = 2,
        int headCount = 4,
        int intermediateSize = 0,
        float ropeTheta = RotaryEmbedding::DefaultBase,
        bool useBias = true,
        int kvHeadCount = -1);
    ~LanguageModel();

    LanguageModel(const LanguageModel&) = delete;
    LanguageModel& operator=(const LanguageModel&) = delete;
    LanguageModel(LanguageModel&&) noexcept;
    LanguageModel& operator=(LanguageModel&&) noexcept;

    /// <summary>FFN intermediate width (gate/up rows); 0 if no blocks</summary>
    int intermediateSize() const;

    /// <summary>RoPE base (HF rope_theta); DefaultBase if no blocks</summary>
    float ropeTheta() const;

    /// <summary>false → FFN/lm_head biases are fixed zeros; true → trainable biases</summary>
    bool useBias() const;

    /// <summary>K/V head count (HF num_key_value_heads); equals query heads for MHA</summary>
    int kvHeadCount() const;

    /// <summary>upload host weights to optional CUDA device mirror for forward and generate</summary>
    void enableCuda();

    /// <summary>opt in to sequential device training instead of OpenMP host training</summary>
    void enableCudaTrain();

    /// <summary>recompute transformer block activations on device backward (true → Selective, false → Off)</summary>
    void enableActivationCheckpointing(bool enabled = true);

    /// <summary>Off / Full / Selective activation checkpointing on the device train mirror</summary>
    void setActivationCheckpointMode(ActivationCheckpointMode mode);

    /// <summary>current device checkpoint mode (Off if no device)</summary>
    ActivationCheckpointMode cudaActivationCheckpointMode() const;

    /// <summary>FP16 GEMMs with dynamic loss scaling for consumer GPU train</summary>
    void setCudaPreferMixedPrecision(bool enabled);

    /// <summary>cap packed token columns on device train (marks manual; skips auto VRAM budget)</summary>
    void setCudaMaxPackedColumns(int columns);

    /// <summary>current device maxPackedColumns (0 if no device)</summary>
    int cudaMaxPackedColumns() const;

    /// <summary>vocab rows per chunked CE/LM-head pass</summary>
    void setCudaLogitChunkRows(int rows);

    /// <summary>store Adam m/v as int8 block scales for low VRAM</summary>
    void setCudaPreferInt8AdamMoments(bool enabled);

    /// <summary>Muon on hidden 2D weights + Adam on embed/norms/biases/head (default on in enableCudaTrain)</summary>
    void setCudaPreferMuon(bool enabled);

    /// <summary>set Newton-Schulz iterations for Muon hidden-weight updates</summary>
    void setCudaMuonNsSteps(int steps);

    /// <summary>ZeRO-Offload Stage-1: keep Adam m/v on host RAM (disables int8 device moments)</summary>
    void setCudaPreferCpuAdamOffload(bool enabled);

    /// <summary>4B PoC path: SGD on host masters (requires cpu-adam / FP16 GPU weights offload)</summary>
    void setCudaPreferHostSgd(bool enabled);

    /// <summary>
    /// Bandwidth-Aware Optimizer: enable unified residency policy (Auto resolves GpuInt8 / HostAdam / HostSgd).
    /// </summary>
    void setCudaPreferSbao(bool enabled);

    /// <summary>SBAO mode override; Auto = selectSbaoMode from free VRAM + param footprint</summary>
    void setCudaSbaoMode(SbaoMode mode);

    /// <summary>last resolved SBAO mode (GpuInt8Adam if SBAO unused)</summary>
    SbaoMode cudaSbaoModeResolved() const;

    /// <summary>toggle flash attention on device blocks</summary>
    void setCudaPreferFlashAttention(bool enabled);

    /// <summary>true when device mirror is active</summary>
    bool cudaEnabled() const;

    /// <summary>true when train uses the device path</summary>
    bool cudaTrainEnabled() const;

    /// <summary>reupload host weights to device mirror if active</summary>
    void syncDevice();

    /// <summary>LM-head weight matrix (embedding when tied)</summary>
    Matrix& lmHeadWeight();
    const Matrix& lmHeadWeight() const;

    /// <summary>enable/disable embed↔head tying; frees untied projection weight when enabling</summary>
    void setTieEmbeddingProjection(bool enabled);

    /// <summary>trainable parameter element count (tied head shares embedding weight)</summary>
    size_t parameterElementCount() const;

    /// <summary>recompute VRAM pack budget on the device train mirror</summary>
    void applyCudaVramPackBudget(float freeFraction = 0.70f, size_t safetyReserveBytes = 2560ull * 1024ull * 1024ull);

    /// <summary>capture fixed-shape CUDA graph for packed microsteps when possible</summary>
    void setCudaPreferTrainGraph(bool enabled);

    /// <summary>synthetic packed train throughput probe (tokens/s); requires enableCudaTrain</summary>
    double probeCudaPackedTrainTokensPerSecond(int sequenceLength, int warmupSteps = 3, int timedSteps = 8);

    /// <summary>time fwd/bwd vs Adam vs Muon sections for one packed shape; prints SmokeLog</summary>
    void probeCudaTrainStepProfile(int sequenceLength, int warmupSteps = 2, int timedSteps = 4);

    /// <summary>causal LM forward to vocab logits (device mirror if enabled)</summary>
    Matrix forward(const std::vector<int>& tokenIds);

    /// <summary>mean CrossEntropy over positions for one example</summary>
    float exampleLoss(const LanguageModelExample& example);

    /// <summary>mean loss over the dataset</summary>
    float averageLoss(const LanguageModelDataset& dataset);

    /// <summary>train with parallel gradient accumulation</summary>
    void train(const LanguageModelDataset& dataset, int epochs, int logEveryEpochs = 1);

    /// <summary>train and also report testLoss every logEveryEpochs</summary>
    void train(const LanguageModelDataset& trainDataset, const LanguageModelDataset& testDataset, int epochs, int logEveryEpochs = 1, int batchSize = 32, int gradientAccumulationSteps = 4);

    /// <summary>stream train chunks from JSONL; testLoss uses the source reservoir</summary>
    void train(LanguageModelChunkSource& source, int epochs, int logEveryEpochs = 1, int batchSize = 32, int gradientAccumulationSteps = 4);

    /// <summary>next-token generation temperature&lt;=0 is greedy otherwise sample with optional topK</summary>
    std::vector<int> generate(const std::vector<int>& promptTokenIds, int newTokenCount, float temperature = 1.0f, int topK = 40, unsigned seed = 42u);

    /// <summary>write weights (+ optional Adam moments) to a binary checkpoint file</summary>
    void saveCheckpoint(const std::string& path, bool includeOptimizer = true);

    /// <summary>load weights (+ Adam moments if present) from a binary checkpoint file</summary>
    void loadCheckpoint(const std::string& path);

    /// <summary>export weights as safetensors (F32); no optimizer state</summary>
    void saveSafeTensors(const std::string& path);

    /// <summary>import weights from safetensors (F32); architecture must already match</summary>
    void loadSafeTensors(const std::string& path);

    /// <summary>save/load roundtrip smoke on a tiny model</summary>
    static void runCheckpointSmokeDemo();

    /// <summary>explicit FFN intermediate_size ctor + safetensors metadata mismatch gate</summary>
    static void runIntermediateSizeSmokeDemo();

    /// <summary>configurable rope_theta tables + safetensors metadata gate</summary>
    static void runRopeThetaSmokeDemo();

    /// <summary>useBias=false zeros + omit bias tensors on save; missing bias → zeros on load</summary>
    static void runBiasPolicySmokeDemo();

    /// <summary>kv_head_count ctor + safetensors metadata mismatch gate</summary>
    static void runKvHeadCountSmokeDemo();

    /// <summary>JSONL chunk source smoke: tiny file, one streamed epoch</summary>
    static void runStreamingSmokeDemo();

private:
    friend class CudaLanguageModel;

    std::unique_ptr<CudaLanguageModel> device;
    bool deviceStale;
    bool deviceTrainEnabled;

    /// <summary>upload host weights when device mirror is stale</summary>
    void syncDeviceIfStale();

    /// <summary>token id with highest logit in the last column</summary>
    static int argmaxLastColumn(const Matrix& logits);

    /// <summary>sample one token from last-column logits with temperature and optional topK</summary>
    static int sampleLastColumn(const Matrix& logits, float temperature, int topK, unsigned& seed);

    /// <summary>sum gradient columns into a bias column vector</summary>
    static Matrix sumColumns(const Matrix& gradient);

    /// <summary>broadcast bias column across sequence length</summary>
    static Matrix broadcastBiasAdd(const Matrix& product, const Matrix& bias);

    /// <summary>forward using only stack local caches (safe under OpenMP)</summary>
    Matrix forwardLocal(const std::vector<int>& tokenIds, LanguageModelCache& cache) const;

    /// <summary>forward + backward into thread local gradient bucket (weights stay read only)</summary>
    float accumulateExample(const LanguageModelExample& example, LanguageModelGradients& gradients, LanguageModelCache& cache) const;

    /// <summary>train one in-memory slice (OpenMP); returns sum of per-example losses</summary>
    float trainOnExamples(const LanguageModelDataset& dataset, int batchSize, std::vector<LanguageModelGradients>& threadGradients, std::vector<LanguageModelCache>& threadCaches, LanguageModelGradients& merged);

    /// <summary>one Adam step from averaged batch gradients</summary>
    void applyGradients(const LanguageModelGradients& gradients);
};

#endif // LANGUAGEMODEL_HPP
