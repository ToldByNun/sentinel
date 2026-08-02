#ifndef LANGUAGEMODEL_HPP
#define LANGUAGEMODEL_HPP

#include "../Data/LanguageModelChunkSource.hpp"
#include "../Data/LanguageModelDataset.hpp"
#include "../Layers/Dense.hpp"
#include "../Layers/Embedding.hpp"
#include "../Layers/RMSNorm.hpp"
#include "../Layers/TransformerBlock.hpp"
#include "../Math/Matrix.hpp"
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

    LanguageModel(int vocabularySize, int embeddingDim, int maximumPositionCount, Adam optimizer, int blockCount = 2, int headCount = 4);
    ~LanguageModel();

    LanguageModel(const LanguageModel&) = delete;
    LanguageModel& operator=(const LanguageModel&) = delete;
    LanguageModel(LanguageModel&&) noexcept;
    LanguageModel& operator=(LanguageModel&&) noexcept;

    /// <summary>upload host weights to optional CUDA device mirror for forward and generate</summary>
    void enableCuda();

    /// <summary>opt in to sequential device training instead of OpenMP host training</summary>
    void enableCudaTrain();

    /// <summary>recompute transformer block activations on device backward (true → Selective, false → Off)</summary>
    void enableActivationCheckpointing(bool enabled = true);

    /// <summary>Off / Full / Selective activation checkpointing on the device train mirror</summary>
    void setActivationCheckpointMode(ActivationCheckpointMode mode);

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

    /// <summary>ZeRO-Offload Stage-1: keep Adam m/v on host RAM (disables int8 device moments)</summary>
    void setCudaPreferCpuAdamOffload(bool enabled);

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
    void applyCudaVramPackBudget(float freeFraction = 0.55f, size_t safetyReserveBytes = 1536ull * 1024ull * 1024ull);

    /// <summary>capture fixed-shape CUDA graph for packed microsteps when possible</summary>
    void setCudaPreferTrainGraph(bool enabled);

    /// <summary>synthetic packed train throughput probe (tokens/s); requires enableCudaTrain</summary>
    double probeCudaPackedTrainTokensPerSecond(int sequenceLength, int warmupSteps = 3, int timedSteps = 8);

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

    /// <summary>save/load roundtrip smoke on a tiny model</summary>
    static void runCheckpointSmokeDemo();

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
