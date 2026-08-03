#ifndef CUDALANGUAGEMODEL_HPP

#define CUDALANGUAGEMODEL_HPP



#include "../Data/LanguageModelChunkSource.hpp"
#include "../Data/LanguageModelDataset.hpp"
#include "../Network/LanguageModel.hpp"

#include "CudaAdam.hpp"

#include "CudaAmp.hpp"

#include "CudaKvCache.hpp"

#include "CudaMatmul.hpp"

#include "CudaMuon.hpp"

#include "CudaOps.hpp"

#include "CudaRMSNorm.hpp"

#include "CudaTransformerBlock.hpp"



#include <vector>



/// <summary>device resident Adam states for one transformer block</summary>
class CudaTransformerBlockAdamStates {
public:
    CudaAdamState queryWeight;
    CudaAdamState keyWeight;
    CudaAdamState valueWeight;
    CudaAdamState attentionOutputWeight;
    CudaAdamState attentionNormGamma;
    CudaAdamState feedForwardNormGamma;
    CudaAdamState feedForwardGateWeight;
    CudaAdamState feedForwardGateBias;
    CudaAdamState feedForwardUpWeight;
    CudaAdamState feedForwardUpBias;
    CudaAdamState feedForwardDownWeight;
    CudaAdamState feedForwardDownBias;

    /// <summary>allocate zero moments matching block parameter shapes</summary>
    void ensureFrom(const CudaTransformerBlock& block);

    /// <summary>release all device moment buffers</summary>
    void free();

    /// <summary>release Adam moments for hidden 2D weights (Muon owns those)</summary>
    void freeMuonManagedWeights();
};

/// <summary>device Muon momentum for hidden 2D weights in one block (fused QKV / gateUp)</summary>
class CudaTransformerBlockMuonStates {
public:
    CudaMuonState qkvWeight;
    CudaMuonState attentionOutputWeight;
    CudaMuonState feedForwardGateUpWeight;
    CudaMuonState feedForwardDownWeight;

    void ensureFrom(const CudaTransformerBlock& block);
    void free();
};

/// <summary>host Adam moments for one transformer block (CPU offload path)</summary>
class CudaTransformerBlockHostAdamStates {
public:
    AdamState queryWeight;
    AdamState keyWeight;
    AdamState valueWeight;
    AdamState attentionOutputWeight;
    AdamState attentionNormGamma;
    AdamState feedForwardNormGamma;
    AdamState feedForwardGateWeight;
    AdamState feedForwardGateBias;
    AdamState feedForwardUpWeight;
    AdamState feedForwardUpBias;
    AdamState feedForwardDownWeight;
    AdamState feedForwardDownBias;

    Matrix queryWeightMaster;
    Matrix keyWeightMaster;
    Matrix valueWeightMaster;
    Matrix attentionOutputWeightMaster;
    Matrix attentionNormGammaMaster;
    Matrix feedForwardNormGammaMaster;
    Matrix feedForwardGateWeightMaster;
    Matrix feedForwardGateBiasMaster;
    Matrix feedForwardUpWeightMaster;
    Matrix feedForwardUpBiasMaster;
    Matrix feedForwardDownWeightMaster;
    Matrix feedForwardDownBiasMaster;

    /// <summary>host-accumulated large weight grads when preferHostGradients</summary>
    Matrix queryWeightGrad;
    Matrix keyWeightGrad;
    Matrix valueWeightGrad;
    Matrix attentionOutputWeightGrad;
    Matrix feedForwardGateWeightGrad;
    Matrix feedForwardUpWeightGrad;
    Matrix feedForwardDownWeightGrad;
};



/// <summary>estimated device VRAM buckets for a train config (dimensional; no allocation)</summary>
struct CudaLanguageModelVramBreakdown {
    size_t parameterElements = 0;
    size_t fp32TrainableWeightBytes = 0;
    size_t fusedMirrorBytes = 0;
    size_t ropeTableBytes = 0;
    size_t gradientBytes = 0;
    size_t fp16AmpMirrorBytes = 0;
    size_t adamMomentBytes = 0;
    size_t muonMomentBytes = 0;
    size_t bytesPerPackedColumn = 0;
    size_t packWorkspaceBytes = 0;
    size_t displaySafetyBytes = 0;
    size_t weightsResidentBytes = 0;
    size_t staticTrainBytes = 0;
    size_t peakEstimateBytes = 0;
};

/// <summary>device resident accumulated gradients for one language model</summary>

class CudaLanguageModelGradients {

public:

    CudaMatrix tokenEmbedding;

    std::vector<CudaTransformerBlockGradients> blocks;

    CudaMatrix finalNormGamma;

    CudaMatrix projectionWeight;

    CudaMatrix projectionBias;



    /// <summary>ensure gradient tensor shapes match model parameters</summary>

    void ensureFrom(const CudaLanguageModel& model);



    /// <summary>set all gradient tensors to zero in place</summary>

    void zeroInPlace();

    /// <summary>zero all grads except tokenEmbedding (sparse-zeroed separately)</summary>
    void zeroInPlaceExceptEmbedding();



    /// <summary>total += other</summary>

    void addInPlace(const CudaLanguageModelGradients& other);



    /// <summary>scale every tensor</summary>

    void scaleInPlace(float scalar);

};



/// <summary>

/// device resident causal LM forward and training

/// token embed -> CudaTransformerBlock x N -> final RMSNorm -> vocab projection

/// </summary>

class CudaLanguageModel {

public:

    CudaMatrix tokenEmbeddingWeight;

    std::vector<CudaTransformerBlock> blocks;

    CudaRMSNorm finalNorm;

    CudaMatrix projectionWeight;

    CudaMatrix projectionBias;



    CudaMatrix hidden;

    CudaMatrix normalized;

    CudaMatrix logits;

    CudaIntBuffer tokenIdsBuffer;

    std::vector<CudaKvCache> kvCaches;

    int maximumPositionCount;

    /// <summary>share tokenEmbeddingWeight with LM head (mirrors LanguageModel::tieEmbeddingProjection)</summary>
    bool tieEmbeddingProjection;

    /// <summary>cap packed columns so attention stays manageable default high with segmented attention</summary>

    int maxPackedColumns;

    /// <summary>true when setCudaMaxPackedColumns / explicit assign should skip auto VRAM budget</summary>
    bool maxPackedColumnsManual;

    /// <summary>vocab rows per CE chunk for low VRAM LM head default 2048</summary>

    int logitChunkRows;

    /// <summary>Adam every N microbatches of batchSize</summary>

    int gradientAccumulationSteps;

    /// <summary>Off / Full (recompute whole block) / Selective (attn recompute, FFN keep)</summary>
    ActivationCheckpointMode activationCheckpointMode;

    /// <summary>capture fixed-shape packed microsteps into a CUDA Graph after warmup</summary>
    bool preferTrainGraph;

    /// <summary>Muon on hidden 2D weights; Adam on embed/norms/biases/head</summary>
    bool preferMuon;

    /// <summary>true when mode is Full or Selective</summary>
    bool activationCheckpointingActive() const;

    /// <summary>set checkpoint mode; Off releases saved block inputs</summary>
    void setActivationCheckpointMode(ActivationCheckpointMode mode);

    /// <summary>enable/disable Muon hybrid; marks train state stale</summary>
    void setPreferMuon(bool enabled);

    /// <summary>log label for mode</summary>
    static const char* activationCheckpointModeName(ActivationCheckpointMode mode);




    CudaAdam adam;

    CudaMuon muon;

    CudaLanguageModelGradients trainGradients;

    CudaAdamState tokenEmbeddingState;
    CudaAdamState finalNormGammaState;
    CudaAdamState projectionWeightState;
    CudaAdamState projectionBiasState;
    std::vector<CudaTransformerBlockAdamStates> blockAdamStates;
    std::vector<CudaTransformerBlockMuonStates> blockMuonStates;

    /// <summary>host Adam moments used when CudaAdam::preferCpuOffload (device states stay empty)</summary>
    AdamState hostTokenEmbeddingState;
    AdamState hostFinalNormGammaState;
    AdamState hostProjectionWeightState;
    AdamState hostProjectionBiasState;
    std::vector<CudaTransformerBlockHostAdamStates> hostBlockAdamStates;

    /// <summary>FP32 Adam masters on host when preferFp16GpuWeights</summary>
    Matrix hostTokenEmbeddingMaster;
    Matrix hostFinalNormGammaMaster;
    Matrix hostProjectionWeightMaster;
    Matrix hostProjectionBiasMaster;

    /// <summary>host-accumulated untied projection weight grads when preferHostGradients</summary>
    Matrix hostProjectionWeightGrad;



    CudaMatrix probabilities;

    CudaMatrix logitGradient;

    CudaMatrix hiddenGradient;

    CudaMatrix finalNormGammaGradient;

    CudaIntBuffer targetTokenIdsBuffer;

    CudaMatrix columnLossScratch;

    CudaMatrix projectionWeightGradient;

    CudaMatrix projectionBiasGradient;

    CudaMatrix blockInputGradientScratch;

    CudaMatrix normInputGradientScratch;

    CudaMatrix epochLossSum;

    /// <summary>chunked LM-head workspaces</summary>

    CudaMatrix logitChunk;

    CudaMatrix logitGradientChunk;

    CudaMatrix projectionWeightGradientChunk;

    CudaMatrix hiddenGradientChunk;

    CudaMatrix onlineSoftmaxMax;

    CudaMatrix onlineSoftmaxSumExp;

    CudaMatrix targetLogits;

    bool trainStateReady;

    std::vector<int> packedInputTokenIds;

    std::vector<int> packedTargetTokenIds;

    std::vector<int> packedMeanDivisors;

    /// <summary>fused [inputs|targets|meanDivisors] host staging for one H2D</summary>
    std::vector<int> packH2dHost;

    CudaIntBuffer packH2dDevice;

    CudaIntBuffer meanDivisorBuffer;

    /// <summary>token ids seen since last Adam step (for sparse embedding grad zero)</summary>
    std::vector<int> adamWindowTokenIds;

    CudaIntBuffer adamWindowTokenIdsBuffer;

    /// <summary>non-default stream for optional CUDA Graph packed microsteps</summary>
    cudaStream_t trainStream;
    cudaGraph_t trainGraph;
    cudaGraphExec_t trainGraphExec;
    int trainGraphSegmentLength;
    int trainGraphExampleCount;
    int trainGraphWarmups;
    int trainGraphSeenSegmentLength;
    int trainGraphSeenExampleCount;
    static constexpr int trainGraphWarmupNeeded = 2;

    static constexpr int padTargetId = -1;
    static constexpr int padInputId = 0;
    static constexpr int lengthBucketStep = 32;

    /// <summary>round true length up to bucket (capped by maximumPositionCount)</summary>
    static int lengthBucket(int trueLength, int maximumPositionCount);

    std::vector<CudaMatrix> blockInputCheckpoints;

    /// <summary>FP16 block input checkpoints when mixed precision is on</summary>

    std::vector<CudaHalfMatrix> blockInputCheckpointsHalf;

    /// <summary>float restore workspace for FP16 checkpoints during backward</summary>

    CudaMatrix checkpointRestoreScratch;



    CudaLanguageModel();



    /// <summary>upload all host LM parameters once</summary>

    void uploadFrom(const LanguageModel& host);



    /// <summary>create device LM from host LM</summary>

    static CudaLanguageModel createFrom(const LanguageModel& host);



    /// <summary>logits vocabSize x sequenceLength staying on device segmentLength packs equal-length sequences</summary>

    void forwardInto(const std::vector<int>& tokenIds, CudaMatrix& outLogits, int segmentLength = 0);



    /// <summary>logits downloaded to host</summary>

    Matrix forward(const std::vector<int>& tokenIds);



    /// <summary>reset and allocate one KV cache per block</summary>

    void resetKvCaches();



    /// <summary>prefill prefix into KV caches writing full-prefix logits</summary>

    void prefillInto(const std::vector<int>& tokenIds, CudaMatrix& outLogits);



    /// <summary>decode one new token against KV caches writing vocab x 1 logits</summary>

    void decodeInto(int tokenId, CudaMatrix& outLogits);



    /// <summary>allocate gradient and Adam state buffers from current weight shapes</summary>

    void ensureTrainState();

    /// <summary>download FP32 masters from GPU, bind FP16 working weights, free FP32 2D weights</summary>
    void materializeFp16GpuWorkingWeights();

    /// <summary>zero GPU train grads + host-offloaded weight grads</summary>
    void zeroAccumulatedGradients();

    /// <summary>zero host-offloaded large weight grads only</summary>
    void zeroHostWeightGradients();

    /// <summary>pre-allocate train activation workspaces to maxPackedColumns</summary>

    void ensureTrainWorkspaces();

    /// <summary>free column-scaled train workspaces (not weights / grads / optimizer)</summary>
    void releasePackedTrainWorkspaces();

    /// <summary>estimated device bytes of train workspaces that scale with one packed column</summary>
    size_t bytesPerPackedColumn() const;

    /// <summary>grads + Adam moments + FP16 weight mirrors still to allocate after weights are resident</summary>
    size_t estimatePendingTrainStaticBytes() const;

    /// <summary>
    /// set maxPackedColumns from cudaMemGetInfo free memory (no-op if maxPackedColumnsManual)
    /// reserves static train overhead + WDDM/display safety headroom, then applies freeFraction
    /// </summary>
    void applyVramPackBudget(float freeFraction = 0.40f, size_t safetyReserveBytes = 2560ull * 1024ull * 1024ull);

    /// <summary>largest pack example count for a fixed segment length under current maxPackedColumns</summary>
    int maxPackExamplesForSegment(int segmentLength) const;



    /// <summary>free block input checkpoints when checkpointing is off</summary>

    void releaseActivationCheckpoints();

    /// <summary>true when block inputs are stored as saturated FP16 (AMP on)</summary>
    bool useHalfActivationCheckpoints() const;

    /// <summary>LM-head weight (token embedding when tied)</summary>
    CudaMatrix& lmHeadWeight();
    const CudaMatrix& lmHeadWeight() const;



    /// <summary>forward trunk embed blocks finalNorm without vocab projection</summary>

    void forwardTrunkInto(const std::vector<int>& tokenIds, int segmentLength = 0);

    /// <summary>forward trunk using tokenIds already resident in tokenIdsBuffer</summary>
    void forwardTrunkFromDevice(size_t tokenCount, int segmentLength = 0);

    /// <summary>zero train grads after Adam; sparse-zeros embedding rows from adamWindowTokenIds</summary>
    void zeroTrainGradientsAfterAdam();



    /// <summary>chunked projection CE loss and grads into gradients using logitChunkRows</summary>

    void accumulateChunkedProjection(size_t tokenCount, int segmentLength, int exampleCountInPack, CudaLanguageModelGradients& gradients);



    /// <summary>download all weights to host language model</summary>
    void downloadTo(LanguageModel& host);

    /// <summary>download device Adam moments into host Adam states (FP32)</summary>
    void downloadOptimizerTo(LanguageModel& host);

    /// <summary>upload host Adam states to device moments</summary>
    void uploadOptimizerFrom(const LanguageModel& host);

    /// <summary>forward backward one example accumulating into gradients returns mean loss</summary>
    float accumulateExample(const LanguageModelExample& example, CudaLanguageModelGradients& gradients);



    /// <summary>forward backward one packed batch of equal-length examples</summary>
    float accumulatePackedExamples(const LanguageModelExample* const* examples, int exampleCount, CudaLanguageModelGradients& gradients);

    /// <summary>pack examples padded to bucketLength; pad targets ignored in CE</summary>
    float accumulateBucketPackedExamples(const LanguageModelExample* const* examples, int exampleCount, int bucketLength, CudaLanguageModelGradients& gradients);

    /// <summary>run packed train step from packedInput/Target/MeanDivisor host buffers</summary>
    float flushPackedHostBuffers(int segmentLength, int exampleCount, CudaLanguageModelGradients& gradients);

    /// <summary>release captured train microstep graph</summary>
    void releaseTrainGraph();

    /// <summary>device work for one packed microstep (tokens already on device)</summary>
    void runPackedTrainDevice(size_t tokenCount, int segmentLength, int exampleCount, CudaLanguageModelGradients& gradients);

    /// <summary>replay captured graph when shape matches; returns true if launched</summary>
    bool tryLaunchTrainGraph(int segmentLength, int exampleCount);

    /// <summary>capture current fixed-shape microstep into trainGraphExec (executes once during capture)</summary>
    bool captureTrainGraph(size_t tokenCount, int segmentLength, int exampleCount, CudaLanguageModelGradients& gradients);

    /// <summary>create non-blocking trainStream once</summary>
    void ensureTrainStream();



    /// <summary>Adam step updating all parameters; gradientScale multiplies grads in-kernel</summary>

    void applyGradients(CudaLanguageModelGradients& gradients, float gradientScale = 1.0f);



    /// <summary>sequential batch training loop on device</summary>
    void train(const LanguageModelDataset& trainDataset, const LanguageModelDataset& testDataset, int epochs, int logEveryEpochs, int batchSize, int gradientAccumulationSteps = 4);

    /// <summary>stream train chunks; testLoss from source.testDataset()</summary>
    void train(LanguageModelChunkSource& source, int epochs, int logEveryEpochs, int batchSize, int gradientAccumulationSteps = 4);

    /// <summary>
    /// train batches from one in-memory slice
    /// keeps Adam accum state across calls; flushRemainder forces a step at the end
    /// packStats counters are accumulated (caller zeros per epoch)
    /// </summary>
    void trainOnExamples(
        const LanguageModelDataset& dataset,
        int batchSize,
        bool flushRemainder,
        int& accumulatedExampleCount,
        int& microbatchesSinceStep,
        int& processedExampleCount,
        int& processedPredictionCount,
        int& packCount,
        int& singleExamplePackCount,
        long long& packedExampleSum,
        long long& packedTokenSum
    );

    /// <summary>mean cross entropy over dataset examples on device</summary>
    float averageLoss(const LanguageModelDataset& dataset);



    /// <summary>compare CPU vs CUDA training step and report tokens per second</summary>
    static void runTrainSmokeDemo(int vocabularySize = 64, int embeddingDim = 32, int sequenceLength = 16, int blockCount = 1, int headCount = 2);

    /// <summary>tiny packed Muon+Adam train: finite loss after a few steps</summary>
    static void runMuonTrainSmokeDemo(int vocabularySize = 128, int embeddingDim = 64, int sequenceLength = 32, int blockCount = 2, int headCount = 4);

    /// <summary>Full vs Selective checkpointing: one packed microstep loss + embed-grad max abs diff</summary>
    static void runSelectiveCheckpointParitySmokeDemo(
        int vocabularySize = 256,
        int embeddingDim = 64,
        int sequenceLength = 32,
        int blockCount = 2,
        int headCount = 4
    );

    /// <summary>compare FP32 vs int8 Adam after several packed train steps</summary>
    static void runTrainInt8AdamSmokeDemo(int vocabularySize = 1000, int embeddingDim = 64, int sequenceLength = 48, int blockCount = 2, int headCount = 4);

    /// <summary>compare GPU FP32 Adam vs CPU-offloaded Adam: weight parity + VRAM delta after moments allocate</summary>
    static void runTrainCpuAdamOffloadSmokeDemo(int vocabularySize = 2000, int embeddingDim = 128, int sequenceLength = 64, int blockCount = 2, int headCount = 4);

    /// <summary>breakdown of one packed train step; packBatchSize 0 => min(32, maxPack/seq); ckpt default off for throughput</summary>
    static void runTrainProfileDemo(int vocabularySize = 1000, int embeddingDim = 64, int sequenceLength = 48, int blockCount = 2, int headCount = 4, bool preferFlash = true, int maxPackedColumns = 0, int packBatchSize = 0, bool activationCheckpointing = false);

    /// <summary>
    /// dimensional VRAM bucket estimate (no alloc); matches train static + pack footprint math
    /// </summary>
    static CudaLanguageModelVramBreakdown estimateVramBreakdown(
        int vocabularySize,
        int embeddingDim,
        int blockCount,
        int headCount,
        int maximumPositionCount,
        ActivationCheckpointMode checkpointMode,
        bool preferMixedPrecision,
        bool preferInt8AdamMoments,
        bool preferCpuAdamOffload,
        bool preferMuon,
        bool tieEmbeddingProjection,
        int maxPackedColumns,
        int logitChunkRows = 2048,
        size_t displaySafetyBytes = 2560ull * 1024ull * 1024ull,
        bool preferFp16GpuWeights = false,
        bool preferHostGradients = false
    );

    /// <summary>
    /// ~4B-shape VRAM probe: log detailed byte buckets for current train layouts, then cudaMalloc fit check
    /// </summary>
    static void runScale4BVramProbeDemo();

    /// <summary>
    /// consumer VRAM proof: larger LM (d&gt;=256), auto pack budget, loss scale, timed packed epoch
    /// logs free/used MiB, maxPackCols, tokens/s
    /// </summary>
    static void runConsumerVramDemo(
        int vocabularySize = 8000,
        int embeddingDim = 256,
        int maximumPositionCount = 512,
        int blockCount = 4,
        int headCount = 8,
        int exampleCount = 4096,
        int epochs = 1,
        int batchSize = 32,
        int gradientAccumulationSteps = 2
    );



    /// <summary>compare CPU LanguageModel forward vs device forward</summary>

    static void runSmokeDemo(int vocabularySize = 128, int embeddingDim = 64, int sequenceLength = 32, int blockCount = 2, int headCount = 4);



    /// <summary>compare prefill plus decode last token vs full forward last column</summary>

    static void runKvCacheSmokeDemo(int vocabularySize = 128, int embeddingDim = 64, int sequenceLength = 32, int blockCount = 2, int headCount = 4);

};



#endif // CUDALANGUAGEMODEL_HPP

