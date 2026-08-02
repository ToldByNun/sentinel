#ifndef CUDALANGUAGEMODEL_HPP

#define CUDALANGUAGEMODEL_HPP



#include "../Data/LanguageModelChunkSource.hpp"
#include "../Data/LanguageModelDataset.hpp"
#include "../Network/LanguageModel.hpp"

#include "CudaAdam.hpp"

#include "CudaAmp.hpp"

#include "CudaKvCache.hpp"

#include "CudaMatmul.hpp"

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

    /// <summary>cap packed columns so attention stays manageable default high with segmented attention</summary>

    int maxPackedColumns;

    /// <summary>vocab rows per CE chunk for low VRAM LM head default 2048</summary>

    int logitChunkRows;

    /// <summary>Adam every N microbatches of batchSize</summary>

    int gradientAccumulationSteps;

    /// <summary>recompute block activations during backward default on for train</summary>

    bool activationCheckpointing;



    CudaAdam adam;

    CudaLanguageModelGradients trainGradients;

    CudaAdamState tokenEmbeddingState;

    CudaAdamState finalNormGammaState;

    CudaAdamState projectionWeightState;

    CudaAdamState projectionBiasState;

    std::vector<CudaTransformerBlockAdamStates> blockAdamStates;



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

    CudaIntBuffer meanDivisorBuffer;

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



    /// <summary>pre-allocate train activation workspaces to maxPackedColumns</summary>

    void ensureTrainWorkspaces();



    /// <summary>free block input checkpoints when checkpointing is off</summary>

    void releaseActivationCheckpoints();



    /// <summary>forward trunk embed blocks finalNorm without vocab projection</summary>

    void forwardTrunkInto(const std::vector<int>& tokenIds, int segmentLength = 0);



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

    /// <summary>compare FP32 vs int8 Adam after several packed train steps</summary>
    static void runTrainInt8AdamSmokeDemo(int vocabularySize = 1000, int embeddingDim = 64, int sequenceLength = 48, int blockCount = 2, int headCount = 4);

    /// <summary>breakdown of one packed train step embed attn ffn ce adam</summary>
    static void runTrainProfileDemo(int vocabularySize = 1000, int embeddingDim = 64, int sequenceLength = 48, int blockCount = 2, int headCount = 4, bool preferFlash = true, int maxPackedColumns = 0);



    /// <summary>compare CPU LanguageModel forward vs device forward</summary>

    static void runSmokeDemo(int vocabularySize = 128, int embeddingDim = 64, int sequenceLength = 32, int blockCount = 2, int headCount = 4);



    /// <summary>compare prefill plus decode last token vs full forward last column</summary>

    static void runKvCacheSmokeDemo(int vocabularySize = 128, int embeddingDim = 64, int sequenceLength = 32, int blockCount = 2, int headCount = 4);

};



#endif // CUDALANGUAGEMODEL_HPP

