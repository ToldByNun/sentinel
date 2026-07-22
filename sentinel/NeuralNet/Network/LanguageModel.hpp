#ifndef LANGUAGEMODEL_HPP
#define LANGUAGEMODEL_HPP

#include "../Data/LanguageModelDataset.hpp"
#include "../Layers/CausalSelfAttention.hpp"
#include "../Layers/Dense.hpp"
#include "../Layers/Embedding.hpp"
#include "../Math/Matrix.hpp"
#include "../Optimizers/Adam.hpp"

#include <vector>

class LanguageModel;

/// <summary>accumulated gradients for one parallel training batch</summary>
class LanguageModelGradients {
public:
    Matrix tokenEmbedding;
    Matrix positionEmbedding;
    Matrix queryWeight;
    Matrix keyWeight;
    Matrix valueWeight;
    Matrix attentionOutputWeight;
    Matrix projectionWeight;
    Matrix projectionBias;

    /// <summary>zero tensors matching model parameter shapes</summary>
    static LanguageModelGradients zerosFrom(const LanguageModel& model);

    /// <summary>total += other</summary>
    void addInPlace(const LanguageModelGradients& other);

    /// <summary>scale every tensor</summary>
    void scaleInPlace(float scalar);
};

/// <summary>
/// tiny causal LM: token + position embed -> attention + residual -> vocab logits
/// </summary>
class LanguageModel {
public:
    Embedding tokenEmbedding;
    Embedding positionEmbedding;
    CausalSelfAttention attention;
    Dense outputProjection;
    Adam optimizer;

    AdamState tokenEmbeddingState;
    AdamState positionEmbeddingState;
    AdamState queryWeightState;
    AdamState keyWeightState;
    AdamState valueWeightState;
    AdamState outputWeightState;
    AdamState projectionWeightState;
    AdamState projectionBiasState;

    int maximumPositionCount;

    LanguageModel(
        int vocabularySize,
        int embeddingDim,
        int maximumPositionCount,
        Adam optimizer
    );

    /// <summary>logits vocabSize x sequenceLength</summary>
    Matrix forward(const std::vector<int>& tokenIds);

    /// <summary>mean CrossEntropy over positions for one example</summary>
    float exampleLoss(const LanguageModelExample& example);

    /// <summary>mean loss over the dataset</summary>
    float averageLoss(const LanguageModelDataset& dataset);

    /// <summary>train with parallel gradient accumulation</summary>
    void train(const LanguageModelDataset& dataset, int epochs, int logEveryEpochs = 1);

    /// <summary>train and also report testLoss every logEveryEpochs</summary>
    void train(
        const LanguageModelDataset& trainDataset,
        const LanguageModelDataset& testDataset,
        int epochs,
        int logEveryEpochs = 1,
        int batchSize = 32
    );

    /// <summary>greedy next-token generation from a prompt</summary>
    std::vector<int> generate(const std::vector<int>& promptTokenIds, int newTokenCount);

private:
    /// <summary>position ids 0 .. sequenceLength-1</summary>
    static std::vector<int> positionIds(size_t sequenceLength);

    /// <summary>sum gradient columns into a bias column vector</summary>
    static Matrix sumColumns(const Matrix& gradient);

    /// <summary>broadcast bias column across sequence length</summary>
    static Matrix broadcastBiasAdd(const Matrix& product, const Matrix& bias);

    /// <summary>forward using only stack local caches (safe under OpenMP)</summary>
    Matrix forwardLocal(
        const std::vector<int>& tokenIds,
        CausalSelfAttentionCache& attentionCache,
        Matrix& projectionInput
    ) const;

    /// <summary>forward + backward into thread local gradient bucket (weights stay read only)</summary>
    float accumulateExample(const LanguageModelExample& example, LanguageModelGradients& gradients) const;

    /// <summary>one Adam step from averaged batch gradients</summary>
    void applyGradients(const LanguageModelGradients& gradients);
};

#endif // LANGUAGEMODEL_HPP
