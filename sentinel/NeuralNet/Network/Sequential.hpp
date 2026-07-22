#ifndef SEQUENTIAL_HPP
#define SEQUENTIAL_HPP

#include "../Data/ClassificationDataset.hpp"
#include "../Layers/Dense.hpp"
#include "../Layers/Dropout.hpp"
#include "../Layers/Embedding.hpp"
#include "../Layers/MeanPool.hpp"
#include "../Math/Matrix.hpp"
#include "../Optimizers/Adam.hpp"

#include <vector>

/// <summary>
/// text path: Embedding -> MeanPool -> Dense -> ReLU -> Dropout -> Dense -> Softmax
/// </summary>
class Sequential {
public:
    Dense layer1;
    Dense layer2;
    Dropout dropout;
    Adam optimizer;

    AdamState layer1WeightState;
    AdamState layer1BiasState;
    AdamState layer2WeightState;
    AdamState layer2BiasState;

    Sequential(Dense layer1, Dense layer2, Adam optimizer, float dropRate = 0.3f);

    /// <summary>forward from a feature vector returns class probs (eval mode)</summary>
    Matrix forward(const Matrix& input);

    /// <summary>forward from token ids (embed + pool + mlp) (eval mode)</summary>
    Matrix forward(Embedding& embedding, MeanPool& meanPool, const std::vector<int>& tokenIds);

    /// <summary>train on one matrix example (no embedding)</summary>
    void train(const Matrix& input, const Matrix& target, int epochs);

    /// <summary>train on one tokenized example</summary>
    void train(Embedding& embedding, MeanPool& meanPool, const std::vector<int>& tokenIds, const Matrix& target, int epochs);

    /// <summary>
    /// train on the full dataset
    /// backprop goes through Dense and into Embedding weights
    /// </summary>
    void train(Embedding& embedding, MeanPool& meanPool, const ClassificationDataset& dataset, int epochs);

    /// <summary>
    /// train with minibatches (accumulate grads then one update)
    /// every logEveryEpochs reports testAccuracy
    /// stops early if testAccuracy does not improve for earlyStoppingPatience checks
    /// restores best weights after stop
    /// </summary>
    void train(Embedding& embedding, MeanPool& meanPool, const ClassificationDataset& trainDataset, const ClassificationDataset& testDataset, int epochs, int logEveryEpochs = 500, int earlyStoppingPatience = 3, int batchSize = 16);

    /// <summary>argmax class for token ids</summary>
    int predictClass(Embedding& embedding, MeanPool& meanPool, const std::vector<int>& tokenIds);

    /// <summary>fraction of correctly classified examples (no training)</summary>
    float accuracy(Embedding& embedding, MeanPool& meanPool, const ClassificationDataset& dataset);

private:
    /// <summary>same shape as matrix filled with zeros</summary>
    static Matrix zerosLike(const Matrix& matrix);

    /// <summary>linear congruential step for shuffle</summary>
    static unsigned advanceSeed(unsigned seed);

    /// <summary>Fisher-Yates order of exampleCount indices</summary>
    static std::vector<size_t> shuffledOrder(size_t exampleCount, unsigned& seed);

    /// <summary>add gradient into total in place</summary>
    static void accumulateGradient(Matrix& total, const Matrix& gradient);

    /// <summary>one Adam step for both Dense layers</summary>
    void updateDenseParameters(const Matrix& weightGradient1, const Matrix& biasGradient1, const Matrix& weightGradient2, const Matrix& biasGradient2);
};

#endif // SEQUENTIAL_HPP
