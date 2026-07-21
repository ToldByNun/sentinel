#ifndef SEQUENTIAL_HPP
#define SEQUENTIAL_HPP

#include "../Data/ClassificationDataset.hpp"
#include "../Layers/Dense.hpp"
#include "../Layers/Embedding.hpp"
#include "../Layers/MeanPool.hpp"
#include "../Math/Matrix.hpp"
#include "../Optimizers/SGD.hpp"

#include <vector>

/// <summary>
/// text path: Embedding -> MeanPool -> Dense -> Softmax
/// </summary>
class Sequential {
public:
    Dense layer1;
    Dense layer2;
    SGD optimizer;

    Sequential(Dense layer1, Dense layer2, SGD optimizer);

    /// <summary>forward from a feature vector returns class probs</summary>
    Matrix forward(const Matrix& input);

    /// <summary>forward from token ids (embed + pool + mlp)</summary>
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
    /// train on trainDataset every logEveryEpochs reports testAccuracy
    /// stops early if testAccuracy does not improve for earlyStoppingPatience checks
    /// restores best weights after stop
    /// </summary>
    void train(
        Embedding& embedding,
        MeanPool& meanPool,
        const ClassificationDataset& trainDataset,
        const ClassificationDataset& testDataset,
        int epochs,
        int logEveryEpochs = 500,
        int earlyStoppingPatience = 3
    );

    /// <summary>argmax class for token ids</summary>
    int predictClass(Embedding& embedding, MeanPool& meanPool, const std::vector<int>& tokenIds);

    /// <summary>fraction of correctly classified examples (no training)</summary>
    float accuracy(Embedding& embedding, MeanPool& meanPool, const ClassificationDataset& dataset);
};

#endif // SEQUENTIAL_HPP
