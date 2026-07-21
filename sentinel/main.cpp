#include "NeuralNet/Tokenizer/BPETokenizer.hpp"
#include "NeuralNet/Layers/Embedding.hpp"
#include "NeuralNet/Layers/Dense.hpp"
#include "NeuralNet/Layers/MeanPool.hpp"
#include "NeuralNet/Network/Sequential.hpp"
#include "NeuralNet/Optimizers/SGD.hpp"
#include "NeuralNet/Math/Matrix.hpp"
#include "NeuralNet/Data/ClassificationDataset.hpp"
#include "NeuralNet/Data/DatasetSplit.hpp"
#include "NeuralNet/Data/JsonlLoader.hpp"
#include "NeuralNet/Initializers/UniformInit.hpp"
#include "NeuralNet/Utils/TextUtil.hpp"

#include <iostream>
#include <string>
#include <vector>

/// <summary>
/// demo: classify SERA rows (T1 vs T2) with a real train/test split
/// tokenizer is fit on train texts only (no test leakage)
/// </summary>
int main() {
    const std::string samplePath = "../SERA-Data/sera_sample.jsonl";
    const int classCount = 2;
    const size_t maximumTextCharacters = 400;
    const float trainRatio = 0.8f;

    std::vector<JsonlRow> rows = JsonlLoader::load(samplePath, 500);

    std::vector<std::string> texts;
    std::vector<int> labels;
    int countClassT1 = 0;
    int countClassT2 = 0;

    for (const JsonlRow& row : rows) {
        const int label = JsonlLoader::sourceToLabel(row.source);
        if (label < 0) continue;

        texts.push_back(TextUtil::truncate(row.text, maximumTextCharacters));
        labels.push_back(label);
        if (label == 0) {
            ++countClassT1;
        }
        else {
            ++countClassT2;
        }
    }

    if (texts.empty()) {
        std::cout << "no usable rows from " << samplePath << '\n';
        return 1;
    }

    DatasetSplit split = DatasetSplit::partition(texts, labels, trainRatio, 42u);

    // fit BPE only on train test stays unseen for vocab merges too
    BPETokenizer tokenizer;
    tokenizer.train(split.trainTexts, 300);

    ClassificationDataset trainDataset = ClassificationDataset::buildLabeled(
        split.trainTexts,
        split.trainLabels,
        tokenizer,
        classCount
    );
    ClassificationDataset testDataset = ClassificationDataset::buildLabeled(
        split.testTexts,
        split.testLabels,
        tokenizer,
        classCount
    );

    const int majorityCount = (countClassT1 > countClassT2) ? countClassT1 : countClassT2;

    std::cout << "rows: " << texts.size()
              << " (T1=" << countClassT1 << ", T2=" << countClassT2 << ")\n";
    std::cout << "train: " << split.trainSize() << " | test: " << split.testSize() << '\n';
    std::cout << "vocab size: " << tokenizer.vocabSize() << '\n';
    std::cout << "majority baseline: "
              << (static_cast<float>(majorityCount) / static_cast<float>(texts.size()))
              << '\n';

    const int embeddingDim = 32;
    const int hidden = 32;

    Matrix weight1 = UniformInit::matrix(hidden, embeddingDim, 0.1f, 1u);
    Matrix bias1 = UniformInit::matrix(hidden, 1, 0.01f, 2u);
    Matrix weight2 = UniformInit::matrix(classCount, hidden, 0.1f, 3u);
    Matrix bias2 = UniformInit::matrix(classCount, 1, 0.01f, 4u);

    Embedding embedding(tokenizer.vocabSize(), embeddingDim);
    MeanPool meanPool;

    Sequential model(
        Dense(std::move(weight1), std::move(bias1)),
        Dense(std::move(weight2), std::move(bias2)),
        SGD(0.05f)
    );

    model.train(embedding, meanPool, trainDataset, testDataset, 2000, 500, 3);

    const float trainAccuracy = model.accuracy(embedding, meanPool, trainDataset);
    const float testAccuracy = model.accuracy(embedding, meanPool, testDataset);

    std::cout << "train accuracy: " << trainAccuracy << '\n';
    std::cout << "test accuracy:  " << testAccuracy << '\n';

    const char* classNames[] = { "T1", "T2" };
    std::cout << "--- test predictions ---\n";
    for (size_t index = 0; index < testDataset.examples.size(); ++index) {
        const ClassificationExample& example = testDataset.examples[index];
        const int predicted = model.predictClass(embedding, meanPool, example.tokenIds);

        std::cout << "test " << index
                  << " | true: " << classNames[example.label]
                  << " | pred: " << classNames[predicted]
                  << '\n';
    }

    return 0;
}
