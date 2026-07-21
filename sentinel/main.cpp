#include "NeuralNet/Tokenizer/BPETokenizer.hpp"
#include "NeuralNet/Layers/Embedding.hpp"
#include "NeuralNet/Layers/Dense.hpp"
#include "NeuralNet/Layers/MeanPool.hpp"
#include "NeuralNet/Network/Sequential.hpp"
#include "NeuralNet/Optimizers/SGD.hpp"
#include "NeuralNet/Math/Matrix.hpp"
#include "NeuralNet/Data/ClassificationDataset.hpp"
#include "NeuralNet/Data/JsonlLoader.hpp"
#include "NeuralNet/Initializers/UniformInit.hpp"
#include "NeuralNet/Utils/TextUtil.hpp"

#include <iostream>
#include <string>
#include <vector>

/// <summary>
/// Demo: classify SERA sample rows by source (T1 vs T2).
/// Flow: jsonl -> truncate -> BPE -> train Embedding+MLP -> print predictions.
/// </summary>
int main() {
    const std::string samplePath = "../SERA-Data/sera_sample.jsonl";
    const int classCount = 2;
    const size_t maximumTextCharacters = 400;

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
        if (label == 0) ++countClassT1;
        else ++countClassT2;
    }

    if (texts.empty()) {
        std::cout << "no usable rows from " << samplePath << '\n';
        return 1;
    }

    const int embeddingDim = 32;
    const int hidden = 32;

    Matrix weight1 = UniformInit::matrix(hidden, embeddingDim, 0.1f, 1u);
    Matrix bias1 = UniformInit::matrix(hidden, 1, 0.01f, 2u);
    Matrix weight2 = UniformInit::matrix(classCount, hidden, 0.1f, 3u);
    Matrix bias2 = UniformInit::matrix(classCount, 1, 0.01f, 4u);

    BPETokenizer tokenizer;
    tokenizer.train(texts, 300);

    ClassificationDataset dataset = ClassificationDataset::buildLabeled(
        texts,
        labels,
        tokenizer,
        classCount
    );

    const int majorityCount = (countClassT1 > countClassT2) ? countClassT1 : countClassT2;

    std::cout << "rows: " << texts.size()
              << " (T1=" << countClassT1 << ", T2=" << countClassT2 << ")\n";
    std::cout << "vocab size: " << tokenizer.vocabSize() << '\n';
    std::cout << "majority baseline: "
              << (static_cast<float>(majorityCount) / static_cast<float>(texts.size()))
              << '\n';

    Embedding embedding(tokenizer.vocabSize(), embeddingDim);
    MeanPool meanPool;

    Sequential model(
        Dense(std::move(weight1), std::move(bias1)),
        Dense(std::move(weight2), std::move(bias2)),
        SGD(0.05f)
    );

    model.train(embedding, meanPool, dataset, 2000);

    const char* classNames[] = { "T1", "T2" };
    int correct = 0;
    for (size_t index = 0; index < dataset.examples.size(); ++index) {
        const ClassificationExample& example = dataset.examples[index];
        const int predicted = model.predictClass(embedding, meanPool, example.tokenIds);
        if (predicted == example.label) ++correct;

        std::cout << "example " << index
                  << " | true: " << classNames[example.label]
                  << " | pred: " << classNames[predicted]
                  << '\n';
    }

    std::cout << "final accuracy: "
              << (static_cast<float>(correct) / static_cast<float>(dataset.size()))
              << '\n';

    return 0;
}
