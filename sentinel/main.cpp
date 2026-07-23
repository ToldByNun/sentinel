#include "NeuralNet/Tokenizer/BPETokenizer.hpp"
#include "NeuralNet/Data/DatasetSplit.hpp"
#include "NeuralNet/Data/JsonlLoader.hpp"
#include "NeuralNet/Data/LanguageModelDataset.hpp"
#include "NeuralNet/Network/LanguageModel.hpp"
#include "NeuralNet/Optimizers/Adam.hpp"
#include "NeuralNet/Utils/TextUtil.hpp"

#include <iostream>
#include <string>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

/// <summary>
/// demo: causal LM on SERA text
/// token embed + RoPE multi head attention + SwiGLU FFN + vocab projection
/// </summary>
int main() {
    const std::string samplePath = "../SERA-Data/sera_sample.jsonl";
    const size_t maximumTextCharacters = 400;
    const size_t maximumTokenCount = 128;
    const float trainRatio = 0.8f;
    const int embeddingDim = 128;
    const int maximumPositionCount = static_cast<int>(maximumTokenCount);

#if defined(_OPENMP)
    std::cout << "OpenMP threads: " << omp_get_max_threads() << '\n';
#else
    std::cout << "OpenMP: disabled\n";
#endif

    // 0 = load entire jsonl (sera_sample has 10000 rows)
    std::vector<JsonlRow> rows = JsonlLoader::load(samplePath, 0);

    std::vector<std::string> texts;
    for (const JsonlRow& row : rows) {
        if (row.text.empty()) continue;
        texts.push_back(TextUtil::truncate(row.text, maximumTextCharacters));
    }

    if (texts.empty()) {
        std::cout << "no usable rows from " << samplePath << '\n';
        return 1;
    }

    DatasetSplit split = DatasetSplit::partitionTexts(texts, trainRatio, 42u);

    BPETokenizer tokenizer;
    tokenizer.train(split.trainTexts, 2500);

    LanguageModelDataset trainDataset = LanguageModelDataset::build(split.trainTexts, tokenizer, maximumTokenCount, false);
    LanguageModelDataset testDataset = LanguageModelDataset::build(split.testTexts, tokenizer, maximumTokenCount, false);

    std::cout << "texts: " << texts.size() << '\n';
    std::cout << "train examples: " << trainDataset.size()
              << " | test examples: " << testDataset.size() << '\n';
    std::cout << "vocab size: " << tokenizer.vocabSize() << '\n';
    std::cout << "next-token positions train: " << trainDataset.totalPredictionCount() << '\n';

    if (trainDataset.examples.empty()) {
        std::cout << "no language-model examples (need sequences with >= 2 tokens)\n";
        return 1;
    }

    LanguageModel model(tokenizer.vocabSize(), embeddingDim, maximumPositionCount, Adam(0.001f), 2, 4);

    std::cout << "blocks: " << model.blocks.size() << " | heads: " << model.blocks[0].attention.headCount << '\n';

    model.train(trainDataset, testDataset, 15, 1, 64);

    std::cout << "final trainLoss: " << model.averageLoss(trainDataset) << '\n';
    if (!testDataset.examples.empty())
        std::cout << "final testLoss:  " << model.averageLoss(testDataset) << '\n';

    const LanguageModelExample& sample = trainDataset.examples[0];
    const size_t promptLength = (sample.inputTokenIds.size() < 12) ? sample.inputTokenIds.size() : 12;
    std::vector<int> prompt;
    for (size_t index = 0; index < promptLength; ++index)
        prompt.push_back(sample.inputTokenIds[index]);

    std::vector<int> greedy = model.generate(prompt, 20, 0.0f);
    std::vector<int> sampled = model.generate(prompt, 20, 0.9f, 40, 42u);

    std::cout << "--- prompt ---\n" << tokenizer.decode(prompt) << '\n';
    std::cout << "--- greedy ---\n" << tokenizer.decode(greedy) << '\n';
    std::cout << "--- sample T=0.9 topK=40 ---\n" << tokenizer.decode(sampled) << '\n';

    return 0;
}
