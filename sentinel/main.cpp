#include "NeuralNet/Tokenizer/BPETokenizer.hpp"
#include "NeuralNet/Data/DatasetSplit.hpp"
#include "NeuralNet/Data/JsonlLoader.hpp"
#include "NeuralNet/Data/LanguageModelDataset.hpp"
#include "NeuralNet/Losses/CrossEntropy.hpp"
#include "NeuralNet/Utils/TextUtil.hpp"

#include <iostream>
#include <string>
#include <vector>

/// <summary>
/// demo: build next-token (shifted) language-model examples from SERA text
/// no sequence model yet — only dataset + CrossEntropy shape check
/// </summary>
int main() {
    const std::string samplePath = "../SERA-Data/sera_sample.jsonl";
    const size_t maximumTextCharacters = 400;
    const size_t maximumTokenCount = 64;
    const float trainRatio = 0.8f;

    std::vector<JsonlRow> rows = JsonlLoader::load(samplePath, 500);

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
    tokenizer.train(split.trainTexts, 300);

    LanguageModelDataset trainDataset = LanguageModelDataset::build(
        split.trainTexts,
        tokenizer,
        maximumTokenCount,
        true
    );
    LanguageModelDataset testDataset = LanguageModelDataset::build(
        split.testTexts,
        tokenizer,
        maximumTokenCount,
        true
    );

    std::cout << "texts: " << texts.size() << '\n';
    std::cout << "train texts: " << split.trainSize() << " | test texts: " << split.testSize() << '\n';
    std::cout << "vocab size: " << tokenizer.vocabSize() << '\n';
    std::cout << "train examples: " << trainDataset.size()
              << " | next-token positions: " << trainDataset.totalPredictionCount() << '\n';
    std::cout << "test examples: " << testDataset.size()
              << " | next-token positions: " << testDataset.totalPredictionCount() << '\n';

    if (trainDataset.examples.empty()) {
        std::cout << "no language-model examples (need sequences with >= 2 tokens)\n";
        return 1;
    }

    const LanguageModelExample& sample = trainDataset.examples[0];
    const size_t previewCount = (sample.inputTokenIds.size() < 8) ? sample.inputTokenIds.size() : 8;

    std::cout << "--- shift preview (first train example, first " << previewCount << " positions) ---\n";
    for (size_t position = 0; position < previewCount; ++position) {
        const int inputId = sample.inputTokenIds[position];
        const int targetId = sample.targetTokenIds[position];
        std::cout << "pos " << position
                  << " | in:  [" << inputId << "] \"" << tokenizer.idToToken(inputId) << "\""
                  << " | out: [" << targetId << "] \"" << tokenizer.idToToken(targetId) << "\""
                  << '\n';
    }

    // smoke test: identical probability/target matrices -> loss near 0
    const float perfectLoss = CrossEntropy::loss(sample.targetOneHot, sample.targetOneHot);
    std::cout << "perfect-prediction CE (should be ~0): " << perfectLoss << '\n';

    return 0;
}
