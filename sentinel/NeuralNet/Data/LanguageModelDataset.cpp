#include "LanguageModelDataset.hpp"

#include <stdexcept>
#include <vector>

Matrix LanguageModelDataset::makeOneHotSequence(const std::vector<int>& targetTokenIds, int vocabularySize) {
    if (vocabularySize <= 0) throw std::invalid_argument("LanguageModelDataset::makeOneHotSequence vocabularySize must be > 0");
    if (targetTokenIds.empty()) throw std::invalid_argument("LanguageModelDataset::makeOneHotSequence empty targets");

    Matrix target;
    target.data = std::vector<std::vector<float>>(
        static_cast<size_t>(vocabularySize),
        std::vector<float>(targetTokenIds.size(), 0.0f)
    );

    for (size_t position = 0; position < targetTokenIds.size(); ++position) {
        const int tokenId = targetTokenIds[position];
        if (tokenId < 0 || tokenId >= vocabularySize)
            throw std::out_of_range("LanguageModelDataset::makeOneHotSequence token id out of range");
        target.data[static_cast<size_t>(tokenId)][position] = 1.0f;
    }

    return target;
}

LanguageModelExample LanguageModelDataset::fromTokenIds(const std::vector<int>& tokenIds, int vocabularySize, bool buildOneHot) {
    if (tokenIds.size() < 2) throw std::invalid_argument("LanguageModelDataset::fromTokenIds needs at least 2 tokens");
    if (vocabularySize <= 0) throw std::invalid_argument("LanguageModelDataset::fromTokenIds vocabularySize must be > 0");

    LanguageModelExample example;
    example.inputTokenIds.assign(tokenIds.begin(), tokenIds.end() - 1);
    example.targetTokenIds.assign(tokenIds.begin() + 1, tokenIds.end());

    if (buildOneHot)
        example.targetOneHot = LanguageModelDataset::makeOneHotSequence(example.targetTokenIds, vocabularySize);

    return example;
}

LanguageModelDataset LanguageModelDataset::build(const std::vector<std::string>& texts, const BPETokenizer& tokenizer, size_t maximumTokenCount, bool buildOneHot) {
    LanguageModelDataset dataset;
    dataset.vocabularySize = tokenizer.vocabSize();

    for (const std::string& text : texts) {
        std::vector<int> tokenIds = tokenizer.encode(text);
        if (maximumTokenCount > 0 && tokenIds.size() > maximumTokenCount)
            tokenIds.resize(maximumTokenCount);
        if (tokenIds.size() < 2) continue;

        dataset.examples.push_back(
            LanguageModelDataset::fromTokenIds(tokenIds, dataset.vocabularySize, buildOneHot)
        );
    }

    return dataset;
}

int LanguageModelDataset::size() const {
    return static_cast<int>(this->examples.size());
}

int LanguageModelDataset::totalPredictionCount() const {
    int total = 0;
    for (const LanguageModelExample& example : this->examples)
        total += static_cast<int>(example.targetTokenIds.size());
    return total;
}
