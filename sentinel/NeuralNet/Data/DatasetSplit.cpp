#include "DatasetSplit.hpp"

#include <stdexcept>
#include <utility>

unsigned DatasetSplit::advanceSeed(unsigned seed) {
    return seed * 1664525u + 1013904223u;
}

DatasetSplit DatasetSplit::partition(const std::vector<std::string>& texts, const std::vector<int>& labels, float trainRatio, unsigned seed) {
    if (texts.size() != labels.size()) throw std::invalid_argument("DatasetSplit::partition size mismatch");
    if (texts.empty()) throw std::invalid_argument("DatasetSplit::partition empty input");
    if (trainRatio <= 0.0f || trainRatio > 1.0f) throw std::invalid_argument("DatasetSplit::partition trainRatio must be in (0, 1]");

    std::vector<size_t> order(texts.size());
    for (size_t index = 0; index < order.size(); ++index) order[index] = index;

    // Fisher-Yates shuffle
    for (size_t index = order.size(); index > 1; --index) {
        seed = DatasetSplit::advanceSeed(seed);
        const size_t swapIndex = static_cast<size_t>(seed % index);
        const size_t lastIndex = index - 1;
        const size_t temporary = order[lastIndex];
        order[lastIndex] = order[swapIndex];
        order[swapIndex] = temporary;
    }

    const size_t trainCount = static_cast<size_t>(static_cast<float>(texts.size()) * trainRatio);
    const size_t clampedTrainCount = (trainCount == 0) ? 1 : trainCount;

    DatasetSplit split;
    for (size_t index = 0; index < order.size(); ++index) {
        const size_t sourceIndex = order[index];
        if (index < clampedTrainCount) {
            split.trainTexts.push_back(texts[sourceIndex]);
            split.trainLabels.push_back(labels[sourceIndex]);
            continue;
        }

        split.testTexts.push_back(texts[sourceIndex]);
        split.testLabels.push_back(labels[sourceIndex]);
    }

    if (split.testTexts.empty() && texts.size() > 1) {
        split.testTexts.push_back(split.trainTexts.back());
        split.testLabels.push_back(split.trainLabels.back());
        split.trainTexts.pop_back();
        split.trainLabels.pop_back();
    }

    return split;
}

DatasetSplit DatasetSplit::partitionTexts(const std::vector<std::string>& texts, float trainRatio, unsigned seed) {
    std::vector<int> dummyLabels(texts.size(), 0);
    DatasetSplit split = DatasetSplit::partition(texts, dummyLabels, trainRatio, seed);
    split.trainLabels.clear();
    split.testLabels.clear();
    return split;
}

int DatasetSplit::trainSize() const {
    return static_cast<int>(this->trainTexts.size());
}

int DatasetSplit::testSize() const {
    return static_cast<int>(this->testTexts.size());
}
