#ifndef DATASETSPLIT_HPP
#define DATASETSPLIT_HPP

#include <string>
#include <vector>

/// <summary>holds raw text/label partitions before tokenization</summary>
class DatasetSplit {
public:
    std::vector<std::string> trainTexts;
    std::vector<int> trainLabels;
    std::vector<std::string> testTexts;
    std::vector<int> testLabels;

    /// <summary>
    /// shuffle then split. trainRatio in (0, 1], e.g. 0.8 -> 80% train
    /// </summary>
    static DatasetSplit partition(
        const std::vector<std::string>& texts,
        const std::vector<int>& labels,
        float trainRatio,
        unsigned seed
    );

    /// <summary>textonly shuffle split for language modeling (labels stay empty)</summary>
    static DatasetSplit partitionTexts(
        const std::vector<std::string>& texts,
        float trainRatio,
        unsigned seed
    );

    int trainSize() const;
    int testSize() const;

private:
    static unsigned advanceSeed(unsigned seed);
};

#endif // DATASETSPLIT_HPP
