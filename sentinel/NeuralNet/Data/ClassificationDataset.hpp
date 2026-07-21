#ifndef CLASSIFICATIONDATASET_HPP
#define CLASSIFICATIONDATASET_HPP

#include "../Math/Matrix.hpp"
#include "../Tokenizer/BPETokenizer.hpp"

#include <string>
#include <vector>

class ClassificationExample {
public:
    std::vector<int> tokenIds;
    Matrix target;
    int label = 0;
};

/// <summary>labeled text samples for Cpp/Json/Python classification</summary>
class ClassificationDataset {
public:
    static constexpr int ClassCpp = 0;
    static constexpr int ClassJson = 1;
    static constexpr int ClassPython = 2;
    static constexpr int ClassCount = 3;

    std::vector<ClassificationExample> examples;

    /// <summary>onehot column vector for a class id</summary>
    static Matrix makeOneHot(int label, int classCount = ClassCount);

    /// <summary>heuristic label from text content</summary>
    static int inferLabel(const std::string& text);

    /// <summary>encode corpus + attach labels. tokenizer must already be trained</summary>
    static ClassificationDataset build(
        const std::vector<std::string>& corpus,
        const BPETokenizer& tokenizer
    );

    /// <summary>encode texts with explicit labels (for SERA etc.).</summary>
    static ClassificationDataset buildLabeled(
        const std::vector<std::string>& texts,
        const std::vector<int>& labels,
        const BPETokenizer& tokenizer,
        int classCount
    );

    int size() const;
};

#endif // CLASSIFICATIONDATASET_HPP
