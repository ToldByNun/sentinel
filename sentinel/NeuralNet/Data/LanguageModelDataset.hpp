#ifndef LANGUAGEMODELDATASET_HPP
#define LANGUAGEMODELDATASET_HPP

#include "../Math/Matrix.hpp"
#include "../Tokenizer/BPETokenizer.hpp"

#include <string>
#include <vector>

/// <summary>
/// one next-token example
/// inputTokenIds predicts targetTokenIds (same length shifted by one)
/// </summary>
class LanguageModelExample {
public:
    std::vector<int> inputTokenIds;
    std::vector<int> targetTokenIds;
    /// <summary>vocabSize x sequenceLength one-hot columns for CrossEntropy</summary>
    Matrix targetOneHot;
};

/// <summary>
/// builds shifted token pairs for language modeling
/// tokens [t0 t1 t2 t3] -> input [t0 t1 t2] target [t1 t2 t3]
/// </summary>
class LanguageModelDataset {
public:
    std::vector<LanguageModelExample> examples;
    int vocabularySize = 0;

    /// <summary>one-hot columns for a target token id sequence</summary>
    static Matrix makeOneHotSequence(const std::vector<int>& targetTokenIds, int vocabularySize);

    /// <summary>shift one token sequence into input/target (+ optional one-hot)</summary>
    static LanguageModelExample fromTokenIds(const std::vector<int>& tokenIds, int vocabularySize, bool buildOneHot = true);

    /// <summary>
    /// encode texts then shift
    /// skips sequences shorter than 2 tokens
    /// if maximumTokenCount > 0 truncates before the shift
    /// </summary>
    static LanguageModelDataset build(const std::vector<std::string>& texts, const BPETokenizer& tokenizer, size_t maximumTokenCount = 0, bool buildOneHot = true);

    int size() const;

    /// <summary>total next-token positions across all examples</summary>
    int totalPredictionCount() const;
};

#endif // LANGUAGEMODELDATASET_HPP
