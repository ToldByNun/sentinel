#ifndef LANGUAGEMODELCHUNKSOURCE_HPP
#define LANGUAGEMODELCHUNKSOURCE_HPP

#include "JsonlLoader.hpp"
#include "LanguageModelDataset.hpp"
#include "../Tokenizer/BPETokenizer.hpp"

#include <string>
#include <vector>

/// <summary>
/// streams JSONL into LanguageModelDataset chunks
/// train/test via deterministic row-index hash; test examples capped in a reservoir
/// </summary>
class LanguageModelChunkSource {
public:
    LanguageModelChunkSource(
        std::string path,
        size_t maximumTextCharacters,
        size_t maximumTokenCount,
        int chunkExampleCount,
        float trainRatio,
        unsigned seed,
        int testReservoirCap
    );

    /// <summary>set tokenizer used for encode / reservoir / train chunks</summary>
    void setTokenizer(const BPETokenizer* tokenizer);

    /// <summary>first maxRows train-hash texts (truncated); rewinds the file</summary>
    std::vector<std::string> prepareTokenizerSample(int maxRows);

    /// <summary>one full pass; fills testDataset up to testReservoirCap (needs tokenizer)</summary>
    void prepareTestReservoir();

    /// <summary>seek to start for a new train epoch</summary>
    void rewindTrain();

    /// <summary>
    /// fill out with up to chunkExampleCount train examples
    /// returns false when the train stream is exhausted (out may still hold a final partial chunk)
    /// </summary>
    bool nextTrainChunk(LanguageModelDataset& out);

    const LanguageModelDataset& testDataset() const;
    const std::string& filePath() const;
    int chunkExampleCount() const;
    float trainRatio() const;

    /// <summary>true when rowIndex belongs to the train partition</summary>
    bool isTrainRow(size_t rowIndex) const;

private:
    bool refillPendingRows();
    bool tryAppendExample(LanguageModelDataset& dataset, const JsonlRow& row) const;
    std::string truncateText(const std::string& text) const;

    JsonlChunkReader reader;
    std::string path;
    size_t maximumTextCharacters = 0;
    size_t maximumTokenCount = 0;
    int chunkExamples = 0;
    float trainSplitRatio = 0.8f;
    unsigned splitSeed = 0;
    int testCap = 0;

    const BPETokenizer* tokenizer = nullptr;
    LanguageModelDataset testReservoir;

    std::vector<JsonlRow> pendingRows;
    size_t pendingBaseIndex = 0;
    size_t pendingCursor = 0;
    bool streamExhausted = false;

    static constexpr int rowReadBatch = 256;
};

#endif // LANGUAGEMODELCHUNKSOURCE_HPP
