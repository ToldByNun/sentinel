#ifndef LANGUAGEMODELCHUNKSOURCE_HPP
#define LANGUAGEMODELCHUNKSOURCE_HPP

#include "JsonlLoader.hpp"
#include "LanguageModelDataset.hpp"
#include "../Tokenizer/BPETokenizer.hpp"

#include <string>
#include <vector>

/// <summary>
/// streams JSONL once into compact token-id examples, then serves train chunks without re-encode
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

    /// <summary>
    /// one JSONL pass: encode all train rows into memory, fill test reservoir up to cap
    /// must be called before nextTrainChunk (avoids re-encoding every epoch)
    /// </summary>
    void materialize();

    /// <summary>alias for materialize (fills testDataset)</summary>
    void prepareTestReservoir();

    /// <summary>seek to start for a new train epoch; length-sorts materialized examples</summary>
    void rewindTrain();

    /// <summary>stable-sort materialized train examples by sequence length (short to long)</summary>
    void sortTrainByLength();

    /// <summary>copy all materialized train examples into out (call after rewindTrain for sorted order)</summary>
    void fillTrainDataset(LanguageModelDataset& out) const;

    /// <summary>
    /// fill out with up to chunkExampleCount train examples from the materialized cache
    /// returns false when the train stream is exhausted (out may still hold a final partial chunk)
    /// </summary>
    bool nextTrainChunk(LanguageModelDataset& out);

    const LanguageModelDataset& testDataset() const;
    const std::string& filePath() const;
    int chunkExampleCount() const;
    float trainRatio() const;
    int trainExampleCount() const;
    int trainPredictionCount() const;
    bool isMaterialized() const;

    /// <summary>true when rowIndex belongs to the train partition</summary>
    bool isTrainRow(size_t rowIndex) const;

private:
    bool tryMakeExample(const JsonlRow& row, LanguageModelExample& out) const;
    bool tryMakeExampleFromText(const std::string& text, LanguageModelExample& out) const;
    std::string truncateText(const std::string& text) const;
    void resetJsonlCursor();

    bool refillPendingRows();
    /// <summary>encode a batch of truncated texts in parallel, then append train/test examples in order</summary>
    void encodeBatchIntoDatasets(const std::vector<size_t>& rowIndices, const std::vector<std::string>& texts);

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
    std::vector<LanguageModelExample> trainExamples;
    int trainPredictions = 0;
    size_t trainCursor = 0;
    bool materialized = false;

    std::vector<JsonlRow> pendingRows;
    size_t pendingBaseIndex = 0;
    size_t pendingCursor = 0;
    bool streamExhausted = false;

    static constexpr int rowReadBatch = 1024;
    static constexpr int encodeParallelBatch = 2048;
};

#endif // LANGUAGEMODELCHUNKSOURCE_HPP
