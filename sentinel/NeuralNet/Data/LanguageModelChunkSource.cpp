#include "LanguageModelChunkSource.hpp"

#include "../Utils/TextUtil.hpp"

#include <stdexcept>
#include <utility>

LanguageModelChunkSource::LanguageModelChunkSource(
    std::string path,
    size_t maximumTextCharacters,
    size_t maximumTokenCount,
    int chunkExampleCount,
    float trainRatio,
    unsigned seed,
    int testReservoirCap
) {
    if (path.empty()) throw std::invalid_argument("LanguageModelChunkSource empty path");
    if (chunkExampleCount <= 0) throw std::invalid_argument("LanguageModelChunkSource chunkExampleCount must be > 0");
    if (trainRatio <= 0.0f || trainRatio > 1.0f) throw std::invalid_argument("LanguageModelChunkSource trainRatio must be in (0, 1]");
    if (testReservoirCap < 0) throw std::invalid_argument("LanguageModelChunkSource testReservoirCap must be >= 0");

    this->path = std::move(path);
    this->maximumTextCharacters = maximumTextCharacters;
    this->maximumTokenCount = maximumTokenCount;
    this->chunkExamples = chunkExampleCount;
    this->trainSplitRatio = trainRatio;
    this->splitSeed = seed;
    this->testCap = testReservoirCap;
    this->reader.open(this->path);
}

void LanguageModelChunkSource::setTokenizer(const BPETokenizer* tokenizer) {
    if (tokenizer == nullptr) throw std::invalid_argument("LanguageModelChunkSource::setTokenizer null");
    this->tokenizer = tokenizer;
    this->materialized = false;
    this->trainExamples.clear();
    this->trainPredictions = 0;
    this->trainCursor = 0;
    this->testReservoir.examples.clear();
}

bool LanguageModelChunkSource::isTrainRow(size_t rowIndex) const {
    if (this->trainSplitRatio >= 1.0f) return true;

    unsigned mixed = this->splitSeed ^ (static_cast<unsigned>(rowIndex + 1u) * 2654435761u);
    mixed = mixed * 1664525u + 1013904223u;
    const float unit = static_cast<float>(mixed % 100000u) / 100000.0f;
    return unit < this->trainSplitRatio;
}

std::string LanguageModelChunkSource::truncateText(const std::string& text) const {
    if (this->maximumTextCharacters == 0) return text;
    return TextUtil::truncate(text, this->maximumTextCharacters);
}

bool LanguageModelChunkSource::tryMakeExample(const JsonlRow& row, LanguageModelExample& out) const {
    if (this->tokenizer == nullptr) throw std::logic_error("LanguageModelChunkSource tokenizer not set");

    std::vector<int> tokenIds = this->tokenizer->encode(this->truncateText(row.text));
    if (this->maximumTokenCount > 0 && tokenIds.size() > this->maximumTokenCount)
        tokenIds.resize(this->maximumTokenCount);
    if (tokenIds.size() < 2) return false;

    out = LanguageModelDataset::fromTokenIds(tokenIds, this->tokenizer->vocabSize(), false);
    return true;
}

void LanguageModelChunkSource::resetJsonlCursor() {
    this->reader.rewind();
    this->pendingRows.clear();
    this->pendingBaseIndex = 0;
    this->pendingCursor = 0;
    this->streamExhausted = false;
}

bool LanguageModelChunkSource::refillPendingRows() {
    // Drain pending first — streamExhausted may be set when the last file batch
    // was read while rows still remain in pendingRows.
    if (this->pendingCursor < this->pendingRows.size()) return true;
    if (this->streamExhausted) return false;

    const bool moreAvailable = this->reader.nextRows(this->pendingRows, LanguageModelChunkSource::rowReadBatch);
    this->pendingBaseIndex = this->reader.rowsRead() - this->pendingRows.size();
    this->pendingCursor = 0;

    if (this->pendingRows.empty()) {
        this->streamExhausted = true;
        return false;
    }

    if (!moreAvailable)
        this->streamExhausted = true;

    return true;
}

std::vector<std::string> LanguageModelChunkSource::prepareTokenizerSample(int maxRows) {
    if (maxRows <= 0) throw std::invalid_argument("LanguageModelChunkSource::prepareTokenizerSample maxRows must be > 0");

    this->resetJsonlCursor();

    std::vector<std::string> texts;
    texts.reserve(static_cast<size_t>(maxRows));

    while (static_cast<int>(texts.size()) < maxRows && this->refillPendingRows()) {
        const size_t rowIndex = this->pendingBaseIndex + this->pendingCursor;
        const JsonlRow& row = this->pendingRows[this->pendingCursor];
        ++this->pendingCursor;

        if (!this->isTrainRow(rowIndex)) continue;
        texts.push_back(this->truncateText(row.text));
    }

    this->resetJsonlCursor();
    return texts;
}

void LanguageModelChunkSource::materialize() {
    if (this->tokenizer == nullptr) throw std::logic_error("LanguageModelChunkSource::materialize tokenizer not set");

    this->trainExamples.clear();
    this->trainPredictions = 0;
    this->trainCursor = 0;
    this->testReservoir.examples.clear();
    this->testReservoir.vocabularySize = this->tokenizer->vocabSize();
    this->materialized = false;

    this->resetJsonlCursor();

    LanguageModelExample example;
    while (this->refillPendingRows()) {
        const size_t rowIndex = this->pendingBaseIndex + this->pendingCursor;
        const JsonlRow& row = this->pendingRows[this->pendingCursor];
        ++this->pendingCursor;

        if (!this->tryMakeExample(row, example)) continue;

        if (this->isTrainRow(rowIndex)) {
            this->trainPredictions += static_cast<int>(example.targetTokenIds.size());
            this->trainExamples.push_back(std::move(example));
            continue;
        }

        if (this->testCap > 0 && static_cast<int>(this->testReservoir.examples.size()) < this->testCap)
            this->testReservoir.examples.push_back(std::move(example));
    }

    this->resetJsonlCursor();
    this->materialized = true;
}

void LanguageModelChunkSource::prepareTestReservoir() {
    this->materialize();
}

void LanguageModelChunkSource::rewindTrain() {
    this->trainCursor = 0;
}

bool LanguageModelChunkSource::nextTrainChunk(LanguageModelDataset& out) {
    if (this->tokenizer == nullptr) throw std::logic_error("LanguageModelChunkSource::nextTrainChunk tokenizer not set");
    if (!this->materialized) throw std::logic_error("LanguageModelChunkSource::nextTrainChunk call materialize() first");

    out.examples.clear();
    out.vocabularySize = this->tokenizer->vocabSize();
    out.examples.reserve(static_cast<size_t>(this->chunkExamples));

    while (static_cast<int>(out.examples.size()) < this->chunkExamples && this->trainCursor < this->trainExamples.size()) {
        out.examples.push_back(this->trainExamples[this->trainCursor]);
        ++this->trainCursor;
    }

    return this->trainCursor < this->trainExamples.size();
}

const LanguageModelDataset& LanguageModelChunkSource::testDataset() const {
    return this->testReservoir;
}

const std::string& LanguageModelChunkSource::filePath() const {
    return this->path;
}

int LanguageModelChunkSource::chunkExampleCount() const {
    return this->chunkExamples;
}

float LanguageModelChunkSource::trainRatio() const {
    return this->trainSplitRatio;
}

int LanguageModelChunkSource::trainExampleCount() const {
    return static_cast<int>(this->trainExamples.size());
}

int LanguageModelChunkSource::trainPredictionCount() const {
    return this->trainPredictions;
}

bool LanguageModelChunkSource::isMaterialized() const {
    return this->materialized;
}
