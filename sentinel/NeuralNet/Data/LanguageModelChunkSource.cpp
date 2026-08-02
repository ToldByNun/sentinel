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

bool LanguageModelChunkSource::tryAppendExample(LanguageModelDataset& dataset, const JsonlRow& row) const {
    if (this->tokenizer == nullptr) throw std::logic_error("LanguageModelChunkSource tokenizer not set");

    std::vector<int> tokenIds = this->tokenizer->encode(this->truncateText(row.text));
    if (this->maximumTokenCount > 0 && tokenIds.size() > this->maximumTokenCount)
        tokenIds.resize(this->maximumTokenCount);
    if (tokenIds.size() < 2) return false;

    dataset.examples.push_back(
        LanguageModelDataset::fromTokenIds(tokenIds, this->tokenizer->vocabSize(), false)
    );
    return true;
}

bool LanguageModelChunkSource::refillPendingRows() {
    if (this->streamExhausted) return false;
    if (this->pendingCursor < this->pendingRows.size()) return true;

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

    this->rewindTrain();

    std::vector<std::string> texts;
    texts.reserve(static_cast<size_t>(maxRows));

    while (static_cast<int>(texts.size()) < maxRows && this->refillPendingRows()) {
        const size_t rowIndex = this->pendingBaseIndex + this->pendingCursor;
        const JsonlRow& row = this->pendingRows[this->pendingCursor];
        ++this->pendingCursor;

        if (!this->isTrainRow(rowIndex)) continue;
        texts.push_back(this->truncateText(row.text));
    }

    this->rewindTrain();
    return texts;
}

void LanguageModelChunkSource::prepareTestReservoir() {
    if (this->tokenizer == nullptr) throw std::logic_error("LanguageModelChunkSource::prepareTestReservoir tokenizer not set");

    this->testReservoir.examples.clear();
    this->testReservoir.vocabularySize = this->tokenizer->vocabSize();
    if (this->testCap == 0) {
        this->rewindTrain();
        return;
    }

    this->rewindTrain();

    while (static_cast<int>(this->testReservoir.examples.size()) < this->testCap && this->refillPendingRows()) {
        const size_t rowIndex = this->pendingBaseIndex + this->pendingCursor;
        const JsonlRow& row = this->pendingRows[this->pendingCursor];
        ++this->pendingCursor;

        if (this->isTrainRow(rowIndex)) continue;
        this->tryAppendExample(this->testReservoir, row);
    }

    this->rewindTrain();
}

void LanguageModelChunkSource::rewindTrain() {
    this->reader.rewind();
    this->pendingRows.clear();
    this->pendingBaseIndex = 0;
    this->pendingCursor = 0;
    this->streamExhausted = false;
}

bool LanguageModelChunkSource::nextTrainChunk(LanguageModelDataset& out) {
    if (this->tokenizer == nullptr) throw std::logic_error("LanguageModelChunkSource::nextTrainChunk tokenizer not set");

    out.examples.clear();
    out.vocabularySize = this->tokenizer->vocabSize();
    out.examples.reserve(static_cast<size_t>(this->chunkExamples));

    while (static_cast<int>(out.examples.size()) < this->chunkExamples && this->refillPendingRows()) {
        const size_t rowIndex = this->pendingBaseIndex + this->pendingCursor;
        const JsonlRow& row = this->pendingRows[this->pendingCursor];
        ++this->pendingCursor;

        if (!this->isTrainRow(rowIndex)) continue;
        this->tryAppendExample(out, row);
    }

    const bool moreTrainData =
        !this->streamExhausted
        || this->pendingCursor < this->pendingRows.size();

    return moreTrainData;
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
