#include "BPETokenizer.hpp"

#include "../Utils/SmokeLog.hpp"

#include <cctype>
#include <chrono>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <iterator>
#include <stdexcept>
#include <utility>

#if defined(_OPENMP)
#include <omp.h>
#endif

size_t BPETokenizer::IntPairHash::operator()(const IntPair& pair) const {
    const std::uint64_t packed = (static_cast<std::uint64_t>(static_cast<std::uint32_t>(pair.left)) << 32)
        | static_cast<std::uint32_t>(pair.right);
    return std::hash<std::uint64_t>{}(packed);
}

void BPETokenizer::train(const std::string& text, int vocabSize) {
    this->train(std::vector<std::string>{text}, vocabSize);
}

void BPETokenizer::train(const std::vector<std::string>& corpus, int vocabSize) {
    if (vocabSize <= 0) throw std::invalid_argument("vocabSize must be > 0");

    const auto start = std::chrono::steady_clock::now();

    this->tokenToId_.clear();
    this->idToToken_.clear();
    this->mergeRules_.clear();
    this->mergeRank_.clear();

    this->addUnknownToken();
    this->buildBaseVocabulary(corpus);
    this->rebuildByteToId();
    this->learnMerges(corpus, vocabSize);
    this->rebuildMergeRank();

    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
#if defined(_OPENMP)
    const int threads = omp_get_max_threads();
#else
    const int threads = 1;
#endif
    SmokeLog::result(
        "BPETokenizer::train",
        "corpus=%zu vocab=%d merges=%zu threads=%d sec=%.2f",
        corpus.size(),
        this->vocabSize(),
        this->mergeRules_.size(),
        threads,
        seconds);
}

void BPETokenizer::addUnknownToken() {
    this->unknownTokenId_ = 0;
    this->tokenToId_[BPETokenizer::UnknownToken] = this->unknownTokenId_;
    this->idToToken_.push_back(BPETokenizer::UnknownToken);
}

int BPETokenizer::lookupTokenId(const std::string& token) const {
    auto found = this->tokenToId_.find(token);
    if (found == this->tokenToId_.end()) return this->unknownTokenId_;
    return found->second;
}

std::vector<int> BPETokenizer::encode(const std::string& text) const {
    std::vector<int> tokenIds;
    const std::vector<std::string> words = this->preTokenize(text);
    tokenIds.reserve(text.size());

    for (const std::string& word : words) {
        std::vector<int> pieces;
        pieces.reserve(word.size());
        for (unsigned char character : word)
            pieces.push_back(this->byteToId_[character]);

        pieces = this->applyMergeRules(std::move(pieces));
        tokenIds.insert(tokenIds.end(), pieces.begin(), pieces.end());
    }

    return tokenIds;
}

std::string BPETokenizer::decode(const std::vector<int>& tokenIds) const {
    std::string result;
    result.reserve(tokenIds.size());

    for (int tokenId : tokenIds) {
        if (tokenId < 0 || tokenId >= static_cast<int>(this->idToToken_.size()))
            throw std::out_of_range("token id out of range");
        result += this->idToToken_[tokenId];
    }

    return result;
}

int BPETokenizer::vocabSize() const {
    return static_cast<int>(this->idToToken_.size());
}

int BPETokenizer::unknownTokenId() const {
    return this->unknownTokenId_;
}

int BPETokenizer::tokenToId(const std::string& token) const {
    return this->lookupTokenId(token);
}

const std::string& BPETokenizer::idToToken(int tokenId) const {
    if (tokenId < 0 || tokenId >= static_cast<int>(this->idToToken_.size()))
        throw std::out_of_range("token id out of range");
    return this->idToToken_[tokenId];
}

void BPETokenizer::buildBaseVocabulary(const std::vector<std::string>& corpus) {
    for (const std::string& text : corpus) {
        for (unsigned char character : text) {
            const char bytes[1] = { static_cast<char>(character) };
            const std::string token(bytes, 1);
            if (this->tokenToId_.find(token) != this->tokenToId_.end()) continue;
            this->tokenToId_[token] = static_cast<int>(this->idToToken_.size());
            this->idToToken_.push_back(token);
        }
    }
}

void BPETokenizer::rebuildByteToId() {
    for (int index = 0; index < 256; ++index)
        this->byteToId_[index] = this->unknownTokenId_;
    for (size_t tokenId = 0; tokenId < this->idToToken_.size(); ++tokenId) {
        const std::string& token = this->idToToken_[tokenId];
        if (token.size() == 1)
            this->byteToId_[static_cast<unsigned char>(token[0])] = static_cast<int>(tokenId);
    }
}

void BPETokenizer::rebuildMergeRank() {
    this->mergeRank_.clear();
    this->mergeRank_.reserve(this->mergeRules_.size() * 2);
    for (size_t index = 0; index < this->mergeRules_.size(); ++index) {
        const MergeRule& rule = this->mergeRules_[index];
        this->mergeRank_[{rule.leftId, rule.rightId}] = static_cast<int>(index);
    }
}

void BPETokenizer::learnMerges(const std::vector<std::string>& corpus, int vocabSize) {
    std::unordered_map<std::string, int> wordFrequency;
    for (const std::string& text : corpus) {
        for (const std::string& word : this->preTokenize(text))
            wordFrequency[word]++;
    }

    struct WordForm {
        std::vector<int> tokens;
        int frequency = 0;
    };

    std::vector<WordForm> words;
    words.reserve(wordFrequency.size());
    for (const auto& entry : wordFrequency) {
        WordForm form;
        form.frequency = entry.second;
        form.tokens.reserve(entry.first.size());
        for (unsigned char character : entry.first)
            form.tokens.push_back(this->byteToId_[character]);
        if (!form.tokens.empty())
            words.push_back(std::move(form));
    }

    const int wordCount = static_cast<int>(words.size());
    const int mergesTarget = vocabSize - this->vocabSize();
    int mergesDone = 0;

    while (this->vocabSize() < vocabSize) {
        std::unordered_map<IntPair, int, IntPairHash> pairCounts;
        pairCounts.reserve(static_cast<size_t>(wordCount) * 2u);

        for (const WordForm& form : words) {
            const std::vector<int>& tokens = form.tokens;
            for (size_t tokenIndex = 0; tokenIndex + 1 < tokens.size(); ++tokenIndex)
                pairCounts[{tokens[tokenIndex], tokens[tokenIndex + 1]}] += form.frequency;
        }

        if (pairCounts.empty()) break;

        auto best = pairCounts.begin();
        for (auto entry = pairCounts.begin(); entry != pairCounts.end(); ++entry) {
            if (entry->second > best->second) best = entry;
        }

        const IntPair bestPair = best->first;
        const std::string merged = this->idToToken_[bestPair.left] + this->idToToken_[bestPair.right];
        const int mergedId = static_cast<int>(this->idToToken_.size());

        this->tokenToId_[merged] = mergedId;
        this->idToToken_.push_back(merged);
        this->mergeRules_.push_back({bestPair.left, bestPair.right, mergedId});
        ++mergesDone;

#if defined(_OPENMP)
        #pragma omp parallel for schedule(dynamic, 64)
#endif
        for (int wordIndex = 0; wordIndex < wordCount; ++wordIndex) {
            std::vector<int>& tokens = words[static_cast<size_t>(wordIndex)].tokens;
            bool changed = false;
            for (size_t tokenIndex = 0; tokenIndex + 1 < tokens.size(); ++tokenIndex) {
                if (tokens[tokenIndex] == bestPair.left && tokens[tokenIndex + 1] == bestPair.right) {
                    changed = true;
                    break;
                }
            }
            if (!changed) continue;

            std::vector<int> updated;
            updated.reserve(tokens.size());
            for (size_t tokenIndex = 0; tokenIndex < tokens.size();) {
                if (tokenIndex + 1 < tokens.size()
                    && tokens[tokenIndex] == bestPair.left
                    && tokens[tokenIndex + 1] == bestPair.right) {
                    updated.push_back(mergedId);
                    tokenIndex += 2;
                    continue;
                }

                updated.push_back(tokens[tokenIndex]);
                ++tokenIndex;
            }
            tokens = std::move(updated);
        }

        if (mergesDone == 1 || mergesDone % 50 == 0 || this->vocabSize() >= vocabSize
            || (mergesTarget > 0 && mergesDone >= mergesTarget)) {
            SmokeLog::progress(
                "BPETokenizer::train",
                "merge %d/%d vocab=%d uniqueWords=%d",
                mergesDone,
                mergesTarget > 0 ? mergesTarget : mergesDone,
                this->vocabSize(),
                wordCount);
        }
    }

    SmokeLog::progressDone();
}

std::vector<std::string> BPETokenizer::preTokenize(const std::string& text) const {
    std::vector<std::string> words;
    const size_t length = text.size();
    size_t index = 0;

    while (index < length) {
        while (index < length && std::isspace(static_cast<unsigned char>(text[index])))
            ++index;
        if (index >= length) break;

        const size_t start = index;
        while (index < length && !std::isspace(static_cast<unsigned char>(text[index])))
            ++index;

        if (words.empty()) {
            words.emplace_back(text, start, index - start);
        } else {
            std::string word;
            word.reserve(1 + (index - start));
            word.push_back(' ');
            word.append(text, start, index - start);
            words.push_back(std::move(word));
        }
    }

    return words;
}

std::vector<int> BPETokenizer::applyMergeRules(std::vector<int> tokenIds) const {
    if (this->mergeRules_.empty() || tokenIds.size() < 2)
        return tokenIds;

    while (tokenIds.size() >= 2) {
        int bestRank = INT_MAX;
        int bestPos = -1;

        for (size_t index = 0; index + 1 < tokenIds.size(); ++index) {
            const auto found = this->mergeRank_.find({tokenIds[index], tokenIds[index + 1]});
            if (found == this->mergeRank_.end()) continue;
            if (found->second < bestRank) {
                bestRank = found->second;
                bestPos = static_cast<int>(index);
            }
        }

        if (bestPos < 0) break;

        const MergeRule& rule = this->mergeRules_[static_cast<size_t>(bestRank)];
        tokenIds[static_cast<size_t>(bestPos)] = rule.resultId;
        tokenIds.erase(tokenIds.begin() + bestPos + 1);
    }

    return tokenIds;
}
