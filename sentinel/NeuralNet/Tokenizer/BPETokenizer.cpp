#include "BPETokenizer.hpp"

#include <chrono>
#include <cstdio>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <utility>

#if defined(_OPENMP)
#include <omp.h>
#endif

size_t BPETokenizer::PairHash::operator()(const Pair& pair) const {
    size_t hashFirst = std::hash<std::string>{}(pair.first);
    size_t hashSecond = std::hash<std::string>{}(pair.second);
    return hashFirst ^ (hashSecond << 1);
}

void BPETokenizer::train(const std::string& text, int vocabSize) {
    this->train(std::vector<std::string>{text}, vocabSize);
}

void BPETokenizer::train(const std::vector<std::string>& corpus, int vocabSize) {
    if (vocabSize <= 0) throw std::invalid_argument("vocabSize must be > 0");

    const auto start = std::chrono::steady_clock::now();

    this->tokenToId_.clear();
    this->idToToken_.clear();
    this->merges_.clear();

    this->addUnknownToken();
    this->buildBaseVocabulary(corpus);
    this->learnMerges(corpus, vocabSize);

    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
#if defined(_OPENMP)
    const int threads = omp_get_max_threads();
#else
    const int threads = 1;
#endif
    std::printf(
        "BPETokenizer::train: corpus=%zu vocab=%d merges=%zu threads=%d sec=%.2f\n",
        corpus.size(),
        this->vocabSize(),
        this->merges_.size(),
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
    std::vector<std::string> words = this->preTokenize(text);

    for (const std::string& word : words) {
        std::vector<std::string> tokens;
        tokens.reserve(word.size());
        for (char character : word)
            tokens.emplace_back(1, character);

        tokens = this->applyMerges(tokens);

        for (const std::string& token : tokens)
            tokenIds.push_back(this->lookupTokenId(token));
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
        for (char character : text) {
            std::string token(1, character);
            if (this->tokenToId_.find(token) != this->tokenToId_.end()) continue;
            this->tokenToId_[token] = static_cast<int>(this->idToToken_.size());
            this->idToToken_.push_back(token);
        }
    }
}

void BPETokenizer::learnMerges(const std::vector<std::string>& corpus, int vocabSize) {
    std::vector<std::vector<std::string>> words;
    const int corpusCount = static_cast<int>(corpus.size());

#if defined(_OPENMP)
    #pragma omp parallel
    {
        std::vector<std::vector<std::string>> localWords;
        #pragma omp for schedule(dynamic, 4) nowait
        for (int textIndex = 0; textIndex < corpusCount; ++textIndex) {
            for (const std::string& word : this->preTokenize(corpus[static_cast<size_t>(textIndex)])) {
                std::vector<std::string> tokens;
                tokens.reserve(word.size());
                for (char character : word)
                    tokens.emplace_back(1, character);
                if (!tokens.empty())
                    localWords.push_back(std::move(tokens));
            }
        }
        #pragma omp critical(bpe_words)
        {
            words.insert(
                words.end(),
                std::make_move_iterator(localWords.begin()),
                std::make_move_iterator(localWords.end()));
        }
    }
#else
    for (const std::string& text : corpus) {
        for (const std::string& word : this->preTokenize(text)) {
            std::vector<std::string> tokens;
            tokens.reserve(word.size());
            for (char character : word)
                tokens.emplace_back(1, character);
            if (!tokens.empty())
                words.push_back(std::move(tokens));
        }
    }
#endif

    const int wordCount = static_cast<int>(words.size());

    while (static_cast<int>(this->idToToken_.size()) < vocabSize) {
        std::unordered_map<Pair, int, PairHash> pairCounts;

#if defined(_OPENMP)
        #pragma omp parallel
        {
            std::unordered_map<Pair, int, PairHash> localCounts;
            #pragma omp for schedule(static) nowait
            for (int wordIndex = 0; wordIndex < wordCount; ++wordIndex) {
                const std::vector<std::string>& tokens = words[static_cast<size_t>(wordIndex)];
                for (size_t tokenIndex = 0; tokenIndex + 1 < tokens.size(); ++tokenIndex)
                    localCounts[{tokens[tokenIndex], tokens[tokenIndex + 1]}]++;
            }
            #pragma omp critical(bpe_pairs)
            {
                for (const auto& entry : localCounts)
                    pairCounts[entry.first] += entry.second;
            }
        }
#else
        for (const auto& tokens : words) {
            for (size_t tokenIndex = 0; tokenIndex + 1 < tokens.size(); ++tokenIndex)
                pairCounts[{tokens[tokenIndex], tokens[tokenIndex + 1]}]++;
        }
#endif

        if (pairCounts.empty()) break;

        auto best = pairCounts.begin();
        for (auto entry = pairCounts.begin(); entry != pairCounts.end(); ++entry) {
            if (entry->second > best->second) best = entry;
        }

        const Pair bestPair = best->first;
        const std::string merged = bestPair.first + bestPair.second;

        this->merges_.push_back(bestPair);
        this->tokenToId_[merged] = static_cast<int>(this->idToToken_.size());
        this->idToToken_.push_back(merged);

#if defined(_OPENMP)
        #pragma omp parallel for schedule(dynamic, 64)
#endif
        for (int wordIndex = 0; wordIndex < wordCount; ++wordIndex) {
            std::vector<std::string>& tokens = words[static_cast<size_t>(wordIndex)];
            std::vector<std::string> updated;
            updated.reserve(tokens.size());

            for (size_t tokenIndex = 0; tokenIndex < tokens.size();) {
                if (tokenIndex + 1 < tokens.size()
                    && tokens[tokenIndex] == bestPair.first
                    && tokens[tokenIndex + 1] == bestPair.second) {
                    updated.push_back(merged);
                    tokenIndex += 2;
                    continue;
                }

                updated.push_back(tokens[tokenIndex]);
                ++tokenIndex;
            }

            tokens = std::move(updated);
        }
    }
}

std::vector<std::string> BPETokenizer::preTokenize(const std::string& text) const {
    std::vector<std::string> words;
    std::istringstream stream(text);
    std::string word;
    bool isFirstWord = true;

    while (stream >> word) {
        if (isFirstWord) {
            words.push_back(word);
            isFirstWord = false;
            continue;
        }

        // keep a leading space so decode can restore whitespace (GPT-2 style)
        words.push_back(" " + word);
    }

    return words;
}

std::vector<std::string> BPETokenizer::applyMerges(const std::vector<std::string>& tokens) const {
    std::vector<std::string> result = tokens;

    for (const Pair& merge : this->merges_) {
        const std::string merged = merge.first + merge.second;
        std::vector<std::string> updated;
        updated.reserve(result.size());

        for (size_t tokenIndex = 0; tokenIndex < result.size();) {
            if (tokenIndex + 1 < result.size()
                && result[tokenIndex] == merge.first
                && result[tokenIndex + 1] == merge.second) {
                updated.push_back(merged);
                tokenIndex += 2;
                continue;
            }

            updated.push_back(result[tokenIndex]);
            ++tokenIndex;
        }

        result = std::move(updated);
    }

    return result;
}
