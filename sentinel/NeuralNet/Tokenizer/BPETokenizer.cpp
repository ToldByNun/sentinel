#include "BPETokenizer.hpp"

#include <algorithm>
#include <limits>
#include <sstream>
#include <stdexcept>

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

    this->tokenToId_.clear();
    this->idToToken_.clear();
    this->merges_.clear();

    this->buildBaseVocabulary(corpus);
    this->learnMerges(corpus, vocabSize);
}

std::vector<int> BPETokenizer::encode(const std::string& text) const {
    std::vector<int> ids;
    std::vector<std::string> words = this->preTokenize(text);

    for (const std::string& word : words) {
        std::vector<std::string> tokens;
        tokens.reserve(word.size());
        for (char character : word) {
            tokens.emplace_back(1, character);
        }

        tokens = this->applyMerges(tokens);

        for (const std::string& token : tokens) {
            auto it = this->tokenToId_.find(token);
            if (it == this->tokenToId_.end()) throw std::runtime_error("unknown token during encode: " + token);
            ids.push_back(it->second);
        }
    }

    return ids;
}

std::string BPETokenizer::decode(const std::vector<int>& tokenIds) const {
    std::string result;
    result.reserve(tokenIds.size());

    for (int tokenId : tokenIds) {
        if (tokenId < 0 || tokenId >= static_cast<int>(this->idToToken_.size())) {
            throw std::out_of_range("token id out of range");
        }
        result += this->idToToken_[tokenId];
    }

    return result;
}

int BPETokenizer::vocabSize() const {
    return static_cast<int>(this->idToToken_.size());
}

int BPETokenizer::tokenToId(const std::string& token) const {
    auto it = this->tokenToId_.find(token);
    if (it == this->tokenToId_.end()) throw std::runtime_error("unknown token: " + token);
    return it->second;
}

const std::string& BPETokenizer::idToToken(int id) const {
    if (id < 0 || id >= static_cast<int>(this->idToToken_.size())) throw std::out_of_range("token id out of range");
    return this->idToToken_[id];
}

void BPETokenizer::buildBaseVocabulary(const std::vector<std::string>& corpus) {
    for (const std::string& text : corpus) {
        for (char character : text) {
            std::string token(1, character);
            if (this->tokenToId_.find(token) == this->tokenToId_.end()) {
                this->tokenToId_[token] = static_cast<int>(this->idToToken_.size());
                this->idToToken_.push_back(token);
            }
        }
    }
}

void BPETokenizer::learnMerges(const std::vector<std::string>& corpus, int vocabSize) {
    std::vector<std::vector<std::string>> words;

    for (const std::string& text : corpus) {
        for (const std::string& word : this->preTokenize(text)) {
            std::vector<std::string> tokens;
            tokens.reserve(word.size());

            for (char character : word) {
                tokens.emplace_back(1, character);
            }

            if (!tokens.empty()) words.push_back(std::move(tokens));
        }
    }

    while (static_cast<int>(this->idToToken_.size()) < vocabSize) {
        std::unordered_map<Pair, int, PairHash> pairCounts;

        for (const auto& tokens : words) {
            for (size_t tokenIndex = 0; tokenIndex + 1 < tokens.size(); ++tokenIndex) {
                pairCounts[{tokens[tokenIndex], tokens[tokenIndex + 1]}]++;
            }
        }

        if (pairCounts.empty()) break;

        auto best = std::max_element(
            pairCounts.begin(),
            pairCounts.end(),
            [](const auto& left, const auto& right) {
                return left.second < right.second;
            }
        );

        const Pair& bestPair = best->first;
        const std::string merged = bestPair.first + bestPair.second;

        this->merges_.push_back(bestPair);
        this->tokenToId_[merged] = static_cast<int>(this->idToToken_.size());
        this->idToToken_.push_back(merged);

        for (auto& tokens : words) {
            std::vector<std::string> updated;
            updated.reserve(tokens.size());

            for (size_t tokenIndex = 0; tokenIndex < tokens.size();) {
                if (tokenIndex + 1 < tokens.size()
                    && tokens[tokenIndex] == bestPair.first
                    && tokens[tokenIndex + 1] == bestPair.second) {
                    updated.push_back(merged);
                    tokenIndex += 2;
                } else {
                    updated.push_back(tokens[tokenIndex]);
                    ++tokenIndex;
                }
            }

            tokens = std::move(updated);
        }
    }
}

std::vector<std::string> BPETokenizer::preTokenize(const std::string& text) const {
    std::vector<std::string> words;
    std::istringstream stream(text);
    std::string word;

    while (stream >> word) {
        words.push_back(word);
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
            } else {
                updated.push_back(result[tokenIndex]);
                ++tokenIndex;
            }
        }

        result = std::move(updated);
    }

    return result;
}
