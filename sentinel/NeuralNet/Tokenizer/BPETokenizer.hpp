#ifndef BPETOKENIZER_HPP
#define BPETOKENIZER_HPP

#include <string>
#include <unordered_map>
#include <vector>

/// <summary>BPE tokenizer call train() before encode/decode unknown pieces map to unk</summary>
class BPETokenizer {
public:
    static constexpr const char* UnknownToken = "<unk>";

    BPETokenizer() = default;

    /// <summary>learn merges from one string</summary>
    void train(const std::string& text, int vocabSize);

    /// <summary>learn merges from a corpus</summary>
    /// <param name="vocabSize">target vocab size (must be bigger than #unique chars + unk)</param>
    void train(const std::vector<std::string>& corpus, int vocabSize);

    /// <summary>text -> token ids unknown tokens become unk spaces kept as leading space on words</summary>
    std::vector<int> encode(const std::string& text) const;

    /// <summary>token ids -> text (concat restores leading spaces)</summary>
    std::string decode(const std::vector<int>& tokenIds) const;

    int vocabSize() const;
    int unknownTokenId() const;
    int tokenToId(const std::string& token) const;
    const std::string& idToToken(int tokenId) const;

    /// <summary>true after train() or a successful load()</summary>
    bool isTrained() const;

    /// <summary>write binary Sentinel BPE (`.sbpe`) — vocab + merge rules</summary>
    void save(const std::string& path) const;

    /// <summary>replace state from a `.sbpe` file written by save()</summary>
    void load(const std::string& path);

    /// <summary>construct and load from path</summary>
    static BPETokenizer loadFrom(const std::string& path);

private:
    struct IntPair {
        int left = 0;
        int right = 0;

        bool operator==(const IntPair& other) const {
            return this->left == other.left && this->right == other.right;
        }
    };

    struct IntPairHash {
        std::size_t operator()(const IntPair& pair) const;
    };

    struct MergeRule {
        int leftId = 0;
        int rightId = 0;
        int resultId = 0;
    };

    std::unordered_map<std::string, int> tokenToId_;
    std::vector<std::string> idToToken_;
    std::vector<MergeRule> mergeRules_;
    std::unordered_map<IntPair, int, IntPairHash> mergeRank_;
    int byteToId_[256] = {};
    int unknownTokenId_ = 0;

    void addUnknownToken();
    void clear();
    void buildBaseVocabulary(const std::vector<std::string>& corpus);
    void learnMerges(const std::vector<std::string>& corpus, int vocabSize);
    void rebuildMergeRank();
    void rebuildByteToId();
    std::vector<std::string> preTokenize(const std::string& text) const;
    std::vector<int> applyMergeRules(std::vector<int> tokenIds) const;
    int lookupTokenId(const std::string& token) const;
};

#endif // BPETOKENIZER_HPP
