#ifndef BPETOKENIZER_HPP
#define BPETOKENIZER_HPP

#include <string>
#include <unordered_map>
#include <utility>
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

    /// <summary>text -> token ids unknown tokens become unk</summary>
    std::vector<int> encode(const std::string& text) const;

    /// <summary>token ids -> text (concat)</summary>
    std::string decode(const std::vector<int>& tokenIds) const;

    int vocabSize() const;
    int unknownTokenId() const;
    int tokenToId(const std::string& token) const;
    const std::string& idToToken(int tokenId) const;

private:
    using Pair = std::pair<std::string, std::string>;

    struct PairHash {
        std::size_t operator()(const Pair& pair) const;
    };

    std::unordered_map<std::string, int> tokenToId_;
    std::vector<std::string> idToToken_;
    std::vector<Pair> merges_;
    int unknownTokenId_ = 0;

    void addUnknownToken();
    void buildBaseVocabulary(const std::vector<std::string>& corpus);
    void learnMerges(const std::vector<std::string>& corpus, int vocabSize);
    std::vector<std::string> preTokenize(const std::string& text) const;
    std::vector<std::string> applyMerges(const std::vector<std::string>& tokens) const;
    int lookupTokenId(const std::string& token) const;
};

#endif // BPETOKENIZER_HPP
