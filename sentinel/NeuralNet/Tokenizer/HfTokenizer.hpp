#ifndef HFTOKENIZER_HPP
#define HFTOKENIZER_HPP

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace HuggingFace {

/// <summary>
/// HuggingFace tokenizers.json loader (BPE + ByteLevel, Llama/Mistral/Qwen2-style).
/// Native Sentinel `.sbpe` remains on BPETokenizer — this is for HF checkpoints only.
/// </summary>
class Tokenizer {
public:
    Tokenizer() = default;

    /// <summary>
    /// load tokenizer.json from a local path, Hub repo id (org/name[@rev]), or HF URL.
    /// Hub downloads use the standard HF cache (`hf` / huggingface_hub).
    /// </summary>
    static Tokenizer load(const std::string& pathOrDirectory);

    /// <summary>text → token ids; optionally prepend BOS when configured</summary>
    std::vector<int> encode(const std::string& text, bool addSpecialTokens = true) const;

    /// <summary>token ids → text; optionally drop special tokens</summary>
    std::string decode(const std::vector<int>& tokenIds, bool skipSpecialTokens = true) const;

    int vocabSize() const;
    int bosTokenId() const; // -1 if none
    int eosTokenId() const;
    int padTokenId() const;
    int unkTokenId() const;
    bool isLoaded() const;
    bool ignoreMerges() const;

    /// <summary>tiny ByteLevel BPE fixture + encode/decode + ignore_merges</summary>
    static void runTokenizerSmokeDemo();

private:
    enum class PreTokenizeKind {
        ByteLevelRaw = 0,
        ByteLevelGpt2Regex = 1,
        LlamaSplitThenByteLevel = 2
    };

    std::unordered_map<std::string, int> tokenToId_;
    std::vector<std::string> idToToken_;
    std::vector<std::pair<std::string, std::string>> merges_; // ordered by rank
    std::unordered_map<std::string, int> mergeRank_; // "left\0right" → rank
    std::unordered_set<int> specialTokenIds_;
    std::string bytesToUnicode_[256];
    std::unordered_map<std::string, unsigned char> unicodeToByte_;
    PreTokenizeKind preTokenizeKind_ = PreTokenizeKind::ByteLevelRaw;
    bool ignoreMerges_ = false;
    bool addPrefixSpace_ = false;
    int bosTokenId_ = -1;
    int eosTokenId_ = -1;
    int padTokenId_ = -1;
    int unkTokenId_ = -1;

    void clear();
    void buildByteMaps();
    void rebuildMergeRank();
    std::string utf8ToByteLevel(const std::string& text) const;
    std::string byteLevelToUtf8(const std::string& token) const;
    std::vector<std::string> preTokenize(const std::string& text) const;
    std::vector<int> bpeEncodePiece(const std::string& byteLevelPiece) const;
    static std::string mergeKey(const std::string& left, const std::string& right);
};

} // namespace HuggingFace

#endif // HFTOKENIZER_HPP
