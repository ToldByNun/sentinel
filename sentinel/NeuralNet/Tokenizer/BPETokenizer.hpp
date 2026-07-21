#ifndef BPETOKENIZER_HPP
#define BPETOKENIZER_HPP

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

class BPETokenizer {
public:
    BPETokenizer() = default;

    void train(const std::string& text, int vocabSize);
    void train(const std::vector<std::string>& corpus, int vocabSize);

    std::vector<int> encode(const std::string& text) const;
    std::string decode(const std::vector<int>& tokenIds) const;

    int vocabSize() const;
    int tokenToId(const std::string& token) const;
    const std::string& idToToken(int id) const;

private:
    using Pair = std::pair<std::string, std::string>;

    struct PairHash {
        std::size_t operator()(const Pair& pair) const;
    };

    std::unordered_map<std::string, int> tokenToId_;
    std::vector<std::string> idToToken_;
    std::vector<Pair> merges_;

    void buildBaseVocabulary(const std::vector<std::string>& corpus);
    void learnMerges(const std::vector<std::string>& corpus, int vocabSize);
    std::vector<std::string> preTokenize(const std::string& text) const;
    std::vector<std::string> applyMerges(const std::vector<std::string>& tokens) const;
};

#endif // BPETOKENIZER_HPP
