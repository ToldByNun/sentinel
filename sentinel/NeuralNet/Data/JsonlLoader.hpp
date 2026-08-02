#ifndef JSONLLOADER_HPP
#define JSONLLOADER_HPP

#include "TextRowReader.hpp"

#include <fstream>
#include <string>
#include <vector>

/// <summary>legacy alias; prefer CorpusRow</summary>
using JsonlRow = CorpusRow;

/// <summary>streams a .jsonl file one chunk of rows at a time</summary>
class JsonlChunkReader : public TextRowReader {
public:
    void open(const std::string& path) override;
    void rewind() override;
    bool nextRows(std::vector<CorpusRow>& out, int maxRows) override;
    bool isOpen() const override;
    size_t rowsRead() const override;

private:
    std::ifstream file;
    std::string path;
    size_t rowsConsumed = 0;
};

/// <summary>tiny JSONL reader no real JSON parser just pulls string fields</summary>
class JsonlLoader {
public:
    /// <summary>load up to maximumRows lines from a .jsonl file (maximumRows less than 1 means no limit)</summary>
    static std::vector<CorpusRow> load(const std::string& path, int maximumRows = 50);

    /// <summary>map source string to class id (T1=0, T2=1, else -1)</summary>
    static int sourceToLabel(const std::string& source);

    /// <summary>parse one JSONL line into a row; returns false if text is empty / unusable</summary>
    static bool tryParseLine(const std::string& line, CorpusRow& out);

private:
    static std::string extractString(const std::string& line, const std::string& key);
};

#endif // JSONLLOADER_HPP
