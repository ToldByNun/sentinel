#ifndef TEXTROWREADER_HPP
#define TEXTROWREADER_HPP

#include <memory>
#include <string>
#include <vector>

/// <summary>one corpus sample row (format-agnostic)</summary>
class CorpusRow {
public:
    std::string text;   // problem_statement / primary text
    std::string source; // e.g. Sera-4.6-Lite-T1 / T2
};

/// <summary>streams corpus rows in chunks; JSONL today, Arrow later</summary>
class TextRowReader {
public:
    virtual ~TextRowReader() = default;

    /// <summary>open path for reading; throws if the path cannot be opened</summary>
    virtual void open(const std::string& path) = 0;

    /// <summary>seek back to the start of the stream</summary>
    virtual void rewind() = 0;

    /// <summary>
    /// fill out with up to maxRows non-empty rows
    /// returns false when the stream is exhausted (out may still hold a final partial chunk)
    /// </summary>
    virtual bool nextRows(std::vector<CorpusRow>& out, int maxRows) = 0;

    virtual bool isOpen() const = 0;

    /// <summary>valid rows returned since open/rewind</summary>
    virtual size_t rowsRead() const = 0;
};

/// <summary>
/// create a reader for path: *.jsonl → JSONL; *.arrow / HF dataset dir → error until Arrow lands
/// </summary>
std::unique_ptr<TextRowReader> createTextRowReader(const std::string& path);

#endif // TEXTROWREADER_HPP
