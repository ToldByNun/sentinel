#ifndef JSONLLOADER_HPP
#define JSONLLOADER_HPP

#include <fstream>
#include <string>
#include <vector>

/// <summary>one SERA sample row (only the fields we care about)</summary>
class JsonlRow {
public:
    std::string text;   // problem_statement
    std::string source; // e.g. Sera-4.6-Lite-T1 / T2
};

/// <summary>streams a .jsonl file one chunk of rows at a time</summary>
class JsonlChunkReader {
public:
    /// <summary>open path for reading; throws if the file cannot be opened</summary>
    void open(const std::string& path);

    /// <summary>seek back to the start of the file</summary>
    void rewind();

    /// <summary>
    /// fill out with up to maxRows non-empty parsed rows
    /// returns false when the file is exhausted (out may still hold a final partial chunk)
    /// </summary>
    bool nextRows(std::vector<JsonlRow>& out, int maxRows);

    bool isOpen() const;

    /// <summary>valid rows returned since open/rewind</summary>
    size_t rowsRead() const;

private:
    std::ifstream file;
    std::string path;
    size_t rowsConsumed = 0;
};

/// <summary>tiny JSONL reader no real JSON parser just pulls string fields</summary>
class JsonlLoader {
public:
    /// <summary>load up to maximumRows lines from a .jsonl file (maximumRows less than 1 means no limit)</summary>
    static std::vector<JsonlRow> load(const std::string& path, int maximumRows = 50);

    /// <summary>map source string to class id (T1=0, T2=1, else -1)</summary>
    static int sourceToLabel(const std::string& source);

    /// <summary>parse one JSONL line into a row; returns false if text is empty / unusable</summary>
    static bool tryParseLine(const std::string& line, JsonlRow& out);

private:
    static std::string extractString(const std::string& line, const std::string& key);
};

#endif // JSONLLOADER_HPP
