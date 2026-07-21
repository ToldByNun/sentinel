#ifndef JSONLLOADER_HPP
#define JSONLLOADER_HPP

#include <string>
#include <vector>

/// <summary>One SERA sample row (only the fields we care about).</summary>
class JsonlRow {
public:
    std::string text;   // problem_statement
    std::string source; // e.g. Sera-4.6-Lite-T1 / T2
};

/// <summary>Tiny JSONL reader. No real JSON parser — just pulls string fields.</summary>
class JsonlLoader {
public:
    /// <summary>Load up to maximumRows lines from a .jsonl file.</summary>
    static std::vector<JsonlRow> load(const std::string& path, int maximumRows = 50);

    /// <summary>Map source string to class id (T1=0, T2=1, else -1).</summary>
    static int sourceToLabel(const std::string& source);

private:
    static std::string extractString(const std::string& line, const std::string& key);
};

#endif // JSONLLOADER_HPP
