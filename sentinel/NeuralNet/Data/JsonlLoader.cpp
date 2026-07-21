#include "JsonlLoader.hpp"

#include <fstream>
#include <stdexcept>

std::string JsonlLoader::extractString(const std::string& line, const std::string& key) {
    std::string pattern = "\"" + key + "\": \"";
    size_t start = line.find(pattern);

    if (start == std::string::npos) {
        pattern = "\"" + key + "\":\"";
        start = line.find(pattern);
    }

    if (start == std::string::npos) return "";

    start += pattern.size();
    std::string result;

    for (size_t index = start; index < line.size(); ++index) {
        const char character = line[index];
        if (character == '\\' && index + 1 < line.size()) {
            const char next = line[index + 1];

            if (next == 'n') result += '\n';
            else if (next == 't') result += '\t';
            else result += next;

            ++index;
            continue;
        }

        if (character == '"') break;

        result += character;
    }

    return result;
}

int JsonlLoader::sourceToLabel(const std::string& source) {
    if (source.find("T1") != std::string::npos) return 0;
    if (source.find("T2") != std::string::npos) return 1;

    return -1;
}

std::vector<JsonlRow> JsonlLoader::load(const std::string& path, int maximumRows) {
    std::ifstream file(path);
    if (!file.is_open()) throw std::runtime_error("JsonlLoader: cannot open " + path);

    std::vector<JsonlRow> rows;
    std::string line;

    while (static_cast<int>(rows.size()) < maximumRows && std::getline(file, line)) {
        if (line.empty()) continue;

        JsonlRow row;
        row.text = JsonlLoader::extractString(line, "problem_statement");
        row.source = JsonlLoader::extractString(line, "source");
        if (row.text.empty()) continue;

        rows.push_back(std::move(row));
    }

    return rows;
}
