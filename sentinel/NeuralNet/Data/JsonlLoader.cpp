#include "JsonlLoader.hpp"

#include <iterator>
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

bool JsonlLoader::tryParseLine(const std::string& line, JsonlRow& out) {
    if (line.empty()) return false;

    out.text = JsonlLoader::extractString(line, "problem_statement");
    out.source = JsonlLoader::extractString(line, "source");
    return !out.text.empty();
}

int JsonlLoader::sourceToLabel(const std::string& source) {
    if (source.find("T1") != std::string::npos) return 0;
    if (source.find("T2") != std::string::npos) return 1;

    return -1;
}

void JsonlChunkReader::open(const std::string& path) {
    if (path.empty()) throw std::invalid_argument("JsonlChunkReader::open empty path");

    this->file.close();
    this->file.clear();
    this->file.open(path);
    if (!this->file.is_open()) throw std::runtime_error("JsonlChunkReader: cannot open " + path);

    this->path = path;
    this->rowsConsumed = 0;
}

void JsonlChunkReader::rewind() {
    if (!this->file.is_open()) throw std::logic_error("JsonlChunkReader::rewind not open");

    this->file.clear();
    this->file.seekg(0, std::ios::beg);
    this->rowsConsumed = 0;
}

bool JsonlChunkReader::nextRows(std::vector<JsonlRow>& out, int maxRows) {
    if (!this->file.is_open()) throw std::logic_error("JsonlChunkReader::nextRows not open");
    if (maxRows <= 0) throw std::invalid_argument("JsonlChunkReader::nextRows maxRows must be > 0");

    out.clear();
    out.reserve(static_cast<size_t>(maxRows));

    std::string line;
    while (static_cast<int>(out.size()) < maxRows && std::getline(this->file, line)) {
        JsonlRow row;
        if (!JsonlLoader::tryParseLine(line, row)) continue;

        out.push_back(std::move(row));
        ++this->rowsConsumed;
    }

    return this->file.good();
}

bool JsonlChunkReader::isOpen() const {
    return this->file.is_open();
}

size_t JsonlChunkReader::rowsRead() const {
    return this->rowsConsumed;
}

std::vector<JsonlRow> JsonlLoader::load(const std::string& path, int maximumRows) {
    JsonlChunkReader reader;
    reader.open(path);

    std::vector<JsonlRow> rows;
    if (maximumRows > 0) {
        reader.nextRows(rows, maximumRows);
        return rows;
    }

    std::vector<JsonlRow> chunk;
    const int chunkSize = 1024;
    while (reader.nextRows(chunk, chunkSize) || !chunk.empty()) {
        rows.insert(rows.end(), std::make_move_iterator(chunk.begin()), std::make_move_iterator(chunk.end()));
        if (chunk.empty()) break;
        chunk.clear();
    }

    return rows;
}
