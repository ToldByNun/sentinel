#include "SentinelModelConfig.hpp"

#include "../Utils/SmokeLog.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace SentinelModel {
namespace {

void skipWs(const std::string& text, size_t& i) {
    while (i < text.size() && (text[i] == ' ' || text[i] == '\n' || text[i] == '\r' || text[i] == '\t'))
        ++i;
}

std::string parseJsonString(const std::string& json, size_t& i) {
    skipWs(json, i);
    if (i >= json.size() || json[i] != '"')
        throw std::runtime_error("SentinelModel config: expected string");
    ++i;
    std::string out;
    while (i < json.size()) {
        const char ch = json[i++];
        if (ch == '"') return out;
        if (ch == '\\') {
            if (i >= json.size()) throw std::runtime_error("SentinelModel config: bad escape");
            const char esc = json[i++];
            switch (esc) {
            case '"': out.push_back('"'); break;
            case '\\': out.push_back('\\'); break;
            case '/': out.push_back('/'); break;
            case 'n': out.push_back('\n'); break;
            case 'r': out.push_back('\r'); break;
            case 't': out.push_back('\t'); break;
            case 'u': {
                if (i + 4 > json.size()) throw std::runtime_error("SentinelModel config: bad unicode escape");
                unsigned code = 0;
                for (int n = 0; n < 4; ++n) {
                    const char h = json[i++];
                    code <<= 4;
                    if (h >= '0' && h <= '9') code |= static_cast<unsigned>(h - '0');
                    else if (h >= 'a' && h <= 'f') code |= static_cast<unsigned>(h - 'a' + 10);
                    else if (h >= 'A' && h <= 'F') code |= static_cast<unsigned>(h - 'A' + 10);
                    else throw std::runtime_error("SentinelModel config: bad unicode escape");
                }
                if (code > 0x7F)
                    throw std::runtime_error("SentinelModel config: non-ASCII unicode not supported");
                out.push_back(static_cast<char>(code));
                break;
            }
            default: throw std::runtime_error("SentinelModel config: unsupported escape");
            }
        } else {
            out.push_back(ch);
        }
    }
    throw std::runtime_error("SentinelModel config: unterminated string");
}

void skipJsonValue(const std::string& json, size_t& i);

void skipJsonObject(const std::string& json, size_t& i) {
    skipWs(json, i);
    if (i >= json.size() || json[i] != '{')
        throw std::runtime_error("SentinelModel config: expected object");
    ++i;
    skipWs(json, i);
    if (i < json.size() && json[i] == '}') {
        ++i;
        return;
    }
    while (true) {
        (void)parseJsonString(json, i);
        skipWs(json, i);
        if (i >= json.size() || json[i] != ':')
            throw std::runtime_error("SentinelModel config: expected ':' in object");
        ++i;
        skipJsonValue(json, i);
        skipWs(json, i);
        if (i >= json.size()) throw std::runtime_error("SentinelModel config: truncated object");
        if (json[i] == '}') {
            ++i;
            return;
        }
        if (json[i] != ',') throw std::runtime_error("SentinelModel config: expected ',' in object");
        ++i;
    }
}

void skipJsonArray(const std::string& json, size_t& i) {
    skipWs(json, i);
    if (i >= json.size() || json[i] != '[')
        throw std::runtime_error("SentinelModel config: expected array");
    ++i;
    skipWs(json, i);
    if (i < json.size() && json[i] == ']') {
        ++i;
        return;
    }
    while (true) {
        skipJsonValue(json, i);
        skipWs(json, i);
        if (i >= json.size()) throw std::runtime_error("SentinelModel config: truncated array");
        if (json[i] == ']') {
            ++i;
            return;
        }
        if (json[i] != ',') throw std::runtime_error("SentinelModel config: expected ',' in array");
        ++i;
    }
}

void skipJsonValue(const std::string& json, size_t& i) {
    skipWs(json, i);
    if (i >= json.size()) throw std::runtime_error("SentinelModel config: truncated value");
    const char ch = json[i];
    if (ch == '"') {
        (void)parseJsonString(json, i);
        return;
    }
    if (ch == '{') {
        skipJsonObject(json, i);
        return;
    }
    if (ch == '[') {
        skipJsonArray(json, i);
        return;
    }
    if (ch == 't' || ch == 'f' || ch == 'n' || ch == '-' || (ch >= '0' && ch <= '9')) {
        while (i < json.size()) {
            const char c = json[i];
            if (c == ',' || c == '}' || c == ']' || c == ' ' || c == '\n' || c == '\r' || c == '\t')
                break;
            ++i;
        }
        return;
    }
    throw std::runtime_error("SentinelModel config: unexpected value");
}

bool tryParseJsonNull(const std::string& json, size_t& i) {
    skipWs(json, i);
    if (i + 4 <= json.size()
        && json[i] == 'n' && json[i + 1] == 'u' && json[i + 2] == 'l' && json[i + 3] == 'l') {
        i += 4;
        return true;
    }
    return false;
}

bool parseJsonBool(const std::string& json, size_t& i) {
    skipWs(json, i);
    if (i + 4 <= json.size()
        && json[i] == 't' && json[i + 1] == 'r' && json[i + 2] == 'u' && json[i + 3] == 'e') {
        i += 4;
        return true;
    }
    if (i + 5 <= json.size()
        && json[i] == 'f' && json[i + 1] == 'a' && json[i + 2] == 'l' && json[i + 3] == 's'
        && json[i + 4] == 'e') {
        i += 5;
        return false;
    }
    throw std::runtime_error("SentinelModel config: expected boolean");
}

double parseJsonNumber(const std::string& json, size_t& i) {
    skipWs(json, i);
    if (i >= json.size()) throw std::runtime_error("SentinelModel config: expected number");
    const size_t begin = i;
    if (json[i] == '-') ++i;
    if (i >= json.size() || (json[i] < '0' || json[i] > '9'))
        throw std::runtime_error("SentinelModel config: expected number");
    while (i < json.size() && json[i] >= '0' && json[i] <= '9') ++i;
    if (i < json.size() && json[i] == '.') {
        ++i;
        while (i < json.size() && json[i] >= '0' && json[i] <= '9') ++i;
    }
    if (i < json.size() && (json[i] == 'e' || json[i] == 'E')) {
        ++i;
        if (i < json.size() && (json[i] == '+' || json[i] == '-')) ++i;
        if (i >= json.size() || json[i] < '0' || json[i] > '9')
            throw std::runtime_error("SentinelModel config: bad exponent");
        while (i < json.size() && json[i] >= '0' && json[i] <= '9') ++i;
    }
    return std::stod(json.substr(begin, i - begin));
}

struct RawFields {
    bool hasFormat = false;
    bool hasVocabSize = false;
    bool hasEmbeddingDim = false;
    bool hasMaxPosition = false;
    bool hasBlockCount = false;
    bool hasHeadCount = false;
    bool hasKvHeadCount = false;
    bool hasIntermediateSize = false;
    bool hasRopeTheta = false;
    bool hasUseBias = false;
    bool hasTieEmbedding = false;
    bool hasRmsNormEps = false;
    bool hasLearningRate = false;
    bool hasWeights = false;

    std::string format;
    int vocabSize = 0;
    int embeddingDim = 0;
    int maxPosition = 0;
    int blockCount = 0;
    int headCount = 0;
    int kvHeadCount = 0;
    int intermediateSize = 0;
    float ropeTheta = 10000.0f;
    bool useBias = true;
    bool tieEmbedding = true;
    float rmsNormEps = 1.0e-5f;
    float learningRate = 3.0e-4f;
    std::string weights;
};

void requirePositive(const char* name, int value) {
    if (value <= 0)
        throw std::runtime_error(std::string("SentinelModel config: ") + name + " must be > 0");
}

Config buildValidated(const RawFields& raw) {
    if (!raw.hasFormat || !isSentinelModelFormat(raw.format))
        throw std::runtime_error(
            "SentinelModel config: format must be \"sentinel-model\""
            + (raw.hasFormat ? (" (got '" + raw.format + "')") : std::string()));

    if (!raw.hasVocabSize) throw std::runtime_error("SentinelModel config: missing vocab_size");
    if (!raw.hasEmbeddingDim) throw std::runtime_error("SentinelModel config: missing embedding_dim");
    if (!raw.hasMaxPosition) throw std::runtime_error("SentinelModel config: missing max_position");
    if (!raw.hasBlockCount) throw std::runtime_error("SentinelModel config: missing block_count");
    if (!raw.hasHeadCount) throw std::runtime_error("SentinelModel config: missing head_count");

    requirePositive("vocab_size", raw.vocabSize);
    requirePositive("embedding_dim", raw.embeddingDim);
    requirePositive("max_position", raw.maxPosition);
    requirePositive("block_count", raw.blockCount);
    requirePositive("head_count", raw.headCount);

    const int kvHeads = raw.hasKvHeadCount ? raw.kvHeadCount : raw.headCount;
    requirePositive("kv_head_count", kvHeads);
    if (raw.headCount % kvHeads != 0)
        throw std::runtime_error("SentinelModel config: head_count must be divisible by kv_head_count");
    if (raw.embeddingDim % raw.headCount != 0)
        throw std::runtime_error("SentinelModel config: embedding_dim must be divisible by head_count");

    if (raw.hasIntermediateSize && raw.intermediateSize < 0)
        throw std::runtime_error("SentinelModel config: intermediate_size must be >= 0");
    if (raw.hasRopeTheta && !(raw.ropeTheta > 0.0f))
        throw std::runtime_error("SentinelModel config: rope_theta must be > 0");
    if (raw.hasRmsNormEps && !(raw.rmsNormEps > 0.0f))
        throw std::runtime_error("SentinelModel config: rms_norm_eps must be > 0");
    if (raw.hasLearningRate && !(raw.learningRate > 0.0f))
        throw std::runtime_error("SentinelModel config: learning_rate must be > 0");

    Config cfg;
    cfg.format = "sentinel-model";
    cfg.vocabSize = raw.vocabSize;
    cfg.embeddingDim = raw.embeddingDim;
    cfg.maxPosition = raw.maxPosition;
    cfg.blockCount = raw.blockCount;
    cfg.headCount = raw.headCount;
    cfg.kvHeadCount = kvHeads;
    cfg.intermediateSize = raw.hasIntermediateSize ? raw.intermediateSize : 0;
    cfg.ropeTheta = raw.hasRopeTheta ? raw.ropeTheta : 10000.0f;
    cfg.useBias = raw.hasUseBias ? raw.useBias : true;
    cfg.tieEmbedding = raw.hasTieEmbedding ? raw.tieEmbedding : true;
    cfg.rmsNormEps = raw.hasRmsNormEps ? raw.rmsNormEps : 1.0e-5f;
    cfg.learningRate = raw.hasLearningRate ? raw.learningRate : 3.0e-4f;
    cfg.weights = raw.hasWeights ? raw.weights : std::string();
    return cfg;
}

void ingestJsonTopLevel(const std::string& json, RawFields& raw) {
    size_t i = 0;
    skipWs(json, i);
    if (i >= json.size() || json[i] != '{')
        throw std::runtime_error("SentinelModel config: expected top-level object");
    ++i;
    skipWs(json, i);
    if (i < json.size() && json[i] == '}') return;

    while (true) {
        const std::string key = parseJsonString(json, i);
        skipWs(json, i);
        if (i >= json.size() || json[i] != ':')
            throw std::runtime_error("SentinelModel config: expected ':' after key");
        ++i;
        skipWs(json, i);

        if (key == "format") {
            raw.format = parseJsonString(json, i);
            raw.hasFormat = true;
        } else if (key == "vocab_size") {
            raw.vocabSize = static_cast<int>(parseJsonNumber(json, i));
            raw.hasVocabSize = true;
        } else if (key == "embedding_dim") {
            raw.embeddingDim = static_cast<int>(parseJsonNumber(json, i));
            raw.hasEmbeddingDim = true;
        } else if (key == "max_position") {
            raw.maxPosition = static_cast<int>(parseJsonNumber(json, i));
            raw.hasMaxPosition = true;
        } else if (key == "block_count") {
            raw.blockCount = static_cast<int>(parseJsonNumber(json, i));
            raw.hasBlockCount = true;
        } else if (key == "head_count") {
            raw.headCount = static_cast<int>(parseJsonNumber(json, i));
            raw.hasHeadCount = true;
        } else if (key == "kv_head_count") {
            raw.kvHeadCount = static_cast<int>(parseJsonNumber(json, i));
            raw.hasKvHeadCount = true;
        } else if (key == "intermediate_size") {
            raw.intermediateSize = static_cast<int>(parseJsonNumber(json, i));
            raw.hasIntermediateSize = true;
        } else if (key == "rope_theta") {
            raw.ropeTheta = static_cast<float>(parseJsonNumber(json, i));
            raw.hasRopeTheta = true;
        } else if (key == "use_bias") {
            raw.useBias = parseJsonBool(json, i);
            raw.hasUseBias = true;
        } else if (key == "tie_embedding") {
            raw.tieEmbedding = parseJsonBool(json, i);
            raw.hasTieEmbedding = true;
        } else if (key == "rms_norm_eps") {
            raw.rmsNormEps = static_cast<float>(parseJsonNumber(json, i));
            raw.hasRmsNormEps = true;
        } else if (key == "learning_rate") {
            raw.learningRate = static_cast<float>(parseJsonNumber(json, i));
            raw.hasLearningRate = true;
        } else if (key == "weights") {
            raw.hasWeights = true;
            if (tryParseJsonNull(json, i))
                raw.weights.clear();
            else
                raw.weights = parseJsonString(json, i);
        } else {
            skipJsonValue(json, i);
        }

        skipWs(json, i);
        if (i >= json.size()) throw std::runtime_error("SentinelModel config: truncated object");
        if (json[i] == '}') {
            ++i;
            break;
        }
        if (json[i] != ',') throw std::runtime_error("SentinelModel config: expected ',' between keys");
        ++i;
    }
}

std::string trimCopy(const std::string& s) {
    size_t begin = 0;
    while (begin < s.size() && std::isspace(static_cast<unsigned char>(s[begin]))) ++begin;
    size_t end = s.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(s[end - 1]))) --end;
    return s.substr(begin, end - begin);
}

std::string stripYamlComment(const std::string& line) {
    bool inSingle = false;
    bool inDouble = false;
    for (size_t i = 0; i < line.size(); ++i) {
        const char ch = line[i];
        if (ch == '\'' && !inDouble) inSingle = !inSingle;
        else if (ch == '"' && !inSingle) inDouble = !inDouble;
        else if (ch == '#' && !inSingle && !inDouble)
            return trimCopy(line.substr(0, i));
    }
    return trimCopy(line);
}

std::string unquoteYamlScalar(const std::string& raw) {
    if (raw.size() >= 2) {
        if ((raw.front() == '"' && raw.back() == '"') || (raw.front() == '\'' && raw.back() == '\''))
            return raw.substr(1, raw.size() - 2);
    }
    return raw;
}

bool parseYamlBool(const std::string& raw, bool& out) {
    std::string lower = raw;
    for (char& c : lower) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (lower == "true" || lower == "yes" || lower == "on") {
        out = true;
        return true;
    }
    if (lower == "false" || lower == "no" || lower == "off") {
        out = false;
        return true;
    }
    return false;
}

bool isYamlNull(const std::string& raw) {
    std::string lower = raw;
    for (char& c : lower) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return lower == "null" || lower == "~" || lower.empty();
}

void setYamlField(RawFields& raw, const std::string& key, const std::string& valueRaw) {
    const std::string value = unquoteYamlScalar(trimCopy(valueRaw));
    if (key == "format") {
        raw.format = value;
        raw.hasFormat = true;
    } else if (key == "vocab_size") {
        raw.vocabSize = std::stoi(value);
        raw.hasVocabSize = true;
    } else if (key == "embedding_dim") {
        raw.embeddingDim = std::stoi(value);
        raw.hasEmbeddingDim = true;
    } else if (key == "max_position") {
        raw.maxPosition = std::stoi(value);
        raw.hasMaxPosition = true;
    } else if (key == "block_count") {
        raw.blockCount = std::stoi(value);
        raw.hasBlockCount = true;
    } else if (key == "head_count") {
        raw.headCount = std::stoi(value);
        raw.hasHeadCount = true;
    } else if (key == "kv_head_count") {
        raw.kvHeadCount = std::stoi(value);
        raw.hasKvHeadCount = true;
    } else if (key == "intermediate_size") {
        raw.intermediateSize = std::stoi(value);
        raw.hasIntermediateSize = true;
    } else if (key == "rope_theta") {
        raw.ropeTheta = std::stof(value);
        raw.hasRopeTheta = true;
    } else if (key == "use_bias") {
        bool b = false;
        if (!parseYamlBool(value, b))
            throw std::runtime_error("SentinelModel config: use_bias must be boolean");
        raw.useBias = b;
        raw.hasUseBias = true;
    } else if (key == "tie_embedding") {
        bool b = false;
        if (!parseYamlBool(value, b))
            throw std::runtime_error("SentinelModel config: tie_embedding must be boolean");
        raw.tieEmbedding = b;
        raw.hasTieEmbedding = true;
    } else if (key == "rms_norm_eps") {
        raw.rmsNormEps = std::stof(value);
        raw.hasRmsNormEps = true;
    } else if (key == "learning_rate") {
        raw.learningRate = std::stof(value);
        raw.hasLearningRate = true;
    } else if (key == "weights") {
        raw.hasWeights = true;
        raw.weights = isYamlNull(value) ? std::string() : value;
    }
    // unknown keys ignored (forward compatible)
}

void ingestYaml(const std::string& yaml, RawFields& raw) {
    std::istringstream in(yaml);
    std::string line;
    int lineNo = 0;
    while (std::getline(in, line)) {
        ++lineNo;
        std::string trimmed = stripYamlComment(line);
        if (trimmed.empty() || trimmed == "---" || trimmed == "...")
            continue;
        if (!trimmed.empty() && (trimmed.front() == ' ' || trimmed.front() == '\t'))
            throw std::runtime_error(
                "SentinelModel config: indented YAML keys are not supported (line "
                + std::to_string(lineNo) + ")");
        if (!trimmed.empty() && (trimmed.front() == '-' || trimmed.front() == '[' || trimmed.front() == '{'))
            throw std::runtime_error(
                "SentinelModel config: YAML lists/nested maps are not supported (line "
                + std::to_string(lineNo) + ")");

        const auto colon = trimmed.find(':');
        if (colon == std::string::npos)
            throw std::runtime_error(
                "SentinelModel config: expected 'key: value' (line " + std::to_string(lineNo) + ")");
        const std::string key = trimCopy(trimmed.substr(0, colon));
        const std::string value = trimCopy(trimmed.substr(colon + 1));
        if (key.empty())
            throw std::runtime_error(
                "SentinelModel config: empty YAML key (line " + std::to_string(lineNo) + ")");
        try {
            setYamlField(raw, key, value);
        } catch (const std::exception& ex) {
            throw std::runtime_error(
                std::string("SentinelModel config: ") + ex.what() + " (line " + std::to_string(lineNo) + ")");
        }
    }
}

std::string readEntireFile(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("SentinelModel config: cannot open " + path);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

std::string toLowerExt(const std::string& path) {
    const auto slash = path.find_last_of("/\\");
    const std::string leaf = (slash == std::string::npos) ? path : path.substr(slash + 1);
    const auto dot = leaf.find_last_of('.');
    if (dot == std::string::npos) return {};
    std::string ext = leaf.substr(dot + 1);
    for (char& c : ext) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return ext;
}

bool pathLooksLikeDirectory(const std::string& path) {
    if (path.empty()) return false;
    const char last = path.back();
    if (last == '/' || last == '\\') return true;
    std::error_code ec;
    if (std::filesystem::is_directory(path, ec)) return true;
    const std::string ext = toLowerExt(path);
    return ext != "json" && ext != "yaml" && ext != "yml";
}

bool looksLikeJson(const std::string& text) {
    size_t i = 0;
    skipWs(text, i);
    return i < text.size() && text[i] == '{';
}

std::string formatFloat(float value) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.8g", static_cast<double>(value));
    return std::string(buf);
}

std::string escapeJsonString(const std::string& value) {
    std::string out;
    out.reserve(value.size() + 8);
    for (char ch : value) {
        switch (ch) {
        case '"': out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default: out.push_back(ch); break;
        }
    }
    return out;
}

bool expectRejectJson(const std::string& json, const char* needle) {
    try {
        (void)parseConfigJson(json);
        return false;
    } catch (const std::exception& ex) {
        return std::string(ex.what()).find(needle) != std::string::npos;
    }
}

} // namespace

bool isSentinelModelFormat(const std::string& format) {
    return format == "sentinel-model";
}

Config parseConfigJson(const std::string& json) {
    RawFields raw;
    ingestJsonTopLevel(json, raw);
    return buildValidated(raw);
}

Config parseConfigYaml(const std::string& yaml) {
    RawFields raw;
    ingestYaml(yaml, raw);
    return buildValidated(raw);
}

Config parseConfigText(const std::string& text, const std::string& pathHint) {
    const std::string ext = toLowerExt(pathHint);
    if (ext == "json" || looksLikeJson(text))
        return parseConfigJson(text);
    if (ext == "yaml" || ext == "yml")
        return parseConfigYaml(text);
    // Bare text: JSON if it starts with '{', otherwise YAML.
    if (looksLikeJson(text))
        return parseConfigJson(text);
    return parseConfigYaml(text);
}

Config loadConfig(const std::string& pathOrDirectory) {
    if (pathOrDirectory.empty())
        throw std::invalid_argument("SentinelModel::loadConfig empty path");

    std::string path = pathOrDirectory;
    if (pathLooksLikeDirectory(path)) {
        namespace fs = std::filesystem;
        const fs::path dir(path);
        const fs::path candidates[] = {
            dir / "model.json",
            dir / "model.yaml",
            dir / "model.yml",
        };
        bool found = false;
        for (const fs::path& candidate : candidates) {
            std::error_code ec;
            if (fs::is_regular_file(candidate, ec)) {
                path = candidate.string();
                found = true;
                break;
            }
        }
        if (!found)
            throw std::runtime_error(
                "SentinelModel::loadConfig no model.json/model.yaml/model.yml in " + pathOrDirectory);
    }

    return parseConfigText(readEntireFile(path), path);
}

std::string configDirectory(const std::string& pathOrDirectory) {
    if (pathOrDirectory.empty()) return {};
    namespace fs = std::filesystem;
    const fs::path p(pathOrDirectory);
    std::error_code ec;
    if (fs::is_directory(p, ec) || pathLooksLikeDirectory(pathOrDirectory))
        return p.string();
    return p.has_parent_path() ? p.parent_path().string() : std::string(".");
}

std::string resolveWeightsPath(const std::string& configPathOrDirectory, const std::string& weights) {
    if (weights.empty()) return {};
    namespace fs = std::filesystem;
    const fs::path w(weights);
    if (w.is_absolute()) return w.string();
    const fs::path base = configDirectory(configPathOrDirectory);
    return (base / w).lexically_normal().string();
}

std::string serializeConfigJson(const Config& config) {
    if (!isSentinelModelFormat(config.format))
        throw std::invalid_argument("SentinelModel::serializeConfigJson format must be sentinel-model");
    if (config.vocabSize <= 0 || config.embeddingDim <= 0 || config.maxPosition <= 0
        || config.blockCount <= 0 || config.headCount <= 0 || config.kvHeadCount <= 0)
        throw std::invalid_argument("SentinelModel::serializeConfigJson incomplete config");

    std::ostringstream out;
    out << "{\n";
    out << "  \"format\": \"sentinel-model\",\n";
    out << "  \"vocab_size\": " << config.vocabSize << ",\n";
    out << "  \"embedding_dim\": " << config.embeddingDim << ",\n";
    out << "  \"max_position\": " << config.maxPosition << ",\n";
    out << "  \"block_count\": " << config.blockCount << ",\n";
    out << "  \"head_count\": " << config.headCount << ",\n";
    out << "  \"kv_head_count\": " << config.kvHeadCount << ",\n";
    out << "  \"intermediate_size\": " << config.intermediateSize << ",\n";
    out << "  \"rope_theta\": " << formatFloat(config.ropeTheta) << ",\n";
    out << "  \"use_bias\": " << (config.useBias ? "true" : "false") << ",\n";
    out << "  \"tie_embedding\": " << (config.tieEmbedding ? "true" : "false") << ",\n";
    out << "  \"rms_norm_eps\": " << formatFloat(config.rmsNormEps) << ",\n";
    out << "  \"learning_rate\": " << formatFloat(config.learningRate) << ",\n";
    if (config.weights.empty())
        out << "  \"weights\": null\n";
    else
        out << "  \"weights\": \"" << escapeJsonString(config.weights) << "\"\n";
    out << "}\n";
    return out.str();
}

std::string serializeConfigYaml(const Config& config) {
    if (!isSentinelModelFormat(config.format))
        throw std::invalid_argument("SentinelModel::serializeConfigYaml format must be sentinel-model");
    if (config.vocabSize <= 0 || config.embeddingDim <= 0 || config.maxPosition <= 0
        || config.blockCount <= 0 || config.headCount <= 0 || config.kvHeadCount <= 0)
        throw std::invalid_argument("SentinelModel::serializeConfigYaml incomplete config");

    std::ostringstream out;
    out << "format: sentinel-model\n";
    out << "vocab_size: " << config.vocabSize << "\n";
    out << "embedding_dim: " << config.embeddingDim << "\n";
    out << "max_position: " << config.maxPosition << "\n";
    out << "block_count: " << config.blockCount << "\n";
    out << "head_count: " << config.headCount << "\n";
    out << "kv_head_count: " << config.kvHeadCount << "\n";
    out << "intermediate_size: " << config.intermediateSize << "\n";
    out << "rope_theta: " << formatFloat(config.ropeTheta) << "\n";
    out << "use_bias: " << (config.useBias ? "true" : "false") << "\n";
    out << "tie_embedding: " << (config.tieEmbedding ? "true" : "false") << "\n";
    out << "rms_norm_eps: " << formatFloat(config.rmsNormEps) << "\n";
    out << "learning_rate: " << formatFloat(config.learningRate) << "\n";
    if (config.weights.empty())
        out << "weights: null\n";
    else
        out << "weights: \"" << config.weights << "\"\n";
    return out.str();
}

void saveConfig(const std::string& pathOrDirectory, const Config& config) {
    if (pathOrDirectory.empty())
        throw std::invalid_argument("SentinelModel::saveConfig empty path");

    namespace fs = std::filesystem;
    fs::path path(pathOrDirectory);
    std::string body;
    if (pathLooksLikeDirectory(pathOrDirectory)) {
        fs::create_directories(path);
        path /= "model.json";
        body = serializeConfigJson(config);
    } else {
        if (path.has_parent_path() && !path.parent_path().empty())
            fs::create_directories(path.parent_path());
        const std::string ext = toLowerExt(path.string());
        if (ext == "yaml" || ext == "yml")
            body = serializeConfigYaml(config);
        else
            body = serializeConfigJson(config);
    }

    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("SentinelModel::saveConfig cannot write " + path.string());
    out << body;
}

void runConfigParseSmokeDemo() {
    const char* json = R"JSON(
{
  "format": "sentinel-model",
  "vocab_size": 32000,
  "embedding_dim": 768,
  "max_position": 512,
  "block_count": 12,
  "head_count": 12,
  "kv_head_count": 12,
  "intermediate_size": 0,
  "rope_theta": 10000,
  "use_bias": true,
  "tie_embedding": true,
  "rms_norm_eps": 1e-5,
  "learning_rate": 3e-4,
  "weights": null
}
)JSON";

    const Config parsed = parseConfigJson(json);
    if (parsed.vocabSize != 32000 || parsed.embeddingDim != 768 || parsed.maxPosition != 512
        || parsed.blockCount != 12 || parsed.headCount != 12 || parsed.kvHeadCount != 12
        || parsed.intermediateSize != 0 || parsed.useBias != true || parsed.tieEmbedding != true
        || !parsed.weights.empty())
        throw std::runtime_error("SentinelModel::runConfigParseSmokeDemo JSON field mismatch");
    if (!(std::fabs(parsed.ropeTheta - 10000.0f) < 1.0e-3f)
        || !(std::fabs(parsed.rmsNormEps - 1.0e-5f) < 1.0e-12f)
        || !(std::fabs(parsed.learningRate - 3.0e-4f) < 1.0e-9f))
        throw std::runtime_error("SentinelModel::runConfigParseSmokeDemo JSON float mismatch");

    const Config fromYaml = parseConfigYaml(serializeConfigYaml(parsed));
    if (fromYaml.vocabSize != parsed.vocabSize || fromYaml.embeddingDim != parsed.embeddingDim
        || fromYaml.blockCount != parsed.blockCount || fromYaml.headCount != parsed.headCount
        || fromYaml.kvHeadCount != parsed.kvHeadCount || fromYaml.useBias != parsed.useBias
        || fromYaml.tieEmbedding != parsed.tieEmbedding)
        throw std::runtime_error("SentinelModel::runConfigParseSmokeDemo YAML roundtrip mismatch");

    const Config again = parseConfigJson(serializeConfigJson(parsed));
    if (again.vocabSize != parsed.vocabSize || again.learningRate != parsed.learningRate)
        throw std::runtime_error("SentinelModel::runConfigParseSmokeDemo JSON roundtrip mismatch");

    if (!expectRejectJson("{\"format\":\"other\",\"vocab_size\":1,\"embedding_dim\":4,\"max_position\":8,\"block_count\":1,\"head_count\":1}", "format"))
        throw std::runtime_error("SentinelModel::runConfigParseSmokeDemo expected format reject");
    if (!expectRejectJson("{\"format\":\"sentinel-model\",\"vocab_size\":1,\"embedding_dim\":5,\"max_position\":8,\"block_count\":1,\"head_count\":2}", "divisible"))
        throw std::runtime_error("SentinelModel::runConfigParseSmokeDemo expected embed/head reject");

    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / "sentinel_model_config_smoke";
    fs::create_directories(dir);
    const fs::path jsonPath = dir / "model.json";
    const fs::path yamlPath = dir / "alt.yaml";
    saveConfig(jsonPath.string(), parsed);
    Config loaded = loadConfig(dir.string());
    if (loaded.vocabSize != parsed.vocabSize)
        throw std::runtime_error("SentinelModel::runConfigParseSmokeDemo loadConfig dir mismatch");
    Config withWeights = parsed;
    withWeights.weights = "weights.safetensors";
    saveConfig(yamlPath.string(), withWeights);
    Config loadedYaml = loadConfig(yamlPath.string());
    if (loadedYaml.weights != "weights.safetensors")
        throw std::runtime_error("SentinelModel::runConfigParseSmokeDemo YAML weights mismatch");
    const std::string resolved = resolveWeightsPath(yamlPath.string(), loadedYaml.weights);
    if (resolved.find("weights.safetensors") == std::string::npos)
        throw std::runtime_error("SentinelModel::runConfigParseSmokeDemo resolveWeightsPath failed");

    std::error_code ec;
    fs::remove_all(dir, ec);

    SmokeLog::result(
        "SentinelModel config",
        "json+yaml parse/serialize/load ok  vocab=%d embed=%d blocks=%d",
        parsed.vocabSize,
        parsed.embeddingDim,
        parsed.blockCount);
}

} // namespace SentinelModel
