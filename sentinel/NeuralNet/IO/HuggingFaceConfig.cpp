#include "HuggingFaceConfig.hpp"

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

namespace HuggingFace {
namespace {

void skipWs(const std::string& json, size_t& i) {
    while (i < json.size() && (json[i] == ' ' || json[i] == '\n' || json[i] == '\r' || json[i] == '\t'))
        ++i;
}

std::string parseString(const std::string& json, size_t& i) {
    skipWs(json, i);
    if (i >= json.size() || json[i] != '"')
        throw std::runtime_error("HuggingFace config: expected string");
    ++i;
    std::string out;
    while (i < json.size()) {
        const char ch = json[i++];
        if (ch == '"') return out;
        if (ch == '\\') {
            if (i >= json.size()) throw std::runtime_error("HuggingFace config: bad escape");
            const char esc = json[i++];
            switch (esc) {
            case '"': out.push_back('"'); break;
            case '\\': out.push_back('\\'); break;
            case '/': out.push_back('/'); break;
            case 'n': out.push_back('\n'); break;
            case 'r': out.push_back('\r'); break;
            case 't': out.push_back('\t'); break;
            case 'u': {
                if (i + 4 > json.size()) throw std::runtime_error("HuggingFace config: bad unicode escape");
                unsigned code = 0;
                for (int n = 0; n < 4; ++n) {
                    const char h = json[i++];
                    code <<= 4;
                    if (h >= '0' && h <= '9') code |= static_cast<unsigned>(h - '0');
                    else if (h >= 'a' && h <= 'f') code |= static_cast<unsigned>(h - 'a' + 10);
                    else if (h >= 'A' && h <= 'F') code |= static_cast<unsigned>(h - 'A' + 10);
                    else throw std::runtime_error("HuggingFace config: bad unicode escape");
                }
                if (code > 0x7F)
                    throw std::runtime_error("HuggingFace config: non-ASCII unicode not supported");
                out.push_back(static_cast<char>(code));
                break;
            }
            default: throw std::runtime_error("HuggingFace config: unsupported escape");
            }
        } else {
            out.push_back(ch);
        }
    }
    throw std::runtime_error("HuggingFace config: unterminated string");
}

void skipValue(const std::string& json, size_t& i);

void skipObject(const std::string& json, size_t& i) {
    skipWs(json, i);
    if (i >= json.size() || json[i] != '{')
        throw std::runtime_error("HuggingFace config: expected object");
    ++i;
    skipWs(json, i);
    if (i < json.size() && json[i] == '}') {
        ++i;
        return;
    }
    while (true) {
        (void)parseString(json, i);
        skipWs(json, i);
        if (i >= json.size() || json[i] != ':')
            throw std::runtime_error("HuggingFace config: expected ':' in object");
        ++i;
        skipValue(json, i);
        skipWs(json, i);
        if (i >= json.size()) throw std::runtime_error("HuggingFace config: truncated object");
        if (json[i] == '}') {
            ++i;
            return;
        }
        if (json[i] != ',') throw std::runtime_error("HuggingFace config: expected ',' in object");
        ++i;
    }
}

void skipArray(const std::string& json, size_t& i) {
    skipWs(json, i);
    if (i >= json.size() || json[i] != '[')
        throw std::runtime_error("HuggingFace config: expected array");
    ++i;
    skipWs(json, i);
    if (i < json.size() && json[i] == ']') {
        ++i;
        return;
    }
    while (true) {
        skipValue(json, i);
        skipWs(json, i);
        if (i >= json.size()) throw std::runtime_error("HuggingFace config: truncated array");
        if (json[i] == ']') {
            ++i;
            return;
        }
        if (json[i] != ',') throw std::runtime_error("HuggingFace config: expected ',' in array");
        ++i;
    }
}

void skipValue(const std::string& json, size_t& i) {
    skipWs(json, i);
    if (i >= json.size()) throw std::runtime_error("HuggingFace config: truncated value");
    const char ch = json[i];
    if (ch == '"') {
        (void)parseString(json, i);
        return;
    }
    if (ch == '{') {
        skipObject(json, i);
        return;
    }
    if (ch == '[') {
        skipArray(json, i);
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
    throw std::runtime_error("HuggingFace config: unexpected value");
}

bool tryParseNull(const std::string& json, size_t& i) {
    skipWs(json, i);
    if (i + 4 <= json.size()
        && json[i] == 'n' && json[i + 1] == 'u' && json[i + 2] == 'l' && json[i + 3] == 'l') {
        i += 4;
        return true;
    }
    return false;
}

bool parseBool(const std::string& json, size_t& i) {
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
    throw std::runtime_error("HuggingFace config: expected boolean");
}

double parseNumber(const std::string& json, size_t& i) {
    skipWs(json, i);
    if (i >= json.size()) throw std::runtime_error("HuggingFace config: expected number");
    const size_t begin = i;
    if (json[i] == '-') ++i;
    if (i >= json.size() || (json[i] < '0' || json[i] > '9'))
        throw std::runtime_error("HuggingFace config: expected number");
    while (i < json.size() && json[i] >= '0' && json[i] <= '9') ++i;
    if (i < json.size() && json[i] == '.') {
        ++i;
        while (i < json.size() && json[i] >= '0' && json[i] <= '9') ++i;
    }
    if (i < json.size() && (json[i] == 'e' || json[i] == 'E')) {
        ++i;
        if (i < json.size() && (json[i] == '+' || json[i] == '-')) ++i;
        if (i >= json.size() || json[i] < '0' || json[i] > '9')
            throw std::runtime_error("HuggingFace config: bad exponent");
        while (i < json.size() && json[i] >= '0' && json[i] <= '9') ++i;
    }
    return std::stod(json.substr(begin, i - begin));
}

std::vector<std::string> parseStringArray(const std::string& json, size_t& i) {
    skipWs(json, i);
    if (i >= json.size() || json[i] != '[')
        throw std::runtime_error("HuggingFace config: expected string array");
    ++i;
    std::vector<std::string> values;
    skipWs(json, i);
    if (i < json.size() && json[i] == ']') {
        ++i;
        return values;
    }
    while (true) {
        values.push_back(parseString(json, i));
        skipWs(json, i);
        if (i >= json.size()) throw std::runtime_error("HuggingFace config: truncated array");
        if (json[i] == ']') {
            ++i;
            break;
        }
        if (json[i] != ',') throw std::runtime_error("HuggingFace config: expected ',' in array");
        ++i;
    }
    return values;
}

struct RawFields {
    bool hasModelType = false;
    bool hasVocabSize = false;
    bool hasHiddenSize = false;
    bool hasIntermediateSize = false;
    bool hasNumHiddenLayers = false;
    bool hasNumAttentionHeads = false;
    bool hasNumKeyValueHeads = false;
    bool hasMaxPositionEmbeddings = false;
    bool hasRmsNormEps = false;
    bool hasRopeTheta = false;
    bool hasTieWordEmbeddings = false;
    bool hasAttentionBias = false;
    bool hasMlpBias = false;
    bool hasArchitectures = false;
    bool hasSlidingWindow = false;
    bool hasQuantizationConfig = false;
    bool hasNumExperts = false;
    bool hasNumLocalExperts = false;

    std::string modelType;
    std::string architecture;
    int vocabSize = 0;
    int hiddenSize = 0;
    int intermediateSize = 0;
    int numHiddenLayers = 0;
    int numAttentionHeads = 0;
    int numKeyValueHeads = 0;
    int maxPositionEmbeddings = 0;
    float rmsNormEps = 1.0e-5f;
    float ropeTheta = 10000.0f;
    bool tieWordEmbeddings = false;
    bool attentionBias = false;
    bool mlpBias = false;
    double slidingWindow = 0.0;
    int numExperts = 0;
    int numLocalExperts = 0;
};

void ingestTopLevel(const std::string& json, RawFields& raw) {
    size_t i = 0;
    skipWs(json, i);
    if (i >= json.size() || json[i] != '{')
        throw std::runtime_error("HuggingFace config: expected top-level object");
    ++i;
    skipWs(json, i);
    if (i < json.size() && json[i] == '}') return;

    while (true) {
        const std::string key = parseString(json, i);
        skipWs(json, i);
        if (i >= json.size() || json[i] != ':')
            throw std::runtime_error("HuggingFace config: expected ':' after key");
        ++i;
        skipWs(json, i);

        if (key == "model_type") {
            raw.modelType = parseString(json, i);
            raw.hasModelType = true;
        } else if (key == "architectures") {
            const auto arches = parseStringArray(json, i);
            raw.hasArchitectures = true;
            if (!arches.empty()) raw.architecture = arches.front();
        } else if (key == "vocab_size") {
            raw.vocabSize = static_cast<int>(parseNumber(json, i));
            raw.hasVocabSize = true;
        } else if (key == "hidden_size") {
            raw.hiddenSize = static_cast<int>(parseNumber(json, i));
            raw.hasHiddenSize = true;
        } else if (key == "intermediate_size") {
            raw.intermediateSize = static_cast<int>(parseNumber(json, i));
            raw.hasIntermediateSize = true;
        } else if (key == "num_hidden_layers") {
            raw.numHiddenLayers = static_cast<int>(parseNumber(json, i));
            raw.hasNumHiddenLayers = true;
        } else if (key == "num_attention_heads") {
            raw.numAttentionHeads = static_cast<int>(parseNumber(json, i));
            raw.hasNumAttentionHeads = true;
        } else if (key == "num_key_value_heads") {
            raw.numKeyValueHeads = static_cast<int>(parseNumber(json, i));
            raw.hasNumKeyValueHeads = true;
        } else if (key == "max_position_embeddings") {
            raw.maxPositionEmbeddings = static_cast<int>(parseNumber(json, i));
            raw.hasMaxPositionEmbeddings = true;
        } else if (key == "rms_norm_eps") {
            raw.rmsNormEps = static_cast<float>(parseNumber(json, i));
            raw.hasRmsNormEps = true;
        } else if (key == "rope_theta") {
            raw.ropeTheta = static_cast<float>(parseNumber(json, i));
            raw.hasRopeTheta = true;
        } else if (key == "tie_word_embeddings") {
            raw.tieWordEmbeddings = parseBool(json, i);
            raw.hasTieWordEmbeddings = true;
        } else if (key == "attention_bias") {
            raw.attentionBias = parseBool(json, i);
            raw.hasAttentionBias = true;
        } else if (key == "mlp_bias") {
            raw.mlpBias = parseBool(json, i);
            raw.hasMlpBias = true;
        } else if (key == "sliding_window") {
            raw.hasSlidingWindow = true;
            if (!tryParseNull(json, i))
                raw.slidingWindow = parseNumber(json, i);
            else
                raw.hasSlidingWindow = false; // null → no sliding window
        } else if (key == "quantization_config") {
            raw.hasQuantizationConfig = true;
            if (!tryParseNull(json, i))
                skipValue(json, i);
            else
                raw.hasQuantizationConfig = false;
        } else if (key == "num_experts") {
            raw.numExperts = static_cast<int>(parseNumber(json, i));
            raw.hasNumExperts = true;
        } else if (key == "num_local_experts") {
            raw.numLocalExperts = static_cast<int>(parseNumber(json, i));
            raw.hasNumLocalExperts = true;
        } else {
            skipValue(json, i);
        }

        skipWs(json, i);
        if (i >= json.size()) throw std::runtime_error("HuggingFace config: truncated object");
        if (json[i] == '}') {
            ++i;
            break;
        }
        if (json[i] != ',') throw std::runtime_error("HuggingFace config: expected ',' between keys");
        ++i;
    }
}

[[noreturn]] void unsupported(const std::string& detail) {
    throw std::runtime_error("HuggingFace config unsupported: " + detail);
}

void requirePositive(const char* name, int value) {
    if (value <= 0)
        throw std::runtime_error(std::string("HuggingFace config: ") + name + " must be > 0");
}

Config buildValidated(const RawFields& raw) {
    if (!raw.hasModelType || raw.modelType.empty())
        throw std::runtime_error("HuggingFace config: missing model_type");
    if (!isSupportedModelType(raw.modelType))
        unsupported("model_type='" + raw.modelType + "' (allowlist: llama, mistral, qwen2)");

    if (raw.hasNumExperts && raw.numExperts > 0)
        unsupported("MoE num_experts=" + std::to_string(raw.numExperts));
    if (raw.hasNumLocalExperts && raw.numLocalExperts > 0)
        unsupported("MoE num_local_experts=" + std::to_string(raw.numLocalExperts));
    if (raw.hasSlidingWindow)
        unsupported("sliding_window (not supported yet)");
    if (raw.hasQuantizationConfig)
        unsupported("quantization_config (GPTQ/AWQ/… not supported)");

    if (!raw.hasVocabSize) throw std::runtime_error("HuggingFace config: missing vocab_size");
    if (!raw.hasHiddenSize) throw std::runtime_error("HuggingFace config: missing hidden_size");
    if (!raw.hasIntermediateSize) throw std::runtime_error("HuggingFace config: missing intermediate_size");
    if (!raw.hasNumHiddenLayers) throw std::runtime_error("HuggingFace config: missing num_hidden_layers");
    if (!raw.hasNumAttentionHeads) throw std::runtime_error("HuggingFace config: missing num_attention_heads");
    if (!raw.hasMaxPositionEmbeddings)
        throw std::runtime_error("HuggingFace config: missing max_position_embeddings");

    requirePositive("vocab_size", raw.vocabSize);
    requirePositive("hidden_size", raw.hiddenSize);
    requirePositive("intermediate_size", raw.intermediateSize);
    requirePositive("num_hidden_layers", raw.numHiddenLayers);
    requirePositive("num_attention_heads", raw.numAttentionHeads);
    requirePositive("max_position_embeddings", raw.maxPositionEmbeddings);

    const int kvHeads = raw.hasNumKeyValueHeads ? raw.numKeyValueHeads : raw.numAttentionHeads;
    requirePositive("num_key_value_heads", kvHeads);
    if (raw.numAttentionHeads % kvHeads != 0)
        throw std::runtime_error("HuggingFace config: num_attention_heads must be divisible by num_key_value_heads");
    if (raw.hiddenSize % raw.numAttentionHeads != 0)
        throw std::runtime_error("HuggingFace config: hidden_size must be divisible by num_attention_heads");

    if (raw.hasRmsNormEps && !(raw.rmsNormEps > 0.0f))
        throw std::runtime_error("HuggingFace config: rms_norm_eps must be > 0");
    if (raw.hasRopeTheta && !(raw.ropeTheta > 0.0f))
        throw std::runtime_error("HuggingFace config: rope_theta must be > 0");

    if (raw.hasArchitectures && !raw.architecture.empty()) {
        const std::string& arch = raw.architecture;
        const bool looksCausal =
            arch.find("ForCausalLM") != std::string::npos
            || arch.find("CausalLM") != std::string::npos;
        if (!looksCausal)
            unsupported("architecture='" + arch + "' (expected *ForCausalLM)");
        if (arch.find("Moe") != std::string::npos || arch.find("MoE") != std::string::npos)
            unsupported("architecture='" + arch + "' (MoE)");
    }

    Config cfg;
    cfg.modelType = raw.modelType;
    cfg.architecture = raw.architecture;
    cfg.vocabSize = raw.vocabSize;
    cfg.hiddenSize = raw.hiddenSize;
    cfg.intermediateSize = raw.intermediateSize;
    cfg.numHiddenLayers = raw.numHiddenLayers;
    cfg.numAttentionHeads = raw.numAttentionHeads;
    cfg.numKeyValueHeads = kvHeads;
    cfg.maxPositionEmbeddings = raw.maxPositionEmbeddings;
    cfg.rmsNormEps = raw.hasRmsNormEps ? raw.rmsNormEps : 1.0e-5f;
    cfg.ropeTheta = raw.hasRopeTheta ? raw.ropeTheta : 10000.0f;
    cfg.tieWordEmbeddings = raw.hasTieWordEmbeddings ? raw.tieWordEmbeddings : false;
    const bool attentionBias = raw.hasAttentionBias ? raw.attentionBias : false;
    const bool mlpBias = raw.hasMlpBias ? raw.mlpBias : false;
    cfg.useBias = attentionBias || mlpBias;
    return cfg;
}

std::string readEntireFile(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("HuggingFace config: cannot open " + path);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

bool pathLooksLikeDirectory(const std::string& path) {
    if (path.empty()) return false;
    const char last = path.back();
    if (last == '/' || last == '\\') return true;
    // Prefer config.json sibling when the path has no .json suffix.
    const auto slash = path.find_last_of("/\\");
    const std::string leaf = (slash == std::string::npos) ? path : path.substr(slash + 1);
    const auto dot = leaf.find_last_of('.');
    if (dot == std::string::npos) return true;
    std::string ext = leaf.substr(dot + 1);
    for (char& c : ext) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return ext != "json";
}

void writeTempConfig(const std::string& path, const std::string& body) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("HuggingFace config smoke: cannot write " + path);
    out << body;
}

bool expectReject(const std::string& json, const char* needle) {
    try {
        (void)parseConfigJson(json);
        return false;
    } catch (const std::exception& ex) {
        const std::string msg = ex.what();
        return msg.find(needle) != std::string::npos;
    }
}

} // namespace

bool isSupportedModelType(const std::string& modelType) {
    return modelType == "llama" || modelType == "mistral" || modelType == "qwen2";
}

std::string defaultArchitectureName(const std::string& modelType) {
    if (modelType == "llama") return "LlamaForCausalLM";
    if (modelType == "mistral") return "MistralForCausalLM";
    if (modelType == "qwen2") return "Qwen2ForCausalLM";
    throw std::invalid_argument(
        "HuggingFace::defaultArchitectureName unsupported model_type='" + modelType + "'");
}

Config parseConfigJson(const std::string& json) {
    RawFields raw;
    ingestTopLevel(json, raw);
    return buildValidated(raw);
}

Config loadConfig(const std::string& pathOrDirectory) {
    if (pathOrDirectory.empty())
        throw std::invalid_argument("HuggingFace::loadConfig empty path");

    std::string path = pathOrDirectory;
    if (pathLooksLikeDirectory(path)) {
        if (!path.empty() && path.back() != '/' && path.back() != '\\')
            path.push_back('/');
        path += "config.json";
    }
    return parseConfigJson(readEntireFile(path));
}

std::string serializeConfigJson(const Config& config) {
    if (!isSupportedModelType(config.modelType))
        throw std::invalid_argument(
            "HuggingFace::serializeConfigJson unsupported model_type='" + config.modelType + "'");
    if (config.vocabSize <= 0 || config.hiddenSize <= 0 || config.intermediateSize <= 0
        || config.numHiddenLayers <= 0 || config.numAttentionHeads <= 0
        || config.numKeyValueHeads <= 0 || config.maxPositionEmbeddings <= 0)
        throw std::invalid_argument("HuggingFace::serializeConfigJson incomplete config");

    const std::string architecture = config.architecture.empty()
        ? defaultArchitectureName(config.modelType)
        : config.architecture;

    auto formatFloat = [](float value) -> std::string {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%.8g", static_cast<double>(value));
        return std::string(buf);
    };

    std::ostringstream out;
    out << "{\n";
    out << "  \"architectures\": [\"" << architecture << "\"],\n";
    out << "  \"model_type\": \"" << config.modelType << "\",\n";
    out << "  \"vocab_size\": " << config.vocabSize << ",\n";
    out << "  \"hidden_size\": " << config.hiddenSize << ",\n";
    out << "  \"intermediate_size\": " << config.intermediateSize << ",\n";
    out << "  \"num_hidden_layers\": " << config.numHiddenLayers << ",\n";
    out << "  \"num_attention_heads\": " << config.numAttentionHeads << ",\n";
    out << "  \"num_key_value_heads\": " << config.numKeyValueHeads << ",\n";
    out << "  \"max_position_embeddings\": " << config.maxPositionEmbeddings << ",\n";
    out << "  \"rms_norm_eps\": " << formatFloat(config.rmsNormEps) << ",\n";
    out << "  \"rope_theta\": " << formatFloat(config.ropeTheta) << ",\n";
    out << "  \"tie_word_embeddings\": " << (config.tieWordEmbeddings ? "true" : "false") << ",\n";
    out << "  \"attention_bias\": " << (config.useBias ? "true" : "false") << ",\n";
    out << "  \"mlp_bias\": " << (config.useBias ? "true" : "false") << ",\n";
    out << "  \"torch_dtype\": \"float32\",\n";
    out << "  \"transformers_version\": \"4.40.0\"\n";
    out << "}\n";
    return out.str();
}

void saveConfig(const std::string& modelDirectory, const Config& config) {
    if (modelDirectory.empty())
        throw std::invalid_argument("HuggingFace::saveConfig empty modelDirectory");

    namespace fs = std::filesystem;
    const fs::path root(modelDirectory);
    fs::create_directories(root);
    const fs::path path = root / "config.json";
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("HuggingFace::saveConfig cannot write " + path.string());
    out << serializeConfigJson(config);
    if (!out) throw std::runtime_error("HuggingFace::saveConfig write failed for " + path.string());
}

void runConfigParseSmokeDemo() {
    const std::string llamaJson = R"json({
  "architectures": ["LlamaForCausalLM"],
  "model_type": "llama",
  "vocab_size": 128256,
  "hidden_size": 2048,
  "intermediate_size": 8192,
  "num_hidden_layers": 16,
  "num_attention_heads": 32,
  "num_key_value_heads": 8,
  "max_position_embeddings": 131072,
  "rms_norm_eps": 1e-5,
  "rope_theta": 500000.0,
  "tie_word_embeddings": true,
  "attention_bias": false,
  "mlp_bias": false,
  "rope_scaling": {"rope_type": "llama3", "factor": 32.0}
})json";

    const Config cfg = parseConfigJson(llamaJson);
    if (cfg.modelType != "llama" || cfg.architecture != "LlamaForCausalLM")
        throw std::runtime_error("HF config smoke: model_type/architecture mismatch");
    if (cfg.vocabSize != 128256 || cfg.hiddenSize != 2048 || cfg.intermediateSize != 8192)
        throw std::runtime_error("HF config smoke: size fields mismatch");
    if (cfg.numHiddenLayers != 16 || cfg.numAttentionHeads != 32 || cfg.numKeyValueHeads != 8)
        throw std::runtime_error("HF config smoke: head/layer fields mismatch");
    if (cfg.maxPositionEmbeddings != 131072)
        throw std::runtime_error("HF config smoke: max_position_embeddings mismatch");
    if (std::fabs(cfg.ropeTheta - 500000.0f) > 1.0f)
        throw std::runtime_error("HF config smoke: rope_theta mismatch");
    if (!cfg.tieWordEmbeddings || cfg.useBias)
        throw std::runtime_error("HF config smoke: tie/useBias mismatch");

    // Missing num_key_value_heads → MHA (= num_attention_heads)
    const Config mha = parseConfigJson(R"json({
  "model_type": "mistral",
  "vocab_size": 32000,
  "hidden_size": 64,
  "intermediate_size": 128,
  "num_hidden_layers": 2,
  "num_attention_heads": 4,
  "max_position_embeddings": 128
})json");
    if (mha.numKeyValueHeads != 4 || mha.useBias)
        throw std::runtime_error("HF config smoke: MHA default kv heads / useBias failed");

    if (!expectReject(
            R"json({"model_type":"gpt2","vocab_size":1,"hidden_size":4,"intermediate_size":8,"num_hidden_layers":1,"num_attention_heads":2,"max_position_embeddings":8})json",
            "model_type"))
        throw std::runtime_error("HF config smoke: should reject unknown model_type");

    if (!expectReject(
            R"json({"model_type":"mixtral","vocab_size":1,"hidden_size":4,"intermediate_size":8,"num_hidden_layers":1,"num_attention_heads":2,"max_position_embeddings":8,"num_local_experts":8})json",
            "model_type"))
        throw std::runtime_error("HF config smoke: should reject mixtral model_type");

    if (!expectReject(
            R"json({"model_type":"llama","vocab_size":1,"hidden_size":4,"intermediate_size":8,"num_hidden_layers":1,"num_attention_heads":2,"max_position_embeddings":8,"num_experts":8})json",
            "MoE"))
        throw std::runtime_error("HF config smoke: should reject MoE num_experts");

    if (!expectReject(
            R"json({"model_type":"mistral","vocab_size":1,"hidden_size":4,"intermediate_size":8,"num_hidden_layers":1,"num_attention_heads":2,"max_position_embeddings":8,"sliding_window":4096})json",
            "sliding_window"))
        throw std::runtime_error("HF config smoke: should reject sliding_window");

    if (!expectReject(
            R"json({"model_type":"llama","vocab_size":1,"hidden_size":4,"intermediate_size":8,"num_hidden_layers":1,"num_attention_heads":2,"max_position_embeddings":8,"quantization_config":{"bits":4}})json",
            "quantization_config"))
        throw std::runtime_error("HF config smoke: should reject quantization_config");

    const std::string path = "hf_config_smoke.json";
    writeTempConfig(path, llamaJson);
    const Config fromFile = loadConfig(path);
    if (fromFile.numKeyValueHeads != 8 || std::fabs(fromFile.ropeTheta - 500000.0f) > 1.0f)
        throw std::runtime_error("HF config smoke: loadConfig file roundtrip failed");
    std::remove(path.c_str());

    const Config roundtrip = parseConfigJson(serializeConfigJson(cfg));
    if (roundtrip.modelType != cfg.modelType
        || roundtrip.vocabSize != cfg.vocabSize
        || roundtrip.numKeyValueHeads != cfg.numKeyValueHeads
        || std::fabs(roundtrip.ropeTheta - cfg.ropeTheta) > 1.0f
        || roundtrip.tieWordEmbeddings != cfg.tieWordEmbeddings
        || roundtrip.useBias != cfg.useBias)
        throw std::runtime_error("HF config smoke: serializeConfigJson roundtrip failed");
    if (defaultArchitectureName("qwen2") != "Qwen2ForCausalLM")
        throw std::runtime_error("HF config smoke: defaultArchitectureName qwen2 mismatch");

    SmokeLog::result(
        "HuggingFace config parse",
        "llama GQA ok  rejects=model_type/MoE/sliding/quant  file=ok  serialize=ok");
}

} // namespace HuggingFace
