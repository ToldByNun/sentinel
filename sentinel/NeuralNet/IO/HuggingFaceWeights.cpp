#include "HuggingFaceWeights.hpp"

#include "PytorchStateDict.hpp"
#include "../Math/Matrix.hpp"
#include "../Utils/SmokeLog.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace HuggingFace {
namespace {

namespace fs = std::filesystem;

void skipWs(const std::string& json, size_t& i) {
    while (i < json.size() && (json[i] == ' ' || json[i] == '\n' || json[i] == '\r' || json[i] == '\t'))
        ++i;
}

std::string parseString(const std::string& json, size_t& i) {
    skipWs(json, i);
    if (i >= json.size() || json[i] != '"')
        throw std::runtime_error("HuggingFace weight index: expected string");
    ++i;
    std::string out;
    while (i < json.size()) {
        const char ch = json[i++];
        if (ch == '"') return out;
        if (ch == '\\') {
            if (i >= json.size()) throw std::runtime_error("HuggingFace weight index: bad escape");
            const char esc = json[i++];
            switch (esc) {
            case '"': out.push_back('"'); break;
            case '\\': out.push_back('\\'); break;
            case '/': out.push_back('/'); break;
            case 'n': out.push_back('\n'); break;
            case 'r': out.push_back('\r'); break;
            case 't': out.push_back('\t'); break;
            case 'u': {
                if (i + 4 > json.size()) throw std::runtime_error("HuggingFace weight index: bad unicode escape");
                i += 4; // skip; shard filenames are ASCII
                break;
            }
            default: throw std::runtime_error("HuggingFace weight index: unsupported escape");
            }
        } else {
            out.push_back(ch);
        }
    }
    throw std::runtime_error("HuggingFace weight index: unterminated string");
}

void skipValue(const std::string& json, size_t& i);

void skipObject(const std::string& json, size_t& i) {
    skipWs(json, i);
    if (i >= json.size() || json[i] != '{')
        throw std::runtime_error("HuggingFace weight index: expected object");
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
            throw std::runtime_error("HuggingFace weight index: expected ':'");
        ++i;
        skipValue(json, i);
        skipWs(json, i);
        if (i >= json.size()) throw std::runtime_error("HuggingFace weight index: truncated object");
        if (json[i] == '}') {
            ++i;
            return;
        }
        if (json[i] != ',') throw std::runtime_error("HuggingFace weight index: expected ','");
        ++i;
    }
}

void skipArray(const std::string& json, size_t& i) {
    skipWs(json, i);
    if (i >= json.size() || json[i] != '[')
        throw std::runtime_error("HuggingFace weight index: expected array");
    ++i;
    skipWs(json, i);
    if (i < json.size() && json[i] == ']') {
        ++i;
        return;
    }
    while (true) {
        skipValue(json, i);
        skipWs(json, i);
        if (i >= json.size()) throw std::runtime_error("HuggingFace weight index: truncated array");
        if (json[i] == ']') {
            ++i;
            return;
        }
        if (json[i] != ',') throw std::runtime_error("HuggingFace weight index: expected ','");
        ++i;
    }
}

void skipValue(const std::string& json, size_t& i) {
    skipWs(json, i);
    if (i >= json.size()) throw std::runtime_error("HuggingFace weight index: truncated value");
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
    throw std::runtime_error("HuggingFace weight index: unexpected value");
}

std::map<std::string, std::string> parseStringStringObject(const std::string& json, size_t& i) {
    skipWs(json, i);
    if (i >= json.size() || json[i] != '{')
        throw std::runtime_error("HuggingFace weight index: expected weight_map object");
    ++i;
    std::map<std::string, std::string> out;
    skipWs(json, i);
    if (i < json.size() && json[i] == '}') {
        ++i;
        return out;
    }
    while (true) {
        const std::string key = parseString(json, i);
        skipWs(json, i);
        if (i >= json.size() || json[i] != ':')
            throw std::runtime_error("HuggingFace weight index: expected ':' in weight_map");
        ++i;
        const std::string value = parseString(json, i);
        out.emplace(key, value);
        skipWs(json, i);
        if (i >= json.size()) throw std::runtime_error("HuggingFace weight index: truncated weight_map");
        if (json[i] == '}') {
            ++i;
            break;
        }
        if (json[i] != ',') throw std::runtime_error("HuggingFace weight index: expected ',' in weight_map");
        ++i;
    }
    return out;
}

std::map<std::string, std::string> parseWeightMapFromIndexJson(const std::string& json) {
    size_t i = 0;
    skipWs(json, i);
    if (i >= json.size() || json[i] != '{')
        throw std::runtime_error("HuggingFace weight index: expected top-level object");
    ++i;
    skipWs(json, i);
    if (i < json.size() && json[i] == '}')
        throw std::runtime_error("HuggingFace weight index: empty index");

    std::map<std::string, std::string> weightMap;
    bool found = false;
    while (true) {
        const std::string key = parseString(json, i);
        skipWs(json, i);
        if (i >= json.size() || json[i] != ':')
            throw std::runtime_error("HuggingFace weight index: expected ':'");
        ++i;
        if (key == "weight_map") {
            weightMap = parseStringStringObject(json, i);
            found = true;
        } else {
            skipValue(json, i);
        }
        skipWs(json, i);
        if (i >= json.size()) throw std::runtime_error("HuggingFace weight index: truncated");
        if (json[i] == '}') {
            ++i;
            break;
        }
        if (json[i] != ',') throw std::runtime_error("HuggingFace weight index: expected ','");
        ++i;
    }
    if (!found || weightMap.empty())
        throw std::runtime_error("HuggingFace weight index: missing or empty weight_map");
    return weightMap;
}

std::string readEntireFile(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("HuggingFace weights: cannot open " + path);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

std::string joinPath(const std::string& directory, const std::string& relative) {
    return (fs::path(directory) / relative).string();
}

bool endsWithIgnoreCase(const std::string& text, const std::string& suffix) {
    if (text.size() < suffix.size()) return false;
    for (size_t i = 0; i < suffix.size(); ++i) {
        const char a = static_cast<char>(std::tolower(static_cast<unsigned char>(text[text.size() - suffix.size() + i])));
        const char b = static_cast<char>(std::tolower(static_cast<unsigned char>(suffix[i])));
        if (a != b) return false;
    }
    return true;
}

void addEntry(
    std::vector<WeightMapEntry>& out,
    std::string sentinel,
    std::string hf,
    size_t rows,
    size_t cols,
    bool optional = false,
    bool transpose = false) {
    WeightMapEntry entry;
    entry.sentinelName = std::move(sentinel);
    entry.hfName = std::move(hf);
    entry.expectedRows = rows;
    entry.expectedCols = cols;
    entry.optional = optional;
    entry.transpose = transpose;
    out.push_back(std::move(entry));
}

std::vector<WeightMapEntry> buildLlamaMistralLikeMap(const Config& config) {
    const size_t vocab = static_cast<size_t>(config.vocabSize);
    const size_t hidden = static_cast<size_t>(config.hiddenSize);
    const size_t intermediate = static_cast<size_t>(config.intermediateSize);
    const size_t headDim = hidden / static_cast<size_t>(config.numAttentionHeads);
    const size_t qRows = hidden;
    const size_t kvRows = static_cast<size_t>(config.numKeyValueHeads) * headDim;

    std::vector<WeightMapEntry> map;
    map.reserve(static_cast<size_t>(8 + config.numHiddenLayers * 12));

    addEntry(map, "token_embedding.weight", "model.embed_tokens.weight", vocab, hidden);

    for (int layer = 0; layer < config.numHiddenLayers; ++layer) {
        const std::string sp = "blocks." + std::to_string(layer) + ".";
        const std::string hp = "model.layers." + std::to_string(layer) + ".";
        addEntry(map, sp + "attn_norm.weight", hp + "input_layernorm.weight", hidden, 1);
        addEntry(map, sp + "attn.q_proj.weight", hp + "self_attn.q_proj.weight", qRows, hidden);
        addEntry(map, sp + "attn.k_proj.weight", hp + "self_attn.k_proj.weight", kvRows, hidden);
        addEntry(map, sp + "attn.v_proj.weight", hp + "self_attn.v_proj.weight", kvRows, hidden);
        addEntry(map, sp + "attn.o_proj.weight", hp + "self_attn.o_proj.weight", hidden, hidden);
        addEntry(map, sp + "ffn_norm.weight", hp + "post_attention_layernorm.weight", hidden, 1);
        addEntry(map, sp + "ffn.gate_proj.weight", hp + "mlp.gate_proj.weight", intermediate, hidden);
        addEntry(map, sp + "ffn.up_proj.weight", hp + "mlp.up_proj.weight", intermediate, hidden);
        addEntry(map, sp + "ffn.down_proj.weight", hp + "mlp.down_proj.weight", hidden, intermediate);
        if (config.useBias) {
            addEntry(map, sp + "ffn.gate_proj.bias", hp + "mlp.gate_proj.bias", intermediate, 1, true);
            addEntry(map, sp + "ffn.up_proj.bias", hp + "mlp.up_proj.bias", intermediate, 1, true);
            addEntry(map, sp + "ffn.down_proj.bias", hp + "mlp.down_proj.bias", hidden, 1, true);
        }
    }

    addEntry(map, "final_norm.weight", "model.norm.weight", hidden, 1);
    addEntry(
        map,
        "lm_head.weight",
        "lm_head.weight",
        vocab,
        hidden,
        /*optional=*/config.tieWordEmbeddings);
    if (config.useBias)
        addEntry(map, "lm_head.bias", "lm_head.bias", vocab, 1, true);

    return map;
}

Matrix asColumnVector(const Matrix& matrix, size_t rows) {
    if (matrix.rows == rows && matrix.cols == 1) return matrix;
    if (matrix.rows == 1 && matrix.cols == rows) {
        Matrix out(rows, 1, 0.0f);
        for (size_t i = 0; i < rows; ++i)
            out.data[i] = matrix.data[i];
        return out;
    }
    if (matrix.rows == rows && matrix.cols == rows && rows == 1) return matrix;
    // Flat 1D stored as [rows, 1] after SafeTensors load of shape [rows]
    if (matrix.rows == rows && matrix.cols == 1) return matrix;
    throw std::runtime_error("HuggingFace weights: expected vector length " + std::to_string(rows));
}

Matrix normalizeLoaded(const Matrix& loaded, const WeightMapEntry& entry) {
    Matrix matrix = entry.transpose ? Matrix::transpose(loaded) : loaded;
    if (entry.expectedCols == 1) {
        matrix = asColumnVector(matrix, entry.expectedRows);
    }
    if (matrix.rows != entry.expectedRows || matrix.cols != entry.expectedCols) {
        throw std::runtime_error(
            "HuggingFace weights: shape mismatch for " + entry.hfName
            + " → " + entry.sentinelName
            + " got [" + std::to_string(matrix.rows) + "," + std::to_string(matrix.cols)
            + "] expected [" + std::to_string(entry.expectedRows) + ","
            + std::to_string(entry.expectedCols) + "]");
    }
    return matrix;
}

void fillArchMetadata(SafeTensors::File& file, const Config& config) {
    file.metadata["format"] = "sentinel";
    file.metadata["arch"] = "causal_lm_rope_swiglu";
    file.metadata["vocab_size"] = std::to_string(config.vocabSize);
    file.metadata["embedding_dim"] = std::to_string(config.hiddenSize);
    file.metadata["max_position"] = std::to_string(config.maxPositionEmbeddings);
    file.metadata["block_count"] = std::to_string(config.numHiddenLayers);
    file.metadata["head_count"] = std::to_string(config.numAttentionHeads);
    file.metadata["kv_head_count"] = std::to_string(config.numKeyValueHeads);
    file.metadata["intermediate_size"] = std::to_string(config.intermediateSize);
    file.metadata["rope_theta"] = std::to_string(config.ropeTheta);
    file.metadata["use_bias"] = config.useBias ? "1" : "0";
    file.metadata["tie_embedding"] = config.tieWordEmbeddings ? "1" : "0";
    file.metadata["model_type"] = config.modelType;
}

Matrix filled(size_t rows, size_t cols, float value) {
    Matrix m(rows, cols, value);
    return m;
}

void writeText(const fs::path& path, const std::string& body) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("HuggingFace weight smoke: cannot write " + path.string());
    out << body;
}

} // namespace

std::vector<WeightMapEntry> buildWeightMap(const Config& config, WeightLayoutFamily family) {
    if (config.vocabSize <= 0 || config.hiddenSize <= 0 || config.numHiddenLayers <= 0
        || config.numAttentionHeads <= 0 || config.numKeyValueHeads <= 0
        || config.intermediateSize <= 0)
        throw std::invalid_argument("HuggingFace::buildWeightMap config incomplete");
    if (config.hiddenSize % config.numAttentionHeads != 0)
        throw std::invalid_argument("HuggingFace::buildWeightMap hidden_size not divisible by heads");
    if (config.numAttentionHeads % config.numKeyValueHeads != 0)
        throw std::invalid_argument("HuggingFace::buildWeightMap heads not divisible by kv heads");

    switch (family) {
    case WeightLayoutFamily::LlamaMistralLike:
        return buildLlamaMistralLikeMap(config);
    }
    throw std::runtime_error("HuggingFace::buildWeightMap unknown layout family");
}

WeightExportFormat parseWeightExportFormat(const std::string& name) {
    std::string lower = name;
    for (char& ch : lower)
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    if (lower == "safetensors" || lower == "safe" || lower == "st")
        return WeightExportFormat::SafeTensors;
    if (lower == "bin" || lower == "pytorch" || lower == "pt" || lower == "pytorch_bin")
        return WeightExportFormat::PytorchBin;
    if (lower == "both" || lower == "all")
        return WeightExportFormat::Both;
    throw std::invalid_argument(
        "HuggingFace::parseWeightExportFormat expected safetensors|bin|both, got '" + name + "'");
}

WeightShardIndex loadWeightShardIndex(const std::string& modelDirectory) {
    if (modelDirectory.empty())
        throw std::invalid_argument("HuggingFace::loadWeightShardIndex empty directory");

    const fs::path root(modelDirectory);
    if (!fs::is_directory(root))
        throw std::runtime_error("HuggingFace weights: not a directory: " + modelDirectory);

    WeightShardIndex index;
    index.directory = root.string();

    const fs::path safeIndexPath = root / "model.safetensors.index.json";
    if (fs::is_regular_file(safeIndexPath)) {
        index.kind = WeightFileKind::SafeTensors;
        index.weightToFile = parseWeightMapFromIndexJson(readEntireFile(safeIndexPath.string()));
        return index;
    }

    const fs::path singleSafe = root / "model.safetensors";
    if (fs::is_regular_file(singleSafe)) {
        index.kind = WeightFileKind::SafeTensors;
        index.singleFile = "model.safetensors";
        return index;
    }

    std::vector<std::string> safeShards;
    for (const auto& entry : fs::directory_iterator(root)) {
        if (!entry.is_regular_file()) continue;
        const std::string name = entry.path().filename().string();
        if (!endsWithIgnoreCase(name, ".safetensors")) continue;
        if (name.find("adapter") != std::string::npos) continue;
        safeShards.push_back(name);
    }
    std::sort(safeShards.begin(), safeShards.end());
    if (safeShards.size() == 1) {
        index.kind = WeightFileKind::SafeTensors;
        index.singleFile = safeShards.front();
        return index;
    }
    if (safeShards.size() > 1) {
        throw std::runtime_error(
            "HuggingFace weights: multiple .safetensors without model.safetensors.index.json under "
            + modelDirectory);
    }

    // PyTorch .bin fallback (modern zip torch.save).
    const fs::path binIndexPath = root / "pytorch_model.bin.index.json";
    if (fs::is_regular_file(binIndexPath)) {
        index.kind = WeightFileKind::PytorchBin;
        index.weightToFile = parseWeightMapFromIndexJson(readEntireFile(binIndexPath.string()));
        return index;
    }

    const fs::path pytorchBin = root / "pytorch_model.bin";
    if (fs::is_regular_file(pytorchBin)) {
        index.kind = WeightFileKind::PytorchBin;
        index.singleFile = "pytorch_model.bin";
        return index;
    }

    const fs::path modelBin = root / "model.bin";
    if (fs::is_regular_file(modelBin)) {
        index.kind = WeightFileKind::PytorchBin;
        index.singleFile = "model.bin";
        return index;
    }

    std::vector<std::string> binShards;
    for (const auto& entry : fs::directory_iterator(root)) {
        if (!entry.is_regular_file()) continue;
        const std::string name = entry.path().filename().string();
        if (!endsWithIgnoreCase(name, ".bin") && !endsWithIgnoreCase(name, ".pt")
            && !endsWithIgnoreCase(name, ".pth"))
            continue;
        if (name.find("adapter") != std::string::npos) continue;
        if (name.find("training_args") != std::string::npos) continue;
        if (name.find("optimizer") != std::string::npos) continue;
        binShards.push_back(name);
    }
    std::sort(binShards.begin(), binShards.end());
    if (binShards.size() == 1) {
        index.kind = WeightFileKind::PytorchBin;
        index.singleFile = binShards.front();
        return index;
    }
    if (binShards.size() > 1) {
        throw std::runtime_error(
            "HuggingFace weights: multiple .bin/.pt shards without pytorch_model.bin.index.json under "
            + modelDirectory);
    }

    throw std::runtime_error(
        "HuggingFace weights: no .safetensors or pytorch .bin under " + modelDirectory);
}

SafeTensors::File loadMappedWeights(
    const std::string& modelDirectory,
    const Config& config,
    WeightLayoutFamily family) {
    const std::vector<WeightMapEntry> map = buildWeightMap(config, family);
    const WeightShardIndex shardIndex = loadWeightShardIndex(modelDirectory);

    auto resolveShardFile = [&](const std::string& hfName) -> std::string {
        if (!shardIndex.singleFile.empty())
            return joinPath(shardIndex.directory, shardIndex.singleFile);

        const auto it = shardIndex.weightToFile.find(hfName);
        if (it == shardIndex.weightToFile.end())
            return {};
        return joinPath(shardIndex.directory, it->second);
    };

    std::unordered_map<std::string, SafeTensors::File> loadedShards;
    auto shardFor = [&](const std::string& path) -> const SafeTensors::File& {
        auto it = loadedShards.find(path);
        if (it != loadedShards.end()) return it->second;
        SafeTensors::File loaded;
        if (shardIndex.kind == WeightFileKind::SafeTensors)
            loaded = SafeTensors::load(path);
        else
            loaded = PytorchStateDict::load(path);
        auto [ins, _] = loadedShards.emplace(path, std::move(loaded));
        return ins->second;
    };

    SafeTensors::File out;
    fillArchMetadata(out, config);

    for (const WeightMapEntry& entry : map) {
        const std::string shardPath = resolveShardFile(entry.hfName);
        bool found = false;
        Matrix matrix;

        if (!shardPath.empty() && fs::is_regular_file(shardPath)) {
            const SafeTensors::File& shard = shardFor(shardPath);
            const auto it = shard.tensors.find(entry.hfName);
            if (it != shard.tensors.end()) {
                matrix = normalizeLoaded(it->second, entry);
                found = true;
            }
        }

        if (!found) {
            if (!entry.optional) {
                throw std::runtime_error(
                    "HuggingFace weights: missing required tensor " + entry.hfName
                    + " (→ " + entry.sentinelName + ")");
            }
            // Optional bias → zeros. Optional lm_head handled after loop via tie.
            if (entry.sentinelName == "lm_head.weight")
                continue;
            matrix = Matrix(entry.expectedRows, entry.expectedCols, 0.0f);
        }

        SafeTensors::putMatrix(out, entry.sentinelName, matrix);
    }

    if (out.tensors.find("lm_head.weight") == out.tensors.end()) {
        if (!config.tieWordEmbeddings)
            throw std::runtime_error("HuggingFace weights: missing lm_head.weight");
        const auto emb = out.tensors.find("token_embedding.weight");
        if (emb == out.tensors.end())
            throw std::runtime_error("HuggingFace weights: cannot tie lm_head without embed_tokens");
        // Leave lm_head absent; LanguageModel::loadSafeTensors uses tie metadata.
        out.metadata["tie_embedding"] = "1";
    } else if (config.tieWordEmbeddings) {
        // Tied configs sometimes still ship lm_head; keep tensor but mark tie for Sentinel.
        out.metadata["tie_embedding"] = "1";
    }

    return out;
}

SafeTensors::File remapSentinelWeightsToHf(
    const SafeTensors::File& sentinelWeights,
    const Config& config,
    WeightLayoutFamily family) {
    const std::vector<WeightMapEntry> map = buildWeightMap(config, family);
    SafeTensors::File out;
    // Transformers often stamps format=pt; keep export recognizable without Sentinel arch keys.
    out.metadata["format"] = "pt";

    for (const WeightMapEntry& entry : map) {
        const auto it = sentinelWeights.tensors.find(entry.sentinelName);
        if (it == sentinelWeights.tensors.end()) {
            if (entry.optional)
                continue;
            throw std::runtime_error(
                "HuggingFace export: missing required Sentinel tensor " + entry.sentinelName
                + " (→ " + entry.hfName + ")");
        }

        Matrix matrix = entry.transpose ? Matrix::transpose(it->second) : it->second;
        if (entry.expectedCols == 1)
            matrix = asColumnVector(matrix, entry.expectedRows);
        if (matrix.rows != entry.expectedRows || matrix.cols != entry.expectedCols) {
            throw std::runtime_error(
                "HuggingFace export: shape mismatch for " + entry.sentinelName
                + " → " + entry.hfName
                + " got [" + std::to_string(matrix.rows) + "," + std::to_string(matrix.cols)
                + "] expected [" + std::to_string(entry.expectedRows) + ","
                + std::to_string(entry.expectedCols) + "]");
        }
        SafeTensors::putMatrix(out, entry.hfName, matrix);
    }

    if (out.tensors.find("model.embed_tokens.weight") == out.tensors.end())
        throw std::runtime_error("HuggingFace export: missing model.embed_tokens.weight");
    if (!config.tieWordEmbeddings && out.tensors.find("lm_head.weight") == out.tensors.end())
        throw std::runtime_error("HuggingFace export: missing lm_head.weight (untied)");

    return out;
}

void saveDirectory(
    const std::string& modelDirectory,
    const Config& config,
    const SafeTensors::File& sentinelWeights,
    WeightLayoutFamily family,
    const std::string& tokenizerSourceDirectory,
    WeightExportFormat weightFormat) {
    if (modelDirectory.empty())
        throw std::invalid_argument("HuggingFace::saveDirectory empty modelDirectory");

    const fs::path root(modelDirectory);
    fs::create_directories(root);

    Config exportConfig = config;
    if (exportConfig.architecture.empty())
        exportConfig.architecture = defaultArchitectureName(exportConfig.modelType);
    saveConfig(root.string(), exportConfig);

    const SafeTensors::File hfWeights = remapSentinelWeightsToHf(sentinelWeights, exportConfig, family);
    if (weightFormat == WeightExportFormat::SafeTensors || weightFormat == WeightExportFormat::Both)
        SafeTensors::save((root / "model.safetensors").string(), hfWeights);
    if (weightFormat == WeightExportFormat::PytorchBin || weightFormat == WeightExportFormat::Both)
        PytorchStateDict::save((root / "pytorch_model.bin").string(), hfWeights);

    if (!tokenizerSourceDirectory.empty()) {
        const fs::path tokRoot(tokenizerSourceDirectory);
        if (!fs::is_directory(tokRoot))
            throw std::runtime_error(
                "HuggingFace::saveDirectory tokenizerSourceDirectory is not a directory: "
                + tokenizerSourceDirectory);

        static const char* const kTokenizerFiles[] = {
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
            "generation_config.json",
        };
        int copied = 0;
        for (const char* name : kTokenizerFiles) {
            const fs::path src = tokRoot / name;
            if (!fs::is_regular_file(src)) continue;
            const fs::path dst = root / name;
            fs::copy_file(src, dst, fs::copy_options::overwrite_existing);
            ++copied;
        }
        if (copied == 0)
            throw std::runtime_error(
                "HuggingFace::saveDirectory tokenizerSourceDirectory has no tokenizer files: "
                + tokenizerSourceDirectory);
    }
}

void runWeightMapSmokeDemo() {
    Config cfg;
    cfg.modelType = "llama";
    cfg.architecture = "LlamaForCausalLM";
    cfg.vocabSize = 32;
    cfg.hiddenSize = 16;
    cfg.intermediateSize = 32;
    cfg.numHiddenLayers = 2;
    cfg.numAttentionHeads = 4;
    cfg.numKeyValueHeads = 2;
    cfg.maxPositionEmbeddings = 64;
    cfg.rmsNormEps = 1.0e-5f;
    cfg.ropeTheta = 10000.0f;
    cfg.tieWordEmbeddings = true;
    cfg.useBias = false;

    const auto map = buildWeightMap(cfg);
    if (map.size() < 20)
        throw std::runtime_error("HF weight map smoke: unexpected entry count");
    bool sawQ = false;
    bool sawLmOptional = false;
    for (const auto& e : map) {
        if (e.hfName == "model.layers.0.self_attn.q_proj.weight") {
            sawQ = true;
            if (e.sentinelName != "blocks.0.attn.q_proj.weight" || e.expectedRows != 16 || e.expectedCols != 16)
                throw std::runtime_error("HF weight map smoke: q_proj mapping wrong");
            if (e.transpose)
                throw std::runtime_error("HF weight map smoke: Llama-like q_proj should not transpose");
        }
        if (e.sentinelName == "lm_head.weight") {
            sawLmOptional = e.optional;
            if (e.hfName != "lm_head.weight")
                throw std::runtime_error("HF weight map smoke: lm_head name wrong");
        }
        if (e.hfName.find("k_proj") != std::string::npos && e.expectedRows != 8)
            throw std::runtime_error("HF weight map smoke: GQA k rows expected 8");
    }
    if (!sawQ || !sawLmOptional)
        throw std::runtime_error("HF weight map smoke: missing q_proj or optional lm_head entry");

    const fs::path dir = fs::path("hf_weight_map_smoke_dir");
    fs::create_directories(dir);

    SafeTensors::File shard0;
    SafeTensors::putMatrix(shard0, "model.embed_tokens.weight", filled(32, 16, 0.11f));
    SafeTensors::putMatrix(shard0, "model.layers.0.input_layernorm.weight", filled(16, 1, 0.21f));
    SafeTensors::putMatrix(shard0, "model.layers.0.self_attn.q_proj.weight", filled(16, 16, 0.31f));
    SafeTensors::putMatrix(shard0, "model.layers.0.self_attn.k_proj.weight", filled(8, 16, 0.32f));
    SafeTensors::putMatrix(shard0, "model.layers.0.self_attn.v_proj.weight", filled(8, 16, 0.33f));
    SafeTensors::putMatrix(shard0, "model.layers.0.self_attn.o_proj.weight", filled(16, 16, 0.34f));
    SafeTensors::putMatrix(shard0, "model.layers.0.post_attention_layernorm.weight", filled(16, 1, 0.22f));
    SafeTensors::putMatrix(shard0, "model.layers.0.mlp.gate_proj.weight", filled(32, 16, 0.41f));
    SafeTensors::putMatrix(shard0, "model.layers.0.mlp.up_proj.weight", filled(32, 16, 0.42f));
    SafeTensors::putMatrix(shard0, "model.layers.0.mlp.down_proj.weight", filled(16, 32, 0.43f));
    SafeTensors::save((dir / "model-00001-of-00002.safetensors").string(), shard0);

    SafeTensors::File shard1;
    SafeTensors::putMatrix(shard1, "model.layers.1.input_layernorm.weight", filled(16, 1, 0.51f));
    SafeTensors::putMatrix(shard1, "model.layers.1.self_attn.q_proj.weight", filled(16, 16, 0.61f));
    SafeTensors::putMatrix(shard1, "model.layers.1.self_attn.k_proj.weight", filled(8, 16, 0.62f));
    SafeTensors::putMatrix(shard1, "model.layers.1.self_attn.v_proj.weight", filled(8, 16, 0.63f));
    SafeTensors::putMatrix(shard1, "model.layers.1.self_attn.o_proj.weight", filled(16, 16, 0.64f));
    SafeTensors::putMatrix(shard1, "model.layers.1.post_attention_layernorm.weight", filled(16, 1, 0.52f));
    SafeTensors::putMatrix(shard1, "model.layers.1.mlp.gate_proj.weight", filled(32, 16, 0.71f));
    SafeTensors::putMatrix(shard1, "model.layers.1.mlp.up_proj.weight", filled(32, 16, 0.72f));
    SafeTensors::putMatrix(shard1, "model.layers.1.mlp.down_proj.weight", filled(16, 32, 0.73f));
    SafeTensors::putMatrix(shard1, "model.norm.weight", filled(16, 1, 0.91f));
    // intentionally no lm_head.weight → tie
    SafeTensors::save((dir / "model-00002-of-00002.safetensors").string(), shard1);

    std::ostringstream indexJson;
    indexJson << "{\"metadata\":{\"total_size\":1},\"weight_map\":{";
    bool first = true;
    auto addMap = [&](const std::string& tensor, const std::string& file) {
        if (!first) indexJson << ',';
        first = false;
        indexJson << "\"" << tensor << "\":\"" << file << "\"";
    };
    for (const auto& [name, _] : shard0.tensors)
        addMap(name, "model-00001-of-00002.safetensors");
    for (const auto& [name, _] : shard1.tensors)
        addMap(name, "model-00002-of-00002.safetensors");
    indexJson << "}}";
    writeText(dir / "model.safetensors.index.json", indexJson.str());

    const WeightShardIndex idx = loadWeightShardIndex(dir.string());
    if (idx.weightToFile.size() != shard0.tensors.size() + shard1.tensors.size())
        throw std::runtime_error("HF weight map smoke: shard index size mismatch");

    const SafeTensors::File mapped = loadMappedWeights(dir.string(), cfg);
    if (mapped.tensors.count("token_embedding.weight") == 0
        || mapped.tensors.count("blocks.0.attn.k_proj.weight") == 0
        || mapped.tensors.count("blocks.1.ffn.down_proj.weight") == 0
        || mapped.tensors.count("final_norm.weight") == 0)
        throw std::runtime_error("HF weight map smoke: missing remapped tensors");
    if (mapped.tensors.count("lm_head.weight") != 0)
        throw std::runtime_error("HF weight map smoke: tied lm_head should stay absent");
    if (mapped.metadata.at("tie_embedding") != "1"
        || mapped.metadata.at("kv_head_count") != "2"
        || mapped.metadata.at("use_bias") != "0")
        throw std::runtime_error("HF weight map smoke: metadata mismatch");

    const Matrix& k0 = mapped.tensors.at("blocks.0.attn.k_proj.weight");
    if (k0.rows != 8 || k0.cols != 16 || std::fabs(k0.data[0] - 0.32f) > 1.0e-6f)
        throw std::runtime_error("HF weight map smoke: k_proj value/shape mismatch");
    const Matrix& emb = mapped.tensors.at("token_embedding.weight");
    if (std::fabs(emb.data[0] - 0.11f) > 1.0e-6f)
        throw std::runtime_error("HF weight map smoke: embed value mismatch");

    // Missing required tensor should fail
    bool rejected = false;
    try {
        fs::remove(dir / "model-00001-of-00002.safetensors");
        (void)loadMappedWeights(dir.string(), cfg);
    } catch (const std::exception&) {
        rejected = true;
    }
    if (!rejected)
        throw std::runtime_error("HF weight map smoke: should reject missing shard tensors");

    fs::remove_all(dir);

    // Export remap: Sentinel names → HF names (tied lm_head omitted).
    SafeTensors::File sentinel;
    fillArchMetadata(sentinel, cfg);
    SafeTensors::putMatrix(sentinel, "token_embedding.weight", filled(32, 16, 0.11f));
    SafeTensors::putMatrix(sentinel, "blocks.0.attn_norm.weight", filled(16, 1, 0.21f));
    SafeTensors::putMatrix(sentinel, "blocks.0.attn.q_proj.weight", filled(16, 16, 0.31f));
    SafeTensors::putMatrix(sentinel, "blocks.0.attn.k_proj.weight", filled(8, 16, 0.32f));
    SafeTensors::putMatrix(sentinel, "blocks.0.attn.v_proj.weight", filled(8, 16, 0.33f));
    SafeTensors::putMatrix(sentinel, "blocks.0.attn.o_proj.weight", filled(16, 16, 0.34f));
    SafeTensors::putMatrix(sentinel, "blocks.0.ffn_norm.weight", filled(16, 1, 0.22f));
    SafeTensors::putMatrix(sentinel, "blocks.0.ffn.gate_proj.weight", filled(32, 16, 0.41f));
    SafeTensors::putMatrix(sentinel, "blocks.0.ffn.up_proj.weight", filled(32, 16, 0.42f));
    SafeTensors::putMatrix(sentinel, "blocks.0.ffn.down_proj.weight", filled(16, 32, 0.43f));
    SafeTensors::putMatrix(sentinel, "blocks.1.attn_norm.weight", filled(16, 1, 0.51f));
    SafeTensors::putMatrix(sentinel, "blocks.1.attn.q_proj.weight", filled(16, 16, 0.61f));
    SafeTensors::putMatrix(sentinel, "blocks.1.attn.k_proj.weight", filled(8, 16, 0.62f));
    SafeTensors::putMatrix(sentinel, "blocks.1.attn.v_proj.weight", filled(8, 16, 0.63f));
    SafeTensors::putMatrix(sentinel, "blocks.1.attn.o_proj.weight", filled(16, 16, 0.64f));
    SafeTensors::putMatrix(sentinel, "blocks.1.ffn_norm.weight", filled(16, 1, 0.52f));
    SafeTensors::putMatrix(sentinel, "blocks.1.ffn.gate_proj.weight", filled(32, 16, 0.71f));
    SafeTensors::putMatrix(sentinel, "blocks.1.ffn.up_proj.weight", filled(32, 16, 0.72f));
    SafeTensors::putMatrix(sentinel, "blocks.1.ffn.down_proj.weight", filled(16, 32, 0.73f));
    SafeTensors::putMatrix(sentinel, "final_norm.weight", filled(16, 1, 0.91f));
    const SafeTensors::File exported = remapSentinelWeightsToHf(sentinel, cfg);
    if (exported.tensors.count("model.embed_tokens.weight") == 0
        || exported.tensors.count("model.layers.0.self_attn.k_proj.weight") == 0
        || exported.tensors.count("lm_head.weight") != 0)
        throw std::runtime_error("HF weight map smoke: export remap failed");
    if (std::fabs(exported.tensors.at("model.layers.0.self_attn.k_proj.weight").data[0] - 0.32f) > 1.0e-6f)
        throw std::runtime_error("HF weight map smoke: export value mismatch");

    SmokeLog::result(
        "HuggingFace weight map",
        "llama-like map=%zu  shards=2  GQA k=8  tie=ok  missing=reject  export=ok",
        map.size());
}

} // namespace HuggingFace
