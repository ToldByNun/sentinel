#include "Gguf.hpp"

#include "../Math/Matrix.hpp"
#include "../Utils/SmokeLog.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Gguf {
namespace {

constexpr std::uint32_t kMagic = 0x46554747u; // "GGUF" LE
constexpr std::uint32_t kVersion = 3u;
constexpr std::uint32_t kDefaultAlignment = 32u;

enum class GgmlType : std::uint32_t {
    F32 = 0,
    F16 = 1,
    BF16 = 30,
};

enum class MetaType : std::uint32_t {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
};

struct TensorInfo {
    std::string name;
    std::vector<std::uint64_t> dims; // ggml order: dims[0] = innermost
    GgmlType type = GgmlType::F32;
    std::uint64_t dataOffset = 0; // relative to tensor_data
};

struct ParsedFile {
    std::uint32_t alignment = kDefaultAlignment;
    std::unordered_map<std::string, std::string> stringMeta;
    std::unordered_map<std::string, std::uint64_t> uintMeta;
    std::unordered_map<std::string, float> floatMeta;
    std::unordered_map<std::string, bool> boolMeta;
    std::unordered_map<std::string, std::uint64_t> arrayLens; // for tokenizer.ggml.tokens etc.
    std::vector<TensorInfo> tensors;
    std::vector<std::uint8_t> tensorData;
};

std::uint64_t alignUp(std::uint64_t value, std::uint64_t alignment) {
    if (alignment == 0) return value;
    return value + (alignment - (value % alignment)) % alignment;
}

void readExact(std::ifstream& in, void* data, size_t bytes) {
    in.read(reinterpret_cast<char*>(data), static_cast<std::streamsize>(bytes));
    if (!in || static_cast<size_t>(in.gcount()) != bytes)
        throw std::runtime_error("Gguf: truncated read");
}

template <typename T>
T readPod(std::ifstream& in) {
    T value{};
    readExact(in, &value, sizeof(value));
    return value;
}

std::string readString(std::ifstream& in) {
    const std::uint64_t len = readPod<std::uint64_t>(in);
    if (len > (1ull << 31))
        throw std::runtime_error("Gguf: string too large");
    std::string out(static_cast<size_t>(len), '\0');
    if (len > 0)
        readExact(in, out.data(), static_cast<size_t>(len));
    return out;
}

void skipValue(std::ifstream& in, MetaType type);

void skipArray(std::ifstream& in) {
    const auto elemType = static_cast<MetaType>(readPod<std::uint32_t>(in));
    const std::uint64_t len = readPod<std::uint64_t>(in);
    for (std::uint64_t i = 0; i < len; ++i)
        skipValue(in, elemType);
}

void skipValue(std::ifstream& in, MetaType type) {
    switch (type) {
    case MetaType::Uint8:
    case MetaType::Int8:
    case MetaType::Bool:
        (void)readPod<std::uint8_t>(in);
        return;
    case MetaType::Uint16:
    case MetaType::Int16:
        (void)readPod<std::uint16_t>(in);
        return;
    case MetaType::Uint32:
    case MetaType::Int32:
    case MetaType::Float32:
        (void)readPod<std::uint32_t>(in);
        return;
    case MetaType::Uint64:
    case MetaType::Int64:
    case MetaType::Float64:
        (void)readPod<std::uint64_t>(in);
        return;
    case MetaType::String:
        (void)readString(in);
        return;
    case MetaType::Array:
        skipArray(in);
        return;
    }
    throw std::runtime_error("Gguf: unknown metadata type");
}

float floatFromBf16Bits(std::uint16_t bits) {
    const std::uint32_t f32Bits = static_cast<std::uint32_t>(bits) << 16;
    float value = 0.0f;
    std::memcpy(&value, &f32Bits, sizeof(value));
    return value;
}

float floatFromF16Bits(std::uint16_t bits) {
    const std::uint32_t sign = (static_cast<std::uint32_t>(bits) & 0x8000u) << 16;
    const std::uint32_t exp = (bits >> 10) & 0x1fu;
    const std::uint32_t mant = bits & 0x3ffu;
    std::uint32_t f32Bits = 0;
    if (exp == 0) {
        if (mant == 0) {
            f32Bits = sign;
        } else {
            std::uint32_t m = mant;
            std::uint32_t e = 127 - 15 + 1;
            while ((m & 0x400u) == 0u) {
                m <<= 1;
                --e;
            }
            m &= 0x3ffu;
            f32Bits = sign | (e << 23) | (m << 13);
        }
    } else if (exp == 31) {
        f32Bits = sign | 0x7f800000u | (mant << 13);
    } else {
        f32Bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
    float value = 0.0f;
    std::memcpy(&value, &f32Bits, sizeof(value));
    return value;
}

size_t typeElementBytes(GgmlType type) {
    switch (type) {
    case GgmlType::F32: return 4;
    case GgmlType::F16:
    case GgmlType::BF16: return 2;
    }
    return 0;
}

size_t tensorElementCount(const TensorInfo& info) {
    size_t count = 1;
    for (std::uint64_t dim : info.dims)
        count *= static_cast<size_t>(dim);
    return count;
}

Matrix matrixFromGgufTensor(const TensorInfo& info, const std::uint8_t* data, size_t dataBytes) {
    if (info.dims.empty() || info.dims.size() > 2)
        throw std::runtime_error("Gguf: unsupported rank for " + info.name);
    const size_t elems = tensorElementCount(info);
    const size_t elemBytes = typeElementBytes(info.type);
    if (elemBytes == 0)
        throw std::runtime_error(
            "Gguf: quantized/unsupported tensor type for " + info.name
            + " (only F32/F16/BF16)");
    if (dataBytes < elems * elemBytes)
        throw std::runtime_error("Gguf: tensor payload too small for " + info.name);

    // ggml dims[0]=innermost → Sentinel Matrix rows=dims[1], cols=dims[0] (2D)
    // or rows=dims[0], cols=1 (1D).
    size_t rows = 0;
    size_t cols = 0;
    if (info.dims.size() == 1) {
        rows = static_cast<size_t>(info.dims[0]);
        cols = 1;
    } else {
        cols = static_cast<size_t>(info.dims[0]);
        rows = static_cast<size_t>(info.dims[1]);
    }

    Matrix out(rows, cols, 0.0f);
    if (info.type == GgmlType::F32) {
        std::memcpy(out.data.data(), data, elems * sizeof(float));
        return out;
    }
    const auto* halfs = reinterpret_cast<const std::uint16_t*>(data);
    for (size_t i = 0; i < elems; ++i) {
        out.data[i] = (info.type == GgmlType::BF16)
            ? floatFromBf16Bits(halfs[i])
            : floatFromF16Bits(halfs[i]);
    }
    return out;
}

void storeMetaValue(ParsedFile& file, const std::string& key, MetaType type, std::ifstream& in) {
    switch (type) {
    case MetaType::Uint8:
        file.uintMeta[key] = readPod<std::uint8_t>(in);
        return;
    case MetaType::Int8:
        file.uintMeta[key] = static_cast<std::uint64_t>(readPod<std::int8_t>(in));
        return;
    case MetaType::Uint16:
        file.uintMeta[key] = readPod<std::uint16_t>(in);
        return;
    case MetaType::Int16:
        file.uintMeta[key] = static_cast<std::uint64_t>(readPod<std::int16_t>(in));
        return;
    case MetaType::Uint32:
        file.uintMeta[key] = readPod<std::uint32_t>(in);
        return;
    case MetaType::Int32:
        file.uintMeta[key] = static_cast<std::uint64_t>(readPod<std::int32_t>(in));
        return;
    case MetaType::Uint64:
        file.uintMeta[key] = readPod<std::uint64_t>(in);
        return;
    case MetaType::Int64:
        file.uintMeta[key] = static_cast<std::uint64_t>(readPod<std::int64_t>(in));
        return;
    case MetaType::Float32:
        file.floatMeta[key] = readPod<float>(in);
        return;
    case MetaType::Float64:
        file.floatMeta[key] = static_cast<float>(readPod<double>(in));
        return;
    case MetaType::Bool: {
        const std::uint8_t value = readPod<std::uint8_t>(in);
        if (value > 1)
            throw std::runtime_error("Gguf: invalid bool metadata for " + key);
        file.boolMeta[key] = value != 0;
        return;
    }
    case MetaType::String:
        file.stringMeta[key] = readString(in);
        return;
    case MetaType::Array: {
        const auto elemType = static_cast<MetaType>(readPod<std::uint32_t>(in));
        const std::uint64_t len = readPod<std::uint64_t>(in);
        file.arrayLens[key] = len;
        for (std::uint64_t i = 0; i < len; ++i)
            skipValue(in, elemType);
        return;
    }
    }
    throw std::runtime_error("Gguf: unsupported metadata type for " + key);
}

ParsedFile parseFile(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw std::runtime_error("Gguf: cannot open " + path);

    const std::uint32_t magic = readPod<std::uint32_t>(in);
    if (magic != kMagic)
        throw std::runtime_error("Gguf: bad magic (expected GGUF)");
    const std::uint32_t version = readPod<std::uint32_t>(in);
    if (version != 2u && version != 3u)
        throw std::runtime_error("Gguf: unsupported version " + std::to_string(version));

    const std::uint64_t tensorCount = readPod<std::uint64_t>(in);
    const std::uint64_t kvCount = readPod<std::uint64_t>(in);
    if (tensorCount > (1ull << 24) || kvCount > (1ull << 24))
        throw std::runtime_error("Gguf: absurd tensor/kv counts");

    ParsedFile file;
    for (std::uint64_t i = 0; i < kvCount; ++i) {
        const std::string key = readString(in);
        const auto type = static_cast<MetaType>(readPod<std::uint32_t>(in));
        storeMetaValue(file, key, type, in);
    }

    auto alignIt = file.uintMeta.find("general.alignment");
    if (alignIt != file.uintMeta.end()) {
        if (alignIt->second == 0 || (alignIt->second % 8u) != 0u)
            throw std::runtime_error("Gguf: invalid general.alignment");
        file.alignment = static_cast<std::uint32_t>(alignIt->second);
    }

    file.tensors.reserve(static_cast<size_t>(tensorCount));
    for (std::uint64_t i = 0; i < tensorCount; ++i) {
        TensorInfo info;
        info.name = readString(in);
        const std::uint32_t nDims = readPod<std::uint32_t>(in);
        if (nDims == 0 || nDims > 4)
            throw std::runtime_error("Gguf: bad n_dimensions for " + info.name);
        info.dims.resize(nDims);
        for (std::uint32_t d = 0; d < nDims; ++d)
            info.dims[d] = readPod<std::uint64_t>(in);
        info.type = static_cast<GgmlType>(readPod<std::uint32_t>(in));
        info.dataOffset = readPod<std::uint64_t>(in);
        file.tensors.push_back(std::move(info));
    }

    const std::uint64_t headerEnd = static_cast<std::uint64_t>(in.tellg());
    const std::uint64_t dataBegin = alignUp(headerEnd, file.alignment);
    in.seekg(0, std::ios::end);
    const std::uint64_t fileSize = static_cast<std::uint64_t>(in.tellg());
    if (fileSize < dataBegin)
        throw std::runtime_error("Gguf: file shorter than tensor_data");
    const size_t payloadBytes = static_cast<size_t>(fileSize - dataBegin);
    file.tensorData.resize(payloadBytes);
    if (payloadBytes > 0) {
        in.seekg(static_cast<std::streamoff>(dataBegin), std::ios::beg);
        readExact(in, file.tensorData.data(), payloadBytes);
    }
    return file;
}

bool tryGetUint(const ParsedFile& file, const std::string& key, std::uint64_t& out) {
    auto it = file.uintMeta.find(key);
    if (it == file.uintMeta.end()) return false;
    out = it->second;
    return true;
}

bool tryGetFloat(const ParsedFile& file, const std::string& key, float& out) {
    auto it = file.floatMeta.find(key);
    if (it == file.floatMeta.end()) return false;
    out = it->second;
    return true;
}

const TensorInfo* findTensor(const ParsedFile& file, const std::string& name) {
    for (const TensorInfo& info : file.tensors) {
        if (info.name == name) return &info;
    }
    return nullptr;
}

void addEntry(
    std::vector<WeightMapEntry>& out,
    std::string sentinel,
    std::string gguf,
    size_t rows,
    size_t cols,
    bool optional = false) {
    WeightMapEntry entry;
    entry.sentinelName = std::move(sentinel);
    entry.ggufName = std::move(gguf);
    entry.expectedRows = rows;
    entry.expectedCols = cols;
    entry.optional = optional;
    out.push_back(std::move(entry));
}

std::vector<WeightMapEntry> buildLlamaLikeMap(const Config& config) {
    const size_t vocab = static_cast<size_t>(config.vocabSize);
    const size_t hidden = static_cast<size_t>(config.hiddenSize);
    const size_t intermediate = static_cast<size_t>(config.intermediateSize);
    const size_t headDim = hidden / static_cast<size_t>(config.numAttentionHeads);
    const size_t qRows = hidden;
    const size_t kvRows = static_cast<size_t>(config.numKeyValueHeads) * headDim;

    std::vector<WeightMapEntry> map;
    map.reserve(static_cast<size_t>(8 + config.numHiddenLayers * 12));

    addEntry(map, "token_embedding.weight", "token_embd.weight", vocab, hidden);

    for (int layer = 0; layer < config.numHiddenLayers; ++layer) {
        const std::string sp = "blocks." + std::to_string(layer) + ".";
        const std::string gp = "blk." + std::to_string(layer) + ".";
        addEntry(map, sp + "attn_norm.weight", gp + "attn_norm.weight", hidden, 1);
        addEntry(map, sp + "attn.q_proj.weight", gp + "attn_q.weight", qRows, hidden);
        addEntry(map, sp + "attn.k_proj.weight", gp + "attn_k.weight", kvRows, hidden);
        addEntry(map, sp + "attn.v_proj.weight", gp + "attn_v.weight", kvRows, hidden);
        addEntry(map, sp + "attn.o_proj.weight", gp + "attn_output.weight", hidden, hidden);
        addEntry(map, sp + "ffn_norm.weight", gp + "ffn_norm.weight", hidden, 1);
        addEntry(map, sp + "ffn.gate_proj.weight", gp + "ffn_gate.weight", intermediate, hidden);
        addEntry(map, sp + "ffn.up_proj.weight", gp + "ffn_up.weight", intermediate, hidden);
        addEntry(map, sp + "ffn.down_proj.weight", gp + "ffn_down.weight", hidden, intermediate);
        if (config.useBias) {
            addEntry(map, sp + "ffn.gate_proj.bias", gp + "ffn_gate.bias", intermediate, 1, true);
            addEntry(map, sp + "ffn.up_proj.bias", gp + "ffn_up.bias", intermediate, 1, true);
            addEntry(map, sp + "ffn.down_proj.bias", gp + "ffn_down.bias", hidden, 1, true);
        }
    }

    addEntry(map, "final_norm.weight", "output_norm.weight", hidden, 1);
    addEntry(
        map,
        "lm_head.weight",
        "output.weight",
        vocab,
        hidden,
        /*optional=*/config.tieWordEmbeddings);
    if (config.useBias)
        addEntry(map, "lm_head.bias", "output.bias", vocab, 1, true);

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
    throw std::runtime_error("Gguf: expected vector length " + std::to_string(rows));
}

Matrix normalizeLoaded(const Matrix& loaded, const WeightMapEntry& entry) {
    Matrix matrix = loaded;
    if (entry.expectedCols == 1)
        matrix = asColumnVector(matrix, entry.expectedRows);
    if (matrix.rows != entry.expectedRows || matrix.cols != entry.expectedCols) {
        throw std::runtime_error(
            "Gguf: shape mismatch for " + entry.ggufName + " → " + entry.sentinelName
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
    file.metadata["model_type"] = config.architecture;
}

void writeExact(std::ofstream& out, const void* data, size_t bytes) {
    out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(bytes));
    if (!out)
        throw std::runtime_error("Gguf: write failed");
}

template <typename T>
void writePod(std::ofstream& out, T value) {
    writeExact(out, &value, sizeof(value));
}

void writeString(std::ofstream& out, const std::string& value) {
    writePod<std::uint64_t>(out, static_cast<std::uint64_t>(value.size()));
    if (!value.empty())
        writeExact(out, value.data(), value.size());
}

void writeMetaString(std::ofstream& out, const std::string& key, const std::string& value) {
    writeString(out, key);
    writePod<std::uint32_t>(out, static_cast<std::uint32_t>(MetaType::String));
    writeString(out, value);
}

void writeMetaU32(std::ofstream& out, const std::string& key, std::uint32_t value) {
    writeString(out, key);
    writePod<std::uint32_t>(out, static_cast<std::uint32_t>(MetaType::Uint32));
    writePod<std::uint32_t>(out, value);
}

void writeMetaU64(std::ofstream& out, const std::string& key, std::uint64_t value) {
    writeString(out, key);
    writePod<std::uint32_t>(out, static_cast<std::uint32_t>(MetaType::Uint64));
    writePod<std::uint64_t>(out, value);
}

void writeMetaF32(std::ofstream& out, const std::string& key, float value) {
    writeString(out, key);
    writePod<std::uint32_t>(out, static_cast<std::uint32_t>(MetaType::Float32));
    writePod<float>(out, value);
}

void writePadding(std::ofstream& out, std::uint64_t alignment) {
    const std::uint64_t pos = static_cast<std::uint64_t>(out.tellp());
    const std::uint64_t padded = alignUp(pos, alignment);
    for (std::uint64_t i = pos; i < padded; ++i)
        writePod<std::uint8_t>(out, 0);
}

Config configFromParsed(const ParsedFile& file) {
    auto archIt = file.stringMeta.find("general.architecture");
    if (archIt == file.stringMeta.end() || archIt->second.empty())
        throw std::runtime_error("Gguf: missing general.architecture");
    if (!isSupportedArchitecture(archIt->second))
        throw std::runtime_error(
            "Gguf: unsupported architecture '" + archIt->second
            + "' (allowlist: llama, mistral, qwen2)");

    // Reject quantized packs early.
    for (const TensorInfo& info : file.tensors) {
        if (typeElementBytes(info.type) == 0)
            throw std::runtime_error(
                "Gguf: quantized tensor '" + info.name
                + "' is unsupported (export/import F32/F16/BF16 only)");
    }

    const std::string& arch = archIt->second;
    Config config;
    config.architecture = arch;

    auto requireU64 = [&](const std::string& key) -> std::uint64_t {
        std::uint64_t value = 0;
        if (!tryGetUint(file, key, value) || value == 0)
            throw std::runtime_error("Gguf: missing/invalid metadata " + key);
        return value;
    };

    config.maxPositionEmbeddings = static_cast<int>(requireU64(arch + ".context_length"));
    config.hiddenSize = static_cast<int>(requireU64(arch + ".embedding_length"));
    config.numHiddenLayers = static_cast<int>(requireU64(arch + ".block_count"));
    config.intermediateSize = static_cast<int>(requireU64(arch + ".feed_forward_length"));
    config.numAttentionHeads = static_cast<int>(requireU64(arch + ".attention.head_count"));

    std::uint64_t kvHeads = 0;
    if (!tryGetUint(file, arch + ".attention.head_count_kv", kvHeads) || kvHeads == 0)
        kvHeads = static_cast<std::uint64_t>(config.numAttentionHeads);
    config.numKeyValueHeads = static_cast<int>(kvHeads);

    float eps = 1.0e-5f;
    if (!tryGetFloat(file, arch + ".attention.layer_norm_rms_epsilon", eps))
        (void)tryGetFloat(file, arch + ".attention.layer_norm_epsilon", eps);
    config.rmsNormEps = eps;

    float rope = 10000.0f;
    (void)tryGetFloat(file, arch + ".rope.freq_base", rope);
    config.ropeTheta = rope;

    // Vocab: prefer tokenizer token count, else token_embd shape.
    std::uint64_t vocab = 0;
    auto tokIt = file.arrayLens.find("tokenizer.ggml.tokens");
    if (tokIt != file.arrayLens.end())
        vocab = tokIt->second;
    if (vocab == 0) {
        const TensorInfo* emb = findTensor(file, "token_embd.weight");
        if (emb == nullptr || emb->dims.size() != 2)
            throw std::runtime_error("Gguf: cannot infer vocab_size (missing token_embd.weight)");
        // ggml: dims=[hidden, vocab]
        vocab = emb->dims[1];
    }
    config.vocabSize = static_cast<int>(vocab);

    config.tieWordEmbeddings = findTensor(file, "output.weight") == nullptr;
    config.useBias = findTensor(file, "blk.0.ffn_gate.bias") != nullptr
        || findTensor(file, "output.bias") != nullptr;

    if (config.hiddenSize % config.numAttentionHeads != 0)
        throw std::runtime_error("Gguf: embedding_length not divisible by head_count");
    if (config.numAttentionHeads % config.numKeyValueHeads != 0)
        throw std::runtime_error("Gguf: head_count not divisible by head_count_kv");

    return config;
}

} // namespace

bool isSupportedArchitecture(const std::string& architecture) {
    return architecture == "llama" || architecture == "mistral" || architecture == "qwen2";
}

bool isGgufFile(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    std::uint32_t magic = 0;
    in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    return static_cast<bool>(in) && magic == kMagic;
}

Config loadConfig(const std::string& path) {
    if (path.empty())
        throw std::invalid_argument("Gguf::loadConfig empty path");
    return configFromParsed(parseFile(path));
}

std::vector<WeightMapEntry> buildWeightMap(const Config& config) {
    if (config.vocabSize <= 0 || config.hiddenSize <= 0 || config.numHiddenLayers <= 0
        || config.numAttentionHeads <= 0 || config.numKeyValueHeads <= 0
        || config.intermediateSize <= 0)
        throw std::invalid_argument("Gguf::buildWeightMap config incomplete");
    if (!isSupportedArchitecture(config.architecture))
        throw std::invalid_argument("Gguf::buildWeightMap unsupported architecture");
    if (config.hiddenSize % config.numAttentionHeads != 0)
        throw std::invalid_argument("Gguf::buildWeightMap hidden not divisible by heads");
    if (config.numAttentionHeads % config.numKeyValueHeads != 0)
        throw std::invalid_argument("Gguf::buildWeightMap heads not divisible by kv heads");
    return buildLlamaLikeMap(config);
}

SafeTensors::File loadMappedWeights(const std::string& path, const Config& config) {
    if (path.empty())
        throw std::invalid_argument("Gguf::loadMappedWeights empty path");
    const ParsedFile parsed = parseFile(path);
    const std::vector<WeightMapEntry> map = buildWeightMap(config);

    std::unordered_map<std::string, const TensorInfo*> byName;
    byName.reserve(parsed.tensors.size());
    for (const TensorInfo& info : parsed.tensors)
        byName.emplace(info.name, &info);

    SafeTensors::File out;
    fillArchMetadata(out, config);

    for (const WeightMapEntry& entry : map) {
        auto it = byName.find(entry.ggufName);
        if (it == byName.end()) {
            if (!entry.optional) {
                throw std::runtime_error("Gguf: missing required tensor " + entry.ggufName);
            }
            if (entry.sentinelName == "lm_head.weight" && config.tieWordEmbeddings)
                continue; // tied — LanguageModel uses token embedding
            out.tensors.emplace(
                entry.sentinelName,
                Matrix(entry.expectedRows, entry.expectedCols, 0.0f));
            continue;
        }

        const TensorInfo& info = *it->second;
        if (info.dataOffset > parsed.tensorData.size())
            throw std::runtime_error("Gguf: tensor offset out of range for " + info.name);
        const std::uint8_t* data = parsed.tensorData.data() + static_cast<size_t>(info.dataOffset);
        const size_t remain = parsed.tensorData.size() - static_cast<size_t>(info.dataOffset);
        Matrix loaded = matrixFromGgufTensor(info, data, remain);
        out.tensors.emplace(entry.sentinelName, normalizeLoaded(loaded, entry));
    }

    return out;
}

void save(
    const std::string& path,
    const Config& config,
    const SafeTensors::File& sentinelWeights) {
    if (path.empty())
        throw std::invalid_argument("Gguf::save empty path");
    if (!isSupportedArchitecture(config.architecture))
        throw std::invalid_argument("Gguf::save unsupported architecture");

    const std::vector<WeightMapEntry> map = buildWeightMap(config);

    struct OutTensor {
        std::string name;
        size_t rows = 0;
        size_t cols = 0;
        const float* data = nullptr;
        size_t offset = 0;
    };
    std::vector<OutTensor> tensors;
    tensors.reserve(map.size());
    size_t dataBytes = 0;
    for (const WeightMapEntry& entry : map) {
        auto it = sentinelWeights.tensors.find(entry.sentinelName);
        if (it == sentinelWeights.tensors.end()) {
            if (entry.optional) continue;
            throw std::runtime_error("Gguf::save missing Sentinel tensor " + entry.sentinelName);
        }
        const Matrix& matrix = it->second;
        if (matrix.rows != entry.expectedRows || matrix.cols != entry.expectedCols)
            throw std::runtime_error("Gguf::save shape mismatch for " + entry.sentinelName);
        OutTensor tensor;
        tensor.name = entry.ggufName;
        tensor.rows = matrix.rows;
        tensor.cols = matrix.cols;
        tensor.data = matrix.data.data();
        dataBytes = static_cast<size_t>(alignUp(static_cast<std::uint64_t>(dataBytes), kDefaultAlignment));
        tensor.offset = dataBytes;
        dataBytes += matrix.data.size() * sizeof(float);
        tensors.push_back(tensor);
    }

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out)
        throw std::runtime_error("Gguf::save cannot open " + path);

    const std::string& arch = config.architecture;
    const std::uint64_t kvCount = 12;

    writePod<std::uint32_t>(out, kMagic);
    writePod<std::uint32_t>(out, kVersion);
    writePod<std::uint64_t>(out, static_cast<std::uint64_t>(tensors.size()));
    writePod<std::uint64_t>(out, kvCount);

    writeMetaString(out, "general.architecture", arch);
    writeMetaString(out, "general.name", "sentinel-" + arch);
    writeMetaU32(out, "general.alignment", kDefaultAlignment);
    writeMetaU32(out, "general.file_type", 0); // ALL_F32
    writeMetaU64(out, arch + ".context_length", static_cast<std::uint64_t>(config.maxPositionEmbeddings));
    writeMetaU64(out, arch + ".embedding_length", static_cast<std::uint64_t>(config.hiddenSize));
    writeMetaU64(out, arch + ".block_count", static_cast<std::uint64_t>(config.numHiddenLayers));
    writeMetaU64(out, arch + ".feed_forward_length", static_cast<std::uint64_t>(config.intermediateSize));
    writeMetaU64(out, arch + ".attention.head_count", static_cast<std::uint64_t>(config.numAttentionHeads));
    writeMetaU64(out, arch + ".attention.head_count_kv", static_cast<std::uint64_t>(config.numKeyValueHeads));
    writeMetaF32(out, arch + ".attention.layer_norm_rms_epsilon", config.rmsNormEps);
    writeMetaF32(out, arch + ".rope.freq_base", config.ropeTheta);

    for (const OutTensor& tensor : tensors) {
        writeString(out, tensor.name);
        if (tensor.cols == 1) {
            writePod<std::uint32_t>(out, 1u);
            writePod<std::uint64_t>(out, static_cast<std::uint64_t>(tensor.rows));
        } else {
            writePod<std::uint32_t>(out, 2u);
            writePod<std::uint64_t>(out, static_cast<std::uint64_t>(tensor.cols)); // ggml innermost
            writePod<std::uint64_t>(out, static_cast<std::uint64_t>(tensor.rows));
        }
        writePod<std::uint32_t>(out, static_cast<std::uint32_t>(GgmlType::F32));
        writePod<std::uint64_t>(out, static_cast<std::uint64_t>(tensor.offset));
    }

    writePadding(out, kDefaultAlignment);
    std::uint64_t written = 0;
    for (size_t i = 0; i < tensors.size(); ++i) {
        const OutTensor& tensor = tensors[i];
        while (written < tensor.offset) {
            writePod<std::uint8_t>(out, 0);
            ++written;
        }
        const size_t nbytes = tensor.rows * tensor.cols * sizeof(float);
        writeExact(out, tensor.data, nbytes);
        written += nbytes;
        const std::uint64_t aligned = alignUp(written, kDefaultAlignment);
        while (written < aligned) {
            writePod<std::uint8_t>(out, 0);
            ++written;
        }
    }
}

void runConfigParseSmokeDemo() {
    // Build a tiny F32 GGUF via save, then parse + reject bad arches.
    Config config;
    config.architecture = "llama";
    config.vocabSize = 16;
    config.hiddenSize = 8;
    config.intermediateSize = 16;
    config.numHiddenLayers = 1;
    config.numAttentionHeads = 2;
    config.numKeyValueHeads = 1;
    config.maxPositionEmbeddings = 32;
    config.rmsNormEps = 1.0e-5f;
    config.ropeTheta = 10000.0f;
    config.tieWordEmbeddings = true;
    config.useBias = false;

    SafeTensors::File weights;
    fillArchMetadata(weights, config);
    const auto map = buildWeightMap(config);
    for (const WeightMapEntry& entry : map) {
        if (entry.optional) continue;
        weights.tensors.emplace(
            entry.sentinelName,
            Matrix(entry.expectedRows, entry.expectedCols, 0.125f));
    }

    const std::string path = "gguf_config_smoke.gguf";
    save(path, config, weights);

    if (!isGgufFile(path))
        throw std::runtime_error("Gguf config smoke: isGgufFile false");
    const Config loaded = loadConfig(path);
    if (loaded.architecture != "llama" || loaded.vocabSize != 16 || loaded.hiddenSize != 8
        || loaded.numKeyValueHeads != 1 || loaded.intermediateSize != 16
        || !loaded.tieWordEmbeddings || loaded.useBias)
        throw std::runtime_error("Gguf config smoke: metadata mismatch");

    bool rejected = false;
    try {
        Config bad = config;
        bad.architecture = "gpt2";
        save("gguf_bad_arch_smoke.gguf", bad, weights);
    } catch (const std::exception&) {
        rejected = true;
    }
    if (!rejected)
        throw std::runtime_error("Gguf config smoke: should reject unsupported architecture on save");

    std::remove(path.c_str());
    SmokeLog::result("Gguf loadConfig", "llama=ok  tie=1  kv=1  reject=arch");
}

void runWeightMapSmokeDemo() {
    Config config;
    config.architecture = "llama";
    config.vocabSize = 32;
    config.hiddenSize = 16;
    config.intermediateSize = 32;
    config.numHiddenLayers = 2;
    config.numAttentionHeads = 4;
    config.numKeyValueHeads = 2;
    config.maxPositionEmbeddings = 64;
    config.rmsNormEps = 1.0e-5f;
    config.ropeTheta = 5000.0f;
    config.tieWordEmbeddings = true;
    config.useBias = false;

    SafeTensors::File weights;
    fillArchMetadata(weights, config);
    const auto map = buildWeightMap(config);
    for (const WeightMapEntry& entry : map) {
        if (entry.optional) continue;
        Matrix m(entry.expectedRows, entry.expectedCols, 0.0f);
        m.data[0] = 0.5f;
        weights.tensors.emplace(entry.sentinelName, std::move(m));
    }
    weights.tensors["token_embedding.weight"].data[0] = 0.125f;
    weights.tensors["blocks.0.attn.k_proj.weight"].data[0] = 0.25f;
    weights.tensors["final_norm.weight"].data[0] = 0.375f;

    const std::string path = "gguf_weight_smoke.gguf";
    save(path, config, weights);

    const SafeTensors::File mapped = loadMappedWeights(path, loadConfig(path));
    if (mapped.tensors.count("token_embedding.weight") == 0
        || mapped.tensors.count("blocks.0.attn.k_proj.weight") == 0
        || mapped.tensors.count("final_norm.weight") == 0)
        throw std::runtime_error("Gguf weight smoke: sentinel tensors missing");
    if (mapped.tensors.count("lm_head.weight") != 0)
        throw std::runtime_error("Gguf weight smoke: tied export should omit lm_head");
    if (std::fabs(mapped.tensors.at("token_embedding.weight").data[0] - 0.125f) > 1.0e-6f
        || std::fabs(mapped.tensors.at("blocks.0.attn.k_proj.weight").data[0] - 0.25f) > 1.0e-6f
        || std::fabs(mapped.tensors.at("final_norm.weight").data[0] - 0.375f) > 1.0e-6f)
        throw std::runtime_error("Gguf weight smoke: value mismatch");
    if (mapped.metadata.at("kv_head_count") != "2"
        || mapped.metadata.at("rope_theta").find("5000") == std::string::npos)
        throw std::runtime_error("Gguf weight smoke: arch metadata mismatch");

    std::remove(path.c_str());
    SmokeLog::result("Gguf weight map", "save/load=ok  tie=omit_output  values=ok");
}

} // namespace Gguf

