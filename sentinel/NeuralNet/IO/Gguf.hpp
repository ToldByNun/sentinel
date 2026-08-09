#ifndef GGUF_HPP
#define GGUF_HPP

#include "SafeTensors.hpp"

#include <cstdint>
#include <string>
#include <vector>

/// <summary>
/// Minimal GGUF (v3) import/export — zero third-party deps.
/// Supports F32 / F16 / BF16 llama / mistral / qwen2-style tensors (no quantized packs).
/// Remaps llama.cpp names ↔ Sentinel tensor names into a SafeTensors::File for LanguageModel.
/// Spec: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
/// </summary>
namespace Gguf {

/// <summary>architecture fields needed to size a LanguageModel</summary>
struct Config {
    std::string architecture; // general.architecture (llama / mistral / qwen2)
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
    /// <summary>trainable FFN / lm_head bias (rare in GGUF llama packs)</summary>
    bool useBias = false;
};

/// <summary>one GGUF ↔ Sentinel tensor mapping entry</summary>
struct WeightMapEntry {
    std::string sentinelName;
    std::string ggufName;
    size_t expectedRows = 0;
    size_t expectedCols = 0;
    bool optional = false;
};

/// <summary>allowlisted general.architecture values</summary>
bool isSupportedArchitecture(const std::string& architecture);

/// <summary>true when path starts with GGUF magic</summary>
bool isGgufFile(const std::string& path);

/// <summary>read architecture metadata (+ infer vocab from token_embd when needed)</summary>
Config loadConfig(const std::string& path);

/// <summary>expand full GGUF↔Sentinel map for a parsed config</summary>
std::vector<WeightMapEntry> buildWeightMap(const Config& config);

/// <summary>
/// load GGUF weights (F32/F16/BF16), remap to Sentinel names + shapes.
/// Sets arch metadata so LanguageModel::loadSafeTensors can consume the File.
/// </summary>
SafeTensors::File loadMappedWeights(const std::string& path, const Config& config);

/// <summary>
/// write a GGUF v3 file (F32 tensors + llama.cpp-style metadata/names).
/// Omits optional tensors that are absent (tied lm_head, unused biases).
/// </summary>
void save(
    const std::string& path,
    const Config& config,
    const SafeTensors::File& sentinelWeights);

/// <summary>parse metadata + reject unsupported / quantized fixtures</summary>
void runConfigParseSmokeDemo();

/// <summary>tiny write → remap shapes/values + tie/bias policy</summary>
void runWeightMapSmokeDemo();

} // namespace Gguf

#endif // GGUF_HPP
