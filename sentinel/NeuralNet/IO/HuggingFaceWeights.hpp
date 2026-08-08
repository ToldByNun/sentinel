#ifndef HUGGINGFACEWEIGHTS_HPP
#define HUGGINGFACEWEIGHTS_HPP

#include "HuggingFaceConfig.hpp"
#include "SafeTensors.hpp"

#include <map>
#include <string>
#include <vector>

namespace HuggingFace {

/// <summary>first supported HF naming family (Llama / Mistral / Qwen2-style keys)</summary>
enum class WeightLayoutFamily {
    LlamaMistralLike = 0
};

/// <summary>one HF → Sentinel tensor mapping entry (shapes validated on load)</summary>
struct WeightMapEntry {
    std::string sentinelName;
    std::string hfName;
    size_t expectedRows = 0;
    size_t expectedCols = 0; // 1 for vectors / norms
    bool transpose = false;  // Llama/Mistral-like Linear is already [out, in]
    bool optional = false;   // missing → zeros (bias) or tie copy (lm_head)
};

/// <summary>shard index: HF tensor name → relative safetensors filename</summary>
struct WeightShardIndex {
    std::string directory;
    /// <summary>from model.safetensors.index.json; empty when using singleFile</summary>
    std::map<std::string, std::string> weightToFile;
    /// <summary>relative filename when the repo is a single .safetensors (no index)</summary>
    std::string singleFile;
};

/// <summary>expand full HF→Sentinel map for a parsed config</summary>
std::vector<WeightMapEntry> buildWeightMap(
    const Config& config,
    WeightLayoutFamily family = WeightLayoutFamily::LlamaMistralLike);

/// <summary>
/// resolve shards under a HF model directory:
/// prefers model.safetensors.index.json, else model.safetensors, else a single *.safetensors
/// </summary>
WeightShardIndex loadWeightShardIndex(const std::string& modelDirectory);

/// <summary>
/// load HF safetensors (possibly sharded), remap to Sentinel tensor names + shapes.
/// Sets arch metadata so LanguageModel::loadSafeTensors can consume the File later.
/// Respects tie_word_embeddings (copies embed → lm_head when head missing) and use_bias.
/// </summary>
SafeTensors::File loadMappedWeights(
    const std::string& modelDirectory,
    const Config& config,
    WeightLayoutFamily family = WeightLayoutFamily::LlamaMistralLike);

/// <summary>
/// remap a Sentinel-named safetensors File to HF tensor names for the given config/layout.
/// Omits optional tensors that are absent (tied lm_head, unused biases).
/// </summary>
SafeTensors::File remapSentinelWeightsToHf(
    const SafeTensors::File& sentinelWeights,
    const Config& config,
    WeightLayoutFamily family = WeightLayoutFamily::LlamaMistralLike);

/// <summary>
/// write a Transformers-compatible model directory:
/// config.json + model.safetensors (HF names, F32).
/// Optionally copies tokenizer*.json / vocab/merges / generation_config.json from tokenizerSourceDirectory.
/// </summary>
void saveDirectory(
    const std::string& modelDirectory,
    const Config& config,
    const SafeTensors::File& sentinelWeights,
    WeightLayoutFamily family = WeightLayoutFamily::LlamaMistralLike,
    const std::string& tokenizerSourceDirectory = "");

/// <summary>tiny sharded HF stub → remap shapes/values + tie/bias policy</summary>
void runWeightMapSmokeDemo();

} // namespace HuggingFace

#endif // HUGGINGFACEWEIGHTS_HPP
