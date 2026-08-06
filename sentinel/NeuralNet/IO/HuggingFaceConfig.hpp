#ifndef HUGGINGFACECONFIG_HPP
#define HUGGINGFACECONFIG_HPP

#include <string>
#include <vector>

/// <summary>
/// Minimal HuggingFace config.json reader — zero third-party deps.
/// Parses arch fields needed to size a Sentinel LanguageModel; rejects unsupported
/// model_type / MoE / sliding-window / quantized checkpoints early.
/// </summary>
namespace HuggingFace {

/// <summary>subset of HF causal-LM config used by Sentinel import</summary>
struct Config {
    std::string modelType;
    /// <summary>first entry of architectures[], if present</summary>
    std::string architecture;
    int vocabSize = 0;
    int hiddenSize = 0;
    int intermediateSize = 0;
    int numHiddenLayers = 0;
    int numAttentionHeads = 0;
    /// <summary>HF num_key_value_heads; equals numAttentionHeads when omitted (MHA)</summary>
    int numKeyValueHeads = 0;
    int maxPositionEmbeddings = 0;
    float rmsNormEps = 1.0e-5f;
    float ropeTheta = 10000.0f;
    bool tieWordEmbeddings = false;
    /// <summary>trainable FFN / lm_head bias; false for typical Llama/Mistral/Qwen2</summary>
    bool useBias = false;
};

/// <summary>allowlisted model_type values (Llama/Mistral/Qwen2-like layout family)</summary>
bool isSupportedModelType(const std::string& modelType);

/// <summary>load config.json from a file path or a HF model directory</summary>
Config loadConfig(const std::string& pathOrDirectory);

/// <summary>parse config.json text (UTF-8); throws on malformed / unsupported</summary>
Config parseConfigJson(const std::string& json);

/// <summary>fixture: allowlisted parse + reject unknown/MoE/sliding_window/quantized</summary>
void runConfigParseSmokeDemo();

} // namespace HuggingFace

#endif // HUGGINGFACECONFIG_HPP
