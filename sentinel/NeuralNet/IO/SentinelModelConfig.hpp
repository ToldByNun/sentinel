#ifndef SENTINELMODELCONFIG_HPP
#define SENTINELMODELCONFIG_HPP

#include <string>

/// <summary>
/// Native Sentinel model architecture config (JSON / YAML) — zero third-party deps.
/// Discriminator: <c>format: "sentinel-model"</c>. Field names match safetensors metadata.
/// </summary>
namespace SentinelModel {

/// <summary>architecture + train defaults used to construct a LanguageModel</summary>
struct Config {
    /// <summary>must be "sentinel-model"</summary>
    std::string format = "sentinel-model";
    int vocabSize = 0;
    int embeddingDim = 0;
    int maxPosition = 0;
    int blockCount = 0;
    int headCount = 0;
    /// <summary>defaults to headCount when omitted (MHA)</summary>
    int kvHeadCount = 0;
    /// <summary>&lt;=0 → legacy expand-4 SwiGLU width</summary>
    int intermediateSize = 0;
    float ropeTheta = 10000.0f;
    bool useBias = true;
    bool tieEmbedding = true;
    float rmsNormEps = 1.0e-5f;
    float learningRate = 3.0e-4f;
    /// <summary>optional weight file (.safetensors / .snlm); empty → random init</summary>
    std::string weights;
};

/// <summary>true when format is the native discriminator</summary>
bool isSentinelModelFormat(const std::string& format);

/// <summary>parse JSON text; requires format == "sentinel-model"</summary>
Config parseConfigJson(const std::string& json);

/// <summary>
/// parse a flat YAML document (key: value). Comments (#) and --- are allowed.
/// Nested maps/lists are not supported.
/// </summary>
Config parseConfigYaml(const std::string& yaml);

/// <summary>auto-detect JSON vs YAML from leading non-ws character / path hint</summary>
Config parseConfigText(const std::string& text, const std::string& pathHint = "");

/// <summary>
/// load model.json / model.yaml / model.yml from a directory, or a concrete file path.
/// Relative <c>weights</c> paths are left as stored; resolve via resolveWeightsPath.
/// </summary>
Config loadConfig(const std::string& pathOrDirectory);

/// <summary>directory containing the config file (or the directory path itself)</summary>
std::string configDirectory(const std::string& pathOrDirectory);

/// <summary>join config directory with relative weights; absolute paths unchanged</summary>
std::string resolveWeightsPath(const std::string& configPathOrDirectory, const std::string& weights);

/// <summary>serialize to JSON (pretty, stable key order)</summary>
std::string serializeConfigJson(const Config& config);

/// <summary>serialize to flat YAML</summary>
std::string serializeConfigYaml(const Config& config);

/// <summary>
/// write config; extension selects JSON (.json) vs YAML (.yaml / .yml).
/// Directory paths write <c>model.json</c>.
/// </summary>
void saveConfig(const std::string& pathOrDirectory, const Config& config);

/// <summary>parse / serialize / load roundtrip smoke (host-only)</summary>
void runConfigParseSmokeDemo();

} // namespace SentinelModel

#endif // SENTINELMODELCONFIG_HPP
