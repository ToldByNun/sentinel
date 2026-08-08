#ifndef HUGGINGFACERESOLVE_HPP
#define HUGGINGFACERESOLVE_HPP

#include <string>

/// <summary>
/// Resolve a HuggingFace model source to a local directory.
/// Accepts: local path, Hub repo id (org/name), or huggingface.co / hf.co URL.
/// Hub downloads use the standard Hugging Face cache (huggingface_hub / hf CLI) —
/// Sentinel does not keep a separate cache.
/// </summary>
namespace HuggingFace {

/// <summary>parsed view of a model source before / after local resolution</summary>
struct ResolvedSource {
    /// <summary>local directory containing config.json (set after resolve; may be empty after parse-only)</summary>
    std::string directory;
    /// <summary>Hub repo id when source is remote (e.g. meta-llama/Llama-3.2-1B)</summary>
    std::string repoId;
    /// <summary>optional revision (branch / tag / commit); empty → Hub default</summary>
    std::string revision;
    /// <summary>true when a Hub download is required</summary>
    bool fromHub = false;
};

/// <summary>
/// classify source without downloading.
/// Local existing paths win over Hub-id interpretation.
/// </summary>
ResolvedSource parseModelSource(const std::string& source);

/// <summary>
/// return a local directory containing config.json.
/// Downloads via `hf download`, `huggingface-cli download`, or
/// `python -c "huggingface_hub.snapshot_download(...)"` into the default Hub cache.
/// </summary>
std::string resolveModelDirectory(const std::string& source);

/// <summary>
/// return a local directory containing tokenizer.json (same Hub sources as resolveModelDirectory).
/// Local tokenizer-only folders are allowed (config.json not required).
/// </summary>
std::string resolveTokenizerDirectory(const std::string& source);

/// <summary>offline parse fixtures (local / URL / repo id / rejects)</summary>
void runResolveSmokeDemo();

} // namespace HuggingFace

#endif // HUGGINGFACERESOLVE_HPP
