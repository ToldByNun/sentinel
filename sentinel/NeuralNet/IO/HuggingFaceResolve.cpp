#include "HuggingFaceResolve.hpp"

#include "../Utils/SmokeLog.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#define SENTINEL_POPEN _popen
#define SENTINEL_PCLOSE _pclose
#else
#include <sys/wait.h>
#define SENTINEL_POPEN popen
#define SENTINEL_PCLOSE pclose
#endif

namespace HuggingFace {
namespace {

namespace fs = std::filesystem;

std::string trimCopy(std::string text) {
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.front())))
        text.erase(text.begin());
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back())))
        text.pop_back();
    return text;
}

bool startsWithIgnoreCase(const std::string& text, const char* prefix) {
    const size_t n = std::char_traits<char>::length(prefix);
    if (text.size() < n) return false;
    for (size_t i = 0; i < n; ++i) {
        const unsigned char a = static_cast<unsigned char>(text[i]);
        const unsigned char b = static_cast<unsigned char>(prefix[i]);
        if (std::tolower(a) != std::tolower(b)) return false;
    }
    return true;
}

bool isSafeHubToken(const std::string& token, bool allowSlash) {
    if (token.empty() || token.size() > 256) return false;
    if (token == "." || token == "..") return false;
    if (token.find("..") != std::string::npos) return false;
    for (unsigned char ch : token) {
        if (std::isalnum(ch) || ch == '.' || ch == '_' || ch == '-' || ch == '+')
            continue;
        if (allowSlash && ch == '/') continue;
        return false;
    }
    if (allowSlash) {
        const size_t slash = token.find('/');
        if (slash == std::string::npos) return true; // single-segment ids exist
        if (slash == 0 || slash + 1 >= token.size()) return false;
        if (token.find('/', slash + 1) != std::string::npos) return false;
    }
    return true;
}

bool isSafeRevision(const std::string& revision) {
    if (revision.empty() || revision.size() > 256) return false;
    if (revision.find("..") != std::string::npos) return false;
    for (unsigned char ch : revision) {
        if (std::isalnum(ch) || ch == '.' || ch == '_' || ch == '-' || ch == '/' || ch == '+')
            continue;
        return false;
    }
    return true;
}

bool directoryHasFile(const fs::path& directory, const char* fileName) {
    std::error_code ec;
    return fs::is_directory(directory, ec) && fs::is_regular_file(directory / fileName, ec);
}

std::string normalizeLocalDirectory(const std::string& source, const char* requiredFile) {
    std::error_code ec;
    fs::path path(source);
    if (fs::is_regular_file(path, ec)) {
        const std::string name = path.filename().string();
        if (name == "config.json" || name == "tokenizer.json" || name == "model.safetensors"
            || name == "pytorch_model.bin") {
            path = path.parent_path();
        } else if (requiredFile != nullptr && name == requiredFile) {
            path = path.parent_path();
        } else {
            throw std::invalid_argument(
                "HuggingFace resolve: file is not a HF model/tokenizer entrypoint: " + source);
        }
    }
    if (!fs::is_directory(path, ec))
        throw std::invalid_argument(
            "HuggingFace resolve: local path is not a directory: " + source);
    if (requiredFile != nullptr && !directoryHasFile(path, requiredFile))
        throw std::runtime_error(
            std::string("HuggingFace resolve: missing ") + requiredFile + " under " + path.string());
    fs::path absolute = fs::absolute(path, ec);
    if (ec) absolute = path;
    return absolute.lexically_normal().string();
}

bool looksLikeUrl(const std::string& source) {
    return startsWithIgnoreCase(source, "https://") || startsWithIgnoreCase(source, "http://");
}

std::string stripUrlQueryFragment(std::string url) {
    const size_t q = url.find_first_of("?#");
    if (q != std::string::npos) url.resize(q);
    while (!url.empty() && url.back() == '/') url.pop_back();
    return url;
}

bool parseHubHostPath(const std::string& source, std::string& hostPath) {
    std::string url = stripUrlQueryFragment(source);
    const char* prefixes[] = {
        "https://huggingface.co/",
        "http://huggingface.co/",
        "https://www.huggingface.co/",
        "http://www.huggingface.co/",
        "https://hf.co/",
        "http://hf.co/",
        "https://www.hf.co/",
        "http://www.hf.co/",
    };
    for (const char* prefix : prefixes) {
        if (startsWithIgnoreCase(url, prefix)) {
            hostPath = url.substr(std::char_traits<char>::length(prefix));
            return true;
        }
    }
    return false;
}

ResolvedSource parseHubUrl(const std::string& source) {
    std::string path;
    if (!parseHubHostPath(source, path))
        throw std::invalid_argument(
            "HuggingFace::parseModelSource: unsupported URL host (need huggingface.co / hf.co): "
            + source);
    if (path.empty())
        throw std::invalid_argument("HuggingFace::parseModelSource: empty Hub URL path");

    // Reject non-model hubs early.
    if (startsWithIgnoreCase(path, "datasets/") || startsWithIgnoreCase(path, "spaces/")
        || startsWithIgnoreCase(path, "datasets") || startsWithIgnoreCase(path, "spaces")) {
        throw std::invalid_argument(
            "HuggingFace::parseModelSource: only model repos are supported (not datasets/spaces): "
            + source);
    }
    if (startsWithIgnoreCase(path, "models/"))
        path = path.substr(7);

    std::vector<std::string> parts;
    {
        std::stringstream ss(path);
        std::string item;
        while (std::getline(ss, item, '/')) {
            if (!item.empty()) parts.push_back(item);
        }
    }
    if (parts.size() < 2)
        throw std::invalid_argument(
            "HuggingFace::parseModelSource: Hub URL must look like https://huggingface.co/org/model");

    ResolvedSource out;
    out.fromHub = true;
    out.repoId = parts[0] + "/" + parts[1];
    if (!isSafeHubToken(out.repoId, true))
        throw std::invalid_argument(
            "HuggingFace::parseModelSource: invalid repo id in URL: " + out.repoId);

    if (parts.size() >= 4) {
        const std::string& kind = parts[2];
        if (kind == "tree" || kind == "resolve" || kind == "blob" || kind == "raw") {
            out.revision = parts[3];
            if (!isSafeRevision(out.revision))
                throw std::invalid_argument(
                    "HuggingFace::parseModelSource: invalid revision in URL: " + out.revision);
        }
    }
    return out;
}

ResolvedSource parseRepoId(const std::string& source) {
    std::string repo = source;
    std::string revision;
    const size_t at = repo.find('@');
    if (at != std::string::npos) {
        revision = repo.substr(at + 1);
        repo.resize(at);
        if (!isSafeRevision(revision))
            throw std::invalid_argument(
                "HuggingFace::parseModelSource: invalid revision '" + revision + "'");
    }
    if (!isSafeHubToken(repo, true))
        throw std::invalid_argument(
            "HuggingFace::parseModelSource: invalid Hub repo id '" + repo
            + "' (expected org/name or name)");

    ResolvedSource out;
    out.fromHub = true;
    out.repoId = repo;
    out.revision = revision;
    return out;
}

bool localPathExists(const std::string& source) {
    std::error_code ec;
    return fs::exists(fs::path(source), ec);
}

std::string shellQuote(const std::string& value) {
#if defined(_WIN32)
    std::string out = "\"";
    for (char ch : value) {
        if (ch == '"') out += "\\\"";
        else out.push_back(ch);
    }
    out.push_back('"');
    return out;
#else
    if (value.find('\'') != std::string::npos)
        throw std::invalid_argument("HuggingFace hub download: argument contains quote");
    return "'" + value + "'";
#endif
}

std::string extractDirectoryFromCommandOutput(const std::string& output) {
    std::string best;
    std::stringstream ss(output);
    std::string line;
    while (std::getline(ss, line)) {
        line = trimCopy(line);
        if (line.empty()) continue;
        if (startsWithIgnoreCase(line, "path="))
            line = trimCopy(line.substr(5));
        // Progress / logs occasionally leak; keep the last path-like existing directory.
        std::error_code ec;
        if (fs::is_directory(line, ec)
            && (directoryHasFile(line, "config.json") || directoryHasFile(line, "tokenizer.json")))
            best = line;
    }
    return best;
}

std::string withSuppressedStderr(const std::string& command) {
#if defined(_WIN32)
    return command + " 2>NUL";
#else
    return command + " 2>/dev/null";
#endif
}

std::string withMergedStderr(const std::string& command) {
#if defined(_WIN32)
    return command + " 2>&1";
#else
    return command + " 2>&1";
#endif
}

bool runCapture(const std::string& command, std::string& stdoutText, int& exitCode) {
    stdoutText.clear();
    FILE* pipe = SENTINEL_POPEN(command.c_str(), "r");
    if (pipe == nullptr) return false;
    char buffer[4096];
    while (std::fgets(buffer, sizeof(buffer), pipe) != nullptr)
        stdoutText += buffer;
    const int status = SENTINEL_PCLOSE(pipe);
#if defined(_WIN32)
    exitCode = status;
#else
    if (status == -1) exitCode = -1;
    else if (WIFEXITED(status)) exitCode = WEXITSTATUS(status);
    else exitCode = -1;
#endif
    return true;
}

std::vector<std::string> candidatePythonCommands() {
    std::vector<std::string> out;
#if defined(_WIN32)
    out.push_back("python");
#else
    out.push_back("python3");
    out.push_back("python");
#endif
    return out;
}

std::vector<std::string> candidateHfCliCommands() {
    std::vector<std::string> out;
#if defined(_WIN32)
    out.push_back("hf");
    out.push_back("huggingface-cli");
#else
    out.push_back("hf");
    out.push_back("huggingface-cli");
    const char* home = std::getenv("HOME");
    if (home != nullptr && home[0] != '\0') {
        out.push_back(std::string(home) + "/.local/bin/hf");
        out.push_back(std::string(home) + "/.local/bin/huggingface-cli");
    }
#endif
    return out;
}

std::string downloadViaHfCli(const std::string& cli, const std::string& repoId, const std::string& revision) {
    std::ostringstream cmd;
    cmd << shellQuote(cli) << " download " << shellQuote(repoId);
    if (!revision.empty())
        cmd << " --revision " << shellQuote(revision);
    const std::string base = cmd.str();

    std::string output;
    int code = -1;
    if (!runCapture(withSuppressedStderr(base), output, code) || code != 0) {
        // Retry without silencing stderr so the exception can include useful text.
        output.clear();
        if (!runCapture(withMergedStderr(base), output, code) || code != 0) {
            throw std::runtime_error(
                "HuggingFace hub download failed via " + cli + " (exit " + std::to_string(code)
                + "):\n" + output);
        }
    }
    const std::string directory = extractDirectoryFromCommandOutput(output);
    if (directory.empty())
        throw std::runtime_error(
            "HuggingFace hub download via " + cli
            + " succeeded but no config.json directory found in output:\n" + output);
    return directory;
}

std::string downloadViaPython(const std::string& pythonCmd, const std::string& repoId, const std::string& revision) {
    // repoId/revision are validated safe tokens — embed as Python single-quoted literals.
    std::ostringstream script;
    script << "from huggingface_hub import snapshot_download; print(snapshot_download(repo_id='"
           << repoId << "'";
    if (!revision.empty())
        script << ", revision='" << revision << "'";
    script << "))";

    const std::string base = pythonCmd + " -c " + shellQuote(script.str());
    std::string output;
    int code = -1;
    if (!runCapture(withSuppressedStderr(base), output, code) || code != 0) {
        output.clear();
        if (!runCapture(withMergedStderr(base), output, code) || code != 0) {
            throw std::runtime_error(
                "HuggingFace hub download failed via " + pythonCmd
                + " huggingface_hub (exit " + std::to_string(code) + "):\n" + output);
        }
    }
    const std::string directory = extractDirectoryFromCommandOutput(output);
    if (directory.empty())
        throw std::runtime_error(
            "HuggingFace hub download via python succeeded but no config.json directory found in output:\n"
            + output);
    return directory;
}

std::string downloadFromHub(const std::string& repoId, const std::string& revision) {
    std::vector<std::string> errors;

    for (const std::string& cli : candidateHfCliCommands()) {
        try {
            return downloadViaHfCli(cli, repoId, revision);
        } catch (const std::exception& ex) {
            errors.push_back(ex.what());
        }
    }
    for (const std::string& py : candidatePythonCommands()) {
        try {
            return downloadViaPython(py, repoId, revision);
        } catch (const std::exception& ex) {
            errors.push_back(ex.what());
        }
    }

    std::ostringstream msg;
    msg << "HuggingFace::resolveModelDirectory: cannot download '" << repoId << "'";
    if (!revision.empty()) msg << " @" << revision;
    msg << ". Install the Hugging Face CLI (`pip install huggingface_hub` → `hf`) "
           "or ensure `python` can `import huggingface_hub`. "
           "Gated models need `huggingface-cli login` / HF_TOKEN. Details:";
    for (const std::string& err : errors) {
        msg << "\n---\n" << err;
    }
    throw std::runtime_error(msg.str());
}

} // namespace

ResolvedSource parseModelSource(const std::string& source) {
    const std::string trimmed = trimCopy(source);
    if (trimmed.empty())
        throw std::invalid_argument("HuggingFace::parseModelSource empty source");

    if (looksLikeUrl(trimmed))
        return parseHubUrl(trimmed);

    // Existing local paths take precedence over Hub-id interpretation
    // (so a relative folder named like org/model still works).
    if (localPathExists(trimmed)) {
        ResolvedSource out;
        out.fromHub = false;
        // Do not require a specific file yet — resolve* helpers enforce that.
        out.directory = normalizeLocalDirectory(trimmed, nullptr);
        return out;
    }

    return parseRepoId(trimmed);
}

std::string finalizeResolvedDirectory(std::string directory, const char* requiredFile) {
    if (!directoryHasFile(directory, requiredFile))
        throw std::runtime_error(
            std::string("HuggingFace resolve: missing ") + requiredFile + " under " + directory);
    std::error_code ec;
    fs::path absolute = fs::absolute(directory, ec);
    if (!ec) directory = absolute.lexically_normal().string();
    return directory;
}

std::string resolveDirectoryFor(const std::string& source, const char* requiredFile) {
    ResolvedSource parsed = parseModelSource(source);
    if (!parsed.fromHub) {
        if (parsed.directory.empty())
            throw std::logic_error("HuggingFace resolve: local directory unset");
        return finalizeResolvedDirectory(parsed.directory, requiredFile);
    }
    return finalizeResolvedDirectory(downloadFromHub(parsed.repoId, parsed.revision), requiredFile);
}

std::string resolveModelDirectory(const std::string& source) {
    return resolveDirectoryFor(source, "config.json");
}

std::string resolveTokenizerDirectory(const std::string& source) {
    return resolveDirectoryFor(source, "tokenizer.json");
}

void runResolveSmokeDemo() {
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / "sentinel_hf_resolve_smoke";
    fs::create_directories(dir);
    {
        std::ofstream out(dir / "config.json");
        out << R"json({"model_type":"llama","architectures":["LlamaForCausalLM"],"vocab_size":32,"hidden_size":16,"intermediate_size":32,"num_hidden_layers":1,"num_attention_heads":4,"num_key_value_heads":2,"max_position_embeddings":64,"rms_norm_eps":1e-5,"rope_theta":10000,"tie_word_embeddings":true,"attention_bias":false,"mlp_bias":false})json";
    }

    const ResolvedSource local = parseModelSource(dir.string());
    if (local.fromHub || local.directory.empty())
        throw std::runtime_error("HF resolve smoke: local directory not classified as local");
    const std::string resolvedLocal = resolveModelDirectory(dir.string());
    if (!fs::is_regular_file(fs::path(resolvedLocal) / "config.json"))
        throw std::runtime_error("HF resolve smoke: resolveModelDirectory local failed");

    const ResolvedSource viaConfig = parseModelSource((dir / "config.json").string());
    if (viaConfig.fromHub)
        throw std::runtime_error("HF resolve smoke: config.json path should be local");

    {
        const fs::path tokOnly = fs::temp_directory_path() / "sentinel_hf_resolve_tok_only";
        fs::create_directories(tokOnly);
        {
            std::ofstream out(tokOnly / "tokenizer.json");
            out << R"json({"version":"1.0","model":{"type":"BPE","vocab":{"a":0},"merges":[]}})json";
        }
        const std::string tokDir = resolveTokenizerDirectory(tokOnly.string());
        if (!fs::is_regular_file(fs::path(tokDir) / "tokenizer.json"))
            throw std::runtime_error("HF resolve smoke: tokenizer-only local resolve failed");
        bool rejectedModel = false;
        try {
            (void)resolveModelDirectory(tokOnly.string());
        } catch (const std::exception&) {
            rejectedModel = true;
        }
        if (!rejectedModel)
            throw std::runtime_error("HF resolve smoke: tokenizer-only dir should fail model resolve");
        fs::remove_all(tokOnly);
    }

    const ResolvedSource repo = parseModelSource("meta-llama/Llama-3.2-1B");
    if (!repo.fromHub || repo.repoId != "meta-llama/Llama-3.2-1B" || !repo.revision.empty())
        throw std::runtime_error("HF resolve smoke: repo id parse failed");

    const ResolvedSource repoRev = parseModelSource("meta-llama/Llama-3.2-1B@main");
    if (!repoRev.fromHub || repoRev.repoId != "meta-llama/Llama-3.2-1B" || repoRev.revision != "main")
        throw std::runtime_error("HF resolve smoke: repo@rev parse failed");

    const ResolvedSource url = parseModelSource("https://huggingface.co/meta-llama/Llama-3.2-1B");
    if (!url.fromHub || url.repoId != "meta-llama/Llama-3.2-1B")
        throw std::runtime_error("HF resolve smoke: URL parse failed");

    const ResolvedSource urlTree =
        parseModelSource("https://hf.co/meta-llama/Llama-3.2-1B/tree/main");
    if (!urlTree.fromHub || urlTree.repoId != "meta-llama/Llama-3.2-1B" || urlTree.revision != "main")
        throw std::runtime_error("HF resolve smoke: URL tree/revision parse failed");

    bool rejectedDataset = false;
    try {
        (void)parseModelSource("https://huggingface.co/datasets/glue");
    } catch (const std::exception&) {
        rejectedDataset = true;
    }
    if (!rejectedDataset)
        throw std::runtime_error("HF resolve smoke: dataset URL should be rejected");

    bool rejectedBad = false;
    try {
        (void)parseModelSource("not a repo!!");
    } catch (const std::exception&) {
        rejectedBad = true;
    }
    if (!rejectedBad)
        throw std::runtime_error("HF resolve smoke: invalid repo id should be rejected");

    fs::remove_all(dir);
    SmokeLog::result(
        "HuggingFace resolve",
        "local=ok  repo=ok  url=ok  @rev=ok  rejects=ok");
}

} // namespace HuggingFace
