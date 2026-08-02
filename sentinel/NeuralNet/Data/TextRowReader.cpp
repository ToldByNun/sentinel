#include "TextRowReader.hpp"
#include "ArrowChunkReader.hpp"
#include "JsonlLoader.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <stdexcept>

namespace {

std::string toLowerAscii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

bool hasExtension(const std::filesystem::path& path, const char* extensionLower) {
    return toLowerAscii(path.extension().string()) == extensionLower;
}

bool looksLikeHfDatasetDir(const std::filesystem::path& directory) {
    namespace fs = std::filesystem;
    if (fs::exists(directory / "dataset_info.json") || fs::exists(directory / "state.json"))
        return true;
    if (fs::exists(directory / "train" / "dataset_info.json") || fs::exists(directory / "train" / "state.json"))
        return true;

    const auto scanForArrow = [](const fs::path& root) {
        if (!fs::exists(root) || !fs::is_directory(root)) return false;
        for (const auto& entry : fs::directory_iterator(root)) {
            if (!entry.is_regular_file()) continue;
            if (hasExtension(entry.path(), ".arrow")) return true;
        }
        return false;
    };

    return scanForArrow(directory) || scanForArrow(directory / "train");
}

} // namespace

std::unique_ptr<TextRowReader> createTextRowReader(const std::string& path) {
    if (path.empty()) throw std::invalid_argument("createTextRowReader empty path");

    namespace fs = std::filesystem;
    const fs::path fsPath(path);

    if (!fs::exists(fsPath))
        throw std::runtime_error("createTextRowReader: path does not exist: " + path);

    if (fs::is_directory(fsPath)) {
        if (looksLikeHfDatasetDir(fsPath)) {
            auto reader = std::make_unique<ArrowChunkReader>();
            reader->open(path);
            return reader;
        }
        throw std::runtime_error(
            "createTextRowReader: unsupported directory (expected HF Arrow dataset or .jsonl file): " + path);
    }

    if (!fs::is_regular_file(fsPath))
        throw std::runtime_error("createTextRowReader: not a file or directory: " + path);

    if (hasExtension(fsPath, ".arrow")) {
        auto reader = std::make_unique<ArrowChunkReader>();
        reader->open(path);
        return reader;
    }

    if (hasExtension(fsPath, ".jsonl")) {
        auto reader = std::make_unique<JsonlChunkReader>();
        reader->open(path);
        return reader;
    }

    throw std::runtime_error(
        "createTextRowReader: unknown corpus format (use .jsonl or .arrow / HF dataset dir): " + path);
}
