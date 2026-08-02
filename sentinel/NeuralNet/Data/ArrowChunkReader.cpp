#include "ArrowChunkReader.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <stdexcept>

namespace {

constexpr char kArrowMagic[6] = { 'A', 'R', 'R', 'O', 'W', '1' };
constexpr std::uint32_t kIpcContinuation = 0xFFFFFFFFu;

std::string toLowerAscii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

bool hasExtension(const std::filesystem::path& path, const char* extensionLower) {
    return toLowerAscii(path.extension().string()) == extensionLower;
}

void appendArrowFiles(const std::filesystem::path& directory, std::vector<std::string>& out) {
    namespace fs = std::filesystem;
    if (!fs::exists(directory) || !fs::is_directory(directory)) return;

    for (const auto& entry : fs::directory_iterator(directory)) {
        if (!entry.is_regular_file()) continue;
        if (!hasExtension(entry.path(), ".arrow")) continue;
        out.push_back(entry.path().string());
    }
}

std::uint32_t readU32Le(const unsigned char* bytes) {
    return static_cast<std::uint32_t>(bytes[0])
        | (static_cast<std::uint32_t>(bytes[1]) << 8)
        | (static_cast<std::uint32_t>(bytes[2]) << 16)
        | (static_cast<std::uint32_t>(bytes[3]) << 24);
}

} // namespace

ArrowChunkReader::IpcKind ArrowChunkReader::detectAndSeekPayload(std::ifstream& file, const std::string& shardPath) {
    unsigned char head[8] = {};
    file.read(reinterpret_cast<char*>(head), 8);
    if (!file || file.gcount() != 8)
        throw std::runtime_error("ArrowChunkReader: cannot read header from " + shardPath);

    if (std::memcmp(head, kArrowMagic, 6) == 0) {
        // File format: magic + 2 pad bytes already consumed; payload starts here.
        return IpcKind::File;
    }

    const std::uint32_t continuation = readU32Le(head);
    if (continuation == kIpcContinuation) {
        // Stream format: rewind so the first framed message starts at byte 0.
        file.clear();
        file.seekg(0, std::ios::beg);
        if (!file)
            throw std::runtime_error("ArrowChunkReader: seek failed on stream " + shardPath);
        return IpcKind::Stream;
    }

    throw std::runtime_error(
        "ArrowChunkReader: unrecognized Arrow IPC header (expected ARROW1 or stream continuation): "
        + shardPath);
}

void ArrowChunkReader::collectShards(const std::string& path) {
    namespace fs = std::filesystem;
    const fs::path fsPath(path);
    this->shardPaths.clear();

    if (fs::is_regular_file(fsPath)) {
        if (!hasExtension(fsPath, ".arrow"))
            throw std::invalid_argument("ArrowChunkReader: expected .arrow file: " + path);
        this->shardPaths.push_back(fsPath.string());
        return;
    }

    if (!fs::is_directory(fsPath))
        throw std::invalid_argument("ArrowChunkReader: path is not file or directory: " + path);

    appendArrowFiles(fsPath, this->shardPaths);
    appendArrowFiles(fsPath / "train", this->shardPaths);

    std::sort(this->shardPaths.begin(), this->shardPaths.end());
    this->shardPaths.erase(
        std::unique(this->shardPaths.begin(), this->shardPaths.end()),
        this->shardPaths.end());

    if (this->shardPaths.empty())
        throw std::runtime_error("ArrowChunkReader: no .arrow shards under " + path);
}

void ArrowChunkReader::openCurrentShard() {
    if (this->shardIndex >= this->shardPaths.size())
        throw std::logic_error("ArrowChunkReader::openCurrentShard out of range");

    this->file.close();
    this->file.clear();
    const std::string& shardPath = this->shardPaths[this->shardIndex];
    this->file.open(shardPath, std::ios::binary);
    if (!this->file.is_open())
        throw std::runtime_error("ArrowChunkReader: cannot open shard " + shardPath);

    this->kind = ArrowChunkReader::detectAndSeekPayload(this->file, shardPath);
}

void ArrowChunkReader::open(const std::string& path) {
    if (path.empty()) throw std::invalid_argument("ArrowChunkReader::open empty path");

    this->rootPath = path;
    this->collectShards(path);
    this->shardIndex = 0;
    this->rowsConsumed = 0;
    this->kind = IpcKind::Unknown;
    this->openCurrentShard();
}

void ArrowChunkReader::rewind() {
    if (this->shardPaths.empty())
        throw std::logic_error("ArrowChunkReader::rewind not open");

    this->shardIndex = 0;
    this->rowsConsumed = 0;
    this->openCurrentShard();
}

bool ArrowChunkReader::nextRows(std::vector<CorpusRow>& out, int maxRows) {
    (void)out;
    if (!this->isOpen()) throw std::logic_error("ArrowChunkReader::nextRows not open");
    if (maxRows <= 0) throw std::invalid_argument("ArrowChunkReader::nextRows maxRows must be > 0");

    const char* kindName = this->kind == IpcKind::File ? "file"
        : this->kind == IpcKind::Stream ? "stream" : "unknown";

    throw std::runtime_error(
        std::string("ArrowChunkReader::nextRows: IPC RecordBatch parsing not implemented yet")
        + " (kind=" + kindName
        + ", shards=" + std::to_string(this->shardPaths.size())
        + ", root=" + this->rootPath + ")");
}

bool ArrowChunkReader::isOpen() const {
    return this->file.is_open();
}

size_t ArrowChunkReader::rowsRead() const {
    return this->rowsConsumed;
}

const std::vector<std::string>& ArrowChunkReader::shards() const {
    return this->shardPaths;
}

ArrowChunkReader::IpcKind ArrowChunkReader::ipcKind() const {
    return this->kind;
}
