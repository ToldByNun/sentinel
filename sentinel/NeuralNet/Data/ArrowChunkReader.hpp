#ifndef ARROWCHUNKREADER_HPP
#define ARROWCHUNKREADER_HPP

#include "TextRowReader.hpp"

#include <fstream>
#include <string>
#include <vector>

/// <summary>
/// from-scratch Arrow IPC corpus reader (no Apache Arrow C++ dependency)
/// this step: discover shards + open/detect IPC file vs stream; RecordBatch → rows comes next
/// </summary>
class ArrowChunkReader : public TextRowReader {
public:
    enum class IpcKind {
        Unknown,
        File,   // ARROW1 … footer … ARROW1
        Stream  // continuation-framed messages (HF datasets default)
    };

    void open(const std::string& path) override;
    void rewind() override;
    bool nextRows(std::vector<CorpusRow>& out, int maxRows) override;
    bool isOpen() const override;
    size_t rowsRead() const override;

    /// <summary>resolved .arrow shard paths after open (sorted)</summary>
    const std::vector<std::string>& shards() const;

    IpcKind ipcKind() const;

private:
    void collectShards(const std::string& path);
    void openCurrentShard();
    /// <summary>peek header; leave file positioned at start of IPC payload (byte 0 for stream, byte 8 for file)</summary>
    static IpcKind detectAndSeekPayload(std::ifstream& file, const std::string& shardPath);

    std::string rootPath;
    std::vector<std::string> shardPaths;
    size_t shardIndex = 0;
    std::ifstream file;
    IpcKind kind = IpcKind::Unknown;
    size_t rowsConsumed = 0;
};

#endif // ARROWCHUNKREADER_HPP
