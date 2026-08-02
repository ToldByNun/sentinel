#ifndef ARROWCHUNKREADER_HPP
#define ARROWCHUNKREADER_HPP

#include "TextRowReader.hpp"

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

/// <summary>
/// from-scratch Arrow IPC corpus reader (no Apache Arrow C++ dependency)
/// supports IPC stream (HF datasets) and file magic; flat Utf8 columns only for now
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

    const std::vector<std::string>& shards() const;
    IpcKind ipcKind() const;

private:
    enum class ColumnKind : std::uint8_t {
        Unsupported = 0,
        Utf8,
        LargeUtf8
    };

    struct SchemaField {
        std::string name;
        ColumnKind kind = ColumnKind::Unsupported;
    };

    struct DecodedBatch {
        std::vector<CorpusRow> rows;
        size_t cursor = 0;
    };

    void collectShards(const std::string& path);
    void openCurrentShard();
    void resetDecodeState();
    static IpcKind detectAndSeekPayload(std::ifstream& file, const std::string& shardPath);

    bool advanceToNextShard();
    bool readEncapsulatedMessage(std::vector<std::uint8_t>& metadata, std::vector<std::uint8_t>& body);
    bool loadNextDecodedBatch();
    void applySchemaMessage(const std::uint8_t* metadata, size_t metadataSize);
    void decodeRecordBatchMessage(const std::uint8_t* metadata, size_t metadataSize, const std::uint8_t* body, size_t bodySize);

    std::string rootPath;
    std::vector<std::string> shardPaths;
    size_t shardIndex = 0;
    std::ifstream file;
    IpcKind kind = IpcKind::Unknown;
    size_t rowsConsumed = 0;

    bool schemaReady = false;
    std::vector<SchemaField> schemaFields;
    int textFieldIndex = -1;
    int sourceFieldIndex = -1;

    DecodedBatch pending;
    bool streamExhausted = false;
};

#endif // ARROWCHUNKREADER_HPP
