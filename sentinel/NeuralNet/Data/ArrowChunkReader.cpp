#include "ArrowChunkReader.hpp"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <stdexcept>

namespace {

constexpr char kArrowMagic[6] = { 'A', 'R', 'R', 'O', 'W', '1' };
constexpr std::uint32_t kIpcContinuation = 0xFFFFFFFFu;

// FlatBuffers union Type (Schema.fbs): NONE=0, Null=1, … Binary=4, Utf8=5, … LargeUtf8=20
constexpr std::uint8_t kTypeUtf8 = 5;
constexpr std::uint8_t kTypeLargeUtf8 = 20;

// FlatBuffers union MessageHeader: Schema=1, DictionaryBatch=2, RecordBatch=3
constexpr std::uint8_t kHeaderSchema = 1;
constexpr std::uint8_t kHeaderDictionaryBatch = 2;
constexpr std::uint8_t kHeaderRecordBatch = 3;

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

std::uint16_t readU16Le(const std::uint8_t* bytes) {
    return static_cast<std::uint16_t>(bytes[0]) | (static_cast<std::uint16_t>(bytes[1]) << 8);
}

std::uint32_t readU32Le(const std::uint8_t* bytes) {
    return static_cast<std::uint32_t>(bytes[0])
        | (static_cast<std::uint32_t>(bytes[1]) << 8)
        | (static_cast<std::uint32_t>(bytes[2]) << 16)
        | (static_cast<std::uint32_t>(bytes[3]) << 24);
}

std::int32_t readI32Le(const std::uint8_t* bytes) {
    return static_cast<std::int32_t>(readU32Le(bytes));
}

std::int64_t readI64Le(const std::uint8_t* bytes) {
    const std::uint64_t low = readU32Le(bytes);
    const std::uint64_t high = readU32Le(bytes + 4);
    return static_cast<std::int64_t>(low | (high << 32));
}

size_t align8(size_t value) {
    return (value + 7u) & ~size_t(7u);
}

struct FbTable {
    const std::uint8_t* base = nullptr;
    size_t size = 0;
    const std::uint8_t* table = nullptr;
    const std::uint8_t* vtable = nullptr;
    std::uint16_t vtableBytes = 0;

    static FbTable fromRoot(const std::uint8_t* buffer, size_t bufferSize) {
        if (buffer == nullptr || bufferSize < 4)
            throw std::runtime_error("ArrowChunkReader: flatbuffer root too small");

        const std::int32_t rootOffset = readI32Le(buffer);
        if (rootOffset < 0 || static_cast<size_t>(rootOffset) + 4 > bufferSize)
            throw std::runtime_error("ArrowChunkReader: flatbuffer root offset out of range");

        FbTable out;
        out.base = buffer;
        out.size = bufferSize;
        out.table = buffer + rootOffset;
        return out.finishInit();
    }

    static FbTable fromTablePtr(const std::uint8_t* buffer, size_t bufferSize, const std::uint8_t* tablePtr) {
        FbTable out;
        out.base = buffer;
        out.size = bufferSize;
        out.table = tablePtr;
        return out.finishInit();
    }

    FbTable finishInit() {
        if (this->table < this->base || this->table + 4 > this->base + this->size)
            throw std::runtime_error("ArrowChunkReader: flatbuffer table out of range");

        const std::int32_t soffset = readI32Le(this->table);
        this->vtable = this->table - soffset;
        if (this->vtable < this->base || this->vtable + 4 > this->base + this->size)
            throw std::runtime_error("ArrowChunkReader: flatbuffer vtable out of range");

        this->vtableBytes = readU16Le(this->vtable);
        if (this->vtableBytes < 4 || this->vtable + this->vtableBytes > this->base + this->size)
            throw std::runtime_error("ArrowChunkReader: flatbuffer vtable size invalid");

        return *this;
    }

    std::uint16_t fieldOffset(int fieldIndex) const {
        const size_t slot = 4u + static_cast<size_t>(fieldIndex) * 2u;
        if (slot + 2u > this->vtableBytes) return 0;
        return readU16Le(this->vtable + slot);
    }

    bool has(int fieldIndex) const {
        return this->fieldOffset(fieldIndex) != 0;
    }

    const std::uint8_t* inlineField(int fieldIndex) const {
        const std::uint16_t off = this->fieldOffset(fieldIndex);
        if (off == 0) return nullptr;
        const std::uint8_t* ptr = this->table + off;
        if (ptr < this->base || ptr >= this->base + this->size)
            throw std::runtime_error("ArrowChunkReader: flatbuffer field out of range");
        return ptr;
    }

    std::uint8_t getU8(int fieldIndex, std::uint8_t defaultValue = 0) const {
        const std::uint8_t* ptr = this->inlineField(fieldIndex);
        return ptr == nullptr ? defaultValue : *ptr;
    }

    std::int16_t getI16(int fieldIndex, std::int16_t defaultValue = 0) const {
        const std::uint8_t* ptr = this->inlineField(fieldIndex);
        return ptr == nullptr ? defaultValue : static_cast<std::int16_t>(readU16Le(ptr));
    }

    std::int32_t getI32(int fieldIndex, std::int32_t defaultValue = 0) const {
        const std::uint8_t* ptr = this->inlineField(fieldIndex);
        return ptr == nullptr ? defaultValue : readI32Le(ptr);
    }

    std::int64_t getI64(int fieldIndex, std::int64_t defaultValue = 0) const {
        const std::uint8_t* ptr = this->inlineField(fieldIndex);
        return ptr == nullptr ? defaultValue : readI64Le(ptr);
    }

    std::string getString(int fieldIndex) const {
        const std::uint8_t* ptr = this->inlineField(fieldIndex);
        if (ptr == nullptr) return {};

        const std::int32_t relative = readI32Le(ptr);
        const std::uint8_t* str = ptr + relative;
        if (str < this->base || str + 4 > this->base + this->size)
            throw std::runtime_error("ArrowChunkReader: flatbuffer string out of range");

        const std::int32_t length = readI32Le(str);
        if (length < 0 || str + 4 + static_cast<size_t>(length) > this->base + this->size)
            throw std::runtime_error("ArrowChunkReader: flatbuffer string length invalid");

        return std::string(reinterpret_cast<const char*>(str + 4), static_cast<size_t>(length));
    }

    const std::uint8_t* vectorBytes(int fieldIndex, std::int32_t* outLength) const {
        const std::uint8_t* ptr = this->inlineField(fieldIndex);
        if (ptr == nullptr) {
            if (outLength != nullptr) *outLength = 0;
            return nullptr;
        }

        const std::int32_t relative = readI32Le(ptr);
        const std::uint8_t* vec = ptr + relative;
        if (vec < this->base || vec + 4 > this->base + this->size)
            throw std::runtime_error("ArrowChunkReader: flatbuffer vector out of range");

        const std::int32_t length = readI32Le(vec);
        if (length < 0)
            throw std::runtime_error("ArrowChunkReader: flatbuffer vector length invalid");

        if (outLength != nullptr) *outLength = length;
        return vec + 4;
    }

    FbTable getTable(int fieldIndex) const {
        const std::uint8_t* ptr = this->inlineField(fieldIndex);
        if (ptr == nullptr)
            throw std::runtime_error("ArrowChunkReader: missing flatbuffer table field");

        const std::int32_t relative = readI32Le(ptr);
        return FbTable::fromTablePtr(this->base, this->size, ptr + relative);
    }

    FbTable getTableFromVector(int fieldIndex, std::int32_t index) const {
        std::int32_t length = 0;
        const std::uint8_t* data = this->vectorBytes(fieldIndex, &length);
        if (data == nullptr || index < 0 || index >= length)
            throw std::runtime_error("ArrowChunkReader: flatbuffer table vector index out of range");

        const std::uint8_t* slot = data + static_cast<size_t>(index) * 4u;
        if (slot + 4 > this->base + this->size)
            throw std::runtime_error("ArrowChunkReader: flatbuffer table vector slot OOB");

        const std::int32_t relative = readI32Le(slot);
        return FbTable::fromTablePtr(this->base, this->size, slot + relative);
    }
};

struct IpcBuffer {
    std::int64_t offset = 0;
    std::int64_t length = 0;
};

struct IpcFieldNode {
    std::int64_t length = 0;
    std::int64_t nullCount = 0;
};

bool bitIsSet(const std::uint8_t* bits, std::int64_t bitLengthBytes, std::int64_t index) {
    if (bits == nullptr || bitLengthBytes <= 0) return true;
    const std::int64_t byteIndex = index >> 3;
    if (byteIndex < 0 || byteIndex >= bitLengthBytes) return true;
    return ((bits[byteIndex] >> (index & 7)) & 1u) != 0;
}

std::string sliceUtf8(
    const std::uint8_t* offsets,
    std::int64_t offsetsLength,
    const std::uint8_t* values,
    std::int64_t valuesLength,
    std::int64_t rowIndex,
    bool largeOffsets
) {
    const std::int64_t offsetWidth = largeOffsets ? 8 : 4;
    const std::int64_t need = (rowIndex + 2) * offsetWidth;
    if (offsets == nullptr || offsetsLength < need)
        throw std::runtime_error("ArrowChunkReader: utf8 offsets buffer too small");

    std::int64_t begin = 0;
    std::int64_t end = 0;
    if (largeOffsets) {
        begin = readI64Le(offsets + static_cast<size_t>(rowIndex) * 8u);
        end = readI64Le(offsets + static_cast<size_t>(rowIndex + 1) * 8u);
    } else {
        begin = readI32Le(offsets + static_cast<size_t>(rowIndex) * 4u);
        end = readI32Le(offsets + static_cast<size_t>(rowIndex + 1) * 4u);
    }

    if (begin < 0 || end < begin || end > valuesLength)
        throw std::runtime_error("ArrowChunkReader: utf8 string slice out of range");

    if (values == nullptr && end > begin)
        throw std::runtime_error("ArrowChunkReader: utf8 values buffer missing");

    if (end == begin) return {};
    return std::string(reinterpret_cast<const char*>(values + begin), static_cast<size_t>(end - begin));
}

const std::uint8_t* bodySlice(const std::uint8_t* body, size_t bodySize, const IpcBuffer& buffer) {
    if (buffer.length <= 0) return nullptr;
    if (buffer.offset < 0 || buffer.length < 0)
        throw std::runtime_error("ArrowChunkReader: negative buffer offset/length");
    if (static_cast<size_t>(buffer.offset + buffer.length) > bodySize)
        throw std::runtime_error("ArrowChunkReader: record batch buffer exceeds body");
    return body + static_cast<size_t>(buffer.offset);
}

} // namespace

ArrowChunkReader::IpcKind ArrowChunkReader::detectAndSeekPayload(std::ifstream& file, const std::string& shardPath) {
    unsigned char head[8] = {};
    file.read(reinterpret_cast<char*>(head), 8);
    if (!file || file.gcount() != 8)
        throw std::runtime_error("ArrowChunkReader: cannot read header from " + shardPath);

    if (std::memcmp(head, kArrowMagic, 6) == 0)
        return IpcKind::File;

    const std::uint32_t continuation = readU32Le(head);
    if (continuation == kIpcContinuation) {
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

    // HF datasets: data-*.arrow hold rows; cache-*.arrow is an index-only stream — skip it.
    std::vector<std::string> dataShards;
    std::vector<std::string> otherShards;
    dataShards.reserve(this->shardPaths.size());
    for (const std::string& shard : this->shardPaths) {
        const std::string name = toLowerAscii(fs::path(shard).filename().string());
        if (name.rfind("cache-", 0) == 0) continue;
        if (name.rfind("data-", 0) == 0) dataShards.push_back(shard);
        else otherShards.push_back(shard);
    }

    if (!dataShards.empty())
        this->shardPaths = std::move(dataShards);
    else
        this->shardPaths = std::move(otherShards);

    if (this->shardPaths.empty())
        throw std::runtime_error("ArrowChunkReader: no .arrow data shards under " + path);
}

void ArrowChunkReader::resetDecodeState() {
    this->schemaReady = false;
    this->schemaFields.clear();
    this->textFieldIndex = -1;
    this->sourceFieldIndex = -1;
    this->pending.rows.clear();
    this->pending.cursor = 0;
    this->streamExhausted = false;
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
    this->schemaReady = false;
    this->schemaFields.clear();
    this->textFieldIndex = -1;
    this->sourceFieldIndex = -1;
    this->pending.rows.clear();
    this->pending.cursor = 0;
}

void ArrowChunkReader::open(const std::string& path) {
    if (path.empty()) throw std::invalid_argument("ArrowChunkReader::open empty path");

    this->rootPath = path;
    this->collectShards(path);
    this->shardIndex = 0;
    this->rowsConsumed = 0;
    this->kind = IpcKind::Unknown;
    this->resetDecodeState();
    this->openCurrentShard();
}

void ArrowChunkReader::rewind() {
    if (this->shardPaths.empty())
        throw std::logic_error("ArrowChunkReader::rewind not open");

    this->shardIndex = 0;
    this->rowsConsumed = 0;
    this->resetDecodeState();
    this->openCurrentShard();
}

bool ArrowChunkReader::advanceToNextShard() {
    if (this->shardIndex + 1 >= this->shardPaths.size()) {
        this->streamExhausted = true;
        return false;
    }

    ++this->shardIndex;
    this->openCurrentShard();
    return true;
}

bool ArrowChunkReader::readEncapsulatedMessage(std::vector<std::uint8_t>& metadata, std::vector<std::uint8_t>& body) {
    unsigned char prefix[8] = {};
    this->file.read(reinterpret_cast<char*>(prefix), 8);
    if (this->file.gcount() == 0)
        return false;
    if (!this->file || this->file.gcount() != 8)
        throw std::runtime_error("ArrowChunkReader: truncated IPC message prefix");

    const std::uint32_t continuation = readU32Le(prefix);
    if (continuation != kIpcContinuation)
        throw std::runtime_error("ArrowChunkReader: expected IPC continuation marker");

    const std::int32_t metadataSize = readI32Le(prefix + 4);
    if (metadataSize == 0)
        return false; // EOS
    if (metadataSize < 0)
        throw std::runtime_error("ArrowChunkReader: negative metadata size");

    metadata.resize(static_cast<size_t>(metadataSize));
    this->file.read(reinterpret_cast<char*>(metadata.data()), metadataSize);
    if (!this->file || this->file.gcount() != metadataSize)
        throw std::runtime_error("ArrowChunkReader: truncated IPC metadata");

    const FbTable message = FbTable::fromRoot(metadata.data(), metadata.size());
    // Message fields: version=0, header_type=1, header=2, bodyLength=3, custom_metadata=4
    const std::int64_t bodyLength = message.getI64(3, 0);
    if (bodyLength < 0)
        throw std::runtime_error("ArrowChunkReader: negative bodyLength");

    body.resize(static_cast<size_t>(bodyLength));
    if (bodyLength > 0) {
        this->file.read(reinterpret_cast<char*>(body.data()), bodyLength);
        if (!this->file || this->file.gcount() != bodyLength)
            throw std::runtime_error("ArrowChunkReader: truncated IPC body");
    }

    const size_t padded = align8(static_cast<size_t>(bodyLength));
    if (padded > static_cast<size_t>(bodyLength)) {
        const std::streamoff skip = static_cast<std::streamoff>(padded - static_cast<size_t>(bodyLength));
        this->file.seekg(skip, std::ios::cur);
        if (!this->file)
            throw std::runtime_error("ArrowChunkReader: failed skipping body padding");
    }

    return true;
}

void ArrowChunkReader::applySchemaMessage(const std::uint8_t* metadata, size_t metadataSize) {
    const FbTable message = FbTable::fromRoot(metadata, metadataSize);
    const std::uint8_t headerType = message.getU8(1);
    if (headerType != kHeaderSchema)
        throw std::logic_error("ArrowChunkReader::applySchemaMessage wrong header type");

    const FbTable schema = message.getTable(2);
    // Schema: endianness=0, fields=1, …
    std::int32_t fieldCount = 0;
    schema.vectorBytes(1, &fieldCount);

    this->schemaFields.clear();
    this->schemaFields.reserve(static_cast<size_t>(fieldCount));
    this->textFieldIndex = -1;
    this->sourceFieldIndex = -1;

    for (std::int32_t index = 0; index < fieldCount; ++index) {
        const FbTable field = schema.getTableFromVector(1, index);
        SchemaField parsed;
        parsed.name = field.getString(0);
        const std::uint8_t typeId = field.getU8(2);
        if (typeId == kTypeUtf8) parsed.kind = ColumnKind::Utf8;
        else if (typeId == kTypeLargeUtf8) parsed.kind = ColumnKind::LargeUtf8;
        else parsed.kind = ColumnKind::Unsupported;

        if (parsed.kind == ColumnKind::Unsupported) {
            throw std::runtime_error(
                "ArrowChunkReader: unsupported column type (need flat Utf8/LargeUtf8): " + parsed.name);
        }

        if (this->textFieldIndex < 0) {
            if (parsed.name == "problem_statement" || parsed.name == "text" || parsed.name == "content")
                this->textFieldIndex = static_cast<int>(this->schemaFields.size());
        }
        if (this->sourceFieldIndex < 0 && parsed.name == "source")
            this->sourceFieldIndex = static_cast<int>(this->schemaFields.size());

        this->schemaFields.push_back(std::move(parsed));
    }

    if (this->textFieldIndex < 0) {
        throw std::runtime_error(
            "ArrowChunkReader: schema has no problem_statement/text/content column (shard="
            + this->shardPaths[this->shardIndex] + ")");
    }

    this->schemaReady = true;
}

void ArrowChunkReader::decodeRecordBatchMessage(
    const std::uint8_t* metadata,
    size_t metadataSize,
    const std::uint8_t* body,
    size_t bodySize
) {
    if (!this->schemaReady)
        throw std::runtime_error("ArrowChunkReader: RecordBatch before Schema");

    const FbTable message = FbTable::fromRoot(metadata, metadataSize);
    const FbTable batch = message.getTable(2);

    // RecordBatch: length=0, nodes=1, buffers=2, compression=3
    if (batch.has(3))
        throw std::runtime_error("ArrowChunkReader: compressed RecordBatch bodies not supported");

    const std::int64_t rowCount = batch.getI64(0, 0);
    if (rowCount < 0)
        throw std::runtime_error("ArrowChunkReader: negative RecordBatch length");

    std::int32_t nodeCount = 0;
    const std::uint8_t* nodeBytes = batch.vectorBytes(1, &nodeCount);
    std::int32_t bufferCount = 0;
    const std::uint8_t* bufferBytes = batch.vectorBytes(2, &bufferCount);

    if (nodeBytes == nullptr || bufferBytes == nullptr)
        throw std::runtime_error("ArrowChunkReader: RecordBatch missing nodes/buffers");
    if (nodeCount != static_cast<std::int32_t>(this->schemaFields.size()))
        throw std::runtime_error("ArrowChunkReader: FieldNode count != schema field count");

    std::vector<IpcFieldNode> nodes(static_cast<size_t>(nodeCount));
    for (std::int32_t index = 0; index < nodeCount; ++index) {
        const std::uint8_t* ptr = nodeBytes + static_cast<size_t>(index) * 16u;
        nodes[static_cast<size_t>(index)].length = readI64Le(ptr);
        nodes[static_cast<size_t>(index)].nullCount = readI64Le(ptr + 8);
    }

    std::vector<IpcBuffer> buffers(static_cast<size_t>(bufferCount));
    for (std::int32_t index = 0; index < bufferCount; ++index) {
        const std::uint8_t* ptr = bufferBytes + static_cast<size_t>(index) * 16u;
        buffers[static_cast<size_t>(index)].offset = readI64Le(ptr);
        buffers[static_cast<size_t>(index)].length = readI64Le(ptr + 8);
    }

    const size_t expectedBuffers = this->schemaFields.size() * 3u;
    if (buffers.size() != expectedBuffers)
        throw std::runtime_error("ArrowChunkReader: expected 3 buffers per Utf8 field");

    std::vector<std::string> texts(static_cast<size_t>(rowCount));
    std::vector<std::string> sources(static_cast<size_t>(rowCount));

    size_t bufferIndex = 0;
    for (size_t fieldIndex = 0; fieldIndex < this->schemaFields.size(); ++fieldIndex) {
        const SchemaField& field = this->schemaFields[fieldIndex];
        const IpcFieldNode& node = nodes[fieldIndex];
        if (node.length != rowCount)
            throw std::runtime_error("ArrowChunkReader: FieldNode length != RecordBatch length");

        if (bufferIndex + 2 >= buffers.size())
            throw std::runtime_error("ArrowChunkReader: buffer index overrun");

        const IpcBuffer validityBuf = buffers[bufferIndex++];
        const IpcBuffer offsetsBuf = buffers[bufferIndex++];
        const IpcBuffer valuesBuf = buffers[bufferIndex++];

        const bool wantText = static_cast<int>(fieldIndex) == this->textFieldIndex;
        const bool wantSource = static_cast<int>(fieldIndex) == this->sourceFieldIndex;
        if (!wantText && !wantSource) continue;

        const std::uint8_t* validity = bodySlice(body, bodySize, validityBuf);
        const std::uint8_t* offsets = bodySlice(body, bodySize, offsetsBuf);
        const std::uint8_t* values = bodySlice(body, bodySize, valuesBuf);
        const bool largeOffsets = field.kind == ColumnKind::LargeUtf8;

        for (std::int64_t row = 0; row < rowCount; ++row) {
            if (!bitIsSet(validity, validityBuf.length, row)) continue;

            std::string value = sliceUtf8(
                offsets,
                offsetsBuf.length,
                values,
                valuesBuf.length,
                row,
                largeOffsets);

            if (wantText) texts[static_cast<size_t>(row)] = std::move(value);
            else sources[static_cast<size_t>(row)] = std::move(value);
        }
    }

    this->pending.rows.clear();
    this->pending.rows.reserve(static_cast<size_t>(rowCount));
    this->pending.cursor = 0;

    for (std::int64_t row = 0; row < rowCount; ++row) {
        CorpusRow outRow;
        outRow.text = std::move(texts[static_cast<size_t>(row)]);
        outRow.source = std::move(sources[static_cast<size_t>(row)]);
        if (outRow.text.empty()) continue;
        this->pending.rows.push_back(std::move(outRow));
    }
}

bool ArrowChunkReader::loadNextDecodedBatch() {
    std::vector<std::uint8_t> metadata;
    std::vector<std::uint8_t> body;

    while (!this->streamExhausted) {
        if (!this->readEncapsulatedMessage(metadata, body)) {
            if (!this->advanceToNextShard())
                return false;
            continue;
        }

        const FbTable message = FbTable::fromRoot(metadata.data(), metadata.size());
        const std::uint8_t headerType = message.getU8(1);

        if (headerType == kHeaderSchema) {
            this->applySchemaMessage(metadata.data(), metadata.size());
            continue;
        }

        if (headerType == kHeaderDictionaryBatch)
            continue;

        if (headerType == kHeaderRecordBatch) {
            this->decodeRecordBatchMessage(metadata.data(), metadata.size(), body.data(), body.size());
            if (!this->pending.rows.empty())
                return true;
            continue;
        }

        throw std::runtime_error(
            "ArrowChunkReader: unsupported MessageHeader type "
            + std::to_string(static_cast<int>(headerType)));
    }

    return false;
}

bool ArrowChunkReader::nextRows(std::vector<CorpusRow>& out, int maxRows) {
    if (this->shardPaths.empty())
        throw std::logic_error("ArrowChunkReader::nextRows not open");
    if (maxRows <= 0) throw std::invalid_argument("ArrowChunkReader::nextRows maxRows must be > 0");

    out.clear();
    out.reserve(static_cast<size_t>(maxRows));

    while (static_cast<int>(out.size()) < maxRows) {
        if (this->pending.cursor >= this->pending.rows.size()) {
            this->pending.rows.clear();
            this->pending.cursor = 0;
            if (!this->loadNextDecodedBatch())
                break;
        }

        out.push_back(std::move(this->pending.rows[this->pending.cursor]));
        ++this->pending.cursor;
        ++this->rowsConsumed;
    }

    const bool morePending = this->pending.cursor < this->pending.rows.size();
    return morePending || !this->streamExhausted;
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
