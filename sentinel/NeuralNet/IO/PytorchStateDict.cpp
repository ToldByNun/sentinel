#include "PytorchStateDict.hpp"

#include "../Utils/SmokeLog.hpp"

#include <algorithm>
#include <cctype>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <memory>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>
#include <zlib.h>

namespace PytorchStateDict {
namespace {

namespace fs = std::filesystem;

// --- half / bfloat16 decode (same bit layouts as SafeTensors) ---

float floatFromF16Bits(std::uint16_t bits) {
    const std::uint32_t sign = (bits >> 15) & 0x1u;
    const std::uint32_t exp = (bits >> 10) & 0x1fu;
    const std::uint32_t mant = bits & 0x3ffu;
    std::uint32_t outBits = 0;
    if (exp == 0) {
        if (mant == 0) {
            outBits = sign << 31;
        } else {
            // subnormal → normalize
            std::uint32_t m = mant;
            std::uint32_t e = 127 - 15 + 1;
            while ((m & 0x400u) == 0) {
                m <<= 1;
                --e;
            }
            m &= 0x3ffu;
            outBits = (sign << 31) | (e << 23) | (m << 13);
        }
    } else if (exp == 31) {
        outBits = (sign << 31) | (0xffu << 23) | (mant << 13);
    } else {
        outBits = (sign << 31) | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
    float value = 0.0f;
    std::memcpy(&value, &outBits, sizeof(value));
    return value;
}

float floatFromBf16Bits(std::uint16_t bits) {
    std::uint32_t outBits = static_cast<std::uint32_t>(bits) << 16;
    float value = 0.0f;
    std::memcpy(&value, &outBits, sizeof(value));
    return value;
}

std::vector<std::uint8_t> readEntireFileBytes(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("PytorchStateDict: cannot open " + path);
    in.seekg(0, std::ios::end);
    const std::streamoff size = in.tellg();
    if (size < 0) throw std::runtime_error("PytorchStateDict: tellg failed for " + path);
    in.seekg(0, std::ios::beg);
    std::vector<std::uint8_t> bytes(static_cast<size_t>(size));
    if (!bytes.empty())
        in.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!in) throw std::runtime_error("PytorchStateDict: read failed for " + path);
    return bytes;
}

void writeEntireFileBytes(const std::string& path, const std::vector<std::uint8_t>& bytes) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("PytorchStateDict: cannot write " + path);
    if (!bytes.empty())
        out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!out) throw std::runtime_error("PytorchStateDict: write failed for " + path);
}

// --- minimal ZIP reader (central directory; STORE + DEFLATE) ---

struct ZipEntry {
    std::string name;
    std::uint32_t compression = 0;
    std::uint64_t compressedSize = 0;
    std::uint64_t uncompressedSize = 0;
    std::uint64_t localHeaderOffset = 0;
    std::uint32_t crc32 = 0;
};

struct ZipArchive {
    std::vector<std::uint8_t> bytes;
    std::vector<ZipEntry> entries;
    std::unordered_map<std::string, size_t> nameToIndex;
};

std::uint16_t readU16(const std::vector<std::uint8_t>& b, size_t off) {
    if (off + 2 > b.size()) throw std::runtime_error("PytorchStateDict ZIP: truncated u16");
    return static_cast<std::uint16_t>(b[off] | (static_cast<std::uint16_t>(b[off + 1]) << 8));
}

std::uint32_t readU32(const std::vector<std::uint8_t>& b, size_t off) {
    if (off + 4 > b.size()) throw std::runtime_error("PytorchStateDict ZIP: truncated u32");
    return static_cast<std::uint32_t>(b[off])
        | (static_cast<std::uint32_t>(b[off + 1]) << 8)
        | (static_cast<std::uint32_t>(b[off + 2]) << 16)
        | (static_cast<std::uint32_t>(b[off + 3]) << 24);
}

std::uint64_t readU64(const std::vector<std::uint8_t>& b, size_t off) {
    if (off + 8 > b.size()) throw std::runtime_error("PytorchStateDict ZIP: truncated u64");
    std::uint64_t value = 0;
    for (int i = 0; i < 8; ++i)
        value |= static_cast<std::uint64_t>(b[off + static_cast<size_t>(i)]) << (8 * i);
    return value;
}

ZipArchive openZip(const std::vector<std::uint8_t>& bytes) {
    if (bytes.size() < 22) throw std::runtime_error("PytorchStateDict ZIP: file too small");
    if (!(bytes[0] == 'P' && bytes[1] == 'K' && bytes[2] == 3 && bytes[3] == 4))
        throw std::runtime_error("PytorchStateDict: not a ZIP archive (need modern torch.save format)");

    // Find EOCD (or ZIP64 locator).
    size_t eocd = static_cast<size_t>(-1);
    const size_t minEocd = bytes.size() >= 22 ? bytes.size() - 22 : 0;
    const size_t searchFrom = bytes.size() > (65535 + 22) ? bytes.size() - (65535 + 22) : 0;
    for (size_t i = bytes.size() - 22; ; --i) {
        if (bytes[i] == 'P' && bytes[i + 1] == 'K' && bytes[i + 2] == 5 && bytes[i + 3] == 6) {
            eocd = i;
            break;
        }
        if (i == searchFrom || i == 0) break;
    }
    if (eocd == static_cast<size_t>(-1))
        throw std::runtime_error("PytorchStateDict ZIP: EOCD not found");

    std::uint64_t cdOffset = readU32(bytes, eocd + 16);
    std::uint64_t cdSize = readU32(bytes, eocd + 12);
    std::uint64_t entryCount = readU16(bytes, eocd + 10);

    // ZIP64
    if (cdOffset == 0xffffffffu || cdSize == 0xffffffffu || entryCount == 0xffffu) {
        if (eocd < 20) throw std::runtime_error("PytorchStateDict ZIP: missing ZIP64 locator");
        size_t loc = eocd - 20;
        if (!(bytes[loc] == 'P' && bytes[loc + 1] == 'K' && bytes[loc + 2] == 6 && bytes[loc + 3] == 7))
            throw std::runtime_error("PytorchStateDict ZIP: ZIP64 locator missing");
        const std::uint64_t zip64Eocd = readU64(bytes, loc + 8);
        if (zip64Eocd + 56 > bytes.size())
            throw std::runtime_error("PytorchStateDict ZIP: bad ZIP64 EOCD offset");
        if (!(bytes[zip64Eocd] == 'P' && bytes[zip64Eocd + 1] == 'K'
                && bytes[zip64Eocd + 2] == 6 && bytes[zip64Eocd + 3] == 6))
            throw std::runtime_error("PytorchStateDict ZIP: bad ZIP64 EOCD signature");
        entryCount = readU64(bytes, zip64Eocd + 32);
        cdSize = readU64(bytes, zip64Eocd + 40);
        cdOffset = readU64(bytes, zip64Eocd + 48);
    }

    if (cdOffset > bytes.size() || cdSize > bytes.size() - cdOffset)
        throw std::runtime_error("PytorchStateDict ZIP: bad central directory range");

    ZipArchive zip;
    zip.bytes = bytes;
    size_t off = static_cast<size_t>(cdOffset);
    const size_t cdEnd = static_cast<size_t>(cdOffset + cdSize);
    for (std::uint64_t i = 0; i < entryCount; ++i) {
        if (off + 46 > cdEnd) throw std::runtime_error("PytorchStateDict ZIP: truncated central entry");
        if (!(zip.bytes[off] == 'P' && zip.bytes[off + 1] == 'K'
                && zip.bytes[off + 2] == 1 && zip.bytes[off + 3] == 2))
            throw std::runtime_error("PytorchStateDict ZIP: bad central signature");
        const std::uint16_t method = readU16(zip.bytes, off + 10);
        const std::uint32_t crc = readU32(zip.bytes, off + 16);
        std::uint64_t csize = readU32(zip.bytes, off + 20);
        std::uint64_t usize = readU32(zip.bytes, off + 24);
        const std::uint16_t nlen = readU16(zip.bytes, off + 28);
        const std::uint16_t elen = readU16(zip.bytes, off + 30);
        const std::uint16_t clen = readU16(zip.bytes, off + 32);
        std::uint64_t localOff = readU32(zip.bytes, off + 42);
        if (off + 46 + nlen + elen + clen > cdEnd)
            throw std::runtime_error("PytorchStateDict ZIP: truncated central name/extra");
        ZipEntry entry;
        entry.name.assign(
            reinterpret_cast<const char*>(zip.bytes.data() + off + 46),
            reinterpret_cast<const char*>(zip.bytes.data() + off + 46 + nlen));
        entry.compression = method;
        entry.crc32 = crc;
        entry.compressedSize = csize;
        entry.uncompressedSize = usize;
        entry.localHeaderOffset = localOff;

        // ZIP64 extra in central directory
        size_t extraOff = off + 46 + nlen;
        size_t extraEnd = extraOff + elen;
        bool needZip64 = (csize == 0xffffffffu || usize == 0xffffffffu || localOff == 0xffffffffu);
        if (needZip64) {
            size_t p = extraOff;
            bool found = false;
            while (p + 4 <= extraEnd) {
                const std::uint16_t id = readU16(zip.bytes, p);
                const std::uint16_t sz = readU16(zip.bytes, p + 2);
                if (p + 4 + sz > extraEnd) break;
                if (id == 1) {
                    size_t q = p + 4;
                    if (usize == 0xffffffffu) {
                        entry.uncompressedSize = readU64(zip.bytes, q);
                        q += 8;
                    }
                    if (csize == 0xffffffffu) {
                        entry.compressedSize = readU64(zip.bytes, q);
                        q += 8;
                    }
                    if (localOff == 0xffffffffu) {
                        entry.localHeaderOffset = readU64(zip.bytes, q);
                    }
                    found = true;
                    break;
                }
                p += 4 + sz;
            }
            if (!found)
                throw std::runtime_error("PytorchStateDict ZIP: ZIP64 sizes missing for " + entry.name);
        }

        zip.nameToIndex.emplace(entry.name, zip.entries.size());
        zip.entries.push_back(std::move(entry));
        off += 46 + nlen + elen + clen;
    }
    return zip;
}

std::vector<std::uint8_t> zipRead(const ZipArchive& zip, const std::string& name) {
    const auto it = zip.nameToIndex.find(name);
    if (it == zip.nameToIndex.end())
        throw std::runtime_error("PytorchStateDict ZIP: missing entry " + name);
    const ZipEntry& entry = zip.entries[it->second];
    const size_t local = static_cast<size_t>(entry.localHeaderOffset);
    if (local + 30 > zip.bytes.size())
        throw std::runtime_error("PytorchStateDict ZIP: bad local header offset for " + name);
    if (!(zip.bytes[local] == 'P' && zip.bytes[local + 1] == 'K'
            && zip.bytes[local + 2] == 3 && zip.bytes[local + 3] == 4))
        throw std::runtime_error("PytorchStateDict ZIP: bad local signature for " + name);
    const std::uint16_t nlen = readU16(zip.bytes, local + 26);
    const std::uint16_t elen = readU16(zip.bytes, local + 28);
    const size_t dataOff = local + 30 + nlen + elen;
    if (dataOff > zip.bytes.size()
        || entry.compressedSize > zip.bytes.size() - dataOff)
        throw std::runtime_error("PytorchStateDict ZIP: truncated data for " + name);

    const std::uint8_t* src = zip.bytes.data() + dataOff;
    if (entry.compression == 0) {
        return std::vector<std::uint8_t>(src, src + entry.compressedSize);
    }
    if (entry.compression == 8) {
        std::vector<std::uint8_t> out(static_cast<size_t>(entry.uncompressedSize));
        z_stream strm{};
        strm.next_in = const_cast<Bytef*>(reinterpret_cast<const Bytef*>(src));
        strm.avail_in = static_cast<uInt>(entry.compressedSize);
        strm.next_out = reinterpret_cast<Bytef*>(out.data());
        strm.avail_out = static_cast<uInt>(out.size());
        // raw inflate (-MAX_WBITS) — ZIP deflate has no zlib header
        if (inflateInit2(&strm, -MAX_WBITS) != Z_OK)
            throw std::runtime_error("PytorchStateDict ZIP: inflateInit2 failed");
        const int rc = inflate(&strm, Z_FINISH);
        inflateEnd(&strm);
        if (rc != Z_STREAM_END)
            throw std::runtime_error("PytorchStateDict ZIP: inflate failed for " + name);
        if (strm.total_out != entry.uncompressedSize)
            throw std::runtime_error("PytorchStateDict ZIP: inflate size mismatch for " + name);
        return out;
    }
    throw std::runtime_error(
        "PytorchStateDict ZIP: unsupported compression " + std::to_string(entry.compression)
        + " for " + name);
}

// --- minimal ZIP writer (STORE only, classic 32-bit) ---

void appendU16(std::vector<std::uint8_t>& out, std::uint16_t value) {
    out.push_back(static_cast<std::uint8_t>(value & 0xffu));
    out.push_back(static_cast<std::uint8_t>((value >> 8) & 0xffu));
}

void appendU32(std::vector<std::uint8_t>& out, std::uint32_t value) {
    for (int i = 0; i < 4; ++i)
        out.push_back(static_cast<std::uint8_t>((value >> (8 * i)) & 0xffu));
}

struct ZipWriteItem {
    std::string name;
    std::vector<std::uint8_t> data;
    std::uint32_t crc = 0;
    std::uint32_t localOffset = 0;
};

std::vector<std::uint8_t> buildZipStore(const std::vector<ZipWriteItem>& items) {
    std::vector<ZipWriteItem> planned = items;
    std::vector<std::uint8_t> out;
    for (ZipWriteItem& item : planned) {
        if (item.data.size() > 0xffffffffu)
            throw std::runtime_error("PytorchStateDict ZIP: entry exceeds 4 GiB (ZIP64 write unsupported)");
        // crc32() accepts nullptr only when length is 0.
        const Bytef* dataPtr = item.data.empty()
            ? nullptr
            : reinterpret_cast<const Bytef*>(item.data.data());
        item.crc = static_cast<std::uint32_t>(::crc32(0L, dataPtr, static_cast<uInt>(item.data.size())));
        if (out.size() > 0xffffffffu)
            throw std::runtime_error("PytorchStateDict ZIP: archive exceeds 4 GiB (ZIP64 write unsupported)");
        item.localOffset = static_cast<std::uint32_t>(out.size());
        // local header
        out.push_back('P'); out.push_back('K'); out.push_back(3); out.push_back(4);
        appendU16(out, 20); // version needed
        appendU16(out, 0);  // flags
        appendU16(out, 0);  // store
        appendU16(out, 0);  // time
        appendU16(out, 0);  // date
        appendU32(out, item.crc);
        appendU32(out, static_cast<std::uint32_t>(item.data.size()));
        appendU32(out, static_cast<std::uint32_t>(item.data.size()));
        appendU16(out, static_cast<std::uint16_t>(item.name.size()));
        appendU16(out, 0); // extra
        out.insert(out.end(), item.name.begin(), item.name.end());
        out.insert(out.end(), item.data.begin(), item.data.end());
    }

    const std::uint32_t cdOffset = static_cast<std::uint32_t>(out.size());
    for (const ZipWriteItem& item : planned) {
        out.push_back('P'); out.push_back('K'); out.push_back(1); out.push_back(2);
        appendU16(out, 20); // version made by
        appendU16(out, 20); // version needed
        appendU16(out, 0);
        appendU16(out, 0);
        appendU16(out, 0);
        appendU16(out, 0);
        appendU32(out, item.crc);
        appendU32(out, static_cast<std::uint32_t>(item.data.size()));
        appendU32(out, static_cast<std::uint32_t>(item.data.size()));
        appendU16(out, static_cast<std::uint16_t>(item.name.size()));
        appendU16(out, 0);
        appendU16(out, 0);
        appendU16(out, 0);
        appendU16(out, 0);
        appendU32(out, 0);
        appendU32(out, item.localOffset);
        out.insert(out.end(), item.name.begin(), item.name.end());
    }
    const std::uint32_t cdSize = static_cast<std::uint32_t>(out.size()) - cdOffset;
    out.push_back('P'); out.push_back('K'); out.push_back(5); out.push_back(6);
    appendU16(out, 0);
    appendU16(out, 0);
    appendU16(out, static_cast<std::uint16_t>(planned.size()));
    appendU16(out, static_cast<std::uint16_t>(planned.size()));
    appendU32(out, cdSize);
    appendU32(out, cdOffset);
    appendU16(out, 0);
    return out;
}

// --- pickle state-dict reader ---

enum class StorageDtype { Float32, Float16, BFloat16 };

struct StorageRef {
    StorageDtype dtype = StorageDtype::Float32;
    std::string key;
    std::string location;
    std::int64_t numel = 0;
};

struct TensorValue {
    StorageRef storage;
    std::int64_t storageOffset = 0;
    std::vector<std::int64_t> size;
    std::vector<std::int64_t> stride;
};

struct GlobalRef {
    std::string module;
    std::string name;
};

struct Mark {};

// Tagged pickle stack value. Tuples are heap-backed so Value stays a complete type.
struct Value {
    enum class Kind {
        None,
        Bool,
        Int,
        String,
        Global,
        Storage,
        Tensor,
        Mark,
        Tuple,
        Dict
    };

    using Tuple = std::vector<Value>;
    using Dict = std::unordered_map<std::string, TensorValue>;

    Kind kind = Kind::None;
    bool boolValue = false;
    std::int64_t intValue = 0;
    std::string stringValue;
    GlobalRef globalValue;
    StorageRef storageValue;
    TensorValue tensorValue;
    std::shared_ptr<Tuple> tupleValue;
    std::shared_ptr<Dict> dictValue;

    Value() = default;
    // Named parameter avoids GCC most-vexing-parse on Value(std::monostate).
    Value(std::monostate /*none*/) : kind(Kind::None) {}
    Value(bool value) : kind(Kind::Bool), boolValue(value) {}
    Value(std::int64_t value) : kind(Kind::Int), intValue(value) {}
    Value(std::string value) : kind(Kind::String), stringValue(std::move(value)) {}
    Value(GlobalRef value) : kind(Kind::Global), globalValue(std::move(value)) {}
    Value(StorageRef value) : kind(Kind::Storage), storageValue(std::move(value)) {}
    Value(TensorValue value) : kind(Kind::Tensor), tensorValue(std::move(value)) {}
    Value(Mark) : kind(Kind::Mark) {}
    Value(Tuple value)
        : kind(Kind::Tuple), tupleValue(std::make_shared<Tuple>(std::move(value))) {}
    Value(Dict value)
        : kind(Kind::Dict), dictValue(std::make_shared<Dict>(std::move(value))) {}
};

template <typename T>
bool valueIs(const Value& value);

template <>
inline bool valueIs<std::monostate>(const Value& value) { return value.kind == Value::Kind::None; }
template <>
inline bool valueIs<bool>(const Value& value) { return value.kind == Value::Kind::Bool; }
template <>
inline bool valueIs<std::int64_t>(const Value& value) { return value.kind == Value::Kind::Int; }
template <>
inline bool valueIs<std::string>(const Value& value) { return value.kind == Value::Kind::String; }
template <>
inline bool valueIs<GlobalRef>(const Value& value) { return value.kind == Value::Kind::Global; }
template <>
inline bool valueIs<StorageRef>(const Value& value) { return value.kind == Value::Kind::Storage; }
template <>
inline bool valueIs<TensorValue>(const Value& value) { return value.kind == Value::Kind::Tensor; }
template <>
inline bool valueIs<Mark>(const Value& value) { return value.kind == Value::Kind::Mark; }
template <>
inline bool valueIs<Value::Tuple>(const Value& value) { return value.kind == Value::Kind::Tuple; }
template <>
inline bool valueIs<Value::Dict>(const Value& value) { return value.kind == Value::Kind::Dict; }

template <typename T>
const T& valueAs(const Value& value);
template <typename T>
T& valueAs(Value& value);

template <>
inline const std::string& valueAs<std::string>(const Value& value) { return value.stringValue; }
template <>
inline const GlobalRef& valueAs<GlobalRef>(const Value& value) { return value.globalValue; }
template <>
inline const StorageRef& valueAs<StorageRef>(const Value& value) { return value.storageValue; }
template <>
inline const TensorValue& valueAs<TensorValue>(const Value& value) { return value.tensorValue; }
template <>
inline const Value::Tuple& valueAs<Value::Tuple>(const Value& value) { return *value.tupleValue; }
template <>
inline const Value::Dict& valueAs<Value::Dict>(const Value& value) { return *value.dictValue; }
template <>
inline const std::int64_t& valueAs<std::int64_t>(const Value& value) { return value.intValue; }

template <>
inline std::string& valueAs<std::string>(Value& value) { return value.stringValue; }
template <>
inline GlobalRef& valueAs<GlobalRef>(Value& value) { return value.globalValue; }
template <>
inline StorageRef& valueAs<StorageRef>(Value& value) { return value.storageValue; }
template <>
inline TensorValue& valueAs<TensorValue>(Value& value) { return value.tensorValue; }
template <>
inline Value::Tuple& valueAs<Value::Tuple>(Value& value) { return *value.tupleValue; }
template <>
inline Value::Dict& valueAs<Value::Dict>(Value& value) { return *value.dictValue; }
template <>
inline std::int64_t& valueAs<std::int64_t>(Value& value) { return value.intValue; }

struct PickleParser {
    const std::vector<std::uint8_t>& pkl;
    size_t pos = 0;
    std::vector<Value> stack;
    std::vector<Value> memo;
    const ZipArchive* zip = nullptr;
    std::string archivePrefix; // e.g. "archive/" or "tiny_state/"

    explicit PickleParser(const std::vector<std::uint8_t>& bytes, const ZipArchive* archive, std::string prefix)
        : pkl(bytes), zip(archive), archivePrefix(std::move(prefix)) {}

    std::uint8_t readByte() {
        if (pos >= pkl.size()) throw std::runtime_error("PytorchStateDict pickle: truncated");
        return pkl[pos++];
    }

    std::string readLineAscii() {
        std::string out;
        while (pos < pkl.size()) {
            const char ch = static_cast<char>(pkl[pos++]);
            if (ch == '\n') break;
            out.push_back(ch);
        }
        return out;
    }

    std::int64_t readInt1() { return static_cast<std::int64_t>(readByte()); }

    std::int64_t readInt2() {
        const std::uint16_t value = static_cast<std::uint16_t>(readByte())
            | (static_cast<std::uint16_t>(readByte()) << 8);
        return static_cast<std::int64_t>(value);
    }

    std::int64_t readInt4() {
        std::int32_t value = 0;
        std::memcpy(&value, pkl.data() + pos, 4);
        pos += 4;
        return static_cast<std::int64_t>(value);
    }

    std::string readBinUnicode() {
        if (pos + 4 > pkl.size()) throw std::runtime_error("PytorchStateDict pickle: bad BINUNICODE");
        const std::uint32_t len = readU32(pkl, pos);
        pos += 4;
        if (pos + len > pkl.size()) throw std::runtime_error("PytorchStateDict pickle: truncated unicode");
        std::string out(reinterpret_cast<const char*>(pkl.data() + pos), len);
        pos += len;
        return out;
    }

    void memoPut(std::uint32_t index, Value value) {
        if (memo.size() <= index)
            memo.resize(static_cast<size_t>(index) + 1);
        memo[index] = std::move(value);
    }

    Value memoGet(std::uint32_t index) const {
        if (index >= memo.size())
            throw std::runtime_error("PytorchStateDict pickle: bad memo get");
        return memo[index];
    }

    Value pop() {
        if (stack.empty()) throw std::runtime_error("PytorchStateDict pickle: stack underflow");
        Value value = std::move(stack.back());
        stack.pop_back();
        return value;
    }

    void push(Value value) { stack.push_back(std::move(value)); }

    std::vector<Value> popMark() {
        std::vector<Value> items;
        while (!stack.empty()) {
            if (valueIs<Mark>(stack.back())) {
                stack.pop_back();
                std::reverse(items.begin(), items.end());
                return items;
            }
            items.push_back(pop());
        }
        throw std::runtime_error("PytorchStateDict pickle: MARK missing");
    }

    StorageRef storageFromPersisted(const Value& value) {
        // Persisted tuple was pushed as vector: ["storage", GlobalRef, key, location, numel]
        if (!valueIs<Value::Tuple>(value))
            throw std::runtime_error("PytorchStateDict pickle: BINPERSID expected tuple");
        const auto& tup = valueAs<Value::Tuple>(value);
        if (tup.size() != 5)
            throw std::runtime_error("PytorchStateDict pickle: storage persistent id arity");
        if (!valueIs<std::string>(tup[0]) || valueAs<std::string>(tup[0]) != "storage")
            throw std::runtime_error("PytorchStateDict pickle: expected storage persistent id");
        if (!valueIs<GlobalRef>(tup[1]))
            throw std::runtime_error("PytorchStateDict pickle: expected storage type global");
        const GlobalRef& type = valueAs<GlobalRef>(tup[1]);
        StorageRef storage;
        if (type.module == "torch" && type.name == "FloatStorage")
            storage.dtype = StorageDtype::Float32;
        else if (type.module == "torch" && type.name == "HalfStorage")
            storage.dtype = StorageDtype::Float16;
        else if (type.module == "torch" && type.name == "BFloat16Storage")
            storage.dtype = StorageDtype::BFloat16;
        else
            throw std::runtime_error(
                "PytorchStateDict: unsupported storage type " + type.module + " " + type.name);
        if (!valueIs<std::string>(tup[2]) || !valueIs<std::string>(tup[3]))
            throw std::runtime_error("PytorchStateDict pickle: storage key/location must be strings");
        if (!valueIs<std::int64_t>(tup[4]))
            throw std::runtime_error("PytorchStateDict pickle: storage numel must be int");
        storage.key = valueAs<std::string>(tup[2]);
        storage.location = valueAs<std::string>(tup[3]);
        storage.numel = valueAs<std::int64_t>(tup[4]);
        if (storage.numel < 0)
            throw std::runtime_error("PytorchStateDict pickle: negative numel");
        return storage;
    }

    TensorValue rebuildTensor(const std::vector<Value>& args) {
        // (storage, offset, size, stride, requires_grad, backward_hooks[, metadata])
        if (args.size() < 6)
            throw std::runtime_error("PytorchStateDict pickle: _rebuild_tensor_v2 arity");
        TensorValue tensor;
        if (valueIs<StorageRef>(args[0]))
            tensor.storage = valueAs<StorageRef>(args[0]);
        else
            throw std::runtime_error("PytorchStateDict pickle: tensor storage missing");
        if (!valueIs<std::int64_t>(args[1]))
            throw std::runtime_error("PytorchStateDict pickle: storage offset must be int");
        tensor.storageOffset = valueAs<std::int64_t>(args[1]);
        if (!valueIs<Value::Tuple>(args[2]) || !valueIs<Value::Tuple>(args[3]))
            throw std::runtime_error("PytorchStateDict pickle: size/stride must be tuples");
        for (const Value& dim : valueAs<Value::Tuple>(args[2])) {
            if (!valueIs<std::int64_t>(dim))
                throw std::runtime_error("PytorchStateDict pickle: size dim must be int");
            tensor.size.push_back(valueAs<std::int64_t>(dim));
        }
        for (const Value& dim : valueAs<Value::Tuple>(args[3])) {
            if (!valueIs<std::int64_t>(dim))
                throw std::runtime_error("PytorchStateDict pickle: stride dim must be int");
            tensor.stride.push_back(valueAs<std::int64_t>(dim));
        }
        if (tensor.size.size() != tensor.stride.size())
            throw std::runtime_error("PytorchStateDict pickle: size/stride rank mismatch");
        if (tensor.size.size() > 2)
            throw std::runtime_error("PytorchStateDict: only 1D/2D tensors supported");
        return tensor;
    }

    std::unordered_map<std::string, TensorValue> parse() {
        while (pos < pkl.size()) {
            const std::uint8_t op = readByte();
            switch (op) {
            case 0x80: { // PROTO
                (void)readByte();
                break;
            }
            case '}': { // EMPTY_DICT
                push(std::unordered_map<std::string, TensorValue>{});
                break;
            }
            case ')': { // EMPTY_TUPLE
                push(std::vector<Value>{});
                break;
            }
            case '(': { // MARK
                push(Mark{});
                break;
            }
            case 'X': { // BINUNICODE
                push(readBinUnicode());
                break;
            }
            case 'q': { // BINPUT
                const std::uint8_t index = readByte();
                if (stack.empty()) throw std::runtime_error("PytorchStateDict pickle: BINPUT empty stack");
                memoPut(index, stack.back());
                break;
            }
            case 'r': { // LONG_BINPUT
                if (pos + 4 > pkl.size()) throw std::runtime_error("PytorchStateDict pickle: bad LONG_BINPUT");
                const std::uint32_t index = readU32(pkl, pos);
                pos += 4;
                if (stack.empty()) throw std::runtime_error("PytorchStateDict pickle: LONG_BINPUT empty stack");
                memoPut(index, stack.back());
                break;
            }
            case 'h': { // BINGET
                push(memoGet(readByte()));
                break;
            }
            case 'j': { // LONG_BINGET
                if (pos + 4 > pkl.size()) throw std::runtime_error("PytorchStateDict pickle: bad LONG_BINGET");
                const std::uint32_t index = readU32(pkl, pos);
                pos += 4;
                push(memoGet(index));
                break;
            }
            case 'c': { // GLOBAL
                const std::string module = readLineAscii();
                const std::string name = readLineAscii();
                push(GlobalRef{ module, name });
                break;
            }
            case 'K': { // BININT1
                push(readInt1());
                break;
            }
            case 'M': { // BININT2
                push(readInt2());
                break;
            }
            case 'J': { // BININT
                if (pos + 4 > pkl.size()) throw std::runtime_error("PytorchStateDict pickle: bad BININT");
                push(readInt4());
                break;
            }
            case 't': { // TUPLE
                push(popMark());
                break;
            }
            case 0x85: { // TUPLE1
                std::vector<Value> tup;
                tup.push_back(pop());
                push(std::move(tup));
                break;
            }
            case 0x86: { // TUPLE2
                Value b = pop();
                Value a = pop();
                std::vector<Value> tup;
                tup.push_back(std::move(a));
                tup.push_back(std::move(b));
                push(std::move(tup));
                break;
            }
            case 0x87: { // TUPLE3
                Value c = pop();
                Value b = pop();
                Value a = pop();
                std::vector<Value> tup;
                tup.push_back(std::move(a));
                tup.push_back(std::move(b));
                tup.push_back(std::move(c));
                push(std::move(tup));
                break;
            }
            case 'Q': { // BINPERSID
                Value pid = pop();
                push(storageFromPersisted(pid));
                break;
            }
            case 0x89: { // NEWFALSE
                push(false);
                break;
            }
            case 0x88: { // NEWTRUE
                push(true);
                break;
            }
            case 'N': { // NONE
                push(std::monostate{});
                break;
            }
            case 'R': { // REDUCE
                Value argValue = pop();
                Value callable = pop();
                if (!valueIs<GlobalRef>(callable))
                    throw std::runtime_error("PytorchStateDict pickle: REDUCE non-global");
                const GlobalRef& global = valueAs<GlobalRef>(callable);
                if (global.module == "collections" && global.name == "OrderedDict") {
                    // empty OrderedDict → empty tensor map (top-level or hooks)
                    push(Value::Dict{});
                    break;
                }
                if (global.module == "torch._utils" && global.name == "_rebuild_tensor_v2") {
                    if (!valueIs<Value::Tuple>(argValue))
                        throw std::runtime_error("PytorchStateDict pickle: tensor args must be tuple");
                    push(rebuildTensor(valueAs<Value::Tuple>(argValue)));
                    break;
                }
                throw std::runtime_error(
                    "PytorchStateDict pickle: unsupported REDUCE " + global.module + " " + global.name);
            }
            case 's': { // SETITEM
                Value value = pop();
                Value key = pop();
                if (stack.empty()) throw std::runtime_error("PytorchStateDict pickle: SETITEM empty stack");
                Value& dictValue = stack.back();
                if (!valueIs<Value::Dict>(dictValue))
                    throw std::runtime_error("PytorchStateDict pickle: SETITEM on non-dict");
                if (!valueIs<std::string>(key) || !valueIs<TensorValue>(value))
                    throw std::runtime_error("PytorchStateDict pickle: SETITEM expects str→tensor");
                valueAs<Value::Dict>(dictValue).emplace(
                    valueAs<std::string>(key),
                    valueAs<TensorValue>(value));
                break;
            }
            case 'u': { // SETITEMS
                std::vector<Value> items = popMark();
                if (items.size() % 2 != 0)
                    throw std::runtime_error("PytorchStateDict pickle: SETITEMS odd count");
                if (stack.empty()) throw std::runtime_error("PytorchStateDict pickle: SETITEMS empty stack");
                Value& dictValue = stack.back();
                if (!valueIs<Value::Dict>(dictValue))
                    throw std::runtime_error("PytorchStateDict pickle: SETITEMS on non-dict");
                auto& dict = valueAs<Value::Dict>(dictValue);
                for (size_t i = 0; i < items.size(); i += 2) {
                    if (!valueIs<std::string>(items[i]) || !valueIs<TensorValue>(items[i + 1]))
                        throw std::runtime_error("PytorchStateDict pickle: SETITEMS expects str→tensor");
                    dict[valueAs<std::string>(items[i])] = valueAs<TensorValue>(items[i + 1]);
                }
                break;
            }
            case '.': { // STOP
                if (stack.size() != 1)
                    throw std::runtime_error("PytorchStateDict pickle: STOP stack size");
                Value root = pop();
                if (!valueIs<Value::Dict>(root))
                    throw std::runtime_error("PytorchStateDict: root object is not a state-dict mapping");
                return valueAs<Value::Dict>(root);
            }
            default:
                throw std::runtime_error(
                    "PytorchStateDict pickle: unsupported opcode "
                    + std::to_string(static_cast<unsigned>(op)));
            }
        }
        throw std::runtime_error("PytorchStateDict pickle: missing STOP");
    }

    Matrix materialize(const TensorValue& tensor) const {
        if (zip == nullptr)
            throw std::logic_error("PytorchStateDict: zip missing during materialize");
        if (tensor.storageOffset < 0)
            throw std::runtime_error("PytorchStateDict: negative storage offset");
        if (tensor.size.empty())
            throw std::runtime_error("PytorchStateDict: scalar tensors unsupported");

        size_t elementBytes = 4;
        if (tensor.storage.dtype == StorageDtype::Float16 || tensor.storage.dtype == StorageDtype::BFloat16)
            elementBytes = 2;

        const std::string storagePath = archivePrefix + "data/" + tensor.storage.key;
        const std::vector<std::uint8_t> raw = zipRead(*zip, storagePath);
        const std::uint64_t expectedBytes =
            static_cast<std::uint64_t>(tensor.storage.numel) * static_cast<std::uint64_t>(elementBytes);
        if (raw.size() < expectedBytes)
            throw std::runtime_error("PytorchStateDict: storage blob shorter than numel for " + storagePath);

        // Contiguous row-major only.
        if (tensor.size.size() == 1) {
            if (tensor.stride.size() != 1 || tensor.stride[0] != 1)
                throw std::runtime_error("PytorchStateDict: non-contiguous 1D tensor");
        } else if (tensor.size.size() == 2) {
            if (tensor.stride.size() != 2
                || tensor.stride[0] != tensor.size[1]
                || tensor.stride[1] != 1)
                throw std::runtime_error("PytorchStateDict: non-contiguous 2D tensor (need row-major)");
        }

        std::int64_t numel = 1;
        for (std::int64_t dim : tensor.size) {
            if (dim < 0) throw std::runtime_error("PytorchStateDict: negative dim");
            numel *= dim;
        }
        if (tensor.storageOffset + numel > tensor.storage.numel)
            throw std::runtime_error("PytorchStateDict: tensor view exceeds storage");

        const size_t rows = static_cast<size_t>(tensor.size[0]);
        const size_t cols = tensor.size.size() == 1 ? 1 : static_cast<size_t>(tensor.size[1]);
        Matrix matrix(rows, cols, 0.0f);
        const size_t base = static_cast<size_t>(tensor.storageOffset) * elementBytes;
        for (size_t i = 0; i < static_cast<size_t>(numel); ++i) {
            const size_t byteIndex = base + i * elementBytes;
            float value = 0.0f;
            if (tensor.storage.dtype == StorageDtype::Float32) {
                std::memcpy(&value, raw.data() + byteIndex, 4);
            } else if (tensor.storage.dtype == StorageDtype::Float16) {
                std::uint16_t bits = 0;
                std::memcpy(&bits, raw.data() + byteIndex, 2);
                value = floatFromF16Bits(bits);
            } else {
                std::uint16_t bits = 0;
                std::memcpy(&bits, raw.data() + byteIndex, 2);
                value = floatFromBf16Bits(bits);
            }
            matrix.data[i] = value;
        }
        return matrix;
    }
};

std::string detectArchivePrefix(const ZipArchive& zip) {
    // Prefer "<prefix>/data.pkl"
    for (const ZipEntry& entry : zip.entries) {
        const std::string& name = entry.name;
        if (name.size() >= 8 && name.compare(name.size() - 8, 8, "data.pkl") == 0) {
            const size_t slash = name.find_last_of('/');
            if (slash == std::string::npos) return {};
            return name.substr(0, slash + 1);
        }
    }
    throw std::runtime_error("PytorchStateDict: data.pkl not found in archive");
}

// --- pickle writer (F32 only) ---

struct PickleWriter {
    std::vector<std::uint8_t> out;
    std::uint32_t memo = 0;
    std::uint32_t rebuildMemo = 0;
    std::uint32_t storageStrMemo = 0;
    std::uint32_t floatStorageMemo = 0;
    std::uint32_t cpuMemo = 0;
    std::uint32_t orderedDictMemo = 0;
    bool hasRebuild = false;
    bool hasStorageStr = false;
    bool hasFloatStorage = false;
    bool hasCpu = false;
    bool hasOrderedDict = false;

    void emit(std::uint8_t byte) { out.push_back(byte); }

    void emitBinUnicode(const std::string& text) {
        emit('X');
        appendU32(out, static_cast<std::uint32_t>(text.size()));
        out.insert(out.end(), text.begin(), text.end());
    }

    void emitBinput() {
        if (memo < 256) {
            emit('q');
            emit(static_cast<std::uint8_t>(memo));
        } else {
            emit('r');
            appendU32(out, memo);
        }
        ++memo;
    }

    void emitBinget(std::uint32_t index) {
        if (index < 256) {
            emit('h');
            emit(static_cast<std::uint8_t>(index));
        } else {
            emit('j');
            appendU32(out, index);
        }
    }

    void emitInt(std::int64_t value) {
        if (value >= 0 && value < 256) {
            emit('K');
            emit(static_cast<std::uint8_t>(value));
        } else if (value >= 0 && value < 65536) {
            emit('M');
            appendU16(out, static_cast<std::uint16_t>(value));
        } else if (value >= INT32_MIN && value <= INT32_MAX) {
            emit('J');
            const std::int32_t narrow = static_cast<std::int32_t>(value);
            std::uint8_t bytes[4];
            std::memcpy(bytes, &narrow, 4);
            out.insert(out.end(), bytes, bytes + 4);
        } else {
            throw std::runtime_error("PytorchStateDict save: integer out of BININT range");
        }
    }

    void emitGlobal(const char* module, const char* name) {
        emit('c');
        const std::string line1 = std::string(module) + "\n";
        const std::string line2 = std::string(name) + "\n";
        out.insert(out.end(), line1.begin(), line1.end());
        out.insert(out.end(), line2.begin(), line2.end());
    }

    void emitSizeTuple(const std::vector<std::int64_t>& dims) {
        if (dims.size() == 1) {
            emitInt(dims[0]);
            emit(0x85); // TUPLE1
            emitBinput();
        } else if (dims.size() == 2) {
            emitInt(dims[0]);
            emitInt(dims[1]);
            emit(0x86); // TUPLE2
            emitBinput();
        } else {
            emit('(');
            for (std::int64_t dim : dims)
                emitInt(dim);
            emit('t');
            emitBinput();
        }
    }

    void emitTensor(const std::string& storageKey, std::int64_t numel, const std::vector<std::int64_t>& shape) {
        if (!hasRebuild) {
            emitGlobal("torch._utils", "_rebuild_tensor_v2");
            rebuildMemo = memo;
            emitBinput();
            hasRebuild = true;
        } else {
            emitBinget(rebuildMemo);
        }

        emit('('); // args MARK
        emit('('); // persid MARK
        if (!hasStorageStr) {
            emitBinUnicode("storage");
            storageStrMemo = memo;
            emitBinput();
            hasStorageStr = true;
        } else {
            emitBinget(storageStrMemo);
        }
        if (!hasFloatStorage) {
            emitGlobal("torch", "FloatStorage");
            floatStorageMemo = memo;
            emitBinput();
            hasFloatStorage = true;
        } else {
            emitBinget(floatStorageMemo);
        }
        emitBinUnicode(storageKey);
        emitBinput();
        if (!hasCpu) {
            emitBinUnicode("cpu");
            cpuMemo = memo;
            emitBinput();
            hasCpu = true;
        } else {
            emitBinget(cpuMemo);
        }
        emitInt(numel);
        emit('t');
        emitBinput();
        emit('Q'); // BINPERSID

        emitInt(0); // storage offset
        emitSizeTuple(shape);
        // contiguous stride
        if (shape.size() == 1)
            emitSizeTuple({ 1 });
        else
            emitSizeTuple({ shape[1], 1 });

        emit(0x89); // NEWFALSE requires_grad
        if (!hasOrderedDict) {
            emitGlobal("collections", "OrderedDict");
            orderedDictMemo = memo;
            emitBinput();
            hasOrderedDict = true;
        } else {
            emitBinget(orderedDictMemo);
        }
        emit(')'); // EMPTY_TUPLE
        emit('R'); // REDUCE empty OD
        emitBinput();

        emit('t'); // args tuple
        emitBinput();
        emit('R'); // REDUCE tensor
        emitBinput();
    }

    std::vector<std::uint8_t> build(const std::vector<std::pair<std::string, Matrix>>& tensors) {
        out.clear();
        memo = 0;
        hasRebuild = hasStorageStr = hasFloatStorage = hasCpu = hasOrderedDict = false;
        emit(0x80);
        emit(2);
        // Match torch.save: collections.OrderedDict() then SETITEMS.
        emitGlobal("collections", "OrderedDict");
        orderedDictMemo = memo;
        emitBinput(); // memo 0 = OrderedDict class (also used for empty hooks)
        hasOrderedDict = true;
        emit(')'); // EMPTY_TUPLE
        emit('R'); // OrderedDict()
        emitBinput(); // memo 1 = empty root mapping
        emit('('); // SETITEMS mark
        std::uint32_t storageIndex = 0;
        for (const auto& [name, matrix] : tensors) {
            emitBinUnicode(name);
            emitBinput();
            const std::int64_t rows = static_cast<std::int64_t>(matrix.rows);
            const std::int64_t cols = static_cast<std::int64_t>(matrix.cols);
            const std::int64_t numel = rows * cols;
            std::vector<std::int64_t> shape;
            if (cols == 1)
                shape = { rows };
            else
                shape = { rows, cols };
            emitTensor(std::to_string(storageIndex), numel, shape);
            ++storageIndex;
        }
        emit('u'); // SETITEMS
        emit('.');
        return out;
    }
};

std::string archiveStemFromPath(const std::string& path) {
    fs::path p(path);
    std::string stem = p.stem().string();
    if (stem.empty()) stem = "archive";
    for (char& ch : stem) {
        if (!(std::isalnum(static_cast<unsigned char>(ch)) || ch == '_' || ch == '-' || ch == '.'))
            ch = '_';
    }
    return stem;
}

} // namespace

bool isPytorchZipFile(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    char magic[4] = {};
    in.read(magic, 4);
    return static_cast<bool>(in) && magic[0] == 'P' && magic[1] == 'K' && magic[2] == 3 && magic[3] == 4;
}

SafeTensors::File load(const std::string& path) {
    if (path.empty()) throw std::invalid_argument("PytorchStateDict::load empty path");
    const std::vector<std::uint8_t> bytes = readEntireFileBytes(path);
    if (bytes.size() >= 2 && bytes[0] == 0x80) {
        throw std::runtime_error(
            "PytorchStateDict::load legacy non-zip pickle is unsupported (" + path
            + "). Re-save with PyTorch ≥ 1.6 (zip format) or convert to safetensors.");
    }
    ZipArchive zip = openZip(bytes);
    const std::string prefix = detectArchivePrefix(zip);
    const std::vector<std::uint8_t> pkl = zipRead(zip, prefix + "data.pkl");
    PickleParser parser(pkl, &zip, prefix);
    const auto tensors = parser.parse();

    SafeTensors::File out;
    out.metadata["format"] = "pt";
    for (const auto& [name, tensor] : tensors)
        SafeTensors::putMatrix(out, name, parser.materialize(tensor));
    if (out.tensors.empty())
        throw std::runtime_error("PytorchStateDict::load empty state dict in " + path);
    return out;
}

void save(const std::string& path, const SafeTensors::File& file) {
    if (path.empty()) throw std::invalid_argument("PytorchStateDict::save empty path");
    if (file.tensors.empty()) throw std::invalid_argument("PytorchStateDict::save no tensors");

    std::vector<std::pair<std::string, Matrix>> ordered;
    ordered.reserve(file.tensors.size());
    for (const auto& entry : file.tensors) // std::map → sorted, stable
        ordered.push_back(entry);

    PickleWriter writer;
    const std::vector<std::uint8_t> pkl = writer.build(ordered);
    const std::string stem = archiveStemFromPath(path);

    std::vector<ZipWriteItem> items;
    items.push_back(ZipWriteItem{ stem + "/data.pkl", pkl, 0, 0 });
    items.push_back(ZipWriteItem{ stem + "/byteorder", std::vector<std::uint8_t>{ 'l','i','t','t','l','e' }, 0, 0 });
    for (size_t i = 0; i < ordered.size(); ++i) {
        const Matrix& matrix = ordered[i].second;
        std::vector<std::uint8_t> raw(matrix.data.size() * sizeof(float));
        if (!raw.empty())
            std::memcpy(raw.data(), matrix.data.data(), raw.size());
        items.push_back(ZipWriteItem{ stem + "/data/" + std::to_string(i), std::move(raw), 0, 0 });
    }
    items.push_back(ZipWriteItem{ stem + "/version", std::vector<std::uint8_t>{ '3', '\n' }, 0, 0 });

    writeEntireFileBytes(path, buildZipStore(items));
}

void runSmokeDemo() {
    SafeTensors::File original;
    SafeTensors::putMatrix(original, "model.embed_tokens.weight", Matrix(2, 3, 0.0f));
    original.tensors["model.embed_tokens.weight"].data = { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f };
    SafeTensors::putMatrix(original, "model.norm.weight", Matrix(3, 1, 0.0f));
    original.tensors["model.norm.weight"].data = { 0.25f, 0.5f, 0.75f };

    const std::string path = "pytorch_state_dict_smoke.bin";
    PytorchStateDict::save(path, original);
    if (!isPytorchZipFile(path))
        throw std::runtime_error("PytorchStateDict smoke: isPytorchZipFile failed");

    const SafeTensors::File loaded = PytorchStateDict::load(path);
    if (loaded.tensors.size() != 2)
        throw std::runtime_error("PytorchStateDict smoke: tensor count mismatch");
    const Matrix& emb = loaded.tensors.at("model.embed_tokens.weight");
    const Matrix& norm = loaded.tensors.at("model.norm.weight");
    if (emb.rows != 2 || emb.cols != 3 || norm.rows != 3 || norm.cols != 1)
        throw std::runtime_error("PytorchStateDict smoke: shape mismatch");
    if (std::fabs(emb.data[0] - 1.f) > 1e-6f || std::fabs(emb.data[5] - 6.f) > 1e-6f
        || std::fabs(norm.data[2] - 0.75f) > 1e-6f)
        throw std::runtime_error("PytorchStateDict smoke: value mismatch");

    fs::remove(path);

    // Optional: load a torch.save fixture (F32/F16/BF16) pointed at by env.
    const char* fixture = std::getenv("SENTINEL_PYTORCH_BIN_FIXTURE");
    if (fixture != nullptr && fixture[0] != '\0') {
        const SafeTensors::File fromTorch = PytorchStateDict::load(fixture);
        if (fromTorch.tensors.empty())
            throw std::runtime_error("PytorchStateDict smoke: empty torch fixture");
        for (const auto& [name, matrix] : fromTorch.tensors) {
            if (matrix.empty())
                throw std::runtime_error("PytorchStateDict smoke: empty tensor in fixture " + name);
            for (float value : matrix.data) {
                if (!std::isfinite(value))
                    throw std::runtime_error("PytorchStateDict smoke: non-finite in fixture " + name);
            }
        }
        SmokeLog::result(
            "PytorchStateDict",
            "zip-save/load=ok  tensors=2  F32=ok  torchFixture=%zu",
            fromTorch.tensors.size());
        return;
    }

    SmokeLog::result("PytorchStateDict", "zip-save/load=ok  tensors=2  F32=ok");
}

} // namespace PytorchStateDict
