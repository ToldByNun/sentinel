#include "SafeTensors.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace SafeTensors {
namespace {

constexpr size_t kMaxHeaderBytes = 100ull * 1024ull * 1024ull; // 100 MiB guard

void writeExact(std::ostream& out, const void* data, size_t bytes) {
    out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(bytes));
    if (!out) throw std::runtime_error("SafeTensors write failed");
}

void readExact(std::istream& in, void* data, size_t bytes) {
    in.read(reinterpret_cast<char*>(data), static_cast<std::streamsize>(bytes));
    if (!in || static_cast<size_t>(in.gcount()) != bytes)
        throw std::runtime_error("SafeTensors read failed");
}

std::string jsonEscape(const std::string& text) {
    std::string out;
    out.reserve(text.size() + 8);
    for (unsigned char ch : text) {
        switch (ch) {
        case '\\': out += "\\\\"; break;
        case '"': out += "\\\""; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default:
            if (ch < 0x20) {
                char buf[8];
                std::snprintf(buf, sizeof(buf), "\\u%04x", ch);
                out += buf;
            } else {
                out.push_back(static_cast<char>(ch));
            }
            break;
        }
    }
    return out;
}

size_t elementCount(const std::vector<size_t>& shape) {
    size_t count = 1;
    if (shape.empty()) return 1; // scalar
    for (size_t dim : shape) {
        if (dim == 0) return 0;
        count *= dim;
    }
    return count;
}

struct ParsedTensor {
    std::string dtype;
    std::vector<size_t> shape;
    size_t begin = 0;
    size_t end = 0;
};

void skipWs(const std::string& json, size_t& i) {
    while (i < json.size() && (json[i] == ' ' || json[i] == '\n' || json[i] == '\r' || json[i] == '\t'))
        ++i;
}

std::string parseString(const std::string& json, size_t& i) {
    skipWs(json, i);
    if (i >= json.size() || json[i] != '"')
        throw std::runtime_error("SafeTensors header: expected string");
    ++i;
    std::string out;
    while (i < json.size()) {
        const char ch = json[i++];
        if (ch == '"') return out;
        if (ch == '\\') {
            if (i >= json.size()) throw std::runtime_error("SafeTensors header: bad escape");
            const char esc = json[i++];
            switch (esc) {
            case '"': out.push_back('"'); break;
            case '\\': out.push_back('\\'); break;
            case '/': out.push_back('/'); break;
            case 'n': out.push_back('\n'); break;
            case 'r': out.push_back('\r'); break;
            case 't': out.push_back('\t'); break;
            case 'u': {
                if (i + 4 > json.size()) throw std::runtime_error("SafeTensors header: bad unicode escape");
                unsigned code = 0;
                for (int n = 0; n < 4; ++n) {
                    const char h = json[i++];
                    code <<= 4;
                    if (h >= '0' && h <= '9') code |= static_cast<unsigned>(h - '0');
                    else if (h >= 'a' && h <= 'f') code |= static_cast<unsigned>(h - 'a' + 10);
                    else if (h >= 'A' && h <= 'F') code |= static_cast<unsigned>(h - 'A' + 10);
                    else throw std::runtime_error("SafeTensors header: bad unicode escape");
                }
                if (code > 0x7F) throw std::runtime_error("SafeTensors header: non-ASCII unicode not supported");
                out.push_back(static_cast<char>(code));
                break;
            }
            default: throw std::runtime_error("SafeTensors header: unsupported escape");
            }
        } else {
            out.push_back(ch);
        }
    }
    throw std::runtime_error("SafeTensors header: unterminated string");
}

std::uint64_t parseUint(const std::string& json, size_t& i) {
    skipWs(json, i);
    if (i >= json.size() || json[i] < '0' || json[i] > '9')
        throw std::runtime_error("SafeTensors header: expected number");
    std::uint64_t value = 0;
    while (i < json.size() && json[i] >= '0' && json[i] <= '9') {
        value = value * 10ull + static_cast<std::uint64_t>(json[i] - '0');
        ++i;
    }
    return value;
}

std::vector<size_t> parseShapeArray(const std::string& json, size_t& i) {
    skipWs(json, i);
    if (i >= json.size() || json[i] != '[')
        throw std::runtime_error("SafeTensors header: expected shape array");
    ++i;
    std::vector<size_t> shape;
    skipWs(json, i);
    if (i < json.size() && json[i] == ']') {
        ++i;
        return shape;
    }
    while (true) {
        shape.push_back(static_cast<size_t>(parseUint(json, i)));
        skipWs(json, i);
        if (i >= json.size()) throw std::runtime_error("SafeTensors header: truncated shape");
        if (json[i] == ']') {
            ++i;
            break;
        }
        if (json[i] != ',') throw std::runtime_error("SafeTensors header: expected ',' in shape");
        ++i;
    }
    return shape;
}

std::pair<size_t, size_t> parseOffsets(const std::string& json, size_t& i) {
    skipWs(json, i);
    if (i >= json.size() || json[i] != '[')
        throw std::runtime_error("SafeTensors header: expected data_offsets array");
    ++i;
    const size_t begin = static_cast<size_t>(parseUint(json, i));
    skipWs(json, i);
    if (i >= json.size() || json[i] != ',')
        throw std::runtime_error("SafeTensors header: expected ',' in data_offsets");
    ++i;
    const size_t end = static_cast<size_t>(parseUint(json, i));
    skipWs(json, i);
    if (i >= json.size() || json[i] != ']')
        throw std::runtime_error("SafeTensors header: expected ']' in data_offsets");
    ++i;
    return { begin, end };
}

std::map<std::string, std::string> parseMetadataObject(const std::string& json, size_t& i) {
    skipWs(json, i);
    if (i >= json.size() || json[i] != '{')
        throw std::runtime_error("SafeTensors header: expected metadata object");
    ++i;
    std::map<std::string, std::string> meta;
    skipWs(json, i);
    if (i < json.size() && json[i] == '}') {
        ++i;
        return meta;
    }
    while (true) {
        const std::string key = parseString(json, i);
        skipWs(json, i);
        if (i >= json.size() || json[i] != ':')
            throw std::runtime_error("SafeTensors header: expected ':' in metadata");
        ++i;
        const std::string value = parseString(json, i);
        meta[key] = value;
        skipWs(json, i);
        if (i >= json.size()) throw std::runtime_error("SafeTensors header: truncated metadata");
        if (json[i] == '}') {
            ++i;
            break;
        }
        if (json[i] != ',') throw std::runtime_error("SafeTensors header: expected ',' in metadata");
        ++i;
    }
    return meta;
}

ParsedTensor parseTensorObject(const std::string& json, size_t& i) {
    skipWs(json, i);
    if (i >= json.size() || json[i] != '{')
        throw std::runtime_error("SafeTensors header: expected tensor object");
    ++i;
    ParsedTensor tensor;
    bool haveDtype = false;
    bool haveShape = false;
    bool haveOffsets = false;
    skipWs(json, i);
    if (i < json.size() && json[i] == '}')
        throw std::runtime_error("SafeTensors header: empty tensor object");
    while (true) {
        const std::string key = parseString(json, i);
        skipWs(json, i);
        if (i >= json.size() || json[i] != ':')
            throw std::runtime_error("SafeTensors header: expected ':' in tensor");
        ++i;
        if (key == "dtype") {
            tensor.dtype = parseString(json, i);
            haveDtype = true;
        } else if (key == "shape") {
            tensor.shape = parseShapeArray(json, i);
            haveShape = true;
        } else if (key == "data_offsets") {
            const auto offsets = parseOffsets(json, i);
            tensor.begin = offsets.first;
            tensor.end = offsets.second;
            haveOffsets = true;
        } else {
            // Skip unknown values that are strings/arrays/objects/numbers — keep parser strict for common cases.
            skipWs(json, i);
            if (i < json.size() && json[i] == '"') {
                (void)parseString(json, i);
            } else if (i < json.size() && json[i] == '[') {
                (void)parseShapeArray(json, i);
            } else if (i < json.size() && (json[i] >= '0' && json[i] <= '9')) {
                (void)parseUint(json, i);
            } else {
                throw std::runtime_error("SafeTensors header: unsupported field value for key " + key);
            }
        }
        skipWs(json, i);
        if (i >= json.size()) throw std::runtime_error("SafeTensors header: truncated tensor object");
        if (json[i] == '}') {
            ++i;
            break;
        }
        if (json[i] != ',') throw std::runtime_error("SafeTensors header: expected ',' in tensor object");
        ++i;
    }
    if (!haveDtype || !haveShape || !haveOffsets)
        throw std::runtime_error("SafeTensors header: tensor missing dtype/shape/data_offsets");
    return tensor;
}

void parseHeader(
    const std::string& json,
    std::map<std::string, std::string>& metadata,
    std::map<std::string, ParsedTensor>& tensors) {
    size_t i = 0;
    skipWs(json, i);
    if (i >= json.size() || json[i] != '{')
        throw std::runtime_error("SafeTensors header must start with '{'");
    ++i;
    skipWs(json, i);
    if (i < json.size() && json[i] == '}') return;

    while (true) {
        const std::string key = parseString(json, i);
        skipWs(json, i);
        if (i >= json.size() || json[i] != ':')
            throw std::runtime_error("SafeTensors header: expected ':'");
        ++i;
        if (key == "__metadata__") {
            metadata = parseMetadataObject(json, i);
        } else {
            tensors[key] = parseTensorObject(json, i);
        }
        skipWs(json, i);
        if (i >= json.size()) throw std::runtime_error("SafeTensors header: truncated root");
        if (json[i] == '}') break;
        if (json[i] != ',') throw std::runtime_error("SafeTensors header: expected ',' in root");
        ++i;
    }
}

Matrix matrixFromBuffer(const std::vector<char>& data, const ParsedTensor& info) {
    if (info.dtype != "F32")
        throw std::runtime_error("SafeTensors: only F32 tensors are supported (got " + info.dtype + ")");
    if (info.end < info.begin)
        throw std::runtime_error("SafeTensors: invalid data_offsets");
    const size_t byteCount = info.end - info.begin;
    const size_t elems = elementCount(info.shape);
    if (byteCount != elems * sizeof(float))
        throw std::runtime_error("SafeTensors: tensor byte size mismatch for " + info.dtype);
    if (info.begin + byteCount > data.size())
        throw std::runtime_error("SafeTensors: tensor extends past file");

    size_t rows = 0;
    size_t cols = 0;
    if (info.shape.empty()) {
        rows = 1;
        cols = 1;
    } else if (info.shape.size() == 1) {
        rows = info.shape[0];
        cols = 1;
    } else if (info.shape.size() == 2) {
        rows = info.shape[0];
        cols = info.shape[1];
    } else {
        throw std::runtime_error("SafeTensors: only rank-0/1/2 F32 tensors supported");
    }

    Matrix matrix(rows, cols, 0.0f);
    if (!matrix.data.empty())
        std::memcpy(matrix.data.data(), data.data() + info.begin, byteCount);
    return matrix;
}

} // namespace

void putMatrix(File& file, const std::string& name, const Matrix& matrix) {
    file.tensors[name] = matrix;
}

Matrix requireMatrix(const File& file, const std::string& name, size_t rows, size_t cols) {
    const auto it = file.tensors.find(name);
    if (it == file.tensors.end())
        throw std::runtime_error("SafeTensors missing tensor: " + name);
    if (it->second.rows != rows || it->second.cols != cols)
        throw std::runtime_error("SafeTensors shape mismatch for " + name);
    return it->second;
}

Matrix requireMatrixFlexible(const File& file, const std::string& name, size_t rows, size_t cols) {
    const auto it = file.tensors.find(name);
    if (it == file.tensors.end())
        throw std::runtime_error("SafeTensors missing tensor: " + name);
    const Matrix& m = it->second;
    if (m.rows == rows && m.cols == cols) return m;
    if (cols == 1 && m.rows == rows && m.cols == 1) return m;
    if (cols == 1 && m.rows == 1 && m.cols == rows) {
        // Accept [1, n] as column vector [n, 1]
        Matrix out(rows, 1, 0.0f);
        for (size_t i = 0; i < rows; ++i)
            out.data[i] = m.data[i];
        return out;
    }
    throw std::runtime_error("SafeTensors shape mismatch for " + name);
}

bool isSafeTensorsFile(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    std::uint64_t headerSize = 0;
    in.read(reinterpret_cast<char*>(&headerSize), 8);
    if (!in || headerSize == 0 || headerSize > kMaxHeaderBytes) return false;
    char brace = 0;
    in.read(&brace, 1);
    return static_cast<bool>(in) && brace == '{';
}

void save(const std::string& path, const File& file) {
    if (path.empty()) throw std::invalid_argument("SafeTensors::save empty path");
    if (file.tensors.empty()) throw std::invalid_argument("SafeTensors::save no tensors");

    struct Planned {
        std::string name;
        std::vector<size_t> shape;
        size_t begin = 0;
        size_t end = 0;
        const Matrix* matrix = nullptr;
    };

    std::vector<Planned> planned;
    planned.reserve(file.tensors.size());
    size_t offset = 0;
    for (const auto& entry : file.tensors) {
        Planned item;
        item.name = entry.first;
        item.matrix = &entry.second;
        if (entry.second.cols == 1 && entry.second.rows > 0)
            item.shape = { entry.second.rows };
        else
            item.shape = { entry.second.rows, entry.second.cols };
        const size_t bytes = entry.second.data.size() * sizeof(float);
        item.begin = offset;
        item.end = offset + bytes;
        offset = item.end;
        planned.push_back(std::move(item));
    }

    // Build JSON header (sorted by name via map iteration above — File uses std::map).
    std::ostringstream header;
    header << '{';
    bool first = true;
    if (!file.metadata.empty()) {
        header << "\"__metadata__\":{";
        bool firstMeta = true;
        for (const auto& meta : file.metadata) {
            if (!firstMeta) header << ',';
            firstMeta = false;
            header << '"' << jsonEscape(meta.first) << "\":\"" << jsonEscape(meta.second) << '"';
        }
        header << '}';
        first = false;
    }
    for (const Planned& item : planned) {
        if (!first) header << ',';
        first = false;
        header << '"' << jsonEscape(item.name) << "\":{";
        header << "\"dtype\":\"F32\",\"shape\":[";
        for (size_t s = 0; s < item.shape.size(); ++s) {
            if (s) header << ',';
            header << item.shape[s];
        }
        header << "],\"data_offsets\":[" << item.begin << ',' << item.end << "]}";
    }
    header << '}';
    std::string headerJson = header.str();

    // Pad header with spaces so 8 + N is a multiple of 8 (common safetensors practice).
    while (((8ull + headerJson.size()) % 8ull) != 0ull)
        headerJson.push_back(' ');

    const std::uint64_t headerSize = static_cast<std::uint64_t>(headerJson.size());
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("SafeTensors::save cannot open file");
    writeExact(out, &headerSize, sizeof(headerSize));
    writeExact(out, headerJson.data(), headerJson.size());
    for (const Planned& item : planned) {
        if (item.matrix->data.empty()) continue;
        writeExact(out, item.matrix->data.data(), item.matrix->data.size() * sizeof(float));
    }
}

File load(const std::string& path) {
    if (path.empty()) throw std::invalid_argument("SafeTensors::load empty path");
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("SafeTensors::load cannot open file");

    std::uint64_t headerSize = 0;
    readExact(in, &headerSize, sizeof(headerSize));
    if (headerSize == 0 || headerSize > kMaxHeaderBytes)
        throw std::runtime_error("SafeTensors::load invalid header size");

    std::string headerJson(static_cast<size_t>(headerSize), '\0');
    readExact(in, headerJson.data(), headerJson.size());
    if (headerJson.empty() || headerJson[0] != '{')
        throw std::runtime_error("SafeTensors::load header must begin with '{'");

    std::map<std::string, std::string> metadata;
    std::map<std::string, ParsedTensor> parsed;
    parseHeader(headerJson, metadata, parsed);

    // Validate coverage: no holes / overlaps, contiguous from 0.
    std::vector<std::pair<size_t, size_t>> ranges;
    ranges.reserve(parsed.size());
    size_t maxEnd = 0;
    for (const auto& entry : parsed) {
        if (entry.second.end < entry.second.begin)
            throw std::runtime_error("SafeTensors::load bad offsets for " + entry.first);
        ranges.push_back({ entry.second.begin, entry.second.end });
        maxEnd = (std::max)(maxEnd, entry.second.end);
    }
    std::sort(ranges.begin(), ranges.end());
    size_t expected = 0;
    for (const auto& range : ranges) {
        if (range.first != expected)
            throw std::runtime_error("SafeTensors::load non-contiguous tensor buffer (holes/overlap)");
        expected = range.second;
    }

    std::vector<char> data(maxEnd);
    if (maxEnd > 0)
        readExact(in, data.data(), data.size());

    File file;
    file.metadata = std::move(metadata);
    for (const auto& entry : parsed)
        file.tensors[entry.first] = matrixFromBuffer(data, entry.second);
    return file;
}

} // namespace SafeTensors
