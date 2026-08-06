#include "HfTokenizer.hpp"

#include "../Utils/SmokeLog.hpp"

#include <algorithm>
#include <cctype>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace HuggingFace {
namespace {

namespace fs = std::filesystem;

void skipWs(const std::string& json, size_t& i) {
    while (i < json.size() && (json[i] == ' ' || json[i] == '\n' || json[i] == '\r' || json[i] == '\t'))
        ++i;
}

void appendUtf8(std::string& out, unsigned codePoint) {
    if (codePoint <= 0x7Fu) {
        out.push_back(static_cast<char>(codePoint));
    } else if (codePoint <= 0x7FFu) {
        out.push_back(static_cast<char>(0xC0u | ((codePoint >> 6) & 0x1Fu)));
        out.push_back(static_cast<char>(0x80u | (codePoint & 0x3Fu)));
    } else if (codePoint <= 0xFFFFu) {
        out.push_back(static_cast<char>(0xE0u | ((codePoint >> 12) & 0x0Fu)));
        out.push_back(static_cast<char>(0x80u | ((codePoint >> 6) & 0x3Fu)));
        out.push_back(static_cast<char>(0x80u | (codePoint & 0x3Fu)));
    } else if (codePoint <= 0x10FFFFu) {
        out.push_back(static_cast<char>(0xF0u | ((codePoint >> 18) & 0x07u)));
        out.push_back(static_cast<char>(0x80u | ((codePoint >> 12) & 0x3Fu)));
        out.push_back(static_cast<char>(0x80u | ((codePoint >> 6) & 0x3Fu)));
        out.push_back(static_cast<char>(0x80u | (codePoint & 0x3Fu)));
    } else {
        throw std::runtime_error("HfTokenizer: invalid unicode code point");
    }
}

unsigned parseHex4(const std::string& json, size_t& i) {
    if (i + 4 > json.size()) throw std::runtime_error("HfTokenizer JSON: bad unicode escape");
    unsigned code = 0;
    for (int n = 0; n < 4; ++n) {
        const char h = json[i++];
        code <<= 4;
        if (h >= '0' && h <= '9') code |= static_cast<unsigned>(h - '0');
        else if (h >= 'a' && h <= 'f') code |= static_cast<unsigned>(h - 'a' + 10);
        else if (h >= 'A' && h <= 'F') code |= static_cast<unsigned>(h - 'A' + 10);
        else throw std::runtime_error("HfTokenizer JSON: bad unicode escape");
    }
    return code;
}

std::string parseString(const std::string& json, size_t& i) {
    skipWs(json, i);
    if (i >= json.size() || json[i] != '"')
        throw std::runtime_error("HfTokenizer JSON: expected string");
    ++i;
    std::string out;
    while (i < json.size()) {
        const char ch = json[i++];
        if (ch == '"') return out;
        if (ch == '\\') {
            if (i >= json.size()) throw std::runtime_error("HfTokenizer JSON: bad escape");
            const char esc = json[i++];
            switch (esc) {
            case '"': out.push_back('"'); break;
            case '\\': out.push_back('\\'); break;
            case '/': out.push_back('/'); break;
            case 'n': out.push_back('\n'); break;
            case 'r': out.push_back('\r'); break;
            case 't': out.push_back('\t'); break;
            case 'b': out.push_back('\b'); break;
            case 'f': out.push_back('\f'); break;
            case 'u': {
                unsigned code = parseHex4(json, i);
                // UTF-16 surrogate pair
                if (code >= 0xD800u && code <= 0xDBFFu) {
                    skipWs(json, i); // not expected between halves, but be strict:
                    if (i + 1 < json.size() && json[i] == '\\' && json[i + 1] == 'u') {
                        i += 2;
                        const unsigned low = parseHex4(json, i);
                        if (low < 0xDC00u || low > 0xDFFFu)
                            throw std::runtime_error("HfTokenizer JSON: bad surrogate pair");
                        code = 0x10000u + (((code - 0xD800u) << 10) | (low - 0xDC00u));
                    } else {
                        throw std::runtime_error("HfTokenizer JSON: lone surrogate");
                    }
                } else if (code >= 0xDC00u && code <= 0xDFFFu) {
                    throw std::runtime_error("HfTokenizer JSON: lone low surrogate");
                }
                appendUtf8(out, code);
                break;
            }
            default: throw std::runtime_error("HfTokenizer JSON: unsupported escape");
            }
        } else {
            out.push_back(ch);
        }
    }
    throw std::runtime_error("HfTokenizer JSON: unterminated string");
}

void skipValue(const std::string& json, size_t& i);

void skipObject(const std::string& json, size_t& i) {
    skipWs(json, i);
    if (i >= json.size() || json[i] != '{')
        throw std::runtime_error("HfTokenizer JSON: expected object");
    ++i;
    skipWs(json, i);
    if (i < json.size() && json[i] == '}') {
        ++i;
        return;
    }
    while (true) {
        (void)parseString(json, i);
        skipWs(json, i);
        if (i >= json.size() || json[i] != ':')
            throw std::runtime_error("HfTokenizer JSON: expected ':'");
        ++i;
        skipValue(json, i);
        skipWs(json, i);
        if (i >= json.size()) throw std::runtime_error("HfTokenizer JSON: truncated object");
        if (json[i] == '}') {
            ++i;
            return;
        }
        if (json[i] != ',') throw std::runtime_error("HfTokenizer JSON: expected ','");
        ++i;
    }
}

void skipArray(const std::string& json, size_t& i) {
    skipWs(json, i);
    if (i >= json.size() || json[i] != '[')
        throw std::runtime_error("HfTokenizer JSON: expected array");
    ++i;
    skipWs(json, i);
    if (i < json.size() && json[i] == ']') {
        ++i;
        return;
    }
    while (true) {
        skipValue(json, i);
        skipWs(json, i);
        if (i >= json.size()) throw std::runtime_error("HfTokenizer JSON: truncated array");
        if (json[i] == ']') {
            ++i;
            return;
        }
        if (json[i] != ',') throw std::runtime_error("HfTokenizer JSON: expected ','");
        ++i;
    }
}

void skipValue(const std::string& json, size_t& i) {
    skipWs(json, i);
    if (i >= json.size()) throw std::runtime_error("HfTokenizer JSON: truncated value");
    const char ch = json[i];
    if (ch == '"') {
        (void)parseString(json, i);
        return;
    }
    if (ch == '{') {
        skipObject(json, i);
        return;
    }
    if (ch == '[') {
        skipArray(json, i);
        return;
    }
    if (ch == 't' || ch == 'f' || ch == 'n' || ch == '-' || (ch >= '0' && ch <= '9')) {
        while (i < json.size()) {
            const char c = json[i];
            if (c == ',' || c == '}' || c == ']' || c == ' ' || c == '\n' || c == '\r' || c == '\t')
                break;
            ++i;
        }
        return;
    }
    throw std::runtime_error("HfTokenizer JSON: unexpected value");
}

bool parseBool(const std::string& json, size_t& i) {
    skipWs(json, i);
    if (i + 4 <= json.size()
        && json[i] == 't' && json[i + 1] == 'r' && json[i + 2] == 'u' && json[i + 3] == 'e') {
        i += 4;
        return true;
    }
    if (i + 5 <= json.size()
        && json[i] == 'f' && json[i + 1] == 'a' && json[i + 2] == 'l' && json[i + 3] == 's'
        && json[i + 4] == 'e') {
        i += 5;
        return false;
    }
    throw std::runtime_error("HfTokenizer JSON: expected boolean");
}

bool tryParseNull(const std::string& json, size_t& i) {
    skipWs(json, i);
    if (i + 4 <= json.size()
        && json[i] == 'n' && json[i + 1] == 'u' && json[i + 2] == 'l' && json[i + 3] == 'l') {
        i += 4;
        return true;
    }
    return false;
}

int parseInt(const std::string& json, size_t& i) {
    skipWs(json, i);
    if (i >= json.size()) throw std::runtime_error("HfTokenizer JSON: expected number");
    const size_t begin = i;
    if (json[i] == '-') ++i;
    if (i >= json.size() || json[i] < '0' || json[i] > '9')
        throw std::runtime_error("HfTokenizer JSON: expected number");
    while (i < json.size() && json[i] >= '0' && json[i] <= '9') ++i;
    return std::stoi(json.substr(begin, i - begin));
}

std::unordered_map<std::string, int> parseVocabObject(const std::string& json, size_t& i) {
    skipWs(json, i);
    if (i >= json.size() || json[i] != '{')
        throw std::runtime_error("HfTokenizer: expected vocab object");
    ++i;
    std::unordered_map<std::string, int> vocab;
    skipWs(json, i);
    if (i < json.size() && json[i] == '}') {
        ++i;
        return vocab;
    }
    while (true) {
        const std::string token = parseString(json, i);
        skipWs(json, i);
        if (i >= json.size() || json[i] != ':')
            throw std::runtime_error("HfTokenizer: expected ':' in vocab");
        ++i;
        const int id = parseInt(json, i);
        vocab.emplace(token, id);
        skipWs(json, i);
        if (i >= json.size()) throw std::runtime_error("HfTokenizer: truncated vocab");
        if (json[i] == '}') {
            ++i;
            break;
        }
        if (json[i] != ',') throw std::runtime_error("HfTokenizer: expected ',' in vocab");
        ++i;
    }
    return vocab;
}

std::vector<std::pair<std::string, std::string>> parseMerges(const std::string& json, size_t& i) {
    skipWs(json, i);
    if (i >= json.size() || json[i] != '[')
        throw std::runtime_error("HfTokenizer: expected merges array");
    ++i;
    std::vector<std::pair<std::string, std::string>> merges;
    skipWs(json, i);
    if (i < json.size() && json[i] == ']') {
        ++i;
        return merges;
    }
    while (true) {
        skipWs(json, i);
        if (i >= json.size()) throw std::runtime_error("HfTokenizer: truncated merges");
        if (json[i] == '[') {
            ++i;
            const std::string left = parseString(json, i);
            skipWs(json, i);
            if (i >= json.size() || json[i] != ',')
                throw std::runtime_error("HfTokenizer: expected ',' in merge pair");
            ++i;
            const std::string right = parseString(json, i);
            skipWs(json, i);
            if (i >= json.size() || json[i] != ']')
                throw std::runtime_error("HfTokenizer: expected ']' in merge pair");
            ++i;
            merges.emplace_back(left, right);
        } else if (json[i] == '"') {
            const std::string merged = parseString(json, i);
            const size_t sp = merged.find(' ');
            if (sp == std::string::npos)
                throw std::runtime_error("HfTokenizer: merge string must contain a space");
            merges.emplace_back(merged.substr(0, sp), merged.substr(sp + 1));
        } else {
            throw std::runtime_error("HfTokenizer: unexpected merge entry");
        }
        skipWs(json, i);
        if (i >= json.size()) throw std::runtime_error("HfTokenizer: truncated merges");
        if (json[i] == ']') {
            ++i;
            break;
        }
        if (json[i] != ',') throw std::runtime_error("HfTokenizer: expected ',' in merges");
        ++i;
    }
    return merges;
}

struct AddedToken {
    std::string content;
    int id = -1;
    bool special = false;
};

void parseAddedTokensArray(const std::string& json, size_t& i, std::vector<AddedToken>& out) {
    skipWs(json, i);
    if (i >= json.size() || json[i] != '[')
        throw std::runtime_error("HfTokenizer: expected added_tokens array");
    ++i;
    skipWs(json, i);
    if (i < json.size() && json[i] == ']') {
        ++i;
        return;
    }
    while (true) {
        skipWs(json, i);
        if (i >= json.size() || json[i] != '{')
            throw std::runtime_error("HfTokenizer: expected added_token object");
        ++i;
        AddedToken tok;
        skipWs(json, i);
        if (i < json.size() && json[i] != '}') {
            while (true) {
                const std::string key = parseString(json, i);
                skipWs(json, i);
                if (i >= json.size() || json[i] != ':')
                    throw std::runtime_error("HfTokenizer: expected ':' in added_token");
                ++i;
                if (key == "id") tok.id = parseInt(json, i);
                else if (key == "content") tok.content = parseString(json, i);
                else if (key == "special") tok.special = parseBool(json, i);
                else skipValue(json, i);
                skipWs(json, i);
                if (i >= json.size()) throw std::runtime_error("HfTokenizer: truncated added_token");
                if (json[i] == '}') {
                    ++i;
                    break;
                }
                if (json[i] != ',') throw std::runtime_error("HfTokenizer: expected ',' in added_token");
                ++i;
            }
        } else if (i < json.size() && json[i] == '}') {
            ++i;
        }
        if (tok.id >= 0 && !tok.content.empty())
            out.push_back(std::move(tok));
        skipWs(json, i);
        if (i >= json.size()) throw std::runtime_error("HfTokenizer: truncated added_tokens");
        if (json[i] == ']') {
            ++i;
            break;
        }
        if (json[i] != ',') throw std::runtime_error("HfTokenizer: expected ',' in added_tokens");
        ++i;
    }
}

bool extractTypeField(const std::string& objectJson, std::string& typeOut) {
    size_t i = 0;
    skipWs(objectJson, i);
    if (i >= objectJson.size() || objectJson[i] != '{') return false;
    ++i;
    skipWs(objectJson, i);
    if (i < objectJson.size() && objectJson[i] == '}') return false;
    while (true) {
        const std::string key = parseString(objectJson, i);
        skipWs(objectJson, i);
        if (i >= objectJson.size() || objectJson[i] != ':') return false;
        ++i;
        if (key == "type") {
            typeOut = parseString(objectJson, i);
            return true;
        }
        skipValue(objectJson, i);
        skipWs(objectJson, i);
        if (i >= objectJson.size()) return false;
        if (objectJson[i] == '}') return false;
        if (objectJson[i] != ',') return false;
        ++i;
    }
}

// Capture a raw JSON value substring starting at i (object/array/string/atom), advancing i.
std::string captureValue(const std::string& json, size_t& i) {
    skipWs(json, i);
    const size_t begin = i;
    skipValue(json, i);
    return json.substr(begin, i - begin);
}

bool looksLikeLetter(uint32_t cp) {
    if ((cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z')) return true;
    // Approximate \p{L}: treat non-ASCII (except some separators) as letters.
    return cp >= 0x80u && cp != 0xA0u;
}

bool looksLikeNumber(uint32_t cp) {
    return cp >= '0' && cp <= '9';
}

bool nextCodePoint(const std::string& text, size_t& i, uint32_t& cp) {
    if (i >= text.size()) return false;
    const unsigned char c0 = static_cast<unsigned char>(text[i]);
    if (c0 < 0x80u) {
        cp = c0;
        ++i;
        return true;
    }
    if ((c0 & 0xE0u) == 0xC0u && i + 1 < text.size()) {
        cp = (static_cast<uint32_t>(c0 & 0x1Fu) << 6)
            | static_cast<uint32_t>(static_cast<unsigned char>(text[i + 1]) & 0x3Fu);
        i += 2;
        return true;
    }
    if ((c0 & 0xF0u) == 0xE0u && i + 2 < text.size()) {
        cp = (static_cast<uint32_t>(c0 & 0x0Fu) << 12)
            | (static_cast<uint32_t>(static_cast<unsigned char>(text[i + 1]) & 0x3Fu) << 6)
            | static_cast<uint32_t>(static_cast<unsigned char>(text[i + 2]) & 0x3Fu);
        i += 3;
        return true;
    }
    if ((c0 & 0xF8u) == 0xF0u && i + 3 < text.size()) {
        cp = (static_cast<uint32_t>(c0 & 0x07u) << 18)
            | (static_cast<uint32_t>(static_cast<unsigned char>(text[i + 1]) & 0x3Fu) << 12)
            | (static_cast<uint32_t>(static_cast<unsigned char>(text[i + 2]) & 0x3Fu) << 6)
            | static_cast<uint32_t>(static_cast<unsigned char>(text[i + 3]) & 0x3Fu);
        i += 4;
        return true;
    }
    cp = c0;
    ++i;
    return true;
}

std::string sliceUtf8(const std::string& text, size_t begin, size_t end) {
    return text.substr(begin, end - begin);
}

// Approximate Llama-3 Isolated Split regex for UTF-8 text (ASCII digits/contractions + letter runs).
std::vector<std::string> llamaLikeSplit(const std::string& text) {
    std::vector<std::string> parts;
    size_t i = 0;
    while (i < text.size()) {
        const size_t start = i;
        uint32_t cp = 0;
        if (!nextCodePoint(text, i, cp)) break;

        // contractions: 's 't 're 've 'm 'll 'd (case insensitive), optionally after letters handled by letter-run
        if (cp == '\'') {
            const size_t afterQuote = i;
            std::string lower;
            size_t j = afterQuote;
            while (j < text.size() && lower.size() < 2) {
                uint32_t c2 = 0;
                const size_t before = j;
                if (!nextCodePoint(text, j, c2)) break;
                if (c2 < 128) lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c2))));
                else {
                    j = before;
                    break;
                }
            }
            if (lower == "s" || lower == "t" || lower == "m" || lower == "d"
                || lower == "re" || lower == "ve" || lower == "ll") {
                i = j;
                parts.push_back(sliceUtf8(text, start, i));
                continue;
            }
            i = afterQuote;
            parts.push_back(sliceUtf8(text, start, i));
            continue;
        }

        if (looksLikeLetter(cp)) {
            while (i < text.size()) {
                size_t mark = i;
                uint32_t c2 = 0;
                if (!nextCodePoint(text, i, c2)) break;
                if (!looksLikeLetter(c2)) {
                    i = mark;
                    break;
                }
            }
            // optional leading non-letter non-newline already consumed only if we started with letter;
            // Llama pattern allows one non-letter before letters — handled when punct starts below.
            parts.push_back(sliceUtf8(text, start, i));
            continue;
        }

        if (looksLikeNumber(cp)) {
            int count = 1;
            while (count < 3 && i < text.size()) {
                size_t mark = i;
                uint32_t c2 = 0;
                if (!nextCodePoint(text, i, c2)) break;
                if (!looksLikeNumber(c2)) {
                    i = mark;
                    break;
                }
                ++count;
            }
            parts.push_back(sliceUtf8(text, start, i));
            continue;
        }

        if (cp == ' ' || cp == '\t') {
            while (i < text.size()) {
                size_t mark = i;
                uint32_t c2 = 0;
                if (!nextCodePoint(text, i, c2)) break;
                if (c2 != ' ' && c2 != '\t') {
                    i = mark;
                    break;
                }
            }
            // If next is a letter/number/punct run with leading space, Llama often keeps space with the next token
            // via " ?..." patterns. Fold one leading space into the following piece when possible.
            if (i < text.size()) {
                size_t mark = i;
                uint32_t c2 = 0;
                if (nextCodePoint(text, i, c2)) {
                    i = mark;
                    if (looksLikeLetter(c2) || looksLikeNumber(c2)
                        || (c2 != ' ' && c2 != '\t' && c2 != '\n' && c2 != '\r')) {
                        // restart loop from whitespace start so letter branch can include a single leading space
                        // by manually building: space + following run
                        const size_t wsEnd = mark;
                        // take last single space only
                        std::string lead = " ";
                        size_t k = wsEnd;
                        uint32_t first = 0;
                        const size_t pieceStart = k;
                        (void)nextCodePoint(text, k, first);
                        if (looksLikeLetter(first)) {
                            while (k < text.size()) {
                                size_t m2 = k;
                                uint32_t c3 = 0;
                                if (!nextCodePoint(text, k, c3)) break;
                                if (!looksLikeLetter(c3)) {
                                    k = m2;
                                    break;
                                }
                            }
                            parts.push_back(lead + sliceUtf8(text, pieceStart, k));
                            i = k;
                            continue;
                        }
                        if (looksLikeNumber(first)) {
                            int count = 1;
                            while (count < 3 && k < text.size()) {
                                size_t m2 = k;
                                uint32_t c3 = 0;
                                if (!nextCodePoint(text, k, c3)) break;
                                if (!looksLikeNumber(c3)) {
                                    k = m2;
                                    break;
                                }
                                ++count;
                            }
                            parts.push_back(lead + sliceUtf8(text, pieceStart, k));
                            i = k;
                            continue;
                        }
                        // punct cluster with leading space
                        while (k < text.size()) {
                            size_t m2 = k;
                            uint32_t c3 = 0;
                            if (!nextCodePoint(text, k, c3)) break;
                            if (c3 == ' ' || c3 == '\t' || c3 == '\n' || c3 == '\r'
                                || looksLikeLetter(c3) || looksLikeNumber(c3)) {
                                k = m2;
                                break;
                            }
                        }
                        while (k < text.size()) {
                            size_t m2 = k;
                            uint32_t c3 = 0;
                            if (!nextCodePoint(text, k, c3)) break;
                            if (c3 != '\n' && c3 != '\r') {
                                k = m2;
                                break;
                            }
                        }
                        parts.push_back(lead + sliceUtf8(text, pieceStart, k));
                        i = k;
                        continue;
                    }
                }
            }
            parts.push_back(sliceUtf8(text, start, i));
            continue;
        }

        if (cp == '\n' || cp == '\r') {
            while (i < text.size()) {
                size_t mark = i;
                uint32_t c2 = 0;
                if (!nextCodePoint(text, i, c2)) break;
                if (c2 != '\n' && c2 != '\r') {
                    i = mark;
                    break;
                }
            }
            parts.push_back(sliceUtf8(text, start, i));
            continue;
        }

        // punctuation / other: optional leading already consumed; consume run + trailing newlines
        while (i < text.size()) {
            size_t mark = i;
            uint32_t c2 = 0;
            if (!nextCodePoint(text, i, c2)) break;
            if (c2 == ' ' || c2 == '\t' || c2 == '\n' || c2 == '\r'
                || looksLikeLetter(c2) || looksLikeNumber(c2)) {
                i = mark;
                break;
            }
        }
        while (i < text.size()) {
            size_t mark = i;
            uint32_t c2 = 0;
            if (!nextCodePoint(text, i, c2)) break;
            if (c2 != '\n' && c2 != '\r') {
                i = mark;
                break;
            }
        }
        parts.push_back(sliceUtf8(text, start, i));
    }
    return parts;
}

// GPT-2 style regex approximation (ASCII-oriented + UTF-8 letter runs).
std::vector<std::string> gpt2LikeSplit(const std::string& text) {
    // Reuse llama-like splitter — close enough for allowlisted causal LMs in Sentinel.
    return llamaLikeSplit(text);
}

std::string readEntireFile(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("HfTokenizer: cannot open " + path);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

bool pathLooksLikeDirectory(const std::string& path) {
    if (path.empty()) return false;
    const char last = path.back();
    if (last == '/' || last == '\\') return true;
    const auto slash = path.find_last_of("/\\");
    const std::string leaf = (slash == std::string::npos) ? path : path.substr(slash + 1);
    const auto dot = leaf.find_last_of('.');
    if (dot == std::string::npos) return true;
    std::string ext = leaf.substr(dot + 1);
    for (char& c : ext) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return ext != "json";
}

int findSpecialId(
    const std::unordered_map<std::string, int>& vocab,
    const std::vector<AddedToken>& added,
    std::initializer_list<const char*> names) {
    for (const char* name : names) {
        for (const AddedToken& t : added) {
            if (t.content == name) return t.id;
        }
        const auto it = vocab.find(name);
        if (it != vocab.end()) return it->second;
    }
    return -1;
}

void writeText(const fs::path& path, const std::string& body) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("HfTokenizer smoke: cannot write " + path.string());
    out << body;
}

} // namespace

void Tokenizer::clear() {
    tokenToId_.clear();
    idToToken_.clear();
    merges_.clear();
    mergeRank_.clear();
    specialTokenIds_.clear();
    unicodeToByte_.clear();
    preTokenizeKind_ = PreTokenizeKind::ByteLevelRaw;
    ignoreMerges_ = false;
    addPrefixSpace_ = false;
    bosTokenId_ = -1;
    eosTokenId_ = -1;
    padTokenId_ = -1;
    unkTokenId_ = -1;
    for (auto& s : bytesToUnicode_)
        s.clear();
}

void Tokenizer::buildByteMaps() {
    // GPT-2 bytes_to_unicode
    std::vector<int> bs;
    std::vector<int> cs;
    for (int i = static_cast<int>('!'); i <= static_cast<int>('~'); ++i) bs.push_back(i);
    for (int i = 0xA1; i <= 0xAC; ++i) bs.push_back(i);
    for (int i = 0xAE; i <= 0xFF; ++i) bs.push_back(i);
    cs = bs;
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.push_back(b);
            cs.push_back(256 + n);
            ++n;
        }
    }
    for (size_t i = 0; i < bs.size(); ++i) {
        std::string uni;
        appendUtf8(uni, static_cast<unsigned>(cs[i]));
        bytesToUnicode_[static_cast<size_t>(bs[i])] = uni;
        unicodeToByte_[uni] = static_cast<unsigned char>(bs[i]);
    }
}

std::string Tokenizer::mergeKey(const std::string& left, const std::string& right) {
    return left + "\x1f" + right;
}

void Tokenizer::rebuildMergeRank() {
    mergeRank_.clear();
    for (size_t rank = 0; rank < merges_.size(); ++rank)
        mergeRank_[mergeKey(merges_[rank].first, merges_[rank].second)] = static_cast<int>(rank);
}

std::string Tokenizer::utf8ToByteLevel(const std::string& text) const {
    std::string out;
    out.reserve(text.size() * 2);
    for (unsigned char byte : text)
        out += bytesToUnicode_[byte];
    return out;
}

std::string Tokenizer::byteLevelToUtf8(const std::string& token) const {
    std::string out;
    size_t i = 0;
    while (i < token.size()) {
        // Match longest unicode map key starting at i (1–4 UTF-8 bytes).
        bool matched = false;
        for (int len = 4; len >= 1; --len) {
            if (i + static_cast<size_t>(len) > token.size()) continue;
            const std::string piece = token.substr(i, static_cast<size_t>(len));
            const auto it = unicodeToByte_.find(piece);
            if (it != unicodeToByte_.end()) {
                out.push_back(static_cast<char>(it->second));
                i += static_cast<size_t>(len);
                matched = true;
                break;
            }
        }
        if (!matched) {
            // Pass through unknown (special tokens).
            out.push_back(token[i]);
            ++i;
        }
    }
    return out;
}

std::vector<std::string> Tokenizer::preTokenize(const std::string& text) const {
    std::string input = text;
    if (addPrefixSpace_ && !input.empty() && input[0] != ' ')
        input.insert(input.begin(), ' ');

    switch (preTokenizeKind_) {
    case PreTokenizeKind::ByteLevelRaw:
        return { input };
    case PreTokenizeKind::ByteLevelGpt2Regex:
        return gpt2LikeSplit(input);
    case PreTokenizeKind::LlamaSplitThenByteLevel:
        return llamaLikeSplit(input);
    }
    return { input };
}

std::vector<int> Tokenizer::bpeEncodePiece(const std::string& byteLevelPiece) const {
    if (byteLevelPiece.empty()) return {};

    if (ignoreMerges_) {
        const auto it = tokenToId_.find(byteLevelPiece);
        if (it != tokenToId_.end())
            return { it->second };
    }

    // Start from individual unicode characters (UTF-8 codepoints in the byte-level alphabet).
    std::vector<std::string> symbols;
    size_t i = 0;
    while (i < byteLevelPiece.size()) {
        size_t len = 1;
        const unsigned char c0 = static_cast<unsigned char>(byteLevelPiece[i]);
        if ((c0 & 0x80u) == 0) len = 1;
        else if ((c0 & 0xE0u) == 0xC0u) len = 2;
        else if ((c0 & 0xF0u) == 0xE0u) len = 3;
        else if ((c0 & 0xF8u) == 0xF0u) len = 4;
        if (i + len > byteLevelPiece.size()) len = 1;
        symbols.push_back(byteLevelPiece.substr(i, len));
        i += len;
    }

    if (symbols.size() == 1) {
        const auto it = tokenToId_.find(symbols[0]);
        if (it == tokenToId_.end()) {
            if (unkTokenId_ >= 0) return { unkTokenId_ };
            throw std::runtime_error("HfTokenizer: unknown symbol '" + symbols[0] + "'");
        }
        return { it->second };
    }

    while (symbols.size() > 1) {
        int bestRank = INT_MAX;
        size_t bestIndex = 0;
        bool found = false;
        for (size_t idx = 0; idx + 1 < symbols.size(); ++idx) {
            const auto it = mergeRank_.find(mergeKey(symbols[idx], symbols[idx + 1]));
            if (it == mergeRank_.end()) continue;
            if (it->second < bestRank) {
                bestRank = it->second;
                bestIndex = idx;
                found = true;
            }
        }
        if (!found) break;
        const std::string merged = symbols[bestIndex] + symbols[bestIndex + 1];
        symbols[bestIndex] = merged;
        symbols.erase(symbols.begin() + static_cast<std::ptrdiff_t>(bestIndex + 1));
    }

    std::vector<int> ids;
    ids.reserve(symbols.size());
    for (const std::string& sym : symbols) {
        const auto it = tokenToId_.find(sym);
        if (it == tokenToId_.end()) {
            if (unkTokenId_ >= 0) ids.push_back(unkTokenId_);
            else throw std::runtime_error("HfTokenizer: unknown BPE piece");
        } else {
            ids.push_back(it->second);
        }
    }
    return ids;
}

Tokenizer Tokenizer::load(const std::string& pathOrDirectory) {
    if (pathOrDirectory.empty())
        throw std::invalid_argument("HfTokenizer::load empty path");

    std::string path = pathOrDirectory;
    if (pathLooksLikeDirectory(path)) {
        if (!path.empty() && path.back() != '/' && path.back() != '\\')
            path.push_back('/');
        path += "tokenizer.json";
    }

    const std::string json = readEntireFile(path);
    Tokenizer tok;
    tok.clear();
    tok.buildByteMaps();

    std::string modelType;
    std::string preTokenizerJson;
    std::string decoderJson;
    std::vector<AddedToken> added;
    bool haveVocab = false;
    bool haveMerges = false;

    size_t i = 0;
    skipWs(json, i);
    if (i >= json.size() || json[i] != '{')
        throw std::runtime_error("HfTokenizer: expected top-level object");
    ++i;
    skipWs(json, i);
    if (i < json.size() && json[i] == '}')
        throw std::runtime_error("HfTokenizer: empty tokenizer.json");

    while (true) {
        const std::string key = parseString(json, i);
        skipWs(json, i);
        if (i >= json.size() || json[i] != ':')
            throw std::runtime_error("HfTokenizer: expected ':'");
        ++i;

        if (key == "model") {
            skipWs(json, i);
            if (i >= json.size() || json[i] != '{')
                throw std::runtime_error("HfTokenizer: expected model object");
            ++i;
            skipWs(json, i);
            while (i < json.size() && json[i] != '}') {
                const std::string mkey = parseString(json, i);
                skipWs(json, i);
                if (i >= json.size() || json[i] != ':')
                    throw std::runtime_error("HfTokenizer: expected ':' in model");
                ++i;
                if (mkey == "type") {
                    modelType = parseString(json, i);
                } else if (mkey == "vocab") {
                    tok.tokenToId_ = parseVocabObject(json, i);
                    haveVocab = true;
                } else if (mkey == "merges") {
                    tok.merges_ = parseMerges(json, i);
                    haveMerges = true;
                } else if (mkey == "ignore_merges") {
                    if (!tryParseNull(json, i))
                        tok.ignoreMerges_ = parseBool(json, i);
                } else if (mkey == "unk_token") {
                    if (!tryParseNull(json, i)) {
                        const std::string unk = parseString(json, i);
                        const auto it = tok.tokenToId_.find(unk);
                        if (it != tok.tokenToId_.end()) tok.unkTokenId_ = it->second;
                    }
                } else {
                    skipValue(json, i);
                }
                skipWs(json, i);
                if (i >= json.size()) throw std::runtime_error("HfTokenizer: truncated model");
                if (json[i] == '}') break;
                if (json[i] != ',') throw std::runtime_error("HfTokenizer: expected ',' in model");
                ++i;
                skipWs(json, i);
            }
            if (i >= json.size() || json[i] != '}')
                throw std::runtime_error("HfTokenizer: expected '}' in model");
            ++i;
        } else if (key == "added_tokens") {
            parseAddedTokensArray(json, i, added);
        } else if (key == "pre_tokenizer") {
            if (!tryParseNull(json, i))
                preTokenizerJson = captureValue(json, i);
        } else if (key == "decoder") {
            if (!tryParseNull(json, i))
                decoderJson = captureValue(json, i);
        } else {
            skipValue(json, i);
        }

        skipWs(json, i);
        if (i >= json.size()) throw std::runtime_error("HfTokenizer: truncated tokenizer.json");
        if (json[i] == '}') {
            ++i;
            break;
        }
        if (json[i] != ',') throw std::runtime_error("HfTokenizer: expected ','");
        ++i;
    }

    if (modelType != "BPE")
        throw std::runtime_error(
            "HfTokenizer unsupported model.type='" + modelType + "' (only BPE)");
    if (!haveVocab || tok.tokenToId_.empty())
        throw std::runtime_error("HfTokenizer: missing vocab");
    if (!haveMerges)
        throw std::runtime_error("HfTokenizer: missing merges");

    // decoder should be ByteLevel (or Sequence containing it) for this family
    if (!decoderJson.empty()) {
        std::string dtype;
        if (extractTypeField(decoderJson, dtype)) {
            if (dtype != "ByteLevel" && dtype != "Sequence")
                throw std::runtime_error(
                    "HfTokenizer unsupported decoder.type='" + dtype + "' (need ByteLevel)");
        }
    }

    // pre_tokenizer detection
    if (preTokenizerJson.empty()) {
        tok.preTokenizeKind_ = PreTokenizeKind::ByteLevelRaw;
    } else {
        std::string ptype;
        if (!extractTypeField(preTokenizerJson, ptype))
            throw std::runtime_error("HfTokenizer: pre_tokenizer missing type");
        if (ptype == "ByteLevel") {
            // inspect use_regex
            if (preTokenizerJson.find("\"use_regex\":false") != std::string::npos
                || preTokenizerJson.find("\"use_regex\": false") != std::string::npos)
                tok.preTokenizeKind_ = PreTokenizeKind::ByteLevelRaw;
            else
                tok.preTokenizeKind_ = PreTokenizeKind::ByteLevelGpt2Regex;
            if (preTokenizerJson.find("\"add_prefix_space\":true") != std::string::npos
                || preTokenizerJson.find("\"add_prefix_space\": true") != std::string::npos)
                tok.addPrefixSpace_ = true;
        } else if (ptype == "Sequence") {
            // Llama/Qwen2: Split + ByteLevel(use_regex=false)
            tok.preTokenizeKind_ = PreTokenizeKind::LlamaSplitThenByteLevel;
            if (preTokenizerJson.find("\"add_prefix_space\":true") != std::string::npos
                || preTokenizerJson.find("\"add_prefix_space\": true") != std::string::npos)
                tok.addPrefixSpace_ = true;
        } else if (ptype == "Metaspace") {
            throw std::runtime_error(
                "HfTokenizer unsupported pre_tokenizer Metaspace (Llama-2 SP); use Llama-3/Qwen2-style ByteLevel BPE");
        } else {
            throw std::runtime_error(
                "HfTokenizer unsupported pre_tokenizer.type='" + ptype + "'");
        }
    }

    int maxId = -1;
    for (const auto& [_, id] : tok.tokenToId_)
        maxId = (std::max)(maxId, id);
    for (const AddedToken& t : added)
        maxId = (std::max)(maxId, t.id);
    if (maxId < 0) throw std::runtime_error("HfTokenizer: empty vocab ids");
    tok.idToToken_.assign(static_cast<size_t>(maxId + 1), std::string{});
    for (const auto& [token, id] : tok.tokenToId_) {
        if (id < 0 || id > maxId) continue;
        tok.idToToken_[static_cast<size_t>(id)] = token;
    }
    for (const AddedToken& t : added) {
        tok.tokenToId_[t.content] = t.id;
        if (t.id >= 0 && t.id <= maxId)
            tok.idToToken_[static_cast<size_t>(t.id)] = t.content;
        if (t.special) tok.specialTokenIds_.insert(t.id);
    }

    tok.bosTokenId_ = findSpecialId(
        tok.tokenToId_, added,
        { "<|begin_of_text|>", "<s>", "<|im_start|>", "[BOS]", "<bos>" });
    tok.eosTokenId_ = findSpecialId(
        tok.tokenToId_, added,
        { "<|end_of_text|>", "</s>", "<|im_end|>", "[EOS]", "<eos>", "<|eot_id|>" });
    tok.padTokenId_ = findSpecialId(
        tok.tokenToId_, added,
        { "<pad>", "[PAD]", "<|padding|>" });
    if (tok.unkTokenId_ < 0) {
        tok.unkTokenId_ = findSpecialId(
            tok.tokenToId_, added,
            { "<unk>", "[UNK]", "<|unk|>" });
    }
    if (tok.bosTokenId_ >= 0) tok.specialTokenIds_.insert(tok.bosTokenId_);
    if (tok.eosTokenId_ >= 0) tok.specialTokenIds_.insert(tok.eosTokenId_);
    if (tok.padTokenId_ >= 0) tok.specialTokenIds_.insert(tok.padTokenId_);
    if (tok.unkTokenId_ >= 0) tok.specialTokenIds_.insert(tok.unkTokenId_);

    tok.rebuildMergeRank();
    return tok;
}

std::vector<int> Tokenizer::encode(const std::string& text, bool addSpecialTokens) const {
    if (!isLoaded()) throw std::logic_error("HfTokenizer::encode not loaded");

    std::vector<int> ids;
    if (addSpecialTokens && bosTokenId_ >= 0)
        ids.push_back(bosTokenId_);

    const std::vector<std::string> pieces = preTokenize(text);
    for (const std::string& piece : pieces) {
        const std::string bl = utf8ToByteLevel(piece);
        const std::vector<int> pieceIds = bpeEncodePiece(bl);
        ids.insert(ids.end(), pieceIds.begin(), pieceIds.end());
    }
    return ids;
}

std::string Tokenizer::decode(const std::vector<int>& tokenIds, bool skipSpecialTokens) const {
    if (!isLoaded()) throw std::logic_error("HfTokenizer::decode not loaded");
    std::string byteLevel;
    for (int id : tokenIds) {
        if (id < 0 || id >= static_cast<int>(idToToken_.size()))
            throw std::out_of_range("HfTokenizer::decode id out of range");
        if (skipSpecialTokens && specialTokenIds_.count(id) != 0)
            continue;
        byteLevel += idToToken_[static_cast<size_t>(id)];
    }
    return byteLevelToUtf8(byteLevel);
}

int Tokenizer::vocabSize() const {
    return static_cast<int>(idToToken_.size());
}

int Tokenizer::bosTokenId() const { return bosTokenId_; }
int Tokenizer::eosTokenId() const { return eosTokenId_; }
int Tokenizer::padTokenId() const { return padTokenId_; }
int Tokenizer::unkTokenId() const { return unkTokenId_; }
bool Tokenizer::isLoaded() const { return !idToToken_.empty(); }
bool Tokenizer::ignoreMerges() const { return ignoreMerges_; }

void Tokenizer::runTokenizerSmokeDemo() {
    // Minimal ByteLevel BPE: vocab for bytes of "hi" / "ab" + merges + specials.
    // 'h','i','a','b' map to themselves in bytes_to_unicode.
    const std::string json = R"json({
  "version": "1.0",
  "added_tokens": [
    {"id": 10, "content": "<bos>", "special": true},
    {"id": 11, "content": "<eos>", "special": true}
  ],
  "pre_tokenizer": {
    "type": "ByteLevel",
    "add_prefix_space": false,
    "trim_offsets": true,
    "use_regex": false
  },
  "decoder": {
    "type": "ByteLevel",
    "add_prefix_space": false,
    "trim_offsets": true,
    "use_regex": false
  },
  "model": {
    "type": "BPE",
    "unk_token": null,
    "ignore_merges": false,
    "vocab": {
      "h": 0,
      "i": 1,
      "a": 2,
      "b": 3,
      "hi": 4,
      "ab": 5,
      "hello": 6,
      "<bos>": 10,
      "<eos>": 11
    },
    "merges": [
      "h i",
      "a b"
    ]
  }
})json";

    const fs::path path = "hf_tokenizer_smoke.json";
    writeText(path, json);
    Tokenizer tok = Tokenizer::load(path.string());
    if (!tok.isLoaded() || tok.vocabSize() < 7)
        throw std::runtime_error("HfTokenizer smoke: load failed");
    if (tok.bosTokenId() != 10 || tok.eosTokenId() != 11)
        throw std::runtime_error("HfTokenizer smoke: special ids mismatch");

    const std::vector<int> ids = tok.encode("hi", true);
    if (ids.size() != 2 || ids[0] != 10 || ids[1] != 4)
        throw std::runtime_error("HfTokenizer smoke: encode(hi) expected [bos, hi]");
    const std::string round = tok.decode(ids, true);
    if (round != "hi")
        throw std::runtime_error("HfTokenizer smoke: decode mismatch got '" + round + "'");

    const std::vector<int> ab = tok.encode("ab", false);
    if (ab.size() != 1 || ab[0] != 5)
        throw std::runtime_error("HfTokenizer smoke: encode(ab) merge failed");

    // ignore_merges: whole piece in vocab → single token without needing merges for internals
    const std::string jsonIgnore = R"json({
  "added_tokens": [{"id": 6, "content": "hello", "special": false}],
  "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "use_regex": false},
  "decoder": {"type": "ByteLevel", "add_prefix_space": false, "use_regex": false},
  "model": {
    "type": "BPE",
    "ignore_merges": true,
    "vocab": {
      "h": 0, "e": 1, "l": 2, "o": 3, "he": 4, "ll": 5, "hello": 6
    },
    "merges": ["h e", "l l"]
  }
})json";
    writeText(path, jsonIgnore);
    Tokenizer tok2 = Tokenizer::load(path.string());
    if (!tok2.ignoreMerges())
        throw std::runtime_error("HfTokenizer smoke: ignore_merges not parsed");
    const std::vector<int> hello = tok2.encode("hello", false);
    if (hello.size() != 1 || hello[0] != 6)
        throw std::runtime_error("HfTokenizer smoke: ignore_merges should keep 'hello' atomic");

    // Reject WordPiece
    bool rejected = false;
    try {
        writeText(path, R"json({"model":{"type":"WordPiece","vocab":{"a":0},"merges":[]},"pre_tokenizer":null,"decoder":null})json");
        (void)Tokenizer::load(path.string());
    } catch (const std::exception&) {
        rejected = true;
    }
    if (!rejected)
        throw std::runtime_error("HfTokenizer smoke: should reject WordPiece");

    fs::remove(path);
    SmokeLog::result(
        "HuggingFace tokenizer",
        "BPE ByteLevel encode/decode=ok  ignore_merges=ok  reject=WordPiece");
}

} // namespace HuggingFace
