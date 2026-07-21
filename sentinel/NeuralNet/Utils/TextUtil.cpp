#include "TextUtil.hpp"

std::string TextUtil::truncate(const std::string& text, size_t maximumCharacters) {
    if (text.size() <= maximumCharacters) return text;
    return text.substr(0, maximumCharacters);
}
