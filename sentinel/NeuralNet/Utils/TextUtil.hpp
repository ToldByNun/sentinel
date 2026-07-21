#ifndef TEXTUTIL_HPP
#define TEXTUTIL_HPP

#include <string>

/// <summary>small string helpers</summary>
class TextUtil {
public:
    /// <summary>keep at most maximumCharacters; otherwise cut from the front</summary>
    static std::string truncate(const std::string& text, size_t maximumCharacters);
};

#endif // TEXTUTIL_HPP
