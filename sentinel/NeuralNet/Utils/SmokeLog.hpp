#ifndef SMOKELOG_HPP
#define SMOKELOG_HPP

#include <cstdarg>
#include <cstdio>

/// <summary>compact aligned console lines for smoke demos</summary>
class SmokeLog {
public:
    /// <summary>blank line then section title</summary>
    static void section(const char* title) {
        std::printf("\n-- %s --\n", title);
    }

    /// <summary>one line note without metrics</summary>
    static void note(const char* text) {
        std::printf("  %s\n", text);
    }

    /// <summary>skip when no CUDA device</summary>
    static void skip(const char* name) {
        std::printf("  %-34s  skipped (no CUDA)\n", name);
    }

    /// <summary>name plus free form metrics after the label</summary>
    static void result(const char* name, const char* format, ...) {
        std::printf("  %-34s  ", name);
        va_list args;
        va_start(args, format);
        std::vprintf(format, args);
        va_end(args);
        std::printf("\n");
    }
};

#endif // SMOKELOG_HPP
