#ifndef SAFETENSORS_HPP
#define SAFETENSORS_HPP

#include "../Math/Matrix.hpp"

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

/// <summary>
/// Minimal safetensors I/O — zero third-party deps.
/// Spec: https://huggingface.co/docs/safetensors
/// Save: F32 row-major only.
/// Load: F32, BF16, and F16 (IEEE) → host F32 Matrix.
/// </summary>
namespace SafeTensors {

struct TensorInfo {
    std::string name;
    std::string dtype = "F32";
    std::vector<size_t> shape;
    size_t dataBegin = 0; // offset into data buffer
    size_t dataEnd = 0;
};

struct File {
    std::map<std::string, std::string> metadata;
    std::map<std::string, Matrix> tensors; // always F32 after load; shape [rows, cols] or [rows] as cols=1
};

/// <summary>write tensors + string metadata; F32 little-endian row-major</summary>
void save(const std::string& path, const File& file);

/// <summary>load tensors as host F32 (accepts F32 / BF16 / F16 storage dtypes)</summary>
File load(const std::string& path);

/// <summary>true if path looks like a safetensors file (header size + '{')</summary>
bool isSafeTensorsFile(const std::string& path);

/// <summary>helper: 2D matrix → tensor with shape [rows, cols]</summary>
void putMatrix(File& file, const std::string& name, const Matrix& matrix);

/// <summary>helper: require tensor present with exact 2D shape</summary>
Matrix requireMatrix(const File& file, const std::string& name, size_t rows, size_t cols);

/// <summary>helper: require tensor present; 1D [n] accepted as [n, 1]</summary>
Matrix requireMatrixFlexible(const File& file, const std::string& name, size_t rows, size_t cols);

/// <summary>write tiny BF16/F16 fixtures, load, check values vs F32 reference</summary>
void runHalfLoadSmokeDemo();

} // namespace SafeTensors

#endif // SAFETENSORS_HPP
