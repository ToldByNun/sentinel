#include "CudaOps.hpp"

#include <cuda_runtime.h>
#include <stdexcept>

__device__ void CudaOps::runBroadcastBiasAddInPlace(float* product, const float* bias, int rowCount, int columnCount) {
    const int elementCount = rowCount * columnCount;
    const int index = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (index >= elementCount) return;

    const int row = index / columnCount;
    product[index] += bias[row];
}

__device__ void CudaOps::runSiluInto(const float* input, float* out, int elementCount) {
    const int index = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (index >= elementCount) return;

    const float value = input[index];
    const float sigmoid = 1.0f / (1.0f + expf(-value));
    out[index] = value * sigmoid;
}

__device__ void CudaOps::runMultiplyElementwiseInto(const float* left, const float* right, float* out, int elementCount) {
    const int index = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (index >= elementCount) return;

    out[index] = left[index] * right[index];
}

__global__ void CudaOpsBroadcastBiasAddEntry(float* product, const float* bias, int rowCount, int columnCount) {
    CudaOps::runBroadcastBiasAddInPlace(product, bias, rowCount, columnCount);
}

__global__ void CudaOpsSiluEntry(const float* input, float* out, int elementCount) {
    CudaOps::runSiluInto(input, out, elementCount);
}

__global__ void CudaOpsMultiplyElementwiseEntry(const float* left, const float* right, float* out, int elementCount) {
    CudaOps::runMultiplyElementwiseInto(left, right, out, elementCount);
}

void CudaOps::broadcastBiasAddInPlace(CudaMatrix& product, const CudaMatrix& bias) {
    if (product.empty()) throw std::invalid_argument("CudaOps::broadcastBiasAddInPlace empty product");
    if (bias.empty()) throw std::invalid_argument("CudaOps::broadcastBiasAddInPlace empty bias");
    if (bias.cols != 1) throw std::invalid_argument("CudaOps::broadcastBiasAddInPlace bias must be a column");
    if (bias.rows != product.rows) throw std::invalid_argument("CudaOps::broadcastBiasAddInPlace row mismatch");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaOps::broadcastBiasAddInPlace no CUDA device");

    const int rowCount = static_cast<int>(product.rows);
    const int columnCount = static_cast<int>(product.cols);
    const int elementCount = rowCount * columnCount;
    const int blockCount = (elementCount + CudaOps::threadCount - 1) / CudaOps::threadCount;
    CudaOpsBroadcastBiasAddEntry<<<blockCount, CudaOps::threadCount>>>(product.buffer.deviceData, bias.buffer.deviceData, rowCount, columnCount);
    CudaMatmul::throwIfCudaFailed(cudaGetLastError(), "CudaOpsBroadcastBiasAddEntry launch");
}

void CudaOps::siluInto(const CudaMatrix& input, CudaMatrix& out) {
    if (input.empty()) throw std::invalid_argument("CudaOps::siluInto empty input");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaOps::siluInto no CUDA device");

    out.ensureSize(input.rows, input.cols);
    const int elementCount = static_cast<int>(input.elementCount());
    const int blockCount = (elementCount + CudaOps::threadCount - 1) / CudaOps::threadCount;
    CudaOpsSiluEntry<<<blockCount, CudaOps::threadCount>>>(input.buffer.deviceData, out.buffer.deviceData, elementCount);
    CudaMatmul::throwIfCudaFailed(cudaGetLastError(), "CudaOpsSiluEntry launch");
}

void CudaOps::multiplyElementwiseInto(const CudaMatrix& left, const CudaMatrix& right, CudaMatrix& out) {
    if (left.empty() || right.empty()) throw std::invalid_argument("CudaOps::multiplyElementwiseInto empty input");
    if (left.rows != right.rows || left.cols != right.cols) throw std::invalid_argument("CudaOps::multiplyElementwiseInto shape mismatch");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaOps::multiplyElementwiseInto no CUDA device");

    out.ensureSize(left.rows, left.cols);
    const int elementCount = static_cast<int>(left.elementCount());
    const int blockCount = (elementCount + CudaOps::threadCount - 1) / CudaOps::threadCount;
    CudaOpsMultiplyElementwiseEntry<<<blockCount, CudaOps::threadCount>>>(left.buffer.deviceData, right.buffer.deviceData, out.buffer.deviceData, elementCount);
    CudaMatmul::throwIfCudaFailed(cudaGetLastError(), "CudaOpsMultiplyElementwiseEntry launch");
}
