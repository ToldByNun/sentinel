#include "CudaMatmul.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

CudaDeviceBuffer::CudaDeviceBuffer() : deviceData(nullptr), capacityBytes(0) {}

CudaDeviceBuffer::~CudaDeviceBuffer() {
    this->free();
}

void CudaDeviceBuffer::ensureCapacity(size_t requiredBytes) {
    if (requiredBytes == 0) return;
    if (this->capacityBytes >= requiredBytes) return;

    this->free();

    float* allocated = nullptr;
    CudaMatmul::throwIfCudaFailed(cudaMalloc(&allocated, requiredBytes), "cudaMalloc");
    this->deviceData = allocated;
    this->capacityBytes = requiredBytes;
}

void CudaDeviceBuffer::copyFromHost(const float* hostData, size_t byteCount) {
    if (hostData == nullptr) throw std::invalid_argument("CudaDeviceBuffer::copyFromHost null hostData");
    if (byteCount == 0) return;
    if (this->deviceData == nullptr) throw std::logic_error("CudaDeviceBuffer::copyFromHost empty buffer");
    if (byteCount > this->capacityBytes) throw std::invalid_argument("CudaDeviceBuffer::copyFromHost exceeds capacity");

    CudaMatmul::throwIfCudaFailed(cudaMemcpy(this->deviceData, hostData, byteCount, cudaMemcpyHostToDevice), "cudaMemcpy host to device");
}

void CudaDeviceBuffer::copyToHost(float* hostData, size_t byteCount) const {
    if (hostData == nullptr) throw std::invalid_argument("CudaDeviceBuffer::copyToHost null hostData");
    if (byteCount == 0) return;
    if (this->deviceData == nullptr) throw std::logic_error("CudaDeviceBuffer::copyToHost empty buffer");
    if (byteCount > this->capacityBytes) throw std::invalid_argument("CudaDeviceBuffer::copyToHost exceeds capacity");

    CudaMatmul::throwIfCudaFailed(cudaMemcpy(hostData, this->deviceData, byteCount, cudaMemcpyDeviceToHost), "cudaMemcpy device to host");
}

void CudaDeviceBuffer::free() {
    if (this->deviceData == nullptr) return;

    cudaFree(this->deviceData);
    this->deviceData = nullptr;
    this->capacityBytes = 0;
}

CudaMatrix::CudaMatrix() : rows(0), cols(0) {}

bool CudaMatrix::empty() const {
    return this->rows == 0 || this->cols == 0;
}

size_t CudaMatrix::elementCount() const {
    return this->rows * this->cols;
}

size_t CudaMatrix::byteCount() const {
    return this->elementCount() * sizeof(float);
}

void CudaMatrix::ensureSize(size_t rowCount, size_t columnCount) {
    this->rows = rowCount;
    this->cols = columnCount;
    this->buffer.ensureCapacity(this->byteCount());
}

void CudaMatrix::upload(const Matrix& host) {
    if (host.empty()) throw std::invalid_argument("CudaMatrix::upload empty host");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaMatrix::upload no CUDA device");

    this->ensureSize(host.rows, host.cols);
    this->buffer.copyFromHost(host.data.data(), this->byteCount());
}

void CudaMatrix::downloadInto(Matrix& host) const {
    if (this->empty()) throw std::invalid_argument("CudaMatrix::downloadInto empty device matrix");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaMatrix::downloadInto no CUDA device");

    CudaMatmul::throwIfCudaFailed(cudaDeviceSynchronize(), "CudaMatrix::downloadInto synchronize");
    host.ensureSize(this->rows, this->cols);
    this->buffer.copyToHost(host.data.data(), this->byteCount());
}

Matrix CudaMatrix::download() const {
    Matrix host;
    this->downloadInto(host);
    return host;
}

void CudaMatrix::multiplyInto(const CudaMatrix& left, const CudaMatrix& right, CudaMatrix& out) {
    if (left.empty() || right.empty()) throw std::invalid_argument("CudaMatrix::multiplyInto empty input");
    if (left.cols != right.rows) throw std::invalid_argument("CudaMatrix::multiplyInto shape mismatch");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaMatrix::multiplyInto no CUDA device");

    out.ensureSize(left.rows, right.cols);
    CudaMatmul::launchSharedMemoryMatmul(left.buffer.deviceData, right.buffer.deviceData, out.buffer.deviceData, static_cast<int>(left.rows), static_cast<int>(right.cols), static_cast<int>(left.cols), nullptr);
}

void CudaMatrix::multiplyIntoTimed(const CudaMatrix& left, const CudaMatrix& right, CudaMatrix& out, double& kernelMilliseconds) {
    if (left.empty() || right.empty()) throw std::invalid_argument("CudaMatrix::multiplyIntoTimed empty input");
    if (left.cols != right.rows) throw std::invalid_argument("CudaMatrix::multiplyIntoTimed shape mismatch");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaMatrix::multiplyIntoTimed no CUDA device");

    out.ensureSize(left.rows, right.cols);
    CudaMatmul::launchSharedMemoryMatmul(left.buffer.deviceData, right.buffer.deviceData, out.buffer.deviceData, static_cast<int>(left.rows), static_cast<int>(right.cols), static_cast<int>(left.cols), &kernelMilliseconds);
}

double CudaMatmulTiming::totalMilliseconds() const {
    return this->ensureCapacityMilliseconds + this->hostToDeviceMilliseconds + this->kernelMilliseconds + this->deviceToHostMilliseconds;
}

CudaMatmul::CudaMatmul() {}

void CudaMatmul::throwIfCudaFailed(int status, const char* operationName) {
    if (status == cudaSuccess) return;
    throw std::runtime_error(std::string(operationName) + ": " + cudaGetErrorString(static_cast<cudaError_t>(status)));
}

__device__ void CudaMatmul::runSharedMemoryMatmul(const float* left, const float* right, float* out, int rowCount, int columnCount, int sharedCount) {
    __shared__ float leftTile[CudaMatmul::tileSize][CudaMatmul::tileSize];
    __shared__ float rightTile[CudaMatmul::tileSize][CudaMatmul::tileSize];

    const int outputRow = static_cast<int>(blockIdx.y) * CudaMatmul::tileSize + static_cast<int>(threadIdx.y);
    const int outputColumn = static_cast<int>(blockIdx.x) * CudaMatmul::tileSize + static_cast<int>(threadIdx.x);
    const int tileCount = (sharedCount + CudaMatmul::tileSize - 1) / CudaMatmul::tileSize;

    float accumulator = 0.0f;

    for (int tileIndex = 0; tileIndex < tileCount; ++tileIndex) {
        const int leftColumn = tileIndex * CudaMatmul::tileSize + static_cast<int>(threadIdx.x);
        const int rightRow = tileIndex * CudaMatmul::tileSize + static_cast<int>(threadIdx.y);

        leftTile[threadIdx.y][threadIdx.x] = 0.0f;
        if (outputRow < rowCount && leftColumn < sharedCount)
            leftTile[threadIdx.y][threadIdx.x] = left[outputRow * sharedCount + leftColumn];

        rightTile[threadIdx.y][threadIdx.x] = 0.0f;
        if (rightRow < sharedCount && outputColumn < columnCount)
            rightTile[threadIdx.y][threadIdx.x] = right[rightRow * columnCount + outputColumn];

        __syncthreads();

        #pragma unroll
        for (int sharedIndex = 0; sharedIndex < CudaMatmul::tileSize; ++sharedIndex)
            accumulator += leftTile[threadIdx.y][sharedIndex] * rightTile[sharedIndex][threadIdx.x];

        __syncthreads();
    }

    if (outputRow >= rowCount) return;
    if (outputColumn >= columnCount) return;
    out[outputRow * columnCount + outputColumn] = accumulator;
}

__global__ void CudaMatmulSharedMemoryEntry(const float* left, const float* right, float* out, int rowCount, int columnCount, int sharedCount) {
    CudaMatmul::runSharedMemoryMatmul(left, right, out, rowCount, columnCount, sharedCount);
}

void CudaMatmul::launchSharedMemoryMatmul(const float* deviceLeft, const float* deviceRight, float* deviceOut, int rowCount, int columnCount, int sharedCount, double* kernelMilliseconds) {
    if (deviceLeft == nullptr || deviceRight == nullptr || deviceOut == nullptr) throw std::invalid_argument("CudaMatmul::launchSharedMemoryMatmul null device pointer");
    if (rowCount <= 0 || columnCount <= 0 || sharedCount <= 0) throw std::invalid_argument("CudaMatmul::launchSharedMemoryMatmul invalid shape");

    const dim3 blockDimension(CudaMatmul::tileSize, CudaMatmul::tileSize);
    const dim3 gridDimension((columnCount + CudaMatmul::tileSize - 1) / CudaMatmul::tileSize, (rowCount + CudaMatmul::tileSize - 1) / CudaMatmul::tileSize);

    if (kernelMilliseconds == nullptr) {
        CudaMatmulSharedMemoryEntry<<<gridDimension, blockDimension>>>(deviceLeft, deviceRight, deviceOut, rowCount, columnCount, sharedCount);
        CudaMatmul::throwIfCudaFailed(cudaGetLastError(), "CudaMatmulSharedMemoryEntry launch");
        return;
    }

    cudaEvent_t kernelStartEvent = nullptr;
    cudaEvent_t kernelStopEvent = nullptr;
    CudaMatmul::throwIfCudaFailed(cudaEventCreate(&kernelStartEvent), "cudaEventCreate start");
    CudaMatmul::throwIfCudaFailed(cudaEventCreate(&kernelStopEvent), "cudaEventCreate stop");
    CudaMatmul::throwIfCudaFailed(cudaEventRecord(kernelStartEvent), "cudaEventRecord start");

    CudaMatmulSharedMemoryEntry<<<gridDimension, blockDimension>>>(deviceLeft, deviceRight, deviceOut, rowCount, columnCount, sharedCount);
    CudaMatmul::throwIfCudaFailed(cudaGetLastError(), "CudaMatmulSharedMemoryEntry launch");
    CudaMatmul::throwIfCudaFailed(cudaEventRecord(kernelStopEvent), "cudaEventRecord stop");
    CudaMatmul::throwIfCudaFailed(cudaEventSynchronize(kernelStopEvent), "cudaEventSynchronize stop");

    float elapsedMilliseconds = 0.0f;
    CudaMatmul::throwIfCudaFailed(cudaEventElapsedTime(&elapsedMilliseconds, kernelStartEvent, kernelStopEvent), "cudaEventElapsedTime");
    *kernelMilliseconds = static_cast<double>(elapsedMilliseconds);

    cudaEventDestroy(kernelStartEvent);
    cudaEventDestroy(kernelStopEvent);
}

Matrix CudaMatmul::makeRandomMatrix(size_t rowCount, size_t columnCount, unsigned seed) {
    Matrix matrix(rowCount, columnCount, 0.0f);
    unsigned state = seed;
    for (size_t index = 0; index < matrix.data.size(); ++index) {
        state = state * 1664525u + 1013904223u;
        matrix.data[index] = (static_cast<float>(state >> 8) / 16777216.0f) * 2.0f - 1.0f;
    }
    return matrix;
}

float CudaMatmul::maximumAbsoluteDifference(const Matrix& left, const Matrix& right) {
    float maximumDifference = 0.0f;
    for (size_t index = 0; index < left.data.size(); ++index)
        maximumDifference = (std::max)(maximumDifference, std::fabs(left.data[index] - right.data[index]));
    return maximumDifference;
}

bool CudaMatmul::isAvailable() {
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess) return false;
    if (deviceCount <= 0) return false;
    return true;
}

void CudaMatmul::multiplyIntoInternal(const Matrix& left, const Matrix& right, Matrix& out, CudaMatmulTiming* timing) {
    if (left.empty() || right.empty()) throw std::invalid_argument("CudaMatmul::multiplyIntoInternal empty input");
    if (left.cols != right.rows) throw std::invalid_argument("CudaMatmul::multiplyIntoInternal shape mismatch");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaMatmul::multiplyIntoInternal no CUDA device");

    const int rowCount = static_cast<int>(left.rows);
    const int sharedCount = static_cast<int>(left.cols);
    const int columnCount = static_cast<int>(right.cols);
    const size_t leftBytes = left.data.size() * sizeof(float);
    const size_t rightBytes = right.data.size() * sizeof(float);
    const size_t outBytes = static_cast<size_t>(rowCount) * static_cast<size_t>(columnCount) * sizeof(float);

    out.ensureSize(left.rows, right.cols);

    if (timing != nullptr) {
        timing->ensureCapacityMilliseconds = 0.0;
        timing->hostToDeviceMilliseconds = 0.0;
        timing->kernelMilliseconds = 0.0;
        timing->deviceToHostMilliseconds = 0.0;
    }

    const auto ensureCapacityStart = std::chrono::steady_clock::now();
    this->deviceLeft.ensureCapacity(leftBytes);
    this->deviceRight.ensureCapacity(rightBytes);
    this->deviceOut.ensureCapacity(outBytes);
    if (timing != nullptr)
        timing->ensureCapacityMilliseconds = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ensureCapacityStart).count();

    const auto hostToDeviceStart = std::chrono::steady_clock::now();
    this->deviceLeft.copyFromHost(left.data.data(), leftBytes);
    this->deviceRight.copyFromHost(right.data.data(), rightBytes);
    if (timing != nullptr)
        timing->hostToDeviceMilliseconds = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - hostToDeviceStart).count();

    double kernelMilliseconds = 0.0;
    if (timing != nullptr) {
        CudaMatmul::launchSharedMemoryMatmul(this->deviceLeft.deviceData, this->deviceRight.deviceData, this->deviceOut.deviceData, rowCount, columnCount, sharedCount, &kernelMilliseconds);
        timing->kernelMilliseconds = kernelMilliseconds;
    } else {
        CudaMatmul::launchSharedMemoryMatmul(this->deviceLeft.deviceData, this->deviceRight.deviceData, this->deviceOut.deviceData, rowCount, columnCount, sharedCount, nullptr);
        CudaMatmul::throwIfCudaFailed(cudaDeviceSynchronize(), "CudaMatmulSharedMemoryEntry synchronize");
    }

    const auto deviceToHostStart = std::chrono::steady_clock::now();
    this->deviceOut.copyToHost(out.data.data(), outBytes);
    if (timing != nullptr)
        timing->deviceToHostMilliseconds = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - deviceToHostStart).count();
}

void CudaMatmul::multiplyInto(const Matrix& left, const Matrix& right, Matrix& out) {
    this->multiplyIntoInternal(left, right, out, nullptr);
}

void CudaMatmul::multiplyIntoTimed(const Matrix& left, const Matrix& right, Matrix& out, CudaMatmulTiming& timing) {
    this->multiplyIntoInternal(left, right, out, &timing);
}

Matrix CudaMatmul::multiply(const Matrix& left, const Matrix& right) {
    Matrix out;
    this->multiplyInto(left, right, out);
    return out;
}

CudaMatmul& CudaMatmul::sharedWorkspace() {
    thread_local CudaMatmul workspace;
    return workspace;
}

void CudaMatmul::multiplyIntoShared(const Matrix& left, const Matrix& right, Matrix& out) {
    CudaMatmul::sharedWorkspace().multiplyInto(left, right, out);
}

void CudaMatmul::runSmokeDemo(size_t matrixSize) {
    if (!CudaMatmul::isAvailable()) {
        std::printf("CUDA matmul smoke: no device\n");
        return;
    }

    Matrix left = CudaMatmul::makeRandomMatrix(matrixSize, matrixSize, 7u);
    Matrix right = CudaMatmul::makeRandomMatrix(matrixSize, matrixSize, 11u);

    Matrix cpuOut;
    const auto cpuStart = std::chrono::steady_clock::now();
    Matrix::gemm(left, right, cpuOut);
    const double cpuMilliseconds = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - cpuStart).count();

    CudaMatmul matmul;
    Matrix hostPathOut;
    CudaMatmulTiming coldTiming;
    matmul.multiplyIntoTimed(left, right, hostPathOut, coldTiming);
    CudaMatmulTiming warmTiming;
    matmul.multiplyIntoTimed(left, right, hostPathOut, warmTiming);
    const float hostPathDifference = CudaMatmul::maximumAbsoluteDifference(cpuOut, hostPathOut);

    CudaMatrix deviceLeft;
    CudaMatrix deviceRight;
    CudaMatrix deviceOut;
    CudaMatrix deviceTemporary;

    const auto uploadStart = std::chrono::steady_clock::now();
    deviceLeft.upload(left);
    deviceRight.upload(right);
    const double uploadMilliseconds = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - uploadStart).count();

    double deviceKernelMilliseconds = 0.0;
    CudaMatrix::multiplyIntoTimed(deviceLeft, deviceRight, deviceOut, deviceKernelMilliseconds);

    const auto downloadStart = std::chrono::steady_clock::now();
    Matrix devicePathOut = deviceOut.download();
    const double downloadMilliseconds = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - downloadStart).count();
    const float devicePathDifference = CudaMatmul::maximumAbsoluteDifference(cpuOut, devicePathOut);

    const int chainCount = 8;
    const auto chainStart = std::chrono::steady_clock::now();
    CudaMatrix::multiplyInto(deviceLeft, deviceRight, deviceTemporary);
    for (int chainIndex = 1; chainIndex < chainCount; ++chainIndex) {
        if ((chainIndex % 2) == 1)
            CudaMatrix::multiplyInto(deviceTemporary, deviceRight, deviceOut);
        else
            CudaMatrix::multiplyInto(deviceOut, deviceRight, deviceTemporary);
    }
    CudaMatmul::throwIfCudaFailed(cudaDeviceSynchronize(), "device chain synchronize");
    const double chainMilliseconds = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - chainStart).count();

    std::printf("CUDA matmul smoke: %zux%zu  cpu=%.2fms\n", matrixSize, matrixSize, cpuMilliseconds);
    std::printf("  host-path cold: ensure=%.2fms  h2d=%.2fms  kernel=%.2fms  d2h=%.2fms  total=%.2fms  maxAbsDiff=%.6g\n", coldTiming.ensureCapacityMilliseconds, coldTiming.hostToDeviceMilliseconds, coldTiming.kernelMilliseconds, coldTiming.deviceToHostMilliseconds, coldTiming.totalMilliseconds(), hostPathDifference);
    std::printf("  host-path warm: ensure=%.2fms  h2d=%.2fms  kernel=%.2fms  d2h=%.2fms  total=%.2fms\n", warmTiming.ensureCapacityMilliseconds, warmTiming.hostToDeviceMilliseconds, warmTiming.kernelMilliseconds, warmTiming.deviceToHostMilliseconds, warmTiming.totalMilliseconds());
    std::printf("  device-resident: upload=%.2fms  kernel=%.2fms  download=%.2fms  maxAbsDiff=%.6g\n", uploadMilliseconds, deviceKernelMilliseconds, downloadMilliseconds, devicePathDifference);
    std::printf("  device-chain x%d gemm sync: %.2fms  (%.3fms/gemm)\n", chainCount, chainMilliseconds, chainMilliseconds / static_cast<double>(chainCount));
}
