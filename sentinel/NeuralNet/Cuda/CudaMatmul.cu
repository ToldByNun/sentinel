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

    const dim3 blockDimension(CudaMatmul::tileSize, CudaMatmul::tileSize);
    const dim3 gridDimension((columnCount + CudaMatmul::tileSize - 1) / CudaMatmul::tileSize, (rowCount + CudaMatmul::tileSize - 1) / CudaMatmul::tileSize);

    cudaEvent_t kernelStartEvent = nullptr;
    cudaEvent_t kernelStopEvent = nullptr;
    if (timing != nullptr) {
        CudaMatmul::throwIfCudaFailed(cudaEventCreate(&kernelStartEvent), "cudaEventCreate start");
        CudaMatmul::throwIfCudaFailed(cudaEventCreate(&kernelStopEvent), "cudaEventCreate stop");
        CudaMatmul::throwIfCudaFailed(cudaEventRecord(kernelStartEvent), "cudaEventRecord start");
    }

    CudaMatmulSharedMemoryEntry<<<gridDimension, blockDimension>>>(this->deviceLeft.deviceData, this->deviceRight.deviceData, this->deviceOut.deviceData, rowCount, columnCount, sharedCount);
    CudaMatmul::throwIfCudaFailed(cudaGetLastError(), "CudaMatmulSharedMemoryEntry launch");

    if (timing != nullptr) {
        CudaMatmul::throwIfCudaFailed(cudaEventRecord(kernelStopEvent), "cudaEventRecord stop");
        CudaMatmul::throwIfCudaFailed(cudaEventSynchronize(kernelStopEvent), "cudaEventSynchronize stop");
        float kernelMilliseconds = 0.0f;
        CudaMatmul::throwIfCudaFailed(cudaEventElapsedTime(&kernelMilliseconds, kernelStartEvent, kernelStopEvent), "cudaEventElapsedTime");
        timing->kernelMilliseconds = static_cast<double>(kernelMilliseconds);
        cudaEventDestroy(kernelStartEvent);
        cudaEventDestroy(kernelStopEvent);
    } else {
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
    Matrix gpuOut;
    CudaMatmulTiming coldTiming;
    matmul.multiplyIntoTimed(left, right, gpuOut, coldTiming);

    CudaMatmulTiming warmTiming;
    matmul.multiplyIntoTimed(left, right, gpuOut, warmTiming);

    const float maximumDifference = CudaMatmul::maximumAbsoluteDifference(cpuOut, gpuOut);
    std::printf("CUDA matmul smoke: %zux%zu  cpu=%.2fms  maxAbsDiff=%.6g\n", matrixSize, matrixSize, cpuMilliseconds, maximumDifference);
    std::printf("  cold: ensure=%.2fms  h2d=%.2fms  kernel=%.2fms  d2h=%.2fms  total=%.2fms\n", coldTiming.ensureCapacityMilliseconds, coldTiming.hostToDeviceMilliseconds, coldTiming.kernelMilliseconds, coldTiming.deviceToHostMilliseconds, coldTiming.totalMilliseconds());
    std::printf("  warm: ensure=%.2fms  h2d=%.2fms  kernel=%.2fms  d2h=%.2fms  total=%.2fms\n", warmTiming.ensureCapacityMilliseconds, warmTiming.hostToDeviceMilliseconds, warmTiming.kernelMilliseconds, warmTiming.deviceToHostMilliseconds, warmTiming.totalMilliseconds());
}
