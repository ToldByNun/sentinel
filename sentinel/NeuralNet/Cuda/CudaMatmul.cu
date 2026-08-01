#include "CudaMatmul.hpp"
#include "CudaAmp.hpp"

#include "../Utils/SmokeLog.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <stdexcept>
#include <string>
#include <utility>

struct CublasLtGemmState {
    cublasLtHandle_t handle;
    bool initAttempted;
    bool initSucceeded;
    CudaDeviceBuffer workspace;
    static constexpr size_t workspaceBytes = 16 * 1024 * 1024;

    CublasLtGemmState() : handle(nullptr), initAttempted(false), initSucceeded(false) {}

    ~CublasLtGemmState() {
        if (this->handle == nullptr) return;
        cublasLtDestroy(this->handle);
        this->handle = nullptr;
    }
};

static CublasLtGemmState& cublasLtGemmState() {
    static CublasLtGemmState state;
    return state;
}

static bool ensureCublasLtHandle() {
    CublasLtGemmState& state = cublasLtGemmState();
    if (state.initAttempted) return state.initSucceeded;

    state.initAttempted = true;
    if (cublasLtCreate(&state.handle) != CUBLAS_STATUS_SUCCESS) return false;

    state.initSucceeded = true;
    return true;
}

CudaDeviceBuffer::CudaDeviceBuffer() : deviceData(nullptr), capacityBytes(0) {}

CudaDeviceBuffer::~CudaDeviceBuffer() {
    this->free();
}

CudaDeviceBuffer::CudaDeviceBuffer(CudaDeviceBuffer&& other) noexcept : deviceData(other.deviceData), capacityBytes(other.capacityBytes) {
    other.deviceData = nullptr;
    other.capacityBytes = 0;
}

CudaDeviceBuffer& CudaDeviceBuffer::operator=(CudaDeviceBuffer&& other) noexcept {
    if (this == &other) return *this;

    this->free();
    this->deviceData = other.deviceData;
    this->capacityBytes = other.capacityBytes;
    other.deviceData = nullptr;
    other.capacityBytes = 0;
    return *this;
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
    this->copyBytesFromHost(hostData, byteCount);
}

void CudaDeviceBuffer::copyBytesFromHost(const void* hostData, size_t byteCount) {
    if (hostData == nullptr) throw std::invalid_argument("CudaDeviceBuffer::copyBytesFromHost null hostData");
    if (byteCount == 0) return;
    if (this->deviceData == nullptr) throw std::logic_error("CudaDeviceBuffer::copyBytesFromHost empty buffer");
    if (byteCount > this->capacityBytes) throw std::invalid_argument("CudaDeviceBuffer::copyBytesFromHost exceeds capacity");

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

CudaIntBuffer::CudaIntBuffer() : deviceData(nullptr), capacityCount(0) {}

CudaIntBuffer::~CudaIntBuffer() {
    this->free();
}

CudaIntBuffer::CudaIntBuffer(CudaIntBuffer&& other) noexcept : deviceData(other.deviceData), capacityCount(other.capacityCount) {
    other.deviceData = nullptr;
    other.capacityCount = 0;
}

CudaIntBuffer& CudaIntBuffer::operator=(CudaIntBuffer&& other) noexcept {
    if (this == &other) return *this;

    this->free();
    this->deviceData = other.deviceData;
    this->capacityCount = other.capacityCount;
    other.deviceData = nullptr;
    other.capacityCount = 0;
    return *this;
}

void CudaIntBuffer::ensureCapacity(size_t requiredCount) {
    if (requiredCount == 0) return;
    if (this->capacityCount >= requiredCount) return;

    this->free();

    int* allocated = nullptr;
    CudaMatmul::throwIfCudaFailed(cudaMalloc(&allocated, requiredCount * sizeof(int)), "cudaMalloc ints");
    this->deviceData = allocated;
    this->capacityCount = requiredCount;
}

void CudaIntBuffer::copyFromHost(const int* hostData, size_t count) {
    if (hostData == nullptr) throw std::invalid_argument("CudaIntBuffer::copyFromHost null hostData");
    if (count == 0) return;
    if (this->deviceData == nullptr) throw std::logic_error("CudaIntBuffer::copyFromHost empty buffer");
    if (count > this->capacityCount) throw std::invalid_argument("CudaIntBuffer::copyFromHost exceeds capacity");

    CudaMatmul::throwIfCudaFailed(cudaMemcpy(this->deviceData, hostData, count * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy ints host to device");
}

void CudaIntBuffer::free() {
    if (this->deviceData == nullptr) return;

    cudaFree(this->deviceData);
    this->deviceData = nullptr;
    this->capacityCount = 0;
}

CudaByteBuffer::CudaByteBuffer() : deviceData(nullptr), capacityCount(0) {}

CudaByteBuffer::~CudaByteBuffer() {
    this->free();
}

CudaByteBuffer::CudaByteBuffer(CudaByteBuffer&& other) noexcept : deviceData(other.deviceData), capacityCount(other.capacityCount) {
    other.deviceData = nullptr;
    other.capacityCount = 0;
}

CudaByteBuffer& CudaByteBuffer::operator=(CudaByteBuffer&& other) noexcept {
    if (this == &other) return *this;

    this->free();
    this->deviceData = other.deviceData;
    this->capacityCount = other.capacityCount;
    other.deviceData = nullptr;
    other.capacityCount = 0;
    return *this;
}

void CudaByteBuffer::ensureCapacity(size_t requiredCount) {
    if (requiredCount == 0) return;
    if (this->capacityCount >= requiredCount) return;

    this->free();

    signed char* allocated = nullptr;
    CudaMatmul::throwIfCudaFailed(cudaMalloc(&allocated, requiredCount * sizeof(signed char)), "cudaMalloc bytes");
    this->deviceData = allocated;
    this->capacityCount = requiredCount;
}

void CudaByteBuffer::zeroInPlace() {
    if (this->deviceData == nullptr || this->capacityCount == 0) return;
    CudaMatmul::throwIfCudaFailed(cudaMemset(this->deviceData, 0, this->capacityCount * sizeof(signed char)), "CudaByteBuffer::zeroInPlace");
}

void CudaByteBuffer::free() {
    if (this->deviceData == nullptr) return;

    cudaFree(this->deviceData);
    this->deviceData = nullptr;
    this->capacityCount = 0;
}

CudaMatrix::CudaMatrix() : rows(0), cols(0) {}

CudaMatrix::CudaMatrix(CudaMatrix&& other) noexcept : rows(other.rows), cols(other.cols), buffer(std::move(other.buffer)) {
    other.rows = 0;
    other.cols = 0;
}

CudaMatrix& CudaMatrix::operator=(CudaMatrix&& other) noexcept {
    if (this == &other) return *this;

    this->rows = other.rows;
    this->cols = other.cols;
    this->buffer = std::move(other.buffer);
    other.rows = 0;
    other.cols = 0;
    return *this;
}

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

void CudaMatrix::free() {
    this->buffer.free();
    this->rows = 0;
    this->cols = 0;
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

void CudaMatrix::multiplyInto(const CudaMatrix& left, const CudaMatrix& right, CudaMatrix& out, bool transposeLeft, bool transposeRight) {
    if (left.empty() || right.empty()) throw std::invalid_argument("CudaMatrix::multiplyInto empty input");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaMatrix::multiplyInto no CUDA device");

    const size_t leftRows = transposeLeft ? left.cols : left.rows;
    const size_t leftCols = transposeLeft ? left.rows : left.cols;
    const size_t rightRows = transposeRight ? right.cols : right.rows;
    const size_t rightCols = transposeRight ? right.rows : right.cols;
    if (leftCols != rightRows) throw std::invalid_argument("CudaMatrix::multiplyInto shape mismatch");

    out.ensureSize(leftRows, rightCols);
    CudaMatmul::launchSharedMemoryMatmul(left.buffer.deviceData, right.buffer.deviceData, out.buffer.deviceData, static_cast<int>(leftRows), static_cast<int>(rightCols), static_cast<int>(leftCols), transposeLeft, transposeRight, nullptr);
}

void CudaMatrix::multiplyPointersInto(const float* left, size_t leftRows, size_t leftCols, const float* right, size_t rightRows, size_t rightCols, float* out, bool transposeLeft, bool transposeRight) {
    if (left == nullptr || right == nullptr || out == nullptr) throw std::invalid_argument("CudaMatrix::multiplyPointersInto null pointer");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaMatrix::multiplyPointersInto no CUDA device");
    if (leftRows == 0 || leftCols == 0 || rightRows == 0 || rightCols == 0) throw std::invalid_argument("CudaMatrix::multiplyPointersInto empty shape");

    const size_t outRows = transposeLeft ? leftCols : leftRows;
    const size_t outCols = transposeRight ? rightRows : rightCols;
    const size_t sharedCount = transposeLeft ? leftRows : leftCols;
    const size_t rightShared = transposeRight ? rightCols : rightRows;
    if (sharedCount != rightShared) throw std::invalid_argument("CudaMatrix::multiplyPointersInto shape mismatch");

    CudaMatmul::launchSharedMemoryMatmul(left, right, out, static_cast<int>(outRows), static_cast<int>(outCols), static_cast<int>(sharedCount), transposeLeft, transposeRight, nullptr);
}

void CudaMatrix::multiplyIntoTimed(const CudaMatrix& left, const CudaMatrix& right, CudaMatrix& out, double& kernelMilliseconds, bool transposeLeft, bool transposeRight) {
    if (left.empty() || right.empty()) throw std::invalid_argument("CudaMatrix::multiplyIntoTimed empty input");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaMatrix::multiplyIntoTimed no CUDA device");

    const size_t leftRows = transposeLeft ? left.cols : left.rows;
    const size_t leftCols = transposeLeft ? left.rows : left.cols;
    const size_t rightRows = transposeRight ? right.cols : right.rows;
    const size_t rightCols = transposeRight ? right.rows : right.cols;
    if (leftCols != rightRows) throw std::invalid_argument("CudaMatrix::multiplyIntoTimed shape mismatch");

    out.ensureSize(leftRows, rightCols);
    CudaMatmul::launchSharedMemoryMatmul(left.buffer.deviceData, right.buffer.deviceData, out.buffer.deviceData, static_cast<int>(leftRows), static_cast<int>(rightCols), static_cast<int>(leftCols), transposeLeft, transposeRight, &kernelMilliseconds);
}

double CudaMatmulTiming::totalMilliseconds() const {
    return this->ensureCapacityMilliseconds + this->hostToDeviceMilliseconds + this->kernelMilliseconds + this->deviceToHostMilliseconds;
}

CudaMatmul::CudaMatmul() {}

void CudaMatmul::throwIfCudaFailed(int status, const char* operationName) {
    if (status == cudaSuccess) return;
    throw std::runtime_error(std::string(operationName) + ": " + cudaGetErrorString(static_cast<cudaError_t>(status)));
}

void CudaMatmul::throwIfCublasLtFailed(int status, const char* operationName) {
    if (status == CUBLAS_STATUS_SUCCESS) return;
    throw std::runtime_error(std::string(operationName) + ": cuBLASLt status " + std::to_string(status));
}

__device__ void CudaMatmul::runSharedMemoryMatmul(const float* left, const float* right, float* out, int rowCount, int columnCount, int sharedCount, bool transposeLeft, bool transposeRight) {
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
        if (outputRow < rowCount && leftColumn < sharedCount) {
            if (transposeLeft)
                leftTile[threadIdx.y][threadIdx.x] = left[leftColumn * rowCount + outputRow];
            else
                leftTile[threadIdx.y][threadIdx.x] = left[outputRow * sharedCount + leftColumn];
        }

        rightTile[threadIdx.y][threadIdx.x] = 0.0f;
        if (rightRow < sharedCount && outputColumn < columnCount) {
            if (transposeRight)
                rightTile[threadIdx.y][threadIdx.x] = right[outputColumn * sharedCount + rightRow];
            else
                rightTile[threadIdx.y][threadIdx.x] = right[rightRow * columnCount + outputColumn];
        }

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

__global__ void CudaMatmulSharedMemoryEntry(const float* left, const float* right, float* out, int rowCount, int columnCount, int sharedCount, bool transposeLeft, bool transposeRight) {
    CudaMatmul::runSharedMemoryMatmul(left, right, out, rowCount, columnCount, sharedCount, transposeLeft, transposeRight);
}

bool CudaMatmul::launchCublasLtMatmul(const float* deviceLeft, const float* deviceRight, float* deviceOut, int rowCount, int columnCount, int sharedCount, bool transposeLeft, bool transposeRight, double* kernelMilliseconds) {
    if (!ensureCublasLtHandle()) return false;

    CublasLtGemmState& state = cublasLtGemmState();
    try {
        state.workspace.ensureCapacity(CublasLtGemmState::workspaceBytes);
    } catch (...) {
        return false;
    }

    cublasLtMatmulDesc_t matmulDesc = nullptr;
    cublasLtMatrixLayout_t layoutLeft = nullptr;
    cublasLtMatrixLayout_t layoutRight = nullptr;
    cublasLtMatrixLayout_t layoutOut = nullptr;

    auto destroyDescriptors = [&]() {
        if (layoutOut != nullptr) cublasLtMatrixLayoutDestroy(layoutOut);
        if (layoutRight != nullptr) cublasLtMatrixLayoutDestroy(layoutRight);
        if (layoutLeft != nullptr) cublasLtMatrixLayoutDestroy(layoutLeft);
        if (matmulDesc != nullptr) cublasLtMatmulDescDestroy(matmulDesc);
    };

    if (cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32F_FAST_TF32, CUDA_R_32F) != CUBLAS_STATUS_SUCCESS) return false;

    const cublasOperation_t transLeft = transposeLeft ? CUBLAS_OP_T : CUBLAS_OP_N;
    const cublasOperation_t transRight = transposeRight ? CUBLAS_OP_T : CUBLAS_OP_N;
    if (cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transLeft, sizeof(transLeft)) != CUBLAS_STATUS_SUCCESS) {
        destroyDescriptors();
        return false;
    }
    if (cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transRight, sizeof(transRight)) != CUBLAS_STATUS_SUCCESS) {
        destroyDescriptors();
        return false;
    }

    const int leftRows = transposeLeft ? sharedCount : rowCount;
    const int leftCols = transposeLeft ? rowCount : sharedCount;
    const int rightRows = transposeRight ? columnCount : sharedCount;
    const int rightCols = transposeRight ? sharedCount : columnCount;
    const cublasLtOrder_t rowMajorOrder = CUBLASLT_ORDER_ROW;

    if (cublasLtMatrixLayoutCreate(&layoutLeft, CUDA_R_32F, leftRows, leftCols, leftCols) != CUBLAS_STATUS_SUCCESS) {
        destroyDescriptors();
        return false;
    }
    if (cublasLtMatrixLayoutCreate(&layoutRight, CUDA_R_32F, rightRows, rightCols, rightCols) != CUBLAS_STATUS_SUCCESS) {
        destroyDescriptors();
        return false;
    }
    if (cublasLtMatrixLayoutCreate(&layoutOut, CUDA_R_32F, rowCount, columnCount, columnCount) != CUBLAS_STATUS_SUCCESS) {
        destroyDescriptors();
        return false;
    }

    if (cublasLtMatrixLayoutSetAttribute(layoutLeft, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowMajorOrder, sizeof(rowMajorOrder)) != CUBLAS_STATUS_SUCCESS) {
        destroyDescriptors();
        return false;
    }
    if (cublasLtMatrixLayoutSetAttribute(layoutRight, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowMajorOrder, sizeof(rowMajorOrder)) != CUBLAS_STATUS_SUCCESS) {
        destroyDescriptors();
        return false;
    }
    if (cublasLtMatrixLayoutSetAttribute(layoutOut, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowMajorOrder, sizeof(rowMajorOrder)) != CUBLAS_STATUS_SUCCESS) {
        destroyDescriptors();
        return false;
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    void* workspacePointer = state.workspace.deviceData;
    const size_t workspaceSize = state.workspace.capacityBytes;

    cudaEvent_t kernelStartEvent = nullptr;
    cudaEvent_t kernelStopEvent = nullptr;
    if (kernelMilliseconds != nullptr) {
        if (cudaEventCreate(&kernelStartEvent) != cudaSuccess) {
            destroyDescriptors();
            return false;
        }
        if (cudaEventCreate(&kernelStopEvent) != cudaSuccess) {
            cudaEventDestroy(kernelStartEvent);
            destroyDescriptors();
            return false;
        }
        if (cudaEventRecord(kernelStartEvent) != cudaSuccess) {
            cudaEventDestroy(kernelStopEvent);
            cudaEventDestroy(kernelStartEvent);
            destroyDescriptors();
            return false;
        }
    }

    const cublasStatus_t matmulStatus = cublasLtMatmul(state.handle, matmulDesc, &alpha, deviceLeft, layoutLeft, deviceRight, layoutRight, &beta, deviceOut, layoutOut, deviceOut, layoutOut, nullptr, workspacePointer, workspaceSize, nullptr);

    if (matmulStatus != CUBLAS_STATUS_SUCCESS) {
        if (kernelStartEvent != nullptr) cudaEventDestroy(kernelStartEvent);
        if (kernelStopEvent != nullptr) cudaEventDestroy(kernelStopEvent);
        destroyDescriptors();
        return false;
    }

    if (kernelMilliseconds != nullptr) {
        if (cudaEventRecord(kernelStopEvent) != cudaSuccess || cudaEventSynchronize(kernelStopEvent) != cudaSuccess) {
            cudaEventDestroy(kernelStopEvent);
            cudaEventDestroy(kernelStartEvent);
            destroyDescriptors();
            return false;
        }

        float elapsedMilliseconds = 0.0f;
        if (cudaEventElapsedTime(&elapsedMilliseconds, kernelStartEvent, kernelStopEvent) != cudaSuccess) {
            cudaEventDestroy(kernelStopEvent);
            cudaEventDestroy(kernelStartEvent);
            destroyDescriptors();
            return false;
        }

        *kernelMilliseconds = static_cast<double>(elapsedMilliseconds);
        cudaEventDestroy(kernelStopEvent);
        cudaEventDestroy(kernelStartEvent);
    }

    destroyDescriptors();
    return true;
}

void CudaMatmul::launchSharedMemoryMatmulKernel(const float* deviceLeft, const float* deviceRight, float* deviceOut, int rowCount, int columnCount, int sharedCount, bool transposeLeft, bool transposeRight, double* kernelMilliseconds) {
    const dim3 blockDimension(CudaMatmul::tileSize, CudaMatmul::tileSize);
    const dim3 gridDimension((columnCount + CudaMatmul::tileSize - 1) / CudaMatmul::tileSize, (rowCount + CudaMatmul::tileSize - 1) / CudaMatmul::tileSize);

    if (kernelMilliseconds == nullptr) {
        CudaMatmulSharedMemoryEntry<<<gridDimension, blockDimension>>>(deviceLeft, deviceRight, deviceOut, rowCount, columnCount, sharedCount, transposeLeft, transposeRight);
        CudaMatmul::throwIfCudaFailed(cudaGetLastError(), "CudaMatmulSharedMemoryEntry launch");
        return;
    }

    cudaEvent_t kernelStartEvent = nullptr;
    cudaEvent_t kernelStopEvent = nullptr;
    CudaMatmul::throwIfCudaFailed(cudaEventCreate(&kernelStartEvent), "cudaEventCreate start");
    CudaMatmul::throwIfCudaFailed(cudaEventCreate(&kernelStopEvent), "cudaEventCreate stop");
    CudaMatmul::throwIfCudaFailed(cudaEventRecord(kernelStartEvent), "cudaEventRecord start");

    CudaMatmulSharedMemoryEntry<<<gridDimension, blockDimension>>>(deviceLeft, deviceRight, deviceOut, rowCount, columnCount, sharedCount, transposeLeft, transposeRight);
    CudaMatmul::throwIfCudaFailed(cudaGetLastError(), "CudaMatmulSharedMemoryEntry launch");
    CudaMatmul::throwIfCudaFailed(cudaEventRecord(kernelStopEvent), "cudaEventRecord stop");
    CudaMatmul::throwIfCudaFailed(cudaEventSynchronize(kernelStopEvent), "cudaEventSynchronize stop");

    float elapsedMilliseconds = 0.0f;
    CudaMatmul::throwIfCudaFailed(cudaEventElapsedTime(&elapsedMilliseconds, kernelStartEvent, kernelStopEvent), "cudaEventElapsedTime");
    *kernelMilliseconds = static_cast<double>(elapsedMilliseconds);

    cudaEventDestroy(kernelStartEvent);
    cudaEventDestroy(kernelStopEvent);
}

void CudaMatmul::launchSharedMemoryMatmul(const float* deviceLeft, const float* deviceRight, float* deviceOut, int rowCount, int columnCount, int sharedCount, bool transposeLeft, bool transposeRight, double* kernelMilliseconds) {
    if (deviceLeft == nullptr || deviceRight == nullptr || deviceOut == nullptr) throw std::invalid_argument("CudaMatmul::launchSharedMemoryMatmul null device pointer");
    if (rowCount <= 0 || columnCount <= 0 || sharedCount <= 0) throw std::invalid_argument("CudaMatmul::launchSharedMemoryMatmul invalid shape");

    if (CudaAmp::launchCublasLtMatmulFp16(deviceLeft, deviceRight, deviceOut, rowCount, columnCount, sharedCount, transposeLeft, transposeRight, kernelMilliseconds)) return;
    if (CudaMatmul::launchCublasLtMatmul(deviceLeft, deviceRight, deviceOut, rowCount, columnCount, sharedCount, transposeLeft, transposeRight, kernelMilliseconds)) return;

    CudaMatmul::launchSharedMemoryMatmulKernel(deviceLeft, deviceRight, deviceOut, rowCount, columnCount, sharedCount, transposeLeft, transposeRight, kernelMilliseconds);
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
        CudaMatmul::launchSharedMemoryMatmul(this->deviceLeft.deviceData, this->deviceRight.deviceData, this->deviceOut.deviceData, rowCount, columnCount, sharedCount, false, false, &kernelMilliseconds);
        timing->kernelMilliseconds = kernelMilliseconds;
    } else {
        CudaMatmul::launchSharedMemoryMatmul(this->deviceLeft.deviceData, this->deviceRight.deviceData, this->deviceOut.deviceData, rowCount, columnCount, sharedCount, false, false, nullptr);
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
        SmokeLog::skip("matmul");
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

    const char* gemmBackend = ensureCublasLtHandle() ? "cuBLASLt TF32" : "shared-mem";
    SmokeLog::result("matmul", "%zux%zu  backend=%s  cpu=%.2fms  gpu=%.2fms  warm=%.2fms  chain=%.3fms/gemm  diff=%.2e",
        matrixSize, matrixSize, gemmBackend, cpuMilliseconds, deviceKernelMilliseconds, warmTiming.totalMilliseconds(),
        chainMilliseconds / static_cast<double>(chainCount), (std::max)(hostPathDifference, devicePathDifference));
    (void)coldTiming;
    (void)uploadMilliseconds;
    (void)downloadMilliseconds;
}
