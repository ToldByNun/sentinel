#include "CudaAmp.hpp"

#include "CudaLanguageModel.hpp"
#include "CudaOps.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

bool CudaAmp::preferMixedPrecision = false;
bool CudaAmp::useLossScaling = false;
CudaLossScaler CudaAmp::lossScaler = CudaLossScaler();
CudaDeviceBuffer CudaAmp::halfScratchLeft = CudaDeviceBuffer();
CudaDeviceBuffer CudaAmp::halfScratchRight = CudaDeviceBuffer();
CudaDeviceBuffer CudaAmp::nonFiniteFlag = CudaDeviceBuffer();

bool CudaAmp::lossScalingActive() {
    return CudaAmp::preferMixedPrecision && CudaAmp::useLossScaling;
}

void CudaAmp::resetLossScaler() {
    CudaAmp::lossScaler = CudaLossScaler();
}

CudaLossScaler::CudaLossScaler()
    : scale(64.0f), growthFactor(2.0f), backoffFactor(0.5f), minScale(1.0f), maxScale(16777216.0f), growthInterval(2000), successfulSteps(0) {}

void CudaLossScaler::updateOnOverflow() {
    this->successfulSteps = 0;
    this->scale *= this->backoffFactor;
    if (this->scale < this->minScale)
        this->scale = this->minScale;
}

void CudaLossScaler::updateOnSuccess() {
    ++this->successfulSteps;
    if (this->successfulSteps < this->growthInterval) return;
    this->successfulSteps = 0;
    this->scale *= this->growthFactor;
    if (this->scale > this->maxScale)
        this->scale = this->maxScale;
}

CudaHalfMatrix::CudaHalfMatrix() : rows(0), cols(0) {}

CudaHalfMatrix::CudaHalfMatrix(CudaHalfMatrix&& other) noexcept : rows(other.rows), cols(other.cols), buffer(std::move(other.buffer)) {
    other.rows = 0;
    other.cols = 0;
}

CudaHalfMatrix& CudaHalfMatrix::operator=(CudaHalfMatrix&& other) noexcept {
    if (this == &other) return *this;
    this->rows = other.rows;
    this->cols = other.cols;
    this->buffer = std::move(other.buffer);
    other.rows = 0;
    other.cols = 0;
    return *this;
}

bool CudaHalfMatrix::empty() const {
    return this->rows == 0 || this->cols == 0;
}

size_t CudaHalfMatrix::elementCount() const {
    return this->rows * this->cols;
}

size_t CudaHalfMatrix::byteCount() const {
    return this->elementCount() * sizeof(__half);
}

void CudaHalfMatrix::ensureSize(size_t rowCount, size_t columnCount) {
    this->rows = rowCount;
    this->cols = columnCount;
    this->buffer.ensureCapacity(this->byteCount());
}

void CudaHalfMatrix::free() {
    this->buffer.free();
    this->rows = 0;
    this->cols = 0;
}

__global__ void CudaAmpCastFloatToHalfEntry(const float* source, __half* destination, int elementCount) {
    const int index = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (index >= elementCount) return;
    destination[index] = __float2half(source[index]);
}

__global__ void CudaAmpCastFloatToHalfSaturatedEntry(const float* source, __half* destination, int elementCount) {
    const int index = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (index >= elementCount) return;
    float value = source[index];
    if (!isfinite(value)) value = 0.0f;
    // FP16 finite max ~65504
    value = fminf(fmaxf(value, -65504.0f), 65504.0f);
    destination[index] = __float2half(value);
}

__global__ void CudaAmpCastHalfToFloatEntry(const __half* source, float* destination, int elementCount) {
    const int index = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (index >= elementCount) return;
    destination[index] = __half2float(source[index]);
}

__global__ void CudaAmpHasNonFiniteEntry(const float* data, int elementCount, int* flag) {
    const int index = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (index >= elementCount) return;
    const float value = data[index];
    if (isnan(value) || isinf(value))
        atomicExch(flag, 1);
}

static void throwIfCudaFailedAmp(cudaError_t status, const char* operationName) {
    if (status == cudaSuccess) return;
    throw std::runtime_error(std::string(operationName) + ": " + cudaGetErrorString(status));
}

static void launchCastFloatToHalf(const float* source, __half* destination, int elementCount) {
    if (elementCount <= 0) return;
    const int threads = 256;
    const int blocks = (elementCount + threads - 1) / threads;
    CudaAmpCastFloatToHalfEntry<<<blocks, threads>>>(source, destination, elementCount);
    throwIfCudaFailedAmp(cudaGetLastError(), "CudaAmpCastFloatToHalfEntry launch");
}

static void launchCastFloatToHalfSaturated(const float* source, __half* destination, int elementCount) {
    if (elementCount <= 0) return;
    const int threads = 256;
    const int blocks = (elementCount + threads - 1) / threads;
    CudaAmpCastFloatToHalfSaturatedEntry<<<blocks, threads>>>(source, destination, elementCount);
    throwIfCudaFailedAmp(cudaGetLastError(), "CudaAmpCastFloatToHalfSaturatedEntry launch");
}

static void launchCastHalfToFloat(const __half* source, float* destination, int elementCount) {
    if (elementCount <= 0) return;
    const int threads = 256;
    const int blocks = (elementCount + threads - 1) / threads;
    CudaAmpCastHalfToFloatEntry<<<blocks, threads>>>(source, destination, elementCount);
    throwIfCudaFailedAmp(cudaGetLastError(), "CudaAmpCastHalfToFloatEntry launch");
}

void CudaAmp::castToHalf(const CudaMatrix& source, CudaHalfMatrix& destination) {
    if (source.empty()) throw std::invalid_argument("CudaAmp::castToHalf empty source");
    destination.ensureSize(source.rows, source.cols);
    launchCastFloatToHalf(source.buffer.deviceData, reinterpret_cast<__half*>(destination.buffer.deviceData), static_cast<int>(source.elementCount()));
}

void CudaAmp::castToHalfSaturated(const CudaMatrix& source, CudaHalfMatrix& destination) {
    if (source.empty()) throw std::invalid_argument("CudaAmp::castToHalfSaturated empty source");
    destination.ensureSize(source.rows, source.cols);
    launchCastFloatToHalfSaturated(source.buffer.deviceData, reinterpret_cast<__half*>(destination.buffer.deviceData), static_cast<int>(source.elementCount()));
}

void CudaAmp::castToFloat(const CudaHalfMatrix& source, CudaMatrix& destination) {
    if (source.empty()) throw std::invalid_argument("CudaAmp::castToFloat empty source");
    destination.ensureSize(source.rows, source.cols);
    launchCastHalfToFloat(reinterpret_cast<const __half*>(source.buffer.deviceData), destination.buffer.deviceData, static_cast<int>(source.elementCount()));
}

void CudaAmp::castFloatBufferToHalf(const float* source, size_t elementCount, CudaDeviceBuffer& destinationHalf) {
    if (source == nullptr) throw std::invalid_argument("CudaAmp::castFloatBufferToHalf null source");
    destinationHalf.ensureCapacity(elementCount * sizeof(__half));
    launchCastFloatToHalf(source, reinterpret_cast<__half*>(destinationHalf.deviceData), static_cast<int>(elementCount));
}

bool CudaAmp::hasNonFinite(const CudaMatrix& matrix) {
    if (matrix.empty()) return false;
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaAmp::hasNonFinite no CUDA device");

    CudaAmp::nonFiniteFlag.ensureCapacity(sizeof(int));
    int zero = 0;
    throwIfCudaFailedAmp(cudaMemcpy(CudaAmp::nonFiniteFlag.deviceData, &zero, sizeof(int), cudaMemcpyHostToDevice), "CudaAmp::hasNonFinite clear flag");

    const int elementCount = static_cast<int>(matrix.elementCount());
    const int threads = 256;
    const int blocks = (elementCount + threads - 1) / threads;
    CudaAmpHasNonFiniteEntry<<<blocks, threads>>>(matrix.buffer.deviceData, elementCount, reinterpret_cast<int*>(CudaAmp::nonFiniteFlag.deviceData));
    throwIfCudaFailedAmp(cudaGetLastError(), "CudaAmpHasNonFiniteEntry launch");

    int flag = 0;
    throwIfCudaFailedAmp(cudaMemcpy(&flag, CudaAmp::nonFiniteFlag.deviceData, sizeof(int), cudaMemcpyDeviceToHost), "CudaAmp::hasNonFinite read flag");
    return flag != 0;
}

static void launchHasNonFiniteIntoFlag(const CudaMatrix& matrix, int* deviceFlag) {
    if (matrix.empty()) return;
    const int elementCount = static_cast<int>(matrix.elementCount());
    const int threads = 256;
    const int blocks = (elementCount + threads - 1) / threads;
    CudaAmpHasNonFiniteEntry<<<blocks, threads>>>(matrix.buffer.deviceData, elementCount, deviceFlag);
    throwIfCudaFailedAmp(cudaGetLastError(), "CudaAmpHasNonFiniteEntry launch");
}

bool CudaAmp::gradientsHaveNonFinite(const CudaLanguageModelGradients& gradients) {
    CudaAmp::nonFiniteFlag.ensureCapacity(sizeof(int));
    int zero = 0;
    throwIfCudaFailedAmp(cudaMemcpy(CudaAmp::nonFiniteFlag.deviceData, &zero, sizeof(int), cudaMemcpyHostToDevice), "CudaAmp::gradientsHaveNonFinite clear flag");
    int* flag = reinterpret_cast<int*>(CudaAmp::nonFiniteFlag.deviceData);

    launchHasNonFiniteIntoFlag(gradients.tokenEmbedding, flag);
    launchHasNonFiniteIntoFlag(gradients.finalNormGamma, flag);
    if (!gradients.projectionWeight.empty())
        launchHasNonFiniteIntoFlag(gradients.projectionWeight, flag);
    launchHasNonFiniteIntoFlag(gradients.projectionBias, flag);

    for (const CudaTransformerBlockGradients& block : gradients.blocks) {
        launchHasNonFiniteIntoFlag(block.queryWeight, flag);
        launchHasNonFiniteIntoFlag(block.keyWeight, flag);
        launchHasNonFiniteIntoFlag(block.valueWeight, flag);
        launchHasNonFiniteIntoFlag(block.attentionOutputWeight, flag);
        launchHasNonFiniteIntoFlag(block.attentionNormGamma, flag);
        launchHasNonFiniteIntoFlag(block.feedForwardNormGamma, flag);
        launchHasNonFiniteIntoFlag(block.feedForwardGateWeight, flag);
        launchHasNonFiniteIntoFlag(block.feedForwardGateBias, flag);
        launchHasNonFiniteIntoFlag(block.feedForwardUpWeight, flag);
        launchHasNonFiniteIntoFlag(block.feedForwardUpBias, flag);
        launchHasNonFiniteIntoFlag(block.feedForwardDownWeight, flag);
        launchHasNonFiniteIntoFlag(block.feedForwardDownBias, flag);
    }

    int hostFlag = 0;
    throwIfCudaFailedAmp(cudaMemcpy(&hostFlag, CudaAmp::nonFiniteFlag.deviceData, sizeof(int), cudaMemcpyDeviceToHost), "CudaAmp::gradientsHaveNonFinite read flag");
    return hostFlag != 0;
}

bool CudaAmp::launchCublasLtMatmulFp16(const float* deviceLeft, const float* deviceRight, float* deviceOut, int rowCount, int columnCount, int sharedCount, bool transposeLeft, bool transposeRight, double* kernelMilliseconds) {
    if (!CudaAmp::preferMixedPrecision) return false;
    if (deviceLeft == nullptr || deviceRight == nullptr || deviceOut == nullptr) return false;
    if (rowCount <= 0 || columnCount <= 0 || sharedCount <= 0) return false;
    // tiny GEMMs (consumer toy dims / attention head pieces) are numerically worse in FP16
    if (sharedCount < 256 || rowCount < 32 || columnCount < 32) return false;

    struct LocalLt {
        cublasLtHandle_t handle;
        bool ok;
        CudaDeviceBuffer workspace;
        LocalLt() : handle(nullptr), ok(false) {
            if (cublasLtCreate(&handle) == CUBLAS_STATUS_SUCCESS) ok = true;
        }
        ~LocalLt() {
            if (handle != nullptr) cublasLtDestroy(handle);
        }
    };
    static LocalLt localLt;
    if (!localLt.ok) return false;

    const size_t workspaceBytes = 16ull * 1024ull * 1024ull;
    try {
        localLt.workspace.ensureCapacity(workspaceBytes);
    } catch (...) {
        return false;
    }

    const int leftRows = transposeLeft ? sharedCount : rowCount;
    const int leftCols = transposeLeft ? rowCount : sharedCount;
    const int rightRows = transposeRight ? columnCount : sharedCount;
    const int rightCols = transposeRight ? sharedCount : columnCount;
    const size_t leftElements = static_cast<size_t>(leftRows) * static_cast<size_t>(leftCols);
    const size_t rightElements = static_cast<size_t>(rightRows) * static_cast<size_t>(rightCols);

    try {
        CudaAmp::castFloatBufferToHalf(deviceLeft, leftElements, CudaAmp::halfScratchLeft);
        CudaAmp::castFloatBufferToHalf(deviceRight, rightElements, CudaAmp::halfScratchRight);
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

    if (cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F) != CUBLAS_STATUS_SUCCESS) return false;

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

    const cublasLtOrder_t rowMajorOrder = CUBLASLT_ORDER_ROW;
    if (cublasLtMatrixLayoutCreate(&layoutLeft, CUDA_R_16F, leftRows, leftCols, leftCols) != CUBLAS_STATUS_SUCCESS) {
        destroyDescriptors();
        return false;
    }
    if (cublasLtMatrixLayoutCreate(&layoutRight, CUDA_R_16F, rightRows, rightCols, rightCols) != CUBLAS_STATUS_SUCCESS) {
        destroyDescriptors();
        return false;
    }
    if (cublasLtMatrixLayoutCreate(&layoutOut, CUDA_R_32F, rowCount, columnCount, columnCount) != CUBLAS_STATUS_SUCCESS) {
        destroyDescriptors();
        return false;
    }

    if (cublasLtMatrixLayoutSetAttribute(layoutLeft, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowMajorOrder, sizeof(rowMajorOrder)) != CUBLAS_STATUS_SUCCESS
        || cublasLtMatrixLayoutSetAttribute(layoutRight, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowMajorOrder, sizeof(rowMajorOrder)) != CUBLAS_STATUS_SUCCESS
        || cublasLtMatrixLayoutSetAttribute(layoutOut, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowMajorOrder, sizeof(rowMajorOrder)) != CUBLAS_STATUS_SUCCESS) {
        destroyDescriptors();
        return false;
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    void* workspacePointer = localLt.workspace.deviceData;
    const size_t workspaceSize = localLt.workspace.capacityBytes;

    cudaEvent_t kernelStartEvent = nullptr;
    cudaEvent_t kernelStopEvent = nullptr;
    if (kernelMilliseconds != nullptr) {
        if (cudaEventCreate(&kernelStartEvent) != cudaSuccess || cudaEventCreate(&kernelStopEvent) != cudaSuccess) {
            if (kernelStartEvent != nullptr) cudaEventDestroy(kernelStartEvent);
            if (kernelStopEvent != nullptr) cudaEventDestroy(kernelStopEvent);
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

    const cublasStatus_t matmulStatus = cublasLtMatmul(
        localLt.handle,
        matmulDesc,
        &alpha,
        CudaAmp::halfScratchLeft.deviceData,
        layoutLeft,
        CudaAmp::halfScratchRight.deviceData,
        layoutRight,
        &beta,
        deviceOut,
        layoutOut,
        deviceOut,
        layoutOut,
        nullptr,
        workspacePointer,
        workspaceSize,
        nullptr);

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
