#include "CudaAmp.hpp"

#include "CudaLanguageModel.hpp"
#include "CudaOps.hpp"

#include <algorithm>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

#if defined(_OPENMP)
#include <omp.h>
#endif

bool CudaAmp::preferMixedPrecision = false;
bool CudaAmp::useLossScaling = false;
CudaLossScaler CudaAmp::lossScaler = CudaLossScaler();
CudaDeviceBuffer CudaAmp::halfScratchLeft = CudaDeviceBuffer();
CudaDeviceBuffer CudaAmp::halfScratchRight = CudaDeviceBuffer();
CudaDeviceBuffer CudaAmp::floatScratchLeft = CudaDeviceBuffer();
CudaDeviceBuffer CudaAmp::floatScratchRight = CudaDeviceBuffer();
CudaDeviceBuffer CudaAmp::nonFiniteFlag = CudaDeviceBuffer();
CudaAmp::MasterWeightHalf CudaAmp::masterWeights[CudaAmp::maxMasterWeights];
int CudaAmp::masterWeightCount = 0;

namespace {
struct PinnedHalfUploadStaging {
    void* buffers[2] = { nullptr, nullptr };
    size_t capacityBytes[2] = { 0, 0 };
    cudaEvent_t readyEvents[2] = { nullptr, nullptr };
    int turn = 0;

    ~PinnedHalfUploadStaging() {
        for (int slot = 0; slot < 2; ++slot) {
            if (this->readyEvents[slot] != nullptr) {
                cudaEventDestroy(this->readyEvents[slot]);
                this->readyEvents[slot] = nullptr;
            }
            if (this->buffers[slot] != nullptr) {
                cudaFreeHost(this->buffers[slot]);
                this->buffers[slot] = nullptr;
            }
            this->capacityBytes[slot] = 0;
        }
    }

    void ensureEvents() {
        for (int slot = 0; slot < 2; ++slot) {
            if (this->readyEvents[slot] != nullptr) continue;
            if (cudaEventCreateWithFlags(&this->readyEvents[slot], cudaEventDisableTiming) != cudaSuccess)
                throw std::runtime_error("PinnedHalfUploadStaging event create failed");
        }
    }

    void* acquire(size_t bytes) {
        this->ensureEvents();
        const int slot = this->turn;
        if (this->readyEvents[slot] != nullptr)
            cudaEventSynchronize(this->readyEvents[slot]);
        if (bytes > this->capacityBytes[slot]) {
            if (this->buffers[slot] != nullptr) {
                cudaFreeHost(this->buffers[slot]);
                this->buffers[slot] = nullptr;
            }
            this->capacityBytes[slot] = 0;
            if (cudaMallocHost(&this->buffers[slot], bytes) != cudaSuccess)
                throw std::runtime_error("PinnedHalfUploadStaging cudaMallocHost failed");
            this->capacityBytes[slot] = bytes;
        }
        return this->buffers[slot];
    }

    void record(cudaStream_t stream) {
        const int slot = this->turn;
        this->ensureEvents();
        if (cudaEventRecord(this->readyEvents[slot], stream) != cudaSuccess)
            throw std::runtime_error("PinnedHalfUploadStaging event record failed");
        this->turn = 1 - this->turn;
    }
};

thread_local PinnedHalfUploadStaging gPinnedHalfUploadStaging;
} // namespace

bool CudaAmp::lossScalingActive() {
    return CudaAmp::preferMixedPrecision && CudaAmp::useLossScaling;
}

void CudaAmp::registerMasterWeight(const float* deviceData, size_t elementCount) {
    if (deviceData == nullptr || elementCount == 0) return;
    for (int index = 0; index < CudaAmp::masterWeightCount; ++index) {
        MasterWeightHalf& entry = CudaAmp::masterWeights[index];
        if (entry.deviceData == deviceData) {
            entry.elementCount = elementCount;
            entry.valid = false;
            entry.fp16Working = false;
            return;
        }
    }
    if (CudaAmp::masterWeightCount >= CudaAmp::maxMasterWeights) return;
    MasterWeightHalf& entry = CudaAmp::masterWeights[CudaAmp::masterWeightCount++];
    entry.deviceData = deviceData;
    entry.elementCount = elementCount;
    entry.valid = false;
    entry.fp16Working = false;
}

void CudaAmp::bindFp16WorkingWeight(CudaMatrix& matrix) {
    if (matrix.empty()) throw std::invalid_argument("CudaAmp::bindFp16WorkingWeight empty matrix");
    if (!matrix.hasDeviceStorage()) throw std::invalid_argument("CudaAmp::bindFp16WorkingWeight needs FP32 device storage to cast");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaAmp::bindFp16WorkingWeight no CUDA device");

    int slot = matrix.ampWeightSlot;
    if (slot < 0) {
        if (CudaAmp::masterWeightCount >= CudaAmp::maxMasterWeights)
            throw std::runtime_error("CudaAmp::bindFp16WorkingWeight master weight table full");
        slot = CudaAmp::masterWeightCount++;
        matrix.ampWeightSlot = slot;
        MasterWeightHalf& entry = CudaAmp::masterWeights[slot];
        entry.deviceData = nullptr;
        entry.elementCount = matrix.elementCount();
        entry.valid = false;
        entry.fp16Working = true;
    }

    MasterWeightHalf& entry = CudaAmp::masterWeights[slot];
    entry.elementCount = matrix.elementCount();
    entry.fp16Working = true;
    CudaAmp::castFloatBufferToHalf(matrix.buffer.deviceData, entry.elementCount, entry.half);
    entry.valid = true;
    entry.deviceData = nullptr;
    matrix.releaseDeviceKeepShape();
}

void CudaAmp::uploadHostMasterToFp16Working(CudaMatrix& matrix, const float* hostMaster) {
    if (matrix.empty()) throw std::invalid_argument("CudaAmp::uploadHostMasterToFp16Working empty matrix");
    if (hostMaster == nullptr) throw std::invalid_argument("CudaAmp::uploadHostMasterToFp16Working null host");
    if (matrix.ampWeightSlot < 0 || matrix.ampWeightSlot >= CudaAmp::masterWeightCount)
        throw std::logic_error("CudaAmp::uploadHostMasterToFp16Working matrix not bound");
    MasterWeightHalf& entry = CudaAmp::masterWeights[matrix.ampWeightSlot];
    if (!entry.fp16Working) throw std::logic_error("CudaAmp::uploadHostMasterToFp16Working not fp16 working");

    const size_t elementCount = entry.elementCount;
    const size_t halfBytes = elementCount * sizeof(__half);
    entry.half.ensureCapacity(halfBytes);

    // Double-buffered pinned staging: cast chunk N+1 while H2D of chunk N runs.
    constexpr size_t kChunkElements = 16ull * 1024ull * 1024ull; // 16M halves ≈ 32 MiB
    cudaStream_t stream = CudaMatmul::activeStream();
    if (stream == nullptr) {
        // Prefer a non-default stream so memcpyAsync can overlap CPU cast without legacy sync.
        static thread_local cudaStream_t uploadStream = nullptr;
        if (uploadStream == nullptr)
            CudaMatmul::throwIfCudaFailed(
                cudaStreamCreateWithFlags(&uploadStream, cudaStreamNonBlocking),
                "CudaAmp upload stream create");
        stream = uploadStream;
    }
    __half* deviceHalf = reinterpret_cast<__half*>(entry.half.deviceData);

    for (size_t offset = 0; offset < elementCount; offset += kChunkElements) {
        const size_t chunk = (std::min)(kChunkElements, elementCount - offset);
        __half* pinnedHalf = reinterpret_cast<__half*>(gPinnedHalfUploadStaging.acquire(chunk * sizeof(__half)));
#if defined(_OPENMP)
        #pragma omp parallel for schedule(static)
#endif
        for (ptrdiff_t index = 0; index < static_cast<ptrdiff_t>(chunk); ++index)
            pinnedHalf[index] = __float2half_rn(hostMaster[offset + static_cast<size_t>(index)]);

        CudaMatmul::throwIfCudaFailed(
            cudaMemcpyAsync(deviceHalf + offset, pinnedHalf, chunk * sizeof(__half), cudaMemcpyHostToDevice, stream),
            "CudaAmp::uploadHostMasterToFp16Working H2D half");
        gPinnedHalfUploadStaging.record(stream);
    }
    entry.valid = true;
}

const void* CudaAmp::fp16WorkingWeightOrNull(const CudaMatrix& matrix) {
    if (matrix.ampWeightSlot < 0 || matrix.ampWeightSlot >= CudaAmp::masterWeightCount)
        return nullptr;
    const MasterWeightHalf& entry = CudaAmp::masterWeights[matrix.ampWeightSlot];
    if (!entry.fp16Working || !entry.valid || entry.half.deviceData == nullptr)
        return nullptr;
    return entry.half.deviceData;
}

void CudaAmp::invalidateMasterWeightHalves() {
    for (int index = 0; index < CudaAmp::masterWeightCount; ++index) {
        MasterWeightHalf& entry = CudaAmp::masterWeights[index];
        if (entry.fp16Working) continue; // working weights stay valid until explicit upload
        entry.valid = false;
    }
}

void CudaAmp::clearMasterWeights() {
    for (int index = 0; index < CudaAmp::masterWeightCount; ++index) {
        CudaAmp::masterWeights[index].half.free();
        CudaAmp::masterWeights[index].deviceData = nullptr;
        CudaAmp::masterWeights[index].elementCount = 0;
        CudaAmp::masterWeights[index].valid = false;
        CudaAmp::masterWeights[index].fp16Working = false;
    }
    CudaAmp::masterWeightCount = 0;
}

const void* CudaAmp::masterWeightHalfOrNull(const float* deviceData, size_t elementCount) {
    if (deviceData == nullptr || elementCount == 0) return nullptr;
    for (int index = 0; index < CudaAmp::masterWeightCount; ++index) {
        MasterWeightHalf& entry = CudaAmp::masterWeights[index];
        if (entry.fp16Working) continue;
        if (entry.deviceData != deviceData || entry.elementCount != elementCount) continue;
        if (!entry.valid) {
            try {
                CudaAmp::castFloatBufferToHalf(deviceData, elementCount, entry.half);
            } catch (...) {
                return nullptr;
            }
            entry.valid = true;
        }
        return entry.half.deviceData;
    }
    return nullptr;
}

void CudaAmp::resetLossScaler() {
    CudaAmp::lossScaler = CudaLossScaler();
}

CudaLossScaler::CudaLossScaler()
    : scale(1024.0f), growthFactor(2.0f), backoffFactor(0.5f), minScale(1.0f), maxScale(16777216.0f), growthInterval(2000), successfulSteps(0), overflowCount(0) {}

void CudaLossScaler::updateOnOverflow() {
    ++this->overflowCount;
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
    CudaAmpCastFloatToHalfEntry<<<blocks, threads, 0, CudaMatmul::activeStream()>>>(source, destination, elementCount);
    throwIfCudaFailedAmp(cudaGetLastError(), "CudaAmpCastFloatToHalfEntry launch");
}

static void launchCastFloatToHalfSaturated(const float* source, __half* destination, int elementCount) {
    if (elementCount <= 0) return;
    const int threads = 256;
    const int blocks = (elementCount + threads - 1) / threads;
    CudaAmpCastFloatToHalfSaturatedEntry<<<blocks, threads, 0, CudaMatmul::activeStream()>>>(source, destination, elementCount);
    throwIfCudaFailedAmp(cudaGetLastError(), "CudaAmpCastFloatToHalfSaturatedEntry launch");
}

static void launchCastHalfToFloat(const __half* source, float* destination, int elementCount) {
    if (elementCount <= 0) return;
    const int threads = 256;
    const int blocks = (elementCount + threads - 1) / threads;
    CudaAmpCastHalfToFloatEntry<<<blocks, threads, 0, CudaMatmul::activeStream()>>>(source, destination, elementCount);
    throwIfCudaFailedAmp(cudaGetLastError(), "CudaAmpCastHalfToFloatEntry launch");
}

const float* CudaAmp::resolveFp32Operand(const CudaMatrix& matrix, CudaDeviceBuffer& floatScratch) {
    if (matrix.hasDeviceStorage())
        return matrix.buffer.deviceData;
    const void* half = CudaAmp::fp16WorkingWeightOrNull(matrix);
    if (half == nullptr)
        return nullptr;
    floatScratch.ensureCapacity(matrix.elementCount() * sizeof(float));
    launchCastHalfToFloat(
        reinterpret_cast<const __half*>(half),
        reinterpret_cast<float*>(floatScratch.deviceData),
        static_cast<int>(matrix.elementCount()));
    return reinterpret_cast<const float*>(floatScratch.deviceData);
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
    CudaAmpHasNonFiniteEntry<<<blocks, threads, 0, CudaMatmul::activeStream()>>>(matrix.buffer.deviceData, elementCount, reinterpret_cast<int*>(CudaAmp::nonFiniteFlag.deviceData));
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
    CudaAmpHasNonFiniteEntry<<<blocks, threads, 0, CudaMatmul::activeStream()>>>(matrix.buffer.deviceData, elementCount, deviceFlag);
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

static bool launchCublasLtMatmulFp16Halves(
    const void* leftHalf,
    const void* rightHalf,
    float* deviceOut,
    int rowCount,
    int columnCount,
    int sharedCount,
    bool transposeLeft,
    bool transposeRight,
    double* kernelMilliseconds,
    const float* deviceBiasOrNull
) {
    if (leftHalf == nullptr || rightHalf == nullptr || deviceOut == nullptr) return false;
    if (rowCount <= 0 || columnCount <= 0 || sharedCount <= 0) return false;

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

    if (deviceBiasOrNull != nullptr) {
        const cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
        if (cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)) != CUBLAS_STATUS_SUCCESS) {
            destroyDescriptors();
            return false;
        }
        if (cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &deviceBiasOrNull, sizeof(deviceBiasOrNull)) != CUBLAS_STATUS_SUCCESS) {
            destroyDescriptors();
            return false;
        }
        const cudaDataType_t biasType = CUDA_R_32F;
        if (cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &biasType, sizeof(biasType)) != CUBLAS_STATUS_SUCCESS) {
            destroyDescriptors();
            return false;
        }
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

    const cublasLtMatmulAlgo_t* algoPointer = nullptr;
    cublasLtMatmulHeuristicResult_t heuristicResults[8]{};
    cublasLtMatmulPreference_t preference = nullptr;
    if (deviceBiasOrNull != nullptr) {
        if (cublasLtMatmulPreferenceCreate(&preference) != CUBLAS_STATUS_SUCCESS) {
            destroyDescriptors();
            return false;
        }
        const size_t workspaceBytesAttr = localLt.workspace.capacityBytes;
        if (cublasLtMatmulPreferenceSetAttribute(
                preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceBytesAttr, sizeof(workspaceBytesAttr)) != CUBLAS_STATUS_SUCCESS) {
            cublasLtMatmulPreferenceDestroy(preference);
            destroyDescriptors();
            return false;
        }
        int returnedResults = 0;
        if (cublasLtMatmulAlgoGetHeuristic(
                localLt.handle, matmulDesc, layoutLeft, layoutRight, layoutOut, layoutOut,
                preference, 8, heuristicResults, &returnedResults) != CUBLAS_STATUS_SUCCESS
            || returnedResults <= 0) {
            cublasLtMatmulPreferenceDestroy(preference);
            destroyDescriptors();
            return false;
        }
        int chosen = -1;
        for (int index = 0; index < returnedResults; ++index) {
            if (heuristicResults[index].state == CUBLAS_STATUS_SUCCESS
                && heuristicResults[index].workspaceSize <= workspaceBytesAttr) {
                chosen = index;
                break;
            }
        }
        cublasLtMatmulPreferenceDestroy(preference);
        preference = nullptr;
        if (chosen < 0) {
            destroyDescriptors();
            return false;
        }
        algoPointer = &heuristicResults[chosen].algo;
    }

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
        leftHalf,
        layoutLeft,
        rightHalf,
        layoutRight,
        &beta,
        deviceOut,
        layoutOut,
        deviceOut,
        layoutOut,
        algoPointer,
        localLt.workspace.deviceData,
        localLt.workspace.capacityBytes,
        CudaMatmul::activeStream());

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

const void* CudaAmp::castActivationToHalfScratch(const float* source, size_t elementCount) {
    if (source == nullptr || elementCount == 0) return nullptr;
    try {
        CudaAmp::castFloatBufferToHalf(source, elementCount, CudaAmp::halfScratchRight);
    } catch (...) {
        return nullptr;
    }
    return CudaAmp::halfScratchRight.deviceData;
}

bool CudaAmp::launchCublasLtMatmulFp16PreCastRight(const float* deviceLeft, const void* rightHalf, float* deviceOut, int rowCount, int columnCount, int sharedCount, bool transposeLeft, bool transposeRight, double* kernelMilliseconds, const float* deviceBiasOrNull) {
    if (!CudaAmp::preferMixedPrecision) return false;
    if (deviceLeft == nullptr || rightHalf == nullptr || deviceOut == nullptr) return false;
    if (rowCount <= 0 || columnCount <= 0 || sharedCount <= 0) return false;
    if (sharedCount < 256 || rowCount < 32 || columnCount < 32) return false;

    const int leftRows = transposeLeft ? sharedCount : rowCount;
    const int leftCols = transposeLeft ? rowCount : sharedCount;
    const size_t leftElements = static_cast<size_t>(leftRows) * static_cast<size_t>(leftCols);

    try {
        const void* cachedLeft = CudaAmp::masterWeightHalfOrNull(deviceLeft, leftElements);
        const void* leftHalf = cachedLeft;
        if (leftHalf == nullptr) {
            CudaAmp::castFloatBufferToHalf(deviceLeft, leftElements, CudaAmp::halfScratchLeft);
            leftHalf = CudaAmp::halfScratchLeft.deviceData;
        }
        return launchCublasLtMatmulFp16Halves(leftHalf, rightHalf, deviceOut, rowCount, columnCount, sharedCount, transposeLeft, transposeRight, kernelMilliseconds, deviceBiasOrNull);
    } catch (...) {
        return false;
    }
}

bool CudaAmp::launchCublasLtMatmulFp16(const float* deviceLeft, const float* deviceRight, float* deviceOut, int rowCount, int columnCount, int sharedCount, bool transposeLeft, bool transposeRight, double* kernelMilliseconds, const float* deviceBiasOrNull) {
    if (!CudaAmp::preferMixedPrecision) return false;
    if (deviceLeft == nullptr || deviceRight == nullptr || deviceOut == nullptr) return false;
    if (rowCount <= 0 || columnCount <= 0 || sharedCount <= 0) return false;
    if (sharedCount < 256 || rowCount < 32 || columnCount < 32) return false;

    const int leftRows = transposeLeft ? sharedCount : rowCount;
    const int leftCols = transposeLeft ? rowCount : sharedCount;
    const int rightRows = transposeRight ? columnCount : sharedCount;
    const int rightCols = transposeRight ? sharedCount : columnCount;
    const size_t leftElements = static_cast<size_t>(leftRows) * static_cast<size_t>(leftCols);
    const size_t rightElements = static_cast<size_t>(rightRows) * static_cast<size_t>(rightCols);

    try {
        const void* cachedLeft = CudaAmp::masterWeightHalfOrNull(deviceLeft, leftElements);
        const void* cachedRight = CudaAmp::masterWeightHalfOrNull(deviceRight, rightElements);
        if (cachedLeft == nullptr)
            CudaAmp::castFloatBufferToHalf(deviceLeft, leftElements, CudaAmp::halfScratchLeft);
        if (cachedRight == nullptr)
            CudaAmp::castFloatBufferToHalf(deviceRight, rightElements, CudaAmp::halfScratchRight);

        const void* leftHalf = cachedLeft != nullptr ? cachedLeft : CudaAmp::halfScratchLeft.deviceData;
        const void* rightHalf = cachedRight != nullptr ? cachedRight : CudaAmp::halfScratchRight.deviceData;
        return launchCublasLtMatmulFp16Halves(leftHalf, rightHalf, deviceOut, rowCount, columnCount, sharedCount, transposeLeft, transposeRight, kernelMilliseconds, deviceBiasOrNull);
    } catch (...) {
        return false;
    }
}

bool CudaAmp::launchCublasLtMatmulFp16Matrices(const CudaMatrix& left, const CudaMatrix& right, CudaMatrix& out, bool transposeLeft, bool transposeRight, double* kernelMilliseconds, const float* deviceBiasOrNull) {
    if (!CudaAmp::preferMixedPrecision) return false;
    if (left.empty() || right.empty() || out.empty()) return false;

    const int rowCount = static_cast<int>(transposeLeft ? left.cols : left.rows);
    const int sharedCount = static_cast<int>(transposeLeft ? left.rows : left.cols);
    const int columnCount = static_cast<int>(transposeRight ? right.rows : right.cols);

    const void* leftWorking = CudaAmp::fp16WorkingWeightOrNull(left);
    const void* rightWorking = CudaAmp::fp16WorkingWeightOrNull(right);
    // Keep the large-GEMM gate even for working weights; tiny shapes fall back to FP32 cast path.
    if (sharedCount < 256 || rowCount < 32 || columnCount < 32) return false;

    try {
        const void* leftHalf = leftWorking;
        const void* rightHalf = rightWorking;

        if (leftHalf == nullptr) {
            if (!left.hasDeviceStorage()) return false;
            const size_t leftElements = left.elementCount();
            leftHalf = CudaAmp::masterWeightHalfOrNull(left.buffer.deviceData, leftElements);
            if (leftHalf == nullptr) {
                CudaAmp::castFloatBufferToHalf(left.buffer.deviceData, leftElements, CudaAmp::halfScratchLeft);
                leftHalf = CudaAmp::halfScratchLeft.deviceData;
            }
        }
        if (rightHalf == nullptr) {
            if (!right.hasDeviceStorage()) return false;
            const size_t rightElements = right.elementCount();
            rightHalf = CudaAmp::masterWeightHalfOrNull(right.buffer.deviceData, rightElements);
            if (rightHalf == nullptr) {
                CudaAmp::castFloatBufferToHalf(right.buffer.deviceData, rightElements, CudaAmp::halfScratchRight);
                rightHalf = CudaAmp::halfScratchRight.deviceData;
            }
        }

        return launchCublasLtMatmulFp16Halves(
            leftHalf, rightHalf, out.buffer.deviceData,
            rowCount, columnCount, sharedCount, transposeLeft, transposeRight, kernelMilliseconds, deviceBiasOrNull);
    } catch (...) {
        return false;
    }
}
