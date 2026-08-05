#include "CudaAmp.hpp"

#include "CudaLanguageModel.hpp"
#include "CudaOps.hpp"

#include <algorithm>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
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
// Two sticky FP16 views of FP32 activations. Separate from GEMM left/right scratch
// so consecutive GEMMs can reuse the same cast (e.g. grad as left then right)
// without one operand's cast clobbering the other mid-setup.
struct ActivationHalfSlot {
    const float* source = nullptr;
    size_t elementCount = 0;
    CudaDeviceBuffer half;
    unsigned long long lastUsed = 0;
};

ActivationHalfSlot g_actHalfSlots[4];
unsigned long long g_actHalfUseCounter = 1;

void clearActivationHalfSlots() {
    for (ActivationHalfSlot& slot : g_actHalfSlots) {
        slot.source = nullptr;
        slot.elementCount = 0;
        slot.lastUsed = 0;
    }
}

const void* getOrCastActivationHalf(const float* source, size_t elementCount) {
    for (ActivationHalfSlot& slot : g_actHalfSlots) {
        if (slot.source == source
            && slot.elementCount == elementCount
            && slot.half.deviceData != nullptr) {
            slot.lastUsed = ++g_actHalfUseCounter;
            return slot.half.deviceData;
        }
    }
    int victim = 0;
    for (int i = 1; i < 4; ++i) {
        if (g_actHalfSlots[i].lastUsed < g_actHalfSlots[victim].lastUsed)
            victim = i;
    }
    ActivationHalfSlot& slot = g_actHalfSlots[victim];
    CudaAmp::castFloatBufferToHalf(source, elementCount, slot.half);
    slot.source = source;
    slot.elementCount = elementCount;
    slot.lastUsed = ++g_actHalfUseCounter;
    return slot.half.deviceData;
}
} // namespace

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
    cudaStream_t stream = CudaMatmul::activeStream();
    if (stream == nullptr) {
        static thread_local cudaStream_t uploadStream = nullptr;
        if (uploadStream == nullptr)
            CudaMatmul::throwIfCudaFailed(
                cudaStreamCreateWithFlags(&uploadStream, cudaStreamNonBlocking),
                "CudaAmp upload stream create");
        stream = uploadStream;
    }
    CudaAmp::uploadHostMasterToFp16Working(matrix, hostMaster, stream);
}

void CudaAmp::uploadHostMasterToFp16Working(CudaMatrix& matrix, const float* hostMaster, cudaStream_t stream) {
    if (matrix.empty()) throw std::invalid_argument("CudaAmp::uploadHostMasterToFp16Working empty matrix");
    if (hostMaster == nullptr) throw std::invalid_argument("CudaAmp::uploadHostMasterToFp16Working null host");
    if (stream == nullptr) throw std::invalid_argument("CudaAmp::uploadHostMasterToFp16Working null stream");
    if (matrix.ampWeightSlot < 0 || matrix.ampWeightSlot >= CudaAmp::masterWeightCount)
        throw std::logic_error("CudaAmp::uploadHostMasterToFp16Working matrix not bound");
    MasterWeightHalf& entry = CudaAmp::masterWeights[matrix.ampWeightSlot];
    if (!entry.fp16Working) throw std::logic_error("CudaAmp::uploadHostMasterToFp16Working not fp16 working");

    const size_t elementCount = entry.elementCount;
    const size_t halfBytes = elementCount * sizeof(__half);
    entry.half.ensureCapacity(halfBytes);

    // Double-buffered pinned staging: cast chunk N+1 while H2D of chunk N runs.
    constexpr size_t kChunkElements = 16ull * 1024ull * 1024ull; // 16M halves ≈ 32 MiB
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

void CudaAmp::uploadHostMastersConcatToFp16Working(
    CudaMatrix& matrix,
    const float* const* parts,
    const size_t* partElements,
    int partCount,
    cudaStream_t stream) {
    if (matrix.empty()) throw std::invalid_argument("CudaAmp::uploadHostMastersConcatToFp16Working empty matrix");
    if (parts == nullptr || partElements == nullptr || partCount <= 0)
        throw std::invalid_argument("CudaAmp::uploadHostMastersConcatToFp16Working bad parts");
    if (stream == nullptr) throw std::invalid_argument("CudaAmp::uploadHostMastersConcatToFp16Working null stream");
    if (matrix.ampWeightSlot < 0 || matrix.ampWeightSlot >= CudaAmp::masterWeightCount)
        throw std::logic_error("CudaAmp::uploadHostMastersConcatToFp16Working matrix not bound");
    MasterWeightHalf& entry = CudaAmp::masterWeights[matrix.ampWeightSlot];
    if (!entry.fp16Working) throw std::logic_error("CudaAmp::uploadHostMastersConcatToFp16Working not fp16 working");

    size_t totalElements = 0;
    for (int part = 0; part < partCount; ++part) {
        if (parts[part] == nullptr) throw std::invalid_argument("CudaAmp::uploadHostMastersConcatToFp16Working null part");
        totalElements += partElements[part];
    }
    if (totalElements != entry.elementCount)
        throw std::invalid_argument("CudaAmp::uploadHostMastersConcatToFp16Working size mismatch");

    entry.half.ensureCapacity(totalElements * sizeof(__half));
    constexpr size_t kChunkElements = 16ull * 1024ull * 1024ull;
    __half* deviceHalf = reinterpret_cast<__half*>(entry.half.deviceData);

    size_t globalOffset = 0;
    int partIndex = 0;
    size_t partOffset = 0;
    while (globalOffset < totalElements) {
        const size_t chunk = (std::min)(kChunkElements, totalElements - globalOffset);
        __half* pinnedHalf = reinterpret_cast<__half*>(gPinnedHalfUploadStaging.acquire(chunk * sizeof(__half)));

        size_t filled = 0;
        while (filled < chunk) {
            const size_t remainInPart = partElements[partIndex] - partOffset;
            const size_t take = (std::min)(remainInPart, chunk - filled);
            const float* src = parts[partIndex] + partOffset;
#if defined(_OPENMP)
            #pragma omp parallel for schedule(static)
#endif
            for (ptrdiff_t index = 0; index < static_cast<ptrdiff_t>(take); ++index)
                pinnedHalf[filled + static_cast<size_t>(index)] = __float2half_rn(src[index]);
            filled += take;
            partOffset += take;
            if (partOffset >= partElements[partIndex]) {
                ++partIndex;
                partOffset = 0;
            }
        }

        CudaMatmul::throwIfCudaFailed(
            cudaMemcpyAsync(deviceHalf + globalOffset, pinnedHalf, chunk * sizeof(__half), cudaMemcpyHostToDevice, stream),
            "CudaAmp::uploadHostMastersConcatToFp16Working H2D half");
        gPinnedHalfUploadStaging.record(stream);
        globalOffset += chunk;
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

void CudaAmp::invalidateActivationHalfCachesFor(const void* devicePtr) {
    if (devicePtr == nullptr) return;
    for (ActivationHalfSlot& slot : g_actHalfSlots) {
        if (slot.source == devicePtr) {
            slot.source = nullptr;
            slot.elementCount = 0;
            slot.lastUsed = 0;
        }
    }
}

void CudaAmp::invalidateActivationHalfCaches() {
    clearActivationHalfSlots();
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
    CudaAmp::invalidateActivationHalfCaches();
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

void CudaHalfMatrix::takeDeviceStorageFrom(CudaHalfMatrix& donor) {
    if (this == &donor) return;
    this->buffer.free();
    this->buffer = std::move(donor.buffer);
    this->rows = 0;
    this->cols = 0;
    donor.rows = 0;
    donor.cols = 0;
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

    struct Fp16DescCacheEntry {
        cublasLtMatmulDesc_t matmulDesc = nullptr;
        cublasLtMatrixLayout_t layoutLeft = nullptr;
        cublasLtMatrixLayout_t layoutRight = nullptr;
        cublasLtMatrixLayout_t layoutOut = nullptr;
        cublasLtMatmulAlgo_t algo{};
        bool hasAlgo = false;
        int rowCount = 0;
        int columnCount = 0;
        int sharedCount = 0;
        bool transposeLeft = false;
        bool transposeRight = false;
        bool hasBias = false;
        bool valid = false;
        unsigned long long lastUsed = 0;
    };

    constexpr int fp16DescCacheSize = 32;
    struct LocalLt {
        cublasLtHandle_t handle = nullptr;
        bool ok = false;
        CudaDeviceBuffer workspace;
        Fp16DescCacheEntry descCache[32]{};
        unsigned long long useCounter = 1;

        LocalLt() {
            if (cublasLtCreate(&handle) == CUBLAS_STATUS_SUCCESS) ok = true;
        }
        ~LocalLt() {
            for (int index = 0; index < 32; ++index)
                destroyDescEntry(descCache[index]);
            if (handle != nullptr) cublasLtDestroy(handle);
        }
        void destroyDescEntry(Fp16DescCacheEntry& entry) {
            if (entry.layoutOut != nullptr) cublasLtMatrixLayoutDestroy(entry.layoutOut);
            if (entry.layoutRight != nullptr) cublasLtMatrixLayoutDestroy(entry.layoutRight);
            if (entry.layoutLeft != nullptr) cublasLtMatrixLayoutDestroy(entry.layoutLeft);
            if (entry.matmulDesc != nullptr) cublasLtMatmulDescDestroy(entry.matmulDesc);
            entry = Fp16DescCacheEntry{};
        }
    };
    static LocalLt localLt;
    if (!localLt.ok) return false;

    constexpr size_t workspaceBytes = 64ull * 1024ull * 1024ull;
    try {
        localLt.workspace.ensureCapacity(workspaceBytes);
    } catch (...) {
        return false;
    }

    const bool wantBias = deviceBiasOrNull != nullptr;
    // Hot path: cache descriptors + heuristic algo by shape (matches FP32 GEMM cache).
    if (kernelMilliseconds == nullptr) {
        Fp16DescCacheEntry* cacheEntry = nullptr;
        for (int index = 0; index < fp16DescCacheSize; ++index) {
            Fp16DescCacheEntry& entry = localLt.descCache[index];
            if (!entry.valid) continue;
            if (entry.rowCount != rowCount || entry.columnCount != columnCount || entry.sharedCount != sharedCount)
                continue;
            if (entry.transposeLeft != transposeLeft || entry.transposeRight != transposeRight)
                continue;
            if (entry.hasBias != wantBias) continue;
            cacheEntry = &entry;
            break;
        }

        if (cacheEntry == nullptr) {
            int victim = 0;
            for (int index = 1; index < fp16DescCacheSize; ++index) {
                if (!localLt.descCache[index].valid) {
                    victim = index;
                    break;
                }
                if (localLt.descCache[index].lastUsed < localLt.descCache[victim].lastUsed)
                    victim = index;
            }
            cacheEntry = &localLt.descCache[victim];
            localLt.destroyDescEntry(*cacheEntry);

            if (cublasLtMatmulDescCreate(&cacheEntry->matmulDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F) != CUBLAS_STATUS_SUCCESS)
                return false;
            const cublasOperation_t transLeft = transposeLeft ? CUBLAS_OP_T : CUBLAS_OP_N;
            const cublasOperation_t transRight = transposeRight ? CUBLAS_OP_T : CUBLAS_OP_N;
            if (cublasLtMatmulDescSetAttribute(cacheEntry->matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transLeft, sizeof(transLeft)) != CUBLAS_STATUS_SUCCESS
                || cublasLtMatmulDescSetAttribute(cacheEntry->matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transRight, sizeof(transRight)) != CUBLAS_STATUS_SUCCESS) {
                localLt.destroyDescEntry(*cacheEntry);
                return false;
            }
            if (wantBias) {
                const cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
                const cudaDataType_t biasType = CUDA_R_32F;
                if (cublasLtMatmulDescSetAttribute(cacheEntry->matmulDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)) != CUBLAS_STATUS_SUCCESS
                    || cublasLtMatmulDescSetAttribute(cacheEntry->matmulDesc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &biasType, sizeof(biasType)) != CUBLAS_STATUS_SUCCESS) {
                    localLt.destroyDescEntry(*cacheEntry);
                    return false;
                }
            }

            const int leftRows = transposeLeft ? sharedCount : rowCount;
            const int leftCols = transposeLeft ? rowCount : sharedCount;
            const int rightRows = transposeRight ? columnCount : sharedCount;
            const int rightCols = transposeRight ? sharedCount : columnCount;
            const cublasLtOrder_t rowMajorOrder = CUBLASLT_ORDER_ROW;
            if (cublasLtMatrixLayoutCreate(&cacheEntry->layoutLeft, CUDA_R_16F, leftRows, leftCols, leftCols) != CUBLAS_STATUS_SUCCESS
                || cublasLtMatrixLayoutCreate(&cacheEntry->layoutRight, CUDA_R_16F, rightRows, rightCols, rightCols) != CUBLAS_STATUS_SUCCESS
                || cublasLtMatrixLayoutCreate(&cacheEntry->layoutOut, CUDA_R_32F, rowCount, columnCount, columnCount) != CUBLAS_STATUS_SUCCESS
                || cublasLtMatrixLayoutSetAttribute(cacheEntry->layoutLeft, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowMajorOrder, sizeof(rowMajorOrder)) != CUBLAS_STATUS_SUCCESS
                || cublasLtMatrixLayoutSetAttribute(cacheEntry->layoutRight, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowMajorOrder, sizeof(rowMajorOrder)) != CUBLAS_STATUS_SUCCESS
                || cublasLtMatrixLayoutSetAttribute(cacheEntry->layoutOut, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowMajorOrder, sizeof(rowMajorOrder)) != CUBLAS_STATUS_SUCCESS) {
                localLt.destroyDescEntry(*cacheEntry);
                return false;
            }

            cublasLtMatmulPreference_t preference = nullptr;
            if (cublasLtMatmulPreferenceCreate(&preference) == CUBLAS_STATUS_SUCCESS) {
                const size_t workspaceBytesAttr = localLt.workspace.capacityBytes;
                if (cublasLtMatmulPreferenceSetAttribute(
                        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceBytesAttr, sizeof(workspaceBytesAttr)) == CUBLAS_STATUS_SUCCESS) {
                    cublasLtMatmulHeuristicResult_t heuristicResults[8]{};
                    int returnedResults = 0;
                    if (cublasLtMatmulAlgoGetHeuristic(
                            localLt.handle, cacheEntry->matmulDesc, cacheEntry->layoutLeft, cacheEntry->layoutRight,
                            cacheEntry->layoutOut, cacheEntry->layoutOut, preference, 8, heuristicResults, &returnedResults) == CUBLAS_STATUS_SUCCESS) {
                        for (int hi = 0; hi < returnedResults; ++hi) {
                            if (heuristicResults[hi].state == CUBLAS_STATUS_SUCCESS
                                && heuristicResults[hi].workspaceSize <= workspaceBytesAttr) {
                                cacheEntry->algo = heuristicResults[hi].algo;
                                cacheEntry->hasAlgo = true;
                                break;
                            }
                        }
                    }
                }
                cublasLtMatmulPreferenceDestroy(preference);
            }

            cacheEntry->rowCount = rowCount;
            cacheEntry->columnCount = columnCount;
            cacheEntry->sharedCount = sharedCount;
            cacheEntry->transposeLeft = transposeLeft;
            cacheEntry->transposeRight = transposeRight;
            cacheEntry->hasBias = wantBias;
            cacheEntry->valid = true;
        }

        cacheEntry->lastUsed = localLt.useCounter++;
        if (wantBias) {
            if (cublasLtMatmulDescSetAttribute(
                    cacheEntry->matmulDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &deviceBiasOrNull, sizeof(deviceBiasOrNull)) != CUBLAS_STATUS_SUCCESS)
                return false;
        }

        const float alpha = 1.0f;
        const float beta = 0.0f;
        const cublasLtMatmulAlgo_t* algoPointer = cacheEntry->hasAlgo ? &cacheEntry->algo : nullptr;
        return cublasLtMatmul(
            localLt.handle,
            cacheEntry->matmulDesc,
            &alpha,
            leftHalf,
            cacheEntry->layoutLeft,
            rightHalf,
            cacheEntry->layoutRight,
            &beta,
            deviceOut,
            cacheEntry->layoutOut,
            deviceOut,
            cacheEntry->layoutOut,
            algoPointer,
            localLt.workspace.deviceData,
            localLt.workspace.capacityBytes,
            CudaMatmul::activeStream()) == CUBLAS_STATUS_SUCCESS;
    }

    // Timed path: keep per-call descriptors (rare).
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
    if (cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transLeft, sizeof(transLeft)) != CUBLAS_STATUS_SUCCESS
        || cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transRight, sizeof(transRight)) != CUBLAS_STATUS_SUCCESS) {
        destroyDescriptors();
        return false;
    }
    if (wantBias) {
        const cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
        const cudaDataType_t biasType = CUDA_R_32F;
        if (cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)) != CUBLAS_STATUS_SUCCESS
            || cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &deviceBiasOrNull, sizeof(deviceBiasOrNull)) != CUBLAS_STATUS_SUCCESS
            || cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &biasType, sizeof(biasType)) != CUBLAS_STATUS_SUCCESS) {
            destroyDescriptors();
            return false;
        }
    }
    const cublasLtOrder_t rowMajorOrder = CUBLASLT_ORDER_ROW;
    if (cublasLtMatrixLayoutCreate(&layoutLeft, CUDA_R_16F, leftRows, leftCols, leftCols) != CUBLAS_STATUS_SUCCESS
        || cublasLtMatrixLayoutCreate(&layoutRight, CUDA_R_16F, rightRows, rightCols, rightCols) != CUBLAS_STATUS_SUCCESS
        || cublasLtMatrixLayoutCreate(&layoutOut, CUDA_R_32F, rowCount, columnCount, columnCount) != CUBLAS_STATUS_SUCCESS
        || cublasLtMatrixLayoutSetAttribute(layoutLeft, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowMajorOrder, sizeof(rowMajorOrder)) != CUBLAS_STATUS_SUCCESS
        || cublasLtMatrixLayoutSetAttribute(layoutRight, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowMajorOrder, sizeof(rowMajorOrder)) != CUBLAS_STATUS_SUCCESS
        || cublasLtMatrixLayoutSetAttribute(layoutOut, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowMajorOrder, sizeof(rowMajorOrder)) != CUBLAS_STATUS_SUCCESS) {
        destroyDescriptors();
        return false;
    }
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cudaEvent_t kernelStartEvent = nullptr;
    cudaEvent_t kernelStopEvent = nullptr;
    if (cudaEventCreate(&kernelStartEvent) != cudaSuccess || cudaEventCreate(&kernelStopEvent) != cudaSuccess) {
        if (kernelStartEvent != nullptr) cudaEventDestroy(kernelStartEvent);
        if (kernelStopEvent != nullptr) cudaEventDestroy(kernelStopEvent);
        destroyDescriptors();
        return false;
    }
    cudaEventRecord(kernelStartEvent);
    if (cublasLtMatmul(
            localLt.handle, matmulDesc, &alpha, leftHalf, layoutLeft, rightHalf, layoutRight, &beta,
            deviceOut, layoutOut, deviceOut, layoutOut, nullptr,
            localLt.workspace.deviceData, localLt.workspace.capacityBytes, CudaMatmul::activeStream()) != CUBLAS_STATUS_SUCCESS) {
        cudaEventDestroy(kernelStopEvent);
        cudaEventDestroy(kernelStartEvent);
        destroyDescriptors();
        return false;
    }
    cudaEventRecord(kernelStopEvent);
    cudaEventSynchronize(kernelStopEvent);
    float elapsedMilliseconds = 0.0f;
    cudaEventElapsedTime(&elapsedMilliseconds, kernelStartEvent, kernelStopEvent);
    *kernelMilliseconds = static_cast<double>(elapsedMilliseconds);
    cudaEventDestroy(kernelStopEvent);
    cudaEventDestroy(kernelStartEvent);
    destroyDescriptors();
    return true;
}

const void* CudaAmp::castActivationToHalfScratch(const float* source, size_t elementCount) {
    if (source == nullptr || elementCount == 0) return nullptr;
    try {
        return getOrCastActivationHalf(source, elementCount);
    } catch (...) {
        return nullptr;
    }
}

void CudaAmp::persistActivationHalf(const float* source, size_t rows, size_t cols, CudaHalfMatrix& destination) {
    if (source == nullptr || rows == 0 || cols == 0)
        throw std::invalid_argument("CudaAmp::persistActivationHalf empty source");
    const size_t elementCount = rows * cols;
    destination.ensureSize(rows, cols);
    const void* cached = getOrCastActivationHalf(source, elementCount);
    CudaMatmul::memcpyDevice(destination.buffer.deviceData, cached, elementCount * sizeof(__half));
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
        const void* leftHalf = cachedLeft != nullptr
            ? cachedLeft
            : getOrCastActivationHalf(deviceLeft, leftElements);
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
        const void* leftHalf = cachedLeft != nullptr
            ? cachedLeft
            : getOrCastActivationHalf(deviceLeft, leftElements);
        const void* rightHalf = cachedRight != nullptr
            ? cachedRight
            : getOrCastActivationHalf(deviceRight, rightElements);
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
    // Sticky FP16 working weights have no FP32 fallback storage — must take the half path
    // even for short eval sequences (columnCount < 32) that the train pack never hits.
    const bool forceWorking = leftWorking != nullptr || rightWorking != nullptr;
    if (!forceWorking && (sharedCount < 256 || rowCount < 32 || columnCount < 32)) return false;
    if (forceWorking && (rowCount <= 0 || columnCount <= 0 || sharedCount <= 0)) return false;

    try {
        const void* leftHalf = leftWorking;
        const void* rightHalf = rightWorking;

        if (leftHalf == nullptr) {
            if (!left.hasDeviceStorage()) return false;
            const size_t leftElements = left.elementCount();
            leftHalf = CudaAmp::masterWeightHalfOrNull(left.buffer.deviceData, leftElements);
            if (leftHalf == nullptr)
                leftHalf = getOrCastActivationHalf(left.buffer.deviceData, leftElements);
        }
        if (rightHalf == nullptr) {
            if (!right.hasDeviceStorage()) return false;
            const size_t rightElements = right.elementCount();
            rightHalf = CudaAmp::masterWeightHalfOrNull(right.buffer.deviceData, rightElements);
            if (rightHalf == nullptr)
                rightHalf = getOrCastActivationHalf(right.buffer.deviceData, rightElements);
        }

        return launchCublasLtMatmulFp16Halves(
            leftHalf, rightHalf, out.buffer.deviceData,
            rowCount, columnCount, sharedCount, transposeLeft, transposeRight, kernelMilliseconds, deviceBiasOrNull);
    } catch (...) {
        return false;
    }
}

bool CudaAmp::launchCublasLtMatmulFp16HalfLeft(
    const void* leftHalf,
    const float* deviceRight,
    float* deviceOut,
    int rowCount,
    int columnCount,
    int sharedCount,
    bool transposeLeft,
    bool transposeRight,
    const float* deviceBiasOrNull
) {
    if (!CudaAmp::preferMixedPrecision) return false;
    if (leftHalf == nullptr || deviceRight == nullptr || deviceOut == nullptr) return false;
    if (rowCount <= 0 || columnCount <= 0 || sharedCount <= 0) return false;
    if (sharedCount < 256 || rowCount < 32 || columnCount < 32) return false;

    const int rightRows = transposeRight ? columnCount : sharedCount;
    const int rightCols = transposeRight ? sharedCount : columnCount;
    const size_t rightElements = static_cast<size_t>(rightRows) * static_cast<size_t>(rightCols);

    try {
        const void* rightHalf = getOrCastActivationHalf(deviceRight, rightElements);
        // Prefer the given half pointer directly when 16-byte aligned (fused up-slice of gateUp).
        // Fall back to a base-allocation copy if cuBLASLt rejects the mid-buffer pointer.
        if ((reinterpret_cast<uintptr_t>(leftHalf) & 15u) == 0u) {
            if (launchCublasLtMatmulFp16Halves(
                    leftHalf, rightHalf, deviceOut,
                    rowCount, columnCount, sharedCount, transposeLeft, transposeRight, nullptr, deviceBiasOrNull))
                return true;
        }
        const int leftRows = transposeLeft ? sharedCount : rowCount;
        const int leftCols = transposeLeft ? rowCount : sharedCount;
        const size_t leftElements = static_cast<size_t>(leftRows) * static_cast<size_t>(leftCols);
        CudaAmp::halfScratchLeft.ensureCapacity(leftElements * 2u);
        CudaMatmul::memcpyDevice(CudaAmp::halfScratchLeft.deviceData, leftHalf, leftElements * 2u);
        return launchCublasLtMatmulFp16Halves(
            CudaAmp::halfScratchLeft.deviceData,
            rightHalf,
            deviceOut,
            rowCount,
            columnCount,
            sharedCount,
            transposeLeft,
            transposeRight,
            nullptr,
            deviceBiasOrNull);
    } catch (...) {
        return false;
    }
}
