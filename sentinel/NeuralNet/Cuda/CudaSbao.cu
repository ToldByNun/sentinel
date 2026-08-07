#include "CudaSbao.hpp"

#include "CudaAdam.hpp"
#include "CudaMatmul.hpp"
#include "../Math/Matrix.hpp"
#include "../Optimizers/Adam.hpp"
#include "../Optimizers/Spulse.hpp"

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

#if defined(_WIN32)
#ifndef NOMINMAX
    #define NOMINMAX
#endif
#include <windows.h>
#elif defined(__linux__)
#include <unistd.h>
#endif

bool CudaSbao::enabled = false;
SbaoMode CudaSbao::request = SbaoMode::Auto;
SbaoMode CudaSbao::resolved = SbaoMode::GpuInt8Adam;

const char* CudaSbao::modeName(SbaoMode mode) {
    switch (mode) {
    case SbaoMode::Auto: return "Auto";
    case SbaoMode::GpuInt8Adam: return "GpuInt8Adam";
    case SbaoMode::HostFusedHalfAdam: return "HostFusedHalfAdam";
    case SbaoMode::HostFusedHalfSgd: return "HostFusedHalfSgd";
    }
    return "Unknown";
}

size_t CudaSbao::queryAvailableHostRamBytes() {
#if defined(_WIN32)
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status) == 0)
        return 0;
    return static_cast<size_t>(status.ullAvailPhys);
#elif defined(__linux__)
    const long pages = sysconf(_SC_AVPHYS_PAGES);
    const long pageSize = sysconf(_SC_PAGESIZE);
    if (pages > 0 && pageSize > 0)
        return static_cast<size_t>(pages) * static_cast<size_t>(pageSize);
    return 0;
#else
    return 0;
#endif
}

SbaoMode CudaSbao::select(size_t freeVramBytes, size_t parameterBytes, size_t hostRamHintBytes) {
    if (parameterBytes == 0)
        return SbaoMode::GpuInt8Adam;

    // Resident GpuInt8 static: FP32 weights + FP32 grads + int8 moments (~0.5×).
    const size_t gpuInt8Static = parameterBytes + parameterBytes + (parameterBytes / 2ull);
    // Prefer pack-capable headroom over a flat 2 GiB wall: need room for a useful pack
    // (~512 MiB) plus ~1 GiB display/runtime slack.
    const size_t minPackBytes = 512ull * 1024ull * 1024ull;
    const size_t runtimeSlack = 1ull * 1024ull * 1024ull * 1024ull;
    if (freeVramBytes > gpuInt8Static + minPackBytes + runtimeSlack)
        return SbaoMode::GpuInt8Adam;

    size_t hostAvail = hostRamHintBytes;
    if (hostAvail == 0)
        hostAvail = queryAvailableHostRamBytes();

    // HostFusedHalfAdam: masters + moments ≈ 2× params (+1 GiB spare). Else HostFusedHalfSgd.
    const size_t hostAdamNeed = 2ull * parameterBytes;
    const size_t hostSpare = 1ull << 30;
    if (hostAvail == 0 || hostAvail > hostAdamNeed + hostSpare)
        return SbaoMode::HostFusedHalfAdam;
    return SbaoMode::HostFusedHalfSgd;
}

void CudaSbao::apply(SbaoMode mode) {
    if (mode == SbaoMode::Auto)
        mode = SbaoMode::GpuInt8Adam;
    resolved = mode;
    enabled = true;
    switch (mode) {
    case SbaoMode::GpuInt8Adam:
        CudaAdam::preferCpuOffload = false;
        CudaAdam::preferFp16GpuWeights = false;
        CudaAdam::preferHostGradients = false;
        CudaAdam::preferHostSgd = false;
        CudaAdam::preferInt8Moments = true;
        break;
    case SbaoMode::HostFusedHalfAdam:
        CudaAdam::preferCpuOffload = true;
        CudaAdam::preferFp16GpuWeights = true;
        CudaAdam::preferHostGradients = true;
        CudaAdam::preferHostSgd = false;
        CudaAdam::preferInt8Moments = false;
        break;
    case SbaoMode::HostFusedHalfSgd:
        CudaAdam::preferCpuOffload = true;
        CudaAdam::preferFp16GpuWeights = true;
        CudaAdam::preferHostGradients = true;
        CudaAdam::preferHostSgd = true;
        CudaAdam::preferInt8Moments = false;
        break;
    case SbaoMode::Auto:
        break;
    }
}

SbaoMode CudaSbao::resolveAndApply(size_t freeVramBytes, size_t parameterBytes, size_t hostRamHintBytes) {
    SbaoMode concrete = request;
    if (concrete == SbaoMode::Auto)
        concrete = select(freeVramBytes, parameterBytes, hostRamHintBytes);
    apply(concrete);
    return concrete;
}

bool CudaSbao::pipelineHostWeightUpdate() {
    return CudaAdam::preferHostGradients && CudaAdam::preferFp16GpuWeights
        && (CudaAdam::preferHostSgd || pipelineHostAdam());
}

bool CudaSbao::pipelineHostAdam() {
    if (!CudaAdam::preferHostGradients || !CudaAdam::preferFp16GpuWeights || CudaAdam::preferHostSgd)
        return false;
    if (resolved == SbaoMode::HostFusedHalfAdam || request == SbaoMode::HostFusedHalfAdam)
        return true;
    return CudaAdam::preferCpuOffload;
}

namespace {
__global__ void CudaSbaoCastFloatToHalfEntry(const float* source, __half* destination, int elementCount) {
    const int index = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (index >= elementCount) return;
    destination[index] = __float2half(source[index]);
}

struct FusedHalfGradPiece {
    Matrix* host = nullptr;
    AdamState* adamState = nullptr;
    SpulseState* spulseState = nullptr;
    const float* deviceFloat = nullptr;
    size_t halfOffset = 0;
    size_t elementCount = 0;
    size_t rows = 0;
    size_t cols = 0;
};

// Pipeline depth: in-flight D2H + async expand/SGD + next commit. Shared (not TLS) so the
// SGD worker can pop batches committed on the main thread.
constexpr int kFusedHalfHostSlots = 8;
struct FusedHalfHostSlot {
    uint16_t* data = nullptr;
    size_t capacityBytes = 0;
    bool inUse = false;
};
struct FusedHalfDeviceSlot {
    CudaDeviceBuffer pack;
    bool inUse = false;
};
struct FusedHalfGradBatch {
    std::vector<FusedHalfGradPiece> pieces;
    uint16_t* hostHalf = nullptr;
    FusedHalfDeviceSlot* deviceSlot = nullptr;
    size_t elementCount = 0;
};

FusedHalfHostSlot g_fusedHalfHostSlots[kFusedHalfHostSlots];
FusedHalfDeviceSlot g_fusedHalfDeviceSlots[kFusedHalfHostSlots];
std::mutex g_fusedHalfReadyMutex;
std::mutex g_fusedHalfFreelistMutex;
std::vector<FusedHalfGradBatch> g_fusedHalfReadyBatches;
thread_local std::vector<FusedHalfGradPiece> g_fusedHalfOpenPieces;
thread_local size_t g_fusedHalfOpenElements = 0;

uint16_t* acquireFusedHalfHost(size_t bytes) {
    std::lock_guard<std::mutex> lock(g_fusedHalfFreelistMutex);
    for (int i = 0; i < kFusedHalfHostSlots; ++i) {
        FusedHalfHostSlot& slot = g_fusedHalfHostSlots[i];
        if (slot.inUse) continue;
        if (slot.capacityBytes < bytes) {
            if (slot.data != nullptr) {
                cudaFreeHost(slot.data);
                slot.data = nullptr;
                slot.capacityBytes = 0;
            }
            void* pinned = nullptr;
            const cudaError_t status = cudaMallocHost(&pinned, bytes);
            if (status != cudaSuccess)
                throw std::runtime_error(std::string("cudaMallocHost fused half grads: ") + cudaGetErrorString(status));
            slot.data = static_cast<uint16_t*>(pinned);
            slot.capacityBytes = bytes;
        }
        slot.inUse = true;
        return slot.data;
    }
    throw std::runtime_error("CudaSbao fused half host freelist exhausted");
}

FusedHalfDeviceSlot* acquireFusedHalfDevice(size_t halfBytes) {
    std::lock_guard<std::mutex> lock(g_fusedHalfFreelistMutex);
    for (int i = 0; i < kFusedHalfHostSlots; ++i) {
        FusedHalfDeviceSlot& slot = g_fusedHalfDeviceSlots[i];
        if (slot.inUse) continue;
        slot.pack.ensureCapacity(halfBytes);
        slot.inUse = true;
        return &slot;
    }
    throw std::runtime_error("CudaSbao fused half device freelist exhausted");
}

void releaseFusedHalfHost(uint16_t* data) {
    if (data == nullptr) return;
    std::lock_guard<std::mutex> lock(g_fusedHalfFreelistMutex);
    for (int i = 0; i < kFusedHalfHostSlots; ++i) {
        if (g_fusedHalfHostSlots[i].data == data) {
            g_fusedHalfHostSlots[i].inUse = false;
            return;
        }
    }
}

void releaseFusedHalfDevice(FusedHalfDeviceSlot* slot) {
    if (slot == nullptr) return;
    std::lock_guard<std::mutex> lock(g_fusedHalfFreelistMutex);
    slot->inUse = false;
}
} // namespace

void CudaSbao::beginFusedHalfGradOffload() {
    g_fusedHalfOpenPieces.clear();
    g_fusedHalfOpenElements = 0;
}

void CudaSbao::appendFusedHalfGradOffload(
    Matrix& hostOut,
    const float* deviceFloat,
    size_t rows,
    size_t cols,
    AdamState* hostAdamState,
    SpulseState* hostSpulseState
) {
    if (deviceFloat == nullptr) throw std::invalid_argument("appendFusedHalfGradOffload null deviceFloat");
    if (rows == 0 || cols == 0) throw std::invalid_argument("appendFusedHalfGradOffload empty shape");
    const size_t elementCount = rows * cols;
    hostOut.ensureSize(rows, cols);
    FusedHalfGradPiece piece;
    piece.host = &hostOut;
    piece.adamState = hostAdamState;
    piece.spulseState = hostSpulseState;
    piece.deviceFloat = deviceFloat;
    piece.halfOffset = g_fusedHalfOpenElements;
    piece.elementCount = elementCount;
    piece.rows = rows;
    piece.cols = cols;
    g_fusedHalfOpenPieces.push_back(piece);
    g_fusedHalfOpenElements += elementCount;
}

void CudaSbao::commitFusedHalfGradOffload(cudaStream_t stream) {
    if (g_fusedHalfOpenPieces.empty() || g_fusedHalfOpenElements == 0) return;
    if (stream == nullptr) throw std::invalid_argument("commitFusedHalfGradOffload null stream");

    const size_t halfBytes = g_fusedHalfOpenElements * sizeof(__half);
    FusedHalfDeviceSlot* deviceSlot = acquireFusedHalfDevice(halfBytes);
    uint16_t* hostHalf = acquireFusedHalfHost(halfBytes);
    __half* deviceHalf = reinterpret_cast<__half*>(deviceSlot->pack.deviceData);

    for (const FusedHalfGradPiece& piece : g_fusedHalfOpenPieces) {
        const int n = static_cast<int>(piece.elementCount);
        const int threads = 256;
        const int blocks = (n + threads - 1) / threads;
        CudaSbaoCastFloatToHalfEntry<<<blocks, threads, 0, stream>>>(
            piece.deviceFloat,
            deviceHalf + piece.halfOffset,
            n);
        CudaMatmul::throwIfCudaFailed(cudaGetLastError(), "CudaSbaoCastFloatToHalfEntry launch");
    }

    CudaMatmul::throwIfCudaFailed(
        cudaMemcpyAsync(hostHalf, deviceHalf, halfBytes, cudaMemcpyDeviceToHost, stream),
        "commitFusedHalfGradOffload D2H");

    FusedHalfGradBatch batch;
    batch.pieces = std::move(g_fusedHalfOpenPieces);
    batch.hostHalf = hostHalf;
    batch.deviceSlot = deviceSlot;
    batch.elementCount = g_fusedHalfOpenElements;
    {
        std::lock_guard<std::mutex> lock(g_fusedHalfReadyMutex);
        g_fusedHalfReadyBatches.push_back(std::move(batch));
    }
    g_fusedHalfOpenPieces.clear();
    g_fusedHalfOpenElements = 0;
}

void CudaSbao::commitPinnedD2hBatch() {
    // Fused half path commits inside enqueueDeferred via commitFusedHalfGradOffload.
}

void CudaSbao::flushPinnedD2hToHost() {
    FusedHalfGradBatch batch;
    {
        std::lock_guard<std::mutex> lock(g_fusedHalfReadyMutex);
        if (g_fusedHalfReadyBatches.empty()) return;
        batch = std::move(g_fusedHalfReadyBatches.front());
        g_fusedHalfReadyBatches.erase(g_fusedHalfReadyBatches.begin());
    }

    if (batch.deviceSlot != nullptr) {
        releaseFusedHalfDevice(batch.deviceSlot);
        batch.deviceSlot = nullptr;
    }

#if defined(_OPENMP)
    #pragma omp parallel for schedule(dynamic)
#endif
    for (int pieceIndex = 0; pieceIndex < static_cast<int>(batch.pieces.size()); ++pieceIndex) {
        const FusedHalfGradPiece& piece = batch.pieces[static_cast<size_t>(pieceIndex)];
        if (piece.host == nullptr || batch.hostHalf == nullptr) continue;
        piece.host->ensureSize(piece.rows, piece.cols);
        float* dst = piece.host->data.data();
        const __half* src = reinterpret_cast<const __half*>(batch.hostHalf + piece.halfOffset);
        for (size_t i = 0; i < piece.elementCount; ++i)
            dst[i] = __half2float(src[i]);
    }
    releaseFusedHalfHost(batch.hostHalf);
}

void CudaSbao::applyFusedHalfHostSgd(float stepScale) {
    FusedHalfGradBatch batch;
    {
        std::lock_guard<std::mutex> lock(g_fusedHalfReadyMutex);
        if (g_fusedHalfReadyBatches.empty()) return;
        batch = std::move(g_fusedHalfReadyBatches.front());
        g_fusedHalfReadyBatches.erase(g_fusedHalfReadyBatches.begin());
    }

    if (batch.deviceSlot != nullptr) {
        releaseFusedHalfDevice(batch.deviceSlot);
        batch.deviceSlot = nullptr;
    }

#if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
#endif
    for (int pieceIndex = 0; pieceIndex < static_cast<int>(batch.pieces.size()); ++pieceIndex) {
        const FusedHalfGradPiece& piece = batch.pieces[static_cast<size_t>(pieceIndex)];
        if (piece.host == nullptr || batch.hostHalf == nullptr) continue;
        if (piece.host->data.size() != piece.elementCount)
            throw std::runtime_error("CudaSbao::applyFusedHalfHostSgd master size mismatch");
        float* __restrict master = piece.host->data.data();
        const __half* __restrict grad = reinterpret_cast<const __half*>(batch.hostHalf + piece.halfOffset);
        const size_t n = piece.elementCount;
        for (size_t i = 0; i < n; ++i)
            master[i] -= stepScale * __half2float(grad[i]);
    }
    releaseFusedHalfHost(batch.hostHalf);
}

void CudaSbao::applyFusedHalfHostAdam(
    float learningRate,
    float beta1,
    float beta2,
    float epsilon,
    int timeStep,
    float gradScale
) {
    if (timeStep <= 0)
        throw std::invalid_argument("CudaSbao::applyFusedHalfHostAdam requires timeStep > 0");
    FusedHalfGradBatch batch;
    {
        std::lock_guard<std::mutex> lock(g_fusedHalfReadyMutex);
        if (g_fusedHalfReadyBatches.empty()) return;
        batch = std::move(g_fusedHalfReadyBatches.front());
        g_fusedHalfReadyBatches.erase(g_fusedHalfReadyBatches.begin());
    }

    if (batch.deviceSlot != nullptr) {
        releaseFusedHalfDevice(batch.deviceSlot);
        batch.deviceSlot = nullptr;
    }

    for (FusedHalfGradPiece& piece : batch.pieces) {
        if (piece.host == nullptr || piece.adamState == nullptr) continue;
        if (piece.adamState->firstMoment.empty()) {
            piece.adamState->firstMoment.ensureSize(piece.rows, piece.cols);
            piece.adamState->firstMoment.fill(0.0f);
            piece.adamState->secondMoment.ensureSize(piece.rows, piece.cols);
            piece.adamState->secondMoment.fill(0.0f);
        }
    }

    const float firstMomentCorrection = 1.0f - std::pow(beta1, static_cast<float>(timeStep));
    const float secondMomentCorrection = 1.0f - std::pow(beta2, static_cast<float>(timeStep));
    const float inverseFirstCorrection = 1.0f / firstMomentCorrection;
    const float inverseSecondCorrection = 1.0f / secondMomentCorrection;

#if defined(_OPENMP)
    #pragma omp parallel for schedule(dynamic)
#endif
    for (int pieceIndex = 0; pieceIndex < static_cast<int>(batch.pieces.size()); ++pieceIndex) {
        const FusedHalfGradPiece& piece = batch.pieces[static_cast<size_t>(pieceIndex)];
        if (piece.host == nullptr || piece.adamState == nullptr || batch.hostHalf == nullptr) continue;
        if (piece.host->data.size() != piece.elementCount)
            throw std::runtime_error("CudaSbao::applyFusedHalfHostAdam master size mismatch");
        float* master = piece.host->data.data();
        float* firstMoment = piece.adamState->firstMoment.data.data();
        float* secondMoment = piece.adamState->secondMoment.data.data();
        const __half* grad = reinterpret_cast<const __half*>(batch.hostHalf + piece.halfOffset);
        for (size_t i = 0; i < piece.elementCount; ++i) {
            const float g = gradScale * __half2float(grad[i]);
            firstMoment[i] = beta1 * firstMoment[i] + (1.0f - beta1) * g;
            secondMoment[i] = beta2 * secondMoment[i] + (1.0f - beta2) * g * g;
            const float correctedFirst = firstMoment[i] * inverseFirstCorrection;
            const float correctedSecond = secondMoment[i] * inverseSecondCorrection;
            master[i] -= learningRate * correctedFirst / (std::sqrt(correctedSecond) + epsilon);
        }
    }
    releaseFusedHalfHost(batch.hostHalf);
}

void CudaSbao::releaseCompletedFusedHalfDevices() {
    std::vector<FusedHalfDeviceSlot*> toRelease;
    {
        std::lock_guard<std::mutex> lock(g_fusedHalfReadyMutex);
        for (FusedHalfGradBatch& batch : g_fusedHalfReadyBatches) {
            if (batch.deviceSlot == nullptr) continue;
            toRelease.push_back(batch.deviceSlot);
            batch.deviceSlot = nullptr;
        }
    }
    for (FusedHalfDeviceSlot* slot : toRelease)
        releaseFusedHalfDevice(slot);
}

bool CudaSbao::takeReadyFusedHalfHostBatch(std::vector<SbaoFusedHalfHostPiece>& outPieces, std::uint16_t*& hostPin) {
    outPieces.clear();
    hostPin = nullptr;
    FusedHalfGradBatch batch;
    {
        std::lock_guard<std::mutex> lock(g_fusedHalfReadyMutex);
        if (g_fusedHalfReadyBatches.empty()) return false;
        batch = std::move(g_fusedHalfReadyBatches.front());
        g_fusedHalfReadyBatches.erase(g_fusedHalfReadyBatches.begin());
    }
    if (batch.deviceSlot != nullptr) {
        releaseFusedHalfDevice(batch.deviceSlot);
        batch.deviceSlot = nullptr;
    }
    outPieces.reserve(batch.pieces.size());
    for (const FusedHalfGradPiece& piece : batch.pieces) {
        SbaoFusedHalfHostPiece view;
        view.host = piece.host;
        view.adamState = piece.adamState;
        view.spulseState = piece.spulseState;
        view.gradHalf = batch.hostHalf != nullptr ? batch.hostHalf + piece.halfOffset : nullptr;
        view.rows = piece.rows;
        view.cols = piece.cols;
        view.elementCount = piece.elementCount;
        outPieces.push_back(view);
    }
    hostPin = batch.hostHalf;
    return true;
}

void CudaSbao::releaseFusedHalfHostPin(std::uint16_t* hostPin) {
    releaseFusedHalfHost(hostPin);
}
