#include "CudaBao.hpp"

#include "CudaAdam.hpp"

#include <cstddef>
#include <cstdio>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__linux__)
#include <unistd.h>
#endif

bool CudaBao::enabled = false;
BaoMode CudaBao::request = BaoMode::Auto;
BaoMode CudaBao::resolved = BaoMode::GpuInt8Adam;

const char* CudaBao::modeName(BaoMode mode) {
    switch (mode) {
    case BaoMode::Auto: return "Auto";
    case BaoMode::GpuInt8Adam: return "GpuInt8Adam";
    case BaoMode::HostFusedHalfAdam: return "HostFusedHalfAdam";
    case BaoMode::HostFusedHalfSgd: return "HostFusedHalfSgd";
    }
    return "Unknown";
}

size_t CudaBao::queryAvailableHostRamBytes() {
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

BaoMode CudaBao::select(size_t freeVramBytes, size_t parameterBytes, size_t hostRamHintBytes) {
    if (parameterBytes == 0)
        return BaoMode::GpuInt8Adam;

    // Resident GpuInt8 static: FP32 weights + FP32 grads + int8 moments (~0.5×).
    const size_t gpuInt8Static = parameterBytes + parameterBytes + (parameterBytes / 2ull);
    // Prefer pack-capable headroom over a flat 2 GiB wall: need room for a useful pack
    // (~512 MiB) plus ~1 GiB display/runtime slack.
    const size_t minPackBytes = 512ull * 1024ull * 1024ull;
    const size_t runtimeSlack = 1ull * 1024ull * 1024ull * 1024ull;
    if (freeVramBytes > gpuInt8Static + minPackBytes + runtimeSlack)
        return BaoMode::GpuInt8Adam;

    size_t hostAvail = hostRamHintBytes;
    if (hostAvail == 0)
        hostAvail = queryAvailableHostRamBytes();

    // HostFusedHalfAdam: masters + moments ≈ 2× params (+1 GiB spare). Else HostFusedHalfSgd.
    const size_t hostAdamNeed = 2ull * parameterBytes;
    const size_t hostSpare = 1ull << 30;
    if (hostAvail == 0 || hostAvail > hostAdamNeed + hostSpare)
        return BaoMode::HostFusedHalfAdam;
    return BaoMode::HostFusedHalfSgd;
}

void CudaBao::apply(BaoMode mode) {
    if (mode == BaoMode::Auto)
        mode = BaoMode::GpuInt8Adam;
    resolved = mode;
    enabled = true;
    switch (mode) {
    case BaoMode::GpuInt8Adam:
        CudaAdam::preferCpuOffload = false;
        CudaAdam::preferFp16GpuWeights = false;
        CudaAdam::preferHostGradients = false;
        CudaAdam::preferHostSgd = false;
        CudaAdam::preferInt8Moments = true;
        break;
    case BaoMode::HostFusedHalfAdam:
        CudaAdam::preferCpuOffload = true;
        CudaAdam::preferFp16GpuWeights = true;
        CudaAdam::preferHostGradients = true;
        CudaAdam::preferHostSgd = false;
        CudaAdam::preferInt8Moments = false;
        break;
    case BaoMode::HostFusedHalfSgd:
        CudaAdam::preferCpuOffload = true;
        CudaAdam::preferFp16GpuWeights = true;
        CudaAdam::preferHostGradients = true;
        CudaAdam::preferHostSgd = true;
        CudaAdam::preferInt8Moments = false;
        break;
    case BaoMode::Auto:
        break;
    }
}

BaoMode CudaBao::resolveAndApply(size_t freeVramBytes, size_t parameterBytes, size_t hostRamHintBytes) {
    BaoMode concrete = request;
    if (concrete == BaoMode::Auto)
        concrete = select(freeVramBytes, parameterBytes, hostRamHintBytes);
    apply(concrete);
    return concrete;
}

bool CudaBao::pipelineHostWeightUpdate() {
    return CudaAdam::preferHostGradients && CudaAdam::preferFp16GpuWeights
        && (CudaAdam::preferHostSgd || pipelineHostAdam());
}

bool CudaBao::pipelineHostAdam() {
    if (!CudaAdam::preferHostGradients || !CudaAdam::preferFp16GpuWeights || CudaAdam::preferHostSgd)
        return false;
    if (resolved == BaoMode::HostFusedHalfAdam || request == BaoMode::HostFusedHalfAdam)
        return true;
    return CudaAdam::preferCpuOffload;
}
