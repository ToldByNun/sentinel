#include "CudaCausalSelfAttention.hpp"

#include "CudaOps.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <stdexcept>

CudaCausalSelfAttention::CudaCausalSelfAttention()
    : headCount(0), headDimension(0), pairCount(0), maximumPositionCount(0) {}

void CudaCausalSelfAttention::uploadFrom(const CausalSelfAttention& host) {
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaCausalSelfAttention::uploadFrom no CUDA device");

    this->queryWeight.upload(host.queryWeight);
    this->keyWeight.upload(host.keyWeight);
    this->valueWeight.upload(host.valueWeight);
    this->outputWeight.upload(host.outputWeight);
    this->headCount = host.headCount;
    this->headDimension = host.headDimension;
    this->pairCount = host.rotaryEmbedding.pairCount;
    this->maximumPositionCount = host.rotaryEmbedding.maximumPositionCount;

    Matrix hostCos(static_cast<size_t>(this->maximumPositionCount), static_cast<size_t>(this->pairCount), 0.0f);
    Matrix hostSin(static_cast<size_t>(this->maximumPositionCount), static_cast<size_t>(this->pairCount), 0.0f);
    for (int position = 0; position < this->maximumPositionCount; ++position) {
        for (int pairIndex = 0; pairIndex < this->pairCount; ++pairIndex) {
            const size_t tableIndex = static_cast<size_t>(position * this->pairCount + pairIndex);
            hostCos.at(static_cast<size_t>(position), static_cast<size_t>(pairIndex)) = host.rotaryEmbedding.cosTable[tableIndex];
            hostSin.at(static_cast<size_t>(position), static_cast<size_t>(pairIndex)) = host.rotaryEmbedding.sinTable[tableIndex];
        }
    }
    this->cosTable.upload(hostCos);
    this->sinTable.upload(hostSin);
}

CudaCausalSelfAttention CudaCausalSelfAttention::createFrom(const CausalSelfAttention& host) {
    CudaCausalSelfAttention device;
    device.uploadFrom(host);
    return device;
}

void CudaCausalSelfAttention::projectAndRotate(const CudaMatrix& input, int positionOffset) {
    CudaMatrix::multiplyInto(this->queryWeight, input, this->query);
    CudaMatrix::multiplyInto(this->keyWeight, input, this->key);
    CudaMatrix::multiplyInto(this->valueWeight, input, this->value);
    CudaOps::rotaryRotateInPlace(this->query, this->headCount, this->headDimension, this->pairCount, this->cosTable, this->sinTable, positionOffset);
    CudaOps::rotaryRotateInPlace(this->key, this->headCount, this->headDimension, this->pairCount, this->cosTable, this->sinTable, positionOffset);
}

void CudaCausalSelfAttention::attendFullSequence(CudaMatrix& out) {
    const size_t sequenceLength = this->query.cols;
    const float scale = 1.0f / std::sqrt(static_cast<float>(this->headDimension));

    this->attended.ensureSize(this->query.rows, sequenceLength);
    CudaOps::zeroInPlace(this->attended);

    for (int headIndex = 0; headIndex < this->headCount; ++headIndex) {
        CudaOps::extractHeadInto(this->query, headIndex, this->headDimension, this->queryHead);
        CudaOps::extractHeadInto(this->key, headIndex, this->headDimension, this->keyHead);
        CudaOps::extractHeadInto(this->value, headIndex, this->headDimension, this->valueHead);

        CudaMatrix::multiplyInto(this->keyHead, this->queryHead, this->scores, true, false);
        CudaOps::scaleInPlace(this->scores, scale);
        CudaOps::applyCausalMaskInPlace(this->scores);
        CudaOps::softmaxInto(this->scores, this->probabilities);
        CudaMatrix::multiplyInto(this->valueHead, this->probabilities, this->attendedHead);
        CudaOps::writeHead(this->attended, headIndex, this->headDimension, this->attendedHead);
    }

    CudaMatrix::multiplyInto(this->outputWeight, this->attended, out);
}

void CudaCausalSelfAttention::attendCachedQuery(const CudaKvCache& cache, CudaMatrix& out) {
    const int usedLength = cache.length;
    const float scale = 1.0f / std::sqrt(static_cast<float>(this->headDimension));

    this->attended.ensureSize(this->query.rows, this->query.cols);
    CudaOps::zeroInPlace(this->attended);

    for (int headIndex = 0; headIndex < this->headCount; ++headIndex) {
        CudaOps::extractHeadInto(this->query, headIndex, this->headDimension, this->queryHead);
        CudaOps::extractHeadInto(cache.key, headIndex, this->headDimension, usedLength, this->keyHead);
        CudaOps::extractHeadInto(cache.value, headIndex, this->headDimension, usedLength, this->valueHead);

        CudaMatrix::multiplyInto(this->keyHead, this->queryHead, this->scores, true, false);
        CudaOps::scaleInPlace(this->scores, scale);
        CudaOps::softmaxInto(this->scores, this->probabilities);
        CudaMatrix::multiplyInto(this->valueHead, this->probabilities, this->attendedHead);
        CudaOps::writeHead(this->attended, headIndex, this->headDimension, this->attendedHead);
    }

    CudaMatrix::multiplyInto(this->outputWeight, this->attended, out);
}

void CudaCausalSelfAttention::forward(const CudaMatrix& input, CudaMatrix& out) {
    if (input.empty()) throw std::invalid_argument("CudaCausalSelfAttention::forward empty input");
    if (this->queryWeight.empty()) throw std::logic_error("CudaCausalSelfAttention::forward weights not uploaded");
    if (this->queryWeight.cols != input.rows) throw std::invalid_argument("CudaCausalSelfAttention::forward embedding dim mismatch");
    if (static_cast<int>(input.cols) > this->maximumPositionCount) throw std::invalid_argument("CudaCausalSelfAttention::forward sequence longer than maximumPositionCount");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaCausalSelfAttention::forward no CUDA device");

    this->projectAndRotate(input, 0);
    this->attendFullSequence(out);
}

void CudaCausalSelfAttention::prefill(const CudaMatrix& input, CudaKvCache& cache, CudaMatrix& out) {
    if (input.empty()) throw std::invalid_argument("CudaCausalSelfAttention::prefill empty input");
    if (this->queryWeight.empty()) throw std::logic_error("CudaCausalSelfAttention::prefill weights not uploaded");
    if (this->queryWeight.cols != input.rows) throw std::invalid_argument("CudaCausalSelfAttention::prefill embedding dim mismatch");
    if (static_cast<int>(input.cols) > this->maximumPositionCount) throw std::invalid_argument("CudaCausalSelfAttention::prefill sequence longer than maximumPositionCount");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaCausalSelfAttention::prefill no CUDA device");

    cache.ensureCapacity(static_cast<int>(input.rows), this->maximumPositionCount);
    this->projectAndRotate(input, 0);
    cache.append(this->key, this->value);
    this->attendFullSequence(out);
}

void CudaCausalSelfAttention::decode(const CudaMatrix& input, CudaKvCache& cache, CudaMatrix& out) {
    if (input.empty()) throw std::invalid_argument("CudaCausalSelfAttention::decode empty input");
    if (input.cols != 1) throw std::invalid_argument("CudaCausalSelfAttention::decode expects a single token column");
    if (this->queryWeight.empty()) throw std::logic_error("CudaCausalSelfAttention::decode weights not uploaded");
    if (this->queryWeight.cols != input.rows) throw std::invalid_argument("CudaCausalSelfAttention::decode embedding dim mismatch");
    if (cache.key.empty() || cache.value.empty()) throw std::logic_error("CudaCausalSelfAttention::decode cache not allocated");
    if (cache.length >= cache.maximumPositionCount) throw std::invalid_argument("CudaCausalSelfAttention::decode cache full");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaCausalSelfAttention::decode no CUDA device");

    const int positionOffset = cache.length;
    this->projectAndRotate(input, positionOffset);
    cache.append(this->key, this->value);
    this->attendCachedQuery(cache, out);
}

void CudaCausalSelfAttention::runSmokeDemo(int embeddingDim, int headCount, int sequenceLength, int maximumPositionCount) {
    if (!CudaMatmul::isAvailable()) {
        std::printf("CUDA Attention smoke: no device\n");
        return;
    }
    if (embeddingDim <= 0 || headCount <= 0 || sequenceLength <= 0 || maximumPositionCount < sequenceLength)
        throw std::invalid_argument("CudaCausalSelfAttention::runSmokeDemo invalid dims");

    CausalSelfAttention host = CausalSelfAttention::create(embeddingDim, headCount, maximumPositionCount, 11u);
    Matrix hostInput(static_cast<size_t>(embeddingDim), static_cast<size_t>(sequenceLength), 0.0f);
    unsigned state = 55u;
    for (size_t index = 0; index < hostInput.data.size(); ++index) {
        state = state * 1664525u + 1013904223u;
        hostInput.data[index] = (static_cast<float>(state >> 8) / 16777216.0f) * 2.0f - 1.0f;
    }

    CausalSelfAttentionCache hostCache;
    const auto cpuStart = std::chrono::steady_clock::now();
    Matrix hostOutput = host.forward(hostInput, hostCache);
    const double cpuMilliseconds = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - cpuStart).count();

    CudaCausalSelfAttention device = CudaCausalSelfAttention::createFrom(host);
    CudaMatrix deviceInput;
    deviceInput.upload(hostInput);
    CudaMatrix deviceOutput;
    device.forward(deviceInput, deviceOutput);
    CudaMatmul::throwIfCudaFailed(cudaDeviceSynchronize(), "CudaCausalSelfAttention warm synchronize");

    cudaEvent_t kernelStartEvent = nullptr;
    cudaEvent_t kernelStopEvent = nullptr;
    CudaMatmul::throwIfCudaFailed(cudaEventCreate(&kernelStartEvent), "cudaEventCreate start");
    CudaMatmul::throwIfCudaFailed(cudaEventCreate(&kernelStopEvent), "cudaEventCreate stop");
    CudaMatmul::throwIfCudaFailed(cudaEventRecord(kernelStartEvent), "cudaEventRecord start");
    device.forward(deviceInput, deviceOutput);
    CudaMatmul::throwIfCudaFailed(cudaEventRecord(kernelStopEvent), "cudaEventRecord stop");
    CudaMatmul::throwIfCudaFailed(cudaEventSynchronize(kernelStopEvent), "cudaEventSynchronize stop");
    float deviceMilliseconds = 0.0f;
    CudaMatmul::throwIfCudaFailed(cudaEventElapsedTime(&deviceMilliseconds, kernelStartEvent, kernelStopEvent), "cudaEventElapsedTime");
    cudaEventDestroy(kernelStartEvent);
    cudaEventDestroy(kernelStopEvent);

    Matrix deviceHostOutput = deviceOutput.download();
    float maximumDifference = 0.0f;
    for (size_t index = 0; index < hostOutput.data.size(); ++index)
        maximumDifference = (std::max)(maximumDifference, std::fabs(hostOutput.data[index] - deviceHostOutput.data[index]));

    std::printf("CUDA Attention smoke: embed=%d heads=%d seq=%d  cpu=%.2fms  device=%.2fms  maxAbsDiff=%.6g\n", embeddingDim, headCount, sequenceLength, cpuMilliseconds, static_cast<double>(deviceMilliseconds), maximumDifference);
}

void CudaCausalSelfAttention::runKvCacheSmokeDemo(int embeddingDim, int headCount, int sequenceLength, int maximumPositionCount) {
    if (!CudaMatmul::isAvailable()) {
        std::printf("CUDA Attention KV smoke: no device\n");
        return;
    }
    if (embeddingDim <= 0 || headCount <= 0 || sequenceLength < 2 || maximumPositionCount < sequenceLength)
        throw std::invalid_argument("CudaCausalSelfAttention::runKvCacheSmokeDemo invalid dims");

    CausalSelfAttention host = CausalSelfAttention::create(embeddingDim, headCount, maximumPositionCount, 13u);
    Matrix hostInput(static_cast<size_t>(embeddingDim), static_cast<size_t>(sequenceLength), 0.0f);
    unsigned state = 59u;
    for (size_t index = 0; index < hostInput.data.size(); ++index) {
        state = state * 1664525u + 1013904223u;
        hostInput.data[index] = (static_cast<float>(state >> 8) / 16777216.0f) * 2.0f - 1.0f;
    }

    CudaCausalSelfAttention device = CudaCausalSelfAttention::createFrom(host);
    CudaMatrix deviceInput;
    deviceInput.upload(hostInput);

    CudaMatrix fullOutput;
    device.forward(deviceInput, fullOutput);

    Matrix hostPrefix(static_cast<size_t>(embeddingDim), static_cast<size_t>(sequenceLength - 1), 0.0f);
    Matrix hostLast(static_cast<size_t>(embeddingDim), 1, 0.0f);
    for (int row = 0; row < embeddingDim; ++row) {
        for (int column = 0; column < sequenceLength - 1; ++column)
            hostPrefix.at(static_cast<size_t>(row), static_cast<size_t>(column)) = hostInput.at(static_cast<size_t>(row), static_cast<size_t>(column));
        hostLast.at(static_cast<size_t>(row), 0) = hostInput.at(static_cast<size_t>(row), static_cast<size_t>(sequenceLength - 1));
    }

    CudaMatrix devicePrefix;
    CudaMatrix deviceLast;
    devicePrefix.upload(hostPrefix);
    deviceLast.upload(hostLast);

    CudaKvCache cache;
    CudaMatrix prefillOutput;
    CudaMatrix decodeOutput;
    device.prefill(devicePrefix, cache, prefillOutput);
    device.decode(deviceLast, cache, decodeOutput);

    Matrix fullHost = fullOutput.download();
    Matrix decodeHost = decodeOutput.download();
    float maximumDifference = 0.0f;
    for (int row = 0; row < embeddingDim; ++row) {
        const float fullValue = fullHost.at(static_cast<size_t>(row), static_cast<size_t>(sequenceLength - 1));
        const float decodeValue = decodeHost.at(static_cast<size_t>(row), 0);
        maximumDifference = (std::max)(maximumDifference, std::fabs(fullValue - decodeValue));
    }

    std::printf("CUDA Attention KV smoke: embed=%d heads=%d seq=%d  prefill+decode vs full last-col maxAbsDiff=%.6g  cacheLen=%d\n", embeddingDim, headCount, sequenceLength, maximumDifference, cache.length);
}
