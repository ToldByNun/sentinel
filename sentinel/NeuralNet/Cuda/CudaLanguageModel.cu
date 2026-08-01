#include "CudaLanguageModel.hpp"

#include "CudaOps.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <stdexcept>
#include <utility>

#include "../Optimizers/Adam.hpp"

CudaLanguageModel::CudaLanguageModel() : maximumPositionCount(0) {}

void CudaLanguageModel::uploadFrom(const LanguageModel& host) {
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaLanguageModel::uploadFrom no CUDA device");

    this->tokenEmbeddingWeight.upload(host.tokenEmbedding.weight);
    this->blocks.clear();
    this->blocks.reserve(host.blocks.size());
    for (const TransformerBlock& block : host.blocks)
        this->blocks.push_back(CudaTransformerBlock::createFrom(block));

    this->finalNorm.uploadFrom(host.finalNorm);
    this->projectionWeight.upload(host.outputProjection.weight);
    this->projectionBias.upload(host.outputProjection.bias);
    this->maximumPositionCount = host.maximumPositionCount;
    this->kvCaches.clear();
    this->kvCaches.resize(this->blocks.size());
}

CudaLanguageModel CudaLanguageModel::createFrom(const LanguageModel& host) {
    CudaLanguageModel device;
    device.uploadFrom(host);
    return device;
}

void CudaLanguageModel::forwardInto(const std::vector<int>& tokenIds, CudaMatrix& outLogits) {
    if (tokenIds.empty()) throw std::invalid_argument("CudaLanguageModel::forwardInto empty tokenIds");
    if (this->tokenEmbeddingWeight.empty()) throw std::logic_error("CudaLanguageModel::forwardInto weights not uploaded");
    if (static_cast<int>(tokenIds.size()) > this->maximumPositionCount)
        throw std::invalid_argument("CudaLanguageModel::forwardInto sequence longer than maximumPositionCount");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaLanguageModel::forwardInto no CUDA device");

    this->tokenIdsBuffer.ensureCapacity(tokenIds.size());
    this->tokenIdsBuffer.copyFromHost(tokenIds.data(), tokenIds.size());

    CudaOps::embeddingGatherInto(this->tokenEmbeddingWeight, this->tokenIdsBuffer, tokenIds.size(), this->hidden);

    for (size_t blockIndex = 0; blockIndex < this->blocks.size(); ++blockIndex) {
        this->blocks[blockIndex].forward(this->hidden, this->normalized);
        CudaMatrix swapBuffer = std::move(this->hidden);
        this->hidden = std::move(this->normalized);
        this->normalized = std::move(swapBuffer);
    }

    this->finalNorm.forward(this->hidden, this->normalized);
    CudaMatrix::multiplyInto(this->projectionWeight, this->normalized, outLogits);
    CudaOps::broadcastBiasAddInPlace(outLogits, this->projectionBias);
}

Matrix CudaLanguageModel::forward(const std::vector<int>& tokenIds) {
    this->forwardInto(tokenIds, this->logits);
    return this->logits.download();
}

void CudaLanguageModel::resetKvCaches() {
    if (this->blocks.empty()) throw std::logic_error("CudaLanguageModel::resetKvCaches no blocks");
    if (this->tokenEmbeddingWeight.empty()) throw std::logic_error("CudaLanguageModel::resetKvCaches weights not uploaded");

    const int embeddingDim = static_cast<int>(this->tokenEmbeddingWeight.cols);
    this->kvCaches.resize(this->blocks.size());
    for (size_t blockIndex = 0; blockIndex < this->kvCaches.size(); ++blockIndex)
        this->kvCaches[blockIndex].ensureCapacity(embeddingDim, this->maximumPositionCount);
}

void CudaLanguageModel::prefillInto(const std::vector<int>& tokenIds, CudaMatrix& outLogits) {
    if (tokenIds.empty()) throw std::invalid_argument("CudaLanguageModel::prefillInto empty tokenIds");
    if (this->tokenEmbeddingWeight.empty()) throw std::logic_error("CudaLanguageModel::prefillInto weights not uploaded");
    if (static_cast<int>(tokenIds.size()) > this->maximumPositionCount)
        throw std::invalid_argument("CudaLanguageModel::prefillInto sequence longer than maximumPositionCount");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaLanguageModel::prefillInto no CUDA device");

    this->resetKvCaches();
    this->tokenIdsBuffer.ensureCapacity(tokenIds.size());
    this->tokenIdsBuffer.copyFromHost(tokenIds.data(), tokenIds.size());
    CudaOps::embeddingGatherInto(this->tokenEmbeddingWeight, this->tokenIdsBuffer, tokenIds.size(), this->hidden);

    for (size_t blockIndex = 0; blockIndex < this->blocks.size(); ++blockIndex) {
        this->blocks[blockIndex].prefill(this->hidden, this->kvCaches[blockIndex], this->normalized);
        CudaMatrix swapBuffer = std::move(this->hidden);
        this->hidden = std::move(this->normalized);
        this->normalized = std::move(swapBuffer);
    }

    this->finalNorm.forward(this->hidden, this->normalized);
    CudaMatrix::multiplyInto(this->projectionWeight, this->normalized, outLogits);
    CudaOps::broadcastBiasAddInPlace(outLogits, this->projectionBias);
}

void CudaLanguageModel::decodeInto(int tokenId, CudaMatrix& outLogits) {
    if (this->tokenEmbeddingWeight.empty()) throw std::logic_error("CudaLanguageModel::decodeInto weights not uploaded");
    if (this->kvCaches.empty()) throw std::logic_error("CudaLanguageModel::decodeInto caches not allocated");
    if (this->kvCaches[0].length >= this->maximumPositionCount)
        throw std::invalid_argument("CudaLanguageModel::decodeInto cache full");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaLanguageModel::decodeInto no CUDA device");

    const int tokenIdsHost[1] = { tokenId };
    this->tokenIdsBuffer.ensureCapacity(1);
    this->tokenIdsBuffer.copyFromHost(tokenIdsHost, 1);
    CudaOps::embeddingGatherInto(this->tokenEmbeddingWeight, this->tokenIdsBuffer, 1, this->hidden);

    for (size_t blockIndex = 0; blockIndex < this->blocks.size(); ++blockIndex) {
        this->blocks[blockIndex].decode(this->hidden, this->kvCaches[blockIndex], this->normalized);
        CudaMatrix swapBuffer = std::move(this->hidden);
        this->hidden = std::move(this->normalized);
        this->normalized = std::move(swapBuffer);
    }

    this->finalNorm.forward(this->hidden, this->normalized);
    CudaMatrix::multiplyInto(this->projectionWeight, this->normalized, outLogits);
    CudaOps::broadcastBiasAddInPlace(outLogits, this->projectionBias);
}

void CudaLanguageModel::runSmokeDemo(int vocabularySize, int embeddingDim, int sequenceLength, int blockCount, int headCount) {
    if (!CudaMatmul::isAvailable()) {
        std::printf("CUDA LanguageModel smoke: no device\n");
        return;
    }
    if (vocabularySize <= 0 || embeddingDim <= 0 || sequenceLength <= 0 || blockCount <= 0 || headCount <= 0)
        throw std::invalid_argument("CudaLanguageModel::runSmokeDemo invalid dims");

    LanguageModel host(vocabularySize, embeddingDim, sequenceLength, Adam(0.001f), blockCount, headCount);

    std::vector<int> tokenIds(static_cast<size_t>(sequenceLength), 0);
    unsigned state = 91u;
    for (size_t index = 0; index < tokenIds.size(); ++index) {
        state = state * 1664525u + 1013904223u;
        tokenIds[index] = static_cast<int>(state % static_cast<unsigned>(vocabularySize));
    }

    const auto cpuStart = std::chrono::steady_clock::now();
    Matrix hostLogits = host.forward(tokenIds);
    const double cpuMilliseconds = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - cpuStart).count();

    const auto uploadStart = std::chrono::steady_clock::now();
    CudaLanguageModel device = CudaLanguageModel::createFrom(host);
    const double uploadMilliseconds = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - uploadStart).count();

    Matrix warmLogits = device.forward(tokenIds);
    CudaMatmul::throwIfCudaFailed(cudaDeviceSynchronize(), "CudaLanguageModel warm synchronize");

    cudaEvent_t kernelStartEvent = nullptr;
    cudaEvent_t kernelStopEvent = nullptr;
    CudaMatmul::throwIfCudaFailed(cudaEventCreate(&kernelStartEvent), "cudaEventCreate start");
    CudaMatmul::throwIfCudaFailed(cudaEventCreate(&kernelStopEvent), "cudaEventCreate stop");
    CudaMatmul::throwIfCudaFailed(cudaEventRecord(kernelStartEvent), "cudaEventRecord start");
    device.forwardInto(tokenIds, device.logits);
    CudaMatmul::throwIfCudaFailed(cudaEventRecord(kernelStopEvent), "cudaEventRecord stop");
    CudaMatmul::throwIfCudaFailed(cudaEventSynchronize(kernelStopEvent), "cudaEventSynchronize stop");
    float deviceMilliseconds = 0.0f;
    CudaMatmul::throwIfCudaFailed(cudaEventElapsedTime(&deviceMilliseconds, kernelStartEvent, kernelStopEvent), "cudaEventElapsedTime");
    cudaEventDestroy(kernelStartEvent);
    cudaEventDestroy(kernelStopEvent);

    const auto downloadStart = std::chrono::steady_clock::now();
    Matrix deviceLogits = device.logits.download();
    const double downloadMilliseconds = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - downloadStart).count();

    float maximumDifference = 0.0f;
    for (size_t index = 0; index < hostLogits.data.size(); ++index)
        maximumDifference = (std::max)(maximumDifference, std::fabs(hostLogits.data[index] - deviceLogits.data[index]));

    (void)warmLogits;
    std::printf("CUDA LanguageModel smoke: vocab=%d embed=%d seq=%d blocks=%d heads=%d\n", vocabularySize, embeddingDim, sequenceLength, blockCount, headCount);
    std::printf("  cpu=%.2fms  upload=%.2fms  device-forward=%.2fms  download=%.2fms  maxAbsDiff=%.6g\n", cpuMilliseconds, uploadMilliseconds, static_cast<double>(deviceMilliseconds), downloadMilliseconds, maximumDifference);
}

void CudaLanguageModel::runKvCacheSmokeDemo(int vocabularySize, int embeddingDim, int sequenceLength, int blockCount, int headCount) {
    if (!CudaMatmul::isAvailable()) {
        std::printf("CUDA LanguageModel KV smoke: no device\n");
        return;
    }
    if (vocabularySize <= 0 || embeddingDim <= 0 || sequenceLength < 2 || blockCount <= 0 || headCount <= 0)
        throw std::invalid_argument("CudaLanguageModel::runKvCacheSmokeDemo invalid dims");

    LanguageModel host(vocabularySize, embeddingDim, sequenceLength, Adam(0.001f), blockCount, headCount);

    std::vector<int> tokenIds(static_cast<size_t>(sequenceLength), 0);
    unsigned state = 97u;
    for (size_t index = 0; index < tokenIds.size(); ++index) {
        state = state * 1664525u + 1013904223u;
        tokenIds[index] = static_cast<int>(state % static_cast<unsigned>(vocabularySize));
    }

    CudaLanguageModel device = CudaLanguageModel::createFrom(host);

    CudaMatrix fullLogits;
    device.forwardInto(tokenIds, fullLogits);

    std::vector<int> prefixIds(tokenIds.begin(), tokenIds.end() - 1);
    const int lastTokenId = tokenIds.back();

    CudaMatrix prefillLogits;
    CudaMatrix decodeLogits;
    device.prefillInto(prefixIds, prefillLogits);
    device.decodeInto(lastTokenId, decodeLogits);

    Matrix fullHost = fullLogits.download();
    Matrix decodeHost = decodeLogits.download();
    const int vocabularyRows = static_cast<int>(fullHost.rows);
    float maximumDifference = 0.0f;
    for (int row = 0; row < vocabularyRows; ++row) {
        const float fullValue = fullHost.at(static_cast<size_t>(row), static_cast<size_t>(sequenceLength - 1));
        const float decodeValue = decodeHost.at(static_cast<size_t>(row), 0);
        maximumDifference = (std::max)(maximumDifference, std::fabs(fullValue - decodeValue));
    }

    std::printf("CUDA LanguageModel KV smoke: vocab=%d embed=%d seq=%d blocks=%d heads=%d\n", vocabularySize, embeddingDim, sequenceLength, blockCount, headCount);
    std::printf("  prefill+decode vs full last-col maxAbsDiff=%.6g  cacheLen=%d\n", maximumDifference, device.kvCaches.empty() ? 0 : device.kvCaches[0].length);
}
