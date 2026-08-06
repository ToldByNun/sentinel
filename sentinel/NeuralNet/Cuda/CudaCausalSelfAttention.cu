#include "CudaCausalSelfAttention.hpp"

#include "CudaAmp.hpp"
#include "CudaFlashAttention.hpp"
#include "CudaOps.hpp"
#include "../Utils/SmokeLog.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <stdexcept>

CudaCausalSelfAttention::CudaCausalSelfAttention()
    : headCount(0), kvHeadCount(0), headDimension(0), pairCount(0), maximumPositionCount(0), windowSize(0), globalTokenCount(0), activeSegmentLength(0), activePackCount(0), preferFlashAttention(true), usedFlashAttention(false) {}

bool CudaCausalSelfAttention::usesFusedQkv() const {
    return this->kvHeadCount > 0 && this->kvHeadCount == this->headCount;
}

int CudaCausalSelfAttention::kvHeadIndexForQueryHead(int queryHeadIndex) const {
    const int groupSize = this->headCount / this->kvHeadCount;
    return queryHeadIndex / groupSize;
}

void CudaCausalSelfAttention::expandKvHeadsInto(const CudaMatrix& compact, CudaMatrix& expanded) {
    expanded.ensureSize(this->query.rows, compact.cols);
    for (int queryHead = 0; queryHead < this->headCount; ++queryHead) {
        const int kvHead = this->kvHeadIndexForQueryHead(queryHead);
        CudaOps::extractHeadInto(compact, kvHead, this->headDimension, this->keyHead);
        CudaOps::writeHead(expanded, queryHead, this->headDimension, this->keyHead);
    }
}

void CudaCausalSelfAttention::reduceKvHeadGradsInto(const CudaMatrix& expandedGrad, CudaMatrix& compactGrad) {
    compactGrad.ensureSize(this->key.rows, expandedGrad.cols);
    CudaOps::zeroInPlace(compactGrad);
    for (int queryHead = 0; queryHead < this->headCount; ++queryHead) {
        const int kvHead = this->kvHeadIndexForQueryHead(queryHead);
        CudaOps::extractHeadInto(expandedGrad, queryHead, this->headDimension, this->keyHead);
        CudaOps::addHeadInto(compactGrad, kvHead, this->headDimension, this->keyHead);
    }
}

bool CudaCausalSelfAttention::canUseFlashAttention(int segmentLength, int strideColumns) const {
    if (!this->preferFlashAttention) return false;
    if (this->headDimension <= 0 || this->headDimension > CudaFlashAttention::maxHeadDimension) return false;
    if (segmentLength <= 0 || strideColumns <= 0) return false;
    if (this->windowSize < segmentLength) return false;
    // Flash cp.async float4 paths need 16-byte-aligned row starts:
    // address ~ base + 4 * (row * strideColumns + columnStart). With columnStart pack-aligned
    // to segmentLength, both stride and segmentLength must be multiples of 4. Train packs
    // usually satisfy this; short eval sequences often do not (→ sticky misalignedAddress).
    if ((strideColumns & 3) != 0 || (segmentLength & 3) != 0) return false;
    return true;
}

void CudaCausalSelfAttention::releaseDenseAttentionScratch() {
    this->scores.releaseDeviceToPool();
    this->probabilities.releaseDeviceToPool();
    for (CudaMatrix& cached : this->cachedHeadProbabilities)
        cached.releaseDeviceToPool();
    this->cachedHeadProbabilities.clear();
    this->querySegment.releaseDeviceToPool();
    this->keySegment.releaseDeviceToPool();
    this->valueSegment.releaseDeviceToPool();
    this->attendedSegment.releaseDeviceToPool();
    this->attendedGradientSegment.releaseDeviceToPool();
    this->queryGradientSegment.releaseDeviceToPool();
    this->keyGradientSegment.releaseDeviceToPool();
    this->valueGradientSegment.releaseDeviceToPool();
    this->probabilityGradient.releaseDeviceToPool();
    this->scoreGradient.releaseDeviceToPool();
    this->valueHeadGradient.releaseDeviceToPool();
    this->queryHeadGradient.releaseDeviceToPool();
    this->keyHeadGradient.releaseDeviceToPool();
}

void CudaCausalSelfAttention::releaseFlashAttentionScratch() {
    this->flashLogSumExp.releaseDeviceToPool();
    this->flashDelta.releaseDeviceToPool();
    this->flashKeyExpanded.releaseDeviceToPool();
    this->flashValueExpanded.releaseDeviceToPool();
    this->flashKeyGradientExpanded.releaseDeviceToPool();
    this->flashValueGradientExpanded.releaseDeviceToPool();
}

void CudaCausalSelfAttention::releaseActivationScratch() {
    this->releaseFlashAttentionScratch();
    this->releaseDenseAttentionScratch();
    this->qkvProjected.releaseDeviceToPool();
    this->query.releaseDeviceToPool();
    this->key.releaseDeviceToPool();
    this->value.releaseDeviceToPool();
    this->queryHead.releaseDeviceToPool();
    this->keyHead.releaseDeviceToPool();
    this->valueHead.releaseDeviceToPool();
    this->attendedHead.releaseDeviceToPool();
    this->attended.releaseDeviceToPool();
    this->output.releaseDeviceToPool();
    this->inputCache.releaseDeviceToPool();
    this->attendedGradient.releaseDeviceToPool();
    this->queryGradient.releaseDeviceToPool();
    this->keyGradient.releaseDeviceToPool();
    this->valueGradient.releaseDeviceToPool();
    this->temp.releaseDeviceToPool();
}

void CudaCausalSelfAttention::uploadFrom(const CausalSelfAttention& host) {
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaCausalSelfAttention::uploadFrom no CUDA device");

    this->queryWeight.upload(host.queryWeight);
    this->keyWeight.upload(host.keyWeight);
    this->valueWeight.upload(host.valueWeight);
    this->outputWeight.upload(host.outputWeight);
    this->headCount = host.headCount;
    this->kvHeadCount = host.kvHeadCount > 0 ? host.kvHeadCount : host.headCount;
    this->headDimension = host.headDimension;
    this->pairCount = host.rotaryEmbedding.pairCount;
    this->maximumPositionCount = host.rotaryEmbedding.maximumPositionCount;
    this->windowSize = host.windowSize;
    this->globalTokenCount = host.globalTokenCount;

    if (this->headCount <= 0 || this->kvHeadCount <= 0 || this->headCount % this->kvHeadCount != 0)
        throw std::invalid_argument("CudaCausalSelfAttention::uploadFrom invalid head / kvHead geometry");
    if (static_cast<int>(host.keyWeight.rows) != this->kvHeadCount * this->headDimension
        || static_cast<int>(host.valueWeight.rows) != this->kvHeadCount * this->headDimension)
        throw std::invalid_argument("CudaCausalSelfAttention::uploadFrom key/value rows mismatch kvHeadCount");

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
    if (this->usesFusedQkv())
        this->syncFusedQkvWeight();
    else
        this->qkvWeight.free();
    CudaAmp::registerMasterWeight(this->queryWeight.buffer.deviceData, this->queryWeight.elementCount());
    CudaAmp::registerMasterWeight(this->keyWeight.buffer.deviceData, this->keyWeight.elementCount());
    CudaAmp::registerMasterWeight(this->valueWeight.buffer.deviceData, this->valueWeight.elementCount());
    CudaAmp::registerMasterWeight(this->outputWeight.buffer.deviceData, this->outputWeight.elementCount());
    if (!this->qkvWeight.empty())
        CudaAmp::registerMasterWeight(this->qkvWeight.buffer.deviceData, this->qkvWeight.elementCount());
}

void CudaCausalSelfAttention::syncFusedQkvWeight() {
    if (!this->usesFusedQkv()) {
        this->qkvWeight.free();
        return;
    }
    if (this->queryWeight.empty() || this->keyWeight.empty() || this->valueWeight.empty()) return;
    if (this->queryWeight.cols != this->keyWeight.cols || this->queryWeight.cols != this->valueWeight.cols)
        throw std::invalid_argument("CudaCausalSelfAttention::syncFusedQkvWeight embed mismatch");
    if (this->queryWeight.rows != this->keyWeight.rows || this->queryWeight.rows != this->valueWeight.rows)
        throw std::invalid_argument("CudaCausalSelfAttention::syncFusedQkvWeight projection rows mismatch");
    if (this->qkvWeight.ampWeightSlot >= 0)
        return; // FP16 working fused mirror owned by LM host-master rebuild

    this->qkvWeight.ensureSize(this->queryWeight.rows * 3ull, this->queryWeight.cols);
    const size_t sliceBytes = this->queryWeight.byteCount();
    CudaMatmul::memcpyDevice(this->qkvWeight.buffer.deviceData, this->queryWeight.buffer.deviceData, sliceBytes);
    CudaMatmul::memcpyDevice(this->qkvWeight.buffer.deviceData + this->queryWeight.elementCount(), this->keyWeight.buffer.deviceData, sliceBytes);
    CudaMatmul::memcpyDevice(this->qkvWeight.buffer.deviceData + 2ull * this->queryWeight.elementCount(), this->valueWeight.buffer.deviceData, sliceBytes);
    CudaAmp::registerMasterWeight(this->qkvWeight.buffer.deviceData, this->qkvWeight.elementCount());
}

CudaCausalSelfAttention CudaCausalSelfAttention::createFrom(const CausalSelfAttention& host) {
    CudaCausalSelfAttention device;
    device.uploadFrom(host);
    return device;
}

void CudaCausalSelfAttention::projectAndRotate(const CudaMatrix& input, int positionOffset, int segmentLength) {
    CudaOps::copyInto(input, this->inputCache);

    this->query.ensureSize(this->queryWeight.rows, input.cols);
    this->key.ensureSize(this->keyWeight.rows, input.cols);
    this->value.ensureSize(this->valueWeight.rows, input.cols);

    if (this->usesFusedQkv()) {
        if (this->qkvWeight.empty())
            this->syncFusedQkvWeight();

        this->qkvProjected.ensureSize(this->qkvWeight.rows, input.cols);
        CudaMatrix::multiplyInto(this->qkvWeight, input, this->qkvProjected);

        const size_t sliceElements = this->queryWeight.rows * input.cols;
        const size_t sliceBytes = sliceElements * sizeof(float);
        CudaMatmul::memcpyDevice(this->query.buffer.deviceData, this->qkvProjected.buffer.deviceData, sliceBytes);
        CudaMatmul::memcpyDevice(this->key.buffer.deviceData, this->qkvProjected.buffer.deviceData + sliceElements, sliceBytes);
        CudaMatmul::memcpyDevice(this->value.buffer.deviceData, this->qkvProjected.buffer.deviceData + 2ull * sliceElements, sliceBytes);
    } else {
        CudaMatrix::multiplyInto(this->queryWeight, input, this->query);
        CudaMatrix::multiplyInto(this->keyWeight, input, this->key);
        CudaMatrix::multiplyInto(this->valueWeight, input, this->value);
    }

    CudaOps::rotaryRotateInPlace(this->query, this->headCount, this->headDimension, this->pairCount, this->cosTable, this->sinTable, positionOffset, segmentLength);
    CudaOps::rotaryRotateInPlace(this->key, this->kvHeadCount, this->headDimension, this->pairCount, this->cosTable, this->sinTable, positionOffset, segmentLength);
}

void CudaCausalSelfAttention::attendFullSequence(CudaMatrix& out, int segmentLength) {
    const size_t sequenceLength = this->query.cols;
    const float scale = 1.0f / std::sqrt(static_cast<float>(this->headDimension));
    if (segmentLength < 0) segmentLength = this->activeSegmentLength;
    if (segmentLength <= 0) segmentLength = static_cast<int>(sequenceLength);
    const int packCount = static_cast<int>(sequenceLength) / segmentLength;
    this->activePackCount = packCount;
    this->usedFlashAttention = false;

    this->attended.ensureSize(this->query.rows, sequenceLength);
    CudaOps::zeroInPlace(this->attended);

    if (this->canUseFlashAttention(segmentLength, static_cast<int>(sequenceLength))) {
        this->usedFlashAttention = true;
        this->releaseDenseAttentionScratch();
        this->flashLogSumExp.ensureSize(static_cast<size_t>(this->headCount), sequenceLength);
        this->attended.ensureSize(this->query.rows, sequenceLength);

        const CudaMatrix* flashKey = &this->key;
        const CudaMatrix* flashValue = &this->value;
        if (!this->usesFusedQkv()) {
            this->expandKvHeadsInto(this->key, this->flashKeyExpanded);
            this->expandKvHeadsInto(this->value, this->flashValueExpanded);
            flashKey = &this->flashKeyExpanded;
            flashValue = &this->flashValueExpanded;
        }

        CudaFlashAttention::forwardMultiHead(
            this->query,
            *flashKey,
            *flashValue,
            this->attended,
            this->flashLogSumExp,
            this->headCount,
            this->headDimension,
            scale,
            true,
            0,
            segmentLength,
            packCount);

        CudaMatrix::multiplyInto(this->outputWeight, this->attended, out);
        return;
    }

    this->releaseFlashAttentionScratch();

    // prefer one dense scores GEMM when width is modest; segment loop for wide packs
    const bool useDensePack = packCount <= 1 || static_cast<int>(sequenceLength) <= 384;

    if (useDensePack) {
        if (this->cachedHeadProbabilities.size() != static_cast<size_t>(this->headCount))
            this->cachedHeadProbabilities.resize(static_cast<size_t>(this->headCount));

        for (int headIndex = 0; headIndex < this->headCount; ++headIndex) {
            const int kvHeadIndex = this->kvHeadIndexForQueryHead(headIndex);
            CudaOps::extractHeadInto(this->query, headIndex, this->headDimension, this->queryHead);
            CudaOps::extractHeadInto(this->key, kvHeadIndex, this->headDimension, this->keyHead);
            CudaOps::extractHeadInto(this->value, kvHeadIndex, this->headDimension, this->valueHead);

            CudaMatrix::multiplyInto(this->keyHead, this->queryHead, this->scores, true, false);
            CudaOps::scaleInPlace(this->scores, scale);
            CudaOps::applySparseAttentionMaskInPlace(this->scores, this->windowSize, this->globalTokenCount, 0, segmentLength);
            CudaOps::softmaxInto(this->scores, this->probabilities);
            CudaOps::copyInto(this->probabilities, this->cachedHeadProbabilities[static_cast<size_t>(headIndex)]);
            CudaMatrix::multiplyInto(this->valueHead, this->probabilities, this->attendedHead);
            CudaOps::writeHead(this->attended, headIndex, this->headDimension, this->attendedHead);
        }

        this->activePackCount = 1;
        CudaMatrix::multiplyInto(this->outputWeight, this->attended, out);
        return;
    }

    const size_t cacheCount = static_cast<size_t>(this->headCount) * static_cast<size_t>(packCount);
    if (this->cachedHeadProbabilities.size() != cacheCount)
        this->cachedHeadProbabilities.resize(cacheCount);

    for (int segmentIndex = 0; segmentIndex < packCount; ++segmentIndex) {
        const int columnStart = segmentIndex * segmentLength;
        CudaOps::extractColumnsInto(this->query, columnStart, segmentLength, this->querySegment);
        CudaOps::extractColumnsInto(this->key, columnStart, segmentLength, this->keySegment);
        CudaOps::extractColumnsInto(this->value, columnStart, segmentLength, this->valueSegment);
        this->attendedSegment.ensureSize(this->query.rows, static_cast<size_t>(segmentLength));
        CudaOps::zeroInPlace(this->attendedSegment);

        for (int headIndex = 0; headIndex < this->headCount; ++headIndex) {
            const int kvHeadIndex = this->kvHeadIndexForQueryHead(headIndex);
            CudaOps::extractHeadInto(this->querySegment, headIndex, this->headDimension, this->queryHead);
            CudaOps::extractHeadInto(this->keySegment, kvHeadIndex, this->headDimension, this->keyHead);
            CudaOps::extractHeadInto(this->valueSegment, kvHeadIndex, this->headDimension, this->valueHead);

            CudaMatrix::multiplyInto(this->keyHead, this->queryHead, this->scores, true, false);
            CudaOps::scaleInPlace(this->scores, scale);
            CudaOps::applySparseAttentionMaskInPlace(this->scores, this->windowSize, this->globalTokenCount, 0, 0);
            CudaOps::softmaxInto(this->scores, this->probabilities);
            const size_t cacheIndex = static_cast<size_t>(headIndex) * static_cast<size_t>(packCount) + static_cast<size_t>(segmentIndex);
            CudaOps::copyInto(this->probabilities, this->cachedHeadProbabilities[cacheIndex]);
            CudaMatrix::multiplyInto(this->valueHead, this->probabilities, this->attendedHead);
            CudaOps::writeHead(this->attendedSegment, headIndex, this->headDimension, this->attendedHead);
        }

        CudaOps::writeColumnsInto(this->attended, columnStart, this->attendedSegment);
    }

    CudaMatrix::multiplyInto(this->outputWeight, this->attended, out);
}

void CudaCausalSelfAttention::attendCachedQuery(const CudaKvCache& cache, CudaMatrix& out) {
    const int usedLength = cache.length;
    const float scale = 1.0f / std::sqrt(static_cast<float>(this->headDimension));
    const int queryPositionStart = usedLength - 1;

    this->attended.ensureSize(this->query.rows, this->query.cols);
    CudaOps::zeroInPlace(this->attended);

    for (int headIndex = 0; headIndex < this->headCount; ++headIndex) {
        const int kvHeadIndex = this->kvHeadIndexForQueryHead(headIndex);
        CudaOps::extractHeadInto(this->query, headIndex, this->headDimension, this->queryHead);
        CudaOps::extractHeadInto(cache.key, kvHeadIndex, this->headDimension, usedLength, this->keyHead);
        CudaOps::extractHeadInto(cache.value, kvHeadIndex, this->headDimension, usedLength, this->valueHead);

        CudaMatrix::multiplyInto(this->keyHead, this->queryHead, this->scores, true, false);
        CudaOps::scaleInPlace(this->scores, scale);
        CudaOps::applySparseAttentionMaskInPlace(this->scores, this->windowSize, this->globalTokenCount, queryPositionStart);
        CudaOps::softmaxInto(this->scores, this->probabilities);
        CudaMatrix::multiplyInto(this->valueHead, this->probabilities, this->attendedHead);
        CudaOps::writeHead(this->attended, headIndex, this->headDimension, this->attendedHead);
    }

    CudaMatrix::multiplyInto(this->outputWeight, this->attended, out);
}

void CudaCausalSelfAttention::forward(const CudaMatrix& input, CudaMatrix& out, int segmentLength) {
    if (input.empty()) throw std::invalid_argument("CudaCausalSelfAttention::forward empty input");
    if (this->queryWeight.empty()) throw std::logic_error("CudaCausalSelfAttention::forward weights not uploaded");
    if (this->queryWeight.cols != input.rows) throw std::invalid_argument("CudaCausalSelfAttention::forward embedding dim mismatch");
    if (segmentLength <= 0) segmentLength = static_cast<int>(input.cols);
    if (static_cast<int>(input.cols) % segmentLength != 0) throw std::invalid_argument("CudaCausalSelfAttention::forward cols not divisible by segmentLength");
    if (segmentLength > this->maximumPositionCount) throw std::invalid_argument("CudaCausalSelfAttention::forward segmentLength exceeds maximumPositionCount");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaCausalSelfAttention::forward no CUDA device");

    this->activeSegmentLength = segmentLength;
    this->projectAndRotate(input, 0, segmentLength);
    this->attendFullSequence(out);
}

void CudaCausalSelfAttention::backward(
    const CudaMatrix& outputGradient,
    CudaMatrix& inputGradient,
    CudaMatrix& queryWeightGradient,
    CudaMatrix& keyWeightGradient,
    CudaMatrix& valueWeightGradient,
    CudaMatrix& outputWeightGradient,
    bool materializeSplitWeightGrads
) {
    if (this->inputCache.empty()) throw std::logic_error("CudaCausalSelfAttention::backward called before forward");
    if (outputGradient.rows != this->outputWeight.rows || outputGradient.cols != this->attended.cols)
        throw std::invalid_argument("CudaCausalSelfAttention::backward output gradient shape mismatch");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaCausalSelfAttention::backward no CUDA device");

    const int segmentLength = this->activeSegmentLength > 0 ? this->activeSegmentLength : static_cast<int>(this->query.cols);
    const int packCount = this->activePackCount > 0 ? this->activePackCount : 1;

    if (this->usedFlashAttention) {
        if (this->flashLogSumExp.rows != static_cast<size_t>(this->headCount) || this->flashLogSumExp.cols != this->query.cols)
            throw std::invalid_argument("CudaCausalSelfAttention::backward flash logSumExp shape mismatch");
    } else {
        const size_t expectedCacheCount = packCount <= 1
            ? static_cast<size_t>(this->headCount)
            : static_cast<size_t>(this->headCount) * static_cast<size_t>(packCount);
        if (this->cachedHeadProbabilities.size() != expectedCacheCount)
            throw std::invalid_argument("CudaCausalSelfAttention::backward head cache size mismatch");
    }

    CudaMatrix::multiplyInto(outputGradient, this->attended, outputWeightGradient, false, true);
    CudaMatrix::multiplyInto(this->outputWeight, outputGradient, this->attendedGradient, true, false);

    this->queryGradient.ensureSize(this->query.rows, this->query.cols);
    this->keyGradient.ensureSize(this->key.rows, this->key.cols);
    this->valueGradient.ensureSize(this->value.rows, this->value.cols);
    // Flash query-outer overwrites dQ; dK/dV need a zero base for atomics (dense path zeros all three).
    if (this->usedFlashAttention) {
        CudaOps::zeroInPlace(this->keyGradient);
        CudaOps::zeroInPlace(this->valueGradient);
    } else {
        CudaOps::zeroInPlace(this->queryGradient);
        CudaOps::zeroInPlace(this->keyGradient);
        CudaOps::zeroInPlace(this->valueGradient);
    }

    const float scale = 1.0f / std::sqrt(static_cast<float>(this->headDimension));

    if (this->usedFlashAttention) {
        const CudaMatrix* flashKey = &this->key;
        const CudaMatrix* flashValue = &this->value;
        CudaMatrix* flashKeyGrad = &this->keyGradient;
        CudaMatrix* flashValueGrad = &this->valueGradient;
        if (!this->usesFusedQkv()) {
            this->expandKvHeadsInto(this->key, this->flashKeyExpanded);
            this->expandKvHeadsInto(this->value, this->flashValueExpanded);
            flashKey = &this->flashKeyExpanded;
            flashValue = &this->flashValueExpanded;
            this->flashKeyGradientExpanded.ensureSize(this->query.rows, this->query.cols);
            this->flashValueGradientExpanded.ensureSize(this->query.rows, this->query.cols);
            CudaOps::zeroInPlace(this->flashKeyGradientExpanded);
            CudaOps::zeroInPlace(this->flashValueGradientExpanded);
            flashKeyGrad = &this->flashKeyGradientExpanded;
            flashValueGrad = &this->flashValueGradientExpanded;
        }

        CudaFlashAttention::backwardMultiHead(
            this->query,
            *flashKey,
            *flashValue,
            this->attended,
            this->flashLogSumExp,
            this->attendedGradient,
            this->queryGradient,
            *flashKeyGrad,
            *flashValueGrad,
            this->flashDelta,
            this->headCount,
            this->headDimension,
            scale,
            true,
            0,
            segmentLength,
            packCount);

        if (!this->usesFusedQkv()) {
            this->reduceKvHeadGradsInto(this->flashKeyGradientExpanded, this->keyGradient);
            this->reduceKvHeadGradsInto(this->flashValueGradientExpanded, this->valueGradient);
        }
    } else if (packCount <= 1) {
        for (int headIndex = 0; headIndex < this->headCount; ++headIndex) {
            const int kvHeadIndex = this->kvHeadIndexForQueryHead(headIndex);
            CudaOps::extractHeadInto(this->attendedGradient, headIndex, this->headDimension, this->attendedHead);
            CudaOps::extractHeadInto(this->query, headIndex, this->headDimension, this->queryHead);
            CudaOps::extractHeadInto(this->key, kvHeadIndex, this->headDimension, this->keyHead);
            CudaOps::extractHeadInto(this->value, kvHeadIndex, this->headDimension, this->valueHead);
            const CudaMatrix& headProbabilities = this->cachedHeadProbabilities[static_cast<size_t>(headIndex)];

            CudaMatrix::multiplyInto(this->attendedHead, headProbabilities, this->valueHeadGradient, false, true);
            CudaMatrix::multiplyInto(this->valueHead, this->attendedHead, this->probabilityGradient, true, false);
            CudaOps::softmaxBackwardInto(headProbabilities, this->probabilityGradient, this->scoreGradient);
            CudaOps::zeroForbiddenScoreGradientsInPlace(this->scoreGradient, this->windowSize, this->globalTokenCount, 0, this->activeSegmentLength);
            CudaOps::scaleInPlace(this->scoreGradient, scale);
            CudaMatrix::multiplyInto(this->keyHead, this->scoreGradient, this->queryHeadGradient);
            CudaMatrix::multiplyInto(this->queryHead, this->scoreGradient, this->keyHeadGradient, false, true);

            CudaOps::writeHead(this->queryGradient, headIndex, this->headDimension, this->queryHeadGradient);
            CudaOps::addHeadInto(this->keyGradient, kvHeadIndex, this->headDimension, this->keyHeadGradient);
            CudaOps::addHeadInto(this->valueGradient, kvHeadIndex, this->headDimension, this->valueHeadGradient);
        }
    } else {
        for (int segmentIndex = 0; segmentIndex < packCount; ++segmentIndex) {
            const int columnStart = segmentIndex * segmentLength;
            CudaOps::extractColumnsInto(this->attendedGradient, columnStart, segmentLength, this->attendedGradientSegment);
            CudaOps::extractColumnsInto(this->query, columnStart, segmentLength, this->querySegment);
            CudaOps::extractColumnsInto(this->key, columnStart, segmentLength, this->keySegment);
            CudaOps::extractColumnsInto(this->value, columnStart, segmentLength, this->valueSegment);

            this->queryGradientSegment.ensureSize(this->query.rows, static_cast<size_t>(segmentLength));
            this->keyGradientSegment.ensureSize(this->key.rows, static_cast<size_t>(segmentLength));
            this->valueGradientSegment.ensureSize(this->value.rows, static_cast<size_t>(segmentLength));
            CudaOps::zeroInPlace(this->queryGradientSegment);
            CudaOps::zeroInPlace(this->keyGradientSegment);
            CudaOps::zeroInPlace(this->valueGradientSegment);

            for (int headIndex = 0; headIndex < this->headCount; ++headIndex) {
                const int kvHeadIndex = this->kvHeadIndexForQueryHead(headIndex);
                CudaOps::extractHeadInto(this->attendedGradientSegment, headIndex, this->headDimension, this->attendedHead);
                CudaOps::extractHeadInto(this->querySegment, headIndex, this->headDimension, this->queryHead);
                CudaOps::extractHeadInto(this->keySegment, kvHeadIndex, this->headDimension, this->keyHead);
                CudaOps::extractHeadInto(this->valueSegment, kvHeadIndex, this->headDimension, this->valueHead);
                const size_t cacheIndex = static_cast<size_t>(headIndex) * static_cast<size_t>(packCount) + static_cast<size_t>(segmentIndex);
                const CudaMatrix& headProbabilities = this->cachedHeadProbabilities[cacheIndex];

                CudaMatrix::multiplyInto(this->attendedHead, headProbabilities, this->valueHeadGradient, false, true);
                CudaMatrix::multiplyInto(this->valueHead, this->attendedHead, this->probabilityGradient, true, false);
                CudaOps::softmaxBackwardInto(headProbabilities, this->probabilityGradient, this->scoreGradient);
                CudaOps::zeroForbiddenScoreGradientsInPlace(this->scoreGradient, this->windowSize, this->globalTokenCount, 0, 0);
                CudaOps::scaleInPlace(this->scoreGradient, scale);
                CudaMatrix::multiplyInto(this->keyHead, this->scoreGradient, this->queryHeadGradient);
                CudaMatrix::multiplyInto(this->queryHead, this->scoreGradient, this->keyHeadGradient, false, true);

                CudaOps::writeHead(this->queryGradientSegment, headIndex, this->headDimension, this->queryHeadGradient);
                CudaOps::addHeadInto(this->keyGradientSegment, kvHeadIndex, this->headDimension, this->keyHeadGradient);
                CudaOps::addHeadInto(this->valueGradientSegment, kvHeadIndex, this->headDimension, this->valueHeadGradient);
            }

            CudaOps::addColumnsInPlace(this->queryGradient, columnStart, this->queryGradientSegment);
            CudaOps::addColumnsInPlace(this->keyGradient, columnStart, this->keyGradientSegment);
            CudaOps::addColumnsInPlace(this->valueGradient, columnStart, this->valueGradientSegment);
        }
    }

    CudaOps::rotaryRotateInverseInPlace(this->queryGradient, this->headCount, this->headDimension, this->pairCount, this->cosTable, this->sinTable, 0, this->activeSegmentLength);
    CudaOps::rotaryRotateInverseInPlace(this->keyGradient, this->kvHeadCount, this->headDimension, this->pairCount, this->cosTable, this->sinTable, 0, this->activeSegmentLength);

    if (this->usesFusedQkv()) {
        if (this->qkvWeight.empty())
            this->syncFusedQkvWeight();

        // Reuse qkvProjected as stacked [dQ; dK; dV] for one dW GEMM and one dX GEMM.
        this->qkvProjected.ensureSize(this->query.rows * 3ull, this->query.cols);
        const size_t sliceElements = this->query.elementCount();
        const size_t sliceBytes = sliceElements * sizeof(float);
        CudaMatmul::memcpyDevice(this->qkvProjected.buffer.deviceData, this->queryGradient.buffer.deviceData, sliceBytes);
        CudaMatmul::memcpyDevice(this->qkvProjected.buffer.deviceData + sliceElements, this->keyGradient.buffer.deviceData, sliceBytes);
        CudaMatmul::memcpyDevice(this->qkvProjected.buffer.deviceData + 2ull * sliceElements, this->valueGradient.buffer.deviceData, sliceBytes);

        CudaMatrix::multiplyInto(this->qkvProjected, this->inputCache, this->qkvWeightGradient, false, true);
        if (materializeSplitWeightGrads) {
            queryWeightGradient.ensureSize(this->queryWeight.rows, this->queryWeight.cols);
            keyWeightGradient.ensureSize(this->keyWeight.rows, this->keyWeight.cols);
            valueWeightGradient.ensureSize(this->valueWeight.rows, this->valueWeight.cols);
            const size_t weightSliceBytes = this->queryWeight.byteCount();
            CudaMatmul::memcpyDevice(queryWeightGradient.buffer.deviceData, this->qkvWeightGradient.buffer.deviceData, weightSliceBytes);
            CudaMatmul::memcpyDevice(keyWeightGradient.buffer.deviceData, this->qkvWeightGradient.buffer.deviceData + this->queryWeight.elementCount(), weightSliceBytes);
            CudaMatmul::memcpyDevice(valueWeightGradient.buffer.deviceData, this->qkvWeightGradient.buffer.deviceData + 2ull * this->queryWeight.elementCount(), weightSliceBytes);
        }

        CudaMatrix::multiplyInto(this->qkvWeight, this->qkvProjected, inputGradient, true, false);
    } else {
        if (materializeSplitWeightGrads) {
            CudaMatrix::multiplyInto(this->queryGradient, this->inputCache, queryWeightGradient, false, true);
            CudaMatrix::multiplyInto(this->keyGradient, this->inputCache, keyWeightGradient, false, true);
            CudaMatrix::multiplyInto(this->valueGradient, this->inputCache, valueWeightGradient, false, true);
        }
        CudaMatrix::multiplyInto(this->queryWeight, this->queryGradient, inputGradient, true, false);
        CudaMatrix::multiplyInto(this->keyWeight, this->keyGradient, this->temp, true, false);
        CudaOps::addInPlace(inputGradient, this->temp);
        CudaMatrix::multiplyInto(this->valueWeight, this->valueGradient, this->temp, true, false);
        CudaOps::addInPlace(inputGradient, this->temp);
    }
}

void CudaCausalSelfAttention::prefill(const CudaMatrix& input, CudaKvCache& cache, CudaMatrix& out) {
    if (input.empty()) throw std::invalid_argument("CudaCausalSelfAttention::prefill empty input");
    if (this->queryWeight.empty()) throw std::logic_error("CudaCausalSelfAttention::prefill weights not uploaded");
    if (this->queryWeight.cols != input.rows) throw std::invalid_argument("CudaCausalSelfAttention::prefill embedding dim mismatch");
    if (static_cast<int>(input.cols) > this->maximumPositionCount) throw std::invalid_argument("CudaCausalSelfAttention::prefill sequence longer than maximumPositionCount");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaCausalSelfAttention::prefill no CUDA device");

    cache.ensureCapacity(static_cast<int>(this->keyWeight.rows), this->maximumPositionCount);
    this->projectAndRotate(input, 0);
    cache.append(this->key, this->value);
    this->attendFullSequence(out, 0);
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
        SmokeLog::skip("Attention fwd");
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

    SmokeLog::result("Attention fwd", "embed=%d heads=%d seq=%d  cpu=%.2fms  gpu=%.2fms  diff=%.2e",
        embeddingDim, headCount, sequenceLength, cpuMilliseconds, static_cast<double>(deviceMilliseconds), maximumDifference);
}

void CudaCausalSelfAttention::runBackwardSmokeDemo(int embeddingDim, int headCount, int sequenceLength, int maximumPositionCount) {
    if (!CudaMatmul::isAvailable()) {
        SmokeLog::skip("Attention bwd");
        return;
    }
    if (embeddingDim <= 0 || headCount <= 0 || sequenceLength <= 0 || maximumPositionCount < sequenceLength)
        throw std::invalid_argument("CudaCausalSelfAttention::runBackwardSmokeDemo invalid dims");

    CausalSelfAttention host = CausalSelfAttention::create(embeddingDim, headCount, maximumPositionCount, 17u, maximumPositionCount, 0);
    Matrix hostInput(static_cast<size_t>(embeddingDim), static_cast<size_t>(sequenceLength), 0.0f);
    unsigned state = 151u;
    for (size_t index = 0; index < hostInput.data.size(); ++index) {
        state = state * 1664525u + 1013904223u;
        hostInput.data[index] = (static_cast<float>(state >> 8) / 16777216.0f) * 2.0f - 1.0f;
    }

    Matrix outputGradient(static_cast<size_t>(embeddingDim), static_cast<size_t>(sequenceLength), 1.0f);

    CausalSelfAttentionCache hostCache;
    host.forward(hostInput, hostCache);
    Matrix hostQueryGrad;
    Matrix hostKeyGrad;
    Matrix hostValueGrad;
    Matrix hostOutputGrad;
    Matrix hostInputGrad = host.backward(outputGradient, hostCache, hostQueryGrad, hostKeyGrad, hostValueGrad, hostOutputGrad);

    CudaCausalSelfAttention device = CudaCausalSelfAttention::createFrom(host);
    CudaMatrix deviceInput;
    deviceInput.upload(hostInput);
    CudaMatrix deviceOutput;
    device.forward(deviceInput, deviceOutput);

    CudaMatrix deviceOutputGradient;
    deviceOutputGradient.upload(outputGradient);
    CudaMatrix deviceInputGrad;
    CudaMatrix deviceQueryGrad;
    CudaMatrix deviceKeyGrad;
    CudaMatrix deviceValueGrad;
    CudaMatrix deviceOutputWeightGrad;
    device.backward(deviceOutputGradient, deviceInputGrad, deviceQueryGrad, deviceKeyGrad, deviceValueGrad, deviceOutputWeightGrad);
    CudaMatmul::throwIfCudaFailed(cudaDeviceSynchronize(), "CudaCausalSelfAttention backward synchronize");

    Matrix deviceHostInputGrad = deviceInputGrad.download();
    Matrix deviceHostQueryGrad = deviceQueryGrad.download();
    Matrix deviceHostKeyGrad = deviceKeyGrad.download();
    Matrix deviceHostValueGrad = deviceValueGrad.download();
    Matrix deviceHostOutputWeightGrad = deviceOutputWeightGrad.download();

    float inputGradDiff = 0.0f;
    for (size_t index = 0; index < hostInputGrad.data.size(); ++index)
        inputGradDiff = (std::max)(inputGradDiff, std::fabs(hostInputGrad.data[index] - deviceHostInputGrad.data[index]));

    float queryWeightGradDiff = 0.0f;
    for (size_t index = 0; index < hostQueryGrad.data.size(); ++index)
        queryWeightGradDiff = (std::max)(queryWeightGradDiff, std::fabs(hostQueryGrad.data[index] - deviceHostQueryGrad.data[index]));

    float keyWeightGradDiff = 0.0f;
    for (size_t index = 0; index < hostKeyGrad.data.size(); ++index)
        keyWeightGradDiff = (std::max)(keyWeightGradDiff, std::fabs(hostKeyGrad.data[index] - deviceHostKeyGrad.data[index]));

    float valueWeightGradDiff = 0.0f;
    for (size_t index = 0; index < hostValueGrad.data.size(); ++index)
        valueWeightGradDiff = (std::max)(valueWeightGradDiff, std::fabs(hostValueGrad.data[index] - deviceHostValueGrad.data[index]));

    float outputWeightGradDiff = 0.0f;
    for (size_t index = 0; index < hostOutputGrad.data.size(); ++index)
        outputWeightGradDiff = (std::max)(outputWeightGradDiff, std::fabs(hostOutputGrad.data[index] - deviceHostOutputWeightGrad.data[index]));

    const float maximumDifference = (std::max)({ inputGradDiff, queryWeightGradDiff, keyWeightGradDiff, valueWeightGradDiff, outputWeightGradDiff });
    SmokeLog::result("Attention bwd", "embed=%d heads=%d seq=%d  diff=%.2e", embeddingDim, headCount, sequenceLength, maximumDifference);
}

void CudaCausalSelfAttention::runKvCacheSmokeDemo(int embeddingDim, int headCount, int sequenceLength, int maximumPositionCount) {
    if (!CudaMatmul::isAvailable()) {
        SmokeLog::skip("Attention KV");
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

    SmokeLog::result("Attention KV", "embed=%d heads=%d seq=%d  cache=%d  diff=%.2e",
        embeddingDim, headCount, sequenceLength, cache.length, maximumDifference);
}

void CudaCausalSelfAttention::runSparseSmokeDemo(int embeddingDim, int headCount, int sequenceLength, int maximumPositionCount, int windowSize, int globalTokenCount) {
    if (!CudaMatmul::isAvailable()) {
        SmokeLog::skip("Sparse Attn S4");
        return;
    }
    if (embeddingDim <= 0 || headCount <= 0 || sequenceLength < 2 || maximumPositionCount < sequenceLength)
        throw std::invalid_argument("CudaCausalSelfAttention::runSparseSmokeDemo invalid dims");
    if (windowSize <= 0 || globalTokenCount < 0)
        throw std::invalid_argument("CudaCausalSelfAttention::runSparseSmokeDemo invalid sparse config");

    CausalSelfAttention host = CausalSelfAttention::create(embeddingDim, headCount, maximumPositionCount, 29u, windowSize, globalTokenCount);
    Matrix hostInput(static_cast<size_t>(embeddingDim), static_cast<size_t>(sequenceLength), 0.0f);
    unsigned state = 101u;
    for (size_t index = 0; index < hostInput.data.size(); ++index) {
        state = state * 1664525u + 1013904223u;
        hostInput.data[index] = (static_cast<float>(state >> 8) / 16777216.0f) * 2.0f - 1.0f;
    }

    CausalSelfAttentionCache hostCache;
    Matrix hostOutput = host.forward(hostInput, hostCache);

    CudaCausalSelfAttention device = CudaCausalSelfAttention::createFrom(host);
    CudaMatrix deviceInput;
    deviceInput.upload(hostInput);
    CudaMatrix deviceOutput;
    device.forward(deviceInput, deviceOutput);
    Matrix deviceHostOutput = deviceOutput.download();

    float forwardDifference = 0.0f;
    for (size_t index = 0; index < hostOutput.data.size(); ++index)
        forwardDifference = (std::max)(forwardDifference, std::fabs(hostOutput.data[index] - deviceHostOutput.data[index]));

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

    Matrix decodeHost = decodeOutput.download();
    float decodeDifference = 0.0f;
    for (int row = 0; row < embeddingDim; ++row) {
        const float fullValue = hostOutput.at(static_cast<size_t>(row), static_cast<size_t>(sequenceLength - 1));
        const float decodeValue = decodeHost.at(static_cast<size_t>(row), 0);
        decodeDifference = (std::max)(decodeDifference, std::fabs(fullValue - decodeValue));
    }

    SmokeLog::result("Sparse Attn S4", "embed=%d heads=%d seq=%d W=%d G=%d  fwd=%.2e  kv=%.2e",
        embeddingDim, headCount, sequenceLength, windowSize, globalTokenCount, forwardDifference, decodeDifference);
}

void CudaCausalSelfAttention::runFlashParitySmokeDemo(int embeddingDim, int headCount, int sequenceLength, int maximumPositionCount) {
    if (!CudaMatmul::isAvailable()) {
        SmokeLog::skip("Attention flash");
        return;
    }
    if (embeddingDim <= 0 || headCount <= 0 || sequenceLength <= 0 || maximumPositionCount < sequenceLength)
        throw std::invalid_argument("CudaCausalSelfAttention::runFlashParitySmokeDemo invalid dims");
    if (embeddingDim / headCount > CudaFlashAttention::maxHeadDimension)
        throw std::invalid_argument("CudaCausalSelfAttention::runFlashParitySmokeDemo headDim exceeds flash max");

    CausalSelfAttention host = CausalSelfAttention::create(embeddingDim, headCount, maximumPositionCount, 41u, maximumPositionCount, 0);
    Matrix hostInput(static_cast<size_t>(embeddingDim), static_cast<size_t>(sequenceLength), 0.0f);
    Matrix hostOutputGradient(static_cast<size_t>(embeddingDim), static_cast<size_t>(sequenceLength), 0.0f);
    unsigned state = 271u;
    for (size_t index = 0; index < hostInput.data.size(); ++index) {
        state = state * 1664525u + 1013904223u;
        hostInput.data[index] = (static_cast<float>(state >> 8) / 16777216.0f) * 2.0f - 1.0f;
        state = state * 1664525u + 1013904223u;
        hostOutputGradient.data[index] = (static_cast<float>(state >> 8) / 16777216.0f) * 2.0f - 1.0f;
    }

    CudaCausalSelfAttention flashDevice = CudaCausalSelfAttention::createFrom(host);
    CudaCausalSelfAttention denseDevice = CudaCausalSelfAttention::createFrom(host);
    flashDevice.preferFlashAttention = true;
    denseDevice.preferFlashAttention = false;

    CudaMatrix deviceInput;
    deviceInput.upload(hostInput);
    CudaMatrix flashOutput;
    CudaMatrix denseOutput;
    flashDevice.forward(deviceInput, flashOutput);
    denseDevice.forward(deviceInput, denseOutput);
    if (!flashDevice.usedFlashAttention)
        throw std::logic_error("CudaCausalSelfAttention::runFlashParitySmokeDemo flash path not selected");
    if (denseDevice.usedFlashAttention)
        throw std::logic_error("CudaCausalSelfAttention::runFlashParitySmokeDemo dense path unexpectedly used flash");

    Matrix flashHostOutput = flashOutput.download();
    Matrix denseHostOutput = denseOutput.download();
    float forwardDifference = 0.0f;
    for (size_t index = 0; index < flashHostOutput.data.size(); ++index)
        forwardDifference = (std::max)(forwardDifference, std::fabs(flashHostOutput.data[index] - denseHostOutput.data[index]));

    CudaMatrix deviceOutputGradient;
    deviceOutputGradient.upload(hostOutputGradient);
    CudaMatrix flashInputGrad;
    CudaMatrix denseInputGrad;
    CudaMatrix flashQueryGrad;
    CudaMatrix denseQueryGrad;
    CudaMatrix flashKeyGrad;
    CudaMatrix denseKeyGrad;
    CudaMatrix flashValueGrad;
    CudaMatrix denseValueGrad;
    CudaMatrix flashOutputWeightGrad;
    CudaMatrix denseOutputWeightGrad;
    flashDevice.backward(deviceOutputGradient, flashInputGrad, flashQueryGrad, flashKeyGrad, flashValueGrad, flashOutputWeightGrad);
    denseDevice.backward(deviceOutputGradient, denseInputGrad, denseQueryGrad, denseKeyGrad, denseValueGrad, denseOutputWeightGrad);
    CudaMatmul::throwIfCudaFailed(cudaDeviceSynchronize(), "CudaCausalSelfAttention flash parity synchronize");

    auto maxAbsDiff = [](const CudaMatrix& left, const CudaMatrix& right) -> float {
        Matrix leftHost = left.download();
        Matrix rightHost = right.download();
        float maximumDifference = 0.0f;
        for (size_t index = 0; index < leftHost.data.size(); ++index)
            maximumDifference = (std::max)(maximumDifference, std::fabs(leftHost.data[index] - rightHost.data[index]));
        return maximumDifference;
    };

    const float inputGradDiff = maxAbsDiff(flashInputGrad, denseInputGrad);
    const float queryGradDiff = maxAbsDiff(flashQueryGrad, denseQueryGrad);
    const float keyGradDiff = maxAbsDiff(flashKeyGrad, denseKeyGrad);
    const float valueGradDiff = maxAbsDiff(flashValueGrad, denseValueGrad);
    const float outputWeightGradDiff = maxAbsDiff(flashOutputWeightGrad, denseOutputWeightGrad);
    const float backwardDifference = (std::max)({ inputGradDiff, queryGradDiff, keyGradDiff, valueGradDiff, outputWeightGradDiff });

    SmokeLog::result("Attention flash", "embed=%d heads=%d seq=%d  fwdDiff=%.2e  bwdDiff=%.2e",
        embeddingDim, headCount, sequenceLength, forwardDifference, backwardDifference);
}

void CudaCausalSelfAttention::runGqaCudaSmokeDemo() {
    if (!CudaMatmul::isAvailable()) {
        SmokeLog::skip("Attention GQA cuda");
        return;
    }

    const int embed = 64;
    const int heads = 8;
    const int kvHeads = 2;
    const int seq = 16; // multiple of 4 → flash-eligible
    const int maxPos = 32;
    const unsigned seed = 53u;

    auto maxAbsDiff = [](const Matrix& a, const Matrix& b) -> float {
        if (a.data.size() != b.data.size())
            throw std::runtime_error("CudaCausalSelfAttention GQA smoke size mismatch");
        float diff = 0.0f;
        for (size_t i = 0; i < a.data.size(); ++i)
            diff = (std::max)(diff, std::fabs(a.data[i] - b.data[i]));
        return diff;
    };

    CausalSelfAttention host = CausalSelfAttention::create(
        embed, heads, maxPos, seed, maxPos, 0, RotaryEmbedding::DefaultBase, kvHeads);
    Matrix hostInput(static_cast<size_t>(embed), static_cast<size_t>(seq), 0.0f);
    unsigned state = 91u;
    for (size_t i = 0; i < hostInput.data.size(); ++i) {
        state = state * 1664525u + 1013904223u;
        hostInput.data[i] = (static_cast<float>(state >> 8) / 16777216.0f) * 2.0f - 1.0f;
    }
    Matrix outputGradient(static_cast<size_t>(embed), static_cast<size_t>(seq), 0.01f);

    CausalSelfAttentionCache hostCache;
    Matrix hostOut = host.forward(hostInput, hostCache);
    Matrix hostQ, hostK, hostV, hostO;
    Matrix hostInGrad = host.backward(outputGradient, hostCache, hostQ, hostK, hostV, hostO);

    CudaMatrix deviceInput;
    deviceInput.upload(hostInput);
    CudaMatrix deviceOutGrad;
    deviceOutGrad.upload(outputGradient);

    auto runDevice = [&](bool preferFlash, Matrix& outHost, Matrix& inGradHost, Matrix& qg, Matrix& kg, Matrix& vg, Matrix& og) {
        CudaCausalSelfAttention device = CudaCausalSelfAttention::createFrom(host);
        device.preferFlashAttention = preferFlash;
        CudaMatrix out;
        device.forward(deviceInput, out);
        CudaMatmul::throwIfCudaFailed(cudaDeviceSynchronize(), "GQA smoke forward sync");
        if (preferFlash && !device.usedFlashAttention)
            throw std::logic_error("CudaCausalSelfAttention GQA smoke expected flash path");
        if (!preferFlash && device.usedFlashAttention)
            throw std::logic_error("CudaCausalSelfAttention GQA smoke dense path used flash");

        CudaMatrix inGrad, dq, dk, dv, dOutW;
        device.backward(deviceOutGrad, inGrad, dq, dk, dv, dOutW);
        CudaMatmul::throwIfCudaFailed(cudaDeviceSynchronize(), "GQA smoke backward sync");
        outHost = out.download();
        inGradHost = inGrad.download();
        qg = dq.download();
        kg = dk.download();
        vg = dv.download();
        og = dOutW.download();
    };

    Matrix denseOut, denseIn, denseQ, denseK, denseV, denseO;
    runDevice(false, denseOut, denseIn, denseQ, denseK, denseV, denseO);
    const float hostDenseFwd = maxAbsDiff(hostOut, denseOut);
    const float hostDenseBwd = (std::max)(
        maxAbsDiff(hostInGrad, denseIn),
        (std::max)(
            maxAbsDiff(hostQ, denseQ),
            (std::max)(
                maxAbsDiff(hostK, denseK),
                (std::max)(maxAbsDiff(hostV, denseV), maxAbsDiff(hostO, denseO)))));
    if (hostDenseFwd > 2.0e-4f || hostDenseBwd > 2.0e-3f)
        throw std::runtime_error("CudaCausalSelfAttention GQA smoke host↔dense parity failed");

    Matrix flashOut, flashIn, flashQ, flashK, flashV, flashO;
    runDevice(true, flashOut, flashIn, flashQ, flashK, flashV, flashO);
    const float flashDenseFwd = maxAbsDiff(flashOut, denseOut);
    const float flashDenseBwd = (std::max)(
        maxAbsDiff(flashIn, denseIn),
        (std::max)(
            maxAbsDiff(flashQ, denseQ),
            (std::max)(
                maxAbsDiff(flashK, denseK),
                (std::max)(maxAbsDiff(flashV, denseV), maxAbsDiff(flashO, denseO)))));
    // Flash uses expanded KV copies; fwd stays tight, bwd allows flash↔dense numeric gap (same as MHA flash parity style).
    if (flashDenseFwd > 1.0e-3f || flashDenseBwd > 1.0e-1f)
        throw std::runtime_error("CudaCausalSelfAttention GQA smoke flash↔dense parity failed");

    SmokeLog::result(
        "Attention GQA cuda",
        "q=%d kv=%d seq=%d  hostDenseFwd=%.2e  hostDenseBwd=%.2e  flashDenseFwd=%.2e  flashDenseBwd=%.2e",
        heads,
        kvHeads,
        seq,
        hostDenseFwd,
        hostDenseBwd,
        flashDenseFwd,
        flashDenseBwd);
}
