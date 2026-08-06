#include "LanguageModel.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

#include "../Activations/Softmax.hpp"
#include "../Cuda/CudaLanguageModel.hpp"
#include "../Cuda/CudaAmp.hpp"
#include "../Cuda/CudaAdam.hpp"
#include "../Cuda/CudaBao.hpp"
#include "../Cuda/CudaMatmul.hpp"
#include "../Cuda/CudaMuon.hpp"
#include "../Initializers/UniformInit.hpp"
#include "../IO/SafeTensors.hpp"
#include "../Losses/CrossEntropy.hpp"
#include "../Tokenizer/BPETokenizer.hpp"
#include "../Utils/SmokeLog.hpp"

#include <cuda_runtime.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

LanguageModelGradients LanguageModelGradients::zerosFrom(const LanguageModel& model) {
    LanguageModelGradients gradients;
    gradients.tokenEmbedding = Matrix::zerosLike(model.tokenEmbedding.weight);
    gradients.blocks.reserve(model.blocks.size());
    for (const TransformerBlock& block : model.blocks)
        gradients.blocks.push_back(TransformerBlockGradients::zerosFrom(block));
    gradients.finalNormGamma = Matrix::zerosLike(model.finalNorm.gamma);
    if (model.tieEmbeddingProjection)
        gradients.projectionWeight = Matrix();
    else
        gradients.projectionWeight = Matrix::zerosLike(model.outputProjection.weight);
    gradients.projectionBias = Matrix::zerosLike(model.outputProjection.bias);
    return gradients;
}

void LanguageModelGradients::zeroInPlace() {
    Matrix::zeroInPlace(this->tokenEmbedding);
    for (TransformerBlockGradients& block : this->blocks)
        block.zeroInPlace();
    Matrix::zeroInPlace(this->finalNormGamma);
    if (!this->projectionWeight.empty())
        Matrix::zeroInPlace(this->projectionWeight);
    Matrix::zeroInPlace(this->projectionBias);
}

void LanguageModelGradients::addInPlace(const LanguageModelGradients& other) {
    Matrix::addInPlace(this->tokenEmbedding, other.tokenEmbedding);
    for (size_t blockIndex = 0; blockIndex < this->blocks.size(); ++blockIndex)
        this->blocks[blockIndex].addInPlace(other.blocks[blockIndex]);
    Matrix::addInPlace(this->finalNormGamma, other.finalNormGamma);
    if (!this->projectionWeight.empty() && !other.projectionWeight.empty())
        Matrix::addInPlace(this->projectionWeight, other.projectionWeight);
    Matrix::addInPlace(this->projectionBias, other.projectionBias);
}

void LanguageModelGradients::scaleInPlace(float scalar) {
    Matrix::scaleInPlace(this->tokenEmbedding, scalar);
    for (TransformerBlockGradients& block : this->blocks)
        block.scaleInPlace(scalar);
    Matrix::scaleInPlace(this->finalNormGamma, scalar);
    if (!this->projectionWeight.empty())
        Matrix::scaleInPlace(this->projectionWeight, scalar);
    Matrix::scaleInPlace(this->projectionBias, scalar);
}

LanguageModel::LanguageModel(int vocabularySize, int embeddingDim, int maximumPositionCount, Adam optimizer, int blockCount, int headCount)
    : tokenEmbedding(vocabularySize, embeddingDim), finalNorm(embeddingDim), outputProjection(UniformInit::matrix(vocabularySize, embeddingDim, 0.1f, 31u), UniformInit::matrix(vocabularySize, 1, 0.01f, 32u)), optimizer(optimizer), maximumPositionCount(maximumPositionCount), tieEmbeddingProjection(true), deviceStale(false), deviceTrainEnabled(false) {
    if (maximumPositionCount <= 0) throw std::invalid_argument("LanguageModel maximumPositionCount must be > 0");
    if (blockCount <= 0) throw std::invalid_argument("LanguageModel blockCount must be > 0");
    if (headCount <= 0) throw std::invalid_argument("LanguageModel headCount must be > 0");

    this->blocks.reserve(static_cast<size_t>(blockCount));
    for (int blockIndex = 0; blockIndex < blockCount; ++blockIndex)
        this->blocks.push_back(TransformerBlock(embeddingDim, headCount, maximumPositionCount, 21u + static_cast<unsigned>(blockIndex) * 100u));

    // weight tying: LM head shares tokenEmbedding; drop untied projection weight + Adam
    this->outputProjection.weight = Matrix();
    // Adam moments allocated lazily on first update (4B ctor must not reserve 2x param RAM).
    this->tokenEmbeddingState = AdamState{};
    this->finalNormGammaState = AdamState{};
    this->projectionWeightState = AdamState{};
    this->projectionBiasState = AdamState{};
}

Matrix& LanguageModel::lmHeadWeight() {
    return this->tieEmbeddingProjection ? this->tokenEmbedding.weight : this->outputProjection.weight;
}

const Matrix& LanguageModel::lmHeadWeight() const {
    return this->tieEmbeddingProjection ? this->tokenEmbedding.weight : this->outputProjection.weight;
}

void LanguageModel::setTieEmbeddingProjection(bool enabled) {
    if (this->tieEmbeddingProjection == enabled) return;

    if (enabled) {
        this->outputProjection.weight = Matrix();
        this->projectionWeightState = AdamState{};
        this->tieEmbeddingProjection = true;
    } else {
        this->outputProjection.weight = this->tokenEmbedding.weight;
        this->projectionWeightState = AdamState::zerosLike(this->outputProjection.weight);
        this->tieEmbeddingProjection = false;
    }

    if (this->device != nullptr)
        this->deviceStale = true;
}

size_t LanguageModel::parameterElementCount() const {
    auto matrixElements = [](const Matrix& matrix) -> size_t {
        return matrix.empty() ? 0ull : matrix.data.size();
    };

    size_t total = 0;
    total += matrixElements(this->tokenEmbedding.weight);
    total += matrixElements(this->finalNorm.gamma);
    if (!this->tieEmbeddingProjection)
        total += matrixElements(this->outputProjection.weight);
    total += matrixElements(this->outputProjection.bias);

    for (const TransformerBlock& block : this->blocks) {
        total += matrixElements(block.attentionNorm.gamma);
        total += matrixElements(block.attention.queryWeight);
        total += matrixElements(block.attention.keyWeight);
        total += matrixElements(block.attention.valueWeight);
        total += matrixElements(block.attention.outputWeight);
        total += matrixElements(block.feedForwardNorm.gamma);
        total += matrixElements(block.feedForward.gateWeight);
        total += matrixElements(block.feedForward.gateBias);
        total += matrixElements(block.feedForward.upWeight);
        total += matrixElements(block.feedForward.upBias);
        total += matrixElements(block.feedForward.downWeight);
        total += matrixElements(block.feedForward.downBias);
    }
    return total;
}

void LanguageModel::applyCudaVramPackBudget(float freeFraction, size_t safetyReserveBytes) {
    if (this->device == nullptr) this->enableCuda();
    if (this->device == nullptr) return;
    this->device->maxPackedColumnsManual = false;
    this->device->applyVramPackBudget(freeFraction, safetyReserveBytes);
    if (this->deviceTrainEnabled) {
        this->device->trainStateReady = false;
        this->device->ensureTrainState();
    }
}

void LanguageModel::setCudaPreferTrainGraph(bool enabled) {
    if (this->device == nullptr) this->enableCuda();
    if (this->device == nullptr) return;
    if (enabled && CudaAdam::preferFp16GpuWeights) {
        std::cout << "LanguageModel::setCudaPreferTrainGraph: keeping graphs off (FP16 working weights)\n";
        enabled = false;
    }
    this->device->preferTrainGraph = enabled;
    if (!enabled)
        this->device->releaseTrainGraph();
}

double LanguageModel::probeCudaPackedTrainTokensPerSecond(int sequenceLength, int warmupSteps, int timedSteps) {
    if (this->device == nullptr || !this->deviceTrainEnabled)
        throw std::logic_error("LanguageModel::probeCudaPackedTrainTokensPerSecond requires enableCudaTrain");
    if (sequenceLength <= 0 || sequenceLength > this->maximumPositionCount)
        throw std::invalid_argument("LanguageModel::probeCudaPackedTrainTokensPerSecond invalid sequenceLength");
    if (warmupSteps < 0 || timedSteps <= 0)
        throw std::invalid_argument("LanguageModel::probeCudaPackedTrainTokensPerSecond invalid step counts");

    CudaLanguageModel& device = *this->device;
    // Preserve preferTrainGraph from caller (do not force capture on one-shot probes).
    device.adam = CudaAdam(this->optimizer.learningRate, this->optimizer.beta1, this->optimizer.beta2, this->optimizer.epsilon);
    device.ensureTrainState();

    const int vocab = this->tokenEmbedding.vocabSize();
    const int packBatch = (std::max)(1, (std::min)(32, device.maxPackExamplesForSegment(sequenceLength)));
    std::vector<LanguageModelExample> examples(static_cast<size_t>(packBatch));
    unsigned rng = 100003u;
    for (int exampleIndex = 0; exampleIndex < packBatch; ++exampleIndex) {
        examples[static_cast<size_t>(exampleIndex)].inputTokenIds.resize(static_cast<size_t>(sequenceLength));
        examples[static_cast<size_t>(exampleIndex)].targetTokenIds.resize(static_cast<size_t>(sequenceLength));
        for (size_t index = 0; index < static_cast<size_t>(sequenceLength); ++index) {
            rng = rng * 1664525u + 1013904223u;
            examples[static_cast<size_t>(exampleIndex)].inputTokenIds[index] = static_cast<int>(rng % static_cast<unsigned>(vocab));
            rng = rng * 1664525u + 1013904223u;
            examples[static_cast<size_t>(exampleIndex)].targetTokenIds[index] = static_cast<int>(rng % static_cast<unsigned>(vocab));
        }
    }
    std::vector<const LanguageModelExample*> packPointers(static_cast<size_t>(packBatch));
    for (int exampleIndex = 0; exampleIndex < packBatch; ++exampleIndex)
        packPointers[static_cast<size_t>(exampleIndex)] = &examples[static_cast<size_t>(exampleIndex)];

    device.preferTrainProgress = false;
#ifdef _MSC_VER
    device.preferTrainPhaseTrace = false;
    {
        char* _sentinel_phase_trace_val = nullptr;
        size_t _sentinel_phase_trace_len = 0;
        if (_dupenv_s(&_sentinel_phase_trace_val, &_sentinel_phase_trace_len, "SENTINEL_PHASE_TRACE") == 0 && _sentinel_phase_trace_val != nullptr) {
            device.preferTrainPhaseTrace = true;
            free(_sentinel_phase_trace_val);
        }
    }
#else
    device.preferTrainPhaseTrace = (std::getenv("SENTINEL_PHASE_TRACE") != nullptr);
#endif
    device.trainProgressIntervalSec = 5.0;
    device.trainProgressEpochStart = {};
    device.trainProgressLastPrint = {};

    char stepLabel[64];
    for (int step = 0; step < warmupSteps; ++step) {
        std::snprintf(stepLabel, sizeof(stepLabel), "probe warmup");
        device.trainProgressLabel = stepLabel;
        device.trainProgressStep = step + 1;
        device.trainProgressStepTotal = warmupSteps;
        device.trainGradients.zeroInPlace();
        device.accumulatePackedExamples(packPointers.data(), packBatch, device.trainGradients);
        device.applyGradients(device.trainGradients, 1.0f / static_cast<float>(packBatch));
        SmokeLog::progressDone();
        std::cout << "probe: warmup " << (step + 1) << "/" << warmupSteps
                  << "  pack=" << packBatch << " seq=" << sequenceLength << " done" << std::endl;
    }
    if (cudaDeviceSynchronize() != cudaSuccess)
        throw std::runtime_error("LanguageModel::probeCudaPackedTrainTokensPerSecond warmup sync failed");

    device.trainProgressEpochStart = {};
    device.trainProgressLastPrint = {};
    const auto start = std::chrono::steady_clock::now();
    for (int step = 0; step < timedSteps; ++step) {
        std::snprintf(stepLabel, sizeof(stepLabel), "probe timed");
        device.trainProgressLabel = stepLabel;
        device.trainProgressStep = step + 1;
        device.trainProgressStepTotal = timedSteps;
        device.trainGradients.zeroInPlace();
        device.accumulatePackedExamples(packPointers.data(), packBatch, device.trainGradients);
        device.applyGradients(device.trainGradients, 1.0f / static_cast<float>(packBatch));
        SmokeLog::progressDone();
        std::cout << "probe: timed " << (step + 1) << "/" << timedSteps
                  << "  pack=" << packBatch << " seq=" << sequenceLength << " done" << std::endl;
    }
    device.trainProgressLabel = "";
    device.trainProgressStep = 0;
    device.trainProgressStepTotal = 0;
    if (cudaDeviceSynchronize() != cudaSuccess)
        throw std::runtime_error("LanguageModel::probeCudaPackedTrainTokensPerSecond timed sync failed");
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    if (seconds <= 0.0) return 0.0;
    return static_cast<double>(sequenceLength) * static_cast<double>(packBatch) * static_cast<double>(timedSteps) / seconds;
}

void LanguageModel::probeCudaTrainStepProfile(int sequenceLength, int warmupSteps, int timedSteps) {
    if (this->device == nullptr || !this->deviceTrainEnabled)
        throw std::logic_error("LanguageModel::probeCudaTrainStepProfile requires enableCudaTrain");
    if (sequenceLength <= 0 || sequenceLength > this->maximumPositionCount)
        throw std::invalid_argument("LanguageModel::probeCudaTrainStepProfile invalid sequenceLength");
    if (warmupSteps < 0 || timedSteps <= 0)
        throw std::invalid_argument("LanguageModel::probeCudaTrainStepProfile invalid step counts");

    CudaLanguageModel& device = *this->device;
    device.preferTrainGraph = true;
    device.adam = CudaAdam(this->optimizer.learningRate, this->optimizer.beta1, this->optimizer.beta2, this->optimizer.epsilon);
    device.ensureTrainState();

    const int vocab = this->tokenEmbedding.vocabSize();
    const int packBatch = (std::max)(1, (std::min)(32, device.maxPackExamplesForSegment(sequenceLength)));
    std::vector<LanguageModelExample> examples(static_cast<size_t>(packBatch));
    unsigned rng = 100003u;
    for (int exampleIndex = 0; exampleIndex < packBatch; ++exampleIndex) {
        examples[static_cast<size_t>(exampleIndex)].inputTokenIds.resize(static_cast<size_t>(sequenceLength));
        examples[static_cast<size_t>(exampleIndex)].targetTokenIds.resize(static_cast<size_t>(sequenceLength));
        for (size_t index = 0; index < static_cast<size_t>(sequenceLength); ++index) {
            rng = rng * 1664525u + 1013904223u;
            examples[static_cast<size_t>(exampleIndex)].inputTokenIds[index] = static_cast<int>(rng % static_cast<unsigned>(vocab));
            rng = rng * 1664525u + 1013904223u;
            examples[static_cast<size_t>(exampleIndex)].targetTokenIds[index] = static_cast<int>(rng % static_cast<unsigned>(vocab));
        }
    }
    std::vector<const LanguageModelExample*> packPointers(static_cast<size_t>(packBatch));
    for (int exampleIndex = 0; exampleIndex < packBatch; ++exampleIndex)
        packPointers[static_cast<size_t>(exampleIndex)] = &examples[static_cast<size_t>(exampleIndex)];

    for (int step = 0; step < warmupSteps; ++step) {
        device.trainGradients.zeroInPlace();
        device.accumulatePackedExamples(packPointers.data(), packBatch, device.trainGradients);
        device.applyGradients(device.trainGradients, 1.0f / static_cast<float>(packBatch));
    }
    if (cudaDeviceSynchronize() != cudaSuccess)
        throw std::runtime_error("LanguageModel::probeCudaTrainStepProfile warmup sync failed");

    device.muon.profileEnabled = device.preferMuon;
    device.muon.profile.reset();

    cudaEvent_t accumStart = nullptr;
    cudaEvent_t accumStop = nullptr;
    cudaEvent_t applyStart = nullptr;
    cudaEvent_t applyStop = nullptr;
    if (cudaEventCreate(&accumStart) != cudaSuccess || cudaEventCreate(&accumStop) != cudaSuccess
        || cudaEventCreate(&applyStart) != cudaSuccess || cudaEventCreate(&applyStop) != cudaSuccess)
        throw std::runtime_error("LanguageModel::probeCudaTrainStepProfile event create failed");

    double accumMs = 0.0;
    double applyMs = 0.0;
    for (int step = 0; step < timedSteps; ++step) {
        device.trainGradients.zeroInPlace();
        cudaEventRecord(accumStart);
        device.accumulatePackedExamples(packPointers.data(), packBatch, device.trainGradients);
        cudaEventRecord(accumStop);
        cudaEventSynchronize(accumStop);
        float stepAccumMs = 0.0f;
        cudaEventElapsedTime(&stepAccumMs, accumStart, accumStop);
        accumMs += static_cast<double>(stepAccumMs);

        cudaEventRecord(applyStart);
        device.applyGradients(device.trainGradients, 1.0f / static_cast<float>(packBatch));
        cudaEventRecord(applyStop);
        cudaEventSynchronize(applyStop);
        float stepApplyMs = 0.0f;
        cudaEventElapsedTime(&stepApplyMs, applyStart, applyStop);
        applyMs += static_cast<double>(stepApplyMs);
    }

    cudaEventDestroy(accumStart);
    cudaEventDestroy(accumStop);
    cudaEventDestroy(applyStart);
    cudaEventDestroy(applyStop);
    device.muon.profileEnabled = false;

    const double invSteps = 1.0 / static_cast<double>(timedSteps);
    const double avgAccum = accumMs * invSteps;
    const double avgApply = applyMs * invSteps;
    const CudaMuonProfile& mp = device.muon.profile;
    const double muonInv = timedSteps > 0 ? invSteps : 1.0;

    SmokeLog::result(
        "step profile",
        "pack=%d seq=%d  fwdBwd=%.2fms  apply=%.2fms  muonUpdates=%d  opt=%s",
        packBatch,
        sequenceLength,
        avgAccum,
        avgApply,
        mp.updateCount,
        device.preferMuon ? "muon+adam" : "adam");

    if (device.preferMuon && mp.updateCount > 0) {
        SmokeLog::result(
            "muon sections /step",
            "momentum=%.2f  normalize=%.2f  nsGemm=%.2f  nsElem=%.2f  apply=%.2f  muonTotal=%.2f  (ms)",
            mp.momentumMs * muonInv,
            mp.normalizeMs * muonInv,
            mp.nsGemmMs * muonInv,
            mp.nsElemwiseMs * muonInv,
            mp.applyMs * muonInv,
            mp.totalMs() * muonInv);
        const double muonTotal = mp.totalMs() * muonInv;
        if (muonTotal > 0.0) {
            SmokeLog::result(
                "muon share",
                "nsGemm=%.0f%%  nsElem=%.0f%%  normalize=%.0f%%  momentum=%.0f%%  apply=%.0f%%  of muon; muon/apply=%.0f%%",
                100.0 * (mp.nsGemmMs * muonInv) / muonTotal,
                100.0 * (mp.nsElemwiseMs * muonInv) / muonTotal,
                100.0 * (mp.normalizeMs * muonInv) / muonTotal,
                100.0 * (mp.momentumMs * muonInv) / muonTotal,
                100.0 * (mp.applyMs * muonInv) / muonTotal,
                100.0 * muonTotal / (std::max)(avgApply, 1e-6));
        }
    }
}

LanguageModel::~LanguageModel() = default;

LanguageModel::LanguageModel(LanguageModel&&) noexcept = default;

LanguageModel& LanguageModel::operator=(LanguageModel&&) noexcept = default;

void LanguageModel::enableCuda() {
    if (!CudaMatmul::isAvailable()) {
        std::cout << "LanguageModel::enableCuda: no CUDA device\n";
        this->device.reset();
        this->deviceStale = false;
        this->deviceTrainEnabled = false;
        return;
    }

    this->device = std::make_unique<CudaLanguageModel>();
    this->device->uploadFrom(*this);
    this->deviceStale = false;
    std::cout << "LanguageModel::enableCuda: device mirror active (forward/generate)\n";
}

void LanguageModel::enableCudaTrain() {
    if (this->device == nullptr) this->enableCuda();
    if (this->device == nullptr) {
        this->deviceTrainEnabled = false;
        return;
    }

    this->deviceTrainEnabled = true;

    // BAO Auto: re-resolve after weights are on device so free VRAM is meaningful.
    if (CudaBao::enabled && CudaBao::request == BaoMode::Auto) {
        size_t freeBytes = 0;
        size_t totalBytes = 0;
        if (CudaMatmul::isAvailable()) {
            cudaDeviceSynchronize();
            cudaMemGetInfo(&freeBytes, &totalBytes);
        }
        const size_t parameterBytes = this->parameterElementCount() * sizeof(float);
        const BaoMode picked = CudaBao::resolveAndApply(freeBytes, parameterBytes, 0);
        std::cout << "LanguageModel::enableCudaTrain: BAO Auto → " << CudaBao::modeName(picked)
                  << "  freeVram=" << (freeBytes / (1024ull * 1024ull)) << " MiB"
                  << "  params=" << (parameterBytes / (1024ull * 1024ull)) << " MiB\n";
    }

    auto applyHostBaoSideEffects = [this]() {
        CudaAmp::preferMixedPrecision = true;
        this->device->preferTrainGraph = false;
        this->device->releaseTrainGraph();
        if (this->device->preferMuon) {
            this->device->preferMuon = false;
            std::cout << "LanguageModel::enableCudaTrain: disabling Muon (host BAO path)\n";
        }
    };

    if (CudaBao::resolved == BaoMode::HostFusedHalfAdam || CudaBao::resolved == BaoMode::HostFusedHalfSgd
        || CudaAdam::preferHostGradients || CudaAdam::preferHostSgd)
        applyHostBaoSideEffects();
    else if (CudaBao::resolved == BaoMode::GpuInt8Adam || !CudaAdam::preferCpuOffload) {
        // Fast path: keep CUDA graphs available (ckpt Off below).
        this->device->preferTrainGraph = true;
    }

    // GpuInt8 / resident: Off (retain acts → graphs).
    // HostFusedHalfAdam: Off on mid (faster); Full on large (matches prior 4B offload setup).
    // HostFusedHalfSgd (~4B): Full from the start.
    if (CudaBao::resolved == BaoMode::HostFusedHalfSgd || CudaAdam::preferHostSgd
        || ((CudaBao::resolved == BaoMode::HostFusedHalfAdam || CudaAdam::preferHostGradients)
            && this->device->blocks.size() >= 24)) {
        this->device->setActivationCheckpointMode(ActivationCheckpointMode::Full);
    } else {
        this->device->setActivationCheckpointMode(ActivationCheckpointMode::Off);
    }

    CudaAmp::preferMixedPrecision = true;
    CudaAmp::useLossScaling = this->tokenEmbedding.embeddingDim() >= 256;
    CudaAmp::resetLossScaler();
    if (CudaAdam::preferCpuOffload)
        CudaAdam::preferInt8Moments = false;
    else
        CudaAdam::preferInt8Moments = true;

    if (CudaBao::resolved == BaoMode::HostFusedHalfSgd || CudaAdam::preferHostSgd) {
        this->device->releasePackedTrainWorkspaces();
        this->device->tuneOffloadCheckpointAndPack(4096);
    } else {
        this->device->trainStateReady = false;
        this->device->ensureTrainState();
        this->device->applyVramPackBudget();
        this->device->trainStateReady = false;
        this->device->ensureTrainState();

        // Auto + GpuInt8 but pack starved to the minimum → fall back once to HostFusedHalfAdam.
        if (CudaBao::enabled && CudaBao::request == BaoMode::Auto
            && CudaBao::resolved == BaoMode::GpuInt8Adam
            && this->device->maxPackedColumns <= CudaLanguageModel::lengthBucketStep) {
            std::cout << "LanguageModel::enableCudaTrain: GpuInt8 pack starved (maxPackCols="
                      << this->device->maxPackedColumns << ") → HostFusedHalfAdam\n";
            CudaBao::request = BaoMode::Auto;
            CudaBao::apply(BaoMode::HostFusedHalfAdam);
            applyHostBaoSideEffects();
            this->device->setActivationCheckpointMode(ActivationCheckpointMode::Off);
            CudaAdam::preferInt8Moments = false;
            this->device->trainStateReady = false;
            this->device->releasePackedTrainWorkspaces();
            this->device->ensureTrainState();
            this->device->applyVramPackBudget();
            this->device->trainStateReady = false;
            this->device->ensureTrainState();
        }
    }
    std::cout << "LanguageModel::enableCudaTrain: device training enabled (packed batches, ckpt="
              << CudaLanguageModel::activationCheckpointModeName(this->device->activationCheckpointMode)
              << ", opt=" << (this->device->preferMuon ? "muon+adam" : "adam")
              << ", bao=" << (CudaBao::enabled ? CudaBao::modeName(CudaBao::resolved) : "off")
              << ", FP16 amp "
              << (CudaAmp::preferMixedPrecision ? "on" : "off")
              << ", lossScale=" << (CudaAmp::useLossScaling ? "on" : "off")
              << ", int8 adam " << (CudaAdam::preferInt8Moments ? "on" : "off")
              << ", cpuAdam=" << (CudaAdam::preferCpuOffload ? "on" : "off")
              << ", hostGrads=" << (CudaAdam::preferHostGradients ? "on" : "off")
              << ", halfCkpt=" << (this->device->useHalfActivationCheckpoints() ? "on" : "off")
              << ", tieEmbed=" << (this->tieEmbeddingProjection ? "on" : "off")
              << ", maxPackCols=" << this->device->maxPackedColumns << ")\n";
}

void LanguageModel::setCudaPreferCpuAdamOffload(bool enabled) {
    if (this->device == nullptr) this->enableCuda();
    if (this->device == nullptr) return;
    if (enabled) {
        // Legacy alias → BAO HostFusedHalfAdam (fused FP16 grad D2H + async host Adam).
        CudaBao::request = BaoMode::HostFusedHalfAdam;
        CudaBao::apply(BaoMode::HostFusedHalfAdam);
        CudaAmp::preferMixedPrecision = true;
        this->device->preferTrainGraph = false;
        this->device->releaseTrainGraph();
        if (this->device->preferMuon) {
            this->device->preferMuon = false;
            std::cout << "LanguageModel::setCudaPreferCpuAdamOffload: disabling Muon (FP16 GPU weights)\n";
        }
        std::cout << "LanguageModel::setCudaPreferCpuAdamOffload: disabling CUDA Graph for FP16 working weights\n";
    } else {
        CudaBao::enabled = false;
        CudaBao::request = BaoMode::Auto;
        CudaBao::resolved = BaoMode::GpuInt8Adam;
        CudaAdam::preferCpuOffload = false;
        CudaAdam::preferFp16GpuWeights = false;
        CudaAdam::preferHostGradients = false;
        CudaAdam::preferHostSgd = false;
        CudaAdam::preferInt8Moments = true;
    }
    this->device->trainStateReady = false;
    if (this->deviceTrainEnabled) {
        if (enabled) {
            const bool largeHostAdam = this->device->blocks.size() >= 24;
            this->device->setActivationCheckpointMode(
                largeHostAdam ? ActivationCheckpointMode::Full : ActivationCheckpointMode::Off);
        } else {
            this->device->setActivationCheckpointMode(ActivationCheckpointMode::Off);
        }
        this->device->ensureTrainState();
        this->device->applyVramPackBudget();
        this->device->trainStateReady = false;
        this->device->ensureTrainState();
    }
    std::cout << "LanguageModel::setCudaPreferCpuAdamOffload: " << (enabled ? "on" : "off")
              << "  bao=" << (CudaBao::enabled ? CudaBao::modeName(CudaBao::resolved) : "off")
              << "  fp16GpuWeights=" << (CudaAdam::preferFp16GpuWeights ? "on" : "off")
              << "  hostGrads=" << (CudaAdam::preferHostGradients ? "on" : "off")
              << "  amp=" << (CudaAmp::preferMixedPrecision ? "on" : "off")
              << "  int8 adam " << (CudaAdam::preferInt8Moments ? "on" : "off") << '\n';
}

void LanguageModel::setCudaPreferHostSgd(bool enabled) {
    if (enabled) {
        CudaBao::request = BaoMode::HostFusedHalfSgd;
        CudaBao::apply(BaoMode::HostFusedHalfSgd);
        CudaAmp::preferMixedPrecision = true;
        if (this->device != nullptr) {
            this->device->preferTrainGraph = false;
            this->device->releaseTrainGraph();
        }
    } else {
        CudaAdam::preferHostSgd = false;
        if (CudaAdam::preferCpuOffload) {
            CudaBao::request = BaoMode::HostFusedHalfAdam;
            CudaBao::apply(BaoMode::HostFusedHalfAdam);
        } else {
            CudaBao::enabled = false;
            CudaBao::request = BaoMode::Auto;
            CudaBao::resolved = BaoMode::GpuInt8Adam;
        }
    }
    if (this->device != nullptr) {
        this->device->trainStateReady = false;
        if (this->deviceTrainEnabled) {
            if (enabled) {
                this->device->setActivationCheckpointMode(ActivationCheckpointMode::Full);
                this->device->releasePackedTrainWorkspaces();
                this->device->tuneOffloadCheckpointAndPack(4096);
            } else {
                this->device->ensureTrainState();
                this->device->applyVramPackBudget();
                this->device->trainStateReady = false;
                this->device->ensureTrainState();
            }
        }
    }
    std::cout << "LanguageModel::setCudaPreferHostSgd: " << (enabled ? "on" : "off")
              << "  bao=" << (CudaBao::enabled ? CudaBao::modeName(CudaBao::resolved) : "off") << '\n';
}

void LanguageModel::setCudaPreferBao(bool enabled) {
    if (this->device == nullptr) this->enableCuda();
    if (this->device == nullptr) return;
    if (!enabled) {
        CudaBao::enabled = false;
        CudaBao::request = BaoMode::Auto;
        CudaBao::resolved = BaoMode::GpuInt8Adam;
        CudaAdam::preferCpuOffload = false;
        CudaAdam::preferFp16GpuWeights = false;
        CudaAdam::preferHostGradients = false;
        CudaAdam::preferHostSgd = false;
        CudaAdam::preferInt8Moments = true;
        this->device->trainStateReady = false;
        if (this->deviceTrainEnabled) {
            this->device->ensureTrainState();
            this->device->applyVramPackBudget();
            this->device->trainStateReady = false;
            this->device->ensureTrainState();
        }
        std::cout << "LanguageModel::setCudaPreferBao: off\n";
        return;
    }
    CudaBao::enabled = true;
    CudaBao::request = BaoMode::Auto;
    // Defer concrete pick until enableCudaTrain (free VRAM after upload). If train is
    // already on, resolve immediately.
    if (this->deviceTrainEnabled)
        this->setCudaBaoMode(BaoMode::Auto);
    else
        std::cout << "LanguageModel::setCudaPreferBao: on  request=Auto (resolve at enable_cuda_train)\n";
}

void LanguageModel::setCudaBaoMode(BaoMode mode) {
    if (this->device == nullptr) this->enableCuda();
    if (this->device == nullptr) return;
    CudaBao::enabled = true;
    CudaBao::request = mode;
    size_t freeBytes = 0;
    size_t totalBytes = 0;
    if (CudaMatmul::isAvailable()) {
        cudaDeviceSynchronize();
        cudaMemGetInfo(&freeBytes, &totalBytes);
    }
    const size_t parameterBytes = this->parameterElementCount() * sizeof(float);
    const BaoMode resolved = CudaBao::resolveAndApply(freeBytes, parameterBytes, 0);
    CudaBao::request = mode; // keep Auto as request when policy-selected
    if (resolved == BaoMode::HostFusedHalfAdam || resolved == BaoMode::HostFusedHalfSgd)
        CudaAmp::preferMixedPrecision = true;
    if (resolved == BaoMode::GpuInt8Adam) {
        this->device->preferTrainGraph = true;
    } else {
        this->device->preferTrainGraph = false;
        this->device->releaseTrainGraph();
        if (this->device->preferMuon) {
            this->device->preferMuon = false;
            std::cout << "LanguageModel::setCudaBaoMode: disabling Muon (host BAO path)\n";
        }
    }
    this->device->trainStateReady = false;
    if (this->deviceTrainEnabled) {
        if (resolved == BaoMode::HostFusedHalfSgd) {
            this->device->setActivationCheckpointMode(ActivationCheckpointMode::Full);
            this->device->releasePackedTrainWorkspaces();
            this->device->tuneOffloadCheckpointAndPack(4096);
        } else if (resolved == BaoMode::HostFusedHalfAdam) {
            const bool largeHostAdam = this->device->blocks.size() >= 24;
            this->device->setActivationCheckpointMode(
                largeHostAdam ? ActivationCheckpointMode::Full : ActivationCheckpointMode::Off);
            this->device->ensureTrainState();
            this->device->applyVramPackBudget();
            this->device->trainStateReady = false;
            this->device->ensureTrainState();
        } else {
            this->device->setActivationCheckpointMode(ActivationCheckpointMode::Off);
            this->device->ensureTrainState();
            this->device->applyVramPackBudget();
            this->device->trainStateReady = false;
            this->device->ensureTrainState();
        }
    }
    std::cout << "LanguageModel::setCudaBaoMode: request=" << CudaBao::modeName(mode)
              << "  resolved=" << CudaBao::modeName(resolved)
              << "  freeVram=" << (freeBytes / (1024ull * 1024ull)) << " MiB\n";
}

BaoMode LanguageModel::cudaBaoModeResolved() const {
    return CudaBao::resolved;
}

void LanguageModel::enableActivationCheckpointing(bool enabled) {
    this->setActivationCheckpointMode(
        enabled ? ActivationCheckpointMode::Selective : ActivationCheckpointMode::Off);
}

void LanguageModel::setActivationCheckpointMode(ActivationCheckpointMode mode) {
    if (this->device == nullptr) this->enableCuda();
    if (this->device == nullptr) return;
    this->device->setActivationCheckpointMode(mode);
    if (this->deviceTrainEnabled) {
        this->device->trainStateReady = false;
        this->device->ensureTrainState();
    }
    std::cout << "LanguageModel::setActivationCheckpointMode: "
              << CudaLanguageModel::activationCheckpointModeName(mode) << '\n';
}

ActivationCheckpointMode LanguageModel::cudaActivationCheckpointMode() const {
    if (this->device == nullptr) return ActivationCheckpointMode::Off;
    return this->device->activationCheckpointMode;
}

void LanguageModel::setCudaPreferMixedPrecision(bool enabled) {
    if (this->device == nullptr) this->enableCuda();
    if (this->device == nullptr) return;
    CudaAmp::preferMixedPrecision = enabled;
    CudaAmp::useLossScaling = enabled && this->tokenEmbedding.embeddingDim() >= 256;
    if (enabled)
        CudaAmp::resetLossScaler();
    if (this->deviceTrainEnabled)
        this->device->ensureTrainWorkspaces();
    std::cout << "LanguageModel::setCudaPreferMixedPrecision: " << (enabled ? "on" : "off")
              << "  lossScale=" << (CudaAmp::useLossScaling ? "on" : "off")
              << "  scale=" << CudaAmp::lossScaler.scale << '\n';
}

void LanguageModel::setCudaMaxPackedColumns(int columns) {
    if (this->device == nullptr) this->enableCuda();
    if (this->device == nullptr) return;
    if (columns <= 0) throw std::invalid_argument("LanguageModel::setCudaMaxPackedColumns columns must be > 0");
    this->device->maxPackedColumns = columns;
    this->device->maxPackedColumnsManual = true;
    if (this->deviceTrainEnabled)
        this->device->ensureTrainWorkspaces();
}

void LanguageModel::setCudaLogitChunkRows(int rows) {
    if (this->device == nullptr) this->enableCuda();
    if (this->device == nullptr) return;
    if (rows <= 0) throw std::invalid_argument("LanguageModel::setCudaLogitChunkRows rows must be > 0");
    this->device->logitChunkRows = rows;
    if (this->deviceTrainEnabled)
        this->device->ensureTrainWorkspaces();
    std::cout << "LanguageModel::setCudaLogitChunkRows: " << rows << '\n';
}

void LanguageModel::setCudaPreferInt8AdamMoments(bool enabled) {
    if (this->device == nullptr) this->enableCuda();
    if (this->device == nullptr) return;
    CudaAdam::preferInt8Moments = enabled;
    if (enabled)
        CudaAdam::preferCpuOffload = false;
    if (this->device != nullptr) {
        this->device->trainStateReady = false;
        if (this->deviceTrainEnabled)
            this->device->ensureTrainState();
    }
    std::cout << "LanguageModel::setCudaPreferInt8AdamMoments: " << (enabled ? "on" : "off")
              << "  blockSize=" << CudaAdam::int8BlockSize
              << "  cpuAdam=" << (CudaAdam::preferCpuOffload ? "on" : "off") << '\n';
}

void LanguageModel::setCudaPreferMuon(bool enabled) {
    if (this->device == nullptr) this->enableCuda();
    if (this->device == nullptr) return;
    this->device->setPreferMuon(enabled);
    if (this->deviceTrainEnabled) {
        this->device->trainStateReady = false;
        this->device->ensureTrainState();
    }
    std::cout << "LanguageModel::setCudaPreferMuon: " << (enabled ? "on" : "off") << '\n';
}

void LanguageModel::setCudaMuonNsSteps(int steps) {
    if (steps <= 0)
        throw std::invalid_argument("LanguageModel::setCudaMuonNsSteps steps must be > 0");
    if (this->device == nullptr) this->enableCuda();
    if (this->device == nullptr) return;
    this->device->muon.nsSteps = steps;
    std::cout << "LanguageModel::setCudaMuonNsSteps: " << steps << '\n';
}

void LanguageModel::setCudaPreferFlashAttention(bool enabled) {
    if (this->device == nullptr) this->enableCuda();
    if (this->device == nullptr) return;
    for (CudaTransformerBlock& block : this->device->blocks) {
        block.attention.preferFlashAttention = enabled;
        if (enabled)
            block.attention.releaseDenseAttentionScratch();
        else
            block.attention.releaseFlashAttentionScratch();
    }
}

bool LanguageModel::cudaEnabled() const {
    return this->device != nullptr;
}

bool LanguageModel::cudaTrainEnabled() const {
    return this->device != nullptr && this->deviceTrainEnabled;
}

int LanguageModel::cudaMaxPackedColumns() const {
    if (this->device == nullptr) return 0;
    return this->device->maxPackedColumns;
}

void LanguageModel::syncDevice() {
    if (this->device == nullptr) return;
    this->device->uploadFrom(*this);
    this->deviceStale = false;
}

void LanguageModel::syncDeviceIfStale() {
    if (this->device == nullptr) return;
    if (!this->deviceStale) return;
    this->syncDevice();
}

Matrix LanguageModel::sumColumns(const Matrix& gradient) {
    Matrix biasGradient(gradient.rows, 1, 0.0f);
    for (size_t row = 0; row < gradient.rows; ++row) {
        float total = 0.0f;
        for (size_t column = 0; column < gradient.cols; ++column)
            total += gradient.at(row, column);
        biasGradient.at(row, 0) = total;
    }
    return biasGradient;
}

Matrix LanguageModel::broadcastBiasAdd(const Matrix& product, const Matrix& bias) {
    Matrix result = product;
    for (size_t row = 0; row < result.rows; ++row) {
        const float biasValue = bias.at(row, 0);
        for (size_t column = 0; column < result.cols; ++column)
            result.at(row, column) += biasValue;
    }
    return result;
}

Matrix LanguageModel::forwardLocal(const std::vector<int>& tokenIds, LanguageModelCache& cache) const {
    if (tokenIds.empty()) throw std::invalid_argument("LanguageModel::forwardLocal empty tokenIds");
    if (static_cast<int>(tokenIds.size()) > this->maximumPositionCount)
        throw std::invalid_argument("LanguageModel::forwardLocal sequence longer than maximumPositionCount");

    if (cache.blockCaches.size() != this->blocks.size())
        cache.blockCaches.resize(this->blocks.size());

    Matrix hidden = this->tokenEmbedding.forward(tokenIds);

    for (size_t blockIndex = 0; blockIndex < this->blocks.size(); ++blockIndex)
        hidden = this->blocks[blockIndex].forward(hidden, cache.blockCaches[blockIndex]);

    cache.blockOutput = this->finalNorm.forward(hidden, cache.finalNormCache);
    Matrix product = Matrix::multiply(this->lmHeadWeight(), cache.blockOutput);
    return LanguageModel::broadcastBiasAdd(product, this->outputProjection.bias);
}

Matrix LanguageModel::forward(const std::vector<int>& tokenIds) {
    if (this->device != nullptr) {
        this->syncDeviceIfStale();
        return this->device->forward(tokenIds);
    }

    LanguageModelCache cache;
    return this->forwardLocal(tokenIds, cache);
}

float LanguageModel::exampleLoss(const LanguageModelExample& example) {
    Matrix logits = this->forward(example.inputTokenIds);
    Matrix probabilities;
    Softmax::applyInto(logits, probabilities);
    Matrix target = example.targetOneHot.empty()
        ? LanguageModelDataset::makeOneHotSequence(example.targetTokenIds, this->tokenEmbedding.vocabSize())
        : example.targetOneHot;
    return CrossEntropy::loss(probabilities, target);
}

float LanguageModel::averageLoss(const LanguageModelDataset& dataset) {
    if (dataset.examples.empty()) return 0.0f;

    if (this->device != nullptr) {
        this->syncDeviceIfStale();
        return this->device->averageLoss(dataset);
    }

    float total = 0.0f;
    const int exampleCount = static_cast<int>(dataset.examples.size());

#if defined(_OPENMP)
    #pragma omp parallel for reduction(+:total) schedule(dynamic, 4)
#endif
    for (int index = 0; index < exampleCount; ++index)
        total += this->exampleLoss(dataset.examples[static_cast<size_t>(index)]);

    return total / static_cast<float>(dataset.size());
}

float LanguageModel::accumulateExample(const LanguageModelExample& example, LanguageModelGradients& gradients, LanguageModelCache& cache) const {
    Matrix logits = this->forwardLocal(example.inputTokenIds, cache);
    Softmax::applyInto(logits, cache.probabilities);
    Matrix target = example.targetOneHot.empty()
        ? LanguageModelDataset::makeOneHotSequence(example.targetTokenIds, this->tokenEmbedding.vocabSize())
        : example.targetOneHot;
    const float loss = CrossEntropy::loss(cache.probabilities, target);

    Matrix logitGradient = CrossEntropy::gradient(cache.probabilities, target);
    Matrix projectionWeightGradient = Matrix::multiply(logitGradient, cache.blockOutput, false, true);
    Matrix projectionBiasGradient = LanguageModel::sumColumns(logitGradient);
    Matrix hiddenGradient = Matrix::multiply(this->lmHeadWeight(), logitGradient, true, false);

    Matrix finalNormGammaGradient;
    hiddenGradient = this->finalNorm.backward(hiddenGradient, cache.finalNormCache, finalNormGammaGradient);

    for (int blockIndex = static_cast<int>(this->blocks.size()) - 1; blockIndex >= 0; --blockIndex)
        hiddenGradient = this->blocks[static_cast<size_t>(blockIndex)].backward(hiddenGradient, cache.blockCaches[static_cast<size_t>(blockIndex)], gradients.blocks[static_cast<size_t>(blockIndex)]);

    Matrix tokenEmbeddingGradient = this->tokenEmbedding.backward(hiddenGradient, example.inputTokenIds);

    if (this->tieEmbeddingProjection)
        Matrix::addInPlace(gradients.tokenEmbedding, projectionWeightGradient);
    else
        Matrix::addInPlace(gradients.projectionWeight, projectionWeightGradient);
    Matrix::addInPlace(gradients.projectionBias, projectionBiasGradient);
    Matrix::addInPlace(gradients.finalNormGamma, finalNormGammaGradient);
    Matrix::addInPlace(gradients.tokenEmbedding, tokenEmbeddingGradient);

    return loss;
}

void LanguageModel::applyGradients(const LanguageModelGradients& gradients) {
    this->optimizer.step();
    if (!this->tieEmbeddingProjection)
        this->optimizer.update(this->outputProjection.weight, this->projectionWeightState, gradients.projectionWeight);
    this->optimizer.update(this->outputProjection.bias, this->projectionBiasState, gradients.projectionBias);
    this->optimizer.update(this->finalNorm.gamma, this->finalNormGammaState, gradients.finalNormGamma);
    for (size_t blockIndex = 0; blockIndex < this->blocks.size(); ++blockIndex)
        this->blocks[blockIndex].applyGradients(this->optimizer, gradients.blocks[blockIndex]);
    this->optimizer.update(this->tokenEmbedding.weight, this->tokenEmbeddingState, gradients.tokenEmbedding);
    if (this->device != nullptr)
        this->deviceStale = true;
}

void LanguageModel::train(const LanguageModelDataset& dataset, int epochs, int logEveryEpochs) {
    LanguageModelDataset emptyTest;
    this->train(dataset, emptyTest, epochs, logEveryEpochs, 32);
}

float LanguageModel::trainOnExamples(
    const LanguageModelDataset& dataset,
    int batchSize,
    std::vector<LanguageModelGradients>& threadGradients,
    std::vector<LanguageModelCache>& threadCaches,
    LanguageModelGradients& merged
) {
    const int exampleCount = static_cast<int>(dataset.examples.size());
    if (exampleCount <= 0) return 0.0f;
    if (batchSize <= 0) batchSize = 32;

    const int threadCount = static_cast<int>(threadGradients.size());
    float epochLoss = 0.0f;

    for (int batchStart = 0; batchStart < exampleCount; batchStart += batchSize) {
        int batchEnd = batchStart + batchSize;
        if (batchEnd > exampleCount) batchEnd = exampleCount;
        const float batchCount = static_cast<float>(batchEnd - batchStart);

        for (int threadIndex = 0; threadIndex < threadCount; ++threadIndex)
            threadGradients[static_cast<size_t>(threadIndex)].zeroInPlace();

        float batchLoss = 0.0f;

#if defined(_OPENMP)
        #pragma omp parallel
#endif
        {
            int threadIndex = 0;
#if defined(_OPENMP)
            threadIndex = omp_get_thread_num();
            #pragma omp for reduction(+:batchLoss) schedule(dynamic, 1)
#endif
            for (int index = batchStart; index < batchEnd; ++index) {
                batchLoss += this->accumulateExample(
                    dataset.examples[static_cast<size_t>(index)],
                    threadGradients[static_cast<size_t>(threadIndex)],
                    threadCaches[static_cast<size_t>(threadIndex)]
                );
            }
        }

        merged.zeroInPlace();
        for (int threadIndex = 0; threadIndex < threadCount; ++threadIndex)
            merged.addInPlace(threadGradients[static_cast<size_t>(threadIndex)]);
        merged.scaleInPlace(1.0f / batchCount);
        this->applyGradients(merged);

        epochLoss += batchLoss;
    }

    return epochLoss;
}

void LanguageModel::train(const LanguageModelDataset& trainDataset, const LanguageModelDataset& testDataset, int epochs, int logEveryEpochs, int batchSize, int gradientAccumulationSteps) {
    if (trainDataset.examples.empty()) return;
    if (logEveryEpochs <= 0) logEveryEpochs = 1;
    if (batchSize <= 0) batchSize = 32;
    if (gradientAccumulationSteps <= 0) gradientAccumulationSteps = 1;

    if (this->cudaTrainEnabled()) {
        this->syncDeviceIfStale();
        this->device->adam = CudaAdam(this->optimizer.learningRate, this->optimizer.beta1, this->optimizer.beta2, this->optimizer.epsilon);
        this->device->adam.timeStep = this->optimizer.timeStep;
        try {
            this->device->train(trainDataset, testDataset, epochs, logEveryEpochs, batchSize, gradientAccumulationSteps);
            this->device->downloadTo(*this);
        } catch (const std::exception& ex) {
            std::cerr << "LanguageModel::train cuda path failed: " << ex.what() << std::endl;
            throw std::runtime_error(std::string("LanguageModel::train cuda path failed: ") + ex.what());
        }
        this->optimizer.timeStep = this->device->adam.timeStep;
        this->deviceStale = false;
        return;
    }

    int threadCount = 1;
#if defined(_OPENMP)
    threadCount = omp_get_max_threads();
    if (threadCount < 1) threadCount = 1;
#endif

    std::vector<LanguageModelGradients> threadGradients(static_cast<size_t>(threadCount));
    std::vector<LanguageModelCache> threadCaches(static_cast<size_t>(threadCount));
    for (int threadIndex = 0; threadIndex < threadCount; ++threadIndex)
        threadGradients[static_cast<size_t>(threadIndex)] = LanguageModelGradients::zerosFrom(*this);

    LanguageModelGradients merged = LanguageModelGradients::zerosFrom(*this);

    for (int epoch = 0; epoch < epochs; ++epoch) {
        const auto epochStart = std::chrono::steady_clock::now();
        const float epochLoss = this->trainOnExamples(trainDataset, batchSize, threadGradients, threadCaches, merged);

        if (epoch % logEveryEpochs != 0) continue;

        const float averageTrainLoss = epochLoss / static_cast<float>(trainDataset.size());
        const auto epochEnd = std::chrono::steady_clock::now();
        const double epochSeconds = std::chrono::duration<double>(epochEnd - epochStart).count();
        const double tokensPerSecond = epochSeconds > 0.0
            ? static_cast<double>(trainDataset.totalPredictionCount()) / epochSeconds
            : 0.0;
        std::cout << "  Epoch " << epoch << "  trainLoss=" << averageTrainLoss
                  << "  sec=" << epochSeconds << "  tokens/s=" << tokensPerSecond
                  << "  backend=cpu-openmp";

        if (!testDataset.examples.empty()) {
            const float testLoss = this->averageLoss(testDataset);
            std::cout << "  testLoss=" << testLoss;
        }

        std::cout << '\n';
    }
}

void LanguageModel::train(LanguageModelChunkSource& source, int epochs, int logEveryEpochs, int batchSize, int gradientAccumulationSteps) {
    if (logEveryEpochs <= 0) logEveryEpochs = 1;
    if (batchSize <= 0) batchSize = 32;
    if (gradientAccumulationSteps <= 0) gradientAccumulationSteps = 1;

    if (this->cudaTrainEnabled()) {
        this->syncDeviceIfStale();
        this->device->adam = CudaAdam(this->optimizer.learningRate, this->optimizer.beta1, this->optimizer.beta2, this->optimizer.epsilon);
        this->device->adam.timeStep = this->optimizer.timeStep;
        try {
            this->device->train(source, epochs, logEveryEpochs, batchSize, gradientAccumulationSteps);
            this->device->downloadTo(*this);
        } catch (const std::exception& ex) {
            std::cerr << "LanguageModel::train cuda path failed: " << ex.what() << std::endl;
            throw std::runtime_error(std::string("LanguageModel::train cuda path failed: ") + ex.what());
        }
        this->optimizer.timeStep = this->device->adam.timeStep;
        this->deviceStale = false;
        return;
    }

    int threadCount = 1;
#if defined(_OPENMP)
    threadCount = omp_get_max_threads();
    if (threadCount < 1) threadCount = 1;
#endif

    std::vector<LanguageModelGradients> threadGradients(static_cast<size_t>(threadCount));
    std::vector<LanguageModelCache> threadCaches(static_cast<size_t>(threadCount));
    for (int threadIndex = 0; threadIndex < threadCount; ++threadIndex)
        threadGradients[static_cast<size_t>(threadIndex)] = LanguageModelGradients::zerosFrom(*this);

    LanguageModelGradients merged = LanguageModelGradients::zerosFrom(*this);
    const LanguageModelDataset& testDataset = source.testDataset();

    for (int epoch = 0; epoch < epochs; ++epoch) {
        const auto epochStart = std::chrono::steady_clock::now();
        float epochLoss = 0.0f;
        int processedExampleCount = 0;
        int processedPredictionCount = 0;

        source.rewindTrain();
        LanguageModelDataset epochDataset;
        source.fillTrainDataset(epochDataset);
        if (!epochDataset.examples.empty()) {
            epochLoss += this->trainOnExamples(epochDataset, batchSize, threadGradients, threadCaches, merged);
            processedExampleCount += epochDataset.size();
            processedPredictionCount += epochDataset.totalPredictionCount();
        }

        if (epoch % logEveryEpochs != 0) continue;

        const float averageTrainLoss = processedExampleCount > 0
            ? epochLoss / static_cast<float>(processedExampleCount)
            : 0.0f;
        const auto epochEnd = std::chrono::steady_clock::now();
        const double epochSeconds = std::chrono::duration<double>(epochEnd - epochStart).count();
        const double tokensPerSecond = epochSeconds > 0.0
            ? static_cast<double>(processedPredictionCount) / epochSeconds
            : 0.0;
        std::cout << "  Epoch " << epoch << "  trainLoss=" << averageTrainLoss
                  << "  sec=" << epochSeconds << "  tokens/s=" << tokensPerSecond
                  << "  backend=cpu-stream";

        if (!testDataset.examples.empty()) {
            const float testLoss = this->averageLoss(testDataset);
            std::cout << "  testLoss=" << testLoss;
        }

        std::cout << '\n';
    }
}

int LanguageModel::argmaxLastColumn(const Matrix& logits) {
    const size_t lastColumn = logits.cols - 1;
    int bestTokenId = 0;
    float bestLogit = logits.at(0, lastColumn);
    for (size_t tokenId = 1; tokenId < logits.rows; ++tokenId) {
        if (logits.at(tokenId, lastColumn) <= bestLogit) continue;
        bestLogit = logits.at(tokenId, lastColumn);
        bestTokenId = static_cast<int>(tokenId);
    }
    return bestTokenId;
}

int LanguageModel::sampleLastColumn(const Matrix& logits, float temperature, int topK, unsigned& seed) {
    if (temperature <= 0.0f) return LanguageModel::argmaxLastColumn(logits);

    const size_t vocabularySize = logits.rows;
    const size_t lastColumn = logits.cols - 1;

    Matrix scaledLogits(vocabularySize, 1, 0.0f);
    for (size_t tokenId = 0; tokenId < vocabularySize; ++tokenId)
        scaledLogits.at(tokenId, 0) = logits.at(tokenId, lastColumn) / temperature;

    Matrix probabilities = Softmax::apply(scaledLogits);

    std::vector<int> candidateTokenIds;
    candidateTokenIds.reserve(vocabularySize);
    for (size_t tokenId = 0; tokenId < vocabularySize; ++tokenId)
        candidateTokenIds.push_back(static_cast<int>(tokenId));

    if (topK > 0 && static_cast<size_t>(topK) < vocabularySize) {
        const size_t keepCount = static_cast<size_t>(topK);
        for (size_t rank = 0; rank < keepCount; ++rank) {
            size_t bestIndex = rank;
            for (size_t index = rank + 1; index < candidateTokenIds.size(); ++index) {
                if (probabilities.at(static_cast<size_t>(candidateTokenIds[index]), 0)
                    <= probabilities.at(static_cast<size_t>(candidateTokenIds[bestIndex]), 0))
                    continue;
                bestIndex = index;
            }
            if (bestIndex != rank)
                std::swap(candidateTokenIds[rank], candidateTokenIds[bestIndex]);
        }
        candidateTokenIds.resize(keepCount);

        float probabilitySum = 0.0f;
        for (int tokenId : candidateTokenIds)
            probabilitySum += probabilities.at(static_cast<size_t>(tokenId), 0);
        if (probabilitySum <= 0.0f) return candidateTokenIds[0];
        for (int tokenId : candidateTokenIds)
            probabilities.at(static_cast<size_t>(tokenId), 0) /= probabilitySum;
    }

    const float unit = UniformInit::unitSample(seed);
    float cumulative = 0.0f;
    for (int tokenId : candidateTokenIds) {
        cumulative += probabilities.at(static_cast<size_t>(tokenId), 0);
        if (unit <= cumulative) return tokenId;
    }
    return candidateTokenIds.back();
}

std::vector<int> LanguageModel::generate(const std::vector<int>& promptTokenIds, int newTokenCount, float temperature, int topK, unsigned seed) {
    if (promptTokenIds.empty()) throw std::invalid_argument("LanguageModel::generate empty prompt");
    if (newTokenCount < 0) throw std::invalid_argument("LanguageModel::generate newTokenCount must be >= 0");

    std::vector<int> tokenIds = promptTokenIds;
    for (int step = 0; step < newTokenCount; ++step) {
        if (static_cast<int>(tokenIds.size()) >= this->maximumPositionCount) break;

        Matrix logits = this->forward(tokenIds);
        const int nextTokenId = LanguageModel::sampleLastColumn(logits, temperature, topK, seed);
        tokenIds.push_back(nextTokenId);
    }

    return tokenIds;
}

namespace {
constexpr char kCheckpointMagic[4] = { 'S', 'N', 'L', 'M' };
constexpr std::int32_t kCheckpointVersion = 3;
constexpr std::int32_t kOptimizerKindAdam = 0;
constexpr std::int32_t kOptimizerKindMuonAdam = 1;

void writePod(std::ostream& out, const void* data, size_t byteCount) {
    out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(byteCount));
    if (!out) throw std::runtime_error("LanguageModel checkpoint write failed");
}

void readPod(std::istream& in, void* data, size_t byteCount) {
    in.read(reinterpret_cast<char*>(data), static_cast<std::streamsize>(byteCount));
    if (!in) throw std::runtime_error("LanguageModel checkpoint read failed");
}

void writeI32(std::ostream& out, std::int32_t value) { writePod(out, &value, sizeof(value)); }
void writeF32(std::ostream& out, float value) { writePod(out, &value, sizeof(value)); }
std::int32_t readI32(std::istream& in) {
    std::int32_t value = 0;
    readPod(in, &value, sizeof(value));
    return value;
}
float readF32(std::istream& in) {
    float value = 0.0f;
    readPod(in, &value, sizeof(value));
    return value;
}

void writeMatrix(std::ostream& out, const Matrix& matrix) {
    writeI32(out, static_cast<std::int32_t>(matrix.rows));
    writeI32(out, static_cast<std::int32_t>(matrix.cols));
    if (matrix.data.empty()) return;
    writePod(out, matrix.data.data(), matrix.data.size() * sizeof(float));
}

Matrix readMatrix(std::istream& in) {
    const std::int32_t rows = readI32(in);
    const std::int32_t cols = readI32(in);
    if (rows < 0 || cols < 0) throw std::runtime_error("LanguageModel checkpoint invalid matrix shape");
    Matrix matrix(static_cast<size_t>(rows), static_cast<size_t>(cols), 0.0f);
    if (matrix.data.empty()) return matrix;
    readPod(in, matrix.data.data(), matrix.data.size() * sizeof(float));
    return matrix;
}

void expectMatrixShape(const Matrix& matrix, size_t rows, size_t cols, const char* name) {
    if (matrix.rows != rows || matrix.cols != cols)
        throw std::runtime_error(std::string("LanguageModel checkpoint shape mismatch: ") + name);
}

void writeAdamState(std::ostream& out, const AdamState& state) {
    writeMatrix(out, state.firstMoment);
    writeMatrix(out, state.secondMoment);
}

AdamState readAdamState(std::istream& in) {
    AdamState state;
    state.firstMoment = readMatrix(in);
    state.secondMoment = readMatrix(in);
    return state;
}

void writeMuonState(std::ostream& out, const MuonState& state) {
    writeMatrix(out, state.momentum);
}

MuonState readMuonState(std::istream& in) {
    MuonState state;
    state.momentum = readMatrix(in);
    return state;
}
}

void LanguageModel::saveCheckpoint(const std::string& path, bool includeOptimizer) {
    if (path.empty()) throw std::invalid_argument("LanguageModel::saveCheckpoint empty path");
    if (this->blocks.empty()) throw std::logic_error("LanguageModel::saveCheckpoint no blocks");

    if (this->cudaTrainEnabled() && this->device != nullptr) {
        this->device->downloadTo(*this);
        if (includeOptimizer)
            this->device->downloadOptimizerTo(*this);
        this->deviceStale = false;
    } else if (this->cudaEnabled() && this->device != nullptr && !this->deviceStale) {
        this->device->downloadTo(*this);
    }

    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("LanguageModel::saveCheckpoint cannot open file");

    writePod(out, kCheckpointMagic, 4);
    writeI32(out, kCheckpointVersion);
    writeI32(out, this->tokenEmbedding.vocabSize());
    writeI32(out, this->tokenEmbedding.embeddingDim());
    writeI32(out, this->maximumPositionCount);
    writeI32(out, static_cast<std::int32_t>(this->blocks.size()));
    writeI32(out, this->blocks[0].attention.headCount);
    writeF32(out, this->optimizer.learningRate);
    writeF32(out, this->optimizer.beta1);
    writeF32(out, this->optimizer.beta2);
    writeF32(out, this->optimizer.epsilon);
    writeI32(out, this->optimizer.timeStep);
    writeI32(out, includeOptimizer ? 1 : 0);
    writeI32(out, this->tieEmbeddingProjection ? 1 : 0);
    const bool useMuonOptimizer = includeOptimizer
        && this->cudaTrainEnabled()
        && this->device != nullptr
        && this->device->preferMuon;
    writeI32(out, useMuonOptimizer ? kOptimizerKindMuonAdam : kOptimizerKindAdam);

    writeMatrix(out, this->tokenEmbedding.weight);
    for (const TransformerBlock& block : this->blocks) {
        writeMatrix(out, block.attention.queryWeight);
        writeMatrix(out, block.attention.keyWeight);
        writeMatrix(out, block.attention.valueWeight);
        writeMatrix(out, block.attention.outputWeight);
        writeMatrix(out, block.attentionNorm.gamma);
        writeMatrix(out, block.feedForwardNorm.gamma);
        writeMatrix(out, block.feedForward.gateWeight);
        writeMatrix(out, block.feedForward.gateBias);
        writeMatrix(out, block.feedForward.upWeight);
        writeMatrix(out, block.feedForward.upBias);
        writeMatrix(out, block.feedForward.downWeight);
        writeMatrix(out, block.feedForward.downBias);
    }
    writeMatrix(out, this->finalNorm.gamma);
    if (!this->tieEmbeddingProjection)
        writeMatrix(out, this->outputProjection.weight);
    writeMatrix(out, this->outputProjection.bias);

    if (includeOptimizer) {
        writeAdamState(out, this->tokenEmbeddingState);
        for (const TransformerBlock& block : this->blocks) {
            if (useMuonOptimizer) {
                writeMuonState(out, block.queryWeightMuon);
                writeMuonState(out, block.keyWeightMuon);
                writeMuonState(out, block.valueWeightMuon);
                writeMuonState(out, block.attentionOutputWeightMuon);
                writeMuonState(out, block.feedForwardGateWeightMuon);
                writeMuonState(out, block.feedForwardUpWeightMuon);
                writeMuonState(out, block.feedForwardDownWeightMuon);
            } else {
                writeAdamState(out, block.queryWeightState);
                writeAdamState(out, block.keyWeightState);
                writeAdamState(out, block.valueWeightState);
                writeAdamState(out, block.attentionOutputWeightState);
                writeAdamState(out, block.feedForwardGateWeightState);
                writeAdamState(out, block.feedForwardUpWeightState);
                writeAdamState(out, block.feedForwardDownWeightState);
            }
            writeAdamState(out, block.attentionNormGammaState);
            writeAdamState(out, block.feedForwardNormGammaState);
            writeAdamState(out, block.feedForwardGateBiasState);
            writeAdamState(out, block.feedForwardUpBiasState);
            writeAdamState(out, block.feedForwardDownBiasState);
        }
        writeAdamState(out, this->finalNormGammaState);
        if (!this->tieEmbeddingProjection)
            writeAdamState(out, this->projectionWeightState);
        writeAdamState(out, this->projectionBiasState);
    }
}

void LanguageModel::loadCheckpoint(const std::string& path) {
    if (path.empty()) throw std::invalid_argument("LanguageModel::loadCheckpoint empty path");
    if (this->blocks.empty()) throw std::logic_error("LanguageModel::loadCheckpoint no blocks");

    // Extension / magic dispatch: .safetensors or safetensors header → SafeTensors path.
    const auto endsWith = [](const std::string& value, const char* suffix) -> bool {
        const size_t n = std::char_traits<char>::length(suffix);
        return value.size() >= n && value.compare(value.size() - n, n, suffix) == 0;
    };
    if (endsWith(path, ".safetensors") || endsWith(path, ".safe") || SafeTensors::isSafeTensorsFile(path)) {
        this->loadSafeTensors(path);
        return;
    }

    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("LanguageModel::loadCheckpoint cannot open file");

    char magic[4] = {};
    readPod(in, magic, 4);
    if (magic[0] != kCheckpointMagic[0] || magic[1] != kCheckpointMagic[1]
        || magic[2] != kCheckpointMagic[2] || magic[3] != kCheckpointMagic[3])
        throw std::runtime_error("LanguageModel::loadCheckpoint bad magic");

    const std::int32_t version = readI32(in);
    if (version != 1 && version != 2 && version != 3)
        throw std::runtime_error("LanguageModel::loadCheckpoint unsupported version");

    const std::int32_t vocabularySize = readI32(in);
    const std::int32_t embeddingDim = readI32(in);
    const std::int32_t maximumPositionCount = readI32(in);
    const std::int32_t blockCount = readI32(in);
    const std::int32_t headCount = readI32(in);
    if (vocabularySize != this->tokenEmbedding.vocabSize()
        || embeddingDim != this->tokenEmbedding.embeddingDim()
        || maximumPositionCount != this->maximumPositionCount
        || blockCount != static_cast<std::int32_t>(this->blocks.size())
        || headCount != this->blocks[0].attention.headCount)
        throw std::runtime_error("LanguageModel::loadCheckpoint architecture mismatch");

    this->optimizer.learningRate = readF32(in);
    this->optimizer.beta1 = readF32(in);
    this->optimizer.beta2 = readF32(in);
    this->optimizer.epsilon = readF32(in);
    this->optimizer.timeStep = readI32(in);
    const bool includeOptimizer = readI32(in) != 0;
    const bool tieWeights = version >= 2 ? (readI32(in) != 0) : false;
    const std::int32_t optimizerKind = version >= 3 ? readI32(in) : kOptimizerKindAdam;
    if (optimizerKind != kOptimizerKindAdam && optimizerKind != kOptimizerKindMuonAdam)
        throw std::runtime_error("LanguageModel::loadCheckpoint unknown optimizer kind");
    const bool useMuonOptimizer = optimizerKind == kOptimizerKindMuonAdam;

    this->tokenEmbedding.weight = readMatrix(in);
    expectMatrixShape(this->tokenEmbedding.weight, static_cast<size_t>(vocabularySize), static_cast<size_t>(embeddingDim), "tokenEmbedding");
    for (TransformerBlock& block : this->blocks) {
        block.attention.queryWeight = readMatrix(in);
        block.attention.keyWeight = readMatrix(in);
        block.attention.valueWeight = readMatrix(in);
        block.attention.outputWeight = readMatrix(in);
        block.attentionNorm.gamma = readMatrix(in);
        block.feedForwardNorm.gamma = readMatrix(in);
        block.feedForward.gateWeight = readMatrix(in);
        block.feedForward.gateBias = readMatrix(in);
        block.feedForward.upWeight = readMatrix(in);
        block.feedForward.upBias = readMatrix(in);
        block.feedForward.downWeight = readMatrix(in);
        block.feedForward.downBias = readMatrix(in);
    }
    this->finalNorm.gamma = readMatrix(in);
    this->tieEmbeddingProjection = tieWeights;
    if (tieWeights) {
        this->outputProjection.weight = Matrix();
        this->projectionWeightState = AdamState{};
    } else {
        this->outputProjection.weight = readMatrix(in);
        expectMatrixShape(this->outputProjection.weight, static_cast<size_t>(vocabularySize), static_cast<size_t>(embeddingDim), "projectionWeight");
    }
    this->outputProjection.bias = readMatrix(in);
    expectMatrixShape(this->outputProjection.bias, static_cast<size_t>(vocabularySize), 1, "projectionBias");

    if (includeOptimizer) {
        this->tokenEmbeddingState = readAdamState(in);
        for (TransformerBlock& block : this->blocks) {
            if (version >= 3) {
                if (useMuonOptimizer) {
                    block.queryWeightMuon = readMuonState(in);
                    block.keyWeightMuon = readMuonState(in);
                    block.valueWeightMuon = readMuonState(in);
                    block.attentionOutputWeightMuon = readMuonState(in);
                    block.feedForwardGateWeightMuon = readMuonState(in);
                    block.feedForwardUpWeightMuon = readMuonState(in);
                    block.feedForwardDownWeightMuon = readMuonState(in);
                    block.queryWeightState = AdamState{};
                    block.keyWeightState = AdamState{};
                    block.valueWeightState = AdamState{};
                    block.attentionOutputWeightState = AdamState{};
                    block.feedForwardGateWeightState = AdamState{};
                    block.feedForwardUpWeightState = AdamState{};
                    block.feedForwardDownWeightState = AdamState{};
                } else {
                    block.queryWeightState = readAdamState(in);
                    block.keyWeightState = readAdamState(in);
                    block.valueWeightState = readAdamState(in);
                    block.attentionOutputWeightState = readAdamState(in);
                    block.feedForwardGateWeightState = readAdamState(in);
                    block.feedForwardUpWeightState = readAdamState(in);
                    block.feedForwardDownWeightState = readAdamState(in);
                    block.queryWeightMuon = MuonState{};
                    block.keyWeightMuon = MuonState{};
                    block.valueWeightMuon = MuonState{};
                    block.attentionOutputWeightMuon = MuonState{};
                    block.feedForwardGateWeightMuon = MuonState{};
                    block.feedForwardUpWeightMuon = MuonState{};
                    block.feedForwardDownWeightMuon = MuonState{};
                }
                block.attentionNormGammaState = readAdamState(in);
                block.feedForwardNormGammaState = readAdamState(in);
                block.feedForwardGateBiasState = readAdamState(in);
                block.feedForwardUpBiasState = readAdamState(in);
                block.feedForwardDownBiasState = readAdamState(in);
            } else {
                // v1/v2 layout: Q K V O attnNorm ffnNorm gateW gateB upW upB downW downB
                block.queryWeightState = readAdamState(in);
                block.keyWeightState = readAdamState(in);
                block.valueWeightState = readAdamState(in);
                block.attentionOutputWeightState = readAdamState(in);
                block.attentionNormGammaState = readAdamState(in);
                block.feedForwardNormGammaState = readAdamState(in);
                block.feedForwardGateWeightState = readAdamState(in);
                block.feedForwardGateBiasState = readAdamState(in);
                block.feedForwardUpWeightState = readAdamState(in);
                block.feedForwardUpBiasState = readAdamState(in);
                block.feedForwardDownWeightState = readAdamState(in);
                block.feedForwardDownBiasState = readAdamState(in);
                block.queryWeightMuon = MuonState{};
                block.keyWeightMuon = MuonState{};
                block.valueWeightMuon = MuonState{};
                block.attentionOutputWeightMuon = MuonState{};
                block.feedForwardGateWeightMuon = MuonState{};
                block.feedForwardUpWeightMuon = MuonState{};
                block.feedForwardDownWeightMuon = MuonState{};
            }
        }
        this->finalNormGammaState = readAdamState(in);
        if (!tieWeights)
            this->projectionWeightState = readAdamState(in);
        this->projectionBiasState = readAdamState(in);
    }

    if (this->device != nullptr) {
        this->device->uploadFrom(*this);
        if (this->deviceTrainEnabled && includeOptimizer) {
            this->device->setPreferMuon(useMuonOptimizer);
            this->device->trainStateReady = false;
            this->device->uploadOptimizerFrom(*this);
        }
        this->deviceStale = false;
    }
}

void LanguageModel::saveSafeTensors(const std::string& path) {
    if (path.empty()) throw std::invalid_argument("LanguageModel::saveSafeTensors empty path");
    if (this->blocks.empty()) throw std::logic_error("LanguageModel::saveSafeTensors no blocks");

    if (this->cudaEnabled() && this->device != nullptr && !this->deviceStale)
        this->device->downloadTo(*this);
    else if (this->cudaTrainEnabled() && this->device != nullptr)
        this->device->downloadTo(*this);

    SafeTensors::File file;
    file.metadata["format"] = "sentinel";
    file.metadata["arch"] = "causal_lm_rope_swiglu";
    file.metadata["vocab_size"] = std::to_string(this->tokenEmbedding.vocabSize());
    file.metadata["embedding_dim"] = std::to_string(this->tokenEmbedding.embeddingDim());
    file.metadata["max_position"] = std::to_string(this->maximumPositionCount);
    file.metadata["block_count"] = std::to_string(this->blocks.size());
    file.metadata["head_count"] = std::to_string(this->blocks[0].attention.headCount);
    file.metadata["tie_embedding"] = this->tieEmbeddingProjection ? "1" : "0";

    SafeTensors::putMatrix(file, "token_embedding.weight", this->tokenEmbedding.weight);
    for (size_t blockIndex = 0; blockIndex < this->blocks.size(); ++blockIndex) {
        const TransformerBlock& block = this->blocks[blockIndex];
        const std::string prefix = "blocks." + std::to_string(blockIndex) + ".";
        SafeTensors::putMatrix(file, prefix + "attn.q_proj.weight", block.attention.queryWeight);
        SafeTensors::putMatrix(file, prefix + "attn.k_proj.weight", block.attention.keyWeight);
        SafeTensors::putMatrix(file, prefix + "attn.v_proj.weight", block.attention.valueWeight);
        SafeTensors::putMatrix(file, prefix + "attn.o_proj.weight", block.attention.outputWeight);
        SafeTensors::putMatrix(file, prefix + "attn_norm.weight", block.attentionNorm.gamma);
        SafeTensors::putMatrix(file, prefix + "ffn_norm.weight", block.feedForwardNorm.gamma);
        SafeTensors::putMatrix(file, prefix + "ffn.gate_proj.weight", block.feedForward.gateWeight);
        SafeTensors::putMatrix(file, prefix + "ffn.gate_proj.bias", block.feedForward.gateBias);
        SafeTensors::putMatrix(file, prefix + "ffn.up_proj.weight", block.feedForward.upWeight);
        SafeTensors::putMatrix(file, prefix + "ffn.up_proj.bias", block.feedForward.upBias);
        SafeTensors::putMatrix(file, prefix + "ffn.down_proj.weight", block.feedForward.downWeight);
        SafeTensors::putMatrix(file, prefix + "ffn.down_proj.bias", block.feedForward.downBias);
    }
    SafeTensors::putMatrix(file, "final_norm.weight", this->finalNorm.gamma);
    if (!this->tieEmbeddingProjection)
        SafeTensors::putMatrix(file, "lm_head.weight", this->outputProjection.weight);
    SafeTensors::putMatrix(file, "lm_head.bias", this->outputProjection.bias);

    SafeTensors::save(path, file);
}

void LanguageModel::loadSafeTensors(const std::string& path) {
    if (path.empty()) throw std::invalid_argument("LanguageModel::loadSafeTensors empty path");
    if (this->blocks.empty()) throw std::logic_error("LanguageModel::loadSafeTensors no blocks");

    const SafeTensors::File file = SafeTensors::load(path);

    auto metaInt = [&](const char* key, int fallback) -> int {
        const auto it = file.metadata.find(key);
        if (it == file.metadata.end()) return fallback;
        return std::stoi(it->second);
    };

    const int vocabularySize = metaInt("vocab_size", this->tokenEmbedding.vocabSize());
    const int embeddingDim = metaInt("embedding_dim", this->tokenEmbedding.embeddingDim());
    const int maximumPositionCount = metaInt("max_position", this->maximumPositionCount);
    const int blockCount = metaInt("block_count", static_cast<int>(this->blocks.size()));
    const int headCount = metaInt("head_count", this->blocks[0].attention.headCount);
    if (vocabularySize != this->tokenEmbedding.vocabSize()
        || embeddingDim != this->tokenEmbedding.embeddingDim()
        || maximumPositionCount != this->maximumPositionCount
        || blockCount != static_cast<int>(this->blocks.size())
        || headCount != this->blocks[0].attention.headCount)
        throw std::runtime_error("LanguageModel::loadSafeTensors architecture mismatch");

    const auto tieIt = file.metadata.find("tie_embedding");
    const bool tieWeights = tieIt == file.metadata.end()
        ? this->tieEmbeddingProjection
        : (tieIt->second == "1" || tieIt->second == "true");

    this->tokenEmbedding.weight = SafeTensors::requireMatrix(
        file, "token_embedding.weight",
        static_cast<size_t>(vocabularySize), static_cast<size_t>(embeddingDim));

    for (size_t blockIndex = 0; blockIndex < this->blocks.size(); ++blockIndex) {
        TransformerBlock& block = this->blocks[blockIndex];
        const std::string prefix = "blocks." + std::to_string(blockIndex) + ".";
        const size_t d = static_cast<size_t>(embeddingDim);
        block.attention.queryWeight = SafeTensors::requireMatrix(file, prefix + "attn.q_proj.weight", block.attention.queryWeight.rows, block.attention.queryWeight.cols);
        block.attention.keyWeight = SafeTensors::requireMatrix(file, prefix + "attn.k_proj.weight", block.attention.keyWeight.rows, block.attention.keyWeight.cols);
        block.attention.valueWeight = SafeTensors::requireMatrix(file, prefix + "attn.v_proj.weight", block.attention.valueWeight.rows, block.attention.valueWeight.cols);
        block.attention.outputWeight = SafeTensors::requireMatrix(file, prefix + "attn.o_proj.weight", block.attention.outputWeight.rows, block.attention.outputWeight.cols);
        block.attentionNorm.gamma = SafeTensors::requireMatrixFlexible(file, prefix + "attn_norm.weight", d, 1);
        block.feedForwardNorm.gamma = SafeTensors::requireMatrixFlexible(file, prefix + "ffn_norm.weight", d, 1);
        block.feedForward.gateWeight = SafeTensors::requireMatrix(file, prefix + "ffn.gate_proj.weight", block.feedForward.gateWeight.rows, block.feedForward.gateWeight.cols);
        block.feedForward.gateBias = SafeTensors::requireMatrixFlexible(file, prefix + "ffn.gate_proj.bias", block.feedForward.gateBias.rows, 1);
        block.feedForward.upWeight = SafeTensors::requireMatrix(file, prefix + "ffn.up_proj.weight", block.feedForward.upWeight.rows, block.feedForward.upWeight.cols);
        block.feedForward.upBias = SafeTensors::requireMatrixFlexible(file, prefix + "ffn.up_proj.bias", block.feedForward.upBias.rows, 1);
        block.feedForward.downWeight = SafeTensors::requireMatrix(file, prefix + "ffn.down_proj.weight", block.feedForward.downWeight.rows, block.feedForward.downWeight.cols);
        block.feedForward.downBias = SafeTensors::requireMatrixFlexible(file, prefix + "ffn.down_proj.bias", block.feedForward.downBias.rows, 1);
    }

    this->finalNorm.gamma = SafeTensors::requireMatrixFlexible(file, "final_norm.weight", static_cast<size_t>(embeddingDim), 1);
    this->tieEmbeddingProjection = tieWeights;
    if (tieWeights) {
        this->outputProjection.weight = Matrix();
        this->projectionWeightState = AdamState{};
    } else {
        this->outputProjection.weight = SafeTensors::requireMatrix(
            file, "lm_head.weight",
            static_cast<size_t>(vocabularySize), static_cast<size_t>(embeddingDim));
    }
    this->outputProjection.bias = SafeTensors::requireMatrixFlexible(
        file, "lm_head.bias", static_cast<size_t>(vocabularySize), 1);

    if (this->device != nullptr) {
        this->device->uploadFrom(*this);
        this->deviceStale = false;
    }
}

void LanguageModel::runCheckpointSmokeDemo() {
    LanguageModel model(64, 32, 16, Adam(0.001f), 1, 2);
    model.enableCuda();
    if (model.cudaEnabled())
        model.enableCudaTrain();

    const std::vector<int> tokenIds = { 1, 2, 3, 4, 5, 6, 7, 8 };
    Matrix before = model.forward(tokenIds);

    // touch optimizer state so moments are non-zero when cuda train is on
    if (model.cudaTrainEnabled()) {
        LanguageModelDataset dataset;
        dataset.vocabularySize = 64;
        LanguageModelExample example;
        example.inputTokenIds = tokenIds;
        example.targetTokenIds = { 2, 3, 4, 5, 6, 7, 8, 9 };
        dataset.examples.push_back(example);
        model.train(dataset, LanguageModelDataset(), 1, 1, 1, 1);
        before = model.forward(tokenIds);
    }

    const std::string path = "checkpoint_smoke.snlm";
    model.saveCheckpoint(path, true);

    LanguageModel restored(64, 32, 16, Adam(0.002f), 1, 2);
    if (model.cudaEnabled()) {
        restored.enableCuda();
        if (model.cudaTrainEnabled())
            restored.enableCudaTrain();
    }
    restored.loadCheckpoint(path);

    Matrix after = restored.forward(tokenIds);
    float maximumDifference = 0.0f;
    for (size_t index = 0; index < before.data.size(); ++index)
        maximumDifference = (std::max)(maximumDifference, std::fabs(before.data[index] - after.data[index]));

    const bool optimizerMatch = restored.optimizer.timeStep == model.optimizer.timeStep
        && restored.optimizer.learningRate == model.optimizer.learningRate;
    SmokeLog::result("LanguageModel checkpoint", "path=%s  logitsDiff=%.2e  timeStep=%d  optOk=%s",
        path.c_str(), maximumDifference, restored.optimizer.timeStep, optimizerMatch ? "yes" : "no");

    const std::string safePath = "checkpoint_smoke.safetensors";
    model.saveSafeTensors(safePath);
    LanguageModel fromSafe(64, 32, 16, Adam(0.001f), 1, 2);
    if (model.cudaEnabled())
        fromSafe.enableCuda();
    fromSafe.loadCheckpoint(safePath); // extension dispatch
    Matrix afterSafe = fromSafe.forward(tokenIds);
    float safeDiff = 0.0f;
    for (size_t index = 0; index < before.data.size(); ++index)
        safeDiff = (std::max)(safeDiff, std::fabs(before.data[index] - afterSafe.data[index]));
    SmokeLog::result("LanguageModel safetensors", "path=%s  logitsDiff=%.2e", safePath.c_str(), safeDiff);

    std::remove(path.c_str());
    std::remove(safePath.c_str());
}

void LanguageModel::runStreamingSmokeDemo() {
    const std::string path = "streaming_smoke.jsonl";
    {
        std::ofstream out(path, std::ios::binary);
        if (!out) throw std::runtime_error("LanguageModel::runStreamingSmokeDemo cannot write temp jsonl");
        const char* rows[] = {
            "{\"problem_statement\": \"alpha beta gamma delta epsilon\", \"source\": \"Sera-T1\"}",
            "{\"problem_statement\": \"one two three four five six\", \"source\": \"Sera-T2\"}",
            "{\"problem_statement\": \"red blue green yellow orange\", \"source\": \"Sera-T1\"}",
            "{\"problem_statement\": \"cat dog bird fish mouse\", \"source\": \"Sera-T2\"}",
            "{\"problem_statement\": \"train test val split batch\", \"source\": \"Sera-T1\"}",
            "{\"problem_statement\": \"cuda kernel launch grid block\", \"source\": \"Sera-T2\"}",
            "{\"problem_statement\": \"token embed rope attention\", \"source\": \"Sera-T1\"}",
            "{\"problem_statement\": \"loss scale adam moments\", \"source\": \"Sera-T2\"}",
        };
        for (const char* row : rows)
            out << row << '\n';
    }

    LanguageModelChunkSource source(path, 0, 32, 3, 0.75f, 7u, 2);
    std::vector<std::string> sample = source.prepareTokenizerSample(8);
    if (sample.empty()) {
        SmokeLog::result("LanguageModel stream", "skipped empty tokenizer sample");
        std::remove(path.c_str());
        return;
    }

    BPETokenizer tokenizer;
    tokenizer.train(sample, 64);
    source.setTokenizer(&tokenizer);
    source.materialize();

    LanguageModel model(tokenizer.vocabSize(), 32, 32, Adam(0.001f), 1, 2);
    model.enableCuda();
    if (model.cudaEnabled())
        model.enableCudaTrain();

    model.train(source, 1, 1, 2, 1);

    int chunkCount = 0;
    int exampleCount = 0;
    LanguageModelDataset chunk;
    source.rewindTrain();
    for (;;) {
        const bool more = source.nextTrainChunk(chunk);
        if (chunk.examples.empty()) break;
        ++chunkCount;
        exampleCount += chunk.size();
        if (!more) break;
    }

    SmokeLog::result("LanguageModel stream", "chunks=%d  trainExamples=%d  test=%d  vocab=%d  backend=%s",
        chunkCount,
        exampleCount,
        source.testDataset().size(),
        tokenizer.vocabSize(),
        model.cudaTrainEnabled() ? "cuda-stream" : "cpu-stream");

    std::remove(path.c_str());
}
