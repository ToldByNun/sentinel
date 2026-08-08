#include "LanguageModel.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

#include "../Activations/Softmax.hpp"
#include "../Cuda/CudaLanguageModel.hpp"
#include "../Cuda/CudaAmp.hpp"
#include "../Cuda/CudaAdam.hpp"
#include "../Cuda/CudaSbao.hpp"
#include "../Cuda/CudaMatmul.hpp"
#include "../Cuda/CudaMuon.hpp"
#include "../Initializers/UniformInit.hpp"
#include "../IO/SafeTensors.hpp"
#include "../IO/HuggingFaceConfig.hpp"
#include "../IO/HuggingFaceWeights.hpp"
#include "../Losses/CrossEntropy.hpp"
#include "../Tokenizer/BPETokenizer.hpp"
#include "../Tokenizer/HfTokenizer.hpp"
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

LanguageModel::LanguageModel(
    int vocabularySize,
    int embeddingDim,
    int maximumPositionCount,
    Adam optimizer,
    int blockCount,
    int headCount,
    int intermediateSize,
    float ropeTheta,
    bool useBias,
    int kvHeadCount)
    : tokenEmbedding(vocabularySize, embeddingDim),
      finalNorm(embeddingDim),
      outputProjection(
          UniformInit::matrix(vocabularySize, embeddingDim, 0.1f, 31u),
          useBias
              ? UniformInit::matrix(vocabularySize, 1, 0.01f, 32u)
              : Matrix(static_cast<size_t>(vocabularySize), 1, 0.0f)),
      optimizer(optimizer),
      maximumPositionCount(maximumPositionCount),
      tieEmbeddingProjection(true),
      deviceStale(false),
      deviceTrainEnabled(false) {
    if (maximumPositionCount <= 0) throw std::invalid_argument("LanguageModel maximumPositionCount must be > 0");
    if (blockCount <= 0) throw std::invalid_argument("LanguageModel blockCount must be > 0");
    if (headCount <= 0) throw std::invalid_argument("LanguageModel headCount must be > 0");
    if (intermediateSize < 0) throw std::invalid_argument("LanguageModel intermediateSize must be >= 0");
    if (ropeTheta <= 0.0f) throw std::invalid_argument("LanguageModel ropeTheta must be > 0");
    const int resolvedKvHeadCount = (kvHeadCount <= 0) ? headCount : kvHeadCount;
    if (resolvedKvHeadCount <= 0) throw std::invalid_argument("LanguageModel kvHeadCount must be > 0");
    if (headCount % resolvedKvHeadCount != 0)
        throw std::invalid_argument("LanguageModel headCount must be divisible by kvHeadCount");

    this->blocks.reserve(static_cast<size_t>(blockCount));
    for (int blockIndex = 0; blockIndex < blockCount; ++blockIndex)
        this->blocks.push_back(TransformerBlock(
            embeddingDim,
            headCount,
            maximumPositionCount,
            21u + static_cast<unsigned>(blockIndex) * 100u,
            intermediateSize,
            ropeTheta,
            useBias,
            resolvedKvHeadCount));

    // weight tying: LM head shares tokenEmbedding; drop untied projection weight + Adam
    this->outputProjection.weight = Matrix();
    // Adam moments allocated lazily on first update (4B ctor must not reserve 2x param RAM).
    this->tokenEmbeddingState = AdamState{};
    this->finalNormGammaState = AdamState{};
    this->projectionWeightState = AdamState{};
    this->projectionBiasState = AdamState{};
}

int LanguageModel::intermediateSize() const {
    if (this->blocks.empty()) return 0;
    return this->blocks[0].feedForward.intermediateSize();
}

float LanguageModel::ropeTheta() const {
    if (this->blocks.empty()) return RotaryEmbedding::DefaultBase;
    return this->blocks[0].attention.rotaryEmbedding.base;
}

bool LanguageModel::useBias() const {
    if (this->blocks.empty()) return true;
    return this->blocks[0].feedForward.useBias;
}

int LanguageModel::kvHeadCount() const {
    if (this->blocks.empty()) return 0;
    return this->blocks[0].attention.kvHeadCount;
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
        device.preferSpulse ? "spulse+adam" : (device.preferMuon ? "muon+adam" : "adam"));

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

    // SBAO Auto: re-resolve after weights are on device so free VRAM is meaningful.
    if (CudaSbao::enabled && CudaSbao::request == SbaoMode::Auto) {
        size_t freeBytes = 0;
        size_t totalBytes = 0;
        if (CudaMatmul::isAvailable()) {
            cudaDeviceSynchronize();
            cudaMemGetInfo(&freeBytes, &totalBytes);
        }
        const size_t parameterBytes = this->parameterElementCount() * sizeof(float);
        const SbaoMode picked = CudaSbao::resolveAndApply(freeBytes, parameterBytes, 0);
        std::cout << "LanguageModel::enableCudaTrain: SBAO Auto → " << CudaSbao::modeName(picked)
                  << "  freeVram=" << (freeBytes / (1024ull * 1024ull)) << " MiB"
                  << "  params=" << (parameterBytes / (1024ull * 1024ull)) << " MiB\n";
    }

    auto applyHostBaoSideEffects = [this]() {
        CudaAmp::preferMixedPrecision = true;
        this->device->preferTrainGraph = false;
        this->device->releaseTrainGraph();
        if (this->device->preferMuon) {
            this->device->preferMuon = false;
            std::cout << "LanguageModel::enableCudaTrain: disabling Muon (host SBAO path)\n";
        }
    };

    if (CudaSbao::resolved == SbaoMode::HostFusedHalfAdam || CudaSbao::resolved == SbaoMode::HostFusedHalfSgd
        || CudaAdam::preferHostGradients || CudaAdam::preferHostSgd)
        applyHostBaoSideEffects();
    else if (CudaSbao::resolved == SbaoMode::GpuInt8Adam || !CudaAdam::preferCpuOffload) {
        // Fast path: keep CUDA graphs available (ckpt Off below).
        this->device->preferTrainGraph = true;
    }

    // GpuInt8 / resident: Off (retain acts → graphs).
    // HostFusedHalfAdam: Off on mid (faster); Full on large (matches prior 4B offload setup).
    // HostFusedHalfSgd (~4B): Full from the start.
    if (CudaSbao::resolved == SbaoMode::HostFusedHalfSgd || CudaAdam::preferHostSgd
        || ((CudaSbao::resolved == SbaoMode::HostFusedHalfAdam || CudaAdam::preferHostGradients)
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

    if (CudaSbao::resolved == SbaoMode::HostFusedHalfSgd || CudaAdam::preferHostSgd) {
        this->device->releasePackedTrainWorkspaces();
        this->device->tuneOffloadCheckpointAndPack(4096);
    } else {
        this->device->trainStateReady = false;
        this->device->ensureTrainState();
        this->device->applyVramPackBudget();
        this->device->trainStateReady = false;
        this->device->ensureTrainState();

        // Auto + GpuInt8 but pack starved to the minimum → fall back once to HostFusedHalfAdam.
        if (CudaSbao::enabled && CudaSbao::request == SbaoMode::Auto
            && CudaSbao::resolved == SbaoMode::GpuInt8Adam
            && this->device->maxPackedColumns <= CudaLanguageModel::lengthBucketStep) {
            std::cout << "LanguageModel::enableCudaTrain: GpuInt8 pack starved (maxPackCols="
                      << this->device->maxPackedColumns << ") → HostFusedHalfAdam\n";
            CudaSbao::request = SbaoMode::Auto;
            CudaSbao::apply(SbaoMode::HostFusedHalfAdam);
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
              << ", opt=" << (this->device->preferSpulse
                    ? (CudaSbao::pipelineHostWeightUpdate() ? "spulse-host+adam" : "spulse+adam")
                    : (this->device->preferMuon ? "muon+adam"
                        : (CudaAdam::preferHostSgd ? "host-sgd"
                            : (CudaSbao::pipelineHostAdam() ? "host-adam" : "adam"))))
              << ", sbao=" << (CudaSbao::enabled ? CudaSbao::modeName(CudaSbao::resolved) : "off")
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
        // Legacy alias → SBAO HostFusedHalfAdam (fused FP16 grad D2H + async host Adam).
        CudaSbao::request = SbaoMode::HostFusedHalfAdam;
        CudaSbao::apply(SbaoMode::HostFusedHalfAdam);
        CudaAmp::preferMixedPrecision = true;
        this->device->preferTrainGraph = false;
        this->device->releaseTrainGraph();
        if (this->device->preferMuon) {
            this->device->preferMuon = false;
            std::cout << "LanguageModel::setCudaPreferCpuAdamOffload: disabling Muon (FP16 GPU weights)\n";
        }
        std::cout << "LanguageModel::setCudaPreferCpuAdamOffload: disabling CUDA Graph for FP16 working weights\n";
    } else {
        CudaSbao::enabled = false;
        CudaSbao::request = SbaoMode::Auto;
        CudaSbao::resolved = SbaoMode::GpuInt8Adam;
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
              << "  sbao=" << (CudaSbao::enabled ? CudaSbao::modeName(CudaSbao::resolved) : "off")
              << "  fp16GpuWeights=" << (CudaAdam::preferFp16GpuWeights ? "on" : "off")
              << "  hostGrads=" << (CudaAdam::preferHostGradients ? "on" : "off")
              << "  amp=" << (CudaAmp::preferMixedPrecision ? "on" : "off")
              << "  int8 adam " << (CudaAdam::preferInt8Moments ? "on" : "off") << '\n';
}

void LanguageModel::setCudaPreferHostSgd(bool enabled) {
    if (enabled) {
        CudaSbao::request = SbaoMode::HostFusedHalfSgd;
        CudaSbao::apply(SbaoMode::HostFusedHalfSgd);
        CudaAmp::preferMixedPrecision = true;
        if (this->device != nullptr) {
            this->device->preferTrainGraph = false;
            this->device->releaseTrainGraph();
        }
    } else {
        CudaAdam::preferHostSgd = false;
        if (CudaAdam::preferCpuOffload) {
            CudaSbao::request = SbaoMode::HostFusedHalfAdam;
            CudaSbao::apply(SbaoMode::HostFusedHalfAdam);
        } else {
            CudaSbao::enabled = false;
            CudaSbao::request = SbaoMode::Auto;
            CudaSbao::resolved = SbaoMode::GpuInt8Adam;
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
              << "  sbao=" << (CudaSbao::enabled ? CudaSbao::modeName(CudaSbao::resolved) : "off") << '\n';
}

void LanguageModel::setCudaPreferSbao(bool enabled) {
    if (this->device == nullptr) this->enableCuda();
    if (this->device == nullptr) return;
    if (!enabled) {
        CudaSbao::enabled = false;
        CudaSbao::request = SbaoMode::Auto;
        CudaSbao::resolved = SbaoMode::GpuInt8Adam;
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
        std::cout << "LanguageModel::setCudaPreferSbao: off\n";
        return;
    }
    CudaSbao::enabled = true;
    CudaSbao::request = SbaoMode::Auto;
    // Defer concrete pick until enableCudaTrain (free VRAM after upload). If train is
    // already on, resolve immediately.
    if (this->deviceTrainEnabled)
        this->setCudaSbaoMode(SbaoMode::Auto);
    else
        std::cout << "LanguageModel::setCudaPreferSbao: on  request=Auto (resolve at enable_cuda_train)\n";
}

void LanguageModel::setCudaSbaoMode(SbaoMode mode) {
    if (this->device == nullptr) this->enableCuda();
    if (this->device == nullptr) return;
    CudaSbao::enabled = true;
    CudaSbao::request = mode;
    size_t freeBytes = 0;
    size_t totalBytes = 0;
    if (CudaMatmul::isAvailable()) {
        cudaDeviceSynchronize();
        cudaMemGetInfo(&freeBytes, &totalBytes);
    }
    const size_t parameterBytes = this->parameterElementCount() * sizeof(float);
    const SbaoMode resolved = CudaSbao::resolveAndApply(freeBytes, parameterBytes, 0);
    CudaSbao::request = mode; // keep Auto as request when policy-selected
    if (resolved == SbaoMode::HostFusedHalfAdam || resolved == SbaoMode::HostFusedHalfSgd)
        CudaAmp::preferMixedPrecision = true;
    if (resolved == SbaoMode::GpuInt8Adam) {
        this->device->preferTrainGraph = true;
    } else {
        this->device->preferTrainGraph = false;
        this->device->releaseTrainGraph();
        if (this->device->preferMuon) {
            this->device->preferMuon = false;
            std::cout << "LanguageModel::setCudaSbaoMode: disabling Muon (host SBAO path)\n";
        }
    }
    this->device->trainStateReady = false;
    if (this->deviceTrainEnabled) {
        if (resolved == SbaoMode::HostFusedHalfSgd) {
            this->device->setActivationCheckpointMode(ActivationCheckpointMode::Full);
            this->device->releasePackedTrainWorkspaces();
            this->device->tuneOffloadCheckpointAndPack(4096);
        } else if (resolved == SbaoMode::HostFusedHalfAdam) {
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
    std::cout << "LanguageModel::setCudaSbaoMode: request=" << CudaSbao::modeName(mode)
              << "  resolved=" << CudaSbao::modeName(resolved)
              << "  freeVram=" << (freeBytes / (1024ull * 1024ull)) << " MiB\n";
}

SbaoMode LanguageModel::cudaSbaoModeResolved() const {
    return CudaSbao::resolved;
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

void LanguageModel::setCudaPreferSpulse(bool enabled) {
    if (this->device == nullptr) this->enableCuda();
    if (this->device == nullptr) return;
    this->device->setPreferSpulse(enabled);
    if (this->deviceTrainEnabled) {
        this->device->trainStateReady = false;
        this->device->ensureTrainState();
    }
    std::cout << "LanguageModel::setCudaPreferSpulse: " << (enabled ? "on" : "off")
              << "  coverage=" << CudaSpulse::coverageName(this->device->spulse.coverage) << '\n';
}

void LanguageModel::setCudaSpulseCoverage(SpulseCoverage coverage) {
    if (coverage == SpulseCoverage::Full)
        throw std::invalid_argument("LanguageModel::setCudaSpulseCoverage Full not implemented yet (use Hybrid)");
    if (this->device == nullptr) this->enableCuda();
    if (this->device == nullptr) return;
    this->device->spulse.coverage = coverage;
    std::cout << "LanguageModel::setCudaSpulseCoverage: " << CudaSpulse::coverageName(coverage) << '\n';
}

void LanguageModel::setCudaSpulseMomentumBeta(float beta) {
    if (beta < 0.0f || beta >= 1.0f)
        throw std::invalid_argument("LanguageModel::setCudaSpulseMomentumBeta must be in [0, 1)");
    if (this->device == nullptr) this->enableCuda();
    if (this->device == nullptr) return;
    this->device->spulse.momentumBeta = beta;
}

void LanguageModel::setCudaSpulseFastBeta(float beta) {
    if (beta < 0.0f || beta >= 1.0f)
        throw std::invalid_argument("LanguageModel::setCudaSpulseFastBeta must be in [0, 1)");
    if (this->device == nullptr) this->enableCuda();
    if (this->device == nullptr) return;
    this->device->spulse.fastBeta = beta;
}

void LanguageModel::setCudaSpulseSlowBeta(float beta) {
    if (beta < 0.0f || beta >= 1.0f)
        throw std::invalid_argument("LanguageModel::setCudaSpulseSlowBeta must be in [0, 1)");
    if (this->device == nullptr) this->enableCuda();
    if (this->device == nullptr) return;
    this->device->spulse.slowBeta = beta;
}

void LanguageModel::setCudaSpulseScaleClip(float scaleMin, float scaleMax) {
    if (scaleMin <= 0.0f || scaleMax < scaleMin)
        throw std::invalid_argument("LanguageModel::setCudaSpulseScaleClip invalid range");
    if (this->device == nullptr) this->enableCuda();
    if (this->device == nullptr) return;
    this->device->spulse.scaleMin = scaleMin;
    this->device->spulse.scaleMax = scaleMax;
}

void LanguageModel::setCudaSpulseMomentumStorage(SpulseMomentumStorage storage) {
    if (this->device == nullptr) this->enableCuda();
    if (this->device == nullptr) return;
    if (this->device->spulse.momentumStorage == storage) return;
    this->device->spulse.momentumStorage = storage;
    this->device->trainStateReady = false;
    if (this->deviceTrainEnabled)
        this->device->ensureTrainState();
    std::cout << "LanguageModel::setCudaSpulseMomentumStorage: "
              << CudaSpulse::momentumStorageName(storage) << '\n';
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
    if (bias.empty()) return result;
    for (size_t row = 0; row < result.rows; ++row) {
        const float biasValue = bias.at(row, 0);
        if (biasValue == 0.0f) continue;
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
    if (!this->useBias())
        return product;
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
    Matrix projectionBiasGradient = this->useBias()
        ? LanguageModel::sumColumns(logitGradient)
        : Matrix(static_cast<size_t>(this->tokenEmbedding.vocabSize()), 1, 0.0f);
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
    if (this->useBias())
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
    file.metadata["kv_head_count"] = std::to_string(this->kvHeadCount());
    file.metadata["intermediate_size"] = std::to_string(this->intermediateSize());
    file.metadata["rope_theta"] = std::to_string(this->ropeTheta());
    file.metadata["use_bias"] = this->useBias() ? "1" : "0";
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
        SafeTensors::putMatrix(file, prefix + "ffn.up_proj.weight", block.feedForward.upWeight);
        SafeTensors::putMatrix(file, prefix + "ffn.down_proj.weight", block.feedForward.downWeight);
        if (this->useBias()) {
            SafeTensors::putMatrix(file, prefix + "ffn.gate_proj.bias", block.feedForward.gateBias);
            SafeTensors::putMatrix(file, prefix + "ffn.up_proj.bias", block.feedForward.upBias);
            SafeTensors::putMatrix(file, prefix + "ffn.down_proj.bias", block.feedForward.downBias);
        }
    }
    SafeTensors::putMatrix(file, "final_norm.weight", this->finalNorm.gamma);
    if (!this->tieEmbeddingProjection)
        SafeTensors::putMatrix(file, "lm_head.weight", this->outputProjection.weight);
    if (this->useBias())
        SafeTensors::putMatrix(file, "lm_head.bias", this->outputProjection.bias);

    SafeTensors::save(path, file);
}

void LanguageModel::loadSafeTensors(const std::string& path) {
    if (path.empty()) throw std::invalid_argument("LanguageModel::loadSafeTensors empty path");
    this->loadSafeTensors(SafeTensors::load(path));
}

void LanguageModel::loadSafeTensors(const SafeTensors::File& file) {
    if (this->blocks.empty()) throw std::logic_error("LanguageModel::loadSafeTensors no blocks");

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
    const int kvHeadCount = metaInt("kv_head_count", this->kvHeadCount());
    const int intermediateSize = metaInt("intermediate_size", this->intermediateSize());
    const auto metaFloat = [&](const char* key, float fallback) -> float {
        const auto it = file.metadata.find(key);
        if (it == file.metadata.end()) return fallback;
        return std::stof(it->second);
    };
    const float ropeTheta = metaFloat("rope_theta", this->ropeTheta());
    const auto metaBool = [&](const char* key, bool fallback) -> bool {
        const auto it = file.metadata.find(key);
        if (it == file.metadata.end()) return fallback;
        return it->second == "1" || it->second == "true";
    };
    const bool useBias = metaBool("use_bias", this->useBias());
    if (vocabularySize != this->tokenEmbedding.vocabSize()
        || embeddingDim != this->tokenEmbedding.embeddingDim()
        || maximumPositionCount != this->maximumPositionCount
        || blockCount != static_cast<int>(this->blocks.size())
        || headCount != this->blocks[0].attention.headCount
        || kvHeadCount != this->kvHeadCount()
        || intermediateSize != this->intermediateSize()
        || std::fabs(ropeTheta - this->ropeTheta()) > 1.0e-3f * (std::max)(1.0f, std::fabs(this->ropeTheta()))
        || useBias != this->useBias())
        throw std::runtime_error("LanguageModel::loadSafeTensors architecture mismatch");

    const auto tieIt = file.metadata.find("tie_embedding");
    const bool tieWeights = tieIt == file.metadata.end()
        ? this->tieEmbeddingProjection
        : (tieIt->second == "1" || tieIt->second == "true");

    auto loadBiasOrZero = [&](const std::string& name, size_t rows) -> Matrix {
        if (file.tensors.find(name) == file.tensors.end())
            return Matrix(rows, 1, 0.0f);
        return SafeTensors::requireMatrixFlexible(file, name, rows, 1);
    };

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
        block.feedForward.upWeight = SafeTensors::requireMatrix(file, prefix + "ffn.up_proj.weight", block.feedForward.upWeight.rows, block.feedForward.upWeight.cols);
        block.feedForward.downWeight = SafeTensors::requireMatrix(file, prefix + "ffn.down_proj.weight", block.feedForward.downWeight.rows, block.feedForward.downWeight.cols);
        block.feedForward.useBias = useBias;
        if (useBias) {
            block.feedForward.gateBias = loadBiasOrZero(prefix + "ffn.gate_proj.bias", block.feedForward.gateBias.rows);
            block.feedForward.upBias = loadBiasOrZero(prefix + "ffn.up_proj.bias", block.feedForward.upBias.rows);
            block.feedForward.downBias = loadBiasOrZero(prefix + "ffn.down_proj.bias", block.feedForward.downBias.rows);
        } else {
            block.feedForward.gateBias = Matrix(block.feedForward.gateBias.rows, 1, 0.0f);
            block.feedForward.upBias = Matrix(block.feedForward.upBias.rows, 1, 0.0f);
            block.feedForward.downBias = Matrix(block.feedForward.downBias.rows, 1, 0.0f);
        }
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
    this->outputProjection.bias = useBias
        ? loadBiasOrZero("lm_head.bias", static_cast<size_t>(vocabularySize))
        : Matrix(static_cast<size_t>(vocabularySize), 1, 0.0f);

    if (this->device != nullptr) {
        this->device->uploadFrom(*this);
        this->deviceStale = false;
    }
}

LanguageModel LanguageModel::loadHuggingFace(const std::string& modelDirectory, float learningRate) {
    if (modelDirectory.empty())
        throw std::invalid_argument("LanguageModel::loadHuggingFace empty modelDirectory");
    if (!(learningRate > 0.0f))
        throw std::invalid_argument("LanguageModel::loadHuggingFace learningRate must be > 0");

    const HuggingFace::Config config = HuggingFace::loadConfig(modelDirectory);
    LanguageModel model(
        config.vocabSize,
        config.hiddenSize,
        config.maxPositionEmbeddings,
        Adam(learningRate),
        config.numHiddenLayers,
        config.numAttentionHeads,
        config.intermediateSize,
        config.ropeTheta,
        config.useBias,
        config.numKeyValueHeads);

    model.finalNorm.epsilon = config.rmsNormEps;
    for (TransformerBlock& block : model.blocks) {
        block.attentionNorm.epsilon = config.rmsNormEps;
        block.feedForwardNorm.epsilon = config.rmsNormEps;
    }

    if (config.tieWordEmbeddings != model.tieEmbeddingProjection)
        model.setTieEmbeddingProjection(config.tieWordEmbeddings);

    const SafeTensors::File mapped = HuggingFace::loadMappedWeights(modelDirectory, config);
    model.loadSafeTensors(mapped);
    return model;
}

void LanguageModel::saveHuggingFace(
    const std::string& modelDirectory,
    const std::string& modelType,
    const std::string& tokenizerSourceDirectory,
    const std::string& weightFormat) {
    if (modelDirectory.empty())
        throw std::invalid_argument("LanguageModel::saveHuggingFace empty modelDirectory");
    if (this->blocks.empty())
        throw std::logic_error("LanguageModel::saveHuggingFace no blocks");
    if (!HuggingFace::isSupportedModelType(modelType))
        throw std::invalid_argument(
            "LanguageModel::saveHuggingFace unsupported model_type='" + modelType
            + "' (allowlist: llama, mistral, qwen2)");
    const HuggingFace::WeightExportFormat exportFormat =
        HuggingFace::parseWeightExportFormat(weightFormat);

    if (this->cudaEnabled() && this->device != nullptr && !this->deviceStale)
        this->device->downloadTo(*this);
    else if (this->cudaTrainEnabled() && this->device != nullptr)
        this->device->downloadTo(*this);

    // Same Sentinel tensor packing as saveSafeTensors (in-memory; no temp file).
    SafeTensors::File sentinelWeights;
    sentinelWeights.metadata["format"] = "sentinel";
    sentinelWeights.metadata["arch"] = "causal_lm_rope_swiglu";
    sentinelWeights.metadata["vocab_size"] = std::to_string(this->tokenEmbedding.vocabSize());
    sentinelWeights.metadata["embedding_dim"] = std::to_string(this->tokenEmbedding.embeddingDim());
    sentinelWeights.metadata["max_position"] = std::to_string(this->maximumPositionCount);
    sentinelWeights.metadata["block_count"] = std::to_string(this->blocks.size());
    sentinelWeights.metadata["head_count"] = std::to_string(this->blocks[0].attention.headCount);
    sentinelWeights.metadata["kv_head_count"] = std::to_string(this->kvHeadCount());
    sentinelWeights.metadata["intermediate_size"] = std::to_string(this->intermediateSize());
    sentinelWeights.metadata["rope_theta"] = std::to_string(this->ropeTheta());
    sentinelWeights.metadata["use_bias"] = this->useBias() ? "1" : "0";
    sentinelWeights.metadata["tie_embedding"] = this->tieEmbeddingProjection ? "1" : "0";

    SafeTensors::putMatrix(sentinelWeights, "token_embedding.weight", this->tokenEmbedding.weight);
    for (size_t blockIndex = 0; blockIndex < this->blocks.size(); ++blockIndex) {
        const TransformerBlock& block = this->blocks[blockIndex];
        const std::string prefix = "blocks." + std::to_string(blockIndex) + ".";
        SafeTensors::putMatrix(sentinelWeights, prefix + "attn.q_proj.weight", block.attention.queryWeight);
        SafeTensors::putMatrix(sentinelWeights, prefix + "attn.k_proj.weight", block.attention.keyWeight);
        SafeTensors::putMatrix(sentinelWeights, prefix + "attn.v_proj.weight", block.attention.valueWeight);
        SafeTensors::putMatrix(sentinelWeights, prefix + "attn.o_proj.weight", block.attention.outputWeight);
        SafeTensors::putMatrix(sentinelWeights, prefix + "attn_norm.weight", block.attentionNorm.gamma);
        SafeTensors::putMatrix(sentinelWeights, prefix + "ffn_norm.weight", block.feedForwardNorm.gamma);
        SafeTensors::putMatrix(sentinelWeights, prefix + "ffn.gate_proj.weight", block.feedForward.gateWeight);
        SafeTensors::putMatrix(sentinelWeights, prefix + "ffn.up_proj.weight", block.feedForward.upWeight);
        SafeTensors::putMatrix(sentinelWeights, prefix + "ffn.down_proj.weight", block.feedForward.downWeight);
        if (this->useBias()) {
            SafeTensors::putMatrix(sentinelWeights, prefix + "ffn.gate_proj.bias", block.feedForward.gateBias);
            SafeTensors::putMatrix(sentinelWeights, prefix + "ffn.up_proj.bias", block.feedForward.upBias);
            SafeTensors::putMatrix(sentinelWeights, prefix + "ffn.down_proj.bias", block.feedForward.downBias);
        }
    }
    SafeTensors::putMatrix(sentinelWeights, "final_norm.weight", this->finalNorm.gamma);
    if (!this->tieEmbeddingProjection)
        SafeTensors::putMatrix(sentinelWeights, "lm_head.weight", this->outputProjection.weight);
    if (this->useBias())
        SafeTensors::putMatrix(sentinelWeights, "lm_head.bias", this->outputProjection.bias);

    HuggingFace::Config config;
    config.modelType = modelType;
    config.architecture = HuggingFace::defaultArchitectureName(modelType);
    config.vocabSize = this->tokenEmbedding.vocabSize();
    config.hiddenSize = this->tokenEmbedding.embeddingDim();
    config.intermediateSize = this->intermediateSize();
    config.numHiddenLayers = static_cast<int>(this->blocks.size());
    config.numAttentionHeads = this->blocks[0].attention.headCount;
    config.numKeyValueHeads = this->kvHeadCount();
    config.maxPositionEmbeddings = this->maximumPositionCount;
    config.rmsNormEps = this->finalNorm.epsilon;
    config.ropeTheta = this->ropeTheta();
    config.tieWordEmbeddings = this->tieEmbeddingProjection;
    config.useBias = this->useBias();

    HuggingFace::saveDirectory(
        modelDirectory,
        config,
        sentinelWeights,
        HuggingFace::WeightLayoutFamily::LlamaMistralLike,
        tokenizerSourceDirectory,
        exportFormat);
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

void LanguageModel::runIntermediateSizeSmokeDemo() {
    FeedForward::runIntermediateSizeSmokeDemo();

    const int embed = 64;
    const int expectedDefault = FeedForward::defaultIntermediateSize(embed, 4);
    const int custom = 256;

    LanguageModel defaultLm(32, embed, 32, Adam(1e-3f), 1, 4, 0);
    LanguageModel customLm(32, embed, 32, Adam(1e-3f), 1, 4, custom);
    if (defaultLm.intermediateSize() != expectedDefault)
        throw std::runtime_error("LanguageModel default intermediateSize mismatch");
    if (customLm.intermediateSize() != custom)
        throw std::runtime_error("LanguageModel explicit intermediateSize mismatch");

    const std::string path = "intermediate_size_smoke.safetensors";
    customLm.saveSafeTensors(path);
    LanguageModel restored(32, embed, 32, Adam(1e-3f), 1, 4, custom);
    restored.loadSafeTensors(path);
    if (restored.intermediateSize() != custom)
        throw std::runtime_error("LanguageModel safetensors intermediate_size metadata roundtrip failed");

    bool rejected = false;
    try {
        LanguageModel wrong(32, embed, 32, Adam(1e-3f), 1, 4, expectedDefault);
        wrong.loadSafeTensors(path);
    } catch (const std::exception&) {
        rejected = true;
    }
    if (!rejected)
        throw std::runtime_error("loadSafeTensors should reject intermediate_size mismatch");

    std::remove(path.c_str());
    SmokeLog::result(
        "LanguageModel intermediate_size",
        "default=%d  custom=%d  safetensors=ok",
        expectedDefault,
        custom);
}

void LanguageModel::runRopeThetaSmokeDemo() {
    const int embed = 64;
    const int heads = 4;
    const int maxPos = 32;
    const float legacy = RotaryEmbedding::DefaultBase;
    const float llama3 = 500000.0f;

    LanguageModel defaultLm(32, embed, maxPos, Adam(1e-3f), 1, heads, 0, legacy);
    LanguageModel llamaLm(32, embed, maxPos, Adam(1e-3f), 1, heads, 0, llama3);
    if (std::fabs(defaultLm.ropeTheta() - legacy) > 0.0f)
        throw std::runtime_error("LanguageModel default ropeTheta mismatch");
    if (std::fabs(llamaLm.ropeTheta() - llama3) > 0.0f)
        throw std::runtime_error("LanguageModel llama3 ropeTheta mismatch");

    const auto& legacyRope = defaultLm.blocks[0].attention.rotaryEmbedding;
    const auto& llamaRope = llamaLm.blocks[0].attention.rotaryEmbedding;
    if (legacyRope.cosTable.size() != llamaRope.cosTable.size() || legacyRope.cosTable.empty())
        throw std::runtime_error("LanguageModel rope table size mismatch");

    float maxAbsDiff = 0.0f;
    for (size_t i = 0; i < legacyRope.cosTable.size(); ++i) {
        maxAbsDiff = (std::max)(maxAbsDiff, std::fabs(legacyRope.cosTable[i] - llamaRope.cosTable[i]));
        maxAbsDiff = (std::max)(maxAbsDiff, std::fabs(legacyRope.sinTable[i] - llamaRope.sinTable[i]));
    }
    if (maxAbsDiff < 1.0e-4f)
        throw std::runtime_error("LanguageModel rope tables should differ for base 1e4 vs 5e5");

    const std::string path = "rope_theta_smoke.safetensors";
    llamaLm.saveSafeTensors(path);
    LanguageModel restored(32, embed, maxPos, Adam(1e-3f), 1, heads, 0, llama3);
    restored.loadSafeTensors(path);
    if (std::fabs(restored.ropeTheta() - llama3) > 1.0e-3f)
        throw std::runtime_error("LanguageModel safetensors rope_theta metadata roundtrip failed");

    bool rejected = false;
    try {
        LanguageModel wrong(32, embed, maxPos, Adam(1e-3f), 1, heads, 0, legacy);
        wrong.loadSafeTensors(path);
    } catch (const std::exception&) {
        rejected = true;
    }
    if (!rejected)
        throw std::runtime_error("loadSafeTensors should reject rope_theta mismatch");

    std::remove(path.c_str());
    SmokeLog::result(
        "LanguageModel rope_theta",
        "default=%.0f  llama3=%.0f  tableDiff=%.3e  safetensors=ok",
        legacy,
        llama3,
        maxAbsDiff);
}

void LanguageModel::runBiasPolicySmokeDemo() {
    const int embed = 64;
    const int heads = 4;
    const int maxPos = 32;

    LanguageModel withBias(32, embed, maxPos, Adam(1e-3f), 1, heads, 0, RotaryEmbedding::DefaultBase, true);
    LanguageModel noBias(32, embed, maxPos, Adam(1e-3f), 1, heads, 0, RotaryEmbedding::DefaultBase, false);
    if (!withBias.useBias() || noBias.useBias())
        throw std::runtime_error("LanguageModel useBias accessor mismatch");

    auto assertAllZero = [](const Matrix& m, const char* name) {
        for (float v : m.data) {
            if (v != 0.0f)
                throw std::runtime_error(std::string("expected zero bias: ") + name);
        }
    };
    assertAllZero(noBias.outputProjection.bias, "lm_head");
    assertAllZero(noBias.blocks[0].feedForward.gateBias, "gate");
    assertAllZero(noBias.blocks[0].feedForward.upBias, "up");
    assertAllZero(noBias.blocks[0].feedForward.downBias, "down");

    LanguageModelDataset dataset;
    LanguageModelExample example;
    example.inputTokenIds = { 1, 2, 3, 4 };
    example.targetTokenIds = { 2, 3, 4, 5 };
    dataset.examples.push_back(example);
    noBias.train(dataset, 1, 0);

    assertAllZero(noBias.outputProjection.bias, "lm_head after train");
    assertAllZero(noBias.blocks[0].feedForward.gateBias, "gate after train");
    assertAllZero(noBias.blocks[0].feedForward.upBias, "up after train");
    assertAllZero(noBias.blocks[0].feedForward.downBias, "down after train");

    const std::string path = "bias_policy_smoke.safetensors";
    noBias.saveSafeTensors(path);
    const SafeTensors::File saved = SafeTensors::load(path);
    if (saved.metadata.at("use_bias") != "0")
        throw std::runtime_error("safetensors use_bias metadata should be 0");
    if (saved.tensors.count("lm_head.bias") != 0
        || saved.tensors.count("blocks.0.ffn.gate_proj.bias") != 0)
        throw std::runtime_error("useBias=false save should omit bias tensors");

    LanguageModel restored(32, embed, maxPos, Adam(1e-3f), 1, heads, 0, RotaryEmbedding::DefaultBase, false);
    restored.loadSafeTensors(path);
    if (restored.useBias())
        throw std::runtime_error("restored useBias should stay false");
    assertAllZero(restored.outputProjection.bias, "restored lm_head");

    Matrix afterTrain = noBias.forward(example.inputTokenIds);
    Matrix afterLoad = restored.forward(example.inputTokenIds);
    float restoreDiff = 0.0f;
    for (size_t i = 0; i < afterTrain.data.size(); ++i)
        restoreDiff = (std::max)(restoreDiff, std::fabs(afterTrain.data[i] - afterLoad.data[i]));
    if (restoreDiff > 1.0e-5f)
        throw std::runtime_error("bias-policy safetensors roundtrip logits mismatch");

    bool rejected = false;
    try {
        withBias.loadSafeTensors(path);
    } catch (const std::exception&) {
        rejected = true;
    }
    if (!rejected)
        throw std::runtime_error("loadSafeTensors should reject use_bias mismatch");

    std::remove(path.c_str());
    SmokeLog::result(
        "LanguageModel bias_policy",
        "use_bias=0  omit_tensors=ok  train_keeps_zero=ok  restoreDiff=%.2e",
        restoreDiff);
}

void LanguageModel::runKvHeadCountSmokeDemo() {
    const int embed = 64;
    const int heads = 8;
    const int kvHeads = 2;
    const int maxPos = 32;

    LanguageModel mhaDefault(32, embed, maxPos, Adam(1e-3f), 1, heads);
    LanguageModel mhaExplicit(
        32, embed, maxPos, Adam(1e-3f), 1, heads, 0, RotaryEmbedding::DefaultBase, true, heads);
    LanguageModel gqa(
        32, embed, maxPos, Adam(1e-3f), 1, heads, 0, RotaryEmbedding::DefaultBase, true, kvHeads);

    if (mhaDefault.kvHeadCount() != heads || mhaExplicit.kvHeadCount() != heads)
        throw std::runtime_error("LanguageModel MHA kvHeadCount mismatch");
    if (gqa.kvHeadCount() != kvHeads)
        throw std::runtime_error("LanguageModel GQA kvHeadCount mismatch");

    const int expectedKvRows = kvHeads * (embed / heads);
    if (static_cast<int>(gqa.blocks[0].attention.keyWeight.rows) != expectedKvRows
        || static_cast<int>(gqa.blocks[0].attention.valueWeight.rows) != expectedKvRows)
        throw std::runtime_error("LanguageModel GQA key/value rows mismatch");

    bool rejectedCtor = false;
    try {
        LanguageModel bad(
            32, embed, maxPos, Adam(1e-3f), 1, heads, 0, RotaryEmbedding::DefaultBase, true, 3);
        (void)bad;
    } catch (const std::exception&) {
        rejectedCtor = true;
    }
    if (!rejectedCtor)
        throw std::runtime_error("LanguageModel should reject headCount not divisible by kvHeadCount");

    const std::string path = "kv_head_count_smoke.safetensors";
    gqa.saveSafeTensors(path);
    const SafeTensors::File saved = SafeTensors::load(path);
    if (saved.metadata.at("kv_head_count") != std::to_string(kvHeads))
        throw std::runtime_error("safetensors kv_head_count metadata mismatch");

    LanguageModel restored(
        32, embed, maxPos, Adam(1e-3f), 1, heads, 0, RotaryEmbedding::DefaultBase, true, kvHeads);
    restored.loadSafeTensors(path);
    if (restored.kvHeadCount() != kvHeads)
        throw std::runtime_error("LanguageModel safetensors kv_head_count roundtrip failed");

    bool rejectedLoad = false;
    try {
        mhaDefault.loadSafeTensors(path);
    } catch (const std::exception&) {
        rejectedLoad = true;
    }
    if (!rejectedLoad)
        throw std::runtime_error("loadSafeTensors should reject kv_head_count mismatch");

    std::remove(path.c_str());
    SmokeLog::result(
        "LanguageModel kv_head_count",
        "mha=%d  gqa=%d  kvRows=%d  safetensors=ok",
        heads,
        kvHeads,
        expectedKvRows);
}

void LanguageModel::runHuggingFaceExportSmokeDemo() {
    namespace fs = std::filesystem;

    const fs::path exportDir = fs::path("hf_export_smoke_out");
    const fs::path tokSrcDir = fs::path("hf_export_smoke_tok");
    fs::remove_all(exportDir);
    fs::remove_all(tokSrcDir);
    fs::create_directories(tokSrcDir);

    {
        std::ofstream out(tokSrcDir / "tokenizer.json", std::ios::binary);
        if (!out) throw std::runtime_error("HF export smoke: cannot write tokenizer.json");
        out << R"json({"version":"1.0","model":{"type":"BPE","vocab":{"a":0},"merges":[]}})json";
    }
    {
        std::ofstream out(tokSrcDir / "tokenizer_config.json", std::ios::binary);
        out << R"json({"tokenizer_class":"PreTrainedTokenizerFast"})json";
    }

    // Tiny GQA + tied + no-bias model (HF-typical).
    LanguageModel model(32, 16, 64, Adam(1e-3f), 2, 4, 32, 10000.0f, false, 2);
    model.setTieEmbeddingProjection(true);
    model.finalNorm.epsilon = 1.0e-5f;
    for (TransformerBlock& block : model.blocks) {
        block.attentionNorm.epsilon = 1.0e-5f;
        block.feedForwardNorm.epsilon = 1.0e-5f;
    }
    // Distinct values so reload parity is meaningful.
    model.tokenEmbedding.weight.data[0] = 0.125f;
    model.blocks[0].attention.keyWeight.data[0] = 0.25f;
    model.blocks[1].feedForward.downWeight.data[0] = 0.375f;
    model.finalNorm.gamma.data[0] = 0.5f;

    const Matrix logitsBefore = model.forward({ 1, 2, 3, 4 });
    model.saveHuggingFace(exportDir.string(), "llama", tokSrcDir.string(), "both");

    if (!fs::is_regular_file(exportDir / "config.json"))
        throw std::runtime_error("HF export smoke: missing config.json");
    if (!fs::is_regular_file(exportDir / "model.safetensors"))
        throw std::runtime_error("HF export smoke: missing model.safetensors");
    if (!fs::is_regular_file(exportDir / "pytorch_model.bin"))
        throw std::runtime_error("HF export smoke: missing pytorch_model.bin");
    if (!fs::is_regular_file(exportDir / "tokenizer.json")
        || !fs::is_regular_file(exportDir / "tokenizer_config.json"))
        throw std::runtime_error("HF export smoke: tokenizer files not copied");

    const HuggingFace::Config cfg = HuggingFace::loadConfig(exportDir.string());
    if (cfg.modelType != "llama" || cfg.architecture != "LlamaForCausalLM")
        throw std::runtime_error("HF export smoke: config model_type/architecture mismatch");
    if (cfg.vocabSize != 32 || cfg.hiddenSize != 16 || cfg.numKeyValueHeads != 2
        || cfg.intermediateSize != 32 || !cfg.tieWordEmbeddings || cfg.useBias)
        throw std::runtime_error("HF export smoke: config arch fields mismatch");

    const SafeTensors::File raw = SafeTensors::load((exportDir / "model.safetensors").string());
    if (raw.tensors.count("model.embed_tokens.weight") == 0
        || raw.tensors.count("model.layers.0.self_attn.k_proj.weight") == 0
        || raw.tensors.count("model.norm.weight") == 0)
        throw std::runtime_error("HF export smoke: HF tensor names missing");
    if (raw.tensors.count("lm_head.weight") != 0)
        throw std::runtime_error("HF export smoke: tied export should omit lm_head.weight");
    if (raw.tensors.count("token_embedding.weight") != 0)
        throw std::runtime_error("HF export smoke: Sentinel names must not appear in HF file");

    LanguageModel reloaded = LanguageModel::loadHuggingFace(exportDir.string(), 1e-3f);
    if (std::fabs(reloaded.tokenEmbedding.weight.data[0] - 0.125f) > 1.0e-6f
        || std::fabs(reloaded.blocks[0].attention.keyWeight.data[0] - 0.25f) > 1.0e-6f
        || std::fabs(reloaded.blocks[1].feedForward.downWeight.data[0] - 0.375f) > 1.0e-6f
        || std::fabs(reloaded.finalNorm.gamma.data[0] - 0.5f) > 1.0e-6f)
        throw std::runtime_error("HF export smoke: reloaded weights mismatch");
    if (!reloaded.tieEmbeddingProjection || reloaded.useBias() || reloaded.kvHeadCount() != 2)
        throw std::runtime_error("HF export smoke: reloaded tie/bias/kv mismatch");

    const Matrix logitsAfter = reloaded.forward({ 1, 2, 3, 4 });
    if (logitsAfter.rows != logitsBefore.rows || logitsAfter.cols != logitsBefore.cols)
        throw std::runtime_error("HF export smoke: logits shape mismatch");
    float maxDiff = 0.0f;
    for (size_t i = 0; i < logitsBefore.data.size(); ++i)
        maxDiff = (std::max)(maxDiff, std::fabs(logitsBefore.data[i] - logitsAfter.data[i]));
    if (maxDiff > 1.0e-5f)
        throw std::runtime_error("HF export smoke: logits drifted after export/import");

    // Bin-only reload: remove safetensors so loadMappedWeights must take pytorch_model.bin.
    fs::remove(exportDir / "model.safetensors");
    LanguageModel fromBin = LanguageModel::loadHuggingFace(exportDir.string(), 1e-3f);
    if (std::fabs(fromBin.tokenEmbedding.weight.data[0] - 0.125f) > 1.0e-6f
        || std::fabs(fromBin.blocks[0].attention.keyWeight.data[0] - 0.25f) > 1.0e-6f)
        throw std::runtime_error("HF export smoke: pytorch_model.bin reload mismatch");

    bool rejected = false;
    try {
        model.saveHuggingFace(exportDir.string(), "gpt2");
    } catch (const std::exception&) {
        rejected = true;
    }
    if (!rejected)
        throw std::runtime_error("HF export smoke: should reject unsupported model_type");

    fs::remove_all(exportDir);
    fs::remove_all(tokSrcDir);
    SmokeLog::result(
        "LanguageModel saveHuggingFace",
        "safe+bin+tokenizer=ok  reload=ok  binOnly=ok  logitsDiff=%.2e  reject=model_type",
        maxDiff);
}

void LanguageModel::runHuggingFaceImportSmokeDemo() {
    namespace fs = std::filesystem;

    const fs::path dir = fs::path("hf_import_smoke_dir");
    fs::create_directories(dir);

    const std::string configJson = R"json({
  "architectures": ["LlamaForCausalLM"],
  "model_type": "llama",
  "vocab_size": 32,
  "hidden_size": 16,
  "intermediate_size": 32,
  "num_hidden_layers": 2,
  "num_attention_heads": 4,
  "num_key_value_heads": 2,
  "max_position_embeddings": 64,
  "rms_norm_eps": 1e-5,
  "rope_theta": 10000.0,
  "tie_word_embeddings": true,
  "attention_bias": false,
  "mlp_bias": false
})json";
    {
        std::ofstream out(dir / "config.json", std::ios::binary);
        if (!out) throw std::runtime_error("HF import smoke: cannot write config.json");
        out << configJson;
    }

    auto filled = [](size_t rows, size_t cols, float value) {
        return Matrix(rows, cols, value);
    };

    SafeTensors::File weights;
    SafeTensors::putMatrix(weights, "model.embed_tokens.weight", filled(32, 16, 0.11f));
    SafeTensors::putMatrix(weights, "model.layers.0.input_layernorm.weight", filled(16, 1, 0.21f));
    SafeTensors::putMatrix(weights, "model.layers.0.self_attn.q_proj.weight", filled(16, 16, 0.31f));
    SafeTensors::putMatrix(weights, "model.layers.0.self_attn.k_proj.weight", filled(8, 16, 0.32f));
    SafeTensors::putMatrix(weights, "model.layers.0.self_attn.v_proj.weight", filled(8, 16, 0.33f));
    SafeTensors::putMatrix(weights, "model.layers.0.self_attn.o_proj.weight", filled(16, 16, 0.34f));
    SafeTensors::putMatrix(weights, "model.layers.0.post_attention_layernorm.weight", filled(16, 1, 0.22f));
    SafeTensors::putMatrix(weights, "model.layers.0.mlp.gate_proj.weight", filled(32, 16, 0.41f));
    SafeTensors::putMatrix(weights, "model.layers.0.mlp.up_proj.weight", filled(32, 16, 0.42f));
    SafeTensors::putMatrix(weights, "model.layers.0.mlp.down_proj.weight", filled(16, 32, 0.43f));
    SafeTensors::putMatrix(weights, "model.layers.1.input_layernorm.weight", filled(16, 1, 0.51f));
    SafeTensors::putMatrix(weights, "model.layers.1.self_attn.q_proj.weight", filled(16, 16, 0.61f));
    SafeTensors::putMatrix(weights, "model.layers.1.self_attn.k_proj.weight", filled(8, 16, 0.62f));
    SafeTensors::putMatrix(weights, "model.layers.1.self_attn.v_proj.weight", filled(8, 16, 0.63f));
    SafeTensors::putMatrix(weights, "model.layers.1.self_attn.o_proj.weight", filled(16, 16, 0.64f));
    SafeTensors::putMatrix(weights, "model.layers.1.post_attention_layernorm.weight", filled(16, 1, 0.52f));
    SafeTensors::putMatrix(weights, "model.layers.1.mlp.gate_proj.weight", filled(32, 16, 0.71f));
    SafeTensors::putMatrix(weights, "model.layers.1.mlp.up_proj.weight", filled(32, 16, 0.72f));
    SafeTensors::putMatrix(weights, "model.layers.1.mlp.down_proj.weight", filled(16, 32, 0.73f));
    SafeTensors::putMatrix(weights, "model.norm.weight", filled(16, 1, 0.91f));
    SafeTensors::save((dir / "model.safetensors").string(), weights);

    LanguageModel model = LanguageModel::loadHuggingFace(dir.string(), 1e-3f);
    if (model.tokenEmbedding.vocabSize() != 32 || model.tokenEmbedding.embeddingDim() != 16)
        throw std::runtime_error("HF import smoke: vocab/embed mismatch");
    if (static_cast<int>(model.blocks.size()) != 2 || model.kvHeadCount() != 2)
        throw std::runtime_error("HF import smoke: blocks/kv heads mismatch");
    if (model.intermediateSize() != 32 || !model.tieEmbeddingProjection || model.useBias())
        throw std::runtime_error("HF import smoke: intermediate/tie/bias mismatch");
    if (std::fabs(model.tokenEmbedding.weight.data[0] - 0.11f) > 1.0e-6f)
        throw std::runtime_error("HF import smoke: embed weight not loaded");
    if (std::fabs(model.blocks[0].attention.keyWeight.data[0] - 0.32f) > 1.0e-6f)
        throw std::runtime_error("HF import smoke: GQA k weight not loaded");

    const Matrix logits = model.forward({ 1, 2, 3, 4 });
    if (logits.rows != 32 || logits.cols != 4)
        throw std::runtime_error("HF import smoke: forward shape mismatch");
    for (float v : logits.data) {
        if (!std::isfinite(v))
            throw std::runtime_error("HF import smoke: non-finite logits");
    }

    bool rejected = false;
    try {
        std::ofstream bad(dir / "config.json", std::ios::binary);
        bad << R"json({"model_type":"gpt2","vocab_size":32,"hidden_size":16,"intermediate_size":32,"num_hidden_layers":2,"num_attention_heads":4,"max_position_embeddings":64})json";
        bad.close();
        (void)LanguageModel::loadHuggingFace(dir.string());
    } catch (const std::exception&) {
        rejected = true;
    }
    if (!rejected)
        throw std::runtime_error("HF import smoke: should reject unsupported model_type");

    fs::remove_all(dir);
    SmokeLog::result(
        "LanguageModel loadHuggingFace",
        "arch=ok  GQA=2  tie=1  forward=ok  reject=model_type");
}

void LanguageModel::runHuggingFaceRoundtripSmokeDemo() {
    namespace fs = std::filesystem;

    const fs::path dir = fs::path("hf_roundtrip_smoke_dir");
    fs::remove_all(dir);
    fs::create_directories(dir);

    {
        std::ofstream out(dir / "config.json", std::ios::binary);
        if (!out) throw std::runtime_error("HF roundtrip smoke: cannot write config.json");
        out << R"json({
  "architectures": ["LlamaForCausalLM"],
  "model_type": "llama",
  "vocab_size": 32,
  "hidden_size": 16,
  "intermediate_size": 32,
  "num_hidden_layers": 2,
  "num_attention_heads": 4,
  "num_key_value_heads": 2,
  "max_position_embeddings": 64,
  "rms_norm_eps": 1e-5,
  "rope_theta": 10000.0,
  "tie_word_embeddings": true,
  "attention_bias": false,
  "mlp_bias": false
})json";
    }

    {
        std::ofstream out(dir / "tokenizer.json", std::ios::binary);
        if (!out) throw std::runtime_error("HF roundtrip smoke: cannot write tokenizer.json");
        // ByteLevel raw BPE; ids stay within model vocab_size=32
        // (ASCII space is not identity under GPT-2 bytes_to_unicode — avoid spaces in the fixture.)
        out << R"json({
  "version": "1.0",
  "added_tokens": [
    {"id": 30, "content": "<bos>", "special": true},
    {"id": 31, "content": "<eos>", "special": true}
  ],
  "pre_tokenizer": {
    "type": "ByteLevel",
    "add_prefix_space": false,
    "trim_offsets": true,
    "use_regex": false
  },
  "decoder": {
    "type": "ByteLevel",
    "add_prefix_space": false,
    "trim_offsets": true,
    "use_regex": false
  },
  "model": {
    "type": "BPE",
    "unk_token": null,
    "ignore_merges": false,
    "vocab": {
      "h": 0,
      "i": 1,
      "a": 2,
      "b": 3,
      "hi": 4,
      "ab": 5,
      "hiab": 6,
      "<bos>": 30,
      "<eos>": 31
    },
    "merges": [
      "h i",
      "a b",
      "hi ab"
    ]
  }
})json";
    }

    auto filled = [](size_t rows, size_t cols, float value) {
        return Matrix(rows, cols, value);
    };
    // Mild magnitudes so one host Adam step stays finite.
    SafeTensors::File weights;
    SafeTensors::putMatrix(weights, "model.embed_tokens.weight", filled(32, 16, 0.02f));
    SafeTensors::putMatrix(weights, "model.layers.0.input_layernorm.weight", filled(16, 1, 1.0f));
    SafeTensors::putMatrix(weights, "model.layers.0.self_attn.q_proj.weight", filled(16, 16, 0.01f));
    SafeTensors::putMatrix(weights, "model.layers.0.self_attn.k_proj.weight", filled(8, 16, 0.01f));
    SafeTensors::putMatrix(weights, "model.layers.0.self_attn.v_proj.weight", filled(8, 16, 0.01f));
    SafeTensors::putMatrix(weights, "model.layers.0.self_attn.o_proj.weight", filled(16, 16, 0.01f));
    SafeTensors::putMatrix(weights, "model.layers.0.post_attention_layernorm.weight", filled(16, 1, 1.0f));
    SafeTensors::putMatrix(weights, "model.layers.0.mlp.gate_proj.weight", filled(32, 16, 0.01f));
    SafeTensors::putMatrix(weights, "model.layers.0.mlp.up_proj.weight", filled(32, 16, 0.01f));
    SafeTensors::putMatrix(weights, "model.layers.0.mlp.down_proj.weight", filled(16, 32, 0.01f));
    SafeTensors::putMatrix(weights, "model.layers.1.input_layernorm.weight", filled(16, 1, 1.0f));
    SafeTensors::putMatrix(weights, "model.layers.1.self_attn.q_proj.weight", filled(16, 16, 0.01f));
    SafeTensors::putMatrix(weights, "model.layers.1.self_attn.k_proj.weight", filled(8, 16, 0.01f));
    SafeTensors::putMatrix(weights, "model.layers.1.self_attn.v_proj.weight", filled(8, 16, 0.01f));
    SafeTensors::putMatrix(weights, "model.layers.1.self_attn.o_proj.weight", filled(16, 16, 0.01f));
    SafeTensors::putMatrix(weights, "model.layers.1.post_attention_layernorm.weight", filled(16, 1, 1.0f));
    SafeTensors::putMatrix(weights, "model.layers.1.mlp.gate_proj.weight", filled(32, 16, 0.01f));
    SafeTensors::putMatrix(weights, "model.layers.1.mlp.up_proj.weight", filled(32, 16, 0.01f));
    SafeTensors::putMatrix(weights, "model.layers.1.mlp.down_proj.weight", filled(16, 32, 0.01f));
    SafeTensors::putMatrix(weights, "model.norm.weight", filled(16, 1, 1.0f));
    SafeTensors::save((dir / "model.safetensors").string(), weights);

    LanguageModel model = LanguageModel::loadHuggingFace(dir.string(), 1e-3f);
    HuggingFace::Tokenizer tokenizer = HuggingFace::Tokenizer::load(dir.string());
    if (!tokenizer.isLoaded() || tokenizer.bosTokenId() != 30)
        throw std::runtime_error("HF roundtrip smoke: tokenizer load failed");

    const std::vector<int> encoded = tokenizer.encode("hiab", true);
    if (encoded.size() < 2)
        throw std::runtime_error("HF roundtrip smoke: encode too short");
    for (int id : encoded) {
        if (id < 0 || id >= model.tokenEmbedding.vocabSize())
            throw std::runtime_error("HF roundtrip smoke: token id out of model vocab");
    }

    const std::string decoded = tokenizer.decode(encoded, true);
    if (decoded != "hiab")
        throw std::runtime_error("HF roundtrip smoke: decode mismatch got '" + decoded + "'");

    Matrix before = model.forward(encoded);
    if (before.rows != static_cast<size_t>(model.tokenEmbedding.vocabSize())
        || before.cols != encoded.size())
        throw std::runtime_error("HF roundtrip smoke: forward shape mismatch");
    for (float v : before.data) {
        if (!std::isfinite(v))
            throw std::runtime_error("HF roundtrip smoke: non-finite logits before train");
    }

    LanguageModelDataset dataset;
    dataset.vocabularySize = model.tokenEmbedding.vocabSize();
    dataset.examples.push_back(
        LanguageModelDataset::fromTokenIds(encoded, dataset.vocabularySize, /*buildOneHot=*/false));

    const float lossBefore = model.averageLoss(dataset);
    if (!std::isfinite(lossBefore))
        throw std::runtime_error("HF roundtrip smoke: non-finite loss before train");

    model.train(dataset, /*epochs=*/1, /*logEveryEpochs=*/0);

    const float lossAfter = model.averageLoss(dataset);
    if (!std::isfinite(lossAfter))
        throw std::runtime_error("HF roundtrip smoke: non-finite loss after train");

    Matrix after = model.forward(encoded);
    for (float v : after.data) {
        if (!std::isfinite(v))
            throw std::runtime_error("HF roundtrip smoke: non-finite logits after train");
    }

    const std::vector<int> generated = model.generate(encoded, /*newTokenCount=*/2, /*temperature=*/0.0f);
    if (generated.size() != encoded.size() + 2)
        throw std::runtime_error("HF roundtrip smoke: generate length mismatch");
    for (size_t i = encoded.size(); i < generated.size(); ++i) {
        const int id = generated[i];
        if (id < 0 || id >= model.tokenEmbedding.vocabSize())
            throw std::runtime_error("HF roundtrip smoke: generated id out of range");
    }

    fs::remove_all(dir);
    SmokeLog::result(
        "LanguageModel HF roundtrip",
        "import+encode+train=ok  loss=%.3f→%.3f  generate=2  finite=ok",
        lossBefore,
        lossAfter);
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
