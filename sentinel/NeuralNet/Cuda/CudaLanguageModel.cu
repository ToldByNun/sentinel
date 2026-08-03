#include "CudaLanguageModel.hpp"

#include "CudaAdam.hpp"
#include "CudaAmp.hpp"
#include "CudaOps.hpp"
#include "../Utils/SmokeLog.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cuda_runtime.h>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "../Optimizers/Adam.hpp"

CudaLanguageModel::CudaLanguageModel()
    : maximumPositionCount(0),
      tieEmbeddingProjection(true),
      maxPackedColumns(8192),
      maxPackedColumnsManual(false),
      logitChunkRows(2048),
      gradientAccumulationSteps(4),
      activationCheckpointMode(ActivationCheckpointMode::Selective),
      preferTrainGraph(true),
      preferMuon(false),
      adam(0.001f),
      muon(0.001f),
      trainStateReady(false),
      trainStream(nullptr),
      trainGraph(nullptr),
      trainGraphExec(nullptr),
      trainGraphSegmentLength(0),
      trainGraphExampleCount(0),
      trainGraphWarmups(0),
      trainGraphSeenSegmentLength(0),
      trainGraphSeenExampleCount(0) {}

bool CudaLanguageModel::activationCheckpointingActive() const {
    return this->activationCheckpointMode != ActivationCheckpointMode::Off;
}

const char* CudaLanguageModel::activationCheckpointModeName(ActivationCheckpointMode mode) {
    switch (mode) {
    case ActivationCheckpointMode::Off: return "off";
    case ActivationCheckpointMode::Full: return "full";
    case ActivationCheckpointMode::Selective: return "selective";
    }
    return "unknown";
}

void CudaLanguageModel::setActivationCheckpointMode(ActivationCheckpointMode mode) {
    this->activationCheckpointMode = mode;
    if (mode == ActivationCheckpointMode::Off) {
        this->releaseActivationCheckpoints();
        return;
    }
    if (!this->tokenEmbeddingWeight.empty())
        this->ensureTrainWorkspaces();
}

void CudaLanguageModel::setPreferMuon(bool enabled) {
    if (this->preferMuon == enabled) return;
    this->preferMuon = enabled;
    this->trainStateReady = false;
    if (!enabled) {
        for (CudaTransformerBlockMuonStates& blockStates : this->blockMuonStates)
            blockStates.free();
        this->blockMuonStates.clear();
    }
}

void CudaLanguageModel::releaseTrainGraph() {
    if (this->trainGraphExec != nullptr) {
        cudaGraphExecDestroy(this->trainGraphExec);
        this->trainGraphExec = nullptr;
    }
    if (this->trainGraph != nullptr) {
        cudaGraphDestroy(this->trainGraph);
        this->trainGraph = nullptr;
    }
    this->trainGraphSegmentLength = 0;
    this->trainGraphExampleCount = 0;
}

void CudaLanguageModel::ensureTrainStream() {
    if (this->trainStream != nullptr) return;
    CudaMatmul::throwIfCudaFailed(cudaStreamCreateWithFlags(&this->trainStream, cudaStreamNonBlocking), "cudaStreamCreate trainStream");
}

void CudaLanguageModel::runPackedTrainDevice(size_t tokenCount, int segmentLength, int exampleCount, CudaLanguageModelGradients& gradients) {
    this->forwardTrunkFromDevice(tokenCount, segmentLength);

    if (this->epochLossSum.rows != 1 || this->epochLossSum.cols != 1)
        this->epochLossSum.ensureSize(1, 1);
    this->accumulateChunkedProjection(tokenCount, segmentLength, exampleCount, gradients);

    this->finalNorm.backward(this->hiddenGradient, this->normInputGradientScratch, this->finalNormGammaGradient);
    CudaOps::addInPlace(gradients.finalNormGamma, this->finalNormGammaGradient);
    std::swap(this->hiddenGradient, this->normInputGradientScratch);

    const bool hostLarge = CudaAdam::preferHostGradients;
    if (hostLarge && this->hostBlockAdamStates.size() != this->blocks.size())
        this->hostBlockAdamStates.resize(this->blocks.size());

    for (int blockIndex = static_cast<int>(this->blocks.size()) - 1; blockIndex >= 0; --blockIndex) {
        CudaTransformerBlock& block = this->blocks[static_cast<size_t>(blockIndex)];
        CudaTransformerBlockHostWeightGrads hostGrads{};
        CudaTransformerBlockHostWeightGrads* hostGradsPtr = nullptr;
        if (hostLarge) {
            CudaTransformerBlockHostAdamStates& hosts = this->hostBlockAdamStates[static_cast<size_t>(blockIndex)];
            hostGrads.queryWeight = &hosts.queryWeightGrad;
            hostGrads.keyWeight = &hosts.keyWeightGrad;
            hostGrads.valueWeight = &hosts.valueWeightGrad;
            hostGrads.attentionOutputWeight = &hosts.attentionOutputWeightGrad;
            hostGrads.feedForwardGateWeight = &hosts.feedForwardGateWeightGrad;
            hostGrads.feedForwardUpWeight = &hosts.feedForwardUpWeightGrad;
            hostGrads.feedForwardDownWeight = &hosts.feedForwardDownWeightGrad;
            hostGradsPtr = &hostGrads;
        }

        if (this->activationCheckpointMode == ActivationCheckpointMode::Selective) {
            const CudaMatrix* blockInput = nullptr;
            if (this->useHalfActivationCheckpoints()) {
                CudaAmp::castToFloat(
                    this->blockInputCheckpointsHalf[static_cast<size_t>(blockIndex)],
                    this->checkpointRestoreScratch);
                this->checkpointRestoreScratch.cols = this->hiddenGradient.cols;
                blockInput = &this->checkpointRestoreScratch;
            } else {
                blockInput = &this->blockInputCheckpoints[static_cast<size_t>(blockIndex)];
            }
            block.backwardSelective(
                this->hiddenGradient,
                this->blockInputGradientScratch,
                gradients.blocks[static_cast<size_t>(blockIndex)],
                *blockInput,
                segmentLength,
                hostGradsPtr);
        } else if (this->activationCheckpointMode == ActivationCheckpointMode::Full) {
            if (this->useHalfActivationCheckpoints()) {
                CudaAmp::castToFloat(
                    this->blockInputCheckpointsHalf[static_cast<size_t>(blockIndex)],
                    this->checkpointRestoreScratch);
                this->checkpointRestoreScratch.cols = this->hiddenGradient.cols;
                block.forward(
                    this->checkpointRestoreScratch,
                    this->normalized,
                    segmentLength);
            } else {
                block.forward(
                    this->blockInputCheckpoints[static_cast<size_t>(blockIndex)],
                    this->normalized,
                    segmentLength);
            }
            block.backward(this->hiddenGradient, this->blockInputGradientScratch, gradients.blocks[static_cast<size_t>(blockIndex)], hostGradsPtr);
        } else {
            block.backward(this->hiddenGradient, this->blockInputGradientScratch, gradients.blocks[static_cast<size_t>(blockIndex)], hostGradsPtr);
        }

        std::swap(this->hiddenGradient, this->blockInputGradientScratch);
    }

    CudaOps::embeddingScatterAddInto(gradients.tokenEmbedding, this->tokenIdsBuffer, tokenCount, this->hiddenGradient);
}

bool CudaLanguageModel::tryLaunchTrainGraph(int segmentLength, int exampleCount) {
    if (!this->preferTrainGraph || this->activationCheckpointingActive()) return false;
    if (this->trainGraphExec == nullptr) return false;
    if (segmentLength != this->trainGraphSegmentLength || exampleCount != this->trainGraphExampleCount) return false;

    CudaMatmul::throwIfCudaFailed(cudaGraphLaunch(this->trainGraphExec, this->trainStream), "cudaGraphLaunch train microstep");
    CudaMatmul::throwIfCudaFailed(cudaStreamSynchronize(this->trainStream), "cudaStreamSynchronize train graph");
    return true;
}

bool CudaLanguageModel::captureTrainGraph(size_t tokenCount, int segmentLength, int exampleCount, CudaLanguageModelGradients& gradients) {
    this->ensureTrainStream();
    this->releaseTrainGraph();

    CudaMatmul::throwIfCudaFailed(
        cudaStreamBeginCapture(this->trainStream, cudaStreamCaptureModeGlobal),
        "cudaStreamBeginCapture train microstep");

    const cudaStream_t previous = CudaMatmul::setActiveStream(this->trainStream);
    bool captureOk = false;
    try {
        this->runPackedTrainDevice(tokenCount, segmentLength, exampleCount, gradients);
        CudaMatmul::throwIfCudaFailed(cudaStreamEndCapture(this->trainStream, &this->trainGraph), "cudaStreamEndCapture train microstep");
        CudaMatmul::throwIfCudaFailed(
            cudaGraphInstantiate(&this->trainGraphExec, this->trainGraph, nullptr, nullptr, 0),
            "cudaGraphInstantiate train microstep");
        this->trainGraphSegmentLength = segmentLength;
        this->trainGraphExampleCount = exampleCount;
        captureOk = true;
    } catch (...) {
        CudaMatmul::setActiveStream(previous);
        // Capture may be invalidated; drain error and fall back.
        cudaGetLastError();
        this->releaseTrainGraph();
        throw;
    }
    CudaMatmul::setActiveStream(previous);
    return captureOk;
}

void CudaTransformerBlockAdamStates::ensureFrom(const CudaTransformerBlock& block) {
    // moments stay empty until first Adam update (lazy allocation for low VRAM before train)
    (void)block;
}

void CudaTransformerBlockAdamStates::free() {
    this->queryWeight.free();
    this->keyWeight.free();
    this->valueWeight.free();
    this->attentionOutputWeight.free();
    this->attentionNormGamma.free();
    this->feedForwardNormGamma.free();
    this->feedForwardGateWeight.free();
    this->feedForwardGateBias.free();
    this->feedForwardUpWeight.free();
    this->feedForwardUpBias.free();
    this->feedForwardDownWeight.free();
    this->feedForwardDownBias.free();
}

void CudaTransformerBlockAdamStates::freeMuonManagedWeights() {
    this->queryWeight.free();
    this->keyWeight.free();
    this->valueWeight.free();
    this->attentionOutputWeight.free();
    this->feedForwardGateWeight.free();
    this->feedForwardUpWeight.free();
    this->feedForwardDownWeight.free();
}

void CudaTransformerBlockMuonStates::ensureFrom(const CudaTransformerBlock& block) {
    if (block.attention.qkvWeight.empty() || block.feedForward.gateUpWeight.empty())
        throw std::logic_error("CudaTransformerBlockMuonStates::ensureFrom requires fused QKV/gateUp weights");
    this->qkvWeight.ensure(block.attention.qkvWeight);
    this->attentionOutputWeight.ensure(block.attention.outputWeight);
    this->feedForwardGateUpWeight.ensure(block.feedForward.gateUpWeight);
    this->feedForwardDownWeight.ensure(block.feedForward.downWeight);
}

void CudaTransformerBlockMuonStates::free() {
    this->qkvWeight.free();
    this->attentionOutputWeight.free();
    this->feedForwardGateUpWeight.free();
    this->feedForwardDownWeight.free();
}

void CudaLanguageModelGradients::ensureFrom(const CudaLanguageModel& model) {
    const bool largeWeightsOnHost = CudaAdam::preferHostGradients;
    this->tokenEmbedding.ensureSize(model.tokenEmbeddingWeight.rows, model.tokenEmbeddingWeight.cols);
    this->blocks.resize(model.blocks.size());
    for (size_t blockIndex = 0; blockIndex < model.blocks.size(); ++blockIndex)
        this->blocks[blockIndex].ensureFrom(model.blocks[blockIndex], largeWeightsOnHost);
    this->finalNormGamma.ensureSize(model.finalNorm.gamma.rows, model.finalNorm.gamma.cols);
    if (model.tieEmbeddingProjection)
        this->projectionWeight.free();
    else
        this->projectionWeight.ensureSize(model.projectionWeight.rows, model.projectionWeight.cols);
    this->projectionBias.ensureSize(model.projectionBias.rows, model.projectionBias.cols);
}

void CudaLanguageModelGradients::zeroInPlace() {
    CudaOps::zeroInPlace(this->tokenEmbedding);
    for (CudaTransformerBlockGradients& block : this->blocks)
        block.zeroInPlace();
    CudaOps::zeroInPlace(this->finalNormGamma);
    if (!this->projectionWeight.empty())
        CudaOps::zeroInPlace(this->projectionWeight);
    CudaOps::zeroInPlace(this->projectionBias);
}

void CudaLanguageModelGradients::zeroInPlaceExceptEmbedding() {
    for (CudaTransformerBlockGradients& block : this->blocks)
        block.zeroInPlace();
    CudaOps::zeroInPlace(this->finalNormGamma);
    if (!this->projectionWeight.empty())
        CudaOps::zeroInPlace(this->projectionWeight);
    CudaOps::zeroInPlace(this->projectionBias);
}

void CudaLanguageModelGradients::addInPlace(const CudaLanguageModelGradients& other) {
    CudaOps::addInPlace(this->tokenEmbedding, other.tokenEmbedding);
    for (size_t blockIndex = 0; blockIndex < this->blocks.size(); ++blockIndex)
        this->blocks[blockIndex].addInPlace(other.blocks[blockIndex]);
    CudaOps::addInPlace(this->finalNormGamma, other.finalNormGamma);
    if (!this->projectionWeight.empty() && !other.projectionWeight.empty())
        CudaOps::addInPlace(this->projectionWeight, other.projectionWeight);
    CudaOps::addInPlace(this->projectionBias, other.projectionBias);
}

void CudaLanguageModelGradients::scaleInPlace(float scalar) {
    CudaOps::scaleInPlace(this->tokenEmbedding, scalar);
    for (CudaTransformerBlockGradients& block : this->blocks)
        block.scaleInPlace(scalar);
    CudaOps::scaleInPlace(this->finalNormGamma, scalar);
    if (!this->projectionWeight.empty())
        CudaOps::scaleInPlace(this->projectionWeight, scalar);
    CudaOps::scaleInPlace(this->projectionBias, scalar);
}

bool CudaLanguageModel::useHalfActivationCheckpoints() const {
    return this->activationCheckpointingActive() && CudaAmp::preferMixedPrecision;
}

CudaMatrix& CudaLanguageModel::lmHeadWeight() {
    return this->tieEmbeddingProjection ? this->tokenEmbeddingWeight : this->projectionWeight;
}

const CudaMatrix& CudaLanguageModel::lmHeadWeight() const {
    return this->tieEmbeddingProjection ? this->tokenEmbeddingWeight : this->projectionWeight;
}

void CudaLanguageModel::ensureTrainState() {
    if (this->trainStateReady) return;
    if (this->tokenEmbeddingWeight.empty()) throw std::logic_error("CudaLanguageModel::ensureTrainState weights not uploaded");

    this->trainGradients.ensureFrom(*this);
    this->trainGradients.zeroInPlace();
    this->muon.learningRate = this->adam.learningRate;

    if (CudaAdam::preferCpuOffload) {
        // moments live on host; free any leftover device Adam buffers
        this->tokenEmbeddingState.free();
        this->finalNormGammaState.free();
        this->projectionWeightState.free();
        this->projectionBiasState.free();
        for (CudaTransformerBlockAdamStates& blockStates : this->blockAdamStates)
            blockStates.free();
        this->blockAdamStates.clear();
        this->hostBlockAdamStates.resize(this->blocks.size());
    } else {
        this->hostBlockAdamStates.clear();
        this->hostTokenEmbeddingState = AdamState{};
        this->hostFinalNormGammaState = AdamState{};
        this->hostProjectionWeightState = AdamState{};
        this->hostProjectionBiasState = AdamState{};
        // Adam moments allocated lazily on first applyGradients
        this->blockAdamStates.resize(this->blocks.size());
        for (size_t blockIndex = 0; blockIndex < this->blocks.size(); ++blockIndex)
            this->blockAdamStates[blockIndex].ensureFrom(this->blocks[blockIndex]);
    }

    if (this->preferMuon) {
        if (CudaAdam::preferFp16GpuWeights)
            throw std::logic_error("CudaLanguageModel::ensureTrainState Muon incompatible with preferFp16GpuWeights");
        this->blockMuonStates.resize(this->blocks.size());
        for (size_t blockIndex = 0; blockIndex < this->blocks.size(); ++blockIndex) {
            this->blocks[blockIndex].attention.syncFusedQkvWeight();
            this->blocks[blockIndex].feedForward.syncFusedGateUpWeight();
            this->blockMuonStates[blockIndex].ensureFrom(this->blocks[blockIndex]);
            if (blockIndex < this->blockAdamStates.size())
                this->blockAdamStates[blockIndex].freeMuonManagedWeights();
        }
    } else {
        for (CudaTransformerBlockMuonStates& blockStates : this->blockMuonStates)
            blockStates.free();
        this->blockMuonStates.clear();
    }

    this->finalNormGammaGradient.ensureSize(this->finalNorm.gamma.rows, this->finalNorm.gamma.cols);

    if (CudaAdam::preferCpuOffload && CudaAdam::preferFp16GpuWeights)
        this->materializeFp16GpuWorkingWeights();

    if (CudaAdam::preferHostGradients) {
        if (this->hostBlockAdamStates.size() != this->blocks.size())
            this->hostBlockAdamStates.resize(this->blocks.size());
        for (size_t blockIndex = 0; blockIndex < this->blocks.size(); ++blockIndex) {
            const CudaTransformerBlock& block = this->blocks[blockIndex];
            CudaTransformerBlockHostAdamStates& hosts = this->hostBlockAdamStates[blockIndex];
            auto ensureHostGrad = [](Matrix& grad, size_t rows, size_t cols) {
                grad.ensureSize(rows, cols);
                grad.fill(0.0f);
            };
            ensureHostGrad(hosts.queryWeightGrad, block.attention.queryWeight.rows, block.attention.queryWeight.cols);
            ensureHostGrad(hosts.keyWeightGrad, block.attention.keyWeight.rows, block.attention.keyWeight.cols);
            ensureHostGrad(hosts.valueWeightGrad, block.attention.valueWeight.rows, block.attention.valueWeight.cols);
            ensureHostGrad(hosts.attentionOutputWeightGrad, block.attention.outputWeight.rows, block.attention.outputWeight.cols);
            ensureHostGrad(hosts.feedForwardGateWeightGrad, block.feedForward.gateWeight.rows, block.feedForward.gateWeight.cols);
            ensureHostGrad(hosts.feedForwardUpWeightGrad, block.feedForward.upWeight.rows, block.feedForward.upWeight.cols);
            ensureHostGrad(hosts.feedForwardDownWeightGrad, block.feedForward.downWeight.rows, block.feedForward.downWeight.cols);
        }
    }

    this->ensureTrainWorkspaces();
    this->trainStateReady = true;
}

void CudaLanguageModel::materializeFp16GpuWorkingWeights() {
    if (!CudaAmp::preferMixedPrecision)
        throw std::logic_error("CudaLanguageModel::materializeFp16GpuWorkingWeights requires AMP");
    if (!CudaAdam::preferCpuOffload || !CudaAdam::preferFp16GpuWeights)
        throw std::logic_error("CudaLanguageModel::materializeFp16GpuWorkingWeights requires cpu offload + fp16 weights");

    auto seedMaster = [](CudaMatrix& deviceWeight, Matrix& hostMaster) {
        if (deviceWeight.empty()) throw std::invalid_argument("materializeFp16GpuWorkingWeights empty weight");
        if (!deviceWeight.hasDeviceStorage())
            throw std::logic_error("materializeFp16GpuWorkingWeights weight missing FP32 storage");
        hostMaster = deviceWeight.download();
    };

    auto bind2d = [&seedMaster](CudaMatrix& deviceWeight, Matrix& hostMaster) {
        seedMaster(deviceWeight, hostMaster);
        deviceWeight.ampWeightSlot = -1;
        CudaAmp::bindFp16WorkingWeight(deviceWeight);
    };

    /// <summary>Adam-only tensors: host master, free GPU (fused mirrors carry the GEMM weights)</summary>
    auto hostOnly2d = [&seedMaster](CudaMatrix& deviceWeight, Matrix& hostMaster) {
        seedMaster(deviceWeight, hostMaster);
        deviceWeight.ampWeightSlot = -1;
        deviceWeight.releaseDeviceKeepShape();
    };

    auto seedFp32Keep = [&seedMaster](CudaMatrix& deviceWeight, Matrix& hostMaster) {
        seedMaster(deviceWeight, hostMaster);
    };

    // Drop sticky FP32 AMP mirrors; working-weight slots own the table after this.
    CudaAmp::clearMasterWeights();

    // LM head / embedding stay FP32 on GPU (CE path uses raw float* rows). Other 2D weights → FP16.
    seedFp32Keep(this->tokenEmbeddingWeight, this->hostTokenEmbeddingMaster);
    seedFp32Keep(this->finalNorm.gamma, this->hostFinalNormGammaMaster);
    seedFp32Keep(this->projectionBias, this->hostProjectionBiasMaster);
    if (!this->tieEmbeddingProjection)
        seedFp32Keep(this->projectionWeight, this->hostProjectionWeightMaster);

    if (this->hostBlockAdamStates.size() != this->blocks.size())
        this->hostBlockAdamStates.resize(this->blocks.size());

    for (size_t blockIndex = 0; blockIndex < this->blocks.size(); ++blockIndex) {
        CudaTransformerBlock& block = this->blocks[blockIndex];
        CudaTransformerBlockHostAdamStates& hosts = this->hostBlockAdamStates[blockIndex];

        hostOnly2d(block.attention.queryWeight, hosts.queryWeightMaster);
        hostOnly2d(block.attention.keyWeight, hosts.keyWeightMaster);
        hostOnly2d(block.attention.valueWeight, hosts.valueWeightMaster);
        bind2d(block.attention.outputWeight, hosts.attentionOutputWeightMaster);
        seedFp32Keep(block.attentionNorm.gamma, hosts.attentionNormGammaMaster);
        seedFp32Keep(block.feedForwardNorm.gamma, hosts.feedForwardNormGammaMaster);
        hostOnly2d(block.feedForward.gateWeight, hosts.feedForwardGateWeightMaster);
        hostOnly2d(block.feedForward.upWeight, hosts.feedForwardUpWeightMaster);
        bind2d(block.feedForward.downWeight, hosts.feedForwardDownWeightMaster);
        seedFp32Keep(block.feedForward.gateBias, hosts.feedForwardGateBiasMaster);
        seedFp32Keep(block.feedForward.upBias, hosts.feedForwardUpBiasMaster);
        seedFp32Keep(block.feedForward.downBias, hosts.feedForwardDownBiasMaster);

        // Rebuild fused mirrors on host then bind as FP16 working (no FP32 fused resident).
        Matrix qkvHost(block.attention.queryWeight.rows * 3ull, block.attention.queryWeight.cols, 0.0f);
        const size_t slice = hosts.queryWeightMaster.data.size();
        std::memcpy(qkvHost.data.data(), hosts.queryWeightMaster.data.data(), slice * sizeof(float));
        std::memcpy(qkvHost.data.data() + slice, hosts.keyWeightMaster.data.data(), slice * sizeof(float));
        std::memcpy(qkvHost.data.data() + 2 * slice, hosts.valueWeightMaster.data.data(), slice * sizeof(float));
        block.attention.qkvWeight.ampWeightSlot = -1;
        block.attention.qkvWeight.ensureSize(qkvHost.rows, qkvHost.cols);
        block.attention.qkvWeight.upload(qkvHost);
        CudaAmp::bindFp16WorkingWeight(block.attention.qkvWeight);

        Matrix gateUpHost(block.feedForward.gateWeight.rows + block.feedForward.upWeight.rows, block.feedForward.gateWeight.cols, 0.0f);
        const size_t gateSlice = hosts.feedForwardGateWeightMaster.data.size();
        std::memcpy(gateUpHost.data.data(), hosts.feedForwardGateWeightMaster.data.data(), gateSlice * sizeof(float));
        std::memcpy(gateUpHost.data.data() + gateSlice, hosts.feedForwardUpWeightMaster.data.data(), hosts.feedForwardUpWeightMaster.data.size() * sizeof(float));
        block.feedForward.gateUpWeight.ampWeightSlot = -1;
        block.feedForward.gateUpWeight.ensureSize(gateUpHost.rows, gateUpHost.cols);
        block.feedForward.gateUpWeight.upload(gateUpHost);
        CudaAmp::bindFp16WorkingWeight(block.feedForward.gateUpWeight);

        if (!block.feedForward.gateBias.empty() && !block.feedForward.upBias.empty()) {
            block.feedForward.gateUpBias.ensureSize(block.feedForward.gateBias.rows + block.feedForward.upBias.rows, 1);
            const size_t gateBiasBytes = block.feedForward.gateBias.byteCount();
            CudaMatmul::memcpyDevice(block.feedForward.gateUpBias.buffer.deviceData, block.feedForward.gateBias.buffer.deviceData, gateBiasBytes);
            CudaMatmul::memcpyDevice(block.feedForward.gateUpBias.buffer.deviceData + block.feedForward.gateBias.elementCount(), block.feedForward.upBias.buffer.deviceData, block.feedForward.upBias.byteCount());
        }
    }
}

void CudaLanguageModel::ensureTrainWorkspaces() {
    if (this->tokenEmbeddingWeight.empty()) throw std::logic_error("CudaLanguageModel::ensureTrainWorkspaces weights not uploaded");
    if (this->maxPackedColumns <= 0) throw std::invalid_argument("CudaLanguageModel::ensureTrainWorkspaces maxPackedColumns must be > 0");
    if (this->logitChunkRows <= 0) throw std::invalid_argument("CudaLanguageModel::ensureTrainWorkspaces logitChunkRows must be > 0");

    const size_t embeddingDim = this->tokenEmbeddingWeight.cols;
    const size_t vocabularySize = this->tokenEmbeddingWeight.rows;
    const size_t maxColumns = static_cast<size_t>(this->maxPackedColumns);
    const size_t chunkRows = static_cast<size_t>((std::min)(this->logitChunkRows, static_cast<int>(vocabularySize)));

    this->hidden.ensureSize(embeddingDim, maxColumns);
    this->normalized.ensureSize(embeddingDim, maxColumns);
    this->hiddenGradient.ensureSize(embeddingDim, maxColumns);
    this->blockInputGradientScratch.ensureSize(embeddingDim, maxColumns);
    this->normInputGradientScratch.ensureSize(embeddingDim, maxColumns);
    this->epochLossSum.ensureSize(1, 1);
    this->tokenIdsBuffer.ensureCapacity(maxColumns);
    this->targetTokenIdsBuffer.ensureCapacity(maxColumns);
    this->meanDivisorBuffer.ensureCapacity(maxColumns);
    this->packH2dDevice.ensureCapacity(3 * maxColumns);
    this->adamWindowTokenIdsBuffer.ensureCapacity(maxColumns);

    // Chunked CE scratch always; full logits when single-pass cache fits (mirrors accumulateChunkedProjection).
    this->logitChunk.ensureSize(chunkRows, maxColumns);
    this->logitGradientChunk.ensureSize(chunkRows, maxColumns);
    this->projectionWeightGradientChunk.ensureSize(chunkRows, embeddingDim);
    this->hiddenGradientChunk.ensureSize(embeddingDim, maxColumns);
    this->onlineSoftmaxMax.ensureSize(1, maxColumns);
    this->onlineSoftmaxSumExp.ensureSize(1, maxColumns);
    this->targetLogits.ensureSize(1, maxColumns);

    constexpr size_t maxCachedLogitsBytes = 512ull * 1024ull * 1024ull;
    const size_t fullLogitsBytes = vocabularySize * maxColumns * sizeof(float);
    if (fullLogitsBytes <= maxCachedLogitsBytes) {
        this->logits.ensureSize(vocabularySize, maxColumns);
        this->probabilities.free();
        this->logitGradient.free();
        this->projectionWeightGradient.free();
        this->projectionBiasGradient.free();
    } else {
        this->logits.free();
        this->probabilities.free();
        this->logitGradient.free();
        this->projectionWeightGradient.free();
        this->projectionBiasGradient.free();
    }

    // Block-input checkpoints: Full and Selective both save every block input.
    // Restore scratch is one FP32 buffer used during backward.
    if (!this->activationCheckpointingActive()) {
        this->releaseActivationCheckpoints();
    } else if (this->useHalfActivationCheckpoints()) {
        for (CudaMatrix& checkpoint : this->blockInputCheckpoints)
            checkpoint.free();
        this->blockInputCheckpoints.clear();
        if (this->blockInputCheckpointsHalf.size() != this->blocks.size())
            this->blockInputCheckpointsHalf.resize(this->blocks.size());
        for (CudaHalfMatrix& checkpoint : this->blockInputCheckpointsHalf)
            checkpoint.ensureSize(embeddingDim, maxColumns);
        this->checkpointRestoreScratch.ensureSize(embeddingDim, maxColumns);
    } else {
        this->checkpointRestoreScratch.free();
        for (CudaHalfMatrix& checkpoint : this->blockInputCheckpointsHalf)
            checkpoint.free();
        this->blockInputCheckpointsHalf.clear();
        if (this->blockInputCheckpoints.size() != this->blocks.size())
            this->blockInputCheckpoints.resize(this->blocks.size());
        for (CudaMatrix& checkpoint : this->blockInputCheckpoints)
            checkpoint.ensureSize(embeddingDim, maxColumns);
    }
    // Attn/FFN column scratch is allocated lazily in block forward; Selective releases
    // Attn QKV/Flash after each block fwd while keeping FFN acts until bwd.
}

size_t CudaLanguageModel::bytesPerPackedColumn() const {
    if (this->tokenEmbeddingWeight.empty())
        throw std::logic_error("CudaLanguageModel::bytesPerPackedColumn weights not uploaded");
    if (this->logitChunkRows <= 0)
        throw std::invalid_argument("CudaLanguageModel::bytesPerPackedColumn logitChunkRows must be > 0");

    const size_t embeddingDim = this->tokenEmbeddingWeight.cols;
    const size_t vocabularySize = this->tokenEmbeddingWeight.rows;
    const size_t chunkRows = static_cast<size_t>((std::min)(this->logitChunkRows, static_cast<int>(vocabularySize)));
    const size_t floatBytes = sizeof(float);
    const size_t intBytes = sizeof(int);
    const size_t ffnHidden = this->blocks.empty() || this->blocks[0].feedForward.gateWeight.empty()
        ? (2ull * embeddingDim * 4ull) / 3ull
        : this->blocks[0].feedForward.gateWeight.rows;

    // LM workspaces that scale with columns (ensureTrainWorkspaces).
    size_t bytes = 0;
    bytes += 5 * embeddingDim * floatBytes; // hidden, normalized, hiddenGradient, blockInputGrad, normInputGrad
    bytes += 2 * chunkRows * floatBytes;    // logitChunk, logitGradientChunk
    bytes += embeddingDim * floatBytes;     // hiddenGradientChunk
    bytes += 3 * floatBytes;                // onlineSoftmaxMax, onlineSoftmaxSumExp, targetLogits
    bytes += 3 * intBytes;                  // tokenIds, targets, meanDivisors
    bytes += 3 * intBytes;                  // packH2d fused staging
    bytes += intBytes;                      // adamWindowTokenIdsBuffer capacity unit

    constexpr size_t maxCachedLogitsBytes = 512ull * 1024ull * 1024ull;
    if (vocabularySize * floatBytes * 8192ull <= maxCachedLogitsBytes)
        bytes += vocabularySize * floatBytes;

    const size_t blockCount = this->blocks.size();
    const size_t attnScratchPerBlock = 15ull * embeddingDim * floatBytes;
    const size_t ffnScratchPerBlock = (8ull * ffnHidden + 4ull * embeddingDim) * floatBytes;

    if (this->activationCheckpointingActive()) {
        // Full + Selective: one saved block input per layer (+ FP32 restore scratch).
        if (this->useHalfActivationCheckpoints())
            bytes += blockCount * embeddingDim * 2u + embeddingDim * floatBytes;
        else
            bytes += blockCount * embeddingDim * floatBytes;
    }

    // Column scratch footprint depends on checkpoint mode:
    // Off/Full: every block may hold Attn+FFN acts after forward (L * (attn+ffn)).
    // Selective: FFN kept for all blocks; Attn QKV/Flash released after each block
    //   (peak ~1 Attn workspace during fwd/recompute).
    if (this->activationCheckpointMode == ActivationCheckpointMode::Selective) {
        bytes += blockCount * ffnScratchPerBlock;
        bytes += attnScratchPerBlock;
    } else {
        bytes += blockCount * (attnScratchPerBlock + ffnScratchPerBlock);
    }

    // AMP activation half cast scratch (shared, one active gemm).
    bytes += embeddingDim * 2ull;

    return bytes;
}

size_t CudaLanguageModel::estimatePendingTrainStaticBytes() const {
    if (this->tokenEmbeddingWeight.empty())
        throw std::logic_error("CudaLanguageModel::estimatePendingTrainStaticBytes weights not uploaded");

    auto matrixBytes = [](const CudaMatrix& matrix) -> size_t {
        return matrix.elementCount() * sizeof(float);
    };

    size_t parameterBytes = 0;
    parameterBytes += matrixBytes(this->tokenEmbeddingWeight);
    parameterBytes += matrixBytes(this->finalNorm.gamma);
    parameterBytes += matrixBytes(this->projectionBias);
    if (!this->tieEmbeddingProjection)
        parameterBytes += matrixBytes(this->projectionWeight);

    for (const CudaTransformerBlock& block : this->blocks) {
        parameterBytes += matrixBytes(block.attention.queryWeight);
        parameterBytes += matrixBytes(block.attention.keyWeight);
        parameterBytes += matrixBytes(block.attention.valueWeight);
        parameterBytes += matrixBytes(block.attention.outputWeight);
        parameterBytes += matrixBytes(block.attentionNorm.gamma);
        parameterBytes += matrixBytes(block.feedForwardNorm.gamma);
        parameterBytes += matrixBytes(block.feedForward.gateWeight);
        parameterBytes += matrixBytes(block.feedForward.gateBias);
        parameterBytes += matrixBytes(block.feedForward.upWeight);
        parameterBytes += matrixBytes(block.feedForward.upBias);
        parameterBytes += matrixBytes(block.feedForward.downWeight);
        parameterBytes += matrixBytes(block.feedForward.downBias);
        // Fused gate|up and QKV mirrors (extra FP32 copies; no separate grads).
        parameterBytes += matrixBytes(block.feedForward.gateWeight) + matrixBytes(block.feedForward.upWeight);
        parameterBytes += matrixBytes(block.attention.queryWeight)
            + matrixBytes(block.attention.keyWeight)
            + matrixBytes(block.attention.valueWeight);
    }

    // Gradients mirror FP32 parameters (except fused mirrors which have no separate grad).
    size_t gradientBytes = parameterBytes;
    for (const CudaTransformerBlock& block : this->blocks) {
        gradientBytes -= matrixBytes(block.feedForward.gateWeight) + matrixBytes(block.feedForward.upWeight);
        gradientBytes -= matrixBytes(block.attention.queryWeight)
            + matrixBytes(block.attention.keyWeight)
            + matrixBytes(block.attention.valueWeight);
    }

    size_t muonWeightBytes = 0;
    size_t adamWeightBytes = 0;
    adamWeightBytes += matrixBytes(this->tokenEmbeddingWeight);
    adamWeightBytes += matrixBytes(this->finalNorm.gamma);
    adamWeightBytes += matrixBytes(this->projectionBias);
    if (!this->tieEmbeddingProjection)
        adamWeightBytes += matrixBytes(this->projectionWeight);

    for (const CudaTransformerBlock& block : this->blocks) {
        const size_t hiddenWeightBytes =
            matrixBytes(block.attention.queryWeight)
            + matrixBytes(block.attention.keyWeight)
            + matrixBytes(block.attention.valueWeight)
            + matrixBytes(block.attention.outputWeight)
            + matrixBytes(block.feedForward.gateWeight)
            + matrixBytes(block.feedForward.upWeight)
            + matrixBytes(block.feedForward.downWeight);
        const size_t auxWeightBytes =
            matrixBytes(block.attentionNorm.gamma)
            + matrixBytes(block.feedForwardNorm.gamma)
            + matrixBytes(block.feedForward.gateBias)
            + matrixBytes(block.feedForward.upBias)
            + matrixBytes(block.feedForward.downBias);
        if (this->preferMuon) {
            muonWeightBytes += hiddenWeightBytes;
            adamWeightBytes += auxWeightBytes;
        } else {
            adamWeightBytes += hiddenWeightBytes + auxWeightBytes;
        }
    }

    size_t pending = 0;

    // Only reserve what is not already resident (re-budget after ensureTrainState must not double-count).
    const bool gradsReady = !this->blocks.empty()
        && this->trainGradients.blocks.size() == this->blocks.size()
        && this->trainGradients.tokenEmbedding.elementCount() == this->tokenEmbeddingWeight.elementCount();
    if (!gradsReady)
        pending += gradientBytes;

    if (this->preferMuon) {
        bool muonReady = this->blockMuonStates.size() == this->blocks.size();
        if (muonReady) {
            for (size_t blockIndex = 0; blockIndex < this->blockMuonStates.size(); ++blockIndex) {
                if (this->blockMuonStates[blockIndex].qkvWeight.momentum.elementCount() == 0) {
                    muonReady = false;
                    break;
                }
            }
        }
        if (!muonReady)
            pending += muonWeightBytes;
    }

    // Adam moments stay lazy until first applyGradients — always reserve unless CPU offload.
    if (!CudaAdam::preferCpuOffload) {
        if (CudaAdam::preferInt8Moments)
            pending += adamWeightBytes / 2ull + adamWeightBytes / 128ull;
        else
            pending += 2ull * adamWeightBytes;
    }

    // Sticky FP16 master-weight mirrors for AMP GEMMs.
    if (CudaAmp::preferMixedPrecision)
        pending += parameterBytes / 2ull;

    return pending;
}

void CudaLanguageModel::releasePackedTrainWorkspaces() {
    this->releaseTrainGraph();
    this->hidden.free();
    this->normalized.free();
    this->hiddenGradient.free();
    this->blockInputGradientScratch.free();
    this->normInputGradientScratch.free();
    this->logits.free();
    this->probabilities.free();
    this->logitGradient.free();
    this->projectionWeightGradient.free();
    this->projectionBiasGradient.free();
    this->logitChunk.free();
    this->logitGradientChunk.free();
    this->projectionWeightGradientChunk.free();
    this->hiddenGradientChunk.free();
    this->onlineSoftmaxMax.free();
    this->onlineSoftmaxSumExp.free();
    this->targetLogits.free();
    this->tokenIdsBuffer.free();
    this->targetTokenIdsBuffer.free();
    this->meanDivisorBuffer.free();
    this->packH2dDevice.free();
    this->adamWindowTokenIdsBuffer.free();
    this->releaseActivationCheckpoints();

    for (CudaTransformerBlock& block : this->blocks) {
        block.attention.releaseActivationScratch();
        block.feedForward.gateUpPreActivation.free();
        block.feedForward.gateUpHiddenGradient.free();
        block.feedForward.gatePreActivation.free();
        block.feedForward.gateActivated.free();
        block.feedForward.up.free();
        block.feedForward.hidden.free();
        block.feedForward.output.free();
        block.feedForward.inputCache.free();
        block.feedForward.hiddenGradient.free();
        block.feedForward.upGradient.free();
        block.feedForward.gateGradient.free();
        block.feedForward.siluDerivative.free();
        block.feedForward.temp.free();
    }
}

int CudaLanguageModel::maxPackExamplesForSegment(int segmentLength) const {
    if (segmentLength <= 0) throw std::invalid_argument("CudaLanguageModel::maxPackExamplesForSegment segmentLength must be > 0");
    if (this->maxPackedColumns <= 0) return 1;
    return (std::max)(1, this->maxPackedColumns / segmentLength);
}

void CudaLanguageModel::applyVramPackBudget(float freeFraction, size_t safetyReserveBytes) {
    if (this->maxPackedColumnsManual) return;
    if (this->tokenEmbeddingWeight.empty())
        throw std::logic_error("CudaLanguageModel::applyVramPackBudget weights not uploaded");
    if (!(freeFraction > 0.0f && freeFraction <= 1.0f))
        throw std::invalid_argument("CudaLanguageModel::applyVramPackBudget freeFraction must be in (0, 1]");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaLanguageModel::applyVramPackBudget no CUDA device");

    // Return pack-scaled workspace to the free pool before measuring; otherwise a second
    // applyVramPackBudget (e.g. after enableCudaTrain) sees free≈0 and collapses to minCols.
    this->releasePackedTrainWorkspaces();
    CudaMatmul::throwIfCudaFailed(cudaDeviceSynchronize(), "CudaLanguageModel::applyVramPackBudget sync after workspace release");

    size_t freeBytes = 0;
    size_t totalBytes = 0;
    CudaMatmul::throwIfCudaFailed(cudaMemGetInfo(&freeBytes, &totalBytes), "CudaLanguageModel::applyVramPackBudget memGetInfo");

    // WDDM shares VRAM with the desktop — keep a hard floor so DWM/apps don't thrash the train working set.
    const size_t displayFloor = (std::max)(safetyReserveBytes, totalBytes / 5ull); // >=20% of card or explicit safety
    const size_t pendingStatic = this->estimatePendingTrainStaticBytes();
    const size_t reserved = pendingStatic + displayFloor;
    size_t usableBytes = freeBytes > reserved ? freeBytes - reserved : 0;

    const size_t perColumn = this->bytesPerPackedColumn();
    if (perColumn == 0) throw std::logic_error("CudaLanguageModel::applyVramPackBudget zero bytesPerPackedColumn");

    // bytesPerPackedColumn undercounts lazy Attn/FFN scratch + cuBLAS workspaces; pad so packs stay safe.
    constexpr double footprintSlack = 1.40;
    const size_t budgetBytes = static_cast<size_t>(static_cast<double>(usableBytes) * static_cast<double>(freeFraction));
    const size_t bytesPerColBudget = static_cast<size_t>(static_cast<double>(perColumn) * footprintSlack);
    size_t cols = bytesPerColBudget > 0 ? budgetBytes / bytesPerColBudget : 0;

    int minCols = CudaLanguageModel::lengthBucketStep;
    if (this->maximumPositionCount > minCols) minCols = this->maximumPositionCount;
    // Beyond ~4k cols throughput saturates on 16GB cards while VRAM pressure kills tok/s (WDDM thrash).
    constexpr int maxCols = 4096;
    if (cols < static_cast<size_t>(minCols)) cols = static_cast<size_t>(minCols);
    if (cols > static_cast<size_t>(maxCols)) cols = static_cast<size_t>(maxCols);

    cols = (cols / 64u) * 64u;
    if (cols < static_cast<size_t>(minCols)) cols = static_cast<size_t>(minCols);

    // Final clamp: planned workspace must leave displayFloor free after pending static.
    while (cols > static_cast<size_t>(minCols)) {
        const size_t plannedWorkspace = cols * bytesPerColBudget;
        if (pendingStatic + plannedWorkspace + displayFloor <= freeBytes) break;
        cols -= 64u;
    }
    if (cols < static_cast<size_t>(minCols)) cols = static_cast<size_t>(minCols);

    this->maxPackedColumns = static_cast<int>(cols);

    const double freeMiB = static_cast<double>(freeBytes) / (1024.0 * 1024.0);
    const double totalMiB = static_cast<double>(totalBytes) / (1024.0 * 1024.0);
    const double pendingMiB = static_cast<double>(pendingStatic) / (1024.0 * 1024.0);
    const double safetyMiB = static_cast<double>(displayFloor) / (1024.0 * 1024.0);
    const double workspaceMiB = static_cast<double>(perColumn) * static_cast<double>(this->maxPackedColumns) / (1024.0 * 1024.0);
    const double plannedMiB = static_cast<double>(bytesPerColBudget) * static_cast<double>(this->maxPackedColumns) / (1024.0 * 1024.0);
    std::printf(
        "CudaLanguageModel::applyVramPackBudget: ckpt=%s  free=%.0f/%.0f MiB  pendingStatic=%.0f safety=%.0f fraction=%.2f  maxPackCols=%d  ~workspace=%.0f MiB (~%.0f w/slack)\n",
        CudaLanguageModel::activationCheckpointModeName(this->activationCheckpointMode),
        freeMiB, totalMiB, pendingMiB, safetyMiB, freeFraction, this->maxPackedColumns, workspaceMiB, plannedMiB);
}

void CudaLanguageModel::releaseActivationCheckpoints() {
    for (CudaMatrix& checkpoint : this->blockInputCheckpoints)
        checkpoint.free();
    this->blockInputCheckpoints.clear();
    for (CudaHalfMatrix& checkpoint : this->blockInputCheckpointsHalf)
        checkpoint.free();
    this->blockInputCheckpointsHalf.clear();
    this->checkpointRestoreScratch.free();
}

void CudaLanguageModel::downloadTo(LanguageModel& host) {
    if (this->tokenEmbeddingWeight.empty()) throw std::logic_error("CudaLanguageModel::downloadTo weights not uploaded");

    if (CudaAdam::preferFp16GpuWeights) {
        auto copyMaster = [](const Matrix& master, Matrix& dest) {
            if (master.empty()) throw std::logic_error("CudaLanguageModel::downloadTo missing host master");
            dest = master;
        };
        copyMaster(this->hostTokenEmbeddingMaster, host.tokenEmbedding.weight);
        for (size_t blockIndex = 0; blockIndex < this->blocks.size(); ++blockIndex) {
            const CudaTransformerBlockHostAdamStates& hosts = this->hostBlockAdamStates[blockIndex];
            TransformerBlock& hostBlock = host.blocks[blockIndex];
            copyMaster(hosts.queryWeightMaster, hostBlock.attention.queryWeight);
            copyMaster(hosts.keyWeightMaster, hostBlock.attention.keyWeight);
            copyMaster(hosts.valueWeightMaster, hostBlock.attention.valueWeight);
            copyMaster(hosts.attentionOutputWeightMaster, hostBlock.attention.outputWeight);
            copyMaster(hosts.attentionNormGammaMaster, hostBlock.attentionNorm.gamma);
            copyMaster(hosts.feedForwardNormGammaMaster, hostBlock.feedForwardNorm.gamma);
            copyMaster(hosts.feedForwardGateWeightMaster, hostBlock.feedForward.gateWeight);
            copyMaster(hosts.feedForwardGateBiasMaster, hostBlock.feedForward.gateBias);
            copyMaster(hosts.feedForwardUpWeightMaster, hostBlock.feedForward.upWeight);
            copyMaster(hosts.feedForwardUpBiasMaster, hostBlock.feedForward.upBias);
            copyMaster(hosts.feedForwardDownWeightMaster, hostBlock.feedForward.downWeight);
            copyMaster(hosts.feedForwardDownBiasMaster, hostBlock.feedForward.downBias);
        }
        copyMaster(this->hostFinalNormGammaMaster, host.finalNorm.gamma);
        host.tieEmbeddingProjection = this->tieEmbeddingProjection;
        if (this->tieEmbeddingProjection) {
            host.outputProjection.weight = Matrix();
            host.projectionWeightState = AdamState{};
        } else {
            copyMaster(this->hostProjectionWeightMaster, host.outputProjection.weight);
        }
        copyMaster(this->hostProjectionBiasMaster, host.outputProjection.bias);
        return;
    }

    this->tokenEmbeddingWeight.downloadInto(host.tokenEmbedding.weight);
    for (size_t blockIndex = 0; blockIndex < this->blocks.size(); ++blockIndex) {
        const CudaTransformerBlock& block = this->blocks[blockIndex];
        TransformerBlock& hostBlock = host.blocks[blockIndex];
        block.attention.queryWeight.downloadInto(hostBlock.attention.queryWeight);
        block.attention.keyWeight.downloadInto(hostBlock.attention.keyWeight);
        block.attention.valueWeight.downloadInto(hostBlock.attention.valueWeight);
        block.attention.outputWeight.downloadInto(hostBlock.attention.outputWeight);
        block.attentionNorm.gamma.downloadInto(hostBlock.attentionNorm.gamma);
        block.feedForwardNorm.gamma.downloadInto(hostBlock.feedForwardNorm.gamma);
        block.feedForward.gateWeight.downloadInto(hostBlock.feedForward.gateWeight);
        block.feedForward.gateBias.downloadInto(hostBlock.feedForward.gateBias);
        block.feedForward.upWeight.downloadInto(hostBlock.feedForward.upWeight);
        block.feedForward.upBias.downloadInto(hostBlock.feedForward.upBias);
        block.feedForward.downWeight.downloadInto(hostBlock.feedForward.downWeight);
        block.feedForward.downBias.downloadInto(hostBlock.feedForward.downBias);
    }
    this->finalNorm.gamma.downloadInto(host.finalNorm.gamma);
    host.tieEmbeddingProjection = this->tieEmbeddingProjection;
    if (this->tieEmbeddingProjection) {
        host.outputProjection.weight = Matrix();
        host.projectionWeightState = AdamState{};
    } else {
        this->projectionWeight.downloadInto(host.outputProjection.weight);
    }
    this->projectionBias.downloadInto(host.outputProjection.bias);
}

void CudaLanguageModel::downloadOptimizerTo(LanguageModel& host) {
    if (this->tokenEmbeddingWeight.empty()) throw std::logic_error("CudaLanguageModel::downloadOptimizerTo weights not uploaded");
    if (host.blocks.size() != this->blocks.size())
        throw std::invalid_argument("CudaLanguageModel::downloadOptimizerTo block count mismatch");

    host.optimizer.learningRate = this->adam.learningRate;
    host.optimizer.beta1 = this->adam.beta1;
    host.optimizer.beta2 = this->adam.beta2;
    host.optimizer.epsilon = this->adam.epsilon;
    host.optimizer.timeStep = this->adam.timeStep;

    auto clearHostMuon = [](TransformerBlock& hostBlock) {
        hostBlock.queryWeightMuon = MuonState{};
        hostBlock.keyWeightMuon = MuonState{};
        hostBlock.valueWeightMuon = MuonState{};
        hostBlock.attentionOutputWeightMuon = MuonState{};
        hostBlock.feedForwardGateWeightMuon = MuonState{};
        hostBlock.feedForwardUpWeightMuon = MuonState{};
        hostBlock.feedForwardDownWeightMuon = MuonState{};
    };

    auto downloadMuonOrZero = [](const CudaMuonState& state, MuonState& hostState, size_t rows, size_t cols) {
        if (state.empty()) {
            hostState = MuonState::zerosLike(Matrix(rows, cols, 0.0f));
            return;
        }
        state.downloadInto(hostState);
    };

    auto downloadOrZero = [](const CudaAdamState& state, AdamState& hostState, size_t rows, size_t cols) {
        if (state.empty()) {
            hostState = AdamState::zerosLike(Matrix(rows, cols, 0.0f));
            return;
        }
        state.downloadInto(hostState, rows, cols);
    };

    auto downloadBlockAuxAdam = [&](TransformerBlock& hostBlock, const CudaTransformerBlock& block, const CudaTransformerBlockAdamStates& states) {
        downloadOrZero(states.attentionNormGamma, hostBlock.attentionNormGammaState, block.attentionNorm.gamma.rows, block.attentionNorm.gamma.cols);
        downloadOrZero(states.feedForwardNormGamma, hostBlock.feedForwardNormGammaState, block.feedForwardNorm.gamma.rows, block.feedForwardNorm.gamma.cols);
        downloadOrZero(states.feedForwardGateBias, hostBlock.feedForwardGateBiasState, block.feedForward.gateBias.rows, block.feedForward.gateBias.cols);
        downloadOrZero(states.feedForwardUpBias, hostBlock.feedForwardUpBiasState, block.feedForward.upBias.rows, block.feedForward.upBias.cols);
        downloadOrZero(states.feedForwardDownBias, hostBlock.feedForwardDownBiasState, block.feedForward.downBias.rows, block.feedForward.downBias.cols);
    };

    auto downloadBlockMuon = [&](TransformerBlock& hostBlock, const CudaTransformerBlock& block, const CudaTransformerBlockMuonStates& muonStates) {
        // fused device momentum → split host Q/K/V and gate/up for checkpoint compatibility
        MuonState qkvHost;
        MuonState gateUpHost;
        downloadMuonOrZero(muonStates.qkvWeight, qkvHost, block.attention.queryWeight.rows * 3ull, block.attention.queryWeight.cols);
        downloadMuonOrZero(muonStates.attentionOutputWeight, hostBlock.attentionOutputWeightMuon, block.attention.outputWeight.rows, block.attention.outputWeight.cols);
        downloadMuonOrZero(muonStates.feedForwardGateUpWeight, gateUpHost, block.feedForward.gateWeight.rows + block.feedForward.upWeight.rows, block.feedForward.gateWeight.cols);
        downloadMuonOrZero(muonStates.feedForwardDownWeight, hostBlock.feedForwardDownWeightMuon, block.feedForward.downWeight.rows, block.feedForward.downWeight.cols);

        const size_t qSlice = block.attention.queryWeight.elementCount();
        hostBlock.queryWeightMuon.momentum = Matrix(block.attention.queryWeight.rows, block.attention.queryWeight.cols, 0.0f);
        hostBlock.keyWeightMuon.momentum = Matrix(block.attention.keyWeight.rows, block.attention.keyWeight.cols, 0.0f);
        hostBlock.valueWeightMuon.momentum = Matrix(block.attention.valueWeight.rows, block.attention.valueWeight.cols, 0.0f);
        if (!qkvHost.momentum.empty() && qkvHost.momentum.data.size() >= 3ull * qSlice) {
            std::copy(qkvHost.momentum.data.begin(), qkvHost.momentum.data.begin() + static_cast<std::ptrdiff_t>(qSlice), hostBlock.queryWeightMuon.momentum.data.begin());
            std::copy(qkvHost.momentum.data.begin() + static_cast<std::ptrdiff_t>(qSlice), qkvHost.momentum.data.begin() + static_cast<std::ptrdiff_t>(2ull * qSlice), hostBlock.keyWeightMuon.momentum.data.begin());
            std::copy(qkvHost.momentum.data.begin() + static_cast<std::ptrdiff_t>(2ull * qSlice), qkvHost.momentum.data.begin() + static_cast<std::ptrdiff_t>(3ull * qSlice), hostBlock.valueWeightMuon.momentum.data.begin());
        }

        const size_t gateSlice = block.feedForward.gateWeight.elementCount();
        hostBlock.feedForwardGateWeightMuon.momentum = Matrix(block.feedForward.gateWeight.rows, block.feedForward.gateWeight.cols, 0.0f);
        hostBlock.feedForwardUpWeightMuon.momentum = Matrix(block.feedForward.upWeight.rows, block.feedForward.upWeight.cols, 0.0f);
        if (!gateUpHost.momentum.empty() && gateUpHost.momentum.data.size() >= gateSlice + block.feedForward.upWeight.elementCount()) {
            std::copy(gateUpHost.momentum.data.begin(), gateUpHost.momentum.data.begin() + static_cast<std::ptrdiff_t>(gateSlice), hostBlock.feedForwardGateWeightMuon.momentum.data.begin());
            std::copy(gateUpHost.momentum.data.begin() + static_cast<std::ptrdiff_t>(gateSlice), gateUpHost.momentum.data.begin() + static_cast<std::ptrdiff_t>(gateSlice + block.feedForward.upWeight.elementCount()), hostBlock.feedForwardUpWeightMuon.momentum.data.begin());
        }

        hostBlock.queryWeightState = AdamState{};
        hostBlock.keyWeightState = AdamState{};
        hostBlock.valueWeightState = AdamState{};
        hostBlock.attentionOutputWeightState = AdamState{};
        hostBlock.feedForwardGateWeightState = AdamState{};
        hostBlock.feedForwardUpWeightState = AdamState{};
        hostBlock.feedForwardDownWeightState = AdamState{};
    };

    if (CudaAdam::preferCpuOffload) {
        host.tokenEmbeddingState = this->hostTokenEmbeddingState;
        host.finalNormGammaState = this->hostFinalNormGammaState;
        if (this->tieEmbeddingProjection)
            host.projectionWeightState = AdamState{};
        else
            host.projectionWeightState = this->hostProjectionWeightState;
        host.projectionBiasState = this->hostProjectionBiasState;

        if (this->hostBlockAdamStates.size() != this->blocks.size())
            this->hostBlockAdamStates.resize(this->blocks.size());

        for (size_t blockIndex = 0; blockIndex < this->blocks.size(); ++blockIndex) {
            TransformerBlock& hostBlock = host.blocks[blockIndex];
            const CudaTransformerBlockHostAdamStates& states = this->hostBlockAdamStates[blockIndex];
            const CudaTransformerBlock& block = this->blocks[blockIndex];

            if (this->preferMuon) {
                if (this->blockMuonStates.size() != this->blocks.size())
                    throw std::logic_error("CudaLanguageModel::downloadOptimizerTo Muon states not ready");
                downloadBlockMuon(hostBlock, block, this->blockMuonStates[blockIndex]);
            } else {
                clearHostMuon(hostBlock);
                hostBlock.queryWeightState = states.queryWeight;
                hostBlock.keyWeightState = states.keyWeight;
                hostBlock.valueWeightState = states.valueWeight;
                hostBlock.attentionOutputWeightState = states.attentionOutputWeight;
                hostBlock.feedForwardGateWeightState = states.feedForwardGateWeight;
                hostBlock.feedForwardUpWeightState = states.feedForwardUpWeight;
                hostBlock.feedForwardDownWeightState = states.feedForwardDownWeight;
            }
            hostBlock.attentionNormGammaState = states.attentionNormGamma;
            hostBlock.feedForwardNormGammaState = states.feedForwardNormGamma;
            hostBlock.feedForwardGateBiasState = states.feedForwardGateBias;
            hostBlock.feedForwardUpBiasState = states.feedForwardUpBias;
            hostBlock.feedForwardDownBiasState = states.feedForwardDownBias;
        }
        return;
    }

    downloadOrZero(this->tokenEmbeddingState, host.tokenEmbeddingState, this->tokenEmbeddingWeight.rows, this->tokenEmbeddingWeight.cols);
    downloadOrZero(this->finalNormGammaState, host.finalNormGammaState, this->finalNorm.gamma.rows, this->finalNorm.gamma.cols);
    if (this->tieEmbeddingProjection)
        host.projectionWeightState = AdamState{};
    else
        downloadOrZero(this->projectionWeightState, host.projectionWeightState, this->projectionWeight.rows, this->projectionWeight.cols);
    downloadOrZero(this->projectionBiasState, host.projectionBiasState, this->projectionBias.rows, this->projectionBias.cols);

    if (this->blockAdamStates.size() != this->blocks.size())
        this->blockAdamStates.resize(this->blocks.size());

    for (size_t blockIndex = 0; blockIndex < this->blocks.size(); ++blockIndex) {
        const CudaTransformerBlock& block = this->blocks[blockIndex];
        TransformerBlock& hostBlock = host.blocks[blockIndex];
        const CudaTransformerBlockAdamStates& states = this->blockAdamStates[blockIndex];

        if (this->preferMuon) {
            if (this->blockMuonStates.size() != this->blocks.size())
                throw std::logic_error("CudaLanguageModel::downloadOptimizerTo Muon states not ready");
            downloadBlockMuon(hostBlock, block, this->blockMuonStates[blockIndex]);
        } else {
            clearHostMuon(hostBlock);
            downloadOrZero(states.queryWeight, hostBlock.queryWeightState, block.attention.queryWeight.rows, block.attention.queryWeight.cols);
            downloadOrZero(states.keyWeight, hostBlock.keyWeightState, block.attention.keyWeight.rows, block.attention.keyWeight.cols);
            downloadOrZero(states.valueWeight, hostBlock.valueWeightState, block.attention.valueWeight.rows, block.attention.valueWeight.cols);
            downloadOrZero(states.attentionOutputWeight, hostBlock.attentionOutputWeightState, block.attention.outputWeight.rows, block.attention.outputWeight.cols);
            downloadOrZero(states.feedForwardGateWeight, hostBlock.feedForwardGateWeightState, block.feedForward.gateWeight.rows, block.feedForward.gateWeight.cols);
            downloadOrZero(states.feedForwardUpWeight, hostBlock.feedForwardUpWeightState, block.feedForward.upWeight.rows, block.feedForward.upWeight.cols);
            downloadOrZero(states.feedForwardDownWeight, hostBlock.feedForwardDownWeightState, block.feedForward.downWeight.rows, block.feedForward.downWeight.cols);
        }
        downloadBlockAuxAdam(hostBlock, block, states);
    }
}

void CudaLanguageModel::uploadOptimizerFrom(const LanguageModel& host) {
    if (this->tokenEmbeddingWeight.empty()) throw std::logic_error("CudaLanguageModel::uploadOptimizerFrom weights not uploaded");
    if (host.blocks.size() != this->blocks.size())
        throw std::invalid_argument("CudaLanguageModel::uploadOptimizerFrom block count mismatch");

    this->ensureTrainState();
    this->adam.learningRate = host.optimizer.learningRate;
    this->adam.beta1 = host.optimizer.beta1;
    this->adam.beta2 = host.optimizer.beta2;
    this->adam.epsilon = host.optimizer.epsilon;
    this->adam.timeStep = host.optimizer.timeStep;
    this->muon.learningRate = this->adam.learningRate;

    if (CudaAdam::preferCpuOffload) {
        this->hostTokenEmbeddingState = host.tokenEmbeddingState;
        this->hostFinalNormGammaState = host.finalNormGammaState;
        if (this->tieEmbeddingProjection)
            this->hostProjectionWeightState = AdamState{};
        else
            this->hostProjectionWeightState = host.projectionWeightState;
        this->hostProjectionBiasState = host.projectionBiasState;

        if (this->hostBlockAdamStates.size() != this->blocks.size())
            this->hostBlockAdamStates.resize(this->blocks.size());

        for (size_t blockIndex = 0; blockIndex < this->blocks.size(); ++blockIndex) {
            const TransformerBlock& hostBlock = host.blocks[blockIndex];
            CudaTransformerBlockHostAdamStates& states = this->hostBlockAdamStates[blockIndex];
            if (this->preferMuon) {
                if (this->blockMuonStates.size() != this->blocks.size())
                    throw std::logic_error("CudaLanguageModel::uploadOptimizerFrom Muon states not ready");
                CudaTransformerBlockMuonStates& muonStates = this->blockMuonStates[blockIndex];
                CudaTransformerBlock& block = this->blocks[blockIndex];
                block.attention.syncFusedQkvWeight();
                block.feedForward.syncFusedGateUpWeight();
                muonStates.ensureFrom(block);

                Matrix qkvMom(block.attention.queryWeight.rows * 3ull, block.attention.queryWeight.cols, 0.0f);
                const size_t qSlice = block.attention.queryWeight.elementCount();
                if (!hostBlock.queryWeightMuon.momentum.empty())
                    std::copy(hostBlock.queryWeightMuon.momentum.data.begin(), hostBlock.queryWeightMuon.momentum.data.end(), qkvMom.data.begin());
                if (!hostBlock.keyWeightMuon.momentum.empty())
                    std::copy(hostBlock.keyWeightMuon.momentum.data.begin(), hostBlock.keyWeightMuon.momentum.data.end(), qkvMom.data.begin() + static_cast<std::ptrdiff_t>(qSlice));
                if (!hostBlock.valueWeightMuon.momentum.empty())
                    std::copy(hostBlock.valueWeightMuon.momentum.data.begin(), hostBlock.valueWeightMuon.momentum.data.end(), qkvMom.data.begin() + static_cast<std::ptrdiff_t>(2ull * qSlice));
                MuonState qkvState;
                qkvState.momentum = std::move(qkvMom);
                muonStates.qkvWeight.uploadFrom(qkvState);

                muonStates.attentionOutputWeight.uploadFrom(hostBlock.attentionOutputWeightMuon);

                Matrix gateUpMom(block.feedForward.gateWeight.rows + block.feedForward.upWeight.rows, block.feedForward.gateWeight.cols, 0.0f);
                const size_t gateSlice = block.feedForward.gateWeight.elementCount();
                if (!hostBlock.feedForwardGateWeightMuon.momentum.empty())
                    std::copy(hostBlock.feedForwardGateWeightMuon.momentum.data.begin(), hostBlock.feedForwardGateWeightMuon.momentum.data.end(), gateUpMom.data.begin());
                if (!hostBlock.feedForwardUpWeightMuon.momentum.empty())
                    std::copy(hostBlock.feedForwardUpWeightMuon.momentum.data.begin(), hostBlock.feedForwardUpWeightMuon.momentum.data.end(), gateUpMom.data.begin() + static_cast<std::ptrdiff_t>(gateSlice));
                MuonState gateUpState;
                gateUpState.momentum = std::move(gateUpMom);
                muonStates.feedForwardGateUpWeight.uploadFrom(gateUpState);

                muonStates.feedForwardDownWeight.uploadFrom(hostBlock.feedForwardDownWeightMuon);
                states.queryWeight = AdamState{};
                states.keyWeight = AdamState{};
                states.valueWeight = AdamState{};
                states.attentionOutputWeight = AdamState{};
                states.feedForwardGateWeight = AdamState{};
                states.feedForwardUpWeight = AdamState{};
                states.feedForwardDownWeight = AdamState{};
            } else {
                states.queryWeight = hostBlock.queryWeightState;
                states.keyWeight = hostBlock.keyWeightState;
                states.valueWeight = hostBlock.valueWeightState;
                states.attentionOutputWeight = hostBlock.attentionOutputWeightState;
                states.feedForwardGateWeight = hostBlock.feedForwardGateWeightState;
                states.feedForwardUpWeight = hostBlock.feedForwardUpWeightState;
                states.feedForwardDownWeight = hostBlock.feedForwardDownWeightState;
            }
            states.attentionNormGamma = hostBlock.attentionNormGammaState;
            states.feedForwardNormGamma = hostBlock.feedForwardNormGammaState;
            states.feedForwardGateBias = hostBlock.feedForwardGateBiasState;
            states.feedForwardUpBias = hostBlock.feedForwardUpBiasState;
            states.feedForwardDownBias = hostBlock.feedForwardDownBiasState;
        }
        return;
    }

    this->tokenEmbeddingState.uploadFrom(host.tokenEmbeddingState);
    this->finalNormGammaState.uploadFrom(host.finalNormGammaState);
    if (!this->tieEmbeddingProjection)
        this->projectionWeightState.uploadFrom(host.projectionWeightState);
    this->projectionBiasState.uploadFrom(host.projectionBiasState);

    if (this->blockAdamStates.size() != this->blocks.size())
        this->blockAdamStates.resize(this->blocks.size());

    for (size_t blockIndex = 0; blockIndex < this->blocks.size(); ++blockIndex) {
        const TransformerBlock& hostBlock = host.blocks[blockIndex];
        CudaTransformerBlockAdamStates& states = this->blockAdamStates[blockIndex];
        if (this->preferMuon) {
            if (this->blockMuonStates.size() != this->blocks.size())
                throw std::logic_error("CudaLanguageModel::uploadOptimizerFrom Muon states not ready");
            CudaTransformerBlockMuonStates& muonStates = this->blockMuonStates[blockIndex];
            CudaTransformerBlock& block = this->blocks[blockIndex];
            block.attention.syncFusedQkvWeight();
            block.feedForward.syncFusedGateUpWeight();
            muonStates.ensureFrom(block);

            Matrix qkvMom(block.attention.queryWeight.rows * 3ull, block.attention.queryWeight.cols, 0.0f);
            const size_t qSlice = block.attention.queryWeight.elementCount();
            if (!hostBlock.queryWeightMuon.momentum.empty())
                std::copy(hostBlock.queryWeightMuon.momentum.data.begin(), hostBlock.queryWeightMuon.momentum.data.end(), qkvMom.data.begin());
            if (!hostBlock.keyWeightMuon.momentum.empty())
                std::copy(hostBlock.keyWeightMuon.momentum.data.begin(), hostBlock.keyWeightMuon.momentum.data.end(), qkvMom.data.begin() + static_cast<std::ptrdiff_t>(qSlice));
            if (!hostBlock.valueWeightMuon.momentum.empty())
                std::copy(hostBlock.valueWeightMuon.momentum.data.begin(), hostBlock.valueWeightMuon.momentum.data.end(), qkvMom.data.begin() + static_cast<std::ptrdiff_t>(2ull * qSlice));
            MuonState qkvState;
            qkvState.momentum = std::move(qkvMom);
            muonStates.qkvWeight.uploadFrom(qkvState);

            muonStates.attentionOutputWeight.uploadFrom(hostBlock.attentionOutputWeightMuon);

            Matrix gateUpMom(block.feedForward.gateWeight.rows + block.feedForward.upWeight.rows, block.feedForward.gateWeight.cols, 0.0f);
            const size_t gateSlice = block.feedForward.gateWeight.elementCount();
            if (!hostBlock.feedForwardGateWeightMuon.momentum.empty())
                std::copy(hostBlock.feedForwardGateWeightMuon.momentum.data.begin(), hostBlock.feedForwardGateWeightMuon.momentum.data.end(), gateUpMom.data.begin());
            if (!hostBlock.feedForwardUpWeightMuon.momentum.empty())
                std::copy(hostBlock.feedForwardUpWeightMuon.momentum.data.begin(), hostBlock.feedForwardUpWeightMuon.momentum.data.end(), gateUpMom.data.begin() + static_cast<std::ptrdiff_t>(gateSlice));
            MuonState gateUpState;
            gateUpState.momentum = std::move(gateUpMom);
            muonStates.feedForwardGateUpWeight.uploadFrom(gateUpState);

            muonStates.feedForwardDownWeight.uploadFrom(hostBlock.feedForwardDownWeightMuon);
            states.freeMuonManagedWeights();
        } else {
            states.queryWeight.uploadFrom(hostBlock.queryWeightState);
            states.keyWeight.uploadFrom(hostBlock.keyWeightState);
            states.valueWeight.uploadFrom(hostBlock.valueWeightState);
            states.attentionOutputWeight.uploadFrom(hostBlock.attentionOutputWeightState);
            states.feedForwardGateWeight.uploadFrom(hostBlock.feedForwardGateWeightState);
            states.feedForwardUpWeight.uploadFrom(hostBlock.feedForwardUpWeightState);
            states.feedForwardDownWeight.uploadFrom(hostBlock.feedForwardDownWeightState);
        }
        states.attentionNormGamma.uploadFrom(hostBlock.attentionNormGammaState);
        states.feedForwardNormGamma.uploadFrom(hostBlock.feedForwardNormGammaState);
        states.feedForwardGateBias.uploadFrom(hostBlock.feedForwardGateBiasState);
        states.feedForwardUpBias.uploadFrom(hostBlock.feedForwardUpBiasState);
        states.feedForwardDownBias.uploadFrom(hostBlock.feedForwardDownBiasState);
    }
}

float CudaLanguageModel::accumulateExample(const LanguageModelExample& example, CudaLanguageModelGradients& gradients) {
    const LanguageModelExample* pointer = &example;
    return this->accumulatePackedExamples(&pointer, 1, gradients);
}

float CudaLanguageModel::flushPackedHostBuffers(int segmentLength, int exampleCount, CudaLanguageModelGradients& gradients) {
    if (segmentLength <= 0) throw std::invalid_argument("CudaLanguageModel::flushPackedHostBuffers segmentLength must be > 0");
    if (exampleCount <= 0) throw std::invalid_argument("CudaLanguageModel::flushPackedHostBuffers exampleCount must be > 0");
    if (gradients.blocks.size() != this->blocks.size())
        throw std::invalid_argument("CudaLanguageModel::flushPackedHostBuffers gradients not initialized");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaLanguageModel::flushPackedHostBuffers no CUDA device");

    const size_t tokenCount = this->packedInputTokenIds.size();
    if (tokenCount == 0) throw std::invalid_argument("CudaLanguageModel::flushPackedHostBuffers empty pack");
    if (tokenCount != this->packedTargetTokenIds.size() || tokenCount != this->packedMeanDivisors.size())
        throw std::invalid_argument("CudaLanguageModel::flushPackedHostBuffers pack buffer size mismatch");
    if (static_cast<size_t>(segmentLength) * static_cast<size_t>(exampleCount) != tokenCount)
        throw std::invalid_argument("CudaLanguageModel::flushPackedHostBuffers tokenCount mismatch");

    if (segmentLength != this->trainGraphSeenSegmentLength || exampleCount != this->trainGraphSeenExampleCount) {
        this->releaseTrainGraph();
        this->trainGraphWarmups = 0;
        this->trainGraphSeenSegmentLength = segmentLength;
        this->trainGraphSeenExampleCount = exampleCount;
    }

    // One H2D for [inputs | targets | meanDivisors], then cheap D2D splits.
    this->packH2dHost.resize(3 * tokenCount);
    std::copy(this->packedInputTokenIds.begin(), this->packedInputTokenIds.end(), this->packH2dHost.begin());
    std::copy(this->packedTargetTokenIds.begin(), this->packedTargetTokenIds.end(), this->packH2dHost.begin() + tokenCount);
    std::copy(this->packedMeanDivisors.begin(), this->packedMeanDivisors.end(), this->packH2dHost.begin() + 2 * tokenCount);

    this->packH2dDevice.ensureCapacity(3 * tokenCount);
    this->packH2dDevice.copyFromHost(this->packH2dHost.data(), 3 * tokenCount);

    this->tokenIdsBuffer.ensureCapacity(tokenCount);
    this->targetTokenIdsBuffer.ensureCapacity(tokenCount);
    this->meanDivisorBuffer.ensureCapacity(tokenCount);
    const size_t bytes = tokenCount * sizeof(int);

    const bool useGraphPath = this->preferTrainGraph && !this->activationCheckpointingActive();
    if (useGraphPath)
        this->ensureTrainStream();

    if (useGraphPath && this->trainStream != nullptr) {
        CudaMatmul::throwIfCudaFailed(
            cudaMemcpyAsync(this->tokenIdsBuffer.deviceData, this->packH2dDevice.deviceData, bytes, cudaMemcpyDeviceToDevice, this->trainStream),
            "flushPackedHostBuffers D2D inputs");
        CudaMatmul::throwIfCudaFailed(
            cudaMemcpyAsync(this->targetTokenIdsBuffer.deviceData, this->packH2dDevice.deviceData + tokenCount, bytes, cudaMemcpyDeviceToDevice, this->trainStream),
            "flushPackedHostBuffers D2D targets");
        CudaMatmul::throwIfCudaFailed(
            cudaMemcpyAsync(this->meanDivisorBuffer.deviceData, this->packH2dDevice.deviceData + 2 * tokenCount, bytes, cudaMemcpyDeviceToDevice, this->trainStream),
            "flushPackedHostBuffers D2D meanDivisors");
    } else {
        CudaMatmul::throwIfCudaFailed(
            cudaMemcpy(this->tokenIdsBuffer.deviceData, this->packH2dDevice.deviceData, bytes, cudaMemcpyDeviceToDevice),
            "CudaLanguageModel::flushPackedHostBuffers D2D inputs");
        CudaMatmul::throwIfCudaFailed(
            cudaMemcpy(this->targetTokenIdsBuffer.deviceData, this->packH2dDevice.deviceData + tokenCount, bytes, cudaMemcpyDeviceToDevice),
            "CudaLanguageModel::flushPackedHostBuffers D2D targets");
        CudaMatmul::throwIfCudaFailed(
            cudaMemcpy(this->meanDivisorBuffer.deviceData, this->packH2dDevice.deviceData + 2 * tokenCount, bytes, cudaMemcpyDeviceToDevice),
            "CudaLanguageModel::flushPackedHostBuffers D2D meanDivisors");
    }

    this->adamWindowTokenIds.insert(
        this->adamWindowTokenIds.end(),
        this->packedInputTokenIds.begin(),
        this->packedInputTokenIds.end());

    if (useGraphPath && this->tryLaunchTrainGraph(segmentLength, exampleCount))
        return 0.0f;

    if (useGraphPath
        && this->trainGraphWarmups >= CudaLanguageModel::trainGraphWarmupNeeded
        && this->trainGraphExec == nullptr) {
        try {
            this->captureTrainGraph(tokenCount, segmentLength, exampleCount, gradients);
            // Capture records only — launch once so this microstep still applies grads.
            CudaMatmul::throwIfCudaFailed(cudaGraphLaunch(this->trainGraphExec, this->trainStream), "flushPackedHostBuffers post-capture launch");
            CudaMatmul::throwIfCudaFailed(cudaStreamSynchronize(this->trainStream), "flushPackedHostBuffers capture sync");
            return 0.0f;
        } catch (const std::exception&) {
            this->releaseTrainGraph();
            this->preferTrainGraph = false;
            // Fall through to eager path once; subsequent steps stay eager.
        }
    }

    if (useGraphPath && this->trainStream != nullptr) {
        const cudaStream_t previous = CudaMatmul::setActiveStream(this->trainStream);
        try {
            this->runPackedTrainDevice(tokenCount, segmentLength, exampleCount, gradients);
            CudaMatmul::throwIfCudaFailed(cudaStreamSynchronize(this->trainStream), "flushPackedHostBuffers eager sync");
        } catch (...) {
            CudaMatmul::setActiveStream(previous);
            throw;
        }
        CudaMatmul::setActiveStream(previous);
        ++this->trainGraphWarmups;
        return 0.0f;
    }

    this->runPackedTrainDevice(tokenCount, segmentLength, exampleCount, gradients);
    return 0.0f;
}

float CudaLanguageModel::accumulatePackedExamples(const LanguageModelExample* const* examples, int exampleCount, CudaLanguageModelGradients& gradients) {
    if (examples == nullptr || exampleCount <= 0) throw std::invalid_argument("CudaLanguageModel::accumulatePackedExamples empty examples");

    const size_t segmentTokenCount = examples[0]->inputTokenIds.size();
    if (segmentTokenCount == 0) throw std::invalid_argument("CudaLanguageModel::accumulatePackedExamples empty inputTokenIds");

    this->packedInputTokenIds.clear();
    this->packedTargetTokenIds.clear();
    this->packedMeanDivisors.clear();
    this->packedInputTokenIds.reserve(segmentTokenCount * static_cast<size_t>(exampleCount));
    this->packedTargetTokenIds.reserve(segmentTokenCount * static_cast<size_t>(exampleCount));
    this->packedMeanDivisors.reserve(segmentTokenCount * static_cast<size_t>(exampleCount));

    for (int exampleIndex = 0; exampleIndex < exampleCount; ++exampleIndex) {
        const LanguageModelExample& example = *examples[exampleIndex];
        if (example.inputTokenIds.size() != segmentTokenCount)
            throw std::invalid_argument("CudaLanguageModel::accumulatePackedExamples unequal input lengths");
        if (example.targetTokenIds.size() != segmentTokenCount)
            throw std::invalid_argument("CudaLanguageModel::accumulatePackedExamples target length mismatch");
        this->packedInputTokenIds.insert(this->packedInputTokenIds.end(), example.inputTokenIds.begin(), example.inputTokenIds.end());
        this->packedTargetTokenIds.insert(this->packedTargetTokenIds.end(), example.targetTokenIds.begin(), example.targetTokenIds.end());
        for (size_t position = 0; position < segmentTokenCount; ++position)
            this->packedMeanDivisors.push_back(static_cast<int>(segmentTokenCount));
    }

    return this->flushPackedHostBuffers(static_cast<int>(segmentTokenCount), exampleCount, gradients);
}

float CudaLanguageModel::accumulateBucketPackedExamples(const LanguageModelExample* const* examples, int exampleCount, int bucketLength, CudaLanguageModelGradients& gradients) {
    if (examples == nullptr || exampleCount <= 0) throw std::invalid_argument("CudaLanguageModel::accumulateBucketPackedExamples empty examples");
    if (bucketLength <= 0) throw std::invalid_argument("CudaLanguageModel::accumulateBucketPackedExamples bucketLength must be > 0");
    if (bucketLength > this->maximumPositionCount)
        throw std::invalid_argument("CudaLanguageModel::accumulateBucketPackedExamples bucket exceeds maximumPositionCount");

    this->packedInputTokenIds.clear();
    this->packedTargetTokenIds.clear();
    this->packedMeanDivisors.clear();
    this->packedInputTokenIds.reserve(static_cast<size_t>(bucketLength) * static_cast<size_t>(exampleCount));
    this->packedTargetTokenIds.reserve(static_cast<size_t>(bucketLength) * static_cast<size_t>(exampleCount));
    this->packedMeanDivisors.reserve(static_cast<size_t>(bucketLength) * static_cast<size_t>(exampleCount));

    for (int exampleIndex = 0; exampleIndex < exampleCount; ++exampleIndex) {
        const LanguageModelExample& example = *examples[exampleIndex];
        const int trueLength = static_cast<int>(example.inputTokenIds.size());
        if (trueLength <= 0) throw std::invalid_argument("CudaLanguageModel::accumulateBucketPackedExamples empty example");
        if (static_cast<int>(example.targetTokenIds.size()) != trueLength)
            throw std::invalid_argument("CudaLanguageModel::accumulateBucketPackedExamples target length mismatch");
        if (trueLength > bucketLength)
            throw std::invalid_argument("CudaLanguageModel::accumulateBucketPackedExamples example longer than bucket");
        if (CudaLanguageModel::lengthBucket(trueLength, this->maximumPositionCount) != bucketLength)
            throw std::invalid_argument("CudaLanguageModel::accumulateBucketPackedExamples bucket mismatch");

        this->packedInputTokenIds.insert(this->packedInputTokenIds.end(), example.inputTokenIds.begin(), example.inputTokenIds.end());
        this->packedTargetTokenIds.insert(this->packedTargetTokenIds.end(), example.targetTokenIds.begin(), example.targetTokenIds.end());
        for (int position = trueLength; position < bucketLength; ++position) {
            this->packedInputTokenIds.push_back(CudaLanguageModel::padInputId);
            this->packedTargetTokenIds.push_back(CudaLanguageModel::padTargetId);
        }
        for (int position = 0; position < bucketLength; ++position)
            this->packedMeanDivisors.push_back(trueLength);
    }

    return this->flushPackedHostBuffers(bucketLength, exampleCount, gradients);
}

int CudaLanguageModel::lengthBucket(int trueLength, int maximumPositionCount) {
    if (maximumPositionCount <= 0) throw std::invalid_argument("CudaLanguageModel::lengthBucket maximumPositionCount must be > 0");
    if (trueLength <= 0) return (std::min)(CudaLanguageModel::lengthBucketStep, maximumPositionCount);
    if (trueLength > maximumPositionCount) trueLength = maximumPositionCount;

    int bucket = ((trueLength + CudaLanguageModel::lengthBucketStep - 1) / CudaLanguageModel::lengthBucketStep) * CudaLanguageModel::lengthBucketStep;
    if (bucket > maximumPositionCount) bucket = maximumPositionCount;
    if (bucket < trueLength) bucket = trueLength;
    return bucket;
}

void CudaLanguageModel::forwardTrunkInto(const std::vector<int>& tokenIds, int segmentLength) {
    if (tokenIds.empty()) throw std::invalid_argument("CudaLanguageModel::forwardTrunkInto empty tokenIds");
    if (this->tokenEmbeddingWeight.empty()) throw std::logic_error("CudaLanguageModel::forwardTrunkInto weights not uploaded");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaLanguageModel::forwardTrunkInto no CUDA device");

    if (segmentLength > 0) {
        if (static_cast<int>(tokenIds.size()) % segmentLength != 0)
            throw std::invalid_argument("CudaLanguageModel::forwardTrunkInto tokenCount not divisible by segmentLength");
        if (segmentLength > this->maximumPositionCount)
            throw std::invalid_argument("CudaLanguageModel::forwardTrunkInto segmentLength exceeds maximumPositionCount");
    } else if (static_cast<int>(tokenIds.size()) > this->maximumPositionCount) {
        throw std::invalid_argument("CudaLanguageModel::forwardTrunkInto sequence longer than maximumPositionCount");
    }

    this->tokenIdsBuffer.ensureCapacity(tokenIds.size());
    this->tokenIdsBuffer.copyFromHost(tokenIds.data(), tokenIds.size());
    this->forwardTrunkFromDevice(tokenIds.size(), segmentLength);
}

void CudaLanguageModel::forwardTrunkFromDevice(size_t tokenCount, int segmentLength) {
    if (tokenCount == 0) throw std::invalid_argument("CudaLanguageModel::forwardTrunkFromDevice empty tokenCount");
    if (this->tokenEmbeddingWeight.empty()) throw std::logic_error("CudaLanguageModel::forwardTrunkFromDevice weights not uploaded");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaLanguageModel::forwardTrunkFromDevice no CUDA device");
    if (this->tokenIdsBuffer.deviceData == nullptr || tokenCount > this->tokenIdsBuffer.capacityCount)
        throw std::invalid_argument("CudaLanguageModel::forwardTrunkFromDevice tokenIdsBuffer not ready");

    if (segmentLength > 0) {
        if (static_cast<int>(tokenCount) % segmentLength != 0)
            throw std::invalid_argument("CudaLanguageModel::forwardTrunkFromDevice tokenCount not divisible by segmentLength");
        if (segmentLength > this->maximumPositionCount)
            throw std::invalid_argument("CudaLanguageModel::forwardTrunkFromDevice segmentLength exceeds maximumPositionCount");
    } else if (static_cast<int>(tokenCount) > this->maximumPositionCount) {
        throw std::invalid_argument("CudaLanguageModel::forwardTrunkFromDevice sequence longer than maximumPositionCount");
    }

    CudaOps::embeddingGatherInto(this->tokenEmbeddingWeight, this->tokenIdsBuffer, tokenCount, this->hidden);

    if (this->activationCheckpointingActive()) {
        if (this->useHalfActivationCheckpoints()) {
            if (this->blockInputCheckpointsHalf.size() != this->blocks.size())
                this->blockInputCheckpointsHalf.resize(this->blocks.size());
            for (CudaHalfMatrix& checkpoint : this->blockInputCheckpointsHalf)
                checkpoint.ensureSize(this->tokenEmbeddingWeight.cols, tokenCount);
            this->checkpointRestoreScratch.ensureSize(this->tokenEmbeddingWeight.cols, tokenCount);
        } else {
            if (this->blockInputCheckpoints.size() != this->blocks.size())
                this->blockInputCheckpoints.resize(this->blocks.size());
            for (CudaMatrix& checkpoint : this->blockInputCheckpoints)
                checkpoint.ensureSize(this->tokenEmbeddingWeight.cols, tokenCount);
        }
    }

    for (size_t blockIndex = 0; blockIndex < this->blocks.size(); ++blockIndex) {
        if (this->activationCheckpointingActive()) {
            if (this->useHalfActivationCheckpoints())
                CudaAmp::castToHalfSaturated(this->hidden, this->blockInputCheckpointsHalf[blockIndex]);
            else
                CudaOps::copyInto(this->hidden, this->blockInputCheckpoints[blockIndex]);
        }
        if (this->activationCheckpointMode == ActivationCheckpointMode::Selective)
            this->blocks[blockIndex].forwardSelectiveTrain(this->hidden, this->normalized, segmentLength);
        else
            this->blocks[blockIndex].forward(this->hidden, this->normalized, segmentLength);
        CudaMatrix swapBuffer = std::move(this->hidden);
        this->hidden = std::move(this->normalized);
        this->normalized = std::move(swapBuffer);
    }

    this->finalNorm.forward(this->hidden, this->normalized);
}

void CudaLanguageModel::zeroHostWeightGradients() {
    if (!CudaAdam::preferHostGradients) return;
    for (CudaTransformerBlockHostAdamStates& hosts : this->hostBlockAdamStates) {
        if (!hosts.queryWeightGrad.empty()) hosts.queryWeightGrad.fill(0.0f);
        if (!hosts.keyWeightGrad.empty()) hosts.keyWeightGrad.fill(0.0f);
        if (!hosts.valueWeightGrad.empty()) hosts.valueWeightGrad.fill(0.0f);
        if (!hosts.attentionOutputWeightGrad.empty()) hosts.attentionOutputWeightGrad.fill(0.0f);
        if (!hosts.feedForwardGateWeightGrad.empty()) hosts.feedForwardGateWeightGrad.fill(0.0f);
        if (!hosts.feedForwardUpWeightGrad.empty()) hosts.feedForwardUpWeightGrad.fill(0.0f);
        if (!hosts.feedForwardDownWeightGrad.empty()) hosts.feedForwardDownWeightGrad.fill(0.0f);
    }
    if (!this->hostProjectionWeightGrad.empty())
        this->hostProjectionWeightGrad.fill(0.0f);
}

void CudaLanguageModel::zeroAccumulatedGradients() {
    this->trainGradients.zeroInPlace();
    this->zeroHostWeightGradients();
}

void CudaLanguageModel::zeroTrainGradientsAfterAdam() {
    this->trainGradients.zeroInPlaceExceptEmbedding();
    this->zeroHostWeightGradients();
    if (this->adamWindowTokenIds.empty()) {
        CudaOps::zeroInPlace(this->trainGradients.tokenEmbedding);
        return;
    }

    const size_t vocabularySize = this->trainGradients.tokenEmbedding.rows;
    std::sort(this->adamWindowTokenIds.begin(), this->adamWindowTokenIds.end());
    this->adamWindowTokenIds.erase(
        std::unique(this->adamWindowTokenIds.begin(), this->adamWindowTokenIds.end()),
        this->adamWindowTokenIds.end());

    // Dense memset is cheaper once unique rows approach vocab size.
    if (this->adamWindowTokenIds.size() * 2 >= vocabularySize) {
        CudaOps::zeroInPlace(this->trainGradients.tokenEmbedding);
        this->adamWindowTokenIds.clear();
        return;
    }

    this->adamWindowTokenIdsBuffer.ensureCapacity(this->adamWindowTokenIds.size());
    this->adamWindowTokenIdsBuffer.copyFromHost(this->adamWindowTokenIds.data(), this->adamWindowTokenIds.size());
    CudaOps::embeddingZeroRows(
        this->trainGradients.tokenEmbedding,
        this->adamWindowTokenIdsBuffer.deviceData,
        this->adamWindowTokenIds.size());
    this->adamWindowTokenIds.clear();
}

void CudaLanguageModel::accumulateChunkedProjection(size_t tokenCount, int segmentLength, int exampleCountInPack, CudaLanguageModelGradients& gradients) {
    if (tokenCount == 0) throw std::invalid_argument("CudaLanguageModel::accumulateChunkedProjection empty tokenCount");
    if (segmentLength <= 0) throw std::invalid_argument("CudaLanguageModel::accumulateChunkedProjection segmentLength must be > 0");
    if (exampleCountInPack <= 0) throw std::invalid_argument("CudaLanguageModel::accumulateChunkedProjection exampleCountInPack must be > 0");
    if (static_cast<size_t>(segmentLength) * static_cast<size_t>(exampleCountInPack) != tokenCount)
        throw std::invalid_argument("CudaLanguageModel::accumulateChunkedProjection tokenCount mismatch");

    const CudaMatrix& headWeight = this->lmHeadWeight();
    if (headWeight.empty()) throw std::logic_error("CudaLanguageModel::accumulateChunkedProjection missing LM head weight");
    if (this->projectionBias.empty()) throw std::logic_error("CudaLanguageModel::accumulateChunkedProjection missing projection bias");

    CudaMatrix& headWeightGradient = this->tieEmbeddingProjection
        ? gradients.tokenEmbedding
        : gradients.projectionWeight;
    if (headWeightGradient.empty())
        throw std::logic_error("CudaLanguageModel::accumulateChunkedProjection missing LM head gradient");

    const int vocabularySize = static_cast<int>(headWeight.rows);
    const int embeddingDim = static_cast<int>(headWeight.cols);
    const float gradScale = CudaAmp::lossScalingActive() ? CudaAmp::lossScaler.scale : 1.0f;

    this->targetLogits.ensureSize(1, tokenCount);
    this->hiddenGradient.ensureSize(static_cast<size_t>(embeddingDim), tokenCount);
    this->hiddenGradientChunk.ensureSize(static_cast<size_t>(embeddingDim), tokenCount);
    CudaOps::onlineSoftmaxReset(this->onlineSoftmaxMax, this->onlineSoftmaxSumExp, tokenCount);
    CudaOps::zeroInPlace(this->targetLogits);
    CudaOps::zeroInPlace(this->hiddenGradient);

    // Cache full logits when they fit: one LM-head GEMM instead of project-twice-per-chunk.
    constexpr size_t maxCachedLogitsBytes = 512ull * 1024ull * 1024ull;
    const size_t logitsBytes = static_cast<size_t>(vocabularySize) * tokenCount * sizeof(float);
    const bool cacheFullLogits = logitsBytes <= maxCachedLogitsBytes;

    auto projectChunk = [&](int rowStart, int chunkRows, float* destination) {
        const float* weightRows = headWeight.buffer.deviceData + static_cast<size_t>(rowStart) * static_cast<size_t>(embeddingDim);
        // LM-head stays FP32 even when amp is on — CE is numerically fragile in FP16
        const bool previousAmp = CudaAmp::preferMixedPrecision;
        CudaAmp::preferMixedPrecision = false;
        CudaMatrix::multiplyPointersInto(
            weightRows, static_cast<size_t>(chunkRows), static_cast<size_t>(embeddingDim),
            this->normalized.buffer.deviceData, static_cast<size_t>(embeddingDim), tokenCount,
            destination,
            false, false);
        CudaAmp::preferMixedPrecision = previousAmp;
    };

    if (cacheFullLogits) {
        this->logits.ensureSize(static_cast<size_t>(vocabularySize), tokenCount);
        const bool previousAmp = CudaAmp::preferMixedPrecision;
        CudaAmp::preferMixedPrecision = false;
        CudaMatrix::multiplyBiasInto(headWeight, this->normalized, this->projectionBias, this->logits);
        CudaAmp::preferMixedPrecision = previousAmp;

        const int chunkCap = (std::min)(this->logitChunkRows, vocabularySize);
        for (int rowStart = 0; rowStart < vocabularySize; rowStart += chunkCap) {
            const int chunkRows = (std::min)(chunkCap, vocabularySize - rowStart);
            const float* chunkPtr = this->logits.buffer.deviceData + static_cast<size_t>(rowStart) * tokenCount;
            CudaOps::onlineSoftmaxUpdateFromChunk(chunkPtr, chunkRows, tokenCount, this->onlineSoftmaxMax, this->onlineSoftmaxSumExp);
            CudaOps::captureTargetLogitFromChunk(chunkPtr, this->targetTokenIdsBuffer, rowStart, chunkRows, tokenCount, this->targetLogits);
        }

        CudaOps::onlineSoftmaxAddMeanCrossEntropy(
            this->targetLogits, this->onlineSoftmaxMax, this->onlineSoftmaxSumExp, tokenCount,
            this->epochLossSum, 1.0f, segmentLength,
            &this->targetTokenIdsBuffer, CudaLanguageModel::padTargetId, &this->meanDivisorBuffer);

        for (int rowStart = 0; rowStart < vocabularySize; rowStart += chunkCap) {
            const int chunkRows = (std::min)(chunkCap, vocabularySize - rowStart);
            const float* chunkPtr = this->logits.buffer.deviceData + static_cast<size_t>(rowStart) * tokenCount;
            CudaOps::onlineSoftmaxLogitGradientChunkInto(
                chunkPtr, this->targetTokenIdsBuffer, rowStart, chunkRows, tokenCount,
                this->onlineSoftmaxMax, this->onlineSoftmaxSumExp, this->logitGradientChunk, gradScale, segmentLength,
                CudaLanguageModel::padTargetId, &this->meanDivisorBuffer);

            this->projectionWeightGradientChunk.ensureSize(static_cast<size_t>(chunkRows), static_cast<size_t>(embeddingDim));
            const bool previousAmpGrad = CudaAmp::preferMixedPrecision;
            CudaAmp::preferMixedPrecision = false;
            CudaMatrix::multiplyInto(this->logitGradientChunk, this->normalized, this->projectionWeightGradientChunk, false, true);
            CudaOps::addRowsInPlace(headWeightGradient, rowStart, this->projectionWeightGradientChunk);
            CudaOps::sumColumnsAddIntoRows(this->logitGradientChunk, gradients.projectionBias, rowStart);

            const float* weightRows = headWeight.buffer.deviceData + static_cast<size_t>(rowStart) * static_cast<size_t>(embeddingDim);
            CudaMatrix::multiplyPointersInto(
                weightRows, static_cast<size_t>(chunkRows), static_cast<size_t>(embeddingDim),
                this->logitGradientChunk.buffer.deviceData, static_cast<size_t>(chunkRows), tokenCount,
                this->hiddenGradientChunk.buffer.deviceData,
                true, false);
            CudaAmp::preferMixedPrecision = previousAmpGrad;
            this->hiddenGradientChunk.rows = static_cast<size_t>(embeddingDim);
            this->hiddenGradientChunk.cols = tokenCount;
            CudaOps::addInPlace(this->hiddenGradient, this->hiddenGradientChunk);
        }
        return;
    }

    const int chunkCap = (std::min)(this->logitChunkRows, vocabularySize);
    for (int rowStart = 0; rowStart < vocabularySize; rowStart += chunkCap) {
        const int chunkRows = (std::min)(chunkCap, vocabularySize - rowStart);
        this->logitChunk.ensureSize(static_cast<size_t>(chunkRows), tokenCount);
        projectChunk(rowStart, chunkRows, this->logitChunk.buffer.deviceData);
        CudaOps::broadcastBiasRowsAddInPlace(this->logitChunk, this->projectionBias, rowStart, chunkRows);
        CudaOps::onlineSoftmaxUpdateFromChunk(this->logitChunk, chunkRows, tokenCount, this->onlineSoftmaxMax, this->onlineSoftmaxSumExp);
        CudaOps::captureTargetLogitFromChunk(this->logitChunk, this->targetTokenIdsBuffer, rowStart, chunkRows, tokenCount, this->targetLogits);
    }

    CudaOps::onlineSoftmaxAddMeanCrossEntropy(
        this->targetLogits, this->onlineSoftmaxMax, this->onlineSoftmaxSumExp, tokenCount,
        this->epochLossSum, 1.0f, segmentLength,
        &this->targetTokenIdsBuffer, CudaLanguageModel::padTargetId, &this->meanDivisorBuffer);

    for (int rowStart = 0; rowStart < vocabularySize; rowStart += chunkCap) {
        const int chunkRows = (std::min)(chunkCap, vocabularySize - rowStart);
        this->logitChunk.ensureSize(static_cast<size_t>(chunkRows), tokenCount);
        projectChunk(rowStart, chunkRows, this->logitChunk.buffer.deviceData);
        CudaOps::broadcastBiasRowsAddInPlace(this->logitChunk, this->projectionBias, rowStart, chunkRows);
        CudaOps::onlineSoftmaxLogitGradientChunkInto(
            this->logitChunk, this->targetTokenIdsBuffer, rowStart, chunkRows, tokenCount,
            this->onlineSoftmaxMax, this->onlineSoftmaxSumExp, this->logitGradientChunk, gradScale, segmentLength,
            CudaLanguageModel::padTargetId, &this->meanDivisorBuffer);

        this->projectionWeightGradientChunk.ensureSize(static_cast<size_t>(chunkRows), static_cast<size_t>(embeddingDim));
        const bool previousAmp = CudaAmp::preferMixedPrecision;
        CudaAmp::preferMixedPrecision = false;
        CudaMatrix::multiplyInto(this->logitGradientChunk, this->normalized, this->projectionWeightGradientChunk, false, true);
        CudaOps::addRowsInPlace(headWeightGradient, rowStart, this->projectionWeightGradientChunk);
        CudaOps::sumColumnsAddIntoRows(this->logitGradientChunk, gradients.projectionBias, rowStart);

        const float* weightRows = headWeight.buffer.deviceData + static_cast<size_t>(rowStart) * static_cast<size_t>(embeddingDim);
        CudaMatrix::multiplyPointersInto(
            weightRows, static_cast<size_t>(chunkRows), static_cast<size_t>(embeddingDim),
            this->logitGradientChunk.buffer.deviceData, static_cast<size_t>(chunkRows), tokenCount,
            this->hiddenGradientChunk.buffer.deviceData,
            true, false);
        CudaAmp::preferMixedPrecision = previousAmp;
        this->hiddenGradientChunk.rows = static_cast<size_t>(embeddingDim);
        this->hiddenGradientChunk.cols = tokenCount;
        CudaOps::addInPlace(this->hiddenGradient, this->hiddenGradientChunk);
    }
}

void CudaLanguageModel::applyGradients(CudaLanguageModelGradients& gradients, float gradientScale) {
    if (!this->trainStateReady) throw std::logic_error("CudaLanguageModel::applyGradients train state not ready");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaLanguageModel::applyGradients no CUDA device");

    float effectiveGradientScale = gradientScale;
    if (CudaAmp::lossScalingActive()) {
        // unscale first so overflow checks and Adam/Muon see true grad magnitudes
        const float inverseLossScale = 1.0f / CudaAmp::lossScaler.scale;
        gradients.scaleInPlace(inverseLossScale);
        if (CudaAmp::gradientsHaveNonFinite(gradients)) {
            CudaAmp::lossScaler.updateOnOverflow();
            // Critical: leave Inf/NaN grads in the buffer and every later microbatch stays poisoned.
            gradients.zeroInPlace();
            this->adamWindowTokenIds.clear();
            return;
        }
        effectiveGradientScale = gradientScale;
    }

    this->adam.step();
    this->muon.learningRate = this->adam.learningRate;

    auto syncFusedMirrors = [this]() {
        if (CudaAdam::preferFp16GpuWeights) {
            for (size_t blockIndex = 0; blockIndex < this->blocks.size(); ++blockIndex) {
                CudaTransformerBlock& block = this->blocks[blockIndex];
                CudaTransformerBlockHostAdamStates& hosts = this->hostBlockAdamStates[blockIndex];
                Matrix qkvHost(block.attention.queryWeight.rows * 3ull, block.attention.queryWeight.cols, 0.0f);
                const size_t slice = hosts.queryWeightMaster.data.size();
                std::memcpy(qkvHost.data.data(), hosts.queryWeightMaster.data.data(), slice * sizeof(float));
                std::memcpy(qkvHost.data.data() + slice, hosts.keyWeightMaster.data.data(), slice * sizeof(float));
                std::memcpy(qkvHost.data.data() + 2 * slice, hosts.valueWeightMaster.data.data(), slice * sizeof(float));
                CudaAmp::uploadHostMasterToFp16Working(block.attention.qkvWeight, qkvHost.data.data());

                Matrix gateUpHost(block.feedForward.gateWeight.rows + block.feedForward.upWeight.rows, block.feedForward.gateWeight.cols, 0.0f);
                const size_t gateSlice = hosts.feedForwardGateWeightMaster.data.size();
                std::memcpy(gateUpHost.data.data(), hosts.feedForwardGateWeightMaster.data.data(), gateSlice * sizeof(float));
                std::memcpy(gateUpHost.data.data() + gateSlice, hosts.feedForwardUpWeightMaster.data.data(), hosts.feedForwardUpWeightMaster.data.size() * sizeof(float));
                CudaAmp::uploadHostMasterToFp16Working(block.feedForward.gateUpWeight, gateUpHost.data.data());
                block.feedForward.syncFusedGateUpWeight(); // bias only when fp16 slot set
            }
            return;
        }
        for (CudaTransformerBlock& block : this->blocks) {
            block.attention.syncFusedQkvWeight();
            block.feedForward.syncFusedGateUpWeight();
        }
        CudaAmp::invalidateMasterWeightHalves();
    };

    auto applyMuonBlockWeights = [this, &gradients, effectiveGradientScale]() {
        if (!this->preferMuon) return;
        if (this->blockMuonStates.size() != this->blocks.size())
            throw std::logic_error("CudaLanguageModel::applyGradients Muon states not ready");
        for (size_t blockIndex = 0; blockIndex < this->blocks.size(); ++blockIndex) {
            CudaTransformerBlock& block = this->blocks[blockIndex];
            CudaTransformerBlockGradients& blockGradients = gradients.blocks[blockIndex];
            CudaTransformerBlockMuonStates& muonStates = this->blockMuonStates[blockIndex];

            // Fused QKV: one NS for Q/K/V instead of three
            block.attention.syncFusedQkvWeight();
            block.attention.qkvWeightGradient.ensureSize(block.attention.qkvWeight.rows, block.attention.qkvWeight.cols);
            const size_t qSliceBytes = block.attention.queryWeight.byteCount();
            CudaMatmul::memcpyDevice(block.attention.qkvWeightGradient.buffer.deviceData, blockGradients.queryWeight.buffer.deviceData, qSliceBytes);
            CudaMatmul::memcpyDevice(block.attention.qkvWeightGradient.buffer.deviceData + block.attention.queryWeight.elementCount(), blockGradients.keyWeight.buffer.deviceData, qSliceBytes);
            CudaMatmul::memcpyDevice(block.attention.qkvWeightGradient.buffer.deviceData + 2ull * block.attention.queryWeight.elementCount(), blockGradients.valueWeight.buffer.deviceData, qSliceBytes);
            this->muon.update(block.attention.qkvWeight, muonStates.qkvWeight, block.attention.qkvWeightGradient, effectiveGradientScale);
            CudaMatmul::memcpyDevice(block.attention.queryWeight.buffer.deviceData, block.attention.qkvWeight.buffer.deviceData, qSliceBytes);
            CudaMatmul::memcpyDevice(block.attention.keyWeight.buffer.deviceData, block.attention.qkvWeight.buffer.deviceData + block.attention.queryWeight.elementCount(), qSliceBytes);
            CudaMatmul::memcpyDevice(block.attention.valueWeight.buffer.deviceData, block.attention.qkvWeight.buffer.deviceData + 2ull * block.attention.queryWeight.elementCount(), qSliceBytes);

            this->muon.update(block.attention.outputWeight, muonStates.attentionOutputWeight, blockGradients.attentionOutputWeight, effectiveGradientScale);

            // Fused gate+up: one NS instead of two
            block.feedForward.syncFusedGateUpWeight();
            block.feedForward.gateUpWeightGradient.ensureSize(block.feedForward.gateUpWeight.rows, block.feedForward.gateUpWeight.cols);
            const size_t gateSliceBytes = block.feedForward.gateWeight.byteCount();
            CudaMatmul::memcpyDevice(block.feedForward.gateUpWeightGradient.buffer.deviceData, blockGradients.feedForwardGateWeight.buffer.deviceData, gateSliceBytes);
            CudaMatmul::memcpyDevice(block.feedForward.gateUpWeightGradient.buffer.deviceData + block.feedForward.gateWeight.elementCount(), blockGradients.feedForwardUpWeight.buffer.deviceData, block.feedForward.upWeight.byteCount());
            this->muon.update(block.feedForward.gateUpWeight, muonStates.feedForwardGateUpWeight, block.feedForward.gateUpWeightGradient, effectiveGradientScale);
            CudaMatmul::memcpyDevice(block.feedForward.gateWeight.buffer.deviceData, block.feedForward.gateUpWeight.buffer.deviceData, gateSliceBytes);
            CudaMatmul::memcpyDevice(block.feedForward.upWeight.buffer.deviceData, block.feedForward.gateUpWeight.buffer.deviceData + block.feedForward.gateWeight.elementCount(), block.feedForward.upWeight.byteCount());

            this->muon.update(block.feedForward.downWeight, muonStates.feedForwardDownWeight, blockGradients.feedForwardDownWeight, effectiveGradientScale);
        }
    };

    if (CudaAdam::preferCpuOffload) {
        if (this->hostBlockAdamStates.size() != this->blocks.size())
            this->hostBlockAdamStates.resize(this->blocks.size());

        std::vector<CudaAdamCpuOffloadItem> offloadItems;
        offloadItems.reserve(16 + this->blocks.size() * 12);

        auto pushOffload = [&offloadItems](CudaMatrix& parameter, AdamState& state, CudaMatrix* deviceGradient, Matrix* hostMaster, Matrix* hostGradient) {
            if (parameter.elementCount() == 0) throw std::invalid_argument("CudaLanguageModel::applyGradients empty parameter");
            if (hostGradient != nullptr) {
                if (hostGradient->empty() || hostGradient->data.size() != parameter.elementCount())
                    throw std::invalid_argument("CudaLanguageModel::applyGradients hostGradient size mismatch");
            } else {
                if (deviceGradient == nullptr || deviceGradient->elementCount() != parameter.elementCount())
                    throw std::invalid_argument("CudaLanguageModel::applyGradients gradient/parameter size mismatch");
            }
            CudaAdamCpuOffloadItem item;
            item.parameter = &parameter;
            item.gradient = deviceGradient;
            item.hostState = &state;
            item.hostMaster = hostMaster;
            item.hostGradient = hostGradient;
            offloadItems.push_back(item);
        };

        if (!this->tieEmbeddingProjection)
            pushOffload(this->projectionWeight, this->hostProjectionWeightState, &gradients.projectionWeight, &this->hostProjectionWeightMaster, nullptr);
        pushOffload(this->projectionBias, this->hostProjectionBiasState, &gradients.projectionBias, &this->hostProjectionBiasMaster, nullptr);
        pushOffload(this->finalNorm.gamma, this->hostFinalNormGammaState, &gradients.finalNormGamma, &this->hostFinalNormGammaMaster, nullptr);

        for (size_t blockIndex = 0; blockIndex < this->blocks.size(); ++blockIndex) {
            CudaTransformerBlock& block = this->blocks[blockIndex];
            CudaTransformerBlockGradients& blockGradients = gradients.blocks[blockIndex];
            CudaTransformerBlockHostAdamStates& blockStates = this->hostBlockAdamStates[blockIndex];
            const bool hostLarge = CudaAdam::preferHostGradients;

            if (!this->preferMuon) {
                pushOffload(block.attention.queryWeight, blockStates.queryWeight, hostLarge ? nullptr : &blockGradients.queryWeight, &blockStates.queryWeightMaster, hostLarge ? &blockStates.queryWeightGrad : nullptr);
                pushOffload(block.attention.keyWeight, blockStates.keyWeight, hostLarge ? nullptr : &blockGradients.keyWeight, &blockStates.keyWeightMaster, hostLarge ? &blockStates.keyWeightGrad : nullptr);
                pushOffload(block.attention.valueWeight, blockStates.valueWeight, hostLarge ? nullptr : &blockGradients.valueWeight, &blockStates.valueWeightMaster, hostLarge ? &blockStates.valueWeightGrad : nullptr);
                pushOffload(block.attention.outputWeight, blockStates.attentionOutputWeight, hostLarge ? nullptr : &blockGradients.attentionOutputWeight, &blockStates.attentionOutputWeightMaster, hostLarge ? &blockStates.attentionOutputWeightGrad : nullptr);
                pushOffload(block.feedForward.gateWeight, blockStates.feedForwardGateWeight, hostLarge ? nullptr : &blockGradients.feedForwardGateWeight, &blockStates.feedForwardGateWeightMaster, hostLarge ? &blockStates.feedForwardGateWeightGrad : nullptr);
                pushOffload(block.feedForward.upWeight, blockStates.feedForwardUpWeight, hostLarge ? nullptr : &blockGradients.feedForwardUpWeight, &blockStates.feedForwardUpWeightMaster, hostLarge ? &blockStates.feedForwardUpWeightGrad : nullptr);
                pushOffload(block.feedForward.downWeight, blockStates.feedForwardDownWeight, hostLarge ? nullptr : &blockGradients.feedForwardDownWeight, &blockStates.feedForwardDownWeightMaster, hostLarge ? &blockStates.feedForwardDownWeightGrad : nullptr);
            }
            pushOffload(block.attentionNorm.gamma, blockStates.attentionNormGamma, &blockGradients.attentionNormGamma, &blockStates.attentionNormGammaMaster, nullptr);
            pushOffload(block.feedForwardNorm.gamma, blockStates.feedForwardNormGamma, &blockGradients.feedForwardNormGamma, &blockStates.feedForwardNormGammaMaster, nullptr);
            pushOffload(block.feedForward.gateBias, blockStates.feedForwardGateBias, &blockGradients.feedForwardGateBias, &blockStates.feedForwardGateBiasMaster, nullptr);
            pushOffload(block.feedForward.upBias, blockStates.feedForwardUpBias, &blockGradients.feedForwardUpBias, &blockStates.feedForwardUpBiasMaster, nullptr);
            pushOffload(block.feedForward.downBias, blockStates.feedForwardDownBias, &blockGradients.feedForwardDownBias, &blockStates.feedForwardDownBiasMaster, nullptr);
        }

        pushOffload(this->tokenEmbeddingWeight, this->hostTokenEmbeddingState, &gradients.tokenEmbedding, &this->hostTokenEmbeddingMaster, nullptr);
        applyMuonBlockWeights();
        this->adam.updateCpuOffloadedMany(offloadItems.data(), static_cast<int>(offloadItems.size()), effectiveGradientScale);
        syncFusedMirrors();

        if (CudaAmp::lossScalingActive())
            CudaAmp::lossScaler.updateOnSuccess();
        return;
    }

    auto ensureMoments = [](CudaAdamState& state, const CudaMatrix& parameter) {
        if (!state.empty()) return;
        if (CudaAdam::preferInt8Moments)
            state.ensureInt8(parameter, CudaAdam::int8BlockSize);
        else
            state.ensureFp32(parameter);
    };

    std::vector<CudaAdamUpdateItem> items;
    items.reserve(16 + this->blocks.size() * 12);

    auto pushItem = [&items, &ensureMoments](CudaMatrix& parameter, CudaAdamState& state, const CudaMatrix& gradient) {
        if (parameter.elementCount() == 0) throw std::invalid_argument("CudaLanguageModel::applyGradients empty parameter");
        if (gradient.elementCount() != parameter.elementCount())
            throw std::invalid_argument("CudaLanguageModel::applyGradients gradient/parameter size mismatch");
        ensureMoments(state, parameter);
        CudaAdamUpdateItem item;
        item.parameter = parameter.buffer.deviceData;
        item.gradient = gradient.buffer.deviceData;
        item.elementCount = static_cast<int>(parameter.elementCount());
        item.useInt8 = CudaAdam::preferInt8Moments;
        if (item.useInt8) {
            item.firstMoment = nullptr;
            item.secondMoment = nullptr;
            item.firstMomentQ = state.firstMomentQ.deviceData;
            item.secondMomentQ = state.secondMomentQ.deviceData;
            item.firstMomentScales = state.firstMomentScales.deviceData;
            item.secondMomentScales = state.secondMomentScales.deviceData;
            item.scaleCount = state.scaleCount;
        } else {
            item.firstMoment = state.firstMoment.buffer.deviceData;
            item.secondMoment = state.secondMoment.buffer.deviceData;
            item.firstMomentQ = nullptr;
            item.secondMomentQ = nullptr;
            item.firstMomentScales = nullptr;
            item.secondMomentScales = nullptr;
            item.scaleCount = 0;
        }
        items.push_back(item);
    };

    if (!this->tieEmbeddingProjection)
        pushItem(this->projectionWeight, this->projectionWeightState, gradients.projectionWeight);
    pushItem(this->projectionBias, this->projectionBiasState, gradients.projectionBias);
    pushItem(this->finalNorm.gamma, this->finalNormGammaState, gradients.finalNormGamma);

    for (size_t blockIndex = 0; blockIndex < this->blocks.size(); ++blockIndex) {
        CudaTransformerBlock& block = this->blocks[blockIndex];
        CudaTransformerBlockGradients& blockGradients = gradients.blocks[blockIndex];
        CudaTransformerBlockAdamStates& blockStates = this->blockAdamStates[blockIndex];

        if (!this->preferMuon) {
            pushItem(block.attention.queryWeight, blockStates.queryWeight, blockGradients.queryWeight);
            pushItem(block.attention.keyWeight, blockStates.keyWeight, blockGradients.keyWeight);
            pushItem(block.attention.valueWeight, blockStates.valueWeight, blockGradients.valueWeight);
            pushItem(block.attention.outputWeight, blockStates.attentionOutputWeight, blockGradients.attentionOutputWeight);
            pushItem(block.feedForward.gateWeight, blockStates.feedForwardGateWeight, blockGradients.feedForwardGateWeight);
            pushItem(block.feedForward.upWeight, blockStates.feedForwardUpWeight, blockGradients.feedForwardUpWeight);
            pushItem(block.feedForward.downWeight, blockStates.feedForwardDownWeight, blockGradients.feedForwardDownWeight);
        }
        pushItem(block.attentionNorm.gamma, blockStates.attentionNormGamma, blockGradients.attentionNormGamma);
        pushItem(block.feedForwardNorm.gamma, blockStates.feedForwardNormGamma, blockGradients.feedForwardNormGamma);
        pushItem(block.feedForward.gateBias, blockStates.feedForwardGateBias, blockGradients.feedForwardGateBias);
        pushItem(block.feedForward.upBias, blockStates.feedForwardUpBias, blockGradients.feedForwardUpBias);
        pushItem(block.feedForward.downBias, blockStates.feedForwardDownBias, blockGradients.feedForwardDownBias);
    }

    pushItem(this->tokenEmbeddingWeight, this->tokenEmbeddingState, gradients.tokenEmbedding);

    // Overlap Muon NS (often dominant) with Adam aux updates on a second stream.
    if (this->preferMuon) {
        this->ensureTrainStream();
        const cudaStream_t previous = CudaMatmul::setActiveStream(this->trainStream);
        applyMuonBlockWeights();
        CudaMatmul::setActiveStream(previous);
        this->adam.updateMany(items.data(), static_cast<int>(items.size()), effectiveGradientScale);
        CudaMatmul::throwIfCudaFailed(cudaStreamSynchronize(this->trainStream), "applyGradients muon stream sync");
    } else {
        this->adam.updateMany(items.data(), static_cast<int>(items.size()), effectiveGradientScale);
    }
    syncFusedMirrors();

    if (CudaAmp::lossScalingActive())
        CudaAmp::lossScaler.updateOnSuccess();
}

float CudaLanguageModel::averageLoss(const LanguageModelDataset& dataset) {
    if (dataset.examples.empty()) return 0.0f;
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaLanguageModel::averageLoss no CUDA device");

    // Eval in FP32: train AMP/half-ckpt state must not leak Inf into the reported test metric.
    const bool previousAmp = CudaAmp::preferMixedPrecision;
    CudaAmp::preferMixedPrecision = false;

    this->epochLossSum.ensureSize(1, 1);
    CudaOps::zeroInPlace(this->epochLossSum);

    for (const LanguageModelExample& example : dataset.examples) {
        if (example.inputTokenIds.empty() || example.targetTokenIds.size() != example.inputTokenIds.size())
            continue;

        this->forwardInto(example.inputTokenIds, this->logits);
        this->targetTokenIdsBuffer.ensureCapacity(example.targetTokenIds.size());
        this->targetTokenIdsBuffer.copyFromHost(example.targetTokenIds.data(), example.targetTokenIds.size());

        // Stable CE from logits (no full softmax materialization dependency on freed train probs).
        CudaOps::softmaxCrossEntropyFromLogitsInto(
            this->logits,
            this->targetTokenIdsBuffer,
            example.targetTokenIds.size(),
            this->probabilities,
            this->logitGradient,
            this->epochLossSum,
            1.0f,
            static_cast<int>(example.targetTokenIds.size()));
    }

    CudaMatmul::throwIfCudaFailed(cudaDeviceSynchronize(), "CudaLanguageModel::averageLoss synchronize");
    CudaAmp::preferMixedPrecision = previousAmp;

    Matrix lossHost = this->epochLossSum.download();
    const float mean = lossHost.at(0, 0) / static_cast<float>(dataset.size());
    return mean;
}

void CudaLanguageModel::trainOnExamples(
    const LanguageModelDataset& dataset,
    int batchSize,
    bool flushRemainder,
    int& accumulatedExampleCount,
    int& microbatchesSinceStep,
    int& processedExampleCount,
    int& processedPredictionCount,
    int& packCount,
    int& singleExamplePackCount,
    long long& packedExampleSum,
    long long& packedTokenSum
) {
    if (batchSize <= 0) batchSize = 32;
    if (this->gradientAccumulationSteps <= 0) this->gradientAccumulationSteps = 1;

    const int exampleCount = static_cast<int>(dataset.examples.size());
    if (exampleCount > 0) {
        processedExampleCount += exampleCount;
        processedPredictionCount += dataset.totalPredictionCount();
    }
    if (exampleCount <= 0) {
        if (flushRemainder && accumulatedExampleCount > 0) {
            this->applyGradients(this->trainGradients, 1.0f / static_cast<float>(accumulatedExampleCount));
            this->zeroTrainGradientsAfterAdam();
            accumulatedExampleCount = 0;
            microbatchesSinceStep = 0;
        }
        return;
    }

    // Packing window is independent of Adam batchSize: wide enough to fill maxPackedColumns
    // at the smallest length bucket (lengthBucketStep).
    const int packWindow = (std::max)(
        batchSize,
        (std::max)(1, this->maxPackedColumns / CudaLanguageModel::lengthBucketStep)
    );
    const int examplesPerAdamStep = batchSize * this->gradientAccumulationSteps;

    std::vector<int> order(static_cast<size_t>(exampleCount));
    for (int index = 0; index < exampleCount; ++index)
        order[static_cast<size_t>(index)] = index;
    std::stable_sort(order.begin(), order.end(), [&dataset](int left, int right) {
        return dataset.examples[static_cast<size_t>(left)].inputTokenIds.size()
            < dataset.examples[static_cast<size_t>(right)].inputTokenIds.size();
    });

    std::vector<const LanguageModelExample*> packPointers;
    packPointers.reserve(static_cast<size_t>((std::max)(1, this->maxPackedColumns / CudaLanguageModel::lengthBucketStep)));

    const auto progressStart = std::chrono::steady_clock::now();
    auto lastProgress = progressStart;
    int doneExamples = 0;
    long long donePredictions = 0;

    for (int windowStart = 0; windowStart < exampleCount; windowStart += packWindow) {
        int windowEnd = windowStart + packWindow;
        if (windowEnd > exampleCount) windowEnd = exampleCount;

        int packStart = windowStart;
        while (packStart < windowEnd) {
            const int trueLength = static_cast<int>(
                dataset.examples[static_cast<size_t>(order[static_cast<size_t>(packStart)])].inputTokenIds.size()
            );
            const int bucketLength = CudaLanguageModel::lengthBucket(trueLength, this->maximumPositionCount);
            int maxExamplesInPack = 1;
            if (bucketLength > 0 && this->maxPackedColumns > 0)
                maxExamplesInPack = (std::max)(1, this->maxPackedColumns / bucketLength);

            packPointers.clear();
            int packEnd = packStart;
            while (packEnd < windowEnd
                && static_cast<int>(packPointers.size()) < maxExamplesInPack) {
                const int candidateLength = static_cast<int>(
                    dataset.examples[static_cast<size_t>(order[static_cast<size_t>(packEnd)])].inputTokenIds.size()
                );
                if (CudaLanguageModel::lengthBucket(candidateLength, this->maximumPositionCount) != bucketLength)
                    break;
                packPointers.push_back(&dataset.examples[static_cast<size_t>(order[static_cast<size_t>(packEnd)])]);
                ++packEnd;
            }

            const int packExampleCount = static_cast<int>(packPointers.size());
            ++packCount;
            if (packExampleCount <= 1) ++singleExamplePackCount;
            packedExampleSum += packExampleCount;
            packedTokenSum += static_cast<long long>(packExampleCount) * static_cast<long long>(bucketLength);

            this->accumulateBucketPackedExamples(packPointers.data(), packExampleCount, bucketLength, this->trainGradients);

            accumulatedExampleCount += packExampleCount;
            doneExamples += packExampleCount;
            for (const LanguageModelExample* example : packPointers)
                donePredictions += static_cast<long long>(example->targetTokenIds.size());

            const auto now = std::chrono::steady_clock::now();
            const double sinceProgress = std::chrono::duration<double>(now - lastProgress).count();
            if (sinceProgress >= 15.0 || doneExamples == exampleCount) {
                const double elapsed = std::chrono::duration<double>(now - progressStart).count();
                const double tokPerSec = elapsed > 0.0 ? static_cast<double>(donePredictions) / elapsed : 0.0;
                const double size1Percent = packCount > 0
                    ? 100.0 * static_cast<double>(singleExamplePackCount) / static_cast<double>(packCount)
                    : 0.0;
                std::printf(
                    "  ... progress  ex=%d/%d  tokens/s=%.0f  maxPackCols=%d  packs=%d  size1=%.1f%%  opt=%s\n",
                    doneExamples,
                    exampleCount,
                    tokPerSec,
                    this->maxPackedColumns,
                    packCount,
                    size1Percent,
                    this->preferMuon ? "muon+adam" : "adam");
                lastProgress = now;
            }

            const bool isLastPack = flushRemainder && packEnd >= exampleCount;
            if (accumulatedExampleCount >= examplesPerAdamStep || isLastPack) {
                this->applyGradients(this->trainGradients, 1.0f / static_cast<float>(accumulatedExampleCount));
                this->zeroTrainGradientsAfterAdam();
                accumulatedExampleCount = 0;
                microbatchesSinceStep = 0;
            }

            packStart = packEnd;
        }
    }

    if (flushRemainder && accumulatedExampleCount > 0) {
        this->applyGradients(this->trainGradients, 1.0f / static_cast<float>(accumulatedExampleCount));
        this->zeroTrainGradientsAfterAdam();
        accumulatedExampleCount = 0;
        microbatchesSinceStep = 0;
    }
}

void CudaLanguageModel::train(const LanguageModelDataset& trainDataset, const LanguageModelDataset& testDataset, int epochs, int logEveryEpochs, int batchSize, int gradientAccumulationSteps) {
    if (trainDataset.examples.empty()) return;
    if (logEveryEpochs <= 0) logEveryEpochs = 1;
    if (batchSize <= 0) batchSize = 32;
    if (gradientAccumulationSteps <= 0) gradientAccumulationSteps = 1;
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaLanguageModel::train no CUDA device");

    this->gradientAccumulationSteps = gradientAccumulationSteps;
    this->ensureTrainState();
    this->ensureTrainWorkspaces();

    for (int epoch = 0; epoch < epochs; ++epoch) {
        CudaOps::zeroInPlace(this->epochLossSum);
        const auto epochStart = std::chrono::steady_clock::now();

        this->trainGradients.zeroInPlace();
        this->adamWindowTokenIds.clear();
        int accumulatedExampleCount = 0;
        int microbatchesSinceStep = 0;
        int processedExampleCount = 0;
        int processedPredictionCount = 0;
        int packCount = 0;
        int singleExamplePackCount = 0;
        long long packedExampleSum = 0;
        long long packedTokenSum = 0;
        this->trainOnExamples(
            trainDataset,
            batchSize,
            true,
            accumulatedExampleCount,
            microbatchesSinceStep,
            processedExampleCount,
            processedPredictionCount,
            packCount,
            singleExamplePackCount,
            packedExampleSum,
            packedTokenSum
        );

        CudaMatmul::throwIfCudaFailed(cudaDeviceSynchronize(), "CudaLanguageModel::train epoch synchronize");
        const auto epochEnd = std::chrono::steady_clock::now();
        const double epochSeconds = std::chrono::duration<double>(epochEnd - epochStart).count();

        if (epoch % logEveryEpochs != 0) continue;

        Matrix lossHost = this->epochLossSum.download();
        const float averageTrainLoss = processedExampleCount > 0
            ? lossHost.at(0, 0) / static_cast<float>(processedExampleCount)
            : 0.0f;
        const double tokensPerSecond = epochSeconds > 0.0 ? static_cast<double>(processedPredictionCount) / epochSeconds : 0.0;
        const float trainPpl = std::exp((std::min)(averageTrainLoss, 20.0f));
        std::printf(
            "  Epoch %-3d  trainLoss=%.6f  trainPpl=%.2f  sec=%.2f  tokens/s=%.0f  backend=cuda",
            epoch,
            averageTrainLoss,
            trainPpl,
            epochSeconds,
            tokensPerSecond);

        if (CudaAmp::lossScalingActive())
            std::printf("  ampScale=%.0f  ampOverflows=%d", CudaAmp::lossScaler.scale, CudaAmp::lossScaler.overflowCount);

        if (!testDataset.examples.empty()) {
            const float testLoss = this->averageLoss(testDataset);
            const float testPpl = std::isfinite(testLoss) ? std::exp((std::min)(testLoss, 20.0f)) : testLoss;
            std::printf("  testLoss=%.6f  testPpl=%.2f", testLoss, testPpl);
        }

        if (packCount > 0) {
            const double avgPackExamples = static_cast<double>(packedExampleSum) / static_cast<double>(packCount);
            const double avgPackTokens = static_cast<double>(packedTokenSum) / static_cast<double>(packCount);
            const double size1Percent = 100.0 * static_cast<double>(singleExamplePackCount) / static_cast<double>(packCount);
            std::printf("\n  pack  packs=%d  avgEx=%.2f  avgTok=%.0f  size1=%.1f%%",
                packCount, avgPackExamples, avgPackTokens, size1Percent);
        }

        std::printf("\n");
    }
}

void CudaLanguageModel::train(LanguageModelChunkSource& source, int epochs, int logEveryEpochs, int batchSize, int gradientAccumulationSteps) {
    if (logEveryEpochs <= 0) logEveryEpochs = 1;
    if (batchSize <= 0) batchSize = 32;
    if (gradientAccumulationSteps <= 0) gradientAccumulationSteps = 1;
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaLanguageModel::train(chunk) no CUDA device");

    this->gradientAccumulationSteps = gradientAccumulationSteps;
    this->ensureTrainState();
    this->ensureTrainWorkspaces();

    LanguageModelDataset epochDataset;
    const LanguageModelDataset& testDataset = source.testDataset();

    for (int epoch = 0; epoch < epochs; ++epoch) {
        CudaOps::zeroInPlace(this->epochLossSum);
        const auto epochStart = std::chrono::steady_clock::now();

        this->trainGradients.zeroInPlace();
        this->adamWindowTokenIds.clear();
        int accumulatedExampleCount = 0;
        int microbatchesSinceStep = 0;
        int processedExampleCount = 0;
        int processedPredictionCount = 0;
        int packCount = 0;
        int singleExamplePackCount = 0;
        long long packedExampleSum = 0;
        long long packedTokenSum = 0;

        source.rewindTrain();
        LanguageModelDataset epochDataset;
        source.fillTrainDataset(epochDataset);
        if (!epochDataset.examples.empty()) {
            this->trainOnExamples(
                epochDataset,
                batchSize,
                true,
                accumulatedExampleCount,
                microbatchesSinceStep,
                processedExampleCount,
                processedPredictionCount,
                packCount,
                singleExamplePackCount,
                packedExampleSum,
                packedTokenSum
            );
        }
        if (accumulatedExampleCount > 0) {
            this->applyGradients(this->trainGradients, 1.0f / static_cast<float>(accumulatedExampleCount));
            this->zeroTrainGradientsAfterAdam();
            accumulatedExampleCount = 0;
            microbatchesSinceStep = 0;
        }

        CudaMatmul::throwIfCudaFailed(cudaDeviceSynchronize(), "CudaLanguageModel::train(chunk) epoch synchronize");
        const auto epochEnd = std::chrono::steady_clock::now();
        const double epochSeconds = std::chrono::duration<double>(epochEnd - epochStart).count();

        if (epoch % logEveryEpochs != 0) continue;

        Matrix lossHost = this->epochLossSum.download();
        const float averageTrainLoss = processedExampleCount > 0
            ? lossHost.at(0, 0) / static_cast<float>(processedExampleCount)
            : 0.0f;
        const double tokensPerSecond = epochSeconds > 0.0 ? static_cast<double>(processedPredictionCount) / epochSeconds : 0.0;
        const float trainPpl = std::exp((std::min)(averageTrainLoss, 20.0f));
        std::printf(
            "  Epoch %-3d  trainLoss=%.6f  trainPpl=%.2f  sec=%.2f  tokens/s=%.0f  backend=cuda-stream",
            epoch,
            averageTrainLoss,
            trainPpl,
            epochSeconds,
            tokensPerSecond);

        if (CudaAmp::lossScalingActive())
            std::printf("  ampScale=%.0f  ampOverflows=%d", CudaAmp::lossScaler.scale, CudaAmp::lossScaler.overflowCount);

        if (!testDataset.examples.empty()) {
            const float testLoss = this->averageLoss(testDataset);
            const float testPpl = std::isfinite(testLoss) ? std::exp((std::min)(testLoss, 20.0f)) : testLoss;
            std::printf("  testLoss=%.6f  testPpl=%.2f", testLoss, testPpl);
        }

        if (packCount > 0) {
            const double avgPackExamples = static_cast<double>(packedExampleSum) / static_cast<double>(packCount);
            const double avgPackTokens = static_cast<double>(packedTokenSum) / static_cast<double>(packCount);
            const double size1Percent = 100.0 * static_cast<double>(singleExamplePackCount) / static_cast<double>(packCount);
            std::printf("\n  pack  packs=%d  avgEx=%.2f  avgTok=%.0f  size1=%.1f%%",
                packCount, avgPackExamples, avgPackTokens, size1Percent);
        }

        std::printf("\n");
    }
}

void CudaLanguageModel::runTrainSmokeDemo(int vocabularySize, int embeddingDim, int sequenceLength, int blockCount, int headCount) {
    if (!CudaMatmul::isAvailable()) {
        SmokeLog::skip("LanguageModel train");
        return;
    }
    if (vocabularySize <= 0 || embeddingDim <= 0 || sequenceLength <= 0 || blockCount <= 0 || headCount <= 0)
        throw std::invalid_argument("CudaLanguageModel::runTrainSmokeDemo invalid dims");

    LanguageModel host(vocabularySize, embeddingDim, sequenceLength, Adam(0.001f), blockCount, headCount);
    CudaLanguageModel device = CudaLanguageModel::createFrom(host);
    device.adam = CudaAdam(host.optimizer.learningRate, host.optimizer.beta1, host.optimizer.beta2, host.optimizer.epsilon);
    const bool previousAmp = CudaAmp::preferMixedPrecision;
    const bool previousInt8 = CudaAdam::preferInt8Moments;
    CudaAmp::preferMixedPrecision = false;
    CudaAdam::preferInt8Moments = false;
    device.ensureTrainState();

    const int packBatchSize = (std::max)(1, (std::min)(16, device.maxPackedColumns / sequenceLength));
    std::vector<LanguageModelExample> examples(static_cast<size_t>(packBatchSize));
    unsigned state = 103u;
    for (int exampleIndex = 0; exampleIndex < packBatchSize; ++exampleIndex) {
        examples[static_cast<size_t>(exampleIndex)].inputTokenIds.resize(static_cast<size_t>(sequenceLength));
        examples[static_cast<size_t>(exampleIndex)].targetTokenIds.resize(static_cast<size_t>(sequenceLength));
        for (size_t index = 0; index < static_cast<size_t>(sequenceLength); ++index) {
            state = state * 1664525u + 1013904223u;
            examples[static_cast<size_t>(exampleIndex)].inputTokenIds[index] = static_cast<int>(state % static_cast<unsigned>(vocabularySize));
            state = state * 1664525u + 1013904223u;
            examples[static_cast<size_t>(exampleIndex)].targetTokenIds[index] = static_cast<int>(state % static_cast<unsigned>(vocabularySize));
        }
    }

    LanguageModelGradients hostGradients = LanguageModelGradients::zerosFrom(host);
    LanguageModelCache hostCache;
    const float hostLoss = host.accumulateExample(examples[0], hostGradients, hostCache);

    CudaLanguageModelGradients deviceGradients;
    deviceGradients.ensureFrom(device);
    deviceGradients.zeroInPlace();
    device.epochLossSum.ensureSize(1, 1);
    CudaOps::zeroInPlace(device.epochLossSum);
    device.accumulateExample(examples[0], deviceGradients);
    CudaMatmul::throwIfCudaFailed(cudaDeviceSynchronize(), "CudaLanguageModel train smoke synchronize");
    const float deviceLoss = device.epochLossSum.download().at(0, 0);
    Matrix deviceHeadWeightGrad = device.tieEmbeddingProjection
        ? deviceGradients.tokenEmbedding.download()
        : deviceGradients.projectionWeight.download();
    const Matrix& hostHeadWeightGrad = host.tieEmbeddingProjection
        ? hostGradients.tokenEmbedding
        : hostGradients.projectionWeight;
    float maximumDifference = 0.0f;
    for (size_t index = 0; index < hostHeadWeightGrad.data.size(); ++index)
        maximumDifference = (std::max)(maximumDifference, std::fabs(hostHeadWeightGrad.data[index] - deviceHeadWeightGrad.data[index]));

    CudaLanguageModelGradients packedGradients;
    packedGradients.ensureFrom(device);
    packedGradients.zeroInPlace();
    CudaLanguageModelGradients sequentialGradients;
    sequentialGradients.ensureFrom(device);
    sequentialGradients.zeroInPlace();
    device.epochLossSum.ensureSize(1, 1);
    CudaOps::zeroInPlace(device.epochLossSum);
    std::vector<const LanguageModelExample*> packPointers(static_cast<size_t>(packBatchSize));
    for (int exampleIndex = 0; exampleIndex < packBatchSize; ++exampleIndex)
        packPointers[static_cast<size_t>(exampleIndex)] = &examples[static_cast<size_t>(exampleIndex)];
    device.accumulatePackedExamples(packPointers.data(), packBatchSize, packedGradients);
    const float packedLoss = device.epochLossSum.download().at(0, 0);

    CudaOps::zeroInPlace(device.epochLossSum);
    for (int exampleIndex = 0; exampleIndex < packBatchSize; ++exampleIndex)
        device.accumulateExample(examples[static_cast<size_t>(exampleIndex)], sequentialGradients);
    const float sequentialLoss = device.epochLossSum.download().at(0, 0);
    Matrix packedHeadWeightGrad = device.tieEmbeddingProjection
        ? packedGradients.tokenEmbedding.download()
        : packedGradients.projectionWeight.download();
    Matrix sequentialHeadWeightGrad = device.tieEmbeddingProjection
        ? sequentialGradients.tokenEmbedding.download()
        : sequentialGradients.projectionWeight.download();
    float packedDifference = 0.0f;
    for (size_t index = 0; index < packedHeadWeightGrad.data.size(); ++index)
        packedDifference = (std::max)(packedDifference, std::fabs(packedHeadWeightGrad.data[index] - sequentialHeadWeightGrad.data[index]));

    const int warmupStepCount = 3;
    const int timedStepCount = 40;
    for (int step = 0; step < warmupStepCount; ++step) {
        device.trainGradients.zeroInPlace();
        device.accumulatePackedExamples(packPointers.data(), packBatchSize, device.trainGradients);
        device.applyGradients(device.trainGradients, 1.0f / static_cast<float>(packBatchSize));
    }
    CudaMatmul::throwIfCudaFailed(cudaDeviceSynchronize(), "CudaLanguageModel train smoke warmup synchronize");

    const auto trainStart = std::chrono::steady_clock::now();
    int totalTokens = 0;
    for (int step = 0; step < timedStepCount; ++step) {
        device.trainGradients.zeroInPlace();
        device.accumulatePackedExamples(packPointers.data(), packBatchSize, device.trainGradients);
        device.applyGradients(device.trainGradients, 1.0f / static_cast<float>(packBatchSize));
        totalTokens += sequenceLength * packBatchSize;
    }
    CudaMatmul::throwIfCudaFailed(cudaDeviceSynchronize(), "CudaLanguageModel train smoke steps synchronize");
    const double trainSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - trainStart).count();
    const double tokensPerSecond = trainSeconds > 0.0 ? static_cast<double>(totalTokens) / trainSeconds : 0.0;

    SmokeLog::result("LanguageModel train", "vocab=%d embed=%d seq=%d pack=%d  loss cpu=%.4f gpu=%.4f  gradDiff=%.2e  packLossDiff=%.2e packGradDiff=%.2e  tokens/s=%.0f",
        vocabularySize, embeddingDim, sequenceLength, packBatchSize, hostLoss, deviceLoss, maximumDifference, std::fabs(packedLoss - sequentialLoss), packedDifference, tokensPerSecond);
    CudaAmp::preferMixedPrecision = previousAmp;
    CudaAdam::preferInt8Moments = previousInt8;
}

void CudaLanguageModel::runMuonTrainSmokeDemo(int vocabularySize, int embeddingDim, int sequenceLength, int blockCount, int headCount) {
    if (!CudaMatmul::isAvailable()) {
        SmokeLog::skip("Muon train");
        return;
    }
    if (vocabularySize <= 0 || embeddingDim <= 0 || sequenceLength <= 0 || blockCount <= 0 || headCount <= 0)
        throw std::invalid_argument("CudaLanguageModel::runMuonTrainSmokeDemo invalid dims");
    if (embeddingDim % headCount != 0)
        throw std::invalid_argument("CudaLanguageModel::runMuonTrainSmokeDemo embed must divide heads");

    const bool previousAmp = CudaAmp::preferMixedPrecision;
    const bool previousLossScale = CudaAmp::useLossScaling;
    const bool previousInt8 = CudaAdam::preferInt8Moments;
    CudaAmp::preferMixedPrecision = false;
    CudaAmp::useLossScaling = false;
    CudaAdam::preferInt8Moments = false;

    LanguageModel host(vocabularySize, embeddingDim, sequenceLength, Adam(0.001f), blockCount, headCount);
    auto device = std::make_unique<CudaLanguageModel>();
    device->uploadFrom(host);
    device->adam = CudaAdam(0.001f);
    device->muon.learningRate = 0.001f;
    device->preferTrainGraph = false;
    device->setPreferMuon(true);
    device->setActivationCheckpointMode(ActivationCheckpointMode::Selective);
    for (CudaTransformerBlock& block : device->blocks)
        block.attention.preferFlashAttention = false;

    device->maxPackedColumns = (std::max)(sequenceLength * 8, sequenceLength);
    device->ensureTrainState();

    const int packBatchSize = (std::max)(1, (std::min)(4, device->maxPackedColumns / sequenceLength));
    std::vector<LanguageModelExample> examples(static_cast<size_t>(packBatchSize));
    unsigned rng = 911u;
    for (int exampleIndex = 0; exampleIndex < packBatchSize; ++exampleIndex) {
        examples[static_cast<size_t>(exampleIndex)].inputTokenIds.resize(static_cast<size_t>(sequenceLength));
        examples[static_cast<size_t>(exampleIndex)].targetTokenIds.resize(static_cast<size_t>(sequenceLength));
        for (size_t index = 0; index < static_cast<size_t>(sequenceLength); ++index) {
            rng = rng * 1664525u + 1013904223u;
            examples[static_cast<size_t>(exampleIndex)].inputTokenIds[index] = static_cast<int>(rng % static_cast<unsigned>(vocabularySize));
            rng = rng * 1664525u + 1013904223u;
            examples[static_cast<size_t>(exampleIndex)].targetTokenIds[index] = static_cast<int>(rng % static_cast<unsigned>(vocabularySize));
        }
    }
    std::vector<const LanguageModelExample*> packPointers(static_cast<size_t>(packBatchSize));
    for (int exampleIndex = 0; exampleIndex < packBatchSize; ++exampleIndex)
        packPointers[static_cast<size_t>(exampleIndex)] = &examples[static_cast<size_t>(exampleIndex)];

    auto runLoss = [&]() -> float {
        device->epochLossSum.ensureSize(1, 1);
        CudaOps::zeroInPlace(device->epochLossSum);
        device->trainGradients.zeroInPlace();
        device->accumulatePackedExamples(packPointers.data(), packBatchSize, device->trainGradients);
        CudaMatmul::throwIfCudaFailed(cudaDeviceSynchronize(), "Muon train smoke loss sync");
        return device->epochLossSum.download().at(0, 0) / static_cast<float>(packBatchSize);
    };

    const float lossBefore = runLoss();
    const int stepCount = 8;
    for (int step = 0; step < stepCount; ++step) {
        device->trainGradients.zeroInPlace();
        device->accumulatePackedExamples(packPointers.data(), packBatchSize, device->trainGradients);
        device->applyGradients(device->trainGradients, 1.0f / static_cast<float>(packBatchSize));
    }
    CudaMatmul::throwIfCudaFailed(cudaDeviceSynchronize(), "Muon train smoke step sync");
    const float lossAfter = runLoss();

    bool anyNonFinite = !std::isfinite(lossBefore) || !std::isfinite(lossAfter);
    Matrix embed = device->tokenEmbeddingWeight.download();
    for (float value : embed.data) {
        if (!std::isfinite(value)) {
            anyNonFinite = true;
            break;
        }
    }

    SmokeLog::result(
        "Muon train",
        "vocab=%d embed=%d seq=%d blocks=%d pack=%d steps=%d  loss0=%.4f lossN=%.4f  finite=%s muon=%s",
        vocabularySize,
        embeddingDim,
        sequenceLength,
        blockCount,
        packBatchSize,
        stepCount,
        lossBefore,
        lossAfter,
        anyNonFinite ? "no" : "yes",
        device->preferMuon ? "on" : "off");

    CudaAmp::preferMixedPrecision = previousAmp;
    CudaAmp::useLossScaling = previousLossScale;
    CudaAdam::preferInt8Moments = previousInt8;
}

void CudaLanguageModel::runSelectiveCheckpointParitySmokeDemo(
    int vocabularySize,
    int embeddingDim,
    int sequenceLength,
    int blockCount,
    int headCount
) {
    if (!CudaMatmul::isAvailable()) {
        SmokeLog::skip("ckpt Full vs Selective");
        return;
    }
    if (vocabularySize <= 0 || embeddingDim <= 0 || sequenceLength <= 0 || blockCount <= 0 || headCount <= 0)
        throw std::invalid_argument("CudaLanguageModel::runSelectiveCheckpointParitySmokeDemo invalid dims");
    if (embeddingDim % headCount != 0)
        throw std::invalid_argument("CudaLanguageModel::runSelectiveCheckpointParitySmokeDemo embed must divide heads");

    const bool previousAmp = CudaAmp::preferMixedPrecision;
    const bool previousLossScale = CudaAmp::useLossScaling;
    const bool previousInt8 = CudaAdam::preferInt8Moments;
    CudaAmp::preferMixedPrecision = false;
    CudaAmp::useLossScaling = false;
    CudaAdam::preferInt8Moments = false;

    // Heap-allocate twin LMs (object too large for two stack locals + grads).
    auto host = std::make_unique<LanguageModel>(vocabularySize, embeddingDim, sequenceLength, Adam(0.001f), blockCount, headCount);
    auto fullDevice = std::make_unique<CudaLanguageModel>();
    fullDevice->uploadFrom(*host);
    auto selectiveDevice = std::make_unique<CudaLanguageModel>();
    selectiveDevice->uploadFrom(*host);
    fullDevice->adam = CudaAdam(0.001f);
    selectiveDevice->adam = CudaAdam(0.001f);
    fullDevice->preferTrainGraph = false;
    selectiveDevice->preferTrainGraph = false;
    fullDevice->setActivationCheckpointMode(ActivationCheckpointMode::Full);
    selectiveDevice->setActivationCheckpointMode(ActivationCheckpointMode::Selective);
    // Dense attn: isolates ckpt math from Flash packed-path issues on some shapes.
    for (CudaTransformerBlock& block : fullDevice->blocks)
        block.attention.preferFlashAttention = false;
    for (CudaTransformerBlock& block : selectiveDevice->blocks)
        block.attention.preferFlashAttention = false;

    fullDevice->maxPackedColumns = (std::max)(sequenceLength * 8, sequenceLength);
    selectiveDevice->maxPackedColumns = fullDevice->maxPackedColumns;
    fullDevice->ensureTrainState();
    selectiveDevice->ensureTrainState();

    const int packBatchSize = (std::max)(1, (std::min)(4, fullDevice->maxPackedColumns / sequenceLength));
    std::vector<LanguageModelExample> examples(static_cast<size_t>(packBatchSize));
    unsigned state = 409u;
    for (int exampleIndex = 0; exampleIndex < packBatchSize; ++exampleIndex) {
        examples[static_cast<size_t>(exampleIndex)].inputTokenIds.resize(static_cast<size_t>(sequenceLength));
        examples[static_cast<size_t>(exampleIndex)].targetTokenIds.resize(static_cast<size_t>(sequenceLength));
        for (size_t index = 0; index < static_cast<size_t>(sequenceLength); ++index) {
            state = state * 1664525u + 1013904223u;
            examples[static_cast<size_t>(exampleIndex)].inputTokenIds[index] = static_cast<int>(state % static_cast<unsigned>(vocabularySize));
            state = state * 1664525u + 1013904223u;
            examples[static_cast<size_t>(exampleIndex)].targetTokenIds[index] = static_cast<int>(state % static_cast<unsigned>(vocabularySize));
        }
    }

    std::vector<const LanguageModelExample*> packPointers(static_cast<size_t>(packBatchSize));
    for (int exampleIndex = 0; exampleIndex < packBatchSize; ++exampleIndex)
        packPointers[static_cast<size_t>(exampleIndex)] = &examples[static_cast<size_t>(exampleIndex)];

    auto fullGradients = std::make_unique<CudaLanguageModelGradients>();
    fullGradients->ensureFrom(*fullDevice);
    fullGradients->zeroInPlace();
    fullDevice->epochLossSum.ensureSize(1, 1);
    CudaOps::zeroInPlace(fullDevice->epochLossSum);
    fullDevice->accumulatePackedExamples(packPointers.data(), packBatchSize, *fullGradients);

    auto selectiveGradients = std::make_unique<CudaLanguageModelGradients>();
    selectiveGradients->ensureFrom(*selectiveDevice);
    selectiveGradients->zeroInPlace();
    selectiveDevice->epochLossSum.ensureSize(1, 1);
    CudaOps::zeroInPlace(selectiveDevice->epochLossSum);
    selectiveDevice->accumulatePackedExamples(packPointers.data(), packBatchSize, *selectiveGradients);
    CudaMatmul::throwIfCudaFailed(cudaDeviceSynchronize(), "ckpt parity synchronize");

    const float fullLoss = fullDevice->epochLossSum.download().at(0, 0);
    const float selectiveLoss = selectiveDevice->epochLossSum.download().at(0, 0);
    Matrix fullEmbedGrad = fullGradients->tokenEmbedding.download();
    Matrix selectiveEmbedGrad = selectiveGradients->tokenEmbedding.download();
    float maxGradDiff = 0.0f;
    for (size_t index = 0; index < fullEmbedGrad.data.size(); ++index)
        maxGradDiff = (std::max)(maxGradDiff, std::fabs(fullEmbedGrad.data[index] - selectiveEmbedGrad.data[index]));

    float maxFfnGradDiff = 0.0f;
    if (!fullGradients->blocks.empty() && !selectiveGradients->blocks.empty()) {
        Matrix fullDownGrad = fullGradients->blocks[0].feedForwardDownWeight.download();
        Matrix selectiveDownGrad = selectiveGradients->blocks[0].feedForwardDownWeight.download();
        for (size_t index = 0; index < fullDownGrad.data.size(); ++index)
            maxFfnGradDiff = (std::max)(maxFfnGradDiff, std::fabs(fullDownGrad.data[index] - selectiveDownGrad.data[index]));
    }

    SmokeLog::result(
        "ckpt Full vs Selective",
        "vocab=%d embed=%d seq=%d blocks=%d pack=%d  lossFull=%.6f lossSel=%.6f lossDiff=%.2e  embedGradDiff=%.2e ffnDownGradDiff=%.2e",
        vocabularySize,
        embeddingDim,
        sequenceLength,
        blockCount,
        packBatchSize,
        fullLoss,
        selectiveLoss,
        std::fabs(fullLoss - selectiveLoss),
        maxGradDiff,
        maxFfnGradDiff);

    CudaAmp::preferMixedPrecision = previousAmp;
    CudaAmp::useLossScaling = previousLossScale;
    CudaAdam::preferInt8Moments = previousInt8;
}

void CudaLanguageModel::runTrainInt8AdamSmokeDemo(int vocabularySize, int embeddingDim, int sequenceLength, int blockCount, int headCount) {
    if (!CudaMatmul::isAvailable()) {
        SmokeLog::skip("LanguageModel train int8 Adam");
        return;
    }
    if (vocabularySize <= 0 || embeddingDim <= 0 || sequenceLength <= 0 || blockCount <= 0 || headCount <= 0)
        throw std::invalid_argument("CudaLanguageModel::runTrainInt8AdamSmokeDemo invalid dims");

    LanguageModel host(vocabularySize, embeddingDim, sequenceLength, Adam(0.001f), blockCount, headCount);
    CudaLanguageModel fp32Device = CudaLanguageModel::createFrom(host);
    CudaLanguageModel int8Device = CudaLanguageModel::createFrom(host);
    fp32Device.adam = CudaAdam(host.optimizer.learningRate, host.optimizer.beta1, host.optimizer.beta2, host.optimizer.epsilon);
    int8Device.adam = CudaAdam(host.optimizer.learningRate, host.optimizer.beta1, host.optimizer.beta2, host.optimizer.epsilon);

    const bool previousAmp = CudaAmp::preferMixedPrecision;
    const bool previousInt8 = CudaAdam::preferInt8Moments;
    CudaAmp::preferMixedPrecision = false;

    const int packBatchSize = (std::max)(1, (std::min)(8, fp32Device.maxPackedColumns / sequenceLength));
    std::vector<LanguageModelExample> examples(static_cast<size_t>(packBatchSize));
    unsigned state = 211u;
    for (int exampleIndex = 0; exampleIndex < packBatchSize; ++exampleIndex) {
        examples[static_cast<size_t>(exampleIndex)].inputTokenIds.resize(static_cast<size_t>(sequenceLength));
        examples[static_cast<size_t>(exampleIndex)].targetTokenIds.resize(static_cast<size_t>(sequenceLength));
        for (size_t index = 0; index < static_cast<size_t>(sequenceLength); ++index) {
            state = state * 1664525u + 1013904223u;
            examples[static_cast<size_t>(exampleIndex)].inputTokenIds[index] = static_cast<int>(state % static_cast<unsigned>(vocabularySize));
            state = state * 1664525u + 1013904223u;
            examples[static_cast<size_t>(exampleIndex)].targetTokenIds[index] = static_cast<int>(state % static_cast<unsigned>(vocabularySize));
        }
    }
    std::vector<const LanguageModelExample*> packPointers(static_cast<size_t>(packBatchSize));
    for (int exampleIndex = 0; exampleIndex < packBatchSize; ++exampleIndex)
        packPointers[static_cast<size_t>(exampleIndex)] = &examples[static_cast<size_t>(exampleIndex)];

    CudaAdam::preferInt8Moments = false;
    fp32Device.ensureTrainState();
    CudaAdam::preferInt8Moments = true;
    int8Device.ensureTrainState();

    const int stepCount = 24;
    for (int step = 0; step < stepCount; ++step) {
        CudaAdam::preferInt8Moments = false;
        fp32Device.trainGradients.zeroInPlace();
        fp32Device.accumulatePackedExamples(packPointers.data(), packBatchSize, fp32Device.trainGradients);
        fp32Device.applyGradients(fp32Device.trainGradients, 1.0f / static_cast<float>(packBatchSize));

        CudaAdam::preferInt8Moments = true;
        int8Device.trainGradients.zeroInPlace();
        int8Device.accumulatePackedExamples(packPointers.data(), packBatchSize, int8Device.trainGradients);
        int8Device.applyGradients(int8Device.trainGradients, 1.0f / static_cast<float>(packBatchSize));
    }
    CudaMatmul::throwIfCudaFailed(cudaDeviceSynchronize(), "CudaLanguageModel train int8 Adam synchronize");

    CudaAdam::preferInt8Moments = false;
    fp32Device.epochLossSum.ensureSize(1, 1);
    CudaOps::zeroInPlace(fp32Device.epochLossSum);
    fp32Device.accumulatePackedExamples(packPointers.data(), packBatchSize, fp32Device.trainGradients);
    const float fp32Loss = fp32Device.epochLossSum.download().at(0, 0) / static_cast<float>(packBatchSize);

    CudaAdam::preferInt8Moments = true;
    int8Device.epochLossSum.ensureSize(1, 1);
    CudaOps::zeroInPlace(int8Device.epochLossSum);
    int8Device.accumulatePackedExamples(packPointers.data(), packBatchSize, int8Device.trainGradients);
    const float int8Loss = int8Device.epochLossSum.download().at(0, 0) / static_cast<float>(packBatchSize);

    Matrix fp32Weight = fp32Device.lmHeadWeight().download();
    Matrix int8Weight = int8Device.lmHeadWeight().download();
    float weightDifference = 0.0f;
    bool anyNonFinite = false;
    for (size_t index = 0; index < fp32Weight.data.size(); ++index) {
        if (!std::isfinite(fp32Weight.data[index]) || !std::isfinite(int8Weight.data[index]))
            anyNonFinite = true;
        weightDifference = (std::max)(weightDifference, std::fabs(fp32Weight.data[index] - int8Weight.data[index]));
    }

    SmokeLog::result("LanguageModel train int8 Adam", "vocab=%d embed=%d seq=%d steps=%d  loss fp32=%.4f int8=%.4f  weightDiff=%.2e  nonFinite=%s",
        vocabularySize, embeddingDim, sequenceLength, stepCount, fp32Loss, int8Loss, weightDifference, anyNonFinite ? "yes" : "no");

    CudaAmp::preferMixedPrecision = previousAmp;
    CudaAdam::preferInt8Moments = previousInt8;
}

void CudaLanguageModel::runTrainCpuAdamOffloadSmokeDemo(int vocabularySize, int embeddingDim, int sequenceLength, int blockCount, int headCount) {
    if (!CudaMatmul::isAvailable()) {
        SmokeLog::skip("LanguageModel train CPU Adam offload");
        return;
    }
    if (vocabularySize <= 0 || embeddingDim <= 0 || sequenceLength <= 0 || blockCount <= 0 || headCount <= 0)
        throw std::invalid_argument("CudaLanguageModel::runTrainCpuAdamOffloadSmokeDemo invalid dims");

    const bool previousAmp = CudaAmp::preferMixedPrecision;
    const bool previousInt8 = CudaAdam::preferInt8Moments;
    const bool previousCpuOffload = CudaAdam::preferCpuOffload;
    const bool previousFp16 = CudaAdam::preferFp16GpuWeights;
    const bool previousHostGrads = CudaAdam::preferHostGradients;
    CudaAmp::preferMixedPrecision = false;
    CudaAdam::preferInt8Moments = false;
    CudaAdam::preferFp16GpuWeights = false;
    CudaAdam::preferHostGradients = false;

    const int packBatchSize = (std::max)(1, (std::min)(8, 8192 / sequenceLength));
    std::vector<LanguageModelExample> examples(static_cast<size_t>(packBatchSize));
    unsigned state = 307u;
    for (int exampleIndex = 0; exampleIndex < packBatchSize; ++exampleIndex) {
        examples[static_cast<size_t>(exampleIndex)].inputTokenIds.resize(static_cast<size_t>(sequenceLength));
        examples[static_cast<size_t>(exampleIndex)].targetTokenIds.resize(static_cast<size_t>(sequenceLength));
        for (size_t index = 0; index < static_cast<size_t>(sequenceLength); ++index) {
            state = state * 1664525u + 1013904223u;
            examples[static_cast<size_t>(exampleIndex)].inputTokenIds[index] = static_cast<int>(state % static_cast<unsigned>(vocabularySize));
            state = state * 1664525u + 1013904223u;
            examples[static_cast<size_t>(exampleIndex)].targetTokenIds[index] = static_cast<int>(state % static_cast<unsigned>(vocabularySize));
        }
    }
    std::vector<const LanguageModelExample*> packPointers(static_cast<size_t>(packBatchSize));
    for (int exampleIndex = 0; exampleIndex < packBatchSize; ++exampleIndex)
        packPointers[static_cast<size_t>(exampleIndex)] = &examples[static_cast<size_t>(exampleIndex)];

    // FP16 GPU working + host weight grads + FP32 CPU Adam
    {
        CudaAmp::preferMixedPrecision = true;
        CudaAdam::preferCpuOffload = true;
        CudaAdam::preferFp16GpuWeights = true;
        CudaAdam::preferHostGradients = true;
        CudaAdam::preferInt8Moments = false;

        LanguageModel fp16Host(vocabularySize, embeddingDim, sequenceLength, Adam(0.001f), blockCount, headCount);
        CudaLanguageModel fp16Device = CudaLanguageModel::createFrom(fp16Host);
        fp16Device.adam = CudaAdam(0.001f);
        fp16Device.ensureTrainState();
        for (int step = 0; step < 4; ++step) {
            fp16Device.zeroAccumulatedGradients();
            fp16Device.accumulatePackedExamples(packPointers.data(), packBatchSize, fp16Device.trainGradients);
            fp16Device.applyGradients(fp16Device.trainGradients, 1.0f / static_cast<float>(packBatchSize));
        }
        CudaMatmul::throwIfCudaFailed(cudaDeviceSynchronize(), "fp16 host-grads smoke synchronize");
        float masterNorm = 0.0f;
        for (float value : fp16Device.hostTokenEmbeddingMaster.data)
            masterNorm += value * value;
        SmokeLog::result(
            "LanguageModel train FP16 GPU + host grads + CPU Adam",
            "vocab=%d embed=%d steps=4  embedMasterL2=%.4f  finite=%s",
            vocabularySize, embeddingDim, std::sqrt(masterNorm),
            std::isfinite(masterNorm) ? "yes" : "no");
        CudaAmp::clearMasterWeights();
    }

    CudaAmp::preferMixedPrecision = previousAmp;
    CudaAdam::preferInt8Moments = previousInt8;
    CudaAdam::preferCpuOffload = previousCpuOffload;
    CudaAdam::preferFp16GpuWeights = previousFp16;
    CudaAdam::preferHostGradients = previousHostGrads;
}

void CudaLanguageModel::runTrainProfileDemo(int vocabularySize, int embeddingDim, int sequenceLength, int blockCount, int headCount, bool preferFlash, int maxPackedColumns, int packBatchSize, bool activationCheckpointing) {
    if (!CudaMatmul::isAvailable()) {
        SmokeLog::skip("LanguageModel profile");
        return;
    }
    if (vocabularySize <= 0 || embeddingDim <= 0 || sequenceLength <= 0 || blockCount <= 0 || headCount <= 0)
        throw std::invalid_argument("CudaLanguageModel::runTrainProfileDemo invalid dims");

    const bool previousAmp = CudaAmp::preferMixedPrecision;
    const bool previousLossScaling = CudaAmp::useLossScaling;
    const bool previousInt8 = CudaAdam::preferInt8Moments;
    const bool previousCpu = CudaAdam::preferCpuOffload;

    LanguageModel host(vocabularySize, embeddingDim, sequenceLength, Adam(0.001f), blockCount, headCount);
    CudaLanguageModel device = CudaLanguageModel::createFrom(host);
    device.adam = CudaAdam(host.optimizer.learningRate, host.optimizer.beta1, host.optimizer.beta2, host.optimizer.epsilon);
    if (maxPackedColumns > 0) {
        device.maxPackedColumns = maxPackedColumns;
        device.maxPackedColumnsManual = true;
    } else {
        device.maxPackedColumnsManual = false;
        device.applyVramPackBudget();
    }
    for (CudaTransformerBlock& block : device.blocks) {
        block.attention.preferFlashAttention = preferFlash;
        if (preferFlash)
            block.attention.releaseDenseAttentionScratch();
        else
            block.attention.releaseFlashAttentionScratch();
    }

    CudaAmp::preferMixedPrecision = true;
    CudaAmp::useLossScaling = embeddingDim >= 256;
    if (CudaAmp::useLossScaling)
        CudaAmp::resetLossScaler();
    CudaAdam::preferCpuOffload = false;
    CudaAdam::preferInt8Moments = true;
    device.setActivationCheckpointMode(
        activationCheckpointing ? ActivationCheckpointMode::Selective : ActivationCheckpointMode::Off);
    device.ensureTrainState();
    device.ensureTrainWorkspaces();
    device.epochLossSum.ensureSize(1, 1);

    const int maxPackByCols = (std::max)(1, device.maxPackedColumns / sequenceLength);
    int resolvedPack = packBatchSize > 0 ? packBatchSize : 32;
    resolvedPack = (std::max)(1, (std::min)(resolvedPack, maxPackByCols));
    packBatchSize = resolvedPack;

    std::vector<LanguageModelExample> examples(static_cast<size_t>(packBatchSize));
    unsigned state = 211u;
    for (int exampleIndex = 0; exampleIndex < packBatchSize; ++exampleIndex) {
        examples[static_cast<size_t>(exampleIndex)].inputTokenIds.resize(static_cast<size_t>(sequenceLength));
        examples[static_cast<size_t>(exampleIndex)].targetTokenIds.resize(static_cast<size_t>(sequenceLength));
        for (size_t index = 0; index < static_cast<size_t>(sequenceLength); ++index) {
            state = state * 1664525u + 1013904223u;
            examples[static_cast<size_t>(exampleIndex)].inputTokenIds[index] = static_cast<int>(state % static_cast<unsigned>(vocabularySize));
            state = state * 1664525u + 1013904223u;
            examples[static_cast<size_t>(exampleIndex)].targetTokenIds[index] = static_cast<int>(state % static_cast<unsigned>(vocabularySize));
        }
    }

    std::vector<const LanguageModelExample*> packPointers(static_cast<size_t>(packBatchSize));
    for (int exampleIndex = 0; exampleIndex < packBatchSize; ++exampleIndex)
        packPointers[static_cast<size_t>(exampleIndex)] = &examples[static_cast<size_t>(exampleIndex)];

    const size_t tokenCount = static_cast<size_t>(sequenceLength) * static_cast<size_t>(packBatchSize);
    const int meanDivisor = sequenceLength;

    cudaEvent_t startEvent = nullptr;
    cudaEvent_t stopEvent = nullptr;
    CudaMatmul::throwIfCudaFailed(cudaEventCreate(&startEvent), "profile cudaEventCreate start");
    CudaMatmul::throwIfCudaFailed(cudaEventCreate(&stopEvent), "profile cudaEventCreate stop");

    auto gpuMs = [&](auto&& work) -> float {
        CudaMatmul::throwIfCudaFailed(cudaEventRecord(startEvent), "profile cudaEventRecord start");
        work();
        CudaMatmul::throwIfCudaFailed(cudaEventRecord(stopEvent), "profile cudaEventRecord stop");
        CudaMatmul::throwIfCudaFailed(cudaEventSynchronize(stopEvent), "profile cudaEventSynchronize");
        float milliseconds = 0.0f;
        CudaMatmul::throwIfCudaFailed(cudaEventElapsedTime(&milliseconds, startEvent, stopEvent), "profile cudaEventElapsedTime");
        return milliseconds;
    };

    auto runTimedStep = [&](float& hostPackMs, float& h2dMs, float& embedMs, float& attnMs, float& ffnMs, float& finalNormMs, float& chunkedHeadCeMs, float& finalNormBwdMs, float& attnBwdMs, float& ffnBwdMs, float& scatterMs, float& adamMs) {
        hostPackMs = 0.0f;
        h2dMs = 0.0f;
        embedMs = 0.0f;
        attnMs = 0.0f;
        ffnMs = 0.0f;
        finalNormMs = 0.0f;
        chunkedHeadCeMs = 0.0f;
        finalNormBwdMs = 0.0f;
        attnBwdMs = 0.0f;
        ffnBwdMs = 0.0f;
        scatterMs = 0.0f;
        adamMs = 0.0f;

        const auto hostPackStart = std::chrono::steady_clock::now();
        device.packedInputTokenIds.clear();
        device.packedTargetTokenIds.clear();
        device.packedMeanDivisors.clear();
        device.packedInputTokenIds.reserve(tokenCount);
        device.packedTargetTokenIds.reserve(tokenCount);
        device.packedMeanDivisors.reserve(tokenCount);
        for (int exampleIndex = 0; exampleIndex < packBatchSize; ++exampleIndex) {
            device.packedInputTokenIds.insert(device.packedInputTokenIds.end(), examples[static_cast<size_t>(exampleIndex)].inputTokenIds.begin(), examples[static_cast<size_t>(exampleIndex)].inputTokenIds.end());
            device.packedTargetTokenIds.insert(device.packedTargetTokenIds.end(), examples[static_cast<size_t>(exampleIndex)].targetTokenIds.begin(), examples[static_cast<size_t>(exampleIndex)].targetTokenIds.end());
            for (int position = 0; position < sequenceLength; ++position)
                device.packedMeanDivisors.push_back(meanDivisor);
        }
        hostPackMs = static_cast<float>(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - hostPackStart).count());

        device.trainGradients.zeroInPlace();

        h2dMs += gpuMs([&]() {
            device.tokenIdsBuffer.ensureCapacity(tokenCount);
            device.targetTokenIdsBuffer.ensureCapacity(tokenCount);
            device.meanDivisorBuffer.ensureCapacity(tokenCount);
            device.tokenIdsBuffer.copyFromHost(device.packedInputTokenIds.data(), tokenCount);
            device.targetTokenIdsBuffer.copyFromHost(device.packedTargetTokenIds.data(), tokenCount);
            device.meanDivisorBuffer.copyFromHost(device.packedMeanDivisors.data(), tokenCount);
        });

        embedMs += gpuMs([&]() {
            CudaOps::embeddingGatherInto(device.tokenEmbeddingWeight, device.tokenIdsBuffer, tokenCount, device.hidden);
        });

        for (size_t blockIndex = 0; blockIndex < device.blocks.size(); ++blockIndex) {
            CudaTransformerBlock& block = device.blocks[blockIndex];
            attnMs += gpuMs([&]() {
                block.attentionNorm.forward(device.hidden, block.attentionInput);
                block.attention.forward(block.attentionInput, block.attended, sequenceLength);
                CudaOps::addInto(device.hidden, block.attended, block.afterAttention);
            });
            ffnMs += gpuMs([&]() {
                block.feedForwardNorm.forward(block.afterAttention, block.feedForwardInput);
                block.feedForward.forward(block.feedForwardInput, block.feedForwardOutput);
                CudaOps::addInto(block.afterAttention, block.feedForwardOutput, device.normalized);
            });
            CudaMatrix swapBuffer = std::move(device.hidden);
            device.hidden = std::move(device.normalized);
            device.normalized = std::move(swapBuffer);
        }

        finalNormMs += gpuMs([&]() {
            device.finalNorm.forward(device.hidden, device.normalized);
        });

        // Production CE path: single-pass / chunked LM head + grads into hiddenGradient
        chunkedHeadCeMs += gpuMs([&]() {
            device.accumulateChunkedProjection(tokenCount, sequenceLength, packBatchSize, device.trainGradients);
        });

        finalNormBwdMs += gpuMs([&]() {
            device.finalNorm.backward(device.hiddenGradient, device.normInputGradientScratch, device.finalNormGammaGradient);
            CudaOps::addInPlace(device.trainGradients.finalNormGamma, device.finalNormGammaGradient);
            std::swap(device.hiddenGradient, device.normInputGradientScratch);
        });

        for (int blockIndex = static_cast<int>(device.blocks.size()) - 1; blockIndex >= 0; --blockIndex) {
            CudaTransformerBlock& block = device.blocks[static_cast<size_t>(blockIndex)];
            CudaTransformerBlockGradients& blockGradients = device.trainGradients.blocks[static_cast<size_t>(blockIndex)];

            ffnBwdMs += gpuMs([&]() {
                block.feedForward.backward(device.hiddenGradient, block.feedForwardInputGradient, block.feedForwardGateWeightGradient, block.feedForwardGateBiasGradient, block.feedForwardUpWeightGradient, block.feedForwardUpBiasGradient, block.feedForwardDownWeightGradient, block.feedForwardDownBiasGradient);
                block.feedForwardNorm.backward(block.feedForwardInputGradient, block.afterAttentionFromFeedForward, block.feedForwardNormGammaGradient);
                CudaOps::addInto(device.hiddenGradient, block.afterAttentionFromFeedForward, block.afterAttentionGradient);
                CudaOps::addInPlace(blockGradients.feedForwardDownWeight, block.feedForwardDownWeightGradient);
                CudaOps::addInPlace(blockGradients.feedForwardDownBias, block.feedForwardDownBiasGradient);
                CudaOps::addInPlace(blockGradients.feedForwardUpWeight, block.feedForwardUpWeightGradient);
                CudaOps::addInPlace(blockGradients.feedForwardUpBias, block.feedForwardUpBiasGradient);
                CudaOps::addInPlace(blockGradients.feedForwardGateWeight, block.feedForwardGateWeightGradient);
                CudaOps::addInPlace(blockGradients.feedForwardGateBias, block.feedForwardGateBiasGradient);
                CudaOps::addInPlace(blockGradients.feedForwardNormGamma, block.feedForwardNormGammaGradient);
            });

            attnBwdMs += gpuMs([&]() {
                block.attention.backward(block.afterAttentionGradient, block.attentionInputGradient, block.queryWeightGradient, block.keyWeightGradient, block.valueWeightGradient, block.attentionOutputWeightGradient);
                block.attentionNorm.backward(block.attentionInputGradient, block.inputFromAttention, block.attentionNormGammaGradient);
                CudaOps::addInPlace(blockGradients.attentionOutputWeight, block.attentionOutputWeightGradient);
                CudaOps::addInPlace(blockGradients.valueWeight, block.valueWeightGradient);
                CudaOps::addInPlace(blockGradients.keyWeight, block.keyWeightGradient);
                CudaOps::addInPlace(blockGradients.queryWeight, block.queryWeightGradient);
                CudaOps::addInPlace(blockGradients.attentionNormGamma, block.attentionNormGammaGradient);
                CudaOps::addInto(block.afterAttentionGradient, block.inputFromAttention, device.blockInputGradientScratch);
            });
            std::swap(device.hiddenGradient, device.blockInputGradientScratch);
        }

        scatterMs += gpuMs([&]() {
            CudaOps::embeddingScatterAddInto(device.trainGradients.tokenEmbedding, device.tokenIdsBuffer, tokenCount, device.hiddenGradient);
        });

        adamMs += gpuMs([&]() {
            device.applyGradients(device.trainGradients, 1.0f / static_cast<float>(packBatchSize));
        });
    };

    float discard[12];
    for (int warm = 0; warm < 3; ++warm)
        runTimedStep(discard[0], discard[1], discard[2], discard[3], discard[4], discard[5], discard[6], discard[7], discard[8], discard[9], discard[10], discard[11]);

    float sumHostPack = 0.0f, sumH2d = 0.0f, sumEmbed = 0.0f, sumAttn = 0.0f, sumFfn = 0.0f, sumFinalNorm = 0.0f;
    float sumChunkedHeadCe = 0.0f, sumFinalNormBwd = 0.0f, sumAttnBwd = 0.0f, sumFfnBwd = 0.0f, sumScatter = 0.0f, sumAdam = 0.0f;
    const int timedStepCount = 20;
    for (int step = 0; step < timedStepCount; ++step) {
        float hostPackMs, h2dMs, embedMs, attnMs, ffnMs, finalNormMs, chunkedHeadCeMs, finalNormBwdMs, attnBwdMs, ffnBwdMs, scatterMs, adamMs;
        CudaOps::zeroInPlace(device.epochLossSum);
        runTimedStep(hostPackMs, h2dMs, embedMs, attnMs, ffnMs, finalNormMs, chunkedHeadCeMs, finalNormBwdMs, attnBwdMs, ffnBwdMs, scatterMs, adamMs);
        sumHostPack += hostPackMs;
        sumH2d += h2dMs;
        sumEmbed += embedMs;
        sumAttn += attnMs;
        sumFfn += ffnMs;
        sumFinalNorm += finalNormMs;
        sumChunkedHeadCe += chunkedHeadCeMs;
        sumFinalNormBwd += finalNormBwdMs;
        sumAttnBwd += attnBwdMs;
        sumFfnBwd += ffnBwdMs;
        sumScatter += scatterMs;
        sumAdam += adamMs;
    }

    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);

    const float inv = 1.0f / static_cast<float>(timedStepCount);
    sumHostPack *= inv; sumH2d *= inv; sumEmbed *= inv; sumAttn *= inv; sumFfn *= inv; sumFinalNorm *= inv;
    sumChunkedHeadCe *= inv; sumFinalNormBwd *= inv; sumAttnBwd *= inv; sumFfnBwd *= inv; sumScatter *= inv; sumAdam *= inv;

    const float attnTotal = sumAttn + sumAttnBwd;
    const float ffnTotal = sumFfn + sumFfnBwd;
    const float headTotal = sumFinalNorm + sumChunkedHeadCe + sumFinalNormBwd;
    const float otherTotal = sumHostPack + sumH2d + sumEmbed + sumScatter + sumAdam;
    const float stepTotal = attnTotal + ffnTotal + headTotal + otherTotal;
    const float percent = stepTotal > 0.0f ? (100.0f / stepTotal) : 0.0f;
    const int tokensPerStep = sequenceLength * packBatchSize;
    const float tokensPerSecond = stepTotal > 0.0f ? (1000.0f * static_cast<float>(tokensPerStep) / stepTotal) : 0.0f;

    SmokeLog::section("train profile");
    SmokeLog::result("profile config", "vocab=%d embed=%d seq=%d blocks=%d heads=%d pack=%d tokens/step=%d flash=%s amp=%s int8Adam=%s ckpt=%s ce=chunked maxPackCols=%d lossScale=%.0f",
        vocabularySize, embeddingDim, sequenceLength, blockCount, headCount, packBatchSize, tokensPerStep,
        preferFlash ? "on" : "off",
        CudaAmp::preferMixedPrecision ? "on" : "off",
        CudaAdam::preferInt8Moments ? "on" : "off",
        CudaLanguageModel::activationCheckpointModeName(device.activationCheckpointMode),
        device.maxPackedColumns, CudaAmp::lossScaler.scale);
    SmokeLog::result("profile step", "avg=%.2fms  ~tokens/s=%.0f", stepTotal, tokensPerSecond);
    SmokeLog::result("  attention", "fwd=%.2fms bwd=%.2fms  total=%.2fms (%.0f%%)", sumAttn, sumAttnBwd, attnTotal, attnTotal * percent);
    SmokeLog::result("  ffn", "fwd=%.2fms bwd=%.2fms  total=%.2fms (%.0f%%)", sumFfn, sumFfnBwd, ffnTotal, ffnTotal * percent);
    SmokeLog::result("  head+ce", "finalNorm=%.2fms chunkedHeadCe=%.2fms finalNormBwd=%.2fms  total=%.2fms (%.0f%%)",
        sumFinalNorm, sumChunkedHeadCe, sumFinalNormBwd, headTotal, headTotal * percent);
    SmokeLog::result("  embed+h2d", "embed=%.2fms scatter=%.2fms h2d=%.2fms hostPack=%.2fms", sumEmbed, sumScatter, sumH2d, sumHostPack);
    SmokeLog::result("  adam", "%.2fms (%.0f%%)", sumAdam, sumAdam * percent);
    SmokeLog::result("  other sum", "%.2fms (%.0f%%)", otherTotal, otherTotal * percent);

    CudaAmp::preferMixedPrecision = previousAmp;
    CudaAmp::useLossScaling = previousLossScaling;
    CudaAdam::preferInt8Moments = previousInt8;
    CudaAdam::preferCpuOffload = previousCpu;
}

void CudaLanguageModel::uploadFrom(const LanguageModel& host) {
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaLanguageModel::uploadFrom no CUDA device");

    this->tieEmbeddingProjection = host.tieEmbeddingProjection;
    this->tokenEmbeddingWeight.upload(host.tokenEmbedding.weight);
    this->blocks.clear();
    this->blocks.reserve(host.blocks.size());
    for (const TransformerBlock& block : host.blocks)
        this->blocks.push_back(CudaTransformerBlock::createFrom(block));

    this->finalNorm.uploadFrom(host.finalNorm);
    if (this->tieEmbeddingProjection)
        this->projectionWeight.free();
    else
        this->projectionWeight.upload(host.outputProjection.weight);
    this->projectionBias.upload(host.outputProjection.bias);
    this->maximumPositionCount = host.maximumPositionCount;
    this->kvCaches.clear();
    this->kvCaches.resize(this->blocks.size());
    this->trainStateReady = false;
    CudaAmp::registerMasterWeight(this->tokenEmbeddingWeight.buffer.deviceData, this->tokenEmbeddingWeight.elementCount());
    if (!this->tieEmbeddingProjection)
        CudaAmp::registerMasterWeight(this->projectionWeight.buffer.deviceData, this->projectionWeight.elementCount());
}

CudaLanguageModel CudaLanguageModel::createFrom(const LanguageModel& host) {
    CudaLanguageModel device;
    device.uploadFrom(host);
    return device;
}

void CudaLanguageModel::forwardInto(const std::vector<int>& tokenIds, CudaMatrix& outLogits, int segmentLength) {
    if (tokenIds.empty()) throw std::invalid_argument("CudaLanguageModel::forwardInto empty tokenIds");
    if (this->tokenEmbeddingWeight.empty()) throw std::logic_error("CudaLanguageModel::forwardInto weights not uploaded");
    if (!CudaMatmul::isAvailable()) throw std::runtime_error("CudaLanguageModel::forwardInto no CUDA device");

    if (segmentLength > 0) {
        if (static_cast<int>(tokenIds.size()) % segmentLength != 0)
            throw std::invalid_argument("CudaLanguageModel::forwardInto tokenCount not divisible by segmentLength");
        if (segmentLength > this->maximumPositionCount)
            throw std::invalid_argument("CudaLanguageModel::forwardInto segmentLength exceeds maximumPositionCount");
    } else if (static_cast<int>(tokenIds.size()) > this->maximumPositionCount) {
        throw std::invalid_argument("CudaLanguageModel::forwardInto sequence longer than maximumPositionCount");
    }

    this->tokenIdsBuffer.ensureCapacity(tokenIds.size());
    this->tokenIdsBuffer.copyFromHost(tokenIds.data(), tokenIds.size());

    CudaOps::embeddingGatherInto(this->tokenEmbeddingWeight, this->tokenIdsBuffer, tokenIds.size(), this->hidden);

    if (this->activationCheckpointingActive()) {
        if (this->useHalfActivationCheckpoints()) {
            if (this->blockInputCheckpointsHalf.size() != this->blocks.size())
                this->blockInputCheckpointsHalf.resize(this->blocks.size());
            for (CudaHalfMatrix& checkpoint : this->blockInputCheckpointsHalf)
                checkpoint.ensureSize(this->tokenEmbeddingWeight.cols, tokenIds.size());
        } else {
            if (this->blockInputCheckpoints.size() != this->blocks.size())
                this->blockInputCheckpoints.resize(this->blocks.size());
            for (CudaMatrix& checkpoint : this->blockInputCheckpoints)
                checkpoint.ensureSize(this->tokenEmbeddingWeight.cols, tokenIds.size());
        }
    }

    for (size_t blockIndex = 0; blockIndex < this->blocks.size(); ++blockIndex) {
        if (this->activationCheckpointingActive()) {
            if (this->useHalfActivationCheckpoints())
                CudaAmp::castToHalfSaturated(this->hidden, this->blockInputCheckpointsHalf[blockIndex]);
            else
                CudaOps::copyInto(this->hidden, this->blockInputCheckpoints[blockIndex]);
        }
        this->blocks[blockIndex].forward(this->hidden, this->normalized, segmentLength);
        CudaMatrix swapBuffer = std::move(this->hidden);
        this->hidden = std::move(this->normalized);
        this->normalized = std::move(swapBuffer);
    }

    this->finalNorm.forward(this->hidden, this->normalized);
    CudaMatrix::multiplyBiasInto(this->lmHeadWeight(), this->normalized, this->projectionBias, outLogits);
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
    CudaMatrix::multiplyBiasInto(this->lmHeadWeight(), this->normalized, this->projectionBias, outLogits);
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
    CudaMatrix::multiplyBiasInto(this->lmHeadWeight(), this->normalized, this->projectionBias, outLogits);
}

CudaLanguageModelVramBreakdown CudaLanguageModel::estimateVramBreakdown(
    int vocabularySize,
    int embeddingDim,
    int blockCount,
    int headCount,
    int maximumPositionCount,
    ActivationCheckpointMode checkpointMode,
    bool preferMixedPrecision,
    bool preferInt8AdamMoments,
    bool preferCpuAdamOffload,
    bool preferMuon,
    bool tieEmbeddingProjection,
    int maxPackedColumns,
    int logitChunkRows,
    size_t displaySafetyBytes,
    bool preferFp16GpuWeights,
    bool preferHostGradients
) {
    if (vocabularySize <= 0 || embeddingDim <= 0 || blockCount <= 0 || headCount <= 0 || maximumPositionCount <= 0)
        throw std::invalid_argument("CudaLanguageModel::estimateVramBreakdown invalid dims");
    if (embeddingDim % headCount != 0)
        throw std::invalid_argument("CudaLanguageModel::estimateVramBreakdown embeddingDim must divide headCount");
    if (maxPackedColumns < 0 || logitChunkRows <= 0)
        throw std::invalid_argument("CudaLanguageModel::estimateVramBreakdown invalid pack/chunk");
    if (preferFp16GpuWeights && (!preferMixedPrecision || !preferCpuAdamOffload))
        throw std::invalid_argument("CudaLanguageModel::estimateVramBreakdown fp16GpuWeights requires AMP + cpuAdam");
    if (preferHostGradients && !preferCpuAdamOffload)
        throw std::invalid_argument("CudaLanguageModel::estimateVramBreakdown hostGrads requires cpuAdam");

    const size_t d = static_cast<size_t>(embeddingDim);
    const size_t v = static_cast<size_t>(vocabularySize);
    const size_t layerCount = static_cast<size_t>(blockCount);
    const size_t pos = static_cast<size_t>(maximumPositionCount);
    const size_t headDim = d / static_cast<size_t>(headCount);
    const size_t pairCount = headDim / 2ull;
    const size_t h = (2ull * d * 4ull) / 3ull;
    const size_t floatBytes = sizeof(float);
    const size_t halfBytes = 2ull;

    size_t paramElements = 0;
    paramElements += v * d;
    paramElements += d;
    paramElements += v;
    if (!tieEmbeddingProjection)
        paramElements += v * d;
    paramElements += layerCount * (
        d
        + 4ull * d * d
        + d
        + 2ull * h * d
        + 2ull * h
        + d * h
        + d
    );

    const size_t fp32TrainableAll = paramElements * floatBytes;
    const size_t fusedMirrorFp32 = layerCount * (3ull * d * d + 2ull * h * d) * floatBytes;
    const size_t ropeTables = layerCount * (2ull * pos * pairCount) * floatBytes;

    // Embedding + untied projection stay FP32 for the CE / LM-head path.
    size_t embedFp32Elements = v * d;
    if (!tieEmbeddingProjection)
        embedFp32Elements += v * d;
    const size_t embedFp32Bytes = embedFp32Elements * floatBytes;

    // Small tensors kept FP32 on GPU in fp16-working mode (gammas + biases + embed/proj).
    size_t smallFp32Elements = d + v; // finalNorm + projection bias
    smallFp32Elements += layerCount * (2ull * d + 2ull * h + d); // attn/ffn gamma + gate/up/down bias
    const size_t smallFp32Bytes = smallFp32Elements * floatBytes + embedFp32Bytes;

    // Gradients: full FP32, or only embed/bias/gamma when large weight grads live on host.
    size_t gradients = fp32TrainableAll;
    if (preferHostGradients) {
        size_t hostGradElements = layerCount * (4ull * d * d + 2ull * h * d + d * h); // q,k,v,o + gate,up + down
        gradients = fp32TrainableAll - hostGradElements * floatBytes;
    }

    // 2D weights that live as FP16 working copies on GPU (out, down, fused qkv+gateUp).
    size_t fp16WorkingElements = 0;
    fp16WorkingElements += layerCount * (
        d * d // output
        + d * h // down
        + 3ull * d * d // fused qkv
        + 2ull * h * d // fused gateUp
    );
    const size_t fp16WorkingBytes = fp16WorkingElements * halfBytes;

    size_t adamWeights = 0;
    size_t muonWeights = 0;
    adamWeights += v * d + d + v;
    if (!tieEmbeddingProjection)
        adamWeights += v * d;
    {
        const size_t hiddenWeight = 4ull * d * d + 2ull * h * d + d * h;
        const size_t auxWeight = 2ull * d + 2ull * h + d;
        if (preferMuon) {
            muonWeights += layerCount * hiddenWeight;
            adamWeights += layerCount * auxWeight;
        } else {
            adamWeights += layerCount * (hiddenWeight + auxWeight);
        }
    }
    adamWeights *= floatBytes;
    muonWeights *= floatBytes;

    size_t adamMoments = 0;
    if (!preferCpuAdamOffload) {
        if (preferInt8AdamMoments)
            adamMoments = adamWeights / 2ull + adamWeights / 128ull;
        else
            adamMoments = 2ull * adamWeights;
    }
    const size_t muonMoments = preferMuon ? muonWeights : 0ull;

    size_t fp16Mirrors = 0;
    size_t fp32Trainable = fp32TrainableAll;
    size_t fusedMirror = fusedMirrorFp32;
    if (preferFp16GpuWeights) {
        fp32Trainable = smallFp32Bytes;
        fusedMirror = 0;
        fp16Mirrors = fp16WorkingBytes;
    } else if (preferMixedPrecision) {
        fp16Mirrors = (fp32TrainableAll + fusedMirrorFp32) / 2ull;
    }

    const size_t chunkRows = static_cast<size_t>((std::min)(logitChunkRows, vocabularySize));
    size_t perColumn = 0;
    perColumn += 5ull * d * floatBytes;
    perColumn += 2ull * chunkRows * floatBytes;
    perColumn += d * floatBytes;
    perColumn += 3ull * floatBytes;
    perColumn += 7ull * sizeof(int);
    constexpr size_t maxCachedLogitsBytes = 512ull * 1024ull * 1024ull;
    if (v * floatBytes * 8192ull <= maxCachedLogitsBytes)
        perColumn += v * floatBytes;

    const size_t attnScratch = 15ull * d * floatBytes;
    const size_t ffnScratch = (8ull * h + 4ull * d) * floatBytes;
    if (checkpointMode != ActivationCheckpointMode::Off) {
        if (preferMixedPrecision)
            perColumn += layerCount * d * 2ull + d * floatBytes;
        else
            perColumn += layerCount * d * floatBytes;
    }
    if (checkpointMode == ActivationCheckpointMode::Selective) {
        perColumn += layerCount * ffnScratch;
        perColumn += attnScratch;
    } else {
        perColumn += layerCount * (attnScratch + ffnScratch);
    }
    if (preferMixedPrecision)
        perColumn += d * 2ull;

    const size_t packCols = static_cast<size_t>((std::max)(0, maxPackedColumns));
    constexpr double footprintSlack = 1.40;
    const size_t packWorkspace = static_cast<size_t>(static_cast<double>(perColumn * packCols) * footprintSlack);

    CudaLanguageModelVramBreakdown out;
    out.parameterElements = paramElements;
    out.fp32TrainableWeightBytes = fp32Trainable;
    out.fusedMirrorBytes = fusedMirror;
    out.ropeTableBytes = ropeTables;
    out.gradientBytes = gradients;
    out.fp16AmpMirrorBytes = fp16Mirrors;
    out.adamMomentBytes = adamMoments;
    out.muonMomentBytes = muonMoments;
    out.bytesPerPackedColumn = perColumn;
    out.packWorkspaceBytes = packWorkspace;
    out.displaySafetyBytes = displaySafetyBytes;
    out.weightsResidentBytes = preferFp16GpuWeights
        ? (fp32Trainable + fp16Mirrors + ropeTables)
        : (fp32Trainable + fusedMirror + ropeTables);
    out.staticTrainBytes =
        out.weightsResidentBytes
        + out.gradientBytes
        + (preferFp16GpuWeights ? 0ull : out.fp16AmpMirrorBytes)
        + out.adamMomentBytes
        + out.muonMomentBytes;
    out.peakEstimateBytes = out.staticTrainBytes + out.packWorkspaceBytes + out.displaySafetyBytes;
    return out;
}

static void logVramBreakdown(const char* label, const CudaLanguageModelVramBreakdown& breakdown) {
    auto miB = [](size_t bytes) -> double {
        return static_cast<double>(bytes) / (1024.0 * 1024.0);
    };
    SmokeLog::result(label,
        "params=%.2fB  weights=%.0f  fusedMirrors=%.0f  rope=%.0f  grads=%.0f  fp16Mirrors=%.0f  adam=%.0f  muon=%.0f MiB",
        static_cast<double>(breakdown.parameterElements) / 1.0e9,
        miB(breakdown.fp32TrainableWeightBytes),
        miB(breakdown.fusedMirrorBytes),
        miB(breakdown.ropeTableBytes),
        miB(breakdown.gradientBytes),
        miB(breakdown.fp16AmpMirrorBytes),
        miB(breakdown.adamMomentBytes),
        miB(breakdown.muonMomentBytes));
    SmokeLog::result("  resident",
        "weights+rope=%.0f  staticTrain=%.0f  packWs=%.0f  safety=%.0f  peak=%.0f MiB  (%.0f B/col)",
        miB(breakdown.weightsResidentBytes),
        miB(breakdown.staticTrainBytes),
        miB(breakdown.packWorkspaceBytes),
        miB(breakdown.displaySafetyBytes),
        miB(breakdown.peakEstimateBytes),
        static_cast<double>(breakdown.bytesPerPackedColumn));
}

void CudaLanguageModel::runScale4BVramProbeDemo() {
    if (!CudaMatmul::isAvailable()) {
        SmokeLog::skip("scale-4B VRAM probe");
        return;
    }

    const int vocab = 32000;
    const int embed = 3072;
    const int blocks = 34;
    const int heads = 48;
    const int pos = 2048;
    const int seq = 512;
    const int packColsTarget = seq;
    const size_t safety = 2560ull * 1024ull * 1024ull;

    SmokeLog::section("scale-4B VRAM probe");
    SmokeLog::result(
        "shape",
        "vocab=%d embed=%d blocks=%d heads=%d headDim=%d pos=%d ffnH=%d seqFloor=%d",
        vocab, embed, blocks, heads, embed / heads, pos, (2 * embed * 4) / 3, seq);

    size_t freeBytes = 0;
    size_t totalBytes = 0;
    CudaMatmul::throwIfCudaFailed(cudaMemGetInfo(&freeBytes, &totalBytes), "scale-4B memGetInfo");
    SmokeLog::result(
        "device",
        "total=%.0f MiB  free=%.0f MiB",
        static_cast<double>(totalBytes) / (1024.0 * 1024.0),
        static_cast<double>(freeBytes) / (1024.0 * 1024.0));

    struct Scenario {
        const char* name;
        ActivationCheckpointMode ckpt;
        bool amp;
        bool int8Adam;
        bool cpuAdam;
        bool muon;
        bool fp16Weights;
        bool hostGrads;
        int packCols;
    };
    const Scenario scenarios[] = {
        {"A adam/int8/gpu ckpt=sel pack=1seq", ActivationCheckpointMode::Selective, true, true, false, false, false, false, packColsTarget},
        {"B fp16w+cpuAdam+hostGrads ckpt=sel", ActivationCheckpointMode::Selective, true, false, true, false, true, true, packColsTarget},
        {"C adam/int8/gpu ckpt=off pack=1seq", ActivationCheckpointMode::Off, true, true, false, false, false, false, packColsTarget},
        {"D muon+adam/int8 ckpt=sel pack=1seq", ActivationCheckpointMode::Selective, true, true, false, true, false, false, packColsTarget},
        {"E fp16w+cpuAdam+hostGrads pack=0", ActivationCheckpointMode::Selective, true, false, true, false, true, true, 0},
    };

    for (const Scenario& scenario : scenarios) {
        const CudaLanguageModelVramBreakdown breakdown = CudaLanguageModel::estimateVramBreakdown(
            vocab, embed, blocks, heads, pos,
            scenario.ckpt, scenario.amp, scenario.int8Adam, scenario.cpuAdam, scenario.muon,
            true, scenario.packCols, 2048, safety, scenario.fp16Weights, scenario.hostGrads);
        logVramBreakdown(scenario.name, breakdown);
        const bool fitsStatic = breakdown.staticTrainBytes + breakdown.displaySafetyBytes <= totalBytes;
        const bool fitsPeak = breakdown.peakEstimateBytes <= totalBytes;
        const double deficitStatic = fitsStatic
            ? 0.0
            : static_cast<double>(breakdown.staticTrainBytes + breakdown.displaySafetyBytes - totalBytes) / (1024.0 * 1024.0);
        const double deficitPeak = fitsPeak
            ? 0.0
            : static_cast<double>(breakdown.peakEstimateBytes - totalBytes) / (1024.0 * 1024.0);
        SmokeLog::result(
            "  fit",
            "static+safety %s  peak %s  deficitStatic=%.0f MiB  deficitPeak=%.0f MiB",
            fitsStatic ? "OK" : "OOM",
            fitsPeak ? "OK" : "OOM",
            deficitStatic,
            deficitPeak);
    }

    SmokeLog::section("scale-4B alloc probe (scenario B: fp16 w + cpu Adam + host grads)");
    const CudaLanguageModelVramBreakdown a = CudaLanguageModel::estimateVramBreakdown(
        vocab, embed, blocks, heads, pos,
        ActivationCheckpointMode::Selective, true, false, true, false,
        true, 0, 2048, safety, true, true);
    const CudaLanguageModelVramBreakdown aPack = CudaLanguageModel::estimateVramBreakdown(
        vocab, embed, blocks, heads, pos,
        ActivationCheckpointMode::Selective, true, false, true, false,
        true, packColsTarget, 2048, safety, true, true);

    struct Piece {
        const char* name;
        size_t bytes;
    };
    const Piece pieces[] = {
        {"fp32 small weights (gamma/bias/embed)", a.fp32TrainableWeightBytes},
        {"fp16 working weights", a.fp16AmpMirrorBytes},
        {"RoPE tables", a.ropeTableBytes},
        {"fp32 grads (embed/bias/gamma only)", a.gradientBytes},
        {"pack workspace (1x seq)", aPack.packWorkspaceBytes},
    };

    size_t allocated = 0;
    std::vector<void*> holds;
    holds.reserve(16);
    auto freeHolds = [&]() {
        for (void* pointer : holds) {
            if (pointer != nullptr)
                cudaFree(pointer);
        }
        holds.clear();
    };

    for (const Piece& piece : pieces) {
        void* pointer = nullptr;
        const cudaError_t status = cudaMalloc(&pointer, piece.bytes);
        if (status != cudaSuccess) {
            size_t freeNow = 0;
            size_t totalNow = 0;
            cudaMemGetInfo(&freeNow, &totalNow);
            SmokeLog::result(
                "alloc FAIL",
                "%s  need=%.0f MiB  alreadyHeld=%.0f MiB  freeNow=%.0f MiB  err=%s",
                piece.name,
                static_cast<double>(piece.bytes) / (1024.0 * 1024.0),
                static_cast<double>(allocated) / (1024.0 * 1024.0),
                static_cast<double>(freeNow) / (1024.0 * 1024.0),
                cudaGetErrorString(status));
            cudaGetLastError();
            freeHolds();
            SmokeLog::note("blocker: FP32 weight residency (and mirrors) before grads/opt — layout change required for 4B");
            return;
        }
        holds.push_back(pointer);
        allocated += piece.bytes;
        size_t freeNow = 0;
        size_t totalNow = 0;
        cudaMemGetInfo(&freeNow, &totalNow);
        SmokeLog::result(
            "alloc OK",
            "%s  +%.0f MiB  held=%.0f  free=%.0f MiB",
            piece.name,
            static_cast<double>(piece.bytes) / (1024.0 * 1024.0),
            static_cast<double>(allocated) / (1024.0 * 1024.0),
            static_cast<double>(freeNow) / (1024.0 * 1024.0));
    }
    freeHolds();
    SmokeLog::note("unexpected: all scenario-A pieces fit as raw cudaMalloc (fragmentation/real objects may still OOM)");
}

void CudaLanguageModel::runConsumerVramDemo(
    int vocabularySize,
    int embeddingDim,
    int maximumPositionCount,
    int blockCount,
    int headCount,
    int exampleCount,
    int epochs,
    int batchSize,
    int gradientAccumulationSteps
) {
    if (!CudaMatmul::isAvailable()) {
        SmokeLog::skip("LanguageModel consumer VRAM");
        return;
    }
    if (vocabularySize <= 0 || embeddingDim <= 0 || maximumPositionCount <= 0
        || blockCount <= 0 || headCount <= 0 || exampleCount <= 0 || epochs <= 0)
        throw std::invalid_argument("CudaLanguageModel::runConsumerVramDemo invalid dims");
    if (embeddingDim % headCount != 0)
        throw std::invalid_argument("CudaLanguageModel::runConsumerVramDemo embeddingDim must divide headCount");

    SmokeLog::section("consumer VRAM");

    size_t freeBefore = 0;
    size_t totalBytes = 0;
    CudaMatmul::throwIfCudaFailed(cudaMemGetInfo(&freeBefore, &totalBytes), "consumer VRAM memGetInfo before");

    LanguageModel host(vocabularySize, embeddingDim, maximumPositionCount, Adam(0.001f), blockCount, headCount);
    host.enableCuda();
    host.enableCudaTrain();
    host.setCudaPreferFlashAttention(true);

    size_t freeAfterSetup = 0;
    CudaMatmul::throwIfCudaFailed(cudaMemGetInfo(&freeAfterSetup, &totalBytes), "consumer VRAM memGetInfo after setup");

    LanguageModelDataset trainDataset;
    trainDataset.vocabularySize = vocabularySize;
    trainDataset.examples.reserve(static_cast<size_t>(exampleCount));
    unsigned state = 42u;
    auto nextU32 = [&state]() -> unsigned {
        state = state * 1664525u + 1013904223u;
        return state;
    };
    for (int exampleIndex = 0; exampleIndex < exampleCount; ++exampleIndex) {
        const float u1 = (static_cast<float>(nextU32() % 10000u) + 0.5f) / 10000.0f;
        const float u2 = (static_cast<float>(nextU32() % 10000u) + 0.5f) / 10000.0f;
        const float radius = std::sqrt(-2.0f * std::log(u1));
        const float angle = 6.2831853f * u2;
        int length = static_cast<int>(std::lround(128.0f + 70.0f * radius * std::cos(angle)));
        if (length < 2) length = 2;
        if (length > maximumPositionCount) length = maximumPositionCount;

        std::vector<int> tokenIds(static_cast<size_t>(length + 1));
        for (int index = 0; index <= length; ++index)
            tokenIds[static_cast<size_t>(index)] = static_cast<int>(nextU32() % static_cast<unsigned>(vocabularySize));
        trainDataset.examples.push_back(LanguageModelDataset::fromTokenIds(tokenIds, vocabularySize, false));
    }

    LanguageModelDataset emptyTest;
    emptyTest.vocabularySize = vocabularySize;

    const int predictions = trainDataset.totalPredictionCount();
    SmokeLog::result(
        "consumer VRAM config",
        "vocab=%d embed=%d pos=%d blocks=%d heads=%d examples=%d preds=%d batch=%d accum=%d maxPackCols=%d lossScale=%s",
        vocabularySize,
        embeddingDim,
        maximumPositionCount,
        blockCount,
        headCount,
        exampleCount,
        predictions,
        batchSize,
        gradientAccumulationSteps,
        host.cudaMaxPackedColumns(),
        (CudaAmp::useLossScaling ? "on" : "off")
    );
    SmokeLog::result(
        "consumer VRAM mem",
        "total=%.0f MiB  freeBefore=%.0f  freeAfterSetup=%.0f  setupUsed=%.0f  maxPackCols=%d",
        static_cast<double>(totalBytes) / (1024.0 * 1024.0),
        static_cast<double>(freeBefore) / (1024.0 * 1024.0),
        static_cast<double>(freeAfterSetup) / (1024.0 * 1024.0),
        static_cast<double>(freeBefore - freeAfterSetup) / (1024.0 * 1024.0),
        host.cudaMaxPackedColumns()
    );

    host.train(trainDataset, emptyTest, epochs, 1, batchSize, gradientAccumulationSteps);

    size_t freeAfterTrain = 0;
    CudaMatmul::throwIfCudaFailed(cudaMemGetInfo(&freeAfterTrain, &totalBytes), "consumer VRAM memGetInfo after train");
    SmokeLog::result(
        "consumer VRAM after",
        "free=%.0f MiB  setupUsed=%.0f MiB",
        static_cast<double>(freeAfterTrain) / (1024.0 * 1024.0),
        static_cast<double>(freeBefore - freeAfterSetup) / (1024.0 * 1024.0)
    );
}

void CudaLanguageModel::runSmokeDemo(int vocabularySize, int embeddingDim, int sequenceLength, int blockCount, int headCount) {
    if (!CudaMatmul::isAvailable()) {
        SmokeLog::skip("LanguageModel fwd");
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

    SmokeLog::result("LanguageModel fwd", "vocab=%d embed=%d seq=%d  cpu=%.2fms  gpu=%.2fms  diff=%.2e",
        vocabularySize, embeddingDim, sequenceLength, cpuMilliseconds, static_cast<double>(deviceMilliseconds), maximumDifference);
    (void)warmLogits;
    (void)uploadMilliseconds;
    (void)downloadMilliseconds;
    (void)blockCount;
    (void)headCount;
}

void CudaLanguageModel::runKvCacheSmokeDemo(int vocabularySize, int embeddingDim, int sequenceLength, int blockCount, int headCount) {
    if (!CudaMatmul::isAvailable()) {
        SmokeLog::skip("LanguageModel KV");
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

    SmokeLog::result("LanguageModel KV", "vocab=%d embed=%d seq=%d  cache=%d  diff=%.2e",
        vocabularySize, embeddingDim, sequenceLength, device.kvCaches.empty() ? 0 : device.kvCaches[0].length, maximumDifference);
    (void)blockCount;
    (void)headCount;
}
