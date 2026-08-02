#include "NeuralNet/Cuda/CudaFeedForward.hpp"
#include "NeuralNet/Cuda/CudaLanguageModel.hpp"
#include "NeuralNet/Cuda/CudaMatmul.hpp"
#include "NeuralNet/Cuda/CudaOps.hpp"
#include "NeuralNet/Cuda/CudaRMSNorm.hpp"
#include "NeuralNet/Cuda/CudaCausalSelfAttention.hpp"
#include "NeuralNet/Cuda/CudaAdam.hpp"
#include "NeuralNet/Cuda/CudaAmp.hpp"
#include "NeuralNet/Cuda/CudaTransformerBlock.hpp"
#include "NeuralNet/Layers/CausalSelfAttention.hpp"
#include "NeuralNet/Tokenizer/BPETokenizer.hpp"
#include "NeuralNet/Data/LanguageModelChunkSource.hpp"
#include "NeuralNet/Data/LanguageModelDataset.hpp"
#include "NeuralNet/Data/ArrowChunkReader.hpp"
#include "NeuralNet/Network/LanguageModel.hpp"
#include "NeuralNet/Optimizers/Adam.hpp"
#include "NeuralNet/Utils/SmokeLog.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

/// <summary>
/// demo: causal LM on SERA text
/// token embed + RoPE multi head attention + RMSNorm + SwiGLU FFN + vocab projection
/// </summary>
int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);

    const bool runSmokes = false;
    const bool runSpeedBench = false;
    const bool runGate40k = false;
    const bool runFlashParity256 = false;
    const bool runScale100M = false;
    const bool runGraphCheck = false;
    const bool runEpilogueCheck = false;
    const bool runFfnBwdCheck = false;
    const bool runQkvCheck = false;
    const bool runPackBudgetBench = false;
    const bool runScaleProfile = false;
    const bool runArrowCorpusSmoke = false;
    const bool runBpeBench = false;

    // Arrow HF disk layout (sera_best_subset) via from-scratch ArrowChunkReader; JSONL still works
    const bool useArrowCorpus = true;
    const std::string samplePath = useArrowCorpus
        ? "../SERA-Data/sera_best_subset"
        : "../SERA-Data/sera_sample.jsonl";
    const size_t maximumTextCharacters = 1200;
    const size_t maximumTokenCount = 512;
    const float trainRatio = 0.8f;
    const int embeddingDim = 768;
    const int blockCount = 8;
    const int headCount = 12;
    const int maximumPositionCount = static_cast<int>(maximumTokenCount);
    const int trainEpochs = 2;
    const int trainBatchSize = 64;
    const int trainGradAccum = 4;
    const int chunkExampleCount = 2048;
    const int tokenizerVocabSize = 4000;
    const int tokenizerSampleRows = 2000;
    const int testReservoirCap = 512;
    const bool preferCpuAdamOffload = false;
    const bool useCheckpointing = false;

    if (runBpeBench) {
        SmokeLog::section("bpe bench");
        const std::string path = useArrowCorpus
            ? "../SERA-Data/sera_best_subset"
            : "../SERA-Data/sera_sample.jsonl";
        LanguageModelChunkSource source(path, maximumTextCharacters, maximumTokenCount, chunkExampleCount, trainRatio, 42u, testReservoirCap);
        std::vector<std::string> sample = source.prepareTokenizerSample(tokenizerSampleRows);
        if (sample.empty()) {
            SmokeLog::note("bpe bench: empty sample");
            return 1;
        }
        BPETokenizer tokenizer;
        tokenizer.train(sample, tokenizerVocabSize);
        const auto encodeStart = std::chrono::steady_clock::now();
        size_t tokenCount = 0;
        for (const std::string& text : sample)
            tokenCount += tokenizer.encode(text).size();
        const double encodeSec = std::chrono::duration<double>(std::chrono::steady_clock::now() - encodeStart).count();
        SmokeLog::result(
            "bpe encode",
            "docs=%zu tokens=%zu sec=%.2f tokens/s=%.0f",
            sample.size(),
            tokenCount,
            encodeSec,
            encodeSec > 0.0 ? static_cast<double>(tokenCount) / encodeSec : 0.0);
        return 0;
    }

    if (runArrowCorpusSmoke) {
        SmokeLog::section("arrow-corpus");
        const std::string arrowPath = "../SERA-Data/sera_best_subset";
        ArrowChunkReader reader;
        reader.open(arrowPath);
        SmokeLog::note(
            ("shards=" + std::to_string(reader.shards().size())
                + " kind=" + std::string(reader.ipcKind() == ArrowChunkReader::IpcKind::Stream ? "stream" : "file"))
                .c_str());

        std::vector<CorpusRow> rows;
        reader.nextRows(rows, 3);
        if (rows.empty()) {
            SmokeLog::note("no rows decoded");
            return 1;
        }

        for (size_t index = 0; index < rows.size(); ++index) {
            const std::string preview = rows[index].text.size() > 80
                ? rows[index].text.substr(0, 80) + "..."
                : rows[index].text;
            SmokeLog::result(
                "arrow-row",
                "i=%zu source=%s textLen=%zu preview=%s",
                index,
                rows[index].source.c_str(),
                rows[index].text.size(),
                preview.c_str());
        }

        // Drain a bit further to ensure multi-batch works.
        size_t total = rows.size();
        while (reader.nextRows(rows, 512) || !rows.empty()) {
            total += rows.size();
            if (total >= 2500) break;
        }
        SmokeLog::result("arrow-corpus", "ok rowsRead=%zu sampled>=%zu", reader.rowsRead(), total);
        return 0;
    }

    if (runScale100M) {
        SmokeLog::section("scale-100M");
        if (!CudaMatmul::isAvailable()) {
            SmokeLog::skip("scale-100M (no CUDA)");
            return 1;
        }

        const std::string scalePath = "../SERA-Data/sera_scale.jsonl";
        const size_t scaleMaxTextChars = 4000;
        const size_t scaleMaxTokens = 512;
        const int scaleVocab = 16000;
        const int scaleEmbed = 768;
        const int scaleBlocks = 12;
        const int scaleHeads = 12;
        const int scalePos = static_cast<int>(scaleMaxTokens);
        const int scaleTokenizerRows = 8000;
        const int scaleChunkExamples = 2048;
        const int scaleTestCap = 512;
        const int scaleBatch = 32;
        const int scaleAccum = 4;
        const int scaleEpochs = 1;
        const int probeSeq = 256;

        try {
            LanguageModelChunkSource source(scalePath, scaleMaxTextChars, scaleMaxTokens, scaleChunkExamples, 0.8f, 42u, scaleTestCap);
            std::vector<std::string> tokenizerSample = source.prepareTokenizerSample(scaleTokenizerRows);
            if (tokenizerSample.empty()) {
                SmokeLog::note(("no usable rows from " + scalePath + " (run SERA-Data/export_sample.py)").c_str());
                return 1;
            }

            BPETokenizer tokenizer;
            tokenizer.train(tokenizerSample, scaleVocab);
            source.setTokenizer(&tokenizer);
            source.materialize();
            if (source.trainExampleCount() <= 0) {
                SmokeLog::note("scale-100M: no train examples");
                return 1;
            }

            LanguageModelDataset promptChunk;
            source.rewindTrain();
            source.nextTrainChunk(promptChunk);
            source.rewindTrain();
            if (promptChunk.examples.empty()) {
                SmokeLog::note("scale-100M: empty prompt chunk");
                return 1;
            }

            SmokeLog::result(
                "data",
                "stream=%s  train=%d  test=%d  vocab=%d  positions=%d  chunk=%d",
                scalePath.c_str(),
                source.trainExampleCount(),
                source.testDataset().size(),
                tokenizer.vocabSize(),
                source.trainPredictionCount(),
                source.chunkExampleCount());

            SmokeLog::note("building ~100M LanguageModel (CPU init may take a while)...");
            LanguageModel model(tokenizer.vocabSize(), scaleEmbed, scalePos, Adam(0.001f), scaleBlocks, scaleHeads);
            const size_t paramCount = model.parameterElementCount();
            SmokeLog::result(
                "model",
                "params=%.2fM  fp32MiB=%.0f  blocks=%d  heads=%d  embed=%d  vocab=%d  maxTok=%d  tieEmbed=on",
                static_cast<double>(paramCount) / 1.0e6,
                static_cast<double>(paramCount) * 4.0 / (1024.0 * 1024.0),
                scaleBlocks,
                scaleHeads,
                scaleEmbed,
                tokenizer.vocabSize(),
                scalePos);

            model.enableCuda();
            model.setCudaPreferCpuAdamOffload(false);
            model.setCudaPreferInt8AdamMoments(true);
            model.setCudaPreferFlashAttention(true);
            model.enableCudaTrain();
            model.enableActivationCheckpointing(true);
            model.setCudaPreferTrainGraph(true);
            model.applyCudaVramPackBudget(0.55f);

            size_t freeAfterSetup = 0;
            size_t totalBytes = 0;
            if (cudaMemGetInfo(&freeAfterSetup, &totalBytes) != cudaSuccess)
                throw std::runtime_error("scale-100M memGetInfo failed");

            SmokeLog::result(
                "vram",
                "maxPackCols=%d  freeMiB=%.0f  totalMiB=%.0f  ckpt=on  int8Adam=on  flash=on  graph=on",
                model.cudaMaxPackedColumns(),
                static_cast<double>(freeAfterSetup) / (1024.0 * 1024.0),
                static_cast<double>(totalBytes) / (1024.0 * 1024.0));

            const double tokensPerSecond = model.probeCudaPackedTrainTokensPerSecond(probeSeq, 3, 8);
            SmokeLog::result(
                "throughput",
                "seq=%d  pack<=32  tokens/s=%.0f",
                probeSeq,
                tokensPerSecond);

            SmokeLog::note("starting 1 epoch streamed train...");
            model.train(source, scaleEpochs, 1, scaleBatch, scaleAccum);
            model.saveCheckpoint("sera_100m.snlm", true);

            if (!source.testDataset().examples.empty())
                SmokeLog::result("final", "testLoss=%.6f", model.averageLoss(source.testDataset()));
            SmokeLog::result("checkpoint", "saved sera_100m.snlm (weights+adam)");

            const std::vector<int> prompt = promptChunk.examples[0].inputTokenIds;
            const std::vector<int> greedy = model.generate(prompt, 32, 0.0f, 0, 7u);
            SmokeLog::section("generate");
            std::cout << "  prompt:  " << tokenizer.decode(prompt) << '\n';
            std::cout << "  greedy:  " << tokenizer.decode(greedy) << '\n';

            SmokeLog::result("scale-100M", "PASS  tokens/s=%.0f  params=%.2fM", tokensPerSecond, static_cast<double>(paramCount) / 1.0e6);
            return 0;
        } catch (const std::exception& ex) {
            SmokeLog::result("scale-100M", "FAILED: %s", ex.what());
            return 1;
        }
    }

    if (runFlashParity256) {
        SmokeLog::section("flash parity seq256");
        if (!CudaMatmul::isAvailable()) {
            SmokeLog::skip("flash parity (no CUDA)");
            return 1;
        }
        try {
            CudaCausalSelfAttention::runFlashParitySmokeDemo(768, 12, 128, 512);
            CudaCausalSelfAttention::runFlashParitySmokeDemo(768, 12, 256, 512);
            CudaCausalSelfAttention::runFlashParitySmokeDemo(768, 12, 1024, 1024);
            return 0;
        } catch (const std::exception& ex) {
            SmokeLog::result("flash parity", "FAILED: %s", ex.what());
            return 1;
        }
    }

    if (runGate40k) {
        SmokeLog::section("gate-40k");
        if (!CudaMatmul::isAvailable()) {
            SmokeLog::skip("gate-40k (no CUDA)");
            return 1;
        }
        try {
            CudaCausalSelfAttention::runFlashParitySmokeDemo(768, 12, 128, 512);
            CudaFeedForward::runBackwardSmokeDemo(256, 128);

            const bool previousAmp = CudaAmp::preferMixedPrecision;
            const bool previousInt8 = CudaAdam::preferInt8Moments;
            const bool previousCpu = CudaAdam::preferCpuOffload;
            CudaAmp::preferMixedPrecision = true;
            CudaAmp::useLossScaling = true;
            CudaAmp::resetLossScaler();
            CudaAdam::preferCpuOffload = false;
            CudaAdam::preferInt8Moments = true;

            const int vocab = 4000;
            const int seq = 256;
            const int warmupSteps = 6;
            const int timedSteps = 16;
            LanguageModel host(vocab, embeddingDim, maximumPositionCount, Adam(0.001f), blockCount, headCount);
            CudaLanguageModel device = CudaLanguageModel::createFrom(host);
            device.adam = CudaAdam(0.001f);
            device.setActivationCheckpointMode(ActivationCheckpointMode::Off);
            device.preferTrainGraph = true;
            device.maxPackedColumnsManual = false;
            device.applyVramPackBudget(0.58f, 1024ull * 1024ull * 1024ull);
            for (CudaTransformerBlock& block : device.blocks)
                block.attention.preferFlashAttention = true;
            device.ensureTrainState();

            const int packBatch = (std::max)(1, (std::min)(32, device.maxPackExamplesForSegment(seq)));
            std::vector<LanguageModelExample> examples(static_cast<size_t>(packBatch));
            unsigned rng = 91u;
            for (int exampleIndex = 0; exampleIndex < packBatch; ++exampleIndex) {
                examples[static_cast<size_t>(exampleIndex)].inputTokenIds.resize(static_cast<size_t>(seq));
                examples[static_cast<size_t>(exampleIndex)].targetTokenIds.resize(static_cast<size_t>(seq));
                for (size_t index = 0; index < static_cast<size_t>(seq); ++index) {
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
                throw std::runtime_error("gate-40k warmup sync failed");

            const auto start = std::chrono::steady_clock::now();
            for (int step = 0; step < timedSteps; ++step) {
                device.trainGradients.zeroInPlace();
                device.accumulatePackedExamples(packPointers.data(), packBatch, device.trainGradients);
                device.applyGradients(device.trainGradients, 1.0f / static_cast<float>(packBatch));
            }
            if (cudaDeviceSynchronize() != cudaSuccess)
                throw std::runtime_error("gate-40k timed sync failed");
            const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            const double tokensPerSecond = seconds > 0.0
                ? static_cast<double>(seq * packBatch) * static_cast<double>(timedSteps) / seconds
                : 0.0;

            size_t freeAfter = 0;
            size_t totalBytes = 0;
            cudaMemGetInfo(&freeAfter, &totalBytes);
            SmokeLog::result(
                "gate int8+noCkpt",
                "pack=%d maxPackCols=%d graph=%s tokens/s=%.0f freeMiB=%.0f %s",
                packBatch, device.maxPackedColumns,
                device.trainGraphExec != nullptr ? "on" : "off",
                tokensPerSecond,
                static_cast<double>(freeAfter) / (1024.0 * 1024.0),
                tokensPerSecond >= 40000.0 ? "PASS" : "BELOW_40k");

            CudaAmp::preferMixedPrecision = previousAmp;
            CudaAdam::preferInt8Moments = previousInt8;
            CudaAdam::preferCpuOffload = previousCpu;

            if (tokensPerSecond < 40000.0)
                return 2;
        } catch (const std::exception& ex) {
            SmokeLog::result("gate-40k", "FAILED: %s", ex.what());
            return 1;
        }
        return 0;
    }

    if (runGraphCheck) {
        SmokeLog::section("cuda graph microstep check");
        if (!CudaMatmul::isAvailable()) {
            SmokeLog::skip("cuda graph microstep check (no CUDA)");
            return 1;
        }
        try {
            const bool previousAmp = CudaAmp::preferMixedPrecision;
            const bool previousInt8 = CudaAdam::preferInt8Moments;
            const bool previousCpu = CudaAdam::preferCpuOffload;
            CudaAmp::preferMixedPrecision = true;
            CudaAmp::useLossScaling = true;
            CudaAmp::resetLossScaler();
            CudaAdam::preferCpuOffload = false;
            CudaAdam::preferInt8Moments = true;

            const int vocab = 4000;
            const int seq = 256;
            LanguageModel host(vocab, embeddingDim, maximumPositionCount, Adam(0.001f), blockCount, headCount);
            CudaLanguageModel device = CudaLanguageModel::createFrom(host);
            device.adam = CudaAdam(0.001f);
            device.setActivationCheckpointMode(ActivationCheckpointMode::Off);
            device.preferTrainGraph = true;
            device.maxPackedColumnsManual = false;
            device.applyVramPackBudget();
            for (CudaTransformerBlock& block : device.blocks)
                block.attention.preferFlashAttention = true;
            device.ensureTrainState();

            const int packBatch = (std::min)(32, device.maxPackExamplesForSegment(seq));
            std::vector<LanguageModelExample> examples(static_cast<size_t>(packBatch));
            unsigned rng = 91u;
            for (int exampleIndex = 0; exampleIndex < packBatch; ++exampleIndex) {
                examples[static_cast<size_t>(exampleIndex)].inputTokenIds.resize(static_cast<size_t>(seq));
                examples[static_cast<size_t>(exampleIndex)].targetTokenIds.resize(static_cast<size_t>(seq));
                for (size_t index = 0; index < static_cast<size_t>(seq); ++index) {
                    rng = rng * 1664525u + 1013904223u;
                    examples[static_cast<size_t>(exampleIndex)].inputTokenIds[index] = static_cast<int>(rng % static_cast<unsigned>(vocab));
                    rng = rng * 1664525u + 1013904223u;
                    examples[static_cast<size_t>(exampleIndex)].targetTokenIds[index] = static_cast<int>(rng % static_cast<unsigned>(vocab));
                }
            }
            std::vector<const LanguageModelExample*> packPointers(static_cast<size_t>(packBatch));
            for (int exampleIndex = 0; exampleIndex < packBatch; ++exampleIndex)
                packPointers[static_cast<size_t>(exampleIndex)] = &examples[static_cast<size_t>(exampleIndex)];

            for (int step = 0; step < 4; ++step) {
                device.trainGradients.zeroInPlace();
                device.accumulatePackedExamples(packPointers.data(), packBatch, device.trainGradients);
                device.applyGradients(device.trainGradients, 1.0f / static_cast<float>(packBatch));
            }
            if (cudaDeviceSynchronize() != cudaSuccess)
                throw std::runtime_error("graph check warmup sync failed");

            const bool graphReady = device.trainGraphExec != nullptr && device.preferTrainGraph;
            SmokeLog::result("cuda graph capture", "ready=%s pack=%d seg=%d",
                graphReady ? "yes" : "no", packBatch, seq);

            const auto start = std::chrono::steady_clock::now();
            const int timedSteps = 6;
            for (int step = 0; step < timedSteps; ++step) {
                device.trainGradients.zeroInPlace();
                device.accumulatePackedExamples(packPointers.data(), packBatch, device.trainGradients);
                device.applyGradients(device.trainGradients, 1.0f / static_cast<float>(packBatch));
            }
            if (cudaDeviceSynchronize() != cudaSuccess)
                throw std::runtime_error("graph check timed sync failed");
            const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            const double tokensPerSecond = seconds > 0.0
                ? static_cast<double>(seq * packBatch) * static_cast<double>(timedSteps) / seconds
                : 0.0;

            size_t freeAfter = 0;
            size_t totalBytes = 0;
            cudaMemGetInfo(&freeAfter, &totalBytes);
            SmokeLog::result("graph throughput", "pack=%d tokens/s=%.0f freeMiB=%.0f graph=%s",
                packBatch, tokensPerSecond, static_cast<double>(freeAfter) / (1024.0 * 1024.0),
                device.trainGraphExec != nullptr ? "on" : "off");

            CudaAmp::preferMixedPrecision = previousAmp;
            CudaAdam::preferInt8Moments = previousInt8;
            CudaAdam::preferCpuOffload = previousCpu;
        } catch (const std::exception& ex) {
            SmokeLog::result("cuda graph microstep check", "FAILED: %s", ex.what());
            return 1;
        }
        return 0;
    }

    if (runEpilogueCheck) {
        SmokeLog::section("bias epilogue check");
        if (!CudaMatmul::isAvailable()) {
            SmokeLog::skip("bias epilogue check (no CUDA)");
            return 1;
        }
        try {
            const bool previousAmp = CudaAmp::preferMixedPrecision;
            CudaAmp::preferMixedPrecision = true;

            CudaFeedForward::runSmokeDemo(256, 128);
            CudaFeedForward::runBackwardSmokeDemo(256, 128);

            // Direct fused-path probe at train-like shape (needs AMP size thresholds)
            FeedForward hostFfn = FeedForward::create(768, 4, 41u);
            CudaFeedForward deviceFfn = CudaFeedForward::createFrom(hostFfn);
            Matrix hostInput(768, 512, 0.0f);
            unsigned state = 77u;
            for (size_t index = 0; index < hostInput.data.size(); ++index) {
                state = state * 1664525u + 1013904223u;
                hostInput.data[index] = (static_cast<float>(state >> 8) / 16777216.0f) * 2.0f - 1.0f;
            }
            CudaMatrix deviceInput;
            deviceInput.upload(hostInput);
            CudaMatrix deviceOut;
            const bool gateUpFused = CudaMatrix::multiplyBiasInto(
                deviceFfn.gateUpWeight, deviceInput, deviceFfn.gateUpBias, deviceOut);
            CudaMatrix refOut;
            CudaMatrix::multiplyInto(deviceFfn.gateUpWeight, deviceInput, refOut);
            CudaOps::broadcastBiasAddInPlace(refOut, deviceFfn.gateUpBias);
            if (cudaDeviceSynchronize() != cudaSuccess)
                throw std::runtime_error("bias epilogue sync failed");
            const Matrix fusedHost = deviceOut.download();
            const Matrix refHost = refOut.download();
            float maxDiff = 0.0f;
            for (size_t index = 0; index < fusedHost.data.size(); ++index)
                maxDiff = (std::max)(maxDiff, std::fabs(fusedHost.data[index] - refHost.data[index]));
            SmokeLog::result("bias epilogue gemm", "fused=%s diff=%.2e", gateUpFused ? "yes" : "fallback", maxDiff);
            if (maxDiff > 5e-2f)
                throw std::runtime_error("bias epilogue numeric mismatch");
            // ROW-major feature-bias epilogue often has no algo on consumer GPUs; fallback is intentional and correct.

            const bool previousInt8 = CudaAdam::preferInt8Moments;
            const bool previousCpu = CudaAdam::preferCpuOffload;
            CudaAmp::useLossScaling = true;
            CudaAmp::resetLossScaler();
            CudaAdam::preferCpuOffload = false;
            CudaAdam::preferInt8Moments = true;

            const int vocab = 4000;
            const int seq = 256;
            LanguageModel host(vocab, embeddingDim, maximumPositionCount, Adam(0.001f), blockCount, headCount);
            CudaLanguageModel device = CudaLanguageModel::createFrom(host);
            device.adam = CudaAdam(0.001f);
            device.setActivationCheckpointMode(ActivationCheckpointMode::Off);
            device.maxPackedColumnsManual = false;
            device.applyVramPackBudget();
            for (CudaTransformerBlock& block : device.blocks)
                block.attention.preferFlashAttention = true;
            device.ensureTrainState();

            const int packBatch = (std::min)(32, device.maxPackExamplesForSegment(seq));
            std::vector<LanguageModelExample> examples(static_cast<size_t>(packBatch));
            unsigned rng = 91u;
            for (int exampleIndex = 0; exampleIndex < packBatch; ++exampleIndex) {
                examples[static_cast<size_t>(exampleIndex)].inputTokenIds.resize(static_cast<size_t>(seq));
                examples[static_cast<size_t>(exampleIndex)].targetTokenIds.resize(static_cast<size_t>(seq));
                for (size_t index = 0; index < static_cast<size_t>(seq); ++index) {
                    rng = rng * 1664525u + 1013904223u;
                    examples[static_cast<size_t>(exampleIndex)].inputTokenIds[index] = static_cast<int>(rng % static_cast<unsigned>(vocab));
                    rng = rng * 1664525u + 1013904223u;
                    examples[static_cast<size_t>(exampleIndex)].targetTokenIds[index] = static_cast<int>(rng % static_cast<unsigned>(vocab));
                }
            }
            std::vector<const LanguageModelExample*> packPointers(static_cast<size_t>(packBatch));
            for (int exampleIndex = 0; exampleIndex < packBatch; ++exampleIndex)
                packPointers[static_cast<size_t>(exampleIndex)] = &examples[static_cast<size_t>(exampleIndex)];

            for (int step = 0; step < 2; ++step) {
                device.trainGradients.zeroInPlace();
                device.accumulatePackedExamples(packPointers.data(), packBatch, device.trainGradients);
                device.applyGradients(device.trainGradients, 1.0f / static_cast<float>(packBatch));
            }
            if (cudaDeviceSynchronize() != cudaSuccess)
                throw std::runtime_error("epilogue check warmup sync failed");

            const auto start = std::chrono::steady_clock::now();
            const int timedSteps = 6;
            for (int step = 0; step < timedSteps; ++step) {
                device.trainGradients.zeroInPlace();
                device.accumulatePackedExamples(packPointers.data(), packBatch, device.trainGradients);
                device.applyGradients(device.trainGradients, 1.0f / static_cast<float>(packBatch));
            }
            if (cudaDeviceSynchronize() != cudaSuccess)
                throw std::runtime_error("epilogue check timed sync failed");
            const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            const double tokensPerSecond = seconds > 0.0
                ? static_cast<double>(seq * packBatch) * static_cast<double>(timedSteps) / seconds
                : 0.0;

            size_t freeAfter = 0;
            size_t totalBytes = 0;
            cudaMemGetInfo(&freeAfter, &totalBytes);
            SmokeLog::result("epilogue throughput", "pack=%d tokens/s=%.0f freeMiB=%.0f",
                packBatch, tokensPerSecond, static_cast<double>(freeAfter) / (1024.0 * 1024.0));

            CudaAmp::preferMixedPrecision = previousAmp;
            CudaAdam::preferInt8Moments = previousInt8;
            CudaAdam::preferCpuOffload = previousCpu;
        } catch (const std::exception& ex) {
            SmokeLog::result("bias epilogue check", "FAILED: %s", ex.what());
            return 1;
        }
        return 0;
    }

    if (runFfnBwdCheck) {
        SmokeLog::section("ffn bwd fusion check");
        if (!CudaMatmul::isAvailable()) {
            SmokeLog::skip("ffn bwd fusion check (no CUDA)");
            return 1;
        }
        try {
            CudaFeedForward::runSmokeDemo(128, 64);
            CudaFeedForward::runBackwardSmokeDemo(64, 32);
            CudaFeedForward::runBackwardSmokeDemo(256, 128);

            const bool previousAmp = CudaAmp::preferMixedPrecision;
            const bool previousInt8 = CudaAdam::preferInt8Moments;
            const bool previousCpu = CudaAdam::preferCpuOffload;
            CudaAmp::preferMixedPrecision = true;
            CudaAmp::useLossScaling = true;
            CudaAmp::resetLossScaler();
            CudaAdam::preferCpuOffload = false;
            CudaAdam::preferInt8Moments = true;

            const int vocab = 4000;
            const int seq = 256;
            LanguageModel host(vocab, embeddingDim, maximumPositionCount, Adam(0.001f), blockCount, headCount);
            CudaLanguageModel device = CudaLanguageModel::createFrom(host);
            device.adam = CudaAdam(0.001f);
            device.setActivationCheckpointMode(ActivationCheckpointMode::Off);
            device.maxPackedColumnsManual = false;
            device.applyVramPackBudget();
            for (CudaTransformerBlock& block : device.blocks)
                block.attention.preferFlashAttention = true;
            device.ensureTrainState();

            const int packBatch = (std::min)(32, device.maxPackExamplesForSegment(seq));
            std::vector<LanguageModelExample> examples(static_cast<size_t>(packBatch));
            unsigned rng = 91u;
            for (int exampleIndex = 0; exampleIndex < packBatch; ++exampleIndex) {
                examples[static_cast<size_t>(exampleIndex)].inputTokenIds.resize(static_cast<size_t>(seq));
                examples[static_cast<size_t>(exampleIndex)].targetTokenIds.resize(static_cast<size_t>(seq));
                for (size_t index = 0; index < static_cast<size_t>(seq); ++index) {
                    rng = rng * 1664525u + 1013904223u;
                    examples[static_cast<size_t>(exampleIndex)].inputTokenIds[index] = static_cast<int>(rng % static_cast<unsigned>(vocab));
                    rng = rng * 1664525u + 1013904223u;
                    examples[static_cast<size_t>(exampleIndex)].targetTokenIds[index] = static_cast<int>(rng % static_cast<unsigned>(vocab));
                }
            }
            std::vector<const LanguageModelExample*> packPointers(static_cast<size_t>(packBatch));
            for (int exampleIndex = 0; exampleIndex < packBatch; ++exampleIndex)
                packPointers[static_cast<size_t>(exampleIndex)] = &examples[static_cast<size_t>(exampleIndex)];

            for (int step = 0; step < 2; ++step) {
                device.trainGradients.zeroInPlace();
                device.accumulatePackedExamples(packPointers.data(), packBatch, device.trainGradients);
                device.applyGradients(device.trainGradients, 1.0f / static_cast<float>(packBatch));
            }
            if (cudaDeviceSynchronize() != cudaSuccess)
                throw std::runtime_error("ffn bwd check warmup sync failed");

            const auto start = std::chrono::steady_clock::now();
            const int timedSteps = 6;
            for (int step = 0; step < timedSteps; ++step) {
                device.trainGradients.zeroInPlace();
                device.accumulatePackedExamples(packPointers.data(), packBatch, device.trainGradients);
                device.applyGradients(device.trainGradients, 1.0f / static_cast<float>(packBatch));
            }
            if (cudaDeviceSynchronize() != cudaSuccess)
                throw std::runtime_error("ffn bwd check timed sync failed");
            const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            const double tokensPerSecond = seconds > 0.0
                ? static_cast<double>(seq * packBatch) * static_cast<double>(timedSteps) / seconds
                : 0.0;

            size_t freeAfter = 0;
            size_t totalBytes = 0;
            cudaMemGetInfo(&freeAfter, &totalBytes);
            SmokeLog::result("ffn bwd throughput", "pack=%d tokens/s=%.0f freeMiB=%.0f",
                packBatch, tokensPerSecond, static_cast<double>(freeAfter) / (1024.0 * 1024.0));

            CudaAmp::preferMixedPrecision = previousAmp;
            CudaAdam::preferInt8Moments = previousInt8;
            CudaAdam::preferCpuOffload = previousCpu;
        } catch (const std::exception& ex) {
            SmokeLog::result("ffn bwd fusion check", "FAILED: %s", ex.what());
            return 1;
        }
        return 0;
    }

    if (runQkvCheck) {
        SmokeLog::section("qkv fusion check");
        if (!CudaMatmul::isAvailable()) {
            SmokeLog::skip("qkv fusion check (no CUDA)");
            return 1;
        }
        try {
            CudaCausalSelfAttention::runSmokeDemo(64, 4, 32, 64);
            CudaCausalSelfAttention::runBackwardSmokeDemo(32, 2, 16, 32);
            CudaCausalSelfAttention::runFlashParitySmokeDemo(64, 4, 48, 64);

            const bool previousAmp = CudaAmp::preferMixedPrecision;
            const bool previousInt8 = CudaAdam::preferInt8Moments;
            const bool previousCpu = CudaAdam::preferCpuOffload;
            CudaAmp::preferMixedPrecision = true;
            CudaAmp::useLossScaling = true;
            CudaAmp::resetLossScaler();
            CudaAdam::preferCpuOffload = false;
            CudaAdam::preferInt8Moments = true;

            const int vocab = 4000;
            const int seq = 256;
            LanguageModel host(vocab, embeddingDim, maximumPositionCount, Adam(0.001f), blockCount, headCount);
            CudaLanguageModel device = CudaLanguageModel::createFrom(host);
            device.adam = CudaAdam(0.001f);
            device.setActivationCheckpointMode(ActivationCheckpointMode::Off);
            device.maxPackedColumnsManual = false;
            device.applyVramPackBudget();
            for (CudaTransformerBlock& block : device.blocks)
                block.attention.preferFlashAttention = true;
            device.ensureTrainState();

            const int packBatch = (std::min)(32, device.maxPackExamplesForSegment(seq));
            std::vector<LanguageModelExample> examples(static_cast<size_t>(packBatch));
            unsigned rng = 91u;
            for (int exampleIndex = 0; exampleIndex < packBatch; ++exampleIndex) {
                examples[static_cast<size_t>(exampleIndex)].inputTokenIds.resize(static_cast<size_t>(seq));
                examples[static_cast<size_t>(exampleIndex)].targetTokenIds.resize(static_cast<size_t>(seq));
                for (size_t index = 0; index < static_cast<size_t>(seq); ++index) {
                    rng = rng * 1664525u + 1013904223u;
                    examples[static_cast<size_t>(exampleIndex)].inputTokenIds[index] = static_cast<int>(rng % static_cast<unsigned>(vocab));
                    rng = rng * 1664525u + 1013904223u;
                    examples[static_cast<size_t>(exampleIndex)].targetTokenIds[index] = static_cast<int>(rng % static_cast<unsigned>(vocab));
                }
            }
            std::vector<const LanguageModelExample*> packPointers(static_cast<size_t>(packBatch));
            for (int exampleIndex = 0; exampleIndex < packBatch; ++exampleIndex)
                packPointers[static_cast<size_t>(exampleIndex)] = &examples[static_cast<size_t>(exampleIndex)];

            for (int step = 0; step < 2; ++step) {
                device.trainGradients.zeroInPlace();
                device.accumulatePackedExamples(packPointers.data(), packBatch, device.trainGradients);
                device.applyGradients(device.trainGradients, 1.0f / static_cast<float>(packBatch));
            }
            if (cudaDeviceSynchronize() != cudaSuccess)
                throw std::runtime_error("qkv check warmup sync failed");

            const auto start = std::chrono::steady_clock::now();
            const int timedSteps = 6;
            for (int step = 0; step < timedSteps; ++step) {
                device.trainGradients.zeroInPlace();
                device.accumulatePackedExamples(packPointers.data(), packBatch, device.trainGradients);
                device.applyGradients(device.trainGradients, 1.0f / static_cast<float>(packBatch));
            }
            if (cudaDeviceSynchronize() != cudaSuccess)
                throw std::runtime_error("qkv check timed sync failed");
            const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            const double tokensPerSecond = seconds > 0.0
                ? static_cast<double>(seq * packBatch) * static_cast<double>(timedSteps) / seconds
                : 0.0;

            size_t freeAfter = 0;
            size_t totalBytes = 0;
            cudaMemGetInfo(&freeAfter, &totalBytes);
            SmokeLog::result("qkv throughput", "pack=%d tokens/s=%.0f freeMiB=%.0f",
                packBatch, tokensPerSecond, static_cast<double>(freeAfter) / (1024.0 * 1024.0));

            CudaAmp::preferMixedPrecision = previousAmp;
            CudaAdam::preferInt8Moments = previousInt8;
            CudaAdam::preferCpuOffload = previousCpu;
        } catch (const std::exception& ex) {
            SmokeLog::result("qkv fusion check", "FAILED: %s", ex.what());
            return 1;
        }
        return 0;
    }

    if (runPackBudgetBench) {
        SmokeLog::section("pack budget bench");
        if (!CudaMatmul::isAvailable()) {
            SmokeLog::skip("pack budget bench (no CUDA)");
            return 1;
        }

        const int vocab = 4000;
        const int seq = 256;
        const int warmupSteps = 2;
        const int timedSteps = 6;

        LanguageModel host(vocab, embeddingDim, maximumPositionCount, Adam(0.001f), blockCount, headCount);

        try {
            const bool previousAmp = CudaAmp::preferMixedPrecision;
            const bool previousInt8 = CudaAdam::preferInt8Moments;
            const bool previousCpu = CudaAdam::preferCpuOffload;
            CudaAmp::preferMixedPrecision = true;
            CudaAmp::useLossScaling = true;
            CudaAmp::resetLossScaler();
            CudaAdam::preferCpuOffload = false;
            CudaAdam::preferInt8Moments = true;

            size_t freeBefore = 0;
            size_t totalBytes = 0;
            if (cudaMemGetInfo(&freeBefore, &totalBytes) != cudaSuccess)
                throw std::runtime_error("pack budget bench memGetInfo before failed");

            CudaLanguageModel device = CudaLanguageModel::createFrom(host);
            device.adam = CudaAdam(0.001f);
            device.setActivationCheckpointMode(ActivationCheckpointMode::Off);
            device.maxPackedColumnsManual = false;
            const size_t pendingStatic = device.estimatePendingTrainStaticBytes();
            device.applyVramPackBudget();
            for (CudaTransformerBlock& block : device.blocks)
                block.attention.preferFlashAttention = true;
            device.ensureTrainState();

            const int autoPack = device.maxPackExamplesForSegment(seq);
            SmokeLog::result("pack budget", "maxPackCols=%d autoPack@seq%d=%d pendingStaticMiB=%.0f",
                device.maxPackedColumns, seq, autoPack,
                static_cast<double>(pendingStatic) / (1024.0 * 1024.0));

            const int packCandidates[] = {
                (std::min)(28, autoPack),
                (std::min)(32, autoPack),
                (std::min)(36, autoPack),
                autoPack
            };

            for (int packBatch : packCandidates) {
                if (packBatch <= 0) continue;
                std::vector<LanguageModelExample> examples(static_cast<size_t>(packBatch));
                unsigned rng = 91u;
                for (int exampleIndex = 0; exampleIndex < packBatch; ++exampleIndex) {
                    examples[static_cast<size_t>(exampleIndex)].inputTokenIds.resize(static_cast<size_t>(seq));
                    examples[static_cast<size_t>(exampleIndex)].targetTokenIds.resize(static_cast<size_t>(seq));
                    for (size_t index = 0; index < static_cast<size_t>(seq); ++index) {
                        rng = rng * 1664525u + 1013904223u;
                        examples[static_cast<size_t>(exampleIndex)].inputTokenIds[index] = static_cast<int>(rng % static_cast<unsigned>(vocab));
                        rng = rng * 1664525u + 1013904223u;
                        examples[static_cast<size_t>(exampleIndex)].targetTokenIds[index] = static_cast<int>(rng % static_cast<unsigned>(vocab));
                    }
                }
                std::vector<const LanguageModelExample*> packPointers(static_cast<size_t>(packBatch));
                for (int exampleIndex = 0; exampleIndex < packBatch; ++exampleIndex)
                    packPointers[static_cast<size_t>(exampleIndex)] = &examples[static_cast<size_t>(exampleIndex)];
                const int tokensPerStep = seq * packBatch;

                for (int step = 0; step < warmupSteps; ++step) {
                    device.trainGradients.zeroInPlace();
                    device.accumulatePackedExamples(packPointers.data(), packBatch, device.trainGradients);
                    device.applyGradients(device.trainGradients, 1.0f / static_cast<float>(packBatch));
                }
                if (cudaDeviceSynchronize() != cudaSuccess)
                    throw std::runtime_error("pack budget warmup sync failed");

                const auto start = std::chrono::steady_clock::now();
                for (int step = 0; step < timedSteps; ++step) {
                    device.trainGradients.zeroInPlace();
                    device.accumulatePackedExamples(packPointers.data(), packBatch, device.trainGradients);
                    device.applyGradients(device.trainGradients, 1.0f / static_cast<float>(packBatch));
                }
                if (cudaDeviceSynchronize() != cudaSuccess)
                    throw std::runtime_error("pack budget timed sync failed");
                const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
                const double tokensPerSecond = seconds > 0.0
                    ? static_cast<double>(tokensPerStep) * static_cast<double>(timedSteps) / seconds
                    : 0.0;

                size_t freeAfter = 0;
                if (cudaMemGetInfo(&freeAfter, &totalBytes) != cudaSuccess)
                    throw std::runtime_error("pack budget memGetInfo after failed");

                SmokeLog::result(
                    "pack",
                    "examples=%d cols=%d tokens/s=%.0f freeMiB=%.0f %s",
                    packBatch, packBatch * seq, tokensPerSecond,
                    static_cast<double>(freeAfter) / (1024.0 * 1024.0),
                    freeAfter < (512ull << 20) ? "WARN_LOW_FREE" : "ok");
            }

            CudaAmp::preferMixedPrecision = previousAmp;
            CudaAdam::preferInt8Moments = previousInt8;
            CudaAdam::preferCpuOffload = previousCpu;
        } catch (const std::exception& ex) {
            SmokeLog::result("pack budget", "FAILED: %s", ex.what());
            return 1;
        }
        return 0;
    }

    if (runScaleProfile) {
        SmokeLog::section("scale profile");
        if (!CudaMatmul::isAvailable()) {
            SmokeLog::skip("scale profile (no CUDA)");
            return 1;
        }
        try {
            // vocab=4000 embed=768 seq=256 blocks=8 heads=12 pack=32 ckpt=off flash=on chunked CE
            CudaLanguageModel::runTrainProfileDemo(4000, 768, 256, 8, 12, true, 0, 32, false);
        } catch (const std::exception& ex) {
            SmokeLog::result("scale profile", "FAILED: %s", ex.what());
            return 1;
        }
        return 0;
    }

    if (runSpeedBench) {
        SmokeLog::section("speed bench");
        if (!CudaMatmul::isAvailable()) {
            SmokeLog::skip("speed bench (no CUDA)");
            return 1;
        }

        const int vocab = 4000;
        const int seq = 256;
        const int warmupSteps = 2;
        const int timedSteps = 8;

        LanguageModel host(vocab, embeddingDim, maximumPositionCount, Adam(0.001f), blockCount, headCount);

        auto timeMode = [&](bool cpuOffload, bool checkpointing, const char* label) {
            const bool previousAmp = CudaAmp::preferMixedPrecision;
            const bool previousInt8 = CudaAdam::preferInt8Moments;
            const bool previousCpu = CudaAdam::preferCpuOffload;
            CudaAmp::preferMixedPrecision = true;
            CudaAmp::useLossScaling = embeddingDim >= 256;
            CudaAmp::resetLossScaler();
            CudaAdam::preferCpuOffload = cpuOffload;
            CudaAdam::preferInt8Moments = !cpuOffload;

            size_t freeBefore = 0;
            size_t totalBytes = 0;
            if (cudaMemGetInfo(&freeBefore, &totalBytes) != cudaSuccess)
                throw std::runtime_error("speed bench memGetInfo before failed");

            CudaLanguageModel device = CudaLanguageModel::createFrom(host);
            device.adam = CudaAdam(0.001f);
            device.setActivationCheckpointMode(
                checkpointing ? ActivationCheckpointMode::Selective : ActivationCheckpointMode::Off);
            device.maxPackedColumnsManual = false;
            device.applyVramPackBudget();
            for (CudaTransformerBlock& block : device.blocks)
                block.attention.preferFlashAttention = true;
            device.ensureTrainState();

            const int packBatch = (std::max)(1, (std::min)(32, device.maxPackedColumns / seq));
            std::vector<LanguageModelExample> examples(static_cast<size_t>(packBatch));
            unsigned rng = 91u;
            for (int exampleIndex = 0; exampleIndex < packBatch; ++exampleIndex) {
                examples[static_cast<size_t>(exampleIndex)].inputTokenIds.resize(static_cast<size_t>(seq));
                examples[static_cast<size_t>(exampleIndex)].targetTokenIds.resize(static_cast<size_t>(seq));
                for (size_t index = 0; index < static_cast<size_t>(seq); ++index) {
                    rng = rng * 1664525u + 1013904223u;
                    examples[static_cast<size_t>(exampleIndex)].inputTokenIds[index] = static_cast<int>(rng % static_cast<unsigned>(vocab));
                    rng = rng * 1664525u + 1013904223u;
                    examples[static_cast<size_t>(exampleIndex)].targetTokenIds[index] = static_cast<int>(rng % static_cast<unsigned>(vocab));
                }
            }
            std::vector<const LanguageModelExample*> packPointers(static_cast<size_t>(packBatch));
            for (int exampleIndex = 0; exampleIndex < packBatch; ++exampleIndex)
                packPointers[static_cast<size_t>(exampleIndex)] = &examples[static_cast<size_t>(exampleIndex)];
            const int tokensPerStep = seq * packBatch;

            size_t freeAfterSetup = 0;
            if (cudaMemGetInfo(&freeAfterSetup, &totalBytes) != cudaSuccess)
                throw std::runtime_error("speed bench memGetInfo after setup failed");

            for (int step = 0; step < warmupSteps; ++step) {
                device.trainGradients.zeroInPlace();
                device.accumulatePackedExamples(packPointers.data(), packBatch, device.trainGradients);
                device.applyGradients(device.trainGradients, 1.0f / static_cast<float>(packBatch));
            }
            if (cudaDeviceSynchronize() != cudaSuccess)
                throw std::runtime_error("speed bench warmup sync failed");

            const auto start = std::chrono::steady_clock::now();
            for (int step = 0; step < timedSteps; ++step) {
                device.trainGradients.zeroInPlace();
                device.accumulatePackedExamples(packPointers.data(), packBatch, device.trainGradients);
                device.applyGradients(device.trainGradients, 1.0f / static_cast<float>(packBatch));
            }
            if (cudaDeviceSynchronize() != cudaSuccess)
                throw std::runtime_error("speed bench timed sync failed");
            const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            const double tokensPerSecond = seconds > 0.0
                ? static_cast<double>(tokensPerStep) * static_cast<double>(timedSteps) / seconds
                : 0.0;

            size_t freeAfterTrain = 0;
            if (cudaMemGetInfo(&freeAfterTrain, &totalBytes) != cudaSuccess)
                throw std::runtime_error("speed bench memGetInfo after train failed");

            const double setupUsedMiB = static_cast<double>(freeBefore - freeAfterSetup) / (1024.0 * 1024.0);
            const double trainUsedMiB = static_cast<double>(freeBefore - freeAfterTrain) / (1024.0 * 1024.0);
            SmokeLog::result(
                "speed",
                "%s  embed=%d blocks=%d seq=%d pack=%d maxPackCols=%d ckpt=%s  tokens/s=%.0f  setupMiB=%.0f trainMiB=%.0f free=%.0f",
                label, embeddingDim, blockCount, seq, packBatch, device.maxPackedColumns,
                checkpointing ? "on" : "off",
                tokensPerSecond, setupUsedMiB, trainUsedMiB,
                static_cast<double>(freeAfterTrain) / (1024.0 * 1024.0));

            CudaAmp::preferMixedPrecision = previousAmp;
            CudaAdam::preferInt8Moments = previousInt8;
            CudaAdam::preferCpuOffload = previousCpu;
        };

        try {
            timeMode(true, true, "cpuAdam+ckpt");
            timeMode(false, true, "int8+ckpt");
            timeMode(false, false, "int8+noCkpt");
        } catch (const std::exception& ex) {
            SmokeLog::result("speed", "FAILED: %s", ex.what());
            return 1;
        }
        return 0;
    }

    if (runSmokes) {
        SmokeLog::section("runtime");
#if defined(_OPENMP)
        SmokeLog::result("OpenMP", "threads=%d", omp_get_max_threads());
#else
        SmokeLog::note("OpenMP disabled");
#endif

        SmokeLog::section("gemm");
        CudaMatmul::runSmokeDemo(512);

        SmokeLog::section("layers");
        CudaFeedForward::runSmokeDemo(128, 64);
        CudaFeedForward::runBackwardSmokeDemo(64, 32);
        CudaRMSNorm::runSmokeDemo(128, 64);
        CudaRMSNorm::runBackwardSmokeDemo(64, 32);
        CudaAdam::runSmokeDemo(128, 64);

        SmokeLog::section("attention");
        CausalSelfAttention::runSparseMaskSmokeDemo(32, 2, 16, 32, 4, 2);
        CausalSelfAttention::runSparseBackwardSmokeDemo(32, 2, 12, 32, 4, 2);
        CausalSelfAttention::runSparseComputeSmokeDemo(32, 2, 16, 32, 4, 2);
        CudaCausalSelfAttention::runSmokeDemo(64, 4, 32, 64);
        CudaCausalSelfAttention::runBackwardSmokeDemo(32, 2, 16, 32);
        CudaCausalSelfAttention::runFlashParitySmokeDemo(64, 4, 48, 64);
        CudaCausalSelfAttention::runKvCacheSmokeDemo(64, 4, 32, 64);
        CudaCausalSelfAttention::runSparseSmokeDemo(32, 2, 16, 32, 4, 2);

        SmokeLog::section("model");
        CudaTransformerBlock::runSmokeDemo(64, 4, 32, 64);
        CudaLanguageModel::runSmokeDemo(128, 64, 32, 2, 4);
        CudaLanguageModel::runKvCacheSmokeDemo(128, 64, 32, 2, 4);
        LanguageModel::runCheckpointSmokeDemo();
        LanguageModel::runStreamingSmokeDemo();
        CudaLanguageModel::runTrainSmokeDemo(64, 32, 16, 1, 2);
        CudaLanguageModel::runTrainSmokeDemo(1000, 64, 48, 2, 4);
        CudaLanguageModel::runTrainInt8AdamSmokeDemo(1000, 64, 48, 2, 4);
        CudaLanguageModel::runTrainCpuAdamOffloadSmokeDemo(2000, 128, 64, 2, 4);
        CudaLanguageModel::runTrainProfileDemo(1000, 64, 48, 2, 4, true, 256);
        CudaLanguageModel::runTrainProfileDemo(1000, 64, 48, 2, 4, false, 256);
        CudaLanguageModel::runTrainProfileDemo(1000, 64, 256, 2, 4, true, 2048);
        CudaLanguageModel::runTrainProfileDemo(1000, 64, 256, 2, 4, false, 2048);
        CudaLanguageModel::runTrainProfileDemo(1000, 64, 512, 2, 4, true, 2048);
        CudaLanguageModel::runTrainProfileDemo(1000, 64, 512, 2, 4, false, 2048);
        CudaLanguageModel::runConsumerVramDemo();
    }

    SmokeLog::section("sera train");
    SmokeLog::note(useArrowCorpus
        ? "corpus=Arrow IPC (HF save_to_disk)"
        : "corpus=JSONL");

    LanguageModelChunkSource source(samplePath, maximumTextCharacters, maximumTokenCount, chunkExampleCount, trainRatio, 42u, testReservoirCap);
    std::vector<std::string> tokenizerSample = source.prepareTokenizerSample(tokenizerSampleRows);
    if (tokenizerSample.empty()) {
        SmokeLog::note(("no usable rows from " + samplePath).c_str());
        return 1;
    }

    BPETokenizer tokenizer;
    tokenizer.train(tokenizerSample, tokenizerVocabSize);
    source.setTokenizer(&tokenizer);
    source.materialize();

    if (source.trainExampleCount() <= 0) {
        SmokeLog::note("no language-model examples (need sequences with >= 2 tokens)");
        return 1;
    }

    LanguageModelDataset promptChunk;
    source.rewindTrain();
    const bool moreAfterPrompt = source.nextTrainChunk(promptChunk);
    source.rewindTrain();
    if (promptChunk.examples.empty()) {
        SmokeLog::note("no prompt example from stream");
        return 1;
    }

    SmokeLog::result("data", "stream=%s  train=%d  test=%d  vocab=%d  positions=%d  chunk=%d  more=%s",
        samplePath.c_str(),
        source.trainExampleCount(),
        source.testDataset().size(),
        tokenizer.vocabSize(),
        source.trainPredictionCount(),
        source.chunkExampleCount(),
        moreAfterPrompt ? "yes" : "no");

    LanguageModel model(tokenizer.vocabSize(), embeddingDim, maximumPositionCount, Adam(0.001f), blockCount, headCount);

    const std::vector<int> parityTokenIds = promptChunk.examples[0].inputTokenIds;
    Matrix cpuLogits = model.forward(parityTokenIds);

    model.enableCuda();
    model.setCudaPreferCpuAdamOffload(preferCpuAdamOffload);
    model.enableCudaTrain();
    model.enableActivationCheckpointing(useCheckpointing);
    model.setCudaPreferFlashAttention(true);
    SmokeLog::result("model", "blocks=%zu  heads=%d  embed=%d  cuda=%s  train=%s  maxTok=%zu maxPackCols=%d flash=on stream=on batch=%d accum=%d cpuAdam=%s ckpt=%s",
        model.blocks.size(),
        model.blocks[0].attention.headCount,
        embeddingDim,
        model.cudaEnabled() ? "on" : "off",
        model.cudaTrainEnabled() ? "cuda" : "cpu-openmp",
        maximumTokenCount,
        model.cudaMaxPackedColumns(),
        trainBatchSize,
        trainGradAccum,
        preferCpuAdamOffload ? "on" : "off",
        useCheckpointing ? "on" : "off");

    if (model.cudaEnabled()) {
        Matrix deviceLogits = model.forward(parityTokenIds);
        float maximumDifference = 0.0f;
        for (size_t index = 0; index < cpuLogits.data.size(); ++index)
            maximumDifference = (std::max)(maximumDifference, std::fabs(cpuLogits.data[index] - deviceLogits.data[index]));
        SmokeLog::result("framework parity", "diff=%.2e", maximumDifference);
    }

    model.train(source, trainEpochs, 1, trainBatchSize, trainGradAccum);
    model.saveCheckpoint("sera_demo.snlm", true);

    SmokeLog::result("final", "trainLoss=n/a (streamed)");
    if (!source.testDataset().examples.empty())
        SmokeLog::result("final", "testLoss=%.6f", model.averageLoss(source.testDataset()));
    SmokeLog::result("checkpoint", "saved sera_demo.snlm (weights+adam)");

    const std::vector<int> prompt = promptChunk.examples[0].inputTokenIds;
    const std::vector<int> greedy = model.generate(prompt, 32, 0.0f, 0, 7u);
    const std::vector<int> sampled = model.generate(prompt, 32, 0.9f, 40, 7u);

    SmokeLog::section("generate");
    std::cout << "  prompt:  " << tokenizer.decode(prompt) << '\n';
    std::cout << "  greedy:  " << tokenizer.decode(greedy) << '\n';
    std::cout << "  sample:  " << tokenizer.decode(sampled) << '\n';

    return 0;
}
