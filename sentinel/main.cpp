#include "NeuralNet/Cuda/CudaFeedForward.hpp"
#include "NeuralNet/Cuda/CudaLanguageModel.hpp"
#include "NeuralNet/Cuda/CudaMatmul.hpp"
#include "NeuralNet/Cuda/CudaRMSNorm.hpp"
#include "NeuralNet/Cuda/CudaCausalSelfAttention.hpp"
#include "NeuralNet/Cuda/CudaAdam.hpp"
#include "NeuralNet/Cuda/CudaTransformerBlock.hpp"
#include "NeuralNet/Layers/CausalSelfAttention.hpp"
#include "NeuralNet/Tokenizer/BPETokenizer.hpp"
#include "NeuralNet/Data/DatasetSplit.hpp"
#include "NeuralNet/Data/JsonlLoader.hpp"
#include "NeuralNet/Data/LanguageModelDataset.hpp"
#include "NeuralNet/Network/LanguageModel.hpp"
#include "NeuralNet/Optimizers/Adam.hpp"
#include "NeuralNet/Utils/SmokeLog.hpp"
#include "NeuralNet/Utils/TextUtil.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
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

    const std::string samplePath = "../SERA-Data/sera_sample.jsonl";
    const size_t maximumTextCharacters = 400;
    const size_t maximumTokenCount = 64;
    const float trainRatio = 0.8f;
    const int embeddingDim = 64;
    const int maximumPositionCount = static_cast<int>(maximumTokenCount);

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
    CudaLanguageModel::runTrainSmokeDemo(64, 32, 16, 1, 2);
    CudaLanguageModel::runTrainSmokeDemo(1000, 64, 48, 2, 4);
    CudaLanguageModel::runTrainProfileDemo(1000, 64, 48, 2, 4);

    SmokeLog::section("sera train");

    // 0 = load entire jsonl (sera_sample has 10000 rows)
    std::vector<JsonlRow> rows = JsonlLoader::load(samplePath, 0);

    std::vector<std::string> texts;
    for (const JsonlRow& row : rows) {
        if (row.text.empty()) continue;
        texts.push_back(TextUtil::truncate(row.text, maximumTextCharacters));
    }

    if (texts.empty()) {
        SmokeLog::note(("no usable rows from " + samplePath).c_str());
        return 1;
    }

    DatasetSplit split = DatasetSplit::partitionTexts(texts, trainRatio, 42u);

    BPETokenizer tokenizer;
    tokenizer.train(split.trainTexts, 1000);

    LanguageModelDataset trainDataset = LanguageModelDataset::build(split.trainTexts, tokenizer, maximumTokenCount, false);
    LanguageModelDataset testDataset = LanguageModelDataset::build(split.testTexts, tokenizer, maximumTokenCount, false);

    SmokeLog::result("data", "texts=%zu  train=%d  test=%d  vocab=%d  positions=%d",
        texts.size(), trainDataset.size(), testDataset.size(), tokenizer.vocabSize(), trainDataset.totalPredictionCount());

    if (trainDataset.examples.empty()) {
        SmokeLog::note("no language-model examples (need sequences with >= 2 tokens)");
        return 1;
    }

    LanguageModel model(tokenizer.vocabSize(), embeddingDim, maximumPositionCount, Adam(0.001f), 2, 4);

    const std::vector<int>& parityTokenIds = trainDataset.examples[0].inputTokenIds;
    Matrix cpuLogits = model.forward(parityTokenIds);

    model.enableCuda();
    model.enableCudaTrain();
    SmokeLog::result("model", "blocks=%zu  heads=%d  cuda=%s  train=%s",
        model.blocks.size(),
        model.blocks[0].attention.headCount,
        model.cudaEnabled() ? "on" : "off",
        model.cudaTrainEnabled() ? "cuda" : "cpu-openmp");

    if (model.cudaEnabled()) {
        Matrix deviceLogits = model.forward(parityTokenIds);
        float maximumDifference = 0.0f;
        for (size_t index = 0; index < cpuLogits.data.size(); ++index)
            maximumDifference = (std::max)(maximumDifference, std::fabs(cpuLogits.data[index] - deviceLogits.data[index]));
        SmokeLog::result("framework parity", "diff=%.2e", maximumDifference);
    }

    // segmented pack attn + grad accum=2 (Adam every 128 examples)
    model.train(trainDataset, testDataset, 5, 1, 64, 2);

    SmokeLog::result("final", "trainLoss=%.6f", model.averageLoss(trainDataset));
    if (!testDataset.examples.empty())
        SmokeLog::result("final", "testLoss=%.6f", model.averageLoss(testDataset));

    const std::vector<int> prompt = trainDataset.examples[0].inputTokenIds;
    const std::vector<int> greedy = model.generate(prompt, 32, 0.0f, 0, 7u);
    const std::vector<int> sampled = model.generate(prompt, 32, 0.9f, 40, 7u);

    SmokeLog::section("generate");
    std::cout << "  prompt:  " << tokenizer.decode(prompt) << '\n';
    std::cout << "  greedy:  " << tokenizer.decode(greedy) << '\n';
    std::cout << "  sample:  " << tokenizer.decode(sampled) << '\n';

    return 0;
}
