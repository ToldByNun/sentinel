/// Minimal end-to-end train example: BPE → tiny causal LM → optional CUDA → checkpoint.
///
///   ./sentinel_train_tiny [checkpoint.snlm]
///
/// No external dataset required.

#include "NeuralNet/Cuda/CudaMatmul.hpp"
#include "NeuralNet/Data/LanguageModelDataset.hpp"
#include "NeuralNet/Network/LanguageModel.hpp"
#include "NeuralNet/Optimizers/Adam.hpp"
#include "NeuralNet/Tokenizer/BPETokenizer.hpp"

#include <cstdio>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);

    const std::string checkpointPath = argc > 1 ? argv[1] : "tiny_demo.snlm";

    const std::vector<std::string> corpus = {
        "the cat sat on the mat",
        "the dog ran in the park",
        "a quick brown fox jumps",
        "sentinel trains causal language models",
        "cuda packs batches for throughput",
        "tiny models learn short patterns",
    };

    BPETokenizer tokenizer;
    const int vocabSize = 256;
    tokenizer.train(corpus, vocabSize);

    LanguageModelDataset train = LanguageModelDataset::build(corpus, tokenizer, /*maximumTokenCount=*/48, /*buildOneHot=*/false);
    if (train.examples.empty()) {
        std::cerr << "train_tiny: no examples after tokenization\n";
        return 1;
    }

    const int embed = 64;
    const int blocks = 2;
    const int heads = 4;
    const int maxPos = 64;
    LanguageModel model(tokenizer.vocabSize(), embed, maxPos, Adam(3e-3f), blocks, heads);

    if (CudaMatmul::isAvailable()) {
        model.enableCuda();
        model.setCudaPreferFlashAttention(true);
        model.enableCudaTrain();
        model.setActivationCheckpointMode(ActivationCheckpointMode::Off);
        std::cout << "train_tiny: CUDA train enabled\n";
    } else {
        std::cout << "train_tiny: CUDA unavailable — host OpenMP train\n";
    }

    LanguageModelDataset emptyTest;
    model.train(train, emptyTest, /*epochs=*/8, /*logEveryEpochs=*/2, /*batchSize=*/4, /*gradientAccumulationSteps=*/1);
    model.saveCheckpoint(checkpointPath, /*includeOptimizer=*/false);
    const std::string safePath = checkpointPath + ".safetensors";
    model.saveSafeTensors(safePath);
    const std::string tokPath = std::filesystem::path(checkpointPath).replace_extension(".sbpe").string();
    tokenizer.save(tokPath);

    LanguageModel roundtrip(tokenizer.vocabSize(), embed, maxPos, Adam(3e-3f), blocks, heads);
    if (CudaMatmul::isAvailable())
        roundtrip.enableCuda();
    roundtrip.loadCheckpoint(safePath);
    const float safeLoss = roundtrip.averageLoss(train);

    BPETokenizer tokRoundtrip = BPETokenizer::loadFrom(tokPath);
    const std::string probe = "the cat";
    if (tokRoundtrip.encode(probe) != tokenizer.encode(probe)
        || tokRoundtrip.decode(tokenizer.encode(probe)) != tokenizer.decode(tokenizer.encode(probe))) {
        std::cerr << "train_tiny: tokenizer roundtrip mismatch\n";
        return 1;
    }

    const float loss = model.averageLoss(train);
    std::cout << "train_tiny: examples=" << train.size()
              << " vocab=" << tokenizer.vocabSize()
              << " params≈" << (model.parameterElementCount() / 1.0e6) << "M"
              << " avgLoss=" << loss
              << " safeLoss=" << safeLoss
              << " wrote " << checkpointPath << " + " << safePath << " + " << tokPath << "\n";
    return 0;
}
