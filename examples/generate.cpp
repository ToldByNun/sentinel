/// Load a checkpoint (or train a tiny model) and generate text.
///
///   ./sentinel_generate [checkpoint.snlm] [prompt]
///
/// If the checkpoint is missing, trains the same toy setup as train_tiny first.

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

namespace {

const std::vector<std::string> kCorpus = {
    "the cat sat on the mat",
    "the dog ran in the park",
    "a quick brown fox jumps",
    "sentinel trains causal language models",
    "cuda packs batches for throughput",
    "tiny models learn short patterns",
};

BPETokenizer makeTokenizer() {
    BPETokenizer tokenizer;
    tokenizer.train(kCorpus, 256);
    return tokenizer;
}

LanguageModel makeTinyModel(int vocabSize) {
    return LanguageModel(vocabSize, 64, 64, Adam(3e-3f), 2, 4);
}

void maybeEnableCuda(LanguageModel& model, bool train) {
    if (!CudaMatmul::isAvailable()) return;
    model.enableCuda();
    model.setCudaPreferFlashAttention(true);
    if (train)
        model.enableCudaTrain();
}

} // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);

    const std::string checkpointPath = argc > 1 ? argv[1] : "tiny_demo.snlm";
    const std::string promptText = argc > 2 ? argv[2] : "the cat";

    BPETokenizer tokenizer = makeTokenizer();
    LanguageModel model = makeTinyModel(tokenizer.vocabSize());

    if (std::filesystem::exists(checkpointPath)) {
        maybeEnableCuda(model, /*train=*/false);
        model.loadCheckpoint(checkpointPath);
        std::cout << "generate: loaded " << checkpointPath << "\n";
    } else {
        std::cout << "generate: no checkpoint at " << checkpointPath << " — training toy model first\n";
        LanguageModelDataset train = LanguageModelDataset::build(kCorpus, tokenizer, 48, false);
        maybeEnableCuda(model, /*train=*/true);
        LanguageModelDataset emptyTest;
        model.train(train, emptyTest, 8, 4, 4, 1);
        model.saveCheckpoint(checkpointPath, false);
    }

    const std::vector<int> promptIds = tokenizer.encode(promptText);
    if (promptIds.empty()) {
        std::cerr << "generate: empty prompt encoding\n";
        return 1;
    }

    const std::vector<int> continuation = model.generate(promptIds, /*newTokenCount=*/32, /*temperature=*/0.9f, /*topK=*/20, /*seed=*/7u);
    std::vector<int> full = promptIds;
    full.insert(full.end(), continuation.begin(), continuation.end());

    std::cout << "prompt: " << promptText << "\n";
    std::cout << "output: " << tokenizer.decode(full) << "\n";
    return 0;
}
