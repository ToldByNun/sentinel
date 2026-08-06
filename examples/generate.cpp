/// Load a checkpoint + tokenizer (or train a tiny model) and generate text.
///
///   ./sentinel_generate [checkpoint.snlm] [prompt]
///
/// Expects sibling `{stem}.sbpe` when loading a checkpoint. If the checkpoint is
/// missing, trains the same toy setup as train_tiny and writes both files.

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

std::string siblingSbpe(const std::string& checkpointPath) {
    return std::filesystem::path(checkpointPath).replace_extension(".sbpe").string();
}

} // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);

    const std::string checkpointPath = argc > 1 ? argv[1] : "tiny_demo.snlm";
    const std::string promptText = argc > 2 ? argv[2] : "the cat";
    const std::string tokPath = siblingSbpe(checkpointPath);

    BPETokenizer tokenizer;
    LanguageModel model = makeTinyModel(/*vocabSize=*/256);

    if (std::filesystem::exists(checkpointPath)) {
        if (!std::filesystem::exists(tokPath)) {
            std::cerr << "generate: missing tokenizer at " << tokPath
                      << " (expected sibling .sbpe of the checkpoint)\n";
            return 1;
        }
        tokenizer.load(tokPath);
        model = makeTinyModel(tokenizer.vocabSize());
        maybeEnableCuda(model, /*train=*/false);
        model.loadCheckpoint(checkpointPath);
        std::cout << "generate: loaded " << checkpointPath << " + " << tokPath << "\n";
    } else {
        std::cout << "generate: no checkpoint at " << checkpointPath << " — training toy model first\n";
        tokenizer.train(kCorpus, 256);
        model = makeTinyModel(tokenizer.vocabSize());
        LanguageModelDataset train = LanguageModelDataset::build(kCorpus, tokenizer, 48, false);
        maybeEnableCuda(model, /*train=*/true);
        LanguageModelDataset emptyTest;
        model.train(train, emptyTest, 8, 4, 4, 1);
        model.saveCheckpoint(checkpointPath, false);
        tokenizer.save(tokPath);
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
