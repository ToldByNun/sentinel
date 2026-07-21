#include "NeuralNet/Tokenizer/BPETokenizer.hpp"
#include "NeuralNet/Layers/Embedding.hpp"
#include "NeuralNet/Layers/Dense.hpp"
#include "NeuralNet/Layers/MeanPool.hpp"
#include "NeuralNet/Network/Sequential.hpp"
#include "NeuralNet/Optimizers/SGD.hpp"
#include "NeuralNet/Math/Matrix.hpp"
#include "NeuralNet/Data/ClassificationDataset.hpp"

#include <iostream>
#include <string>
#include <vector>

/// <summary>
/// Demo: classify corpus snippets as Cpp / Json / Python
/// Flow: BPE train -> build dataset -> train Embedding+MLP -> print predictions
/// </summary>
int main() {
    std::vector<std::string> corpus = {
        // C++
        "#include <iostream>\n#include <vector>\n#include <string>\n#include <memory>",
        "template<typename T>\nclass Tensor {\npublic:\n    std::vector<T> data;\n    size_t rows, cols;\n};",
        "int main() {\n    auto ptr = std::make_unique<int>(42);\n    if (ptr != nullptr) {\n        std::cout << \"Value: \" << *ptr << std::endl;\n    }\n    return 0;\n}",
        "for (size_t i = 0; i < vec.size(); ++i) { vec[i] = vec[i] * 2.0f; }",

        // JSON
        "{\"instruction\": \"Fix memory leak in C++ vector\", \"output\": \"Use std::vector::clear() and std::move()\"}",
        "{\"role\": \"system\", \"content\": \"You are an expert C++ and AI engine software architect.\"}",
        "{\"role\": \"user\", \"content\": \"Write a fast matrix multiplication algorithm using raw pointers.\"}",

        // Python
        "import json\nimport datasets\nfrom transformers import AutoTokenizer",
        "def load_dataset(file_path):\n    with open(file_path, 'r', encoding='utf-8') as f:\n        return [json.loads(line) for line in f]",

        // heuristic maps
        "Segmentation fault (core dumped) in function forwardPassAtLayer(int index)",
        "Optimization flag: -O3 -march=native -std=c++20",
        "Backpropagation gradient check failed: expected dW1 to match numerical derivative."
    };

    const int embeddingDim = 8;
    const int hidden = 4;
    const int outDim = ClassificationDataset::ClassCount;

    // dense1: hidden x embeddingDim, Dense2: classes x hidden
    Matrix weight1(std::vector<std::vector<float>>(hidden, std::vector<float>(embeddingDim, 0.1f)));
    Matrix bias1(std::vector<std::vector<float>>(hidden, std::vector<float>(1, 0.0f)));
    Matrix weight2(std::vector<std::vector<float>>(outDim, std::vector<float>(hidden, 0.1f)));
    Matrix bias2(std::vector<std::vector<float>>(outDim, std::vector<float>(1, 0.0f)));

    BPETokenizer tokenizer;
    tokenizer.train(corpus, 150);

    ClassificationDataset dataset = ClassificationDataset::build(corpus, tokenizer);

    std::cout << "corpus size: " << corpus.size() << '\n';
    std::cout << "vocab size: " << tokenizer.vocabSize() << '\n';
    std::cout << "dataset size: " << dataset.size() << '\n';

    Embedding embedding(tokenizer.vocabSize(), embeddingDim);
    MeanPool meanPool;

    Sequential model(
        Dense(std::move(weight1), std::move(bias1)),
        Dense(std::move(weight2), std::move(bias2)),
        SGD(0.01f)
    );

    model.train(embedding, meanPool, dataset, 3000);

    const char* classNames[] = { "Cpp", "Json", "Python" };
    for (size_t index = 0; index < dataset.examples.size(); ++index) {
        const ClassificationExample& example = dataset.examples[index];
        const int predicted = model.predictClass(embedding, meanPool, example.tokenIds);

        std::cout << "example " << index
                  << " | true: " << classNames[example.label]
                  << " | pred: " << classNames[predicted]
                  << '\n';
    }

    return 0;
}
