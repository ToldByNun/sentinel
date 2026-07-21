#include "NeuralNet/Tokenizer/BPETokenizer.hpp"
#include "NeuralNet/Layers/Embedding.hpp"
#include "NeuralNet/Layers/Dense.hpp"
#include "NeuralNet/Network/Sequential.hpp"
#include "NeuralNet/Optimizers/SGD.hpp"
#include "NeuralNet/Math/Matrix.hpp"

#include <iostream>
#include <string>
#include <vector>

int main() {
    std::vector<std::string> corpus = {
        // Test corpus
        "#include <iostream>\n#include <vector>\n#include <string>\n#include <memory>",
        "template<typename T>\nclass Tensor {\npublic:\n    std::vector<T> data;\n    size_t rows, cols;\n};",
        "int main() {\n    auto ptr = std::make_unique<int>(42);\n    if (ptr != nullptr) {\n        std::cout << \"Value: \" << *ptr << std::endl;\n    }\n    return 0;\n}",
        "for (size_t i = 0; i < vec.size(); ++i) { vec[i] = vec[i] * 2.0f; }",

        "{\"instruction\": \"Fix memory leak in C++ vector\", \"output\": \"Use std::vector::clear() and std::move()\"}",
        "{\"role\": \"system\", \"content\": \"You are an expert C++ and AI engine software architect.\"}",
        "{\"role\": \"user\", \"content\": \"Write a fast matrix multiplication algorithm using raw pointers.\"}",

        "import json\nimport datasets\nfrom transformers import AutoTokenizer",
        "def load_dataset(file_path):\n    with open(file_path, 'r', encoding='utf-8') as f:\n        return [json.loads(line) for line in f]",

        "Segmentation fault (core dumped) in function forwardPassAtLayer(int index)",
        "Optimization flag: -O3 -march=native -std=c++20",
        "Backpropagation gradient check failed: expected dW1 to match numerical derivative."
    };

    BPETokenizer tokenizer;
    tokenizer.train(corpus, 150);

    std::cout << "corpus size: " << corpus.size() << '\n';
    std::cout << "vocab size: " << tokenizer.vocabSize() << '\n';

    const int embeddingDim = 8;
    Embedding embedding(tokenizer.vocabSize(), embeddingDim);

    std::vector<int> ids = tokenizer.encode("main vector");
    Matrix x = embedding.forward(ids);

    std::cout << "token count: " << ids.size() << '\n';
    std::cout << "embedding shape: " << x.data.size() << " x " << (x.data.empty() ? 0 : x.data[0].size()) << '\n';

    Matrix input({ {1.0f}, {2.0f}, {0.5f} });
    Matrix target({ {2.0f}, {4.0f}, {1.0f} });

    Matrix weight1({
        {0.1f, 0.2f, 0.1f},
        {0.1f, 0.1f, 0.2f},
        {0.2f, 0.1f, 0.1f},
        {0.1f, 0.1f, 0.1f}
    });
    Matrix bias1({ {0.0f}, {0.0f}, {0.0f}, {0.0f} });

    Matrix weight2({
        {0.1f, 0.1f, 0.1f, 0.1f},
        {0.1f, 0.2f, 0.1f, 0.1f},
        {0.1f, 0.1f, 0.2f, 0.1f}
    });
    Matrix bias2({ {0.0f}, {0.0f}, {0.0f} });

    Sequential model(
        Dense(std::move(weight1), std::move(bias1)),
        Dense(std::move(weight2), std::move(bias2)),
        SGD(0.01f)
    );

    model.train(input, target, 10000);

    return 0;
}
