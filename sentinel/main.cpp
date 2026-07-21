#include "NeuralNet/Tokenizer/BPETokenizer.hpp"
#include "NeuralNet/Layers/Embedding.hpp"
#include "NeuralNet/Layers/Dense.hpp"
#include "NeuralNet/Layers/MeanPool.hpp"
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

    int embeddingDim = 8;
    int hidden = 4;
    int outDim = 3;

    Matrix weight1(std::vector<std::vector<float>>(hidden, std::vector<float>(embeddingDim, 0.1f)));
    Matrix bias1(std::vector<std::vector<float>>(hidden, std::vector<float>(1.0f, 0.1f)));
    
    Matrix weight2(std::vector<std::vector<float>>(outDim, std::vector<float>(hidden, 0.1f)));
    Matrix bias2(std::vector<std::vector<float>>(outDim, std::vector<float>(1.0f, 0.1f)));

    BPETokenizer tokenizer;
    tokenizer.train(corpus, 150);

    std::cout << "corpus size: " << corpus.size() << '\n';
    std::cout << "vocab size: " << tokenizer.vocabSize() << '\n';

    Embedding embedding(tokenizer.vocabSize(), embeddingDim);

    std::vector<int> ids = tokenizer.encode("main vector");
    Matrix x = MeanPool::meanPool(embedding.forward(ids));

    Matrix target({ {1.0f}, {0.0f}, {0.0f} });

    Sequential model(
        Dense(std::move(weight1), std::move(bias1)),
        Dense(std::move(weight2), std::move(bias2)),
        SGD(0.01f)
    );
    model.train(x, target, 10000);

    return 0;
}
