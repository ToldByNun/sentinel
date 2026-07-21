#ifndef SEQUENTIAL_HPP
#define SEQUENTIAL_HPP

#include "../Layers/Dense.hpp"
#include "../Layers/Embedding.hpp"
#include "../Layers/MeanPool.hpp"
#include "../Math/Matrix.hpp"
#include "../Optimizers/SGD.hpp"

#include <vector>

class Sequential {
public:
    Dense layer1;
    Dense layer2;
    SGD optimizer;

    Sequential(Dense layer1, Dense layer2, SGD optimizer);

    Matrix forward(const Matrix& input);
    void train(const Matrix& input, const Matrix& target, int epochs);

    void train(
        Embedding& embedding,
        MeanPool& meanPool,
        const std::vector<int>& tokenIds,
        const Matrix& target,
        int epochs
    );
};

#endif // SEQUENTIAL_HPP
