#include "Math/Matrix.hpp"
#include "Layers/Dense.hpp"
#include "Network/Sequential.hpp"
#include "Optimizers/SGD.hpp"

#include <utility>

int main() {
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
