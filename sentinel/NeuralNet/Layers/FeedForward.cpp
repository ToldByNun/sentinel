#include "FeedForward.hpp"

#include "../Activations/ReLU.hpp"
#include "../Initializers/UniformInit.hpp"

#include <stdexcept>
#include <utility>
#include <vector>

FeedForward::FeedForward(Matrix firstWeight, Matrix firstBias, Matrix secondWeight, Matrix secondBias)
    : firstWeight(std::move(firstWeight)),
      firstBias(std::move(firstBias)),
      secondWeight(std::move(secondWeight)),
      secondBias(std::move(secondBias)) {}

FeedForward FeedForward::create(int embeddingDim, int expandRatio, unsigned seed) {
    if (embeddingDim <= 0) throw std::invalid_argument("FeedForward::create embeddingDim must be > 0");
    if (expandRatio <= 0) throw std::invalid_argument("FeedForward::create expandRatio must be > 0");

    const int hiddenDim = embeddingDim * expandRatio;
    return FeedForward(
        UniformInit::matrix(hiddenDim, embeddingDim, 0.1f, seed),
        UniformInit::matrix(hiddenDim, 1, 0.01f, seed + 1u),
        UniformInit::matrix(embeddingDim, hiddenDim, 0.1f, seed + 2u),
        UniformInit::matrix(embeddingDim, 1, 0.01f, seed + 3u)
    );
}

Matrix FeedForward::broadcastBiasAdd(const Matrix& product, const Matrix& bias) {
    Matrix result = product;
    for (size_t row = 0; row < result.data.size(); ++row) {
        const float biasValue = bias.data[row][0];
        for (size_t column = 0; column < result.data[row].size(); ++column)
            result.data[row][column] += biasValue;
    }
    return result;
}

Matrix FeedForward::sumColumns(const Matrix& gradient) {
    Matrix biasGradient;
    biasGradient.data = std::vector<std::vector<float>>(gradient.data.size(), std::vector<float>(1, 0.0f));
    for (size_t row = 0; row < gradient.data.size(); ++row) {
        float total = 0.0f;
        for (size_t column = 0; column < gradient.data[row].size(); ++column)
            total += gradient.data[row][column];
        biasGradient.data[row][0] = total;
    }
    return biasGradient;
}

Matrix FeedForward::forward(const Matrix& input, FeedForwardCache& cache) const {
    if (input.data.empty() || input.data[0].empty()) throw std::invalid_argument("FeedForward::forward empty input");
    if (this->firstWeight.data[0].size() != input.data.size())
        throw std::invalid_argument("FeedForward::forward embedding dim mismatch");

    cache.input = input;
    cache.hiddenPreActivation = FeedForward::broadcastBiasAdd(
        Matrix::multiply(this->firstWeight, input),
        this->firstBias
    );
    cache.hiddenActivated = ReLU::apply(cache.hiddenPreActivation);
    return FeedForward::broadcastBiasAdd(
        Matrix::multiply(this->secondWeight, cache.hiddenActivated),
        this->secondBias
    );
}

Matrix FeedForward::backward(const Matrix& outputGradient, const FeedForwardCache& cache, Matrix& firstWeightGradient, Matrix& firstBiasGradient, Matrix& secondWeightGradient, Matrix& secondBiasGradient) const {
    if (cache.input.data.empty()) throw std::logic_error("FeedForward::backward called before forward");
    if (outputGradient.data.size() != this->secondWeight.data.size() || outputGradient.data[0].size() != cache.input.data[0].size())
        throw std::invalid_argument("FeedForward::backward shape mismatch");

    secondWeightGradient = Matrix::multiply(outputGradient, Matrix::transpose(cache.hiddenActivated));
    secondBiasGradient = FeedForward::sumColumns(outputGradient);

    Matrix hiddenGradient = Matrix::multiply(Matrix::transpose(this->secondWeight), outputGradient);
    hiddenGradient = Matrix::multiplyElementwise(hiddenGradient, ReLU::derivative(cache.hiddenPreActivation));

    firstWeightGradient = Matrix::multiply(hiddenGradient, Matrix::transpose(cache.input));
    firstBiasGradient = FeedForward::sumColumns(hiddenGradient);

    return Matrix::multiply(Matrix::transpose(this->firstWeight), hiddenGradient);
}
