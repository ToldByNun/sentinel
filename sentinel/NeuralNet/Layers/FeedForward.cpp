#include "FeedForward.hpp"

#include "../Activations/ReLU.hpp"
#include "../Initializers/UniformInit.hpp"

#include <stdexcept>
#include <utility>

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
    for (size_t row = 0; row < result.rows; ++row) {
        const float biasValue = bias.at(row, 0);
        for (size_t column = 0; column < result.cols; ++column)
            result.at(row, column) += biasValue;
    }
    return result;
}

Matrix FeedForward::sumColumns(const Matrix& gradient) {
    Matrix biasGradient(gradient.rows, 1, 0.0f);
    for (size_t row = 0; row < gradient.rows; ++row) {
        float total = 0.0f;
        for (size_t column = 0; column < gradient.cols; ++column)
            total += gradient.at(row, column);
        biasGradient.at(row, 0) = total;
    }
    return biasGradient;
}

Matrix FeedForward::forward(const Matrix& input, FeedForwardCache& cache) const {
    if (input.empty()) throw std::invalid_argument("FeedForward::forward empty input");
    if (this->firstWeight.cols != input.rows)
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
    if (cache.input.empty()) throw std::logic_error("FeedForward::backward called before forward");
    if (outputGradient.rows != this->secondWeight.rows || outputGradient.cols != cache.input.cols)
        throw std::invalid_argument("FeedForward::backward shape mismatch");

    secondWeightGradient = Matrix::multiply(outputGradient, cache.hiddenActivated, false, true);
    secondBiasGradient = FeedForward::sumColumns(outputGradient);

    Matrix hiddenGradient = Matrix::multiply(this->secondWeight, outputGradient, true, false);
    hiddenGradient = Matrix::multiplyElementwise(hiddenGradient, ReLU::derivative(cache.hiddenPreActivation));

    firstWeightGradient = Matrix::multiply(hiddenGradient, cache.input, false, true);
    firstBiasGradient = FeedForward::sumColumns(hiddenGradient);

    return Matrix::multiply(this->firstWeight, hiddenGradient, true, false);
}
