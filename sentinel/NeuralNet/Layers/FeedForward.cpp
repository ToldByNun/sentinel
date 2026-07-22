#include "FeedForward.hpp"

#include "../Activations/ReLU.hpp"
#include "../Initializers/UniformInit.hpp"

#include <stdexcept>
#include <utility>

FeedForward::FeedForward(Matrix firstWeight, Matrix firstBias, Matrix secondWeight, Matrix secondBias)
    : firstWeight(std::move(firstWeight)), firstBias(std::move(firstBias)), secondWeight(std::move(secondWeight)), secondBias(std::move(secondBias)) {}

FeedForward FeedForward::create(int embeddingDim, int expandRatio, unsigned seed) {
    if (embeddingDim <= 0) throw std::invalid_argument("FeedForward::create embeddingDim must be > 0");
    if (expandRatio <= 0) throw std::invalid_argument("FeedForward::create expandRatio must be > 0");

    const int hiddenDim = embeddingDim * expandRatio;
    return FeedForward(UniformInit::matrix(hiddenDim, embeddingDim, 0.1f, seed), UniformInit::matrix(hiddenDim, 1, 0.01f, seed + 1u), UniformInit::matrix(embeddingDim, hiddenDim, 0.1f, seed + 2u), UniformInit::matrix(embeddingDim, 1, 0.01f, seed + 3u));
}

void FeedForward::broadcastBiasAddInPlace(Matrix& product, const Matrix& bias) {
    for (size_t row = 0; row < product.rows; ++row) {
        const float biasValue = bias.at(row, 0);
        for (size_t column = 0; column < product.cols; ++column)
            product.at(row, column) += biasValue;
    }
}

void FeedForward::sumColumnsInto(const Matrix& gradient, Matrix& biasGradient) {
    biasGradient.ensureSize(gradient.rows, 1);
    for (size_t row = 0; row < gradient.rows; ++row) {
        float total = 0.0f;
        for (size_t column = 0; column < gradient.cols; ++column)
            total += gradient.at(row, column);
        biasGradient.at(row, 0) = total;
    }
}

Matrix FeedForward::forward(const Matrix& input, FeedForwardCache& cache) const {
    if (input.empty()) throw std::invalid_argument("FeedForward::forward empty input");
    if (this->firstWeight.cols != input.rows) throw std::invalid_argument("FeedForward::forward embedding dim mismatch");

    cache.input = input;
    Matrix::gemm(this->firstWeight, input, cache.hiddenPreActivation);
    FeedForward::broadcastBiasAddInPlace(cache.hiddenPreActivation, this->firstBias);
    ReLU::applyInto(cache.hiddenPreActivation, cache.hiddenActivated);

    Matrix::gemm(this->secondWeight, cache.hiddenActivated, cache.output);
    FeedForward::broadcastBiasAddInPlace(cache.output, this->secondBias);
    return cache.output;
}

Matrix FeedForward::backward(const Matrix& outputGradient, FeedForwardCache& cache, Matrix& firstWeightGradient, Matrix& firstBiasGradient, Matrix& secondWeightGradient, Matrix& secondBiasGradient) const {
    if (cache.input.empty()) throw std::logic_error("FeedForward::backward called before forward");
    if (outputGradient.rows != this->secondWeight.rows || outputGradient.cols != cache.input.cols) throw std::invalid_argument("FeedForward::backward shape mismatch");

    Matrix::gemm(outputGradient, cache.hiddenActivated, secondWeightGradient, false, true);
    FeedForward::sumColumnsInto(outputGradient, secondBiasGradient);

    Matrix::gemm(this->secondWeight, outputGradient, cache.hiddenGradient, true, false);
    ReLU::derivativeInto(cache.hiddenPreActivation, cache.reluDerivative);
    Matrix::multiplyElementwiseInPlace(cache.hiddenGradient, cache.reluDerivative);

    Matrix::gemm(cache.hiddenGradient, cache.input, firstWeightGradient, false, true);
    FeedForward::sumColumnsInto(cache.hiddenGradient, firstBiasGradient);

    Matrix::gemm(this->firstWeight, cache.hiddenGradient, cache.inputGradient, true, false);
    return cache.inputGradient;
}
