#include "FeedForward.hpp"

#include "../Activations/SiLU.hpp"
#include "../Initializers/UniformInit.hpp"

#include <stdexcept>
#include <utility>

FeedForward::FeedForward(Matrix gateWeight, Matrix gateBias, Matrix upWeight, Matrix upBias, Matrix downWeight, Matrix downBias)
    : gateWeight(std::move(gateWeight)), gateBias(std::move(gateBias)), upWeight(std::move(upWeight)), upBias(std::move(upBias)), downWeight(std::move(downWeight)), downBias(std::move(downBias)) {}

FeedForward FeedForward::create(int embeddingDim, int expandRatio, unsigned seed) {
    if (embeddingDim <= 0) throw std::invalid_argument("FeedForward::create embeddingDim must be > 0");
    if (expandRatio <= 0) throw std::invalid_argument("FeedForward::create expandRatio must be > 0");

    const int hiddenDim = (2 * embeddingDim * expandRatio) / 3;
    if (hiddenDim <= 0) throw std::invalid_argument("FeedForward::create hiddenDim must be > 0");

    return FeedForward(UniformInit::matrix(hiddenDim, embeddingDim, 0.1f, seed), UniformInit::matrix(hiddenDim, 1, 0.01f, seed + 1u), UniformInit::matrix(hiddenDim, embeddingDim, 0.1f, seed + 2u), UniformInit::matrix(hiddenDim, 1, 0.01f, seed + 3u), UniformInit::matrix(embeddingDim, hiddenDim, 0.1f, seed + 4u), UniformInit::matrix(embeddingDim, 1, 0.01f, seed + 5u));
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

void FeedForward::multiplyElementwiseInto(const Matrix& left, const Matrix& right, Matrix& out) {
    if (left.rows != right.rows || left.cols != right.cols) throw std::invalid_argument("FeedForward::multiplyElementwiseInto shape mismatch");
    out.ensureSize(left.rows, left.cols);
    for (size_t index = 0; index < left.data.size(); ++index)
        out.data[index] = left.data[index] * right.data[index];
}

Matrix FeedForward::forward(const Matrix& input, FeedForwardCache& cache) const {
    if (input.empty()) throw std::invalid_argument("FeedForward::forward empty input");
    if (this->gateWeight.cols != input.rows) throw std::invalid_argument("FeedForward::forward embedding dim mismatch");

    cache.input = input;
    Matrix::gemm(this->gateWeight, input, cache.gatePreActivation);
    FeedForward::broadcastBiasAddInPlace(cache.gatePreActivation, this->gateBias);
    SiLU::applyInto(cache.gatePreActivation, cache.gateActivated);

    Matrix::gemm(this->upWeight, input, cache.up);
    FeedForward::broadcastBiasAddInPlace(cache.up, this->upBias);
    FeedForward::multiplyElementwiseInto(cache.gateActivated, cache.up, cache.hidden);

    Matrix::gemm(this->downWeight, cache.hidden, cache.output);
    FeedForward::broadcastBiasAddInPlace(cache.output, this->downBias);
    return cache.output;
}

Matrix FeedForward::backward(const Matrix& outputGradient, FeedForwardCache& cache, Matrix& gateWeightGradient, Matrix& gateBiasGradient, Matrix& upWeightGradient, Matrix& upBiasGradient, Matrix& downWeightGradient, Matrix& downBiasGradient) const {
    if (cache.input.empty()) throw std::logic_error("FeedForward::backward called before forward");
    if (outputGradient.rows != this->downWeight.rows || outputGradient.cols != cache.input.cols) throw std::invalid_argument("FeedForward::backward shape mismatch");

    Matrix::gemm(outputGradient, cache.hidden, downWeightGradient, false, true);
    FeedForward::sumColumnsInto(outputGradient, downBiasGradient);

    Matrix::gemm(this->downWeight, outputGradient, cache.hiddenGradient, true, false);
    FeedForward::multiplyElementwiseInto(cache.hiddenGradient, cache.gateActivated, cache.upGradient);
    FeedForward::multiplyElementwiseInto(cache.hiddenGradient, cache.up, cache.gateGradient);

    SiLU::derivativeInto(cache.gatePreActivation, cache.siluDerivative);
    Matrix::multiplyElementwiseInPlace(cache.gateGradient, cache.siluDerivative);

    Matrix::gemm(cache.gateGradient, cache.input, gateWeightGradient, false, true);
    FeedForward::sumColumnsInto(cache.gateGradient, gateBiasGradient);
    Matrix::gemm(cache.upGradient, cache.input, upWeightGradient, false, true);
    FeedForward::sumColumnsInto(cache.upGradient, upBiasGradient);

    Matrix::gemm(this->gateWeight, cache.gateGradient, cache.inputGradient, true, false);
    Matrix::gemm(this->upWeight, cache.upGradient, cache.temp, true, false);
    Matrix::addInPlace(cache.inputGradient, cache.temp);
    return cache.inputGradient;
}
