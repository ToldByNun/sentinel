#include "TransformerBlock.hpp"

#include <stdexcept>

TransformerBlockGradients TransformerBlockGradients::zerosFrom(const TransformerBlock& block) {
    TransformerBlockGradients gradients;
    gradients.queryWeight = Matrix::zerosLike(block.attention.queryWeight);
    gradients.keyWeight = Matrix::zerosLike(block.attention.keyWeight);
    gradients.valueWeight = Matrix::zerosLike(block.attention.valueWeight);
    gradients.attentionOutputWeight = Matrix::zerosLike(block.attention.outputWeight);
    gradients.attentionNormGamma = Matrix::zerosLike(block.attentionNorm.gamma);
    gradients.attentionNormBeta = Matrix::zerosLike(block.attentionNorm.beta);
    gradients.feedForwardNormGamma = Matrix::zerosLike(block.feedForwardNorm.gamma);
    gradients.feedForwardNormBeta = Matrix::zerosLike(block.feedForwardNorm.beta);
    gradients.feedForwardFirstWeight = Matrix::zerosLike(block.feedForward.firstWeight);
    gradients.feedForwardFirstBias = Matrix::zerosLike(block.feedForward.firstBias);
    gradients.feedForwardSecondWeight = Matrix::zerosLike(block.feedForward.secondWeight);
    gradients.feedForwardSecondBias = Matrix::zerosLike(block.feedForward.secondBias);
    return gradients;
}

void TransformerBlockGradients::zeroInPlace() {
    Matrix::zeroInPlace(this->queryWeight);
    Matrix::zeroInPlace(this->keyWeight);
    Matrix::zeroInPlace(this->valueWeight);
    Matrix::zeroInPlace(this->attentionOutputWeight);
    Matrix::zeroInPlace(this->attentionNormGamma);
    Matrix::zeroInPlace(this->attentionNormBeta);
    Matrix::zeroInPlace(this->feedForwardNormGamma);
    Matrix::zeroInPlace(this->feedForwardNormBeta);
    Matrix::zeroInPlace(this->feedForwardFirstWeight);
    Matrix::zeroInPlace(this->feedForwardFirstBias);
    Matrix::zeroInPlace(this->feedForwardSecondWeight);
    Matrix::zeroInPlace(this->feedForwardSecondBias);
}

void TransformerBlockGradients::addInPlace(const TransformerBlockGradients& other) {
    Matrix::addInPlace(this->queryWeight, other.queryWeight);
    Matrix::addInPlace(this->keyWeight, other.keyWeight);
    Matrix::addInPlace(this->valueWeight, other.valueWeight);
    Matrix::addInPlace(this->attentionOutputWeight, other.attentionOutputWeight);
    Matrix::addInPlace(this->attentionNormGamma, other.attentionNormGamma);
    Matrix::addInPlace(this->attentionNormBeta, other.attentionNormBeta);
    Matrix::addInPlace(this->feedForwardNormGamma, other.feedForwardNormGamma);
    Matrix::addInPlace(this->feedForwardNormBeta, other.feedForwardNormBeta);
    Matrix::addInPlace(this->feedForwardFirstWeight, other.feedForwardFirstWeight);
    Matrix::addInPlace(this->feedForwardFirstBias, other.feedForwardFirstBias);
    Matrix::addInPlace(this->feedForwardSecondWeight, other.feedForwardSecondWeight);
    Matrix::addInPlace(this->feedForwardSecondBias, other.feedForwardSecondBias);
}

void TransformerBlockGradients::scaleInPlace(float scalar) {
    Matrix::scaleInPlace(this->queryWeight, scalar);
    Matrix::scaleInPlace(this->keyWeight, scalar);
    Matrix::scaleInPlace(this->valueWeight, scalar);
    Matrix::scaleInPlace(this->attentionOutputWeight, scalar);
    Matrix::scaleInPlace(this->attentionNormGamma, scalar);
    Matrix::scaleInPlace(this->attentionNormBeta, scalar);
    Matrix::scaleInPlace(this->feedForwardNormGamma, scalar);
    Matrix::scaleInPlace(this->feedForwardNormBeta, scalar);
    Matrix::scaleInPlace(this->feedForwardFirstWeight, scalar);
    Matrix::scaleInPlace(this->feedForwardFirstBias, scalar);
    Matrix::scaleInPlace(this->feedForwardSecondWeight, scalar);
    Matrix::scaleInPlace(this->feedForwardSecondBias, scalar);
}

TransformerBlock::TransformerBlock(int embeddingDim, int headCount, unsigned seed)
    : attentionNorm(embeddingDim), attention(CausalSelfAttention::create(embeddingDim, headCount, seed)), feedForwardNorm(embeddingDim), feedForward(FeedForward::create(embeddingDim, 4, seed + 20u)) {
    if (embeddingDim <= 0) throw std::invalid_argument("TransformerBlock embeddingDim must be > 0");

    this->queryWeightState = AdamState::zerosLike(this->attention.queryWeight);
    this->keyWeightState = AdamState::zerosLike(this->attention.keyWeight);
    this->valueWeightState = AdamState::zerosLike(this->attention.valueWeight);
    this->attentionOutputWeightState = AdamState::zerosLike(this->attention.outputWeight);
    this->attentionNormGammaState = AdamState::zerosLike(this->attentionNorm.gamma);
    this->attentionNormBetaState = AdamState::zerosLike(this->attentionNorm.beta);
    this->feedForwardNormGammaState = AdamState::zerosLike(this->feedForwardNorm.gamma);
    this->feedForwardNormBetaState = AdamState::zerosLike(this->feedForwardNorm.beta);
    this->feedForwardFirstWeightState = AdamState::zerosLike(this->feedForward.firstWeight);
    this->feedForwardFirstBiasState = AdamState::zerosLike(this->feedForward.firstBias);
    this->feedForwardSecondWeightState = AdamState::zerosLike(this->feedForward.secondWeight);
    this->feedForwardSecondBiasState = AdamState::zerosLike(this->feedForward.secondBias);
}

Matrix TransformerBlock::forward(const Matrix& input, TransformerBlockCache& cache) const {
    cache.input = input;

    Matrix attentionInput = this->attentionNorm.forward(input, cache.attentionNormCache);
    Matrix attended = this->attention.forward(attentionInput, cache.attentionCache);
    cache.afterAttention = Matrix::add(input, attended);

    Matrix feedForwardInput = this->feedForwardNorm.forward(cache.afterAttention, cache.feedForwardNormCache);
    Matrix feedForwardOutput = this->feedForward.forward(feedForwardInput, cache.feedForwardCache);
    cache.output = Matrix::add(cache.afterAttention, feedForwardOutput);
    return cache.output;
}

Matrix TransformerBlock::backward(const Matrix& outputGradient, const TransformerBlockCache& cache, TransformerBlockGradients& gradients) const {
    Matrix feedForwardFirstWeightGradient;
    Matrix feedForwardFirstBiasGradient;
    Matrix feedForwardSecondWeightGradient;
    Matrix feedForwardSecondBiasGradient;
    Matrix feedForwardInputGradient = this->feedForward.backward(outputGradient, cache.feedForwardCache, feedForwardFirstWeightGradient, feedForwardFirstBiasGradient, feedForwardSecondWeightGradient, feedForwardSecondBiasGradient);

    Matrix feedForwardNormGammaGradient;
    Matrix feedForwardNormBetaGradient;
    Matrix afterAttentionFromFeedForward = this->feedForwardNorm.backward(feedForwardInputGradient, cache.feedForwardNormCache, feedForwardNormGammaGradient, feedForwardNormBetaGradient);

    Matrix afterAttentionGradient = Matrix::add(outputGradient, afterAttentionFromFeedForward);

    Matrix queryWeightGradient;
    Matrix keyWeightGradient;
    Matrix valueWeightGradient;
    Matrix attentionOutputWeightGradient;
    Matrix attentionInputGradient = this->attention.backward(afterAttentionGradient, cache.attentionCache, queryWeightGradient, keyWeightGradient, valueWeightGradient, attentionOutputWeightGradient);

    Matrix attentionNormGammaGradient;
    Matrix attentionNormBetaGradient;
    Matrix inputFromAttention = this->attentionNorm.backward(attentionInputGradient, cache.attentionNormCache, attentionNormGammaGradient, attentionNormBetaGradient);

    Matrix::addInPlace(gradients.feedForwardSecondWeight, feedForwardSecondWeightGradient);
    Matrix::addInPlace(gradients.feedForwardSecondBias, feedForwardSecondBiasGradient);
    Matrix::addInPlace(gradients.feedForwardFirstWeight, feedForwardFirstWeightGradient);
    Matrix::addInPlace(gradients.feedForwardFirstBias, feedForwardFirstBiasGradient);
    Matrix::addInPlace(gradients.feedForwardNormGamma, feedForwardNormGammaGradient);
    Matrix::addInPlace(gradients.feedForwardNormBeta, feedForwardNormBetaGradient);
    Matrix::addInPlace(gradients.attentionOutputWeight, attentionOutputWeightGradient);
    Matrix::addInPlace(gradients.valueWeight, valueWeightGradient);
    Matrix::addInPlace(gradients.keyWeight, keyWeightGradient);
    Matrix::addInPlace(gradients.queryWeight, queryWeightGradient);
    Matrix::addInPlace(gradients.attentionNormGamma, attentionNormGammaGradient);
    Matrix::addInPlace(gradients.attentionNormBeta, attentionNormBetaGradient);

    return Matrix::add(afterAttentionGradient, inputFromAttention);
}

void TransformerBlock::applyGradients(Adam& optimizer, const TransformerBlockGradients& gradients) {
    optimizer.update(this->feedForward.secondWeight, this->feedForwardSecondWeightState, gradients.feedForwardSecondWeight);
    optimizer.update(this->feedForward.secondBias, this->feedForwardSecondBiasState, gradients.feedForwardSecondBias);
    optimizer.update(this->feedForward.firstWeight, this->feedForwardFirstWeightState, gradients.feedForwardFirstWeight);
    optimizer.update(this->feedForward.firstBias, this->feedForwardFirstBiasState, gradients.feedForwardFirstBias);
    optimizer.update(this->feedForwardNorm.gamma, this->feedForwardNormGammaState, gradients.feedForwardNormGamma);
    optimizer.update(this->feedForwardNorm.beta, this->feedForwardNormBetaState, gradients.feedForwardNormBeta);
    optimizer.update(this->attention.outputWeight, this->attentionOutputWeightState, gradients.attentionOutputWeight);
    optimizer.update(this->attention.valueWeight, this->valueWeightState, gradients.valueWeight);
    optimizer.update(this->attention.keyWeight, this->keyWeightState, gradients.keyWeight);
    optimizer.update(this->attention.queryWeight, this->queryWeightState, gradients.queryWeight);
    optimizer.update(this->attentionNorm.gamma, this->attentionNormGammaState, gradients.attentionNormGamma);
    optimizer.update(this->attentionNorm.beta, this->attentionNormBetaState, gradients.attentionNormBeta);
}
