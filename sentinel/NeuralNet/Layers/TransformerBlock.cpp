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
    gradients.feedForwardGateWeight = Matrix::zerosLike(block.feedForward.gateWeight);
    gradients.feedForwardGateBias = Matrix::zerosLike(block.feedForward.gateBias);
    gradients.feedForwardUpWeight = Matrix::zerosLike(block.feedForward.upWeight);
    gradients.feedForwardUpBias = Matrix::zerosLike(block.feedForward.upBias);
    gradients.feedForwardDownWeight = Matrix::zerosLike(block.feedForward.downWeight);
    gradients.feedForwardDownBias = Matrix::zerosLike(block.feedForward.downBias);
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
    Matrix::zeroInPlace(this->feedForwardGateWeight);
    Matrix::zeroInPlace(this->feedForwardGateBias);
    Matrix::zeroInPlace(this->feedForwardUpWeight);
    Matrix::zeroInPlace(this->feedForwardUpBias);
    Matrix::zeroInPlace(this->feedForwardDownWeight);
    Matrix::zeroInPlace(this->feedForwardDownBias);
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
    Matrix::addInPlace(this->feedForwardGateWeight, other.feedForwardGateWeight);
    Matrix::addInPlace(this->feedForwardGateBias, other.feedForwardGateBias);
    Matrix::addInPlace(this->feedForwardUpWeight, other.feedForwardUpWeight);
    Matrix::addInPlace(this->feedForwardUpBias, other.feedForwardUpBias);
    Matrix::addInPlace(this->feedForwardDownWeight, other.feedForwardDownWeight);
    Matrix::addInPlace(this->feedForwardDownBias, other.feedForwardDownBias);
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
    Matrix::scaleInPlace(this->feedForwardGateWeight, scalar);
    Matrix::scaleInPlace(this->feedForwardGateBias, scalar);
    Matrix::scaleInPlace(this->feedForwardUpWeight, scalar);
    Matrix::scaleInPlace(this->feedForwardUpBias, scalar);
    Matrix::scaleInPlace(this->feedForwardDownWeight, scalar);
    Matrix::scaleInPlace(this->feedForwardDownBias, scalar);
}

TransformerBlock::TransformerBlock(int embeddingDim, int headCount, int maximumPositionCount, unsigned seed)
    : attentionNorm(embeddingDim), attention(CausalSelfAttention::create(embeddingDim, headCount, maximumPositionCount, seed)), feedForwardNorm(embeddingDim), feedForward(FeedForward::create(embeddingDim, 4, seed + 20u)) {
    if (embeddingDim <= 0) throw std::invalid_argument("TransformerBlock embeddingDim must be > 0");

    this->queryWeightState = AdamState::zerosLike(this->attention.queryWeight);
    this->keyWeightState = AdamState::zerosLike(this->attention.keyWeight);
    this->valueWeightState = AdamState::zerosLike(this->attention.valueWeight);
    this->attentionOutputWeightState = AdamState::zerosLike(this->attention.outputWeight);
    this->attentionNormGammaState = AdamState::zerosLike(this->attentionNorm.gamma);
    this->attentionNormBetaState = AdamState::zerosLike(this->attentionNorm.beta);
    this->feedForwardNormGammaState = AdamState::zerosLike(this->feedForwardNorm.gamma);
    this->feedForwardNormBetaState = AdamState::zerosLike(this->feedForwardNorm.beta);
    this->feedForwardGateWeightState = AdamState::zerosLike(this->feedForward.gateWeight);
    this->feedForwardGateBiasState = AdamState::zerosLike(this->feedForward.gateBias);
    this->feedForwardUpWeightState = AdamState::zerosLike(this->feedForward.upWeight);
    this->feedForwardUpBiasState = AdamState::zerosLike(this->feedForward.upBias);
    this->feedForwardDownWeightState = AdamState::zerosLike(this->feedForward.downWeight);
    this->feedForwardDownBiasState = AdamState::zerosLike(this->feedForward.downBias);
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

Matrix TransformerBlock::backward(const Matrix& outputGradient, TransformerBlockCache& cache, TransformerBlockGradients& gradients) const {
    Matrix feedForwardGateWeightGradient;
    Matrix feedForwardGateBiasGradient;
    Matrix feedForwardUpWeightGradient;
    Matrix feedForwardUpBiasGradient;
    Matrix feedForwardDownWeightGradient;
    Matrix feedForwardDownBiasGradient;
    Matrix feedForwardInputGradient = this->feedForward.backward(outputGradient, cache.feedForwardCache, feedForwardGateWeightGradient, feedForwardGateBiasGradient, feedForwardUpWeightGradient, feedForwardUpBiasGradient, feedForwardDownWeightGradient, feedForwardDownBiasGradient);

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

    Matrix::addInPlace(gradients.feedForwardDownWeight, feedForwardDownWeightGradient);
    Matrix::addInPlace(gradients.feedForwardDownBias, feedForwardDownBiasGradient);
    Matrix::addInPlace(gradients.feedForwardUpWeight, feedForwardUpWeightGradient);
    Matrix::addInPlace(gradients.feedForwardUpBias, feedForwardUpBiasGradient);
    Matrix::addInPlace(gradients.feedForwardGateWeight, feedForwardGateWeightGradient);
    Matrix::addInPlace(gradients.feedForwardGateBias, feedForwardGateBiasGradient);
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
    optimizer.update(this->feedForward.downWeight, this->feedForwardDownWeightState, gradients.feedForwardDownWeight);
    optimizer.update(this->feedForward.downBias, this->feedForwardDownBiasState, gradients.feedForwardDownBias);
    optimizer.update(this->feedForward.upWeight, this->feedForwardUpWeightState, gradients.feedForwardUpWeight);
    optimizer.update(this->feedForward.upBias, this->feedForwardUpBiasState, gradients.feedForwardUpBias);
    optimizer.update(this->feedForward.gateWeight, this->feedForwardGateWeightState, gradients.feedForwardGateWeight);
    optimizer.update(this->feedForward.gateBias, this->feedForwardGateBiasState, gradients.feedForwardGateBias);
    optimizer.update(this->feedForwardNorm.gamma, this->feedForwardNormGammaState, gradients.feedForwardNormGamma);
    optimizer.update(this->feedForwardNorm.beta, this->feedForwardNormBetaState, gradients.feedForwardNormBeta);
    optimizer.update(this->attention.outputWeight, this->attentionOutputWeightState, gradients.attentionOutputWeight);
    optimizer.update(this->attention.valueWeight, this->valueWeightState, gradients.valueWeight);
    optimizer.update(this->attention.keyWeight, this->keyWeightState, gradients.keyWeight);
    optimizer.update(this->attention.queryWeight, this->queryWeightState, gradients.queryWeight);
    optimizer.update(this->attentionNorm.gamma, this->attentionNormGammaState, gradients.attentionNormGamma);
    optimizer.update(this->attentionNorm.beta, this->attentionNormBetaState, gradients.attentionNormBeta);
}
