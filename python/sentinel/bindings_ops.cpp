#include "bindings_ops.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <stdexcept>
#include <utility>
#include <vector>

#include "NeuralNet/Activations/ReLU.hpp"
#include "NeuralNet/Activations/SiLU.hpp"
#include "NeuralNet/Activations/Softmax.hpp"
#include "NeuralNet/Cuda/CudaSPULSE.hpp"
#include "NeuralNet/Initializers/UniformInit.hpp"
#include "NeuralNet/Layers/CausalSelfAttention.hpp"
#include "NeuralNet/Layers/Dense.hpp"
#include "NeuralNet/Layers/Dropout.hpp"
#include "NeuralNet/Layers/Embedding.hpp"
#include "NeuralNet/Layers/FeedForward.hpp"
#include "NeuralNet/Layers/MeanPool.hpp"
#include "NeuralNet/Layers/RMSNorm.hpp"
#include "NeuralNet/Layers/RotaryEmbedding.hpp"
#include "NeuralNet/Layers/TransformerBlock.hpp"
#include "NeuralNet/Losses/CrossEntropy.hpp"
#include "NeuralNet/Losses/MSE.hpp"
#include "NeuralNet/Math/Matrix.hpp"
#include "NeuralNet/Optimizers/Adam.hpp"
#include "NeuralNet/Optimizers/SGD.hpp"
#include "NeuralNet/Optimizers/Spulse.hpp"

namespace nb = nanobind;

namespace {

Matrix matrixFromList(size_t rows, size_t cols, const std::vector<float>& values) {
    if (rows == 0 || cols == 0)
        throw std::invalid_argument("Matrix.from_list rows/cols must be > 0");
    if (values.size() != rows * cols)
        throw std::invalid_argument("Matrix.from_list data size must equal rows*cols");
    Matrix matrix(rows, cols, 0.0f);
    matrix.data = values;
    return matrix;
}

} // namespace

void registerSentinelOps(nb::module_& m) {
    nb::class_<Matrix>(m, "Matrix")
        .def(nb::init<>())
        .def(
            nb::init<size_t, size_t, float>(),
            nb::arg("rows"),
            nb::arg("cols"),
            nb::arg("fill") = 0.0f)
        .def_prop_ro("rows", [](const Matrix& matrix) { return matrix.rows; })
        .def_prop_ro("cols", [](const Matrix& matrix) { return matrix.cols; })
        .def_prop_ro(
            "shape",
            [](const Matrix& matrix) {
                return nb::make_tuple(matrix.rows, matrix.cols);
            })
        .def_prop_ro("empty", &Matrix::empty)
        .def(
            "at",
            [](const Matrix& matrix, size_t row, size_t col) { return matrix.at(row, col); },
            nb::arg("row"),
            nb::arg("col"))
        .def(
            "set",
            [](Matrix& matrix, size_t row, size_t col, float value) { matrix.at(row, col) = value; },
            nb::arg("row"),
            nb::arg("col"),
            nb::arg("value"))
        .def("fill", &Matrix::fill, nb::arg("value"))
        .def("resize", &Matrix::resize, nb::arg("rows"), nb::arg("cols"), nb::arg("fill") = 0.0f)
        .def("ensure_size", &Matrix::ensureSize, nb::arg("rows"), nb::arg("cols"))
        .def(
            "to_list",
            [](const Matrix& matrix) { return matrix.data; },
            "Row-major flat float list")
        .def_static(
            "from_list",
            &matrixFromList,
            nb::arg("rows"),
            nb::arg("cols"),
            nb::arg("values"),
            "Build from a row-major flat float list")
        .def_static("zeros_like", &Matrix::zerosLike, nb::arg("matrix"))
        .def_static("zero_in_place", &Matrix::zeroInPlace, nb::arg("matrix"))
        .def_static("transpose", &Matrix::transpose, nb::arg("matrix"))
        .def_static("add", &Matrix::add, nb::arg("left"), nb::arg("right"))
        .def_static("subtract", &Matrix::subtract, nb::arg("left"), nb::arg("right"))
        .def_static("scale", &Matrix::scale, nb::arg("matrix"), nb::arg("scalar"))
        .def_static(
            "multiply",
            nb::overload_cast<const Matrix&, const Matrix&>(&Matrix::multiply),
            nb::arg("left"),
            nb::arg("right"))
        .def_static(
            "multiply",
            nb::overload_cast<const Matrix&, const Matrix&, bool, bool>(&Matrix::multiply),
            nb::arg("left"),
            nb::arg("right"),
            nb::arg("transpose_left"),
            nb::arg("transpose_right"))
        .def_static(
            "multiply_elementwise",
            &Matrix::multiplyElementwise,
            nb::arg("left"),
            nb::arg("right"))
        .def_static("add_in_place", &Matrix::addInPlace, nb::arg("total"), nb::arg("delta"))
        .def_static("scale_in_place", &Matrix::scaleInPlace, nb::arg("matrix"), nb::arg("scalar"))
        .def_static(
            "gemm",
            &Matrix::gemm,
            nb::arg("left"),
            nb::arg("right"),
            nb::arg("out"),
            nb::arg("transpose_left") = false,
            nb::arg("transpose_right") = false,
            "C = op(A) @ op(B) into out");

    nb::class_<Softmax>(m, "Softmax")
        .def_static("apply", &Softmax::apply, nb::arg("logits"), "Column-wise stable softmax")
        .def_static(
            "apply_into",
            &Softmax::applyInto,
            nb::arg("logits"),
            nb::arg("out"));

    nb::class_<SiLU>(m, "SiLU")
        .def_static("apply_into", &SiLU::applyInto, nb::arg("input"), nb::arg("out"))
        .def_static("derivative_into", &SiLU::derivativeInto, nb::arg("input"), nb::arg("out"));

    nb::class_<ReLU>(m, "ReLU")
        .def_static("apply", &ReLU::apply, nb::arg("matrix"))
        .def_static("apply_into", &ReLU::applyInto, nb::arg("input"), nb::arg("out"))
        .def_static("derivative", &ReLU::derivative, nb::arg("matrix"))
        .def_static("derivative_into", &ReLU::derivativeInto, nb::arg("input"), nb::arg("out"));

    nb::class_<CrossEntropy>(m, "CrossEntropy")
        .def_static(
            "loss",
            &CrossEntropy::loss,
            nb::arg("probabilities"),
            nb::arg("target"),
            "Mean -sum(target * log(p)) over columns")
        .def_static(
            "gradient",
            &CrossEntropy::gradient,
            nb::arg("probabilities"),
            nb::arg("target"),
            "dL/dlogits for mean Softmax+CE: (p - target) / columns");

    nb::class_<MSE>(m, "MSE")
        .def_static("loss", &MSE::loss, nb::arg("prediction"), nb::arg("target"))
        .def_static("gradient", &MSE::gradient, nb::arg("prediction"), nb::arg("target"));

    nb::class_<UniformInit>(m, "UniformInit")
        .def_static(
            "matrix",
            &UniformInit::matrix,
            nb::arg("rows"),
            nb::arg("cols"),
            nb::arg("scale"),
            nb::arg("seed"),
            "Create a rows×cols matrix with values in [-scale, scale]")
        .def_static(
            "fill",
            &UniformInit::fill,
            nb::arg("matrix"),
            nb::arg("scale"),
            nb::arg("seed"));

    nb::class_<AdamState>(m, "AdamState")
        .def(nb::init<>())
        .def_rw("first_moment", &AdamState::firstMoment)
        .def_rw("second_moment", &AdamState::secondMoment)
        .def_static("zeros_like", &AdamState::zerosLike, nb::arg("parameter"));

    nb::class_<MuonState>(m, "MuonState")
        .def(nb::init<>())
        .def_rw("momentum", &MuonState::momentum)
        .def_static("zeros_like", &MuonState::zerosLike, nb::arg("parameter"));

    nb::class_<Adam>(m, "Adam")
        .def(
            nb::init<float, float, float, float>(),
            nb::arg("learning_rate"),
            nb::arg("beta1") = 0.9f,
            nb::arg("beta2") = 0.999f,
            nb::arg("epsilon") = 1e-8f)
        .def_rw("learning_rate", &Adam::learningRate)
        .def_rw("beta1", &Adam::beta1)
        .def_rw("beta2", &Adam::beta2)
        .def_rw("epsilon", &Adam::epsilon)
        .def_rw("time_step", &Adam::timeStep)
        .def("step", &Adam::step, "Advance bias-correction time step (call once per batch)")
        .def(
            "update",
            &Adam::update,
            nb::arg("parameter"),
            nb::arg("state"),
            nb::arg("gradient"))
        .def(
            "update_selected_rows",
            &Adam::updateSelectedRows,
            nb::arg("parameter"),
            nb::arg("state"),
            nb::arg("gradient"),
            nb::arg("row_indices"));

    nb::class_<SGD>(m, "SGD")
        .def(nb::init<float>(), nb::arg("learning_rate"))
        .def_rw("learning_rate", &SGD::learningRate)
        .def("update", &SGD::update, nb::arg("parameter"), nb::arg("gradient"));

    nb::class_<SpulseState>(m, "SpulseState")
        .def(nb::init<>())
        .def_rw("momentum", &SpulseState::momentum)
        .def_rw("energy_fast", &SpulseState::energyFast)
        .def_rw("energy_slow", &SpulseState::energySlow)
        .def_rw("scale", &SpulseState::scale)
        .def_static("zeros_like", &SpulseState::zerosLike, nb::arg("parameter"))
        .def("ensure", &SpulseState::ensure, nb::arg("parameter"))
        .def("clear", &SpulseState::clear);

    // Host SPULSE stepper (CudaSpulse::updateHost); device path stays on LanguageModel.set_prefer_spulse.
    nb::class_<CudaSpulse>(m, "Spulse")
        .def(
            nb::init<float, float, float, float, float, float, float, float, SpulseCoverage, bool, SpulseMomentumStorage, int>(),
            nb::arg("learning_rate") = 3e-3f,
            nb::arg("momentum_beta") = 0.9f,
            nb::arg("fast_beta") = 0.9f,
            nb::arg("slow_beta") = 0.999f,
            nb::arg("epsilon") = 1e-8f,
            nb::arg("scale_min") = 0.25f,
            nb::arg("scale_max") = 4.0f,
            nb::arg("weight_decay") = 0.0f,
            nb::arg("coverage") = SpulseCoverage::Hybrid,
            nb::arg("host_lightweight") = false,
            nb::arg("momentum_storage") = SpulseMomentumStorage::Fp32,
            nb::arg("int8_block_size") = 256)
        .def_rw("learning_rate", &CudaSpulse::learningRate)
        .def_rw("momentum_beta", &CudaSpulse::momentumBeta)
        .def_rw("fast_beta", &CudaSpulse::fastBeta)
        .def_rw("slow_beta", &CudaSpulse::slowBeta)
        .def_rw("epsilon", &CudaSpulse::epsilon)
        .def_rw("scale_min", &CudaSpulse::scaleMin)
        .def_rw("scale_max", &CudaSpulse::scaleMax)
        .def_rw("weight_decay", &CudaSpulse::weightDecay)
        .def_rw("coverage", &CudaSpulse::coverage)
        .def_rw("time_step", &CudaSpulse::timeStep)
        .def("step", &CudaSpulse::step)
        .def(
            "update",
            &CudaSpulse::updateHost,
            nb::arg("parameter"),
            nb::arg("state"),
            nb::arg("gradient"),
            nb::arg("gradient_scale") = 1.0f,
            "Host SPULSE update (dual-horizon energy-scaled momentum)");

    nb::class_<Embedding>(m, "Embedding")
        .def(nb::init<int, int>(), nb::arg("vocab_size"), nb::arg("embedding_dim"))
        .def(nb::init<Matrix>(), nb::arg("weight"))
        .def_rw("weight", &Embedding::weight)
        .def_prop_ro("vocab_size", &Embedding::vocabSize)
        .def_prop_ro("embedding_dim", &Embedding::embeddingDim)
        .def("forward", &Embedding::forward, nb::arg("token_ids"))
        .def(
            "backward",
            &Embedding::backward,
            nb::arg("output_gradient"),
            nb::arg("token_ids"),
            "Scatter output grads into embedding rows");

    nb::class_<Dense>(m, "Dense")
        .def(nb::init<Matrix, Matrix>(), nb::arg("weight"), nb::arg("bias"))
        .def_rw("weight", &Dense::weight)
        .def_rw("bias", &Dense::bias)
        .def("forward", &Dense::forward, nb::arg("input"), "z = W @ x + b");

    nb::class_<Dropout>(m, "Dropout")
        .def(nb::init<float, unsigned>(), nb::arg("drop_rate") = 0.3f, nb::arg("seed") = 7u)
        .def_rw("drop_rate", &Dropout::dropRate)
        .def_rw("training", &Dropout::training)
        .def_rw("seed", &Dropout::seed)
        .def("forward", &Dropout::forward, nb::arg("input"))
        .def("backward", &Dropout::backward, nb::arg("output_gradient"));

    nb::class_<MeanPool>(m, "MeanPool")
        .def(nb::init<>())
        .def_rw("last_sequence_length", &MeanPool::lastSequenceLength)
        .def("forward", &MeanPool::forward, nb::arg("embeddings"))
        .def("backward", &MeanPool::backward, nb::arg("pooled_gradient"));

    nb::class_<RMSNormCache>(m, "RMSNormCache")
        .def(nb::init<>());

    nb::class_<RMSNorm>(m, "RMSNorm")
        .def(nb::init<int, float>(), nb::arg("embedding_dim"), nb::arg("epsilon") = 1e-5f)
        .def_rw("gamma", &RMSNorm::gamma)
        .def_rw("epsilon", &RMSNorm::epsilon)
        .def("forward", &RMSNorm::forward, nb::arg("input"), nb::arg("cache"))
        .def(
            "backward",
            [](const RMSNorm& norm, const Matrix& outputGradient, const RMSNormCache& cache) {
                Matrix gammaGradient;
                Matrix inputGradient = norm.backward(outputGradient, cache, gammaGradient);
                return nb::make_tuple(std::move(inputGradient), std::move(gammaGradient));
            },
            nb::arg("output_gradient"),
            nb::arg("cache"),
            "Returns (input_gradient, gamma_gradient)");

    nb::class_<RotaryEmbedding>(m, "RotaryEmbedding")
        .def(nb::init<>())
        .def(
            nb::init<int, int, float>(),
            nb::arg("head_dimension"),
            nb::arg("maximum_position_count"),
            nb::arg("base") = RotaryEmbedding::DefaultBase)
        .def_ro("head_dimension", &RotaryEmbedding::headDimension)
        .def_ro("maximum_position_count", &RotaryEmbedding::maximumPositionCount)
        .def_ro("base", &RotaryEmbedding::base)
        .def(
            "rotate_in_place",
            &RotaryEmbedding::rotateInPlace,
            nb::arg("tensor"),
            nb::arg("head_count"))
        .def(
            "rotate_inverse_in_place",
            &RotaryEmbedding::rotateInverseInPlace,
            nb::arg("tensor"),
            nb::arg("head_count"));

    nb::class_<CausalSelfAttentionCache>(m, "CausalSelfAttentionCache")
        .def(nb::init<>());

    nb::class_<CausalSelfAttention>(m, "CausalSelfAttention")
        .def_static(
            "create",
            &CausalSelfAttention::create,
            nb::arg("embedding_dim"),
            nb::arg("head_count"),
            nb::arg("maximum_position_count"),
            nb::arg("seed") = 11u,
            nb::arg("window_size") = -1,
            nb::arg("global_token_count") = 0,
            nb::arg("rope_base") = RotaryEmbedding::DefaultBase,
            nb::arg("kv_head_count") = -1)
        .def_rw("query_weight", &CausalSelfAttention::queryWeight)
        .def_rw("key_weight", &CausalSelfAttention::keyWeight)
        .def_rw("value_weight", &CausalSelfAttention::valueWeight)
        .def_rw("output_weight", &CausalSelfAttention::outputWeight)
        .def_rw("rotary_embedding", &CausalSelfAttention::rotaryEmbedding)
        .def_rw("head_count", &CausalSelfAttention::headCount)
        .def_rw("kv_head_count", &CausalSelfAttention::kvHeadCount)
        .def_rw("head_dimension", &CausalSelfAttention::headDimension)
        .def_rw("window_size", &CausalSelfAttention::windowSize)
        .def_rw("global_token_count", &CausalSelfAttention::globalTokenCount)
        .def_rw("prefer_sparse_compute", &CausalSelfAttention::preferSparseCompute)
        .def("forward", &CausalSelfAttention::forward, nb::arg("input"), nb::arg("cache"))
        .def(
            "backward",
            [](const CausalSelfAttention& attention,
               const Matrix& outputGradient,
               CausalSelfAttentionCache& cache) {
                Matrix queryGrad;
                Matrix keyGrad;
                Matrix valueGrad;
                Matrix outputGrad;
                Matrix inputGrad = attention.backward(
                    outputGradient, cache, queryGrad, keyGrad, valueGrad, outputGrad);
                return nb::make_tuple(
                    std::move(inputGrad),
                    std::move(queryGrad),
                    std::move(keyGrad),
                    std::move(valueGrad),
                    std::move(outputGrad));
            },
            nb::arg("output_gradient"),
            nb::arg("cache"),
            "Returns (input_grad, q_w_grad, k_w_grad, v_w_grad, o_w_grad)");

    nb::class_<FeedForwardCache>(m, "FeedForwardCache")
        .def(nb::init<>());

    nb::class_<FeedForward>(m, "FeedForward")
        .def_static(
            "create",
            &FeedForward::create,
            nb::arg("embedding_dim"),
            nb::arg("expand_ratio") = 4,
            nb::arg("seed") = 41u,
            nb::arg("use_bias") = true)
        .def_static(
            "create_with_intermediate_size",
            &FeedForward::createWithIntermediateSize,
            nb::arg("embedding_dim"),
            nb::arg("intermediate_size"),
            nb::arg("seed") = 41u,
            nb::arg("use_bias") = true)
        .def_static(
            "default_intermediate_size",
            &FeedForward::defaultIntermediateSize,
            nb::arg("embedding_dim"),
            nb::arg("expand_ratio") = 4)
        .def_rw("gate_weight", &FeedForward::gateWeight)
        .def_rw("gate_bias", &FeedForward::gateBias)
        .def_rw("up_weight", &FeedForward::upWeight)
        .def_rw("up_bias", &FeedForward::upBias)
        .def_rw("down_weight", &FeedForward::downWeight)
        .def_rw("down_bias", &FeedForward::downBias)
        .def_rw("use_bias", &FeedForward::useBias)
        .def_prop_ro("intermediate_size", &FeedForward::intermediateSize)
        .def("forward", &FeedForward::forward, nb::arg("input"), nb::arg("cache"))
        .def(
            "backward",
            [](const FeedForward& ffn, const Matrix& outputGradient, FeedForwardCache& cache) {
                Matrix gateW;
                Matrix gateB;
                Matrix upW;
                Matrix upB;
                Matrix downW;
                Matrix downB;
                Matrix inputGrad = ffn.backward(
                    outputGradient, cache, gateW, gateB, upW, upB, downW, downB);
                return nb::make_tuple(
                    std::move(inputGrad),
                    std::move(gateW),
                    std::move(gateB),
                    std::move(upW),
                    std::move(upB),
                    std::move(downW),
                    std::move(downB));
            },
            nb::arg("output_gradient"),
            nb::arg("cache"),
            "Returns (input_grad, gate_w, gate_b, up_w, up_b, down_w, down_b)");

    nb::class_<TransformerBlockCache>(m, "TransformerBlockCache")
        .def(nb::init<>());

    nb::class_<TransformerBlockGradients>(m, "TransformerBlockGradients")
        .def(nb::init<>())
        .def_rw("query_weight", &TransformerBlockGradients::queryWeight)
        .def_rw("key_weight", &TransformerBlockGradients::keyWeight)
        .def_rw("value_weight", &TransformerBlockGradients::valueWeight)
        .def_rw("attention_output_weight", &TransformerBlockGradients::attentionOutputWeight)
        .def_rw("attention_norm_gamma", &TransformerBlockGradients::attentionNormGamma)
        .def_rw("feed_forward_norm_gamma", &TransformerBlockGradients::feedForwardNormGamma)
        .def_rw("feed_forward_gate_weight", &TransformerBlockGradients::feedForwardGateWeight)
        .def_rw("feed_forward_gate_bias", &TransformerBlockGradients::feedForwardGateBias)
        .def_rw("feed_forward_up_weight", &TransformerBlockGradients::feedForwardUpWeight)
        .def_rw("feed_forward_up_bias", &TransformerBlockGradients::feedForwardUpBias)
        .def_rw("feed_forward_down_weight", &TransformerBlockGradients::feedForwardDownWeight)
        .def_rw("feed_forward_down_bias", &TransformerBlockGradients::feedForwardDownBias)
        .def_static("zeros_from", &TransformerBlockGradients::zerosFrom, nb::arg("block"))
        .def("zero_in_place", &TransformerBlockGradients::zeroInPlace)
        .def("add_in_place", &TransformerBlockGradients::addInPlace, nb::arg("other"))
        .def("scale_in_place", &TransformerBlockGradients::scaleInPlace, nb::arg("scalar"));

    nb::class_<TransformerBlock>(m, "TransformerBlock")
        .def(
            nb::init<int, int, int, unsigned, int, float, bool, int>(),
            nb::arg("embedding_dim"),
            nb::arg("head_count"),
            nb::arg("maximum_position_count"),
            nb::arg("seed") = 41u,
            nb::arg("intermediate_size") = 0,
            nb::arg("rope_theta") = 10000.0f,
            nb::arg("use_bias") = true,
            nb::arg("kv_head_count") = -1)
        .def_prop_ro(
            "attention",
            [](TransformerBlock& block) -> CausalSelfAttention& { return block.attention; },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "feed_forward",
            [](TransformerBlock& block) -> FeedForward& { return block.feedForward; },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "attention_norm",
            [](TransformerBlock& block) -> RMSNorm& { return block.attentionNorm; },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "feed_forward_norm",
            [](TransformerBlock& block) -> RMSNorm& { return block.feedForwardNorm; },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "query_weight",
            [](TransformerBlock& block) -> Matrix& { return block.attention.queryWeight; },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "key_weight",
            [](TransformerBlock& block) -> Matrix& { return block.attention.keyWeight; },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "value_weight",
            [](TransformerBlock& block) -> Matrix& { return block.attention.valueWeight; },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "attention_output_weight",
            [](TransformerBlock& block) -> Matrix& { return block.attention.outputWeight; },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "gate_weight",
            [](TransformerBlock& block) -> Matrix& { return block.feedForward.gateWeight; },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "up_weight",
            [](TransformerBlock& block) -> Matrix& { return block.feedForward.upWeight; },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "down_weight",
            [](TransformerBlock& block) -> Matrix& { return block.feedForward.downWeight; },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "head_count",
            [](const TransformerBlock& block) { return block.attention.headCount; })
        .def_prop_ro(
            "kv_head_count",
            [](const TransformerBlock& block) { return block.attention.kvHeadCount; })
        .def(
            "forward",
            &TransformerBlock::forward,
            nb::arg("input"),
            nb::arg("cache"))
        .def(
            "backward",
            &TransformerBlock::backward,
            nb::arg("output_gradient"),
            nb::arg("cache"),
            nb::arg("gradients"))
        .def(
            "apply_gradients",
            &TransformerBlock::applyGradients,
            nb::arg("optimizer"),
            nb::arg("gradients"));
}
