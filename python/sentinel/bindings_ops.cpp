#include "bindings_ops.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <stdexcept>
#include <utility>
#include <vector>

#include "NeuralNet/Activations/SiLU.hpp"
#include "NeuralNet/Activations/Softmax.hpp"
#include "NeuralNet/Layers/Embedding.hpp"
#include "NeuralNet/Layers/RMSNorm.hpp"
#include "NeuralNet/Layers/TransformerBlock.hpp"
#include "NeuralNet/Losses/CrossEntropy.hpp"
#include "NeuralNet/Math/Matrix.hpp"
#include "NeuralNet/Optimizers/Adam.hpp"
#include "NeuralNet/Optimizers/SGD.hpp"

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

    nb::class_<AdamState>(m, "AdamState")
        .def(nb::init<>())
        .def_rw("first_moment", &AdamState::firstMoment)
        .def_rw("second_moment", &AdamState::secondMoment)
        .def_static("zeros_like", &AdamState::zerosLike, nb::arg("parameter"));

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
