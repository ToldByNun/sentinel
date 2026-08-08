#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "NeuralNet/Cuda/CudaMatmul.hpp"
#include "NeuralNet/Data/LanguageModelDataset.hpp"
#include "NeuralNet/Network/LanguageModel.hpp"
#include "NeuralNet/Optimizers/Adam.hpp"
#include "NeuralNet/Tokenizer/BPETokenizer.hpp"
#include "NeuralNet/Tokenizer/HfTokenizer.hpp"

namespace nb = nanobind;

namespace {

LanguageModel makeLanguageModel(
    int vocabularySize,
    int embeddingDim,
    int maximumPositionCount,
    float learningRate = 3e-4f,
    int blockCount = 2,
    int headCount = 4,
    int intermediateSize = 0,
    float ropeTheta = 10000.0f,
    bool useBias = true,
    int kvHeadCount = -1) {
    return LanguageModel(
        vocabularySize,
        embeddingDim,
        maximumPositionCount,
        Adam(learningRate),
        blockCount,
        headCount,
        intermediateSize,
        ropeTheta,
        useBias,
        kvHeadCount);
}

} // namespace

NB_MODULE(_core, m) {
    m.doc() = "Sentinel C++/CUDA language-model bindings";
    m.attr("__version__") = "0.1.0";
    m.def("cuda_available", &CudaMatmul::isAvailable, "True if a CUDA device is usable");

    nb::enum_<ActivationCheckpointMode>(m, "ActivationCheckpointMode")
        .value("Off", ActivationCheckpointMode::Off)
        .value("Full", ActivationCheckpointMode::Full)
        .value("Selective", ActivationCheckpointMode::Selective);

    nb::enum_<SbaoMode>(m, "SbaoMode")
        .value("Auto", SbaoMode::Auto)
        .value("GpuInt8Adam", SbaoMode::GpuInt8Adam)
        .value("HostFusedHalfAdam", SbaoMode::HostFusedHalfAdam)
        .value("HostFusedHalfSgd", SbaoMode::HostFusedHalfSgd);

    nb::enum_<SpulseCoverage>(m, "SpulseCoverage")
        .value("Hybrid", SpulseCoverage::Hybrid)
        .value("Full", SpulseCoverage::Full);

    nb::enum_<SpulseMomentumStorage>(m, "SpulseMomentumStorage")
        .value("Fp32", SpulseMomentumStorage::Fp32)
        .value("Fp16", SpulseMomentumStorage::Fp16)
        .value("Int8", SpulseMomentumStorage::Int8);

    nb::class_<BPETokenizer>(m, "BPETokenizer")
        .def(nb::init<>())
        .def(
            "train",
            nb::overload_cast<const std::vector<std::string>&, int>(&BPETokenizer::train),
            nb::arg("corpus"),
            nb::arg("vocab_size"),
            "Learn BPE merges from a list of strings")
        .def("encode", &BPETokenizer::encode, nb::arg("text"))
        .def("decode", &BPETokenizer::decode, nb::arg("token_ids"))
        .def("save", &BPETokenizer::save, nb::arg("path"), "Write binary Sentinel BPE (.sbpe)")
        .def("load", &BPETokenizer::load, nb::arg("path"), "Load binary Sentinel BPE (.sbpe)")
        .def_static(
            "load_from",
            &BPETokenizer::loadFrom,
            nb::arg("path"),
            "Construct a tokenizer from a .sbpe file")
        .def_prop_ro("vocab_size", &BPETokenizer::vocabSize)
        .def_prop_ro("unknown_token_id", &BPETokenizer::unknownTokenId)
        .def_prop_ro("is_trained", &BPETokenizer::isTrained);

    nb::class_<HuggingFace::Tokenizer>(m, "HfTokenizer")
        .def(nb::init<>())
        .def_static(
            "load",
            &HuggingFace::Tokenizer::load,
            nb::arg("path"),
            "Load HuggingFace tokenizer.json from a file or model directory")
        .def(
            "encode",
            &HuggingFace::Tokenizer::encode,
            nb::arg("text"),
            nb::arg("add_special_tokens") = true)
        .def(
            "decode",
            &HuggingFace::Tokenizer::decode,
            nb::arg("token_ids"),
            nb::arg("skip_special_tokens") = true)
        .def_prop_ro("vocab_size", &HuggingFace::Tokenizer::vocabSize)
        .def_prop_ro("bos_token_id", &HuggingFace::Tokenizer::bosTokenId)
        .def_prop_ro("eos_token_id", &HuggingFace::Tokenizer::eosTokenId)
        .def_prop_ro("pad_token_id", &HuggingFace::Tokenizer::padTokenId)
        .def_prop_ro("unk_token_id", &HuggingFace::Tokenizer::unkTokenId)
        .def_prop_ro("is_loaded", &HuggingFace::Tokenizer::isLoaded)
        .def_prop_ro("ignore_merges", &HuggingFace::Tokenizer::ignoreMerges);

    nb::class_<LanguageModelDataset>(m, "LanguageModelDataset")
        .def(nb::init<>())
        .def_static(
            "build",
            &LanguageModelDataset::build,
            nb::arg("texts"),
            nb::arg("tokenizer"),
            nb::arg("maximum_token_count") = 0,
            nb::arg("build_one_hot") = false,
            "Encode texts and build shifted next-token examples")
        .def_prop_ro("size", &LanguageModelDataset::size)
        .def_prop_ro("total_prediction_count", &LanguageModelDataset::totalPredictionCount)
        .def_rw("vocabulary_size", &LanguageModelDataset::vocabularySize);

    nb::class_<LanguageModel>(m, "LanguageModel")
        .def(
            "__init__",
            [](LanguageModel* self,
               int vocabularySize,
               int embeddingDim,
               int maximumPositionCount,
               float learningRate,
               int blockCount,
               int headCount,
               int intermediateSize,
               float ropeTheta,
               bool useBias,
               int kvHeadCount) {
                new (self) LanguageModel(makeLanguageModel(
                    vocabularySize,
                    embeddingDim,
                    maximumPositionCount,
                    learningRate,
                    blockCount,
                    headCount,
                    intermediateSize,
                    ropeTheta,
                    useBias,
                    kvHeadCount));
            },
            nb::arg("vocabulary_size"),
            nb::arg("embedding_dim"),
            nb::arg("maximum_position_count"),
            nb::arg("learning_rate") = 3e-4f,
            nb::arg("block_count") = 2,
            nb::arg("head_count") = 4,
            nb::arg("intermediate_size") = 0,
            nb::arg("rope_theta") = 10000.0f,
            nb::arg("use_bias") = true,
            nb::arg("kv_head_count") = -1,
            "intermediate_size<=0 uses legacy expand-4 SwiGLU width; rope_theta is HF RoPE base; use_bias=false when the HF arch has no FFN/lm_head bias; kv_head_count<=0 → MHA (= head_count)")
        .def("enable_cuda", &LanguageModel::enableCuda)
        .def("enable_cuda_train", &LanguageModel::enableCudaTrain)
        .def(
            "set_activation_checkpoint_mode",
            &LanguageModel::setActivationCheckpointMode,
            nb::arg("mode"))
        .def(
            "set_prefer_flash_attention",
            &LanguageModel::setCudaPreferFlashAttention,
            nb::arg("enabled"))
        .def(
            "set_prefer_muon",
            &LanguageModel::setCudaPreferMuon,
            nb::arg("enabled"))
        .def(
            "set_prefer_spulse",
            &LanguageModel::setCudaPreferSpulse,
            nb::arg("enabled"))
        .def(
            "set_spulse_coverage",
            &LanguageModel::setCudaSpulseCoverage,
            nb::arg("coverage"))
        .def(
            "set_spulse_momentum_beta",
            &LanguageModel::setCudaSpulseMomentumBeta,
            nb::arg("beta"))
        .def(
            "set_spulse_fast_beta",
            &LanguageModel::setCudaSpulseFastBeta,
            nb::arg("beta"))
        .def(
            "set_spulse_slow_beta",
            &LanguageModel::setCudaSpulseSlowBeta,
            nb::arg("beta"))
        .def(
            "set_spulse_scale_clip",
            &LanguageModel::setCudaSpulseScaleClip,
            nb::arg("scale_min"),
            nb::arg("scale_max"))
        .def(
            "set_spulse_momentum_storage",
            &LanguageModel::setCudaSpulseMomentumStorage,
            nb::arg("storage"),
            "Device u storage: Fp32 (default), Fp16, or Int8")
        .def(
            "set_prefer_int8_adam_moments",
            &LanguageModel::setCudaPreferInt8AdamMoments,
            nb::arg("enabled"))
        .def(
            "set_prefer_cpu_adam_offload",
            &LanguageModel::setCudaPreferCpuAdamOffload,
            nb::arg("enabled"))
        .def(
            "set_prefer_host_sgd",
            &LanguageModel::setCudaPreferHostSgd,
            nb::arg("enabled"))
        .def(
            "set_prefer_sbao",
            &LanguageModel::setCudaPreferSbao,
            nb::arg("enabled"))
        .def(
            "set_sbao_mode",
            &LanguageModel::setCudaSbaoMode,
            nb::arg("mode"))
        .def_prop_ro("sbao_mode_resolved", &LanguageModel::cudaSbaoModeResolved)
        .def(
            "set_prefer_train_graph",
            &LanguageModel::setCudaPreferTrainGraph,
            nb::arg("enabled"))
        .def(
            "set_max_packed_columns",
            &LanguageModel::setCudaMaxPackedColumns,
            nb::arg("columns"))
        .def(
            "apply_vram_pack_budget",
            &LanguageModel::applyCudaVramPackBudget,
            nb::arg("free_fraction") = 0.70f,
            nb::arg("safety_reserve_bytes") = 2560ull * 1024ull * 1024ull)
        .def(
            "probe_cuda_packed_train_tokens_per_second",
            &LanguageModel::probeCudaPackedTrainTokensPerSecond,
            nb::arg("sequence_length"),
            nb::arg("warmup_steps") = 3,
            nb::arg("timed_steps") = 8,
            "Synthetic packed-train throughput probe (tok/s)")
        .def_prop_ro("cuda_enabled", &LanguageModel::cudaEnabled)
        .def_prop_ro("cuda_train_enabled", &LanguageModel::cudaTrainEnabled)
        .def_prop_ro("parameter_count", &LanguageModel::parameterElementCount)
        .def_prop_ro("intermediate_size", &LanguageModel::intermediateSize)
        .def_prop_ro("rope_theta", &LanguageModel::ropeTheta)
        .def_prop_ro("use_bias", &LanguageModel::useBias)
        .def_prop_ro("kv_head_count", &LanguageModel::kvHeadCount)
        .def_prop_ro("max_packed_columns", &LanguageModel::cudaMaxPackedColumns)
        .def(
            "average_loss",
            &LanguageModel::averageLoss,
            nb::arg("dataset"))
        .def(
            "train",
            [](LanguageModel& model,
               const LanguageModelDataset& train,
               int epochs,
               int batchSize,
               int gradientAccumulationSteps,
               int logEveryEpochs,
               nb::object testObj) {
                LanguageModelDataset empty;
                if (testObj.is_none()) {
                    model.train(train, empty, epochs, logEveryEpochs, batchSize, gradientAccumulationSteps);
                } else {
                    model.train(
                        train,
                        nb::cast<const LanguageModelDataset&>(testObj),
                        epochs,
                        logEveryEpochs,
                        batchSize,
                        gradientAccumulationSteps);
                }
            },
            nb::arg("train"),
            nb::arg("epochs") = 1,
            nb::arg("batch_size") = 32,
            nb::arg("gradient_accumulation_steps") = 1,
            nb::arg("log_every_epochs") = 1,
            nb::arg("test") = nb::none(),
            "Train on an in-memory dataset (optional test set)")
        .def(
            "generate",
            &LanguageModel::generate,
            nb::arg("prompt_token_ids"),
            nb::arg("new_token_count"),
            nb::arg("temperature") = 1.0f,
            nb::arg("top_k") = 40,
            nb::arg("seed") = 42u)
        .def(
            "save_checkpoint",
            &LanguageModel::saveCheckpoint,
            nb::arg("path"),
            nb::arg("include_optimizer") = true)
        .def(
            "load_checkpoint",
            &LanguageModel::loadCheckpoint,
            nb::arg("path"))
        .def(
            "save_safetensors",
            &LanguageModel::saveSafeTensors,
            nb::arg("path"))
        .def(
            "load_safetensors",
            static_cast<void (LanguageModel::*)(const std::string&)>(&LanguageModel::loadSafeTensors),
            nb::arg("path"))
        .def_static(
            "load_huggingface",
            &LanguageModel::loadHuggingFace,
            nb::arg("path"),
            nb::arg("learning_rate") = 3e-4f,
            "Build + load a causal LM from a HuggingFace model directory (config.json + safetensors)")
        .def(
            "save_huggingface",
            &LanguageModel::saveHuggingFace,
            nb::arg("path"),
            nb::arg("model_type") = "llama",
            nb::arg("tokenizer_source_directory") = "",
            "Export a Transformers-compatible directory (config.json + model.safetensors; optional tokenizer copy)");
}
