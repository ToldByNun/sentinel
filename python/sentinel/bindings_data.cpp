#include "bindings_data.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <stdexcept>
#include <utility>
#include <vector>

#include "NeuralNet/Data/ClassificationDataset.hpp"
#include "NeuralNet/Data/JsonlLoader.hpp"
#include "NeuralNet/Data/LanguageModelChunkSource.hpp"
#include "NeuralNet/Data/LanguageModelDataset.hpp"
#include "NeuralNet/Data/TextRowReader.hpp"
#include "NeuralNet/IO/SafeTensors.hpp"
#include "NeuralNet/Layers/Dense.hpp"
#include "NeuralNet/Layers/Dropout.hpp"
#include "NeuralNet/Layers/Embedding.hpp"
#include "NeuralNet/Layers/MeanPool.hpp"
#include "NeuralNet/Math/Matrix.hpp"
#include "NeuralNet/Network/Sequential.hpp"
#include "NeuralNet/Optimizers/Adam.hpp"
#include "NeuralNet/Tokenizer/BPETokenizer.hpp"

namespace nb = nanobind;

void registerSentinelData(nb::module_& m) {
    nb::class_<CorpusRow>(m, "CorpusRow")
        .def(nb::init<>())
        .def_rw("text", &CorpusRow::text)
        .def_rw("source", &CorpusRow::source);

    nb::class_<JsonlLoader>(m, "JsonlLoader")
        .def_static(
            "load",
            &JsonlLoader::load,
            nb::arg("path"),
            nb::arg("maximum_rows") = 50,
            "Load up to maximum_rows CorpusRow entries from a .jsonl file (≤0 = no limit)")
        .def_static(
            "try_parse_line",
            [](const std::string& line) -> nb::object {
                CorpusRow row;
                if (!JsonlLoader::tryParseLine(line, row))
                    return nb::none();
                return nb::cast(std::move(row));
            },
            nb::arg("line"))
        .def_static("source_to_label", &JsonlLoader::sourceToLabel, nb::arg("source"));

    nb::class_<LanguageModelChunkSource>(m, "LanguageModelChunkSource")
        .def(
            nb::init<std::string, size_t, size_t, int, float, unsigned, int>(),
            nb::arg("path"),
            nb::arg("maximum_text_characters") = 0,
            nb::arg("maximum_token_count") = 0,
            nb::arg("chunk_example_count") = 64,
            nb::arg("train_ratio") = 0.9f,
            nb::arg("seed") = 42u,
            nb::arg("test_reservoir_cap") = 256)
        .def(
            "set_tokenizer",
            [](LanguageModelChunkSource& source, const BPETokenizer& tokenizer) {
                source.setTokenizer(&tokenizer);
            },
            nb::arg("tokenizer"),
            nb::keep_alive<1, 2>(),
            "Tokenizer must outlive the chunk source")
        .def(
            "prepare_tokenizer_sample",
            &LanguageModelChunkSource::prepareTokenizerSample,
            nb::arg("max_rows"))
        .def("materialize", &LanguageModelChunkSource::materialize)
        .def("prepare_test_reservoir", &LanguageModelChunkSource::prepareTestReservoir)
        .def("rewind_train", &LanguageModelChunkSource::rewindTrain)
        .def("sort_train_by_length", &LanguageModelChunkSource::sortTrainByLength)
        .def(
            "fill_train_dataset",
            &LanguageModelChunkSource::fillTrainDataset,
            nb::arg("out"))
        .def(
            "next_train_chunk",
            &LanguageModelChunkSource::nextTrainChunk,
            nb::arg("out"),
            "Fill out with the next train chunk; returns False when exhausted")
        .def_prop_ro(
            "test_dataset",
            [](const LanguageModelChunkSource& source) -> const LanguageModelDataset& {
                return source.testDataset();
            },
            nb::rv_policy::reference_internal)
        .def_prop_ro("file_path", &LanguageModelChunkSource::filePath)
        .def_prop_ro("chunk_example_count", &LanguageModelChunkSource::chunkExampleCount)
        .def_prop_ro("train_ratio", &LanguageModelChunkSource::trainRatio)
        .def_prop_ro("train_example_count", &LanguageModelChunkSource::trainExampleCount)
        .def_prop_ro("train_prediction_count", &LanguageModelChunkSource::trainPredictionCount)
        .def_prop_ro("is_materialized", &LanguageModelChunkSource::isMaterialized);

    nb::class_<ClassificationExample>(m, "ClassificationExample")
        .def(nb::init<>())
        .def_rw("token_ids", &ClassificationExample::tokenIds)
        .def_rw("target", &ClassificationExample::target)
        .def_rw("label", &ClassificationExample::label);

    nb::class_<ClassificationDataset>(m, "ClassificationDataset")
        .def(nb::init<>())
        .def_static(
            "make_one_hot",
            &ClassificationDataset::makeOneHot,
            nb::arg("label"),
            nb::arg("class_count") = ClassificationDataset::ClassCount)
        .def_static("infer_label", &ClassificationDataset::inferLabel, nb::arg("text"))
        .def_static(
            "build",
            &ClassificationDataset::build,
            nb::arg("corpus"),
            nb::arg("tokenizer"))
        .def_static(
            "build_labeled",
            &ClassificationDataset::buildLabeled,
            nb::arg("texts"),
            nb::arg("labels"),
            nb::arg("tokenizer"),
            nb::arg("class_count"))
        .def_prop_ro("size", &ClassificationDataset::size)
        .def_prop_ro(
            "examples",
            [](ClassificationDataset& dataset) -> std::vector<ClassificationExample>& {
                return dataset.examples;
            },
            nb::rv_policy::reference_internal);

    m.attr("CLASS_CPP") = ClassificationDataset::ClassCpp;
    m.attr("CLASS_JSON") = ClassificationDataset::ClassJson;
    m.attr("CLASS_PYTHON") = ClassificationDataset::ClassPython;
    m.attr("CLASS_COUNT") = ClassificationDataset::ClassCount;

    nb::class_<Sequential>(m, "Sequential")
        .def(
            nb::init<Dense, Dense, Adam, float>(),
            nb::arg("layer1"),
            nb::arg("layer2"),
            nb::arg("optimizer"),
            nb::arg("drop_rate") = 0.3f,
            "Embedding→MeanPool→Dense→ReLU→Dropout→Dense→Softmax classifier stack")
        .def_rw("layer1", &Sequential::layer1)
        .def_rw("layer2", &Sequential::layer2)
        .def_rw("dropout", &Sequential::dropout)
        .def_rw("optimizer", &Sequential::optimizer)
        .def(
            "forward",
            nb::overload_cast<const Matrix&>(&Sequential::forward),
            nb::arg("input"))
        .def(
            "forward",
            nb::overload_cast<Embedding&, MeanPool&, const std::vector<int>&>(&Sequential::forward),
            nb::arg("embedding"),
            nb::arg("mean_pool"),
            nb::arg("token_ids"))
        .def(
            "train",
            nb::overload_cast<const Matrix&, const Matrix&, int>(&Sequential::train),
            nb::arg("input"),
            nb::arg("target"),
            nb::arg("epochs"))
        .def(
            "train",
            nb::overload_cast<Embedding&, MeanPool&, const ClassificationDataset&, int>(&Sequential::train),
            nb::arg("embedding"),
            nb::arg("mean_pool"),
            nb::arg("dataset"),
            nb::arg("epochs"))
        .def(
            "train",
            nb::overload_cast<
                Embedding&,
                MeanPool&,
                const ClassificationDataset&,
                const ClassificationDataset&,
                int,
                int,
                int,
                int>(&Sequential::train),
            nb::arg("embedding"),
            nb::arg("mean_pool"),
            nb::arg("train"),
            nb::arg("test"),
            nb::arg("epochs"),
            nb::arg("log_every_epochs") = 500,
            nb::arg("early_stopping_patience") = 3,
            nb::arg("batch_size") = 16)
        .def(
            "predict_class",
            &Sequential::predictClass,
            nb::arg("embedding"),
            nb::arg("mean_pool"),
            nb::arg("token_ids"))
        .def(
            "accuracy",
            &Sequential::accuracy,
            nb::arg("embedding"),
            nb::arg("mean_pool"),
            nb::arg("dataset"));

    nb::class_<SafeTensors::File>(m, "SafeTensorsFile")
        .def(nb::init<>())
        .def_rw("metadata", &SafeTensors::File::metadata)
        .def_prop_ro(
            "tensor_names",
            [](const SafeTensors::File& file) {
                std::vector<std::string> names;
                names.reserve(file.tensors.size());
                for (const auto& entry : file.tensors)
                    names.push_back(entry.first);
                return names;
            })
        .def(
            "has_tensor",
            [](const SafeTensors::File& file, const std::string& name) {
                return file.tensors.find(name) != file.tensors.end();
            },
            nb::arg("name"))
        .def(
            "get_tensor",
            [](SafeTensors::File& file, const std::string& name) -> Matrix& {
                auto it = file.tensors.find(name);
                if (it == file.tensors.end())
                    throw std::out_of_range("SafeTensorsFile missing tensor: " + name);
                return it->second;
            },
            nb::arg("name"),
            nb::rv_policy::reference_internal)
        .def(
            "set_tensor",
            [](SafeTensors::File& file, const std::string& name, const Matrix& matrix) {
                file.tensors[name] = matrix;
            },
            nb::arg("name"),
            nb::arg("matrix"))
        .def(
            "put_matrix",
            [](SafeTensors::File& file, const std::string& name, const Matrix& matrix) {
                SafeTensors::putMatrix(file, name, matrix);
            },
            nb::arg("name"),
            nb::arg("matrix"));

    m.def(
        "safetensors_load",
        &SafeTensors::load,
        nb::arg("path"),
        "Load a .safetensors file into a SafeTensorsFile (F32/BF16/F16 → F32)");
    m.def(
        "safetensors_save",
        &SafeTensors::save,
        nb::arg("path"),
        nb::arg("file"),
        "Write a SafeTensorsFile (F32)");
    m.def(
        "is_safetensors_file",
        &SafeTensors::isSafeTensorsFile,
        nb::arg("path"));
}
