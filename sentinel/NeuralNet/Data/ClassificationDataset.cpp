#include "ClassificationDataset.hpp"

#include <stdexcept>

Matrix ClassificationDataset::makeOneHot(int label, int classCount) {
    if (label < 0 || label >= classCount) throw std::out_of_range("ClassificationDataset::makeOneHot label out of range");

    Matrix target;
    target.data = std::vector<std::vector<float>>(
        static_cast<size_t>(classCount),
        std::vector<float>(1, 0.0f)
    );
    target.data[label][0] = 1.0f;
    return target;
}

int ClassificationDataset::inferLabel(const std::string& text) {
    if (text.find('{') != std::string::npos && text.find('"') != std::string::npos) return ClassJson;
    if (text.find("import ") != std::string::npos || text.find("def ") != std::string::npos) return ClassPython;
    return ClassCpp;
}

ClassificationDataset ClassificationDataset::build(
    const std::vector<std::string>& corpus,
    const BPETokenizer& tokenizer
) {
    ClassificationDataset dataset;

    for (const std::string& text : corpus) {
        ClassificationExample example;
        example.label = ClassificationDataset::inferLabel(text);
        example.tokenIds = tokenizer.encode(text);
        example.target = ClassificationDataset::makeOneHot(example.label, ClassificationDataset::ClassCount);
        dataset.examples.push_back(std::move(example));
    }

    return dataset;
}

ClassificationDataset ClassificationDataset::buildLabeled(
    const std::vector<std::string>& texts,
    const std::vector<int>& labels,
    const BPETokenizer& tokenizer,
    int classCount
) {
    if (texts.size() != labels.size())
        throw std::invalid_argument("ClassificationDataset::buildLabeled size mismatch");

    ClassificationDataset dataset;
    for (size_t index = 0; index < texts.size(); ++index) {
        ClassificationExample example;
        example.label = labels[index];
        example.tokenIds = tokenizer.encode(texts[index]);
        example.target = ClassificationDataset::makeOneHot(example.label, classCount);
        dataset.examples.push_back(std::move(example));
    }
    return dataset;
}

int ClassificationDataset::size() const {
    return static_cast<int>(this->examples.size());
}
