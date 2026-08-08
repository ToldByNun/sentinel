#ifndef PYTORCHSTATEDICT_HPP
#define PYTORCHSTATEDICT_HPP

#include "SafeTensors.hpp"

#include <string>

/// <summary>
/// Minimal PyTorch zipfile state-dict I/O — no libtorch.
/// Supports modern torch.save ZIP archives (PyTorch ≥ 1.6) containing a dict /
/// OrderedDict of CPU tensors (Float / Half / BFloat16 → host F32 Matrix).
/// Legacy non-zip pickle .bin files are rejected with a clear error.
/// </summary>
namespace PytorchStateDict {

/// <summary>true when path is a ZIP archive (PK\\x03\\x04), i.e. modern torch.save</summary>
bool isPytorchZipFile(const std::string& path);

/// <summary>load state-dict tensors from pytorch_model.bin / *.pt / *.bin (zip format)</summary>
SafeTensors::File load(const std::string& path);

/// <summary>
/// write a torch.load-compatible ZIP state-dict (F32 FloatStorage, protocol-2 pickle).
/// Archive layout: {stem}/data.pkl, data/{i}, byteorder, version.
/// </summary>
void save(const std::string& path, const SafeTensors::File& file);

/// <summary>roundtrip + dtype + torch-format fixture smoke</summary>
void runSmokeDemo();

} // namespace PytorchStateDict

#endif // PYTORCHSTATEDICT_HPP
