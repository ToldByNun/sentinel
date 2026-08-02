# Sentinel

Small C++/CUDA neural net framework focused on **full-train** causal language models (no LoRA).

CPU/OpenMP path plus CUDA device training: packed batches, activation checkpointing, optional FP16 AMP, flash attention.

## Requirements

- Windows, Visual Studio 2022+ (toolset `v145`)
- CUDA Toolkit (project points at `v13.3`)
- NVIDIA GPU matching the project arch (`sm_120` in the vcxproj — change if needed)
- Optional: `SERA-Data/sera_sample.jsonl` next to the repo for the train demo

## Build

```bat
msbuild sentinel\sentinel.vcxproj /p:Configuration=Release /p:Platform=x64
```

Or open `sentinel.slnx` in Visual Studio and build Release|x64.

Output: `x64\Release\sentinel.exe`

## Run

Working directory must be `sentinel\` (relative data path):

```bat
cd sentinel
..\x64\Release\sentinel.exe
```

`main.cpp` runs smoke tests (matmul, layers, attention, Adam, streaming dataset, …) then a short streamed SERA LM train (2 epochs, embed=64).

## Layout

| Path | Contents |
|------|----------|
| `sentinel/NeuralNet/Network/` | `LanguageModel`, CPU training |
| `sentinel/NeuralNet/Cuda/` | Device mirror, train kernels, AMP, Adam, flash |
| `sentinel/NeuralNet/Layers/` | Attention, FFN, RMSNorm, embedding, … |
| `sentinel/NeuralNet/Data/` | JSONL, dataset split, LM examples |
| `sentinel/main.cpp` | Smoke + SERA demo |

## Checkpoints

```cpp
model.saveCheckpoint("run.snlm", true);   // weights + Adam moments (FP32 on disk)
model.loadCheckpoint("run.snlm");         // restores weights, Adam, timeStep
```

Binary format `SNLM` v1. Architecture must match. Int8 Adam moments are dequantized to FP32 for the file.

## Streaming data

```cpp
LanguageModelChunkSource source(path, maxChars, maxTokens, /*chunk*/256, /*trainRatio*/0.8f, seed, /*testCap*/256);
auto sample = source.prepareTokenizerSample(2000);
tokenizer.train(sample, 1000);
source.setTokenizer(&tokenizer);
source.prepareTestReservoir();
model.train(source, epochs, logEvery, batchSize, gradAccum);  // testLoss from reservoir
```

JSONL is read in chunks (peak RAM ≈ chunk + test reservoir + tokenizer sample). Train/test split is a deterministic row-index hash.

## Training (quick)

```cpp
LanguageModel model(vocab, embedDim, maxPos, Adam(0.001f), blocks, heads);
model.enableCuda();
model.enableCudaTrain();          // packed batches, checkpointing, AMP
model.setCudaMaxPackedColumns(2048);
model.setCudaPreferFlashAttention(true);
model.train(trainSet, testSet, epochs, logEvery, batchSize, gradAccum);
```

- AMP loss scaling only kicks in for larger embed dims (when FP16 GEMMs can run).
- 8-bit Adam moments default on: `setCudaPreferInt8AdamMoments(false)` to disable.
