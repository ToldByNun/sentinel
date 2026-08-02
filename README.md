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
source.materialize();  // one JSONL+BPE pass; later epochs reuse token ids
model.train(source, epochs, logEvery, batchSize, gradAccum);  // testLoss from reservoir
```

JSONL is read once and tokenized into compact examples (peak text RAM only during that pass). Later epochs only slice cached token ids. Train/test split is a deterministic row-index hash.

## Training (quick)

```cpp
LanguageModel model(vocab, embedDim, maxPos, Adam(0.001f), blocks, heads);
model.enableCuda();
model.enableCudaTrain();          // packed batches, checkpointing, AMP, auto VRAM pack budget
model.setCudaPreferFlashAttention(true);
model.train(trainSet, testSet, epochs, logEvery, batchSize, gradAccum);
```

- `enableCudaTrain` sets `maxPackedColumns` from free GPU memory (`applyVramPackBudget`). Override with `setCudaMaxPackedColumns(n)` (marks manual; auto will not overwrite).
- AMP on stores **saturated FP16** block-input checkpoints (half VRAM vs FP32; restore scratch is one FP32 buffer). Non-finite activations are zeroed on cast.
- AMP loss scaling only kicks in for larger embed dims (when FP16 GEMMs can run).
- 8-bit Adam moments default on: `setCudaPreferInt8AdamMoments(false)` to disable.

### Consumer VRAM proof (RTX 5070 Ti 16 GB)

`CudaLanguageModel::runConsumerVramDemo()` — vocab 8k, embed 256, pos 512, 4 blocks / 8 heads, flash, int8 Adam, auto pack budget, loss scale on:

| | |
|---|---|
| maxPackCols | 16384 (auto, free≈15 GiB × 0.55) |
| setup VRAM | ~460 MiB (weights + workspaces, FP16 checkpoints) |
| after 1 epoch | free still ~11 GiB (Adam moments grow lazily) |
| throughput | ~66k tok/s (epoch, packed) |
| pack | packs=47 avgTok≈12.5k size1=0% |

Fits comfortably on 16 GB; setup footprint leaves headroom for larger packs/models.
