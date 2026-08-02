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
- CPU Adam offload (ZeRO-Offload Stage-1): `setCudaPreferCpuAdamOffload(true)` keeps `m`/`v` in host RAM (disables int8 device moments). Call before or after `enableCudaTrain`; re-inits train state.
- **Weight tying** default on (`tieEmbeddingProjection`): LM head shares the token embedding matrix (saves ~vocab×embed params + Adam moments). Toggle with `setTieEmbeddingProjection(false)`.

### Consumer VRAM proof (RTX 5070 Ti 16 GB)

Measured with `runScale100M` in `main.cpp` (SERA scale JSONL, Release|x64) — full-train causal LM, **no LoRA**:

| | |
|---|---|
| model | **97.32M** params (tied embed), vocab 16k, d=768, L=12, H=12, maxTok=512 (~371 MiB FP32 weights) |
| flags | activation ckpt, FP16 AMP, int8 Adam, flash attention, CUDA graph |
| VRAM after setup | free **~13.0 GiB** / 15.9 GiB total; auto `maxPackCols`≈3840 |
| probe | **~21.8k** tok/s (seq=256, pack≤32) |
| 1 epoch | trainLoss 6.76, testLoss 7.50, **~19.0k** tok/s, ~603 s |
| data | 37 989 train / 512 test / ~11.4M prediction positions |

Fits on a 16 GB consumer card with several GiB free after setup. Smaller footprint smoke: `CudaLanguageModel::runConsumerVramDemo()` (8k/256/4L — ~66k tok/s packed).

