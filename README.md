# Sentinel

C++/CUDA framework for **full-train** causal LMs (no LoRA): CPU/OpenMP plus a device train path with packed batches, flash attention, FP16 AMP, int8 Adam, and selective activation checkpointing.

## Requirements

- Windows, Visual Studio 2022+ (`v145`)
- CUDA Toolkit (project targets **v13.3**)
- NVIDIA GPU — vcxproj currently builds `sm_120` (RTX 50-series); change `CodeGeneration` for other arches (intended floor: Turing `sm_75`+)
- Data (optional for demos): `SERA-Data/sera_best_subset` (HF Arrow) and/or `*.jsonl`

## Build & run

```bat
msbuild sentinel\sentinel.vcxproj /p:Configuration=Release /p:Platform=x64
cd sentinel
..\x64\Release\sentinel.exe
```

Working directory must be `sentinel\` (relative `../SERA-Data/...`). Flags in `main.cpp` gate smokes, benches, and the SERA train demo (default: Arrow corpus, ~8×768).

## Layout

| Path | Role |
|------|------|
| `NeuralNet/Network/` | `LanguageModel`, host train, `.snlm` checkpoints |
| `NeuralNet/Cuda/` | Device mirror, flash attn, AMP, Adam, packed train |
| `NeuralNet/Layers/` | Attention, SwiGLU FFN, RMSNorm, embedding, … |
| `NeuralNet/Data/` | `TextRowReader`, JSONL + from-scratch Arrow IPC reader, chunk source |
| `NeuralNet/Tokenizer/` | BPE (freq-based train, ranked encode) |
| `main.cpp` | Smokes / scale harness / SERA demo |

## Data

```cpp
LanguageModelChunkSource source(path, maxChars, maxTokens, chunk, trainRatio, seed, testCap);
auto sample = source.prepareTokenizerSample(2000);
tokenizer.train(sample, vocabSize);
source.setTokenizer(&tokenizer);
source.materialize();   // one corpus+BPE pass → cached token ids
model.train(source, epochs, logEvery, batchSize, gradAccum);
```

- **JSONL** (`.jsonl`) or **HF Arrow** (`.arrow` / `save_to_disk` dir with `data-*.arrow`; `cache-*` skipped)
- Path auto-detect via `createTextRowReader` — no Apache Arrow C++ dependency (custom IPC stream reader)
- Progress logs for sample / BPE / materialize (`SmokeLog::progress`)

## Training

```cpp
LanguageModel model(vocab, embed, maxPos, Adam(0.001f), blocks, heads);
model.enableCuda();
model.enableCudaTrain();  // pack budget, AMP, int8 Adam, ckpt=Selective
model.setCudaPreferFlashAttention(true);
model.train(source, epochs, 1, batchSize, gradAccum);
```

| Knob | Default / notes |
|------|-----------------|
| Pack budget | `applyVramPackBudget` from free VRAM; override `setCudaMaxPackedColumns` |
| Flash attention | Prefer on for train/forward |
| AMP | FP16 GEMMs + saturated FP16 block-input checkpoints; loss scale when embed≥256 |
| Adam | Int8 moments on; or `setCudaPreferCpuAdamOffload(true)` (host `m`/`v`) |
| Weight tying | On — LM head shares token embedding |
| Activation ckpt | `Off` / `Full` / **`Selective`** (default): keep FFN acts, recompute Attn; `enableActivationCheckpointing(bool)` maps true→Selective, false→Off |
| CUDA graphs | Used when checkpointing is **Off** and shapes are stable |

```cpp
model.setActivationCheckpointMode(ActivationCheckpointMode::Selective); // or Full / Off
model.saveCheckpoint("run.snlm", true);
model.loadCheckpoint("run.snlm");
```

Checkpoint file: `SNLM` (weights + optional Adam; int8 moments stored as FP32).

## Consumer VRAM proof (RTX 5070 Ti 16 GB)

`runScale100M` — full train, no LoRA:

| | |
|---|---|
| Model | **97.32M** (tied), vocab 16k, d=768, L=12, H=12, maxTok=512 |
| Flags | ckpt + FP16 AMP + int8 Adam + flash + graph |
| After setup | free **~13 GiB** / 15.9 GiB; `maxPackCols`≈3840 |
| Probe / epoch | **~21.8k** tok/s (seq=256) · epoch **~19.0k** tok/s (~603 s) |
| Data | ~38k train / 512 test / ~11.4M positions |

Smaller smoke: `CudaLanguageModel::runConsumerVramDemo()` (~8k/256/4L, ~66k tok/s packed).
