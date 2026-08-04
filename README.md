# Sentinel

C++/CUDA framework for **full-train** causal LMs (no LoRA): CPU/OpenMP plus a device train path with packed batches, flash attention, FP16 AMP, int8 Adam / Muon, selective activation checkpointing, and a consumer-GPU **~4B offload** path (FP16 GPU weights + host masters/grads + host SGD).

## Requirements

- **OS:** Windows or Linux (x86_64)
- C++20 compiler — MSVC (`v145+`), GCC ≥ 11, or Clang ≥ 14 — plus **CMake ≥ 3.24**
- CUDA Toolkit (**13.x**; developed against v13.3) with matching NVIDIA driver
- NVIDIA GPU — GeForce **20 / 30 / 40 / 50** (`sm_75` / `sm_80`+`sm_86` / `sm_89` / `sm_120`). CUDA 13 floor is Turing.
- OpenMP (libgomp / LLVM OpenMP / MSVC OpenMP)
- Data (not included): `SERA-Data/sera_best_subset` (HF Arrow) and/or `*.jsonl`



## Build & run

### CMake (recommended)

**Linux**

```bash
# CUDA on PATH, e.g. export PATH=/usr/local/cuda/bin:$PATH
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
cd sentinel
../build/bin/sentinel
```

**Windows**

```bat
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j
cd sentinel
..\build\bin\sentinel.exe
```

Fast local rebuild for your GPU only: `-DSENTINEL_CUDA_ARCHITECTURES=native`.  
Default fat binary arches: `75;80;86;89;120` (RTX 20 / 30 / 40 / 50).

Working directory must be `sentinel/` (relative `../SERA-Data/...`).

### MSBuild / Visual Studio (Windows only)

```bat
msbuild sentinel\sentinel.vcxproj /p:Configuration=Release /p:Platform=x64
cd sentinel
..\x64\Release\sentinel.exe
```

Flags in `main.cpp` gate demos:


| Flag                     | Purpose                                                         |
| ------------------------ | --------------------------------------------------------------- |
| `runSmallSuite`          | Smokes + parities + speed 768 + ~100M probe (no corpus / no 4B) |
| `runScale100M`           | Full multi-epoch ~100M train on `sera_scale.jsonl`              |
| `runMuonThroughputProbe` | Adam vs Muon × ckpt tok/s table                                 |
| `runScale4BTrainStep`    | One packed ~4B train step (FP16w + host SGD)                    |
| `runSpeedBench` / others | Focused benches                                                 |


Default SERA demo (when suite flags are off): Arrow corpus, ~8×768.

## Layout


| Path                   | Role                                                                 |
| ---------------------- | -------------------------------------------------------------------- |
| `NeuralNet/Network/`   | `LanguageModel`, host train, `.snlm` checkpoints                     |
| `NeuralNet/Cuda/`      | Device mirror, flash attn, AMP, Adam, Muon, packed train, 4B offload |
| `NeuralNet/Layers/`    | Attention, SwiGLU FFN, RMSNorm, embedding, …                         |
| `NeuralNet/Data/`      | `TextRowReader`, JSONL + from-scratch Arrow IPC reader, chunk source |
| `NeuralNet/Tokenizer/` | BPE (freq-based train, ranked encode)                                |
| `main.cpp`             | Smokes / scale harness / SERA demo                                   |




## Data

```cpp
LanguageModelChunkSource source(path, maxChars, maxTokens, chunk, trainRatio, seed, testCap);
auto sample = source.prepareTokenizerSample(2000);
tokenizer.train(sample, vocabSize);
source.setTokenizer(&tokenizer);
source.materialize();   // one corpus+BPE pass → cached token ids
model.train(source, epochs, logEvery, batchSize, gradAccum);
```

- **JSONL** (`.jsonl`) or **HF Arrow** (`.arrow` / `save_to_disk` dir with `data-*.arrow`; `cache-`* skipped)
- Path auto-detect via `createTextRowReader` — no Apache Arrow C++ dependency (custom IPC stream reader)
- Progress logs for sample / BPE / materialize (`SmokeLog::progress`)



## Training

```cpp
LanguageModel model(vocab, embed, maxPos, Adam(0.001f), blocks, heads);
model.enableCuda();
model.enableCudaTrain();  // pack budget, AMP, int8 Adam, ckpt=Off (peak tok/s)
model.setCudaPreferFlashAttention(true);
model.train(source, epochs, 1, batchSize, gradAccum);
```


| Knob                | Default / notes                                                                                                                                                  |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Pack budget         | `applyVramPackBudget` (default **70%** of usable free VRAM, **≥20%** display floor, slack 1.2×, cap **4096** cols). Override with `setCudaMaxPackedColumns`.     |
| Flash attention     | Prefer on for train/forward                                                                                                                                      |
| AMP                 | FP16 GEMMs + saturated FP16 block-input checkpoints; loss scale when embed≥256                                                                                   |
| Adam                | Int8 moments on GPU; or `setCudaPreferCpuAdamOffload(true)` (host `m`/`v`)                                                                                       |
| Muon                | `setCudaPreferMuon(true)` — Newton–Schulz on 2D hidden weights; Adam on embed / norms / biases / head                                                            |
| Large-model offload | `preferFp16GpuWeights` + `preferHostGradients` + `preferHostSgd` — FP16 weights on GPU, FP32 masters/grads on host, SGD (no Adam `m`/`v`). Used by the 4B probe. |
| Weight tying        | On — LM head shares token embedding                                                                                                                              |
| Activation ckpt     | Default **Off** (retain act scratch across steps). `Selective`/`Full` when VRAM is tight; `enableActivationCheckpointing(bool)` maps true→Selective, false→Off   |
| CUDA graphs         | Only when checkpointing is **Off** and shapes are stable (`preferTrainGraph`)                                                                                    |


```cpp
model.setActivationCheckpointMode(ActivationCheckpointMode::Selective); // or Full / Off
model.saveCheckpoint("run.snlm", true);
model.loadCheckpoint("run.snlm");
```

Checkpoint file: `SNLM` (weights + optional Adam; int8 moments stored as FP32).

## Measured throughput (RTX 5070 Ti 16 GB)

Numbers below are **fresh synthetic packed-train probes** (`probeCudaPackedTrainTokensPerSecond`, seq=256) unless noted. Desktop WDDM / pack budget matter — re-measure after big train-path changes (`runMuonThroughputProbe`, `runSmallSuite`).

### ~60M — 8×768 (speed path)


|       |                                                                    |
| ----- | ------------------------------------------------------------------ |
| Shape | vocab 4k, d=768, L=8, H=12, maxPos=512                             |
| Flags | Selective ckpt, FP16 AMP, int8 Adam, flash, auto pack (~4096 cols) |
| Probe | **~18–20k** tok/s                                                  |




### ~97M — 12×768 (consumer VRAM proof)

`runScale100M` — full multi-epoch train on `sera_scale.jsonl` (tied embed, no LoRA). Headline probe is Adam with **ckpt=off** (act scratch retained; graphs on).


|                         |                                                                                                    |
| ----------------------- | -------------------------------------------------------------------------------------------------- |
| Model                   | **~97M** (tied), vocab 16k, d=768, L=12, H=12, maxTok=512                                          |
| Flags                   | ckpt=off, FP16 AMP, int8 Adam, flash (WMMA TC hot path), train graph on                            |
| Pack                    | auto budget (~3840–4096 cols on 16 GB)                                                             |
| Probe (Adam + ckpt=off) | **~25–26k** tok/s                                                                                  |
| Note                    | Selective is slower here (~11–13k) because Attn is recomputed each layer — use when VRAM is tight. |


#### Adam vs Muon × checkpoint (same 12×768 shape)

Fresh one-shot models (`runMuonThroughputProbe`), FP16 AMP + int8 Adam + flash:


| Optimizer | ckpt      | pack              | tok/s       |
| --------- | --------- | ----------------- | ----------- |
| Adam      | **Off**   | auto / ~3840–4096 | **~25–26k** |
| Adam      | Selective | ~3840             | ~11–13k     |
| Muon      | Off / Sel | auto / ~3840      | lower (NS)  |


Muon adds Newton–Schulz cost on hidden weights; comparing a Muon probe to an Adam README number looks like a 2× “regression” even when Adam is unchanged.

Full SERA epoch numbers still need `sera_scale.jsonl` + `runScale100M=true`.

### ~4B — FP16 GPU + host SGD (fits 16 GB)

`runScale4BTrainStep` — one packed train step (not multi-epoch):


|            |                                                                                               |
| ---------- | --------------------------------------------------------------------------------------------- |
| Shape      | vocab 32k, d=3072, L=34, H=48, maxPos=2048, seq=512, **pack=8** (4096 tokens)                 |
| Mode       | FP16 GPU weights, host FP32 masters + grads, **host SGD** (no Adam moments), Full ckpt, flash |
| Host RAM   | ~15 GiB masters+grads (Adam `m`/`v` would roughly double that)                                |
| Full step  | **~320–330** tok/s (accumulate alone ~380–420)                                                |
| Bottleneck | Full-ckpt bwd recompute + end-of-step H2D; async Grad-D2H is mostly overlapped                |


Target ≥600 tok/s full-step is not yet reached on this offload+Full-ckpt path.

Smaller smoke: `CudaLanguageModel::runConsumerVramDemo()` (~8k/256/4L).

## Verify

```bash
# Linux
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
# Windows (or MSBuild — see above)
# cmake -S . -B build && cmake --build build --config Release -j
# in main.cpp: runSmallSuite = true  (other scale flags false)
cd sentinel
../build/bin/sentinel   # Windows: ..\build\bin\sentinel.exe
```

Expect smokes/parities OK, **speed 768 ~18–20k** tok/s, **scale-100M probe ~25k** (Adam ckpt=off, WMMA flash).

FlashAttention `headDim=64` / tiles `64×64` uses CUDA WMMA Tensor Cores (toolkit only — no cuDNN). Toggle `runWmmaFaVerify` in `main.cpp` for a quick parity + probe.
