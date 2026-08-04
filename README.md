# Sentinel

C++/CUDA framework for **full-train** causal LMs (no LoRA): CPU/OpenMP plus a device train path with packed batches, flash attention, FP16 AMP, int8 Adam / Muon, selective activation checkpointing, and a consumer-GPU **~4B offload** path (FP16 GPU weights + host masters/grads + host SGD).

**v0.1** ships as an installable C++ library (`Sentinel::sentinel`), a demo harness, examples, and a **pip-installable Python package** (`sentinel-lm`).

> **Note on “zero-dependency”:** That claim (e.g. in the GitHub *About* blurb — *zero-dependency C++/CUDA DL framework focused on single consumer GPUs*) refers to the **engine runtime itself**: the C++/CUDA library links only against the CUDA toolkit (cuBLAS / cuBLASLt), OpenMP, and the C++ standard library — no cuDNN, no PyTorch/TensorFlow, no other third-party DL stacks at train/infer time. It does **not** mean the optional **Python bindings** are dependency-free: building `sentinel-lm` needs a Python toolchain plus build-time packages (`scikit-build-core`, `nanobind`, CMake, a CUDA-capable compiler). At Python *runtime* there are still no extra PyPI deps — only the same CUDA / driver / OpenMP requirements as the C++ engine.

## Requirements

- **OS:** Windows or Linux (x86_64)
- C++20 compiler — MSVC (`v145+`), GCC ≥ 11, or Clang ≥ 14 — plus **CMake ≥ 3.24**
- CUDA Toolkit (**13.x**; developed against v13.3) with matching NVIDIA driver
- NVIDIA GPU — GeForce **20 / 30 / 40 / 50** (`sm_75` / `sm_80`+`sm_86` / `sm_89` / `sm_120`). CUDA 13 floor is Turing.
- OpenMP (libgomp / LLVM OpenMP / MSVC OpenMP)
- **Python ≥ 3.10** (only for the Python package)
- Data for the SERA demo only (not included): `SERA-Data/sera_best_subset` (HF Arrow) and/or `*.jsonl`

## Python (pip)

Needs CUDA toolkit on the **build** machine (compiles the same C++/CUDA core) and at **runtime** (cuBLASLt DLLs / `.so`). No extra PyPI runtime deps.

On Windows, `import sentinel` auto-adds `$CUDA_PATH\bin\x64` (and `bin`) via `os.add_dll_directory` so Python 3.8+ can load `cublasLt64_*.dll`. Set `CUDA_PATH` if the toolkit is not under the default `Program Files\NVIDIA GPU Computing Toolkit\CUDA` layout.

```bash
# From repo root (editable / local wheel)
pip install .

# Or editable while hacking bindings (needs build deps already installed):
pip install -e . --no-build-isolation

python examples/python/train_tiny.py
python examples/python/generate.py tiny_demo.snlm "the cat"
```

```python
import sentinel as S

tok = S.BPETokenizer()
tok.train(["the cat sat on the mat", "cuda packs batches"], vocab_size=256)
data = S.LanguageModelDataset.build(["the cat sat on the mat"], tok, maximum_token_count=48)

model = S.LanguageModel(tok.vocab_size, embedding_dim=64, maximum_position_count=64, learning_rate=3e-3)
if S.cuda_available():
    model.enable_cuda()
    model.set_prefer_flash_attention(True)
    model.enable_cuda_train()

model.train(data, epochs=4, batch_size=4)
model.save_safetensors("toy.safetensors")
print(tok.decode(model.generate(tok.encode("the"), new_token_count=16)))
```

Override CUDA arches (e.g. faster local build for your GPU only):

```bash
pip install . -C cmake.define.SENTINEL_CUDA_ARCHITECTURES=native
# or a subset:
pip install . -C cmake.define.SENTINEL_CUDA_ARCHITECTURES="75;86;89;120"
```

### Publishing wheels (GitHub → PyPI)

Workflow: [`.github/workflows/publish.yml`](.github/workflows/publish.yml). Builds **sdist** + **cp310–cp312** wheels for **Linux** and **Windows** (CUDA 13 toolkit on the runner; fat binary arches). Publishes on a GitHub **Release**, or via *workflow_dispatch* with `publish_to_pypi=true`.

One-time setup:

1. On [PyPI Trusted Publishers](https://pypi.org/manage/account/publishing/), add publisher for project **`sentinel-lm`**: repo + workflow `publish.yml` + environment `pypi`.
2. In GitHub → Settings → Environments, create **`pypi`** (optional protection rules).
3. Cut a release (or run the workflow manually).

Wheels do **not** bundle the CUDA runtime — installers still need CUDA 13 + a matching driver, same as the C++ engine.

## Build & run (C++)

### CMake (recommended)

Builds **`libsentinel`** (static by default), the **`sentinel`** demo harness, and C++ examples.

**Linux**

```bash
# CUDA on PATH, e.g. export PATH=/usr/local/cuda/bin:$PATH
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"

# Tiny end-to-end (no dataset)
./build/bin/sentinel_train_tiny tiny_demo.snlm
./build/bin/sentinel_generate tiny_demo.snlm "the cat"

# Full harness (cwd matters for SERA paths)
cd sentinel && ../build/bin/sentinel
```

**Windows**

```bat
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j
build\bin\sentinel_train_tiny.exe tiny_demo.snlm
build\bin\sentinel_generate.exe tiny_demo.snlm "the cat"
cd sentinel
..\build\bin\sentinel.exe
```

Useful options:

| Option | Default | Meaning |
| ------ | ------- | ------- |
| `SENTINEL_CUDA_ARCHITECTURES` | `75;80;86;89;120` | Fat binary arches (`native` = host GPU only) |
| `SENTINEL_BUILD_SHARED` | `OFF` | Shared library instead of static |
| `SENTINEL_BUILD_DEMO` | `ON` | `sentinel` harness from `main.cpp` |
| `SENTINEL_BUILD_EXAMPLES` | `ON` | `sentinel_train_tiny`, `sentinel_generate` |
| `SENTINEL_BUILD_PYTHON` | `OFF` (`ON` via pip) | nanobind module `sentinel._core` |

### Install / consume from another CMake project

```bash
cmake --install build --prefix /path/to/prefix
```

```cmake
find_package(Sentinel 0.1 REQUIRED)
target_link_libraries(my_app PRIVATE Sentinel::sentinel)
# #include "NeuralNet/Network/LanguageModel.hpp"
```

### MSBuild / Visual Studio (Windows only)

Monolithic `sentinel.vcxproj` still builds the harness only (no separate lib). Prefer CMake for library + examples.

```bat
msbuild sentinel\sentinel.vcxproj /p:Configuration=Release /p:Platform=x64
cd sentinel
..\x64\Release\sentinel.exe
```

### Demo harness flags (`main.cpp`)

| Flag                     | Purpose                                                         |
| ------------------------ | --------------------------------------------------------------- |
| `runSmallSuite`          | Smokes + parities + speed 768 + ~100M probe (no corpus / no 4B) |
| `runScale100M`           | Full multi-epoch ~100M train on `sera_scale.jsonl`              |
| `runMuonThroughputProbe` | Adam vs Muon × ckpt tok/s table                                 |
| `runScale4BTrainStep`    | One packed ~4B train step (FP16w + host SGD)                    |
| `runSpeedBench` / others | Focused benches                                                 |

Default SERA demo (when suite flags are off): Arrow corpus, ~8×768. Working directory must be `sentinel/` for relative `../SERA-Data/...`.

## Layout

| Path                   | Role                                                                 |
| ---------------------- | -------------------------------------------------------------------- |
| `CMakeLists.txt`       | `Sentinel::sentinel` library, demo, examples, install/export         |
| `pyproject.toml`       | pip package `sentinel-lm` (scikit-build-core + nanobind)             |
| `python/sentinel/`     | Python package + nanobind `_core` (import name still `sentinel`)     |
| `sentinel/`            | C++/CUDA engine sources (`NeuralNet/`, demo `main.cpp`, …)           |
| `examples/`            | C++ + `examples/python/` — no external data                          |
| `cmake/`               | `find_package(Sentinel)` package config                              |
| `sentinel/NeuralNet/`  | Engine: Network, Cuda, Layers, Data, Tokenizer, …                    |
| `sentinel/main.cpp`    | Smokes / scale harness / SERA demo                                   |

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
model.saveSafeTensors("run.safetensors");   // weights only, HF-compatible container
model.loadSafeTensors("run.safetensors");   // or loadCheckpoint("*.safetensors")
```

Checkpoint files:
- **`.snlm`** — native `SNLM` (weights + optional Adam/Muon)
- **`.safetensors`** — zero-dep F32 safetensors export (`token_embedding.weight`, `blocks.{i}.attn.*`, `ffn.*`, `final_norm.weight`, `lm_head.*`); metadata carries arch dims / `tie_embedding`

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
| Mode       | FP16 GPU weights, host FP32 masters/grads, **host SGD** (no Adam moments), Full ckpt, flash |
| Host RAM   | ~15 GiB masters+grads (Adam `m`/`v` would roughly double that)                                |
| Full step  | **~320–330** tok/s (accumulate alone ~380–420)                                                |
| Bottleneck | Full-ckpt bwd recompute + end-of-step H2D; async Grad-D2H is mostly overlapped                |

Target ≥600 tok/s full-step is not yet reached on this offload+Full-ckpt path.

Smaller smoke: `CudaLanguageModel::runConsumerVramDemo()` (~8k/256/4L).

## Verify

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./build/bin/sentinel_train_tiny
./build/bin/sentinel_generate tiny_demo.snlm "the cat"
# Optional: in main.cpp set runSmallSuite=true, then:
# cd sentinel && ../build/bin/sentinel
```

Expect tiny examples to run without external data. Full suite: smokes/parities OK, **speed 768 ~18–20k** tok/s, **scale-100M probe ~25k** (Adam ckpt=off, WMMA flash).

FlashAttention `headDim=64` / tiles `64×64` uses CUDA WMMA Tensor Cores (toolkit only — no cuDNN). Toggle `runWmmaFaVerify` in `main.cpp` for a quick parity + probe.
