# C++ API

Consume the installed library via CMake:

```cmake
find_package(Sentinel 0.1 REQUIRED)
target_link_libraries(my_app PRIVATE Sentinel::sentinel)
```

Includes are rooted at the NeuralNet tree, e.g.:

```cpp
#include "NeuralNet/Network/LanguageModel.hpp"
#include "NeuralNet/Data/LanguageModelDataset.hpp"
#include "NeuralNet/Data/LanguageModelChunkSource.hpp"
#include "NeuralNet/Tokenizer/BPETokenizer.hpp"
#include "NeuralNet/Optimizers/Adam.hpp"
#include "NeuralNet/Cuda/CudaSbao.hpp"
#include "NeuralNet/Cuda/CudaMatmul.hpp"   // CudaMatmul::isAvailable()
```

Sources live under [`sentinel/NeuralNet/`](../sentinel/NeuralNet/). The `sentinel` executable from `main.cpp` is a **harness**, not the API contract.

---

## Quick pattern

```cpp
BPETokenizer tok;
tok.train(corpus, /*vocabSize=*/256);
auto train = LanguageModelDataset::build(corpus, tok, /*maxTokens=*/48, /*oneHot=*/false);

LanguageModel model(tok.vocabSize(), /*embed=*/64, /*maxPos=*/64, Adam(3e-3f), /*blocks=*/2, /*heads=*/4);
if (CudaMatmul::isAvailable()) {
    model.enableCuda();
    model.setCudaPreferFlashAttention(true);
    model.enableCudaTrain();
}
LanguageModelDataset emptyTest;
model.train(train, emptyTest, /*epochs=*/4, /*logEvery=*/1, /*batch=*/4, /*accum=*/1);
model.saveSafeTensors("toy.safetensors");
```

Tiny binaries: [`examples/train_tiny.cpp`](../examples/train_tiny.cpp), [`examples/generate.cpp`](../examples/generate.cpp).

---

## Enums

### `ActivationCheckpointMode` (`LanguageModel.hpp`)

| Value | Meaning |
| ----- | ------- |
| `Off` | Retain activations |
| `Full` | Full recompute |
| `Selective` | Partial recompute |

`enableActivationCheckpointing(true)` maps to **Selective**; `false` → **Off**. Prefer `setActivationCheckpointMode` for explicit control.

### `SbaoMode` (`Cuda/CudaSbao.hpp`)

| Value | Meaning |
| ----- | ------- |
| `Auto` | Resolve at train enable |
| `GpuInt8Adam` | Resident int8 Adam |
| `HostFusedHalfAdam` | Fused-half D2H + host Adam |
| `HostFusedHalfSgd` | Fused-half D2H + host SGD |

`CudaSbao::select` / `resolveAndApply` / `modeName` are available for policy introspection. Prefer driving policy through `LanguageModel::setCudaPreferSbao` / `setCudaSbaoMode`.

---

## `BPETokenizer`

Header: `Tokenizer/BPETokenizer.hpp`

| Method | Notes |
| ------ | ----- |
| `train(const std::string&, int vocabSize)` | Single string |
| `train(const std::vector<std::string>&, int vocabSize)` | Corpus |
| `encode` / `decode` | |
| `vocabSize` / `unknownTokenId` / `tokenToId` / `idToToken` | |
| `isTrained()` | True after `train` / `load` |
| `save(path)` / `load(path)` | Binary **`.sbpe`** (magic `SBPE`, vocab + merges) |
| `loadFrom(path)` | static constructor |

Unknown pieces map to `<unk>`. Prefer a sibling `{stem}.sbpe` next to `.snlm` / `.safetensors`.

---

## `LanguageModelDataset` / `LanguageModelExample`

Header: `Data/LanguageModelDataset.hpp`

| API | Notes |
| --- | ----- |
| `LanguageModelDataset::build(texts, tok, maxTokens=0, buildOneHot=true)` | Skip len &lt; 2; optional truncate |
| `fromTokenIds` / `makeOneHotSequence` | Lower-level helpers |
| `size()` / `totalPredictionCount()` | |
| `examples` / `vocabularySize` | Public fields |

Device train paths typically use `buildOneHot=false` (token-id CE).

---

## `LanguageModelChunkSource` (streaming)

Header: `Data/LanguageModelChunkSource.hpp` — **not wrapped in Python**.

Streams `.jsonl` or HF Arrow dirs via `createTextRowReader`. JSONL rows use the **`problem_statement`** string field (SERA-style); see `JsonlLoader`.

```cpp
LanguageModelChunkSource source(
    path,
    /*maxChars=*/0,
    /*maxTokens=*/512,
    /*chunkExamples=*/32,
    /*trainRatio=*/0.9f,
    /*seed=*/42u,
    /*testReservoirCap=*/256);

auto sample = source.prepareTokenizerSample(2000);
tok.train(sample, vocabSize);
source.setTokenizer(&tok);
source.materialize();
model.train(source, /*epochs=*/2, /*logEvery=*/1, /*batch=*/32, /*accum=*/4);
```

| Method | Notes |
| ------ | ----- |
| `setTokenizer` | Required before materialize / encode |
| `prepareTokenizerSample` | First N train-hash texts; rewinds |
| `materialize` | One corpus pass → cached token ids |
| `rewindTrain` / `nextTrainChunk` | Epoch iteration |
| `fillTrainDataset` | Copy all train examples into a dataset |
| `testDataset` | Reservoir after materialize |

---

## `LanguageModel`

Header: `Network/LanguageModel.hpp`

Move-only. Construct with `Adam` (or pass learning rate via `Adam(lr)`).

### Lifecycle / device

| Method | Notes |
| ------ | ----- |
| `enableCuda()` | Host → device mirror for forward/generate |
| `enableCudaTrain()` | Packed device train; applies pack budget + SBAO resolve |
| `syncDevice()` | Re-upload host weights if mirror active |
| `cudaEnabled()` / `cudaTrainEnabled()` | |
| `parameterElementCount()` | Tied head shares embedding |

### Train knobs

| Method | Default intent |
| ------ | -------------- |
| `setCudaPreferFlashAttention(bool)` | Prefer on |
| `setCudaPreferMixedPrecision(bool)` | FP16 GEMMs + loss scale when large |
| `setCudaPreferInt8AdamMoments(bool)` | Low-VRAM Adam |
| `setCudaPreferMuon(bool)` / `setCudaMuonNsSteps(int)` | Hidden-weight Muon |
| `setCudaPreferCpuAdamOffload(bool)` | Host Adam / fused-half Adam |
| `setCudaPreferHostSgd(bool)` | Host SGD masters |
| `setCudaPreferSbao(bool)` / `setCudaSbaoMode(SbaoMode)` / `cudaSbaoModeResolved()` | Unified policy |
| `setActivationCheckpointMode` / `cudaActivationCheckpointMode` | Off / Selective / Full |
| `enableActivationCheckpointing(bool)` | Legacy bool → Selective/Off |
| `setCudaMaxPackedColumns(int)` / `cudaMaxPackedColumns()` | Manual pack; skips auto budget |
| `applyCudaVramPackBudget(fraction=0.70, safetyReserve=…)` | Auto pack from free VRAM |
| `setCudaLogitChunkRows(int)` | Chunked CE / LM-head |
| `setCudaPreferTrainGraph(bool)` | Graphs when ckpt Off |
| `setTieEmbeddingProjection(bool)` | Default on |
| `lmHeadWeight()` | Embedding when tied |

Set offload / SBAO preferences **before** `enableCudaTrain()` when pack and checkpoint mode must follow the large-model layout.

### Train / loss / generate

| Method | Notes |
| ------ | ----- |
| `train(dataset, epochs, logEvery=1)` | Host OpenMP path defaults |
| `train(train, test, epochs, logEvery, batchSize, gradAccum)` | In-memory + optional test |
| `train(LanguageModelChunkSource&, …)` | Streaming epochs |
| `averageLoss` / `exampleLoss` | |
| `forward(tokenIds)` | Logits `Matrix` (device if enabled) |
| `generate(prompt, newTokens, temperature=1, topK=40, seed=42)` | New tokens only; `temperature <= 0` greedy |

### Checkpoints / I/O

| Method | Notes |
| ------ | ----- |
| `saveCheckpoint(path, includeOptimizer=true)` | Native `.snlm` |
| `loadCheckpoint(path)` | `.snlm` or routes `.safetensors` |
| `saveSafeTensors` / `loadSafeTensors` | Weights + metadata (`IO/SafeTensors.hpp`) |

Safetensors names follow HF-style keys (`token_embedding.weight`, `blocks.{i}.attn.*`, `ffn.*`, `final_norm.weight`, `lm_head.*`).

### Probes

| Method | Notes |
| ------ | ----- |
| `probeCudaPackedTrainTokensPerSecond(seq, warmup=3, timed=8)` | Synthetic packed tok/s |
| `probeCudaTrainStepProfile(seq, …)` | Prints fwd/bwd/opt breakdown |

Unset env `SENTINEL_PHASE_TRACE` when quoting throughput.

---

## Related headers (library)

| Header | Role |
| ------ | ---- |
| `Optimizers/Adam.hpp` / `SGD.hpp` | Host optimizers |
| `IO/SafeTensors.hpp` | Weight file format — **load** F32/BF16/F16 → host F32; **save** F32 |
| `Data/TextRowReader.hpp` / `JsonlLoader.hpp` / `ArrowChunkReader.hpp` | Corpus I/O |
| `Cuda/CudaMatmul.hpp` | `CudaMatmul::isAvailable()` |
| `Cuda/CudaSbao.hpp` | SBAO policy + fused-half offload helpers |
| `Layers/*` | Block primitives (usually not app-facing) |

---

## Install layout

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
cmake --install build --prefix /path/to/prefix
```

Useful options: `SENTINEL_BUILD_SHARED`, `SENTINEL_CUDA_ARCHITECTURES`, `SENTINEL_BUILD_EXAMPLES`, `SENTINEL_BUILD_DEMO` — see root [README](../README.md).

---

## Python parity

Most train/generate/checkpoint knobs are also on the [Python API](python.md). Gaps may include streaming `ChunkSource`, `forward` logits, and some fine-grained CUDA setters. Tokenizer **`.sbpe`** I/O is available on both sides.
