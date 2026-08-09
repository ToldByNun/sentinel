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
#include "NeuralNet/Tokenizer/HfTokenizer.hpp"
#include "NeuralNet/IO/SentinelModelConfig.hpp"
#include "NeuralNet/IO/HuggingFaceResolve.hpp"
#include "NeuralNet/Optimizers/Adam.hpp"
#include "NeuralNet/Optimizers/Spulse.hpp"
#include "NeuralNet/Cuda/CudaSbao.hpp"
#include "NeuralNet/Cuda/CudaMatmul.hpp"   // CudaMatmul::isAvailable()
```

Sources live under [`sentinel/NeuralNet/`](../sentinel/NeuralNet/). The `sentinel` executable from `main.cpp` is a **harness**, not the API contract.

HuggingFace checkpoint import / export: **[huggingface.md](huggingface.md)**.

---

## Quick pattern

```cpp
BPETokenizer tok;
tok.train(corpus, /*vocabSize=*/256);
auto train = LanguageModelDataset::build(corpus, tok, /*maxTokens=*/48, /*oneHot=*/false);

LanguageModel model(tok.vocabSize(), /*embed=*/64, /*maxPos=*/64, Adam(3e-3f), /*blocks=*/2, /*heads=*/4, /*intermediateSize=*/0);
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

### `SpulseCoverage` / `SpulseMomentumStorage` (`Optimizers/Spulse.hpp`)

Also included via `Cuda/CudaSPULSE.hpp`.

| `SpulseCoverage` | Meaning |
| ---------------- | ------- |
| `Hybrid` | Hidden 2D block weights; Adam keeps embed/norms/biases/head |
| `Full` | All trainable params; Adam idle while SPULSE is on |

| `SpulseMomentumStorage` | Meaning |
| ----------------------- | ------- |
| `Fp32` | Default / best parity |
| `Fp16` | ~2× less VRAM for `u` |
| `Int8` | ~4× less VRAM (absmax blocks) |

SPULSE is an optimizer (`setCudaPreferSpulse` / `setCudaSpulseCoverage`); SBAO remains residency policy.
On the host fused-half path, Hybrid keeps momentum `u` on the GPU by default and downloads half-precision deltas (HostSGD-shaped host apply); Full still applies norms/biases/embed/head on device. `CudaSpulse::hostLightweight` is a Hybrid VRAM fallback (drops device `u`; forced off for Full) — there is no `LanguageModel` setter for it.
Device `u` storage: `setCudaSpulseMomentumStorage`.

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

## `HuggingFace::Tokenizer`

Header: `Tokenizer/HfTokenizer.hpp`

| Method | Notes |
| ------ | ----- |
| `load(pathOrDirectory)` | static; `tokenizer.json` file or HF model dir |
| `encode(text, addSpecialTokens=true)` | |
| `decode(ids, skipSpecialTokens=true)` | |
| `vocabSize` / `bosTokenId` / `eosTokenId` / `padTokenId` / `unkTokenId` | specials `-1` when absent |
| `isLoaded` / `ignoreMerges` | |

ByteLevel BPE only — see [huggingface.md](huggingface.md).

---

## `LanguageModelDataset` / `LanguageModelExample`

Header: `Data/LanguageModelDataset.hpp`

| API | Notes |
| --- | ----- |
| `LanguageModelDataset::build(texts, tok, maxTokens=0, buildOneHot=true)` | `BPETokenizer` or `HuggingFace::Tokenizer`; skip len &lt; 2; optional truncate |
| `fromTokenIds` / `makeOneHotSequence` | Lower-level helpers |
| `size()` / `totalPredictionCount()` | |
| `examples` / `vocabularySize` | Public fields |

Device train paths typically use `buildOneHot=false` (token-id CE).

Public host step helpers on `LanguageModel.hpp`: `LanguageModelCache`, `LanguageModelGradients` (`zerosFrom`, `zeroInPlace`, `addInPlace`, `scaleInPlace`) for `accumulateExample` / `applyGradients`.

---

## `LanguageModelChunkSource` (streaming)

Header: `Data/LanguageModelChunkSource.hpp` — also wrapped in Python (`LanguageModelChunkSource` + `train_chunks`).

Streams `.jsonl` or HF Arrow dirs via `createTextRowReader`. JSONL string fields: **`problem_statement`**, **`text`**, or **`content`** (see `JsonlLoader`).

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
source.setTokenizer(&tok);   // BPETokenizer*; must outlive source
source.materialize();
model.train(source, /*epochs=*/2, /*logEvery=*/1, /*batch=*/32, /*accum=*/4);
```

| Method | Notes |
| ------ | ----- |
| `setTokenizer` | Required before materialize / encode (`BPETokenizer*`) |
| `prepareTokenizerSample` | First N train-hash texts; rewinds |
| `materialize` / `prepareTestReservoir` | One corpus pass → cached token ids |
| `rewindTrain` / `sortTrainByLength` | Epoch iteration / length sort |
| `nextTrainChunk` / `fillTrainDataset` | Chunk or full copy |
| `testDataset` | Reservoir after materialize |
| `filePath` / `chunkExampleCount` / `trainRatio` | |
| `trainExampleCount` / `trainPredictionCount` / `isMaterialized` / `isTrainRow` | |

Related: `Data/TextRowReader.hpp`, `JsonlLoader.hpp` (`load` / `tryParseLine` / `sourceToLabel`, `CorpusRow`), `ArrowChunkReader.hpp`.

---

## `LanguageModel`

Header: `Network/LanguageModel.hpp`

| Method | Notes |
| ------ | ----- |
| ctor `(vocab, embed, maxPos, Adam, blocks=2, heads=4, intermediateSize=0, ropeTheta=10000, useBias=true, kvHeadCount=-1)` | `intermediateSize<=0` → legacy `(2*embed*4)/3` SwiGLU width; `useBias=false` → fixed-zero FFN/`lm_head` biases; `kvHeadCount<=0` → MHA |
| `intermediateSize()` / `ropeTheta()` / `useBias()` / `kvHeadCount()` | Arch mirrors |
| Public members | `tokenEmbedding`, `blocks`, `finalNorm`, `outputProjection`, `optimizer`, Adam states, `maximumPositionCount`, `tieEmbeddingProjection` |

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
| `setCudaPreferSpulse(bool)` / `setCudaSpulseCoverage(SpulseCoverage)` | SPULSE Hybrid or Full |
| `setCudaSpulseMomentumBeta` / `setCudaSpulseFastBeta` / `setCudaSpulseSlowBeta` | EMA knobs |
| `setCudaSpulseScaleClip(scaleMin, scaleMax)` | Clip dual-horizon scale |
| `setCudaSpulseMomentumStorage(SpulseMomentumStorage)` | Device `u` Fp32/Fp16/Int8 |
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
| `train(train, test, epochs, logEvery, batchSize=32, gradAccum=4)` | In-memory + optional test |
| `train(LanguageModelChunkSource&, epochs, logEvery=1, batchSize=32, gradAccum=4)` | Streaming epochs |
| `averageLoss` / `exampleLoss` | |
| `forward(tokenIds)` | Logits `Matrix` (device if enabled) |
| `generate(prompt, newTokens, temperature=1, topK=40, seed=42)` | New tokens only; `temperature <= 0` greedy |
| `accumulateExample` / `applyGradients` | Public host custom-loop step API (also bound in Python) |

### Checkpoints / I/O

| Method | Notes |
| ------ | ----- |
| `saveCheckpoint(path, includeOptimizer=true)` | Native `.snlm` |
| `loadCheckpoint(path)` | `.snlm` or routes `.safetensors` |
| `saveSafeTensors(path)` / `loadSafeTensors(path)` | Weights + metadata (`IO/SafeTensors.hpp`) |
| `loadSafeTensors(const SafeTensors::File&)` | In-memory file (load only) |
| `loadHuggingFace(dir, lr=3e-4)` | Static: local HF dir → size + remap weights; see [huggingface.md](huggingface.md) |
| `loadGguf(path, lr=3e-4)` | Static: GGUF file → size + remap weights; see [gguf.md](gguf.md) |
| `fromSentinelConfig` / `loadSentinelModel` | Static: native `sentinel-model` JSON/YAML; optional `weights` |
| `sentinelConfig` / `saveSentinelConfig` | Snapshot / write native model config |
| `saveHuggingFace(dir, modelType="llama", tokenizerSource="", weightFormat="safetensors")` | Export Transformers-compatible dir. `weightFormat`: `safetensors` \| `bin` \| `both` |
| `saveGguf(path, architecture="llama")` | Export GGUF v3 F32 (`llama` / `mistral` / `qwen2`) |

Safetensors names follow HF-style keys (`token_embedding.weight`, `blocks.{i}.attn.*`, `ffn.*`, `final_norm.weight`, `lm_head.*`).

### Probes

| Method | Notes |
| ------ | ----- |
| `probeCudaPackedTrainTokensPerSecond(seq, warmup=3, timed=8)` | Synthetic packed tok/s |
| `probeCudaTrainStepProfile(seq, warmup=2, timed=4)` | Prints fwd/bwd/opt breakdown |

Unset env `SENTINEL_PHASE_TRACE` when quoting throughput.

---

## Native `sentinel-model` config

Header: `IO/SentinelModelConfig.hpp` (`namespace SentinelModel`)

| Piece | Notes |
| ----- | ----- |
| `Config` fields | `format` (`"sentinel-model"`), `vocabSize`, `embeddingDim`, `maxPosition`, `blockCount`, `headCount`, `kvHeadCount`, `intermediateSize`, `ropeTheta`, `useBias`, `tieEmbedding`, `rmsNormEps`, `learningRate`, `weights` |
| `parseConfigJson` / `parseConfigYaml` / `parseConfigText` | Flat YAML only |
| `loadConfig` / `saveConfig` | File or directory (`model.json`) |
| `serializeConfigJson` / `serializeConfigYaml` | |
| `resolveWeightsPath` / `configDirectory` | Relative weights next to config |
| `LanguageModel::fromSentinelConfig` / `loadSentinelModel` | Build (+ optional weights) |

Examples: [`examples/configs/`](../examples/configs/) (`tiny.json`, `tiny.yaml`, `base-768.json`). Smoke: `SENTINEL_MODEL_CONFIG_SMOKE=1`.

---

## Related headers (library)

| Header | Role |
| ------ | ---- |
| `Optimizers/Adam.hpp` / `SGD.hpp` | Host Adam (+ `MuonState`) / SGD |
| `Optimizers/Spulse.hpp` | `SpulseCoverage`, `SpulseMomentumStorage`, `SpulseState` |
| `Cuda/CudaSPULSE.hpp` | Device / host SPULSE (`CudaSpulse`) |
| `IO/SafeTensors.hpp` | Weight file — **load** F32/BF16/F16 → host F32; **save** F32 |
| `IO/SentinelModelConfig.hpp` | Native `format: "sentinel-model"` JSON/YAML |
| `IO/PytorchStateDict.hpp` | Modern torch ZIP state-dict — **load** F32/F16/BF16 → host F32; **save** F32 (no libtorch; zlib) |
| `IO/HuggingFaceConfig.hpp` | Minimal `config.json` parse/serialize; allowlist `llama`/`mistral`/`qwen2` |
| `IO/HuggingFaceWeights.hpp` | HF↔Sentinel tensor remap + shard index; safetensors + `.bin` |
| `IO/HuggingFaceResolve.hpp` | Hub / URL / local path → local directory (`resolveModelDirectory`) — **not** called by `loadHuggingFace` |
| `IO/Gguf.hpp` | GGUF v3 load/save + remap (`loadConfig` / `loadMappedWeights` / `save`) |
| `Tokenizer/HfTokenizer.hpp` | HF `tokenizer.json` ByteLevel BPE |
| `Data/TextRowReader.hpp` / `JsonlLoader.hpp` / `ArrowChunkReader.hpp` | Corpus I/O |
| `Data/ClassificationDataset.hpp` / `Network/Sequential.hpp` | Small classifier stack |
| `Cuda/CudaMatmul.hpp` | `CudaMatmul::isAvailable()` |
| `Cuda/CudaSbao.hpp` | SBAO policy + fused-half offload helpers |
| `Layers/*` | Block primitives |

---

## Install layout

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
cmake --install build --prefix /path/to/prefix
```

| Option | Default | Meaning |
| ------ | ------- | ------- |
| `SENTINEL_CUDA_ARCHITECTURES` | `75;80;86;89;120` | Fat binary (`native` = host GPU) |
| `SENTINEL_BUILD_SHARED` | `OFF` | Shared instead of static |
| `SENTINEL_BUILD_DEMO` | `ON` | `sentinel` harness (`main.cpp`) |
| `SENTINEL_BUILD_EXAMPLES` | `ON` | `sentinel_train_tiny`, `sentinel_generate` |
| `SENTINEL_BUILD_PYTHON` | `OFF` (`ON` via pip / `SKBUILD`) | nanobind `sentinel._core` |
| `SENTINEL_INSTALL` | `ON` | Install / export rules |

Install destinations: library → `lib/`, headers `NeuralNet/**/*.hpp` → include root, CMake package → `lib/cmake/Sentinel`, binaries `sentinel` / `sentinel_train_tiny` / `sentinel_generate` when enabled. Pip/`SKBUILD` forces demo+examples OFF and Python ON.

Also see root [README](../README.md).

---

## Harness smokes

Focused early-exit env gates on the `sentinel` demo binary (`main.cpp`):

| Env | What |
| --- | ---- |
| `SENTINEL_SAFETENSORS_HALF_SMOKE=1` | BF16/F16 SafeTensors load |
| `SENTINEL_PYTORCH_BIN_SMOKE=1` | ZIP state-dict roundtrip |
| `SENTINEL_INTERMEDIATE_SIZE_SMOKE=1` | Explicit FFN width |
| `SENTINEL_ROPE_THETA_SMOKE=1` | RoPE base |
| `SENTINEL_BIAS_POLICY_SMOKE=1` | `useBias` |
| `SENTINEL_KV_HEAD_COUNT_SMOKE=1` | LM GQA field |
| `SENTINEL_GQA_HOST_SMOKE=1` / `SENTINEL_GQA_CUDA_SMOKE=1` | Host / CUDA GQA |
| `SENTINEL_MODEL_CONFIG_SMOKE=1` | Native `sentinel-model` config |
| HF smokes | See [huggingface.md](huggingface.md#smokes-harness) |

Optional fixture: `SENTINEL_PYTORCH_BIN_FIXTURE` for the PyTorch ZIP smoke. Unset `SENTINEL_PHASE_TRACE` when quoting tok/s.

---

## Python parity

Train / generate / checkpoint knobs, streaming `LanguageModelChunkSource` (`train_chunks`), SafeTensors / GGUF helpers, mid-level layers/optimizers, and native config are also on the [Python API](python.md). Still C++-only: raw `ArrowChunkReader`, `HuggingFaceResolve`, low-level HF/GGUF config/weight-map helpers, `PytorchStateDict`, and harness smoke entry points.
