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

HuggingFace checkpoint import / export (allowlist, weight map, tokenizer, VRAM): **[huggingface.md](huggingface.md)**.

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

### `SpulseCoverage` (`Cuda/CudaSPULSE.hpp`)

| Value | Meaning |
| ----- | ------- |
| `Hybrid` | Hidden 2D block weights; Adam keeps embed/norms/biases/head |
| `Full` | All trainable params; Adam idle while SPULSE is on |

SPULSE is an optimizer (`setCudaPreferSpulse` / `setCudaSpulseCoverage`); SBAO remains residency policy.
On the host fused-half path, Hybrid keeps momentum `u` on the GPU by default and downloads half-precision deltas (HostSGD-shaped host apply); Full still applies norms/biases/embed/head on device. Set `CudaSpulse::hostLightweight` only as a Hybrid VRAM fallback (drops device `u`; forced off for Full).
Device `u` storage: `SpulseMomentumStorage::{Fp32,Fp16,Int8}` via `setCudaSpulseMomentumStorage` (Fp16/Int8 trade parity for VRAM).

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
| `LanguageModelDataset::build(texts, tok, maxTokens=0, buildOneHot=true)` | `BPETokenizer` or `HuggingFace::Tokenizer`; skip len &lt; 2; optional truncate |
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

| Method | Notes |
| ------ | ----- |
| ctor `(vocab, embed, maxPos, Adam, blocks=2, heads=4, intermediateSize=0, ropeTheta=10000, useBias=true, kvHeadCount=-1)` | `intermediateSize<=0` → legacy `(2*embed*4)/3` SwiGLU width; `useBias=false` → fixed-zero FFN/`lm_head` biases; `kvHeadCount<=0` → MHA |
| `intermediateSize()` | gate/up rows |
| `ropeTheta()` | RoPE base (HF `rope_theta`) |
| `useBias()` | trainable FFN/`lm_head` biases |
| `kvHeadCount()` | K/V heads (HF `num_key_value_heads`) |

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
| `saveSafeTensors` / `loadSafeTensors` | Weights + metadata (`IO/SafeTensors.hpp`); overload accepts in-memory `SafeTensors::File` |
| `loadHuggingFace(dir, lr=3e-4)` | Static: parse HF `config.json`, size model, remap/load safetensors or modern `pytorch_model.bin`; see [huggingface.md](huggingface.md) |
| `fromSentinelConfig` / `loadSentinelModel` | Static: native `sentinel-model` JSON/YAML (`IO/SentinelModelConfig.hpp`); optional `weights` |
| `sentinelConfig` / `saveSentinelConfig` | Snapshot / write native model config |
| `saveHuggingFace(dir, modelType="llama", tokenizerSource="", weightFormat="safetensors")` | Export Transformers-compatible dir (`config.json` + `model.safetensors` and/or `pytorch_model.bin`; optional tokenizer copy). `weightFormat`: `safetensors` \| `bin` \| `both` |

Focused smoke: `SENTINEL_HF_ROUNDTRIP_SMOKE=1` — stub HF dir → import + `HfTokenizer` encode + 1 host train step + generate (finite gate).

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
| `IO/SentinelModelConfig.hpp` | Native `format: "sentinel-model"` JSON/YAML (flat YAML); size + optional weights |
| `IO/PytorchStateDict.hpp` | Modern torch ZIP state-dict — **load** F32/F16/BF16 → host F32; **save** F32 (no libtorch; zlib) |
| `IO/HuggingFaceConfig.hpp` | Minimal `config.json` parse/serialize; allowlist `llama`/`mistral`/`qwen2`; rejects MoE / sliding-window / quantized |
| `IO/HuggingFaceWeights.hpp` | HF↔Sentinel tensor remap + shard index (`loadMappedWeights` / `saveDirectory`); safetensors + `.bin`; first family: Llama/Mistral-like names |
| `Tokenizer/HfTokenizer.hpp` | HF `tokenizer.json` ByteLevel BPE (`HuggingFace::Tokenizer`); `.sbpe` stays on `BPETokenizer` |
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

Most train/generate/checkpoint knobs are also on the [Python API](python.md). Gaps may include streaming `ChunkSource`, `forward` logits, and some fine-grained CUDA setters. Tokenizer **`.sbpe`** I/O is available on both sides; HF `tokenizer.json` is `HfTokenizer` / `HuggingFace::Tokenizer` — see [huggingface.md](huggingface.md).
