# Changelog

All notable changes to this project are documented here.

## [Unreleased]

### Fixed

- CUDA packed train: epoch shuffle + **window-local** length packing (was global short→long curriculum) and Adam-step pack caps so effective batch matches `batch×accum`. SERA harness default LR `3e-4`, batch `32` (was `1e-3` / `64`) for stable descending train/test loss.

### Changed

- **Docs refreshed** to match the current public surface: Python mid-level ops, streaming `LanguageModelChunkSource` / `train_chunks`, SafeTensors helpers, `Sequential` / classification, `sentinel-model` config, SPULSE Full/Hybrid, HF import/export. Removed stale “ChunkSource C++-only” notes from README / examples / API pages.

### Added

- **Python mid-level ops bindings** (expanded): `Matrix`, activations/losses (`Softmax`/`SiLU`/`ReLU`/`CrossEntropy`/`MSE`), `UniformInit`, `Adam`/`AdamState`/`SGD`/`MuonState`, host `Spulse`/`SpulseState`, layers (`Embedding`/`Dense`/`Dropout`/`MeanPool`/`RMSNorm`/`RotaryEmbedding`/`CausalSelfAttention`/`FeedForward`/`TransformerBlock` + caches), LM step API (`forward`/`accumulate_example`/`apply_gradients`/`train_step`), streaming (`LanguageModelChunkSource`/`train_chunks`), SafeTensors I/O, `Sequential`/`ClassificationDataset`. Examples: [`custom_train_loop.py`](examples/python/custom_train_loop.py), [`custom_layers_demo.py`](examples/python/custom_layers_demo.py).
- **Native `sentinel-model` JSON/YAML config** (`IO/SentinelModelConfig.*`): `format: "sentinel-model"` with safetensors-aligned fields (`vocab_size`, `embedding_dim`, `max_position`, …, `rms_norm_eps`, `learning_rate`, optional `weights`). Flat YAML only. `LanguageModel::fromSentinelConfig` / `loadSentinelModel` / `sentinelConfig` / `saveSentinelConfig`; Python `SentinelModelConfig`, `from_config` / `load_sentinel_model`. Examples under [`examples/configs/`](examples/configs/) + [`examples/python/train_from_config.py`](examples/python/train_from_config.py). Smoke: `SENTINEL_MODEL_CONFIG_SMOKE=1`.
- Example [`examples/python/finetune_hf.py`](examples/python/finetune_hf.py): HF import → JSONL fine-tune → `save_huggingface` export (local dir or Hub via `huggingface_hub`; `--demo` offline stub). `LanguageModelDataset::build` / Python `build` also accept `HfTokenizer`.
- **HuggingFace export**: `LanguageModel::saveHuggingFace` / Python `save_huggingface` writes a Transformers-compatible directory (`config.json` + F32 `model.safetensors` and/or `pytorch_model.bin` with Llama/Mistral/Qwen2-style keys). `weight_format`: `safetensors` \| `bin` \| `both`. Optional `tokenizer_source_directory` copies tokenizer files. Helpers: `HuggingFace::serializeConfigJson` / `saveConfig`, `remapSentinelWeightsToHf` / `saveDirectory`. Smoke: `SENTINEL_HF_EXPORT_SMOKE=1`.
- **PyTorch `.bin` state-dict I/O** (no libtorch): `PytorchStateDict::load` / `save` for modern ZIP `torch.save` archives (F32 write; F32/F16/BF16 read → host F32). HF import falls back to `pytorch_model.bin` / index shards when safetensors is absent. Legacy non-zip pickle rejected. Smoke: `SENTINEL_PYTORCH_BIN_SMOKE=1`.
- **SPULSE** optimizer (opt-in): dual-horizon energy-scaled momentum (`set_prefer_spulse` / `setCudaPreferSpulse`). **Hybrid** (default): hidden 2D block weights + Adam on embed/norms/biases/head. **Full** (`set_spulse_coverage(Full)` / `SpulseCoverage::Full`): all trainable params; Adam idle. Logic lives in `CudaSPULSE.*`. GPU-resident path keeps full `u` on device. **Host fused-half**: Hybrid keeps `u` on GPU, bakes `delta = lr·s·û` into the grad buffer before half D2H, then HostSGD-shaped host axpy (same PCIe volume as HostSGD); Full still updates aux/embed/head on device. EMA momentum uses **Adam-style bias correction** (`û = u / (1-β^t)`) so cold-start steps match HostSGD magnitude. Kernels use **float4/half2** loads and fuse energy commit into the last grid block (no `<<<1,1>>>` launch). `hostLightweight` is a Hybrid VRAM fallback (no device `u`; forced off for Full). Device `u` storage selectable via `SpulseMomentumStorage` / `set_spulse_momentum_storage`: **Fp32** (default), **Fp16**, or **Int8** (absmax blocks) for VRAM experiments. Distinct from SBAO (policy). Harness: `runSpulseThroughputCompare` (GPU), `runSpulseHostQualityCompare` (HostSGD vs SPULSE-Host: tok/s **and** loss quality — speed without quality fails). Smoke: `runSpulseTrainSmokeDemo` covers Hybrid + Full.
- `BPETokenizer` persistence: binary `.sbpe` via `save` / `load` / `loadFrom` (C++ + Python)
- Examples write/load sibling `{stem}.sbpe` next to checkpoints
- SafeTensors load accepts **BF16** / **F16** (converts to host F32); save remains F32
- Explicit FFN `intermediate_size` on `FeedForward` / `TransformerBlock` / `LanguageModel` (+ safetensors metadata); `<=0` keeps legacy expand-4 width
- Configurable RoPE `rope_theta` (base) through Attention / Block / LM (+ safetensors metadata); default `10000`
- Optional FFN / `lm_head` bias via `use_bias` (default `true`); `false` keeps zero-shaped biases for CUDA, omits bias tensors on safetensors save, and skips Adam updates (HF arches without those biases)
- Host `CausalSelfAttention` **GQA** (`kvHeadCount`); MHA when `kvHeadCount == headCount`
- CUDA / flash attention GQA (dense Repeat-KV; flash via expanded K/V heads) + host/flash parity smoke
- `kv_head_count` on `TransformerBlock` / `LanguageModel` / Python (+ safetensors metadata); `<=0` → MHA
- Minimal HuggingFace `config.json` parser (`HuggingFace::loadConfig` / `parseConfigJson`); allowlist `llama` / `mistral` / `qwen2`; rejects MoE, sliding-window, quantized, unknown `model_type`
- HF→Sentinel weight map + sharded safetensors (`HuggingFace::loadMappedWeights`, Llama/Mistral-like layout family); respects `tie_word_embeddings` / `use_bias`
- `LanguageModel::loadHuggingFace` / Python `LanguageModel.load_huggingface` — config-driven arch + remapped weights from an HF model directory
- `HuggingFace::Tokenizer` / Python `HfTokenizer` — load HF `tokenizer.json` (ByteLevel BPE, Llama-3 `ignore_merges`); encode/decode + specials
- HF roundtrip smoke gate: import + encode + host microtrain + generate (finite logits/loss)
- Docs: [`docs/huggingface.md`](docs/huggingface.md) — supported HF arches, rejects, tokenizer, VRAM notes (+ links from Python/C++ API docs)

## [0.1.0] — 2026-08-06

First library-oriented release surface.

### Added

- Installable C++ library (`Sentinel::sentinel`) via CMake install / `find_package`
- Pip package `sentinel-lm` (import `sentinel`) with nanobind bindings for train / generate / checkpoints
- CUDA train path: packed batches, flash attention, FP16 AMP, int8 Adam, Muon, activation checkpointing
- **SBAO** (Sentinel Backend Adaptive Optimization): GpuInt8Adam, HostFusedHalfAdam, HostFusedHalfSgd
- Safetensors import/export (weights + arch metadata)
- Examples: C++ `train_tiny` / `generate`, Python `train_tiny` / `generate` / `train_jsonl`
- API docs under `docs/` (Python + C++)
- Paper benchmark harness under `benchmarks/`
- GitHub Actions publish workflow for CUDA wheels

### Notes

- Python v0.1 started with in-memory datasets (`LanguageModelDataset`). Streaming (`LanguageModelChunkSource` / `train_chunks`) and mid-level ops landed later — see **[Unreleased]**.
- Throughput numbers in the README are indicative (RTX 5070 Ti 16 GB); re-measure on your machine.
- SafeTensors half-load smoke: `SENTINEL_SAFETENSORS_HALF_SMOKE=1` when running the `sentinel` harness.
- Intermediate-size smoke: `SENTINEL_INTERMEDIATE_SIZE_SMOKE=1`.
- RoPE-theta smoke: `SENTINEL_ROPE_THETA_SMOKE=1`.
- Bias-policy smoke: `SENTINEL_BIAS_POLICY_SMOKE=1`.
- Host GQA smoke: `SENTINEL_GQA_HOST_SMOKE=1`.
- CUDA GQA smoke: `SENTINEL_GQA_CUDA_SMOKE=1`.
- LM `kv_head_count` smoke: `SENTINEL_KV_HEAD_COUNT_SMOKE=1`.
- HF config parse smoke: `SENTINEL_HF_CONFIG_SMOKE=1`.
- HF weight-map smoke: `SENTINEL_HF_WEIGHT_MAP_SMOKE=1`.
- HF import smoke: `SENTINEL_HF_IMPORT_SMOKE=1`.
- HF tokenizer smoke: `SENTINEL_HF_TOKENIZER_SMOKE=1`.
- HF roundtrip smoke: `SENTINEL_HF_ROUNDTRIP_SMOKE=1`.
- HF export smoke: `SENTINEL_HF_EXPORT_SMOKE=1`.
- PyTorch ZIP `.bin` smoke: `SENTINEL_PYTORCH_BIN_SMOKE=1`.
