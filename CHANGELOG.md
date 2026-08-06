# Changelog

All notable changes to this project are documented here.

## [Unreleased]

### Added

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

- Python v0.1 exposes **in-memory** datasets (`LanguageModelDataset`). Streaming JSONL/Arrow epochs are C++ (`LanguageModelChunkSource`) unless wrapped in bindings.
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
