# Changelog

All notable changes to this project are documented here.

## [Unreleased]

### Added

- `BPETokenizer` persistence: binary `.sbpe` via `save` / `load` / `loadFrom` (C++ + Python)
- Examples write/load sibling `{stem}.sbpe` next to checkpoints
- SafeTensors load accepts **BF16** / **F16** (converts to host F32); save remains F32
- Explicit FFN `intermediate_size` on `FeedForward` / `TransformerBlock` / `LanguageModel` (+ safetensors metadata); `<=0` keeps legacy expand-4 width

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
