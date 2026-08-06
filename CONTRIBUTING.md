# Contributing

Thanks for hacking on Sentinel. Keep the library usable: small diffs, measured train-path changes, and a clear public API.

## Setup

**C++**

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSENTINEL_CUDA_ARCHITECTURES=native
cmake --build build -j
./build/bin/sentinel_train_tiny
```

**Python (editable)**

```bash
# CUDA + CMake on PATH (VS Native Tools on Windows)
pip install -e . --no-build-isolation
python examples/python/train_tiny.py
```

## Public API vs harness

| Use | Avoid treating as API |
| --- | --------------------- |
| `NeuralNet/Network/LanguageModel.hpp` + Data/Tokenizer/IO headers | One-off flags inside `sentinel/main.cpp` |
| `python/sentinel/` + bindings in `python/sentinel/_core.cpp` | Ad-hoc probe scripts unless documented |

If you add a user-facing knob, expose it on `LanguageModel` (and Python if it belongs in v0.1) and mention it in the README API table.

## CUDA / throughput changes

Train-path edits need evidence:

1. State the hypothesis and expected delta (e.g. “algo cache hit → ≥3% mean tok/s”).
2. Measure with a fixed probe (`benchmarks/_lib/probe_4b_safe.py` or `probe_cuda_packed_train_tokens_per_second`) — multi-run mean; phase trace only when diagnosing.
3. **Keep** only if the gate is met; otherwise **revert** in the same change set.

Do not stack speculative stream / D2H / kernel rewrites without a keep/revert result.

Unset `SENTINEL_PHASE_TRACE` when quoting tok/s (trace skews wall time).

## Docs

- User entry point: [`README.md`](README.md)
- API reference: [`docs/`](docs/README.md) — [Python](docs/python.md), [C++](docs/cpp.md)
- Examples: [`examples/README.md`](examples/README.md)
- Benchmarks: [`benchmarks/README.md`](benchmarks/README.md)
- Release notes: [`CHANGELOG.md`](CHANGELOG.md)

When you change a public method, update the matching docs page in the same PR.

## Style

- Match neighboring code (C++/CUDA in `sentinel/NeuralNet/`, Python examples under `examples/python/`).
- Prefer focused PRs; no drive-by renames or unrelated formatting.
- Do not commit build trees (`build/`), local checkpoints, or datasets.
