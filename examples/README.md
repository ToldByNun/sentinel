# Examples

Minimal programs that use the **library API** (not the `main.cpp` harness). No external dataset required unless you pass your own JSONL.

API reference: [docs/python.md](../docs/python.md) · [docs/cpp.md](../docs/cpp.md)

## Python

| Script | What it does |
| ------ | ------------ |
| [`python/train_tiny.py`](python/train_tiny.py) | In-memory toy corpus → train → `.snlm` + `.safetensors` |
| [`python/train_from_config.py`](python/train_from_config.py) | Size model from [`configs/tiny.json`](configs/tiny.json) → train → config + weights |
| [`python/generate.py`](python/generate.py) | Load checkpoint (or train toy) → sample |
| [`python/train_jsonl.py`](python/train_jsonl.py) | Load JSONL texts → train → save weights |
| [`python/finetune_hf.py`](python/finetune_hf.py) | HF import → JSONL fine-tune → `save_huggingface` export (`--demo` offline stub) |

```bash
pip install -e . --no-build-isolation   # from repo root
python examples/python/train_tiny.py
python examples/python/train_from_config.py --config examples/configs/tiny.yaml
python examples/python/generate.py tiny_demo.snlm "the cat"
python examples/python/train_jsonl.py examples/data/sample.jsonl --out run.safetensors
python examples/python/finetune_hf.py --demo --out hf_demo_out
# python examples/python/finetune_hf.py /path/to/hf_model examples/data/sample.jsonl --out hf_finetuned
```

### Native model configs

[`configs/`](configs/) — `format: "sentinel-model"` JSON/YAML (arch + optional `weights`). Load with `LanguageModel.from_config(...)` / `load_sentinel_model`. See [docs/python.md](../docs/python.md).

### JSONL shape (Python example)

Each non-empty line is either:

- a JSON object with a string field `text`, `content`, or `problem_statement`, or
- a raw text line

The C++ streaming path (`LanguageModelChunkSource` + `JsonlLoader`) currently expects **`problem_statement`** (SERA-style). The Python example accepts the common aliases above so small demos are less awkward.

Streaming / chunked epoch loops over large corpora use the C++ `LanguageModelChunkSource` API (and Python bindings if you exposed them). Keep `{stem}.sbpe` next to checkpoints for generate.

## C++

Built when `SENTINEL_BUILD_EXAMPLES=ON` (default for a normal CMake build):

| Target | Source |
| ------ | ------ |
| `sentinel_train_tiny` | [`train_tiny.cpp`](train_tiny.cpp) |
| `sentinel_generate` | [`generate.cpp`](generate.cpp) |

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/bin/sentinel_train_tiny tiny_demo.snlm
./build/bin/sentinel_generate tiny_demo.snlm "the cat"
```

## Sample data

[`data/sample.jsonl`](data/sample.jsonl) — a handful of lines for `train_jsonl.py`.
