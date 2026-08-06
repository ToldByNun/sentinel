# Examples

Minimal programs that use the **library API** (not the `main.cpp` harness). No external dataset required unless you pass your own JSONL.

API reference: [docs/python.md](../docs/python.md) · [docs/cpp.md](../docs/cpp.md)

## Python

| Script | What it does |
| ------ | ------------ |
| [`python/train_tiny.py`](python/train_tiny.py) | In-memory toy corpus → train → `.snlm` + `.safetensors` |
| [`python/generate.py`](python/generate.py) | Load checkpoint (or train toy) → sample |
| [`python/train_jsonl.py`](python/train_jsonl.py) | Load JSONL texts → train → save weights |

```bash
pip install -e . --no-build-isolation   # from repo root
python examples/python/train_tiny.py
python examples/python/generate.py tiny_demo.snlm "the cat"
python examples/python/train_jsonl.py examples/data/sample.jsonl --out run.safetensors
```

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
