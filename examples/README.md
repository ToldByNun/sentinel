# Examples

Minimal programs that use the **library API** (not the `main.cpp` harness). No external dataset required unless you pass your own JSONL.

API reference: [docs/python.md](../docs/python.md) · [docs/cpp.md](../docs/cpp.md) · [docs/huggingface.md](../docs/huggingface.md) · [docs/gguf.md](../docs/gguf.md)

## Python

| Script | What it does |
| ------ | ------------ |
| [`python/train_tiny.py`](python/train_tiny.py) | In-memory toy corpus → train → `.snlm` + `.safetensors` |
| [`python/train_from_config.py`](python/train_from_config.py) | Size model from [`configs/tiny.json`](configs/tiny.json) → train → config + weights |
| [`python/custom_train_loop.py`](python/custom_train_loop.py) | Mid-level ops: `forward` / `accumulate_example` / `Adam` (no `train()`) |
| [`python/custom_layers_demo.py`](python/custom_layers_demo.py) | Attention / FFN / Spulse / SafeTensors bindings smoke |
| [`python/generate.py`](python/generate.py) | Load checkpoint (or train toy) → sample |
| [`python/train_jsonl.py`](python/train_jsonl.py) | Load JSONL texts → train → save weights |
| [`python/train_chunks.py`](python/train_chunks.py) | Stream JSONL/Arrow via `LanguageModelChunkSource` + `train_chunks` / `iter_train_chunks` |
| [`python/finetune_hf.py`](python/finetune_hf.py) | HF import → JSONL fine-tune → `save_huggingface` export (`--demo` offline stub) |
| [`python/gguf_roundtrip.py`](python/gguf_roundtrip.py) | `save_gguf` → `load_gguf` roundtrip |
| [`python/ci_host_smoke.py`](python/ci_host_smoke.py) | Host-only CI gate (import, train, streaming, config, HF/GGUF export) — no GPU |

```bash
pip install -e . --no-build-isolation   # from repo root
python examples/python/train_tiny.py
python examples/python/train_from_config.py --config examples/configs/tiny.yaml
python examples/python/custom_train_loop.py
python examples/python/custom_layers_demo.py
python examples/python/generate.py tiny_demo.snlm "the cat"
python examples/python/train_jsonl.py examples/data/sample.jsonl --out run.safetensors
python examples/python/train_chunks.py examples/data/sample.jsonl --out chunks_demo.snlm
python examples/python/finetune_hf.py --demo --out hf_demo_out
python examples/python/gguf_roundtrip.py --out toy.gguf
# python examples/python/finetune_hf.py /path/to/hf_model examples/data/sample.jsonl --out hf_finetuned
```

### Native model configs

[`configs/`](configs/) — `format: "sentinel-model"` JSON/YAML (arch + optional `weights`):

| File | Shape |
| ---- | ----- |
| [`tiny.json`](configs/tiny.json) / [`tiny.yaml`](configs/tiny.yaml) | 2×64 toy |
| [`base-768.json`](configs/base-768.json) | 12×768 / vocab 32k |

Load with `LanguageModel.from_config(...)` / `load_sentinel_model`. See [docs/python.md](../docs/python.md).

### JSONL shape

**In-memory demos** (`train_jsonl.py`, `finetune_hf.py`) accept each non-empty line as either:

- a JSON object with a string field `text`, `content`, or `problem_statement`, or
- a raw text line

**Streaming** (`LanguageModelChunkSource` / `JsonlLoader`) accepts JSON string fields **`problem_statement`**, **`text`**, or **`content`**. Example:

```python
source = S.LanguageModelChunkSource("corpus.jsonl", maximum_token_count=512)
sample = source.prepare_tokenizer_sample(2000)
tok.train(sample, vocab_size=8000)
source.set_tokenizer(tok)   # BPETokenizer only
source.materialize()
model.train_chunks(source, epochs=1)
# or: for chunk in source.iter_train_chunks(): model.train_step(chunk.examples)
```

See [`python/train_chunks.py`](python/train_chunks.py).

Keep `{stem}.sbpe` next to checkpoints for generate.

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
