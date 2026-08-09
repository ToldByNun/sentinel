# GGUF import / export

Sentinel can **export** and **import** [GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) v3 files for llama.cpp-style causal LMs — same allowlist as HuggingFace (`llama` / `mistral` / `qwen2`).

Weights remapped into Sentinel’s internal tensor names, then loaded via `LanguageModel::loadSafeTensors`. No third-party GGUF dependency.

## Python

```python
import sentinel as S

model = S.LanguageModel(
    vocabulary_size=64,
    embedding_dim=32,
    maximum_position_count=64,
    block_count=2,
    head_count=4,
    intermediate_size=64,
    use_bias=False,
    kv_head_count=2,
)
model.set_tie_embedding(True)
model.save_gguf("toy.gguf", architecture="llama")

loaded = S.LanguageModel.load_gguf("toy.gguf", learning_rate=3e-4)
```

Example: [`examples/python/gguf_roundtrip.py`](../examples/python/gguf_roundtrip.py).

| Method | Notes |
| ------ | ----- |
| `LanguageModel.save_gguf(path, architecture="llama")` | Writes GGUF v3 **F32** + metadata |
| `LanguageModel.load_gguf(path, learning_rate=3e-4)` | Reads F32 / F16 / BF16; sizes model from metadata |

## C++

```cpp
#include "NeuralNet/Network/LanguageModel.hpp"
#include "NeuralNet/IO/Gguf.hpp"

model.saveGguf("toy.gguf", "llama");
LanguageModel loaded = LanguageModel::loadGguf("toy.gguf", 3e-4f);

Gguf::Config cfg = Gguf::loadConfig("toy.gguf");
SafeTensors::File mapped = Gguf::loadMappedWeights("toy.gguf", cfg);
```

Headers: `NeuralNet/IO/Gguf.hpp`.

## Tensor names (llama.cpp ↔ Sentinel)

| Sentinel | GGUF |
| -------- | ---- |
| `token_embedding.weight` | `token_embd.weight` |
| `blocks.{i}.attn_norm.weight` | `blk.{i}.attn_norm.weight` |
| `blocks.{i}.attn.q_proj.weight` | `blk.{i}.attn_q.weight` |
| `blocks.{i}.attn.k_proj.weight` | `blk.{i}.attn_k.weight` |
| `blocks.{i}.attn.v_proj.weight` | `blk.{i}.attn_v.weight` |
| `blocks.{i}.attn.o_proj.weight` | `blk.{i}.attn_output.weight` |
| `blocks.{i}.ffn_norm.weight` | `blk.{i}.ffn_norm.weight` |
| `blocks.{i}.ffn.gate_proj.weight` | `blk.{i}.ffn_gate.weight` |
| `blocks.{i}.ffn.up_proj.weight` | `blk.{i}.ffn_up.weight` |
| `blocks.{i}.ffn.down_proj.weight` | `blk.{i}.ffn_down.weight` |
| `final_norm.weight` | `output_norm.weight` |
| `lm_head.weight` | `output.weight` (omitted when tied) |

FFN / `output` biases are written only when `use_bias` is true.

## Metadata written on export

- `general.architecture`, `general.name`, `general.alignment` (32), `general.file_type` (`ALL_F32`)
- `{arch}.context_length`, `.embedding_length`, `.block_count`, `.feed_forward_length`
- `{arch}.attention.head_count`, `.attention.head_count_kv`, `.attention.layer_norm_rms_epsilon`
- `{arch}.rope.freq_base`

Vocab size on import is inferred from `token_embd.weight` (ggml dims `[hidden, vocab]`) or `tokenizer.ggml.tokens` length when present.

## Limits

| Supported | Not supported |
| --------- | ------------- |
| GGUF v2/v3 little-endian | Big-endian packs |
| F32 / F16 / BF16 tensors | Quantized types (`Q4_0`, `Q5_K`, …) |
| `llama` / `mistral` / `qwen2` | Other arches, MoE, multimodal |
| Tied embeddings (omit `output.weight`) | Tokenizer embed inside GGUF (no `.sbpe` synthesis) |

## Smoke / CI

- Env: `SENTINEL_GGUF_SMOKE=1` → `LanguageModel::runGgufExportSmokeDemo` (config + weight map + export/import logits)
- Python host: `ci_host_smoke.py` `test_gguf_roundtrip` + `examples/python/gguf_roundtrip.py`
