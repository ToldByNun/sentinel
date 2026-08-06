# HuggingFace causal-LM import

Sentinel can **import HuggingFace causal LM checkpoints** that fit the engine surface (RoPE + RMSNorm + SwiGLU + optional GQA), then fine-tune with the usual C++ / Python train path.

This is **not** a Llama-only importer. Public APIs are HF-generic (`loadHuggingFace` / `load_huggingface`, `HfTokenizer`). Llama / Mistral / Qwen2-style repos are the **first supported layout family** because they share tensor names and math.

API details: [Python](python.md) · [C++](cpp.md). Example (next milestone): `examples/python/finetune_hf.py`.

---

## Quick start

### Python

```python
import sentinel as S

model = S.LanguageModel.load_huggingface("/path/to/hf_model", learning_rate=3e-4)
tok = S.HfTokenizer.load("/path/to/hf_model")

ids = tok.encode("hello world", add_special_tokens=True)
# Fine-tune: encode corpus → next-token pairs → model.train(...).
# See recipes in python.md; end-to-end script: examples/python/finetune_hf.py (when present).

if S.cuda_available():
    model.enable_cuda()
    model.set_prefer_flash_attention(True)
    model.enable_cuda_train()
```

Expected directory contents:

| File | Role |
| ---- | ---- |
| `config.json` | Arch (required) |
| `model.safetensors` **or** sharded `*.safetensors` + `model.safetensors.index.json` | Weights |
| `tokenizer.json` | ByteLevel BPE (for `HfTokenizer`) |

After fine-tune, save Sentinel weights with `model.save_safetensors(...)`. Keep the HF `tokenizer.json` (or convert later); native `.sbpe` is only for Sentinel-trained BPEs.

### C++

```cpp
#include "NeuralNet/Network/LanguageModel.hpp"
#include "NeuralNet/Tokenizer/HfTokenizer.hpp"

LanguageModel model = LanguageModel::loadHuggingFace("/path/to/hf_model", 3e-4f);
HuggingFace::Tokenizer tok = HuggingFace::Tokenizer::load("/path/to/hf_model");
std::vector<int> ids = tok.encode("hello world", /*addSpecialTokens=*/true);
```

Lower-level pieces (if you need them): `HuggingFace::loadConfig`, `HuggingFace::loadMappedWeights` under `IO/`.

---

## Supported

### `model_type` allowlist

| `model_type` | Notes |
| ------------ | ----- |
| `llama` | Llama-3.x ByteLevel BPE + GQA |
| `mistral` | Same weight names; **no** sliding-window attention |
| `qwen2` | Same layout family |

Architectures should look like `*ForCausalLM` when `architectures` is present.

### Features mapped from `config.json`

| HF field | Sentinel |
| -------- | -------- |
| `vocab_size` | embedding / LM-head rows |
| `hidden_size` | `embeddingDim` |
| `num_hidden_layers` | block count |
| `num_attention_heads` | query heads |
| `num_key_value_heads` | GQA (`kv_head_count`); omitted → MHA |
| `intermediate_size` | SwiGLU width |
| `max_position_embeddings` | RoPE / max positions |
| `rope_theta` | RoPE base (default `10000`) |
| `rms_norm_eps` | RMSNorm ε |
| `tie_word_embeddings` | embed ↔ LM-head tying |
| `attention_bias` / `mlp_bias` | `use_bias` (either true → biases trainable) |

Weights load **BF16 / F16 / F32** safetensors → host F32. Shards via `model.safetensors.index.json`.

### Weight names (first layout family)

Llama/Mistral/Qwen2-style keys, e.g.:

- `model.embed_tokens.weight`
- `model.layers.{i}.self_attn.{q,k,v,o}_proj.weight`
- `model.layers.{i}.mlp.{gate,up,down}_proj.weight`
- `model.layers.{i}.input_layernorm.weight` / `post_attention_layernorm.weight`
- `model.norm.weight`
- `lm_head.weight` (optional when tied)

Mapped to Sentinel safetensors names (`token_embedding.weight`, `blocks.{i}.attn.*`, `ffn.*`, …). Linear layout is already `[out, in]` — no transpose for this family.

### Tokenizer

`HfTokenizer` / `HuggingFace::Tokenizer` loads **`tokenizer.json`** with:

- `model.type == "BPE"`
- ByteLevel decoder
- Pre-tokenizers: `ByteLevel` and `Sequence` (Split + ByteLevel), Llama-3-style `ignore_merges`

Native Sentinel **`.sbpe`** (`BPETokenizer`) remains for models trained from scratch in Sentinel.

---

## Unsupported (rejected early)

Clear errors on import / config parse — do not expect silent fallbacks.

| Category | Examples |
| -------- | -------- |
| Unknown / other arches | `gpt2`, `gemma`, `gemma2`, `phi3`, … |
| MoE | `mixtral`, `num_experts` / `num_local_experts` > 0 |
| Sliding window | `sliding_window` set (classic Mistral SWA) |
| Quantized packs | `quantization_config` (GPTQ / AWQ / …) |
| Other tensor layouts | non–Llama/Mistral-like names |
| Other tokenizers | WordPiece, Unigram, Metaspace (Llama-2 SentencePiece) |
| Multimodal / vision | not in scope |

Also out of scope for now: bit-identical loss vs Transformers, GGUF, and separate brand importers (`loadHuggingFaceLlama`, …).

---

## Fine-tune / VRAM notes

- Import builds a **host** `LanguageModel` sized from config. Call `enable_cuda` / `enable_cuda_train` before heavy train when using GPU.
- Consumer 16 GB: prefer flash attention; for multi‑B models use HostSGD / SBAO (`set_prefer_host_sgd` or `SbaoMode.HostFusedHalfSgd`) and activation checkpointing `Full` — see [Python](python.md) / [C++](cpp.md) device setup.
- GQA reduces KV footprint vs MHA at the same width; pack budget still auto-scales from free VRAM.
- After fine-tune, `save_safetensors` writes Sentinel layout + arch metadata (`kv_head_count`, `rope_theta`, `use_bias`, …). Reload with a matching ctor or keep using the same imported model object.

---

## Smokes (harness)

With the `sentinel` demo binary:

| Env | What |
| --- | ---- |
| `SENTINEL_HF_CONFIG_SMOKE=1` | `config.json` allowlist / rejects |
| `SENTINEL_HF_WEIGHT_MAP_SMOKE=1` | shard remap |
| `SENTINEL_HF_IMPORT_SMOKE=1` | `loadHuggingFace` |
| `SENTINEL_HF_TOKENIZER_SMOKE=1` | `tokenizer.json` encode/decode |
| `SENTINEL_HF_ROUNDTRIP_SMOKE=1` | import + encode + 1 host train step + generate (finite gate) |

---

## Related headers / bindings

| C++ | Python |
| --- | ------ |
| `LanguageModel::loadHuggingFace` | `LanguageModel.load_huggingface` |
| `HuggingFace::Tokenizer` (`Tokenizer/HfTokenizer.hpp`) | `HfTokenizer` |
| `HuggingFace::loadConfig` | (C++ only) |
| `HuggingFace::loadMappedWeights` | (C++ only) |
