# HuggingFace causal-LM import / export

Sentinel can **import** HuggingFace causal LM checkpoints that fit the engine surface (RoPE + RMSNorm + SwiGLU + optional GQA), fine-tune with the usual C++ / Python train path, then **export** back to a Transformers-compatible directory (`config.json` + `model.safetensors` and/or `pytorch_model.bin` with HF tensor names).

This is **not** a Llama-only importer/exporter. Public APIs are HF-generic (`loadHuggingFace` / `saveHuggingFace`, `load_huggingface` / `save_huggingface`, `HfTokenizer`). Llama / Mistral / Qwen2-style repos are the **first supported layout family** because they share tensor names and math.

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

# After fine-tune: HF-compatible directory (weights + config; copy tokenizer from the source repo)
model.save_huggingface(
    "/path/to/hf_export",
    model_type="llama",
    tokenizer_source_directory="/path/to/hf_model",
    weight_format="both",  # "safetensors" | "bin" | "both"
)
```

Expected directory contents (import **and** export):

| File | Role |
| ---- | ---- |
| `config.json` | Arch (required) |
| `model.safetensors` **or** sharded `*.safetensors` + `model.safetensors.index.json` | Weights (preferred on import; export default) |
| `pytorch_model.bin` **or** sharded `*.bin` + `pytorch_model.bin.index.json` | PyTorch ZIP state-dict fallback (modern `torch.save`; F32/F16/BF16 → host F32) |
| `tokenizer.json` | ByteLevel BPE (for `HfTokenizer`; export copies when `tokenizer_source_directory` is set) |

`save_safetensors` still writes **Sentinel** tensor names + arch metadata. Prefer `save_huggingface` when the consumer is Transformers / `from_pretrained`. Native `.sbpe` is only for Sentinel-trained BPEs.

### C++

```cpp
#include "NeuralNet/Network/LanguageModel.hpp"
#include "NeuralNet/Tokenizer/HfTokenizer.hpp"

LanguageModel model = LanguageModel::loadHuggingFace("/path/to/hf_model", 3e-4f);
HuggingFace::Tokenizer tok = HuggingFace::Tokenizer::load("/path/to/hf_model");
std::vector<int> ids = tok.encode("hello world", /*addSpecialTokens=*/true);

model.saveHuggingFace("/path/to/hf_export", "llama", "/path/to/hf_model");
```

Lower-level pieces (if you need them): `HuggingFace::loadConfig` / `saveConfig`, `HuggingFace::loadMappedWeights` / `saveDirectory` under `IO/`.

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

Weights load **BF16 / F16 / F32** (safetensors or modern PyTorch ZIP `.bin`) → host F32. Prefer safetensors when both exist; else `pytorch_model.bin` / index shards. Legacy non-zip pickle `.bin` is rejected.

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

## Export notes

- Writes **F32** weights with Llama/Mistral/Qwen2-style keys (`model.embed_tokens.weight`, `model.layers.{i}.*`, `model.norm.weight`, optional `lm_head.weight`).
- `weight_format`: `"safetensors"` (default) → `model.safetensors`; `"bin"` → `pytorch_model.bin` (torch.load-compatible ZIP); `"both"` → both.
- `config.json` includes the allowlisted `model_type`, matching `architectures[]`, GQA heads, `rope_theta`, `rms_norm_eps`, `tie_word_embeddings`, and `attention_bias` / `mlp_bias` from Sentinel `use_bias`.
- Tied embeddings: export **omits** `lm_head.weight` (same as many HF repos); import reloads via `tie_word_embeddings`.
- Tokenizer files are **not** synthesized from `.sbpe`. Pass `tokenizer_source_directory` to copy `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, `vocab.json`, `merges.txt`, and `generation_config.json` when present.
- Export is single-file per format (no shard index). Re-import via `loadHuggingFace` / `load_huggingface` is covered by the export smoke (`both` + bin-only reload).

---

## Fine-tune / VRAM notes

- Import builds a **host** `LanguageModel` sized from config. Call `enable_cuda` / `enable_cuda_train` before heavy train when using GPU.
- Consumer 16 GB: prefer flash attention; for multi‑B models use HostSGD / SBAO (`set_prefer_host_sgd` or `SbaoMode.HostFusedHalfSgd`) and activation checkpointing `Full` — see [Python](python.md) / [C++](cpp.md) device setup.
- GQA reduces KV footprint vs MHA at the same width; pack budget still auto-scales from free VRAM.
- After fine-tune prefer `save_huggingface` for Transformers; `save_safetensors` still writes Sentinel layout + arch metadata for in-engine reload.

---

## Smokes (harness)

With the `sentinel` demo binary:

| Env | What |
| --- | ---- |
| `SENTINEL_HF_CONFIG_SMOKE=1` | `config.json` allowlist / rejects / serialize |
| `SENTINEL_HF_WEIGHT_MAP_SMOKE=1` | shard remap + export remap |
| `SENTINEL_HF_IMPORT_SMOKE=1` | `loadHuggingFace` |
| `SENTINEL_HF_EXPORT_SMOKE=1` | `saveHuggingFace` → reload parity + tokenizer copy + `pytorch_model.bin` |
| `SENTINEL_PYTORCH_BIN_SMOKE=1` | ZIP state-dict save/load roundtrip (`PytorchStateDict`) |
| `SENTINEL_HF_TOKENIZER_SMOKE=1` | `tokenizer.json` encode/decode |
| `SENTINEL_HF_ROUNDTRIP_SMOKE=1` | import + encode + 1 host train step + generate (finite gate) |

---

## Related headers / bindings

| C++ | Python |
| --- | ------ |
| `LanguageModel::loadHuggingFace` | `LanguageModel.load_huggingface` |
| `LanguageModel::saveHuggingFace` | `LanguageModel.save_huggingface` |
| `HuggingFace::Tokenizer` (`Tokenizer/HfTokenizer.hpp`) | `HfTokenizer` |
| `HuggingFace::loadConfig` / `saveConfig` | (C++ only) |
| `HuggingFace::loadMappedWeights` / `saveDirectory` | (C++ only) |
| `PytorchStateDict::load` / `save` | (C++ only; used by HF import/export) |
