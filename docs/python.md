# Python API

Package name on PyPI: **`sentinel-lm`**. Import name: **`sentinel`**.

```python
import sentinel as S
S.cuda_available()
S.__version__
```

Bindings live in [`python/sentinel/_core.cpp`](../python/sentinel/_core.cpp), [`bindings_ops.cpp`](../python/sentinel/bindings_ops.cpp), [`bindings_data.cpp`](../python/sentinel/bindings_data.cpp) and are re-exported from [`python/sentinel/__init__.py`](../python/sentinel/__init__.py).

HuggingFace checkpoint import / export (supported `model_type`s, rejects, VRAM): **[huggingface.md](huggingface.md)**.

On Windows, importing `sentinel` adds CUDA `bin\x64` / `bin` via `os.add_dll_directory` so `cublasLt` loads. Set `CUDA_PATH` if the toolkit is non-default.

---

## Module

| Name | Type | Description |
| ---- | ---- | ----------- |
| `cuda_available()` | `() -> bool` | True if a CUDA device is usable |
| `__version__` | `str` | Binding / core version (e.g. `"0.1.0"`) |
| `Matrix`, `Adam`, `Softmax`, … | classes | Mid-level ops — see [below](#mid-level-ops-custom-loops) |

---

## Enums

### `ActivationCheckpointMode`

Device train activation checkpointing.

| Value | Meaning |
| ----- | ------- |
| `Off` | Keep activations (fastest when VRAM fits; enables train graphs when shapes are stable) |
| `Selective` | Recompute selected block work on backward |
| `Full` | Full recompute (typical for ~4B HostSGD on 16 GB) |

### `SbaoMode`

Sentinel Backend Adaptive Optimization residency policy.

| Value | Meaning |
| ----- | ------- |
| `Auto` | Resolve at `enable_cuda_train` from free VRAM + param footprint |
| `GpuInt8Adam` | GPU weights + int8 Adam moments |
| `HostFusedHalfAdam` | FP16 GPU weights; fused FP16 grad D2H + async host Adam |
| `HostFusedHalfSgd` | Same D2H path with host SGD (no Adam `m`/`v`; ~4B path) |

### `SpulseCoverage`

SPULSE tensor ownership (optimizer, not SBAO). Host fused-half keeps momentum on GPU by default (half-delta D2H); same PCIe volume as HostSGD.

| Value | Meaning |
| ----- | ------- |
| `Hybrid` | Hidden 2D block weights (default); Adam keeps embed/norms/biases/head |
| `Full` | All trainable params; Adam idle while SPULSE is on |

### `SpulseMomentumStorage`

Device storage for SPULSE momentum `u` (`set_spulse_momentum_storage`).

| Value | Meaning |
| ----- | ------- |
| `Fp32` | Default reference / best parity |
| `Fp16` | ~2× less VRAM for `u` |
| `Int8` | ~4× less VRAM (per-block absmax); larger step drift |

---

## `BPETokenizer`

```python
tok = S.BPETokenizer()
tok.train(["hello world", "cuda packs"], vocab_size=256)
tok.save("run.sbpe")

tok2 = S.BPETokenizer.load_from("run.sbpe")
ids = tok2.encode("hello")
text = tok2.decode(ids)
```

| Method / property | Signature | Notes |
| ----------------- | --------- | ----- |
| `train` | `(corpus: list[str], vocab_size: int) -> None` | Learn merges; `vocab_size` must exceed unique chars + unk |
| `encode` | `(text: str) -> list[int]` | Unknown pieces → unk |
| `decode` | `(token_ids: list[int]) -> str` | Concat; leading spaces restored |
| `save` | `(path: str) -> None` | Binary Sentinel BPE (`.sbpe`) |
| `load` | `(path: str) -> None` | Replace state from `.sbpe` |
| `load_from` | `(path: str) -> BPETokenizer` | static constructor |
| `vocab_size` | `int` (ro) | |
| `unknown_token_id` | `int` (ro) | |
| `is_trained` | `bool` (ro) | True after `train` / `load` |

Convention: keep the tokenizer next to weights as `{stem}.sbpe` (e.g. `run.safetensors` → `run.sbpe`). Examples write/load that sibling automatically.

---

## `HfTokenizer`

Load a HuggingFace `tokenizer.json` (ByteLevel BPE; Llama-3/Qwen2-style `ignore_merges` supported). Native `.sbpe` training stays on `BPETokenizer`.

```python
tok = S.HfTokenizer.load("/path/to/hf_model")  # or .../tokenizer.json
ids = tok.encode("hello", add_special_tokens=True)
text = tok.decode(ids, skip_special_tokens=True)
```

| Method / property | Notes |
| ----------------- | ----- |
| `load` (static) | file or HF model directory |
| `encode` / `decode` | optional BOS / skip specials |
| `vocab_size`, `bos_token_id`, `eos_token_id`, `pad_token_id`, `unk_token_id` | `-1` when absent |
| `ignore_merges` | Llama-3-style whole-piece vocab hits |

Unsupported: WordPiece / Unigram / Metaspace (Llama-2 SentencePiece). Full import guide: [huggingface.md](huggingface.md).

---

## `LanguageModelDataset`

In-memory next-token examples. Streaming JSONL/Arrow epochs are **C++-only** (`LanguageModelChunkSource`) — see [C++ API](cpp.md).

```python
data = S.LanguageModelDataset.build(
    texts,
    tok,
    maximum_token_count=48,
    build_one_hot=False,
)
n = data.size
positions = data.total_prediction_count
```

| Member | Signature | Notes |
| ------ | --------- | ----- |
| `build` (static) | `(texts, tokenizer, maximum_token_count=0, build_one_hot=False) -> LanguageModelDataset` | `tokenizer` is `BPETokenizer` or `HfTokenizer`; skips sequences shorter than 2 tokens; truncates if `maximum_token_count > 0` |
| `size` | `int` (ro) | Number of examples |
| `total_prediction_count` | `int` (ro) | Sum of next-token positions |
| `vocabulary_size` | `int` (rw) | Set by `build` |

For JSONL → list of strings without bindings, use [`examples/python/train_jsonl.py`](../examples/python/train_jsonl.py).

---

## `LanguageModel`

Causal LM: embed → N× (RMSNorm + Attn + SwiGLU) → final RMSNorm → vocab. Default **weight tying** (LM head shares token embedding).

### Constructor

```python
model = S.LanguageModel(
    vocabulary_size,
    embedding_dim,
    maximum_position_count,
    learning_rate=3e-4,
    block_count=2,
    head_count=4,
    intermediate_size=0,  # 0 = legacy expand-4 SwiGLU width; set for HF-style MLP width
    rope_theta=10000.0,   # HF rope_theta
    use_bias=True,        # False when HF config has no FFN/lm_head bias
    kv_head_count=-1,     # <=0 → MHA (= head_count); else GQA (HF num_key_value_heads)
)
```

`embedding_dim` must be divisible by `head_count`. `intermediate_size` is the SwiGLU gate/up row count (HF `intermediate_size`). For GQA, `head_count` must be divisible by `kv_head_count`.

Or load a HuggingFace causal-LM directory (allowlisted `model_type`: `llama` / `mistral` / `qwen2`):

```python
model = S.LanguageModel.load_huggingface("/path/to/hf_model", learning_rate=3e-4)
```

Or size / load from a native **`sentinel-model`** JSON/YAML config (field names match safetensors metadata):

```python
model = S.LanguageModel.from_config("examples/configs/tiny.json")
# dict / SentinelModelConfig also work:
model = S.LanguageModel.from_config({
    "format": "sentinel-model",
    "vocab_size": 32000,
    "embedding_dim": 768,
    "max_position": 512,
    "block_count": 12,
    "head_count": 12,
    "kv_head_count": 12,
    "intermediate_size": 0,
    "rope_theta": 10000,
    "use_bias": True,
    "tie_embedding": True,
    "rms_norm_eps": 1e-5,
    "learning_rate": 3e-4,
    "weights": None,  # or "weights.safetensors" next to the config
})
model.save_sentinel_config("out/model.yaml")
```

### `SentinelModelConfig`

| Field | Notes |
| ----- | ----- |
| `format` | Must be `"sentinel-model"` |
| `vocab_size` / `embedding_dim` / `max_position` / `block_count` / `head_count` | Required |
| `kv_head_count` | Optional; defaults to `head_count` (MHA) |
| `intermediate_size` | `0` = legacy expand-4 SwiGLU |
| `rope_theta` / `use_bias` / `tie_embedding` / `rms_norm_eps` / `learning_rate` | Optional (defaults match ctor) |
| `weights` | `null`/empty = random init; else `.safetensors` / `.snlm` path (relative to config file) |

| Method | Notes |
| ------ | ----- |
| `SentinelModelConfig.load(path)` | `model.json` / `.yaml` / `.yml` or directory |
| `parse_json` / `parse_yaml` | From text |
| `to_json` / `to_yaml` / `save` | Serialize |
| `LanguageModel.from_config` / `load_sentinel_model` | Build (+ optional weights) |
| `sentinel_config` / `save_sentinel_config` | Snapshot / write |

YAML support is flat `key: value` only (no nested maps/lists). Smoke: `SENTINEL_MODEL_CONFIG_SMOKE=1`.

### Device setup

Call **before** heavy train when using CUDA:

```python
if S.cuda_available():
    model.enable_cuda()                 # forward / generate on device
    model.set_prefer_flash_attention(True)
    model.set_prefer_host_sgd(True)     # optional large-model path
    model.enable_cuda_train()           # packed train; resolves SBAO if enabled
    model.set_activation_checkpoint_mode(S.ActivationCheckpointMode.Full)
```

| Method / property | Signature | Notes |
| ----------------- | --------- | ----- |
| `enable_cuda` | `() -> None` | Upload host weights; inference mirror |
| `enable_cuda_train` | `() -> None` | Packed device train (AMP, pack budget, optimizer policy) |
| `cuda_enabled` | `bool` (ro) | |
| `cuda_train_enabled` | `bool` (ro) | |
| `set_prefer_flash_attention` | `(enabled: bool) -> None` | |
| `set_prefer_muon` | `(enabled: bool) -> None` | Muon on hidden 2D weights |
| `set_prefer_spulse` | `(enabled: bool) -> None` | SPULSE (mutex with Muon); GPU + host fused-half |
| `set_spulse_coverage` | `(coverage: SpulseCoverage) -> None` | `Hybrid` (default) or `Full` |
| `set_spulse_momentum_beta` / `set_spulse_fast_beta` / `set_spulse_slow_beta` | `(beta: float) -> None` | SPULSE EMA knobs |
| `set_spulse_scale_clip` | `(scale_min, scale_max) -> None` | Clip dual-horizon scale |
| `set_spulse_momentum_storage` | `(SpulseMomentumStorage) -> None` | Device `u` as Fp32 / Fp16 / Int8 |
| `set_prefer_int8_adam_moments` | `(enabled: bool) -> None` | |
| `set_prefer_cpu_adam_offload` | `(enabled: bool) -> None` | Host Adam moments / HostFusedHalfAdam path |
| `set_prefer_host_sgd` | `(enabled: bool) -> None` | HostSGD masters (4B-style) |
| `set_prefer_sbao` | `(enabled: bool) -> None` | Enable SBAO policy |
| `set_sbao_mode` | `(mode: SbaoMode) -> None` | |
| `sbao_mode_resolved` | `SbaoMode` (ro) | Concrete mode after `enable_cuda_train` |
| `set_activation_checkpoint_mode` | `(mode: ActivationCheckpointMode) -> None` | |
| `set_prefer_train_graph` | `(enabled: bool) -> None` | Graphs only useful with ckpt `Off` |
| `set_max_packed_columns` | `(columns: int) -> None` | Manual pack cap; skips auto budget |
| `max_packed_columns` | `int` (ro) | |
| `apply_vram_pack_budget` | `(free_fraction=0.70, safety_reserve_bytes=…) -> None` | Recompute pack from free VRAM |
| `parameter_count` | `int` (ro) | Trainable elements (tied head counted once) |
| `intermediate_size` | `int` (ro) | FFN gate/up width (`0` only if no blocks) |
| `rope_theta` | `float` (ro) | RoPE base (HF `rope_theta`) |
| `use_bias` | `bool` (ro) | `False` → fixed-zero FFN/`lm_head` biases (common HF causal LMs) |
| `kv_head_count` | `int` (ro) | K/V heads (HF `num_key_value_heads`); equals `head_count` for MHA |

Prefer setting host-SGD / SBAO **before** `enable_cuda_train` so pack/ckpt resolve correctly.

### Train / eval / generate

```python
model.train(
    train,
    epochs=4,
    batch_size=8,
    gradient_accumulation_steps=1,
    log_every_epochs=1,
    test=None,   # optional LanguageModelDataset
)
loss = model.average_loss(train)
cont = model.generate(prompt_ids, new_token_count=32, temperature=0.9, top_k=20, seed=7)
```

| Method | Signature | Notes |
| ------ | --------- | ----- |
| `train` | `(train, epochs=1, batch_size=32, gradient_accumulation_steps=1, log_every_epochs=1, test=None) -> None` | In-memory only |
| `forward` | `(token_ids) -> Matrix` | Logits `vocab × seq` |
| `example_loss` / `average_loss` | example or dataset → `float` | |
| `accumulate_example` / `apply_gradients` / `train_step` | Host custom-loop step API | See [Mid-level ops](#mid-level-ops-custom-loops) |
| `generate` | `(prompt_token_ids, new_token_count, temperature=1.0, top_k=40, seed=42) -> list[int]` | Returns **new** tokens only (not the prompt). `temperature <= 0` → greedy |

### Checkpoints

| Method | Signature | Notes |
| ------ | --------- | ----- |
| `save_checkpoint` | `(path, include_optimizer=True) -> None` | Native `.snlm` |
| `load_checkpoint` | `(path) -> None` | `.snlm` or `.safetensors` |
| `save_safetensors` | `(path) -> None` | Weights + arch metadata (Sentinel tensor names) |
| `load_safetensors` | `(path) -> None` | Architecture must already match |
| `load_huggingface` | `(path, learning_rate=3e-4) -> LanguageModel` | Static: HF dir (`config.json` + safetensors) → sized model + weights; see [huggingface.md](huggingface.md) |
| `save_huggingface` | `(path, model_type="llama", tokenizer_source_directory="", weight_format="safetensors") -> None` | Export Transformers-compatible dir (`config.json` + `model.safetensors` and/or `pytorch_model.bin`; optional tokenizer copy). `weight_format`: `"safetensors"` \| `"bin"` \| `"both"` |
| `from_config` / `load_sentinel_model` | `(path\|dict\|SentinelModelConfig, …) -> LanguageModel` | Native `sentinel-model` JSON/YAML (+ optional weights) |
| `sentinel_config` / `save_sentinel_config` | snapshot / write native config | |

Rebuild a model with the **same** dims before `load_safetensors`. Tokenizer is separate (`BPETokenizer` / `HfTokenizer`).

### Probe

```python
tok_s = model.probe_cuda_packed_train_tokens_per_second(512, warmup_steps=3, timed_steps=4)
```

Requires `enable_cuda_train`. Unset `SENTINEL_PHASE_TRACE` when quoting tok/s.

---

## Typical recipes

**Tiny CPU/GPU toy** — [`examples/python/train_tiny.py`](../examples/python/train_tiny.py)

**From JSON/YAML config** — [`examples/python/train_from_config.py`](../examples/python/train_from_config.py) + [`examples/configs/`](../examples/configs/)

**Custom host train loop** — [`examples/python/custom_train_loop.py`](../examples/python/custom_train_loop.py)

**Layer / SafeTensors / Spulse ops** — [`examples/python/custom_layers_demo.py`](../examples/python/custom_layers_demo.py)

**JSONL (in-memory)** — [`examples/python/train_jsonl.py`](../examples/python/train_jsonl.py)

**HuggingFace fine-tune + export** — [`examples/python/finetune_hf.py`](../examples/python/finetune_hf.py) (details: [huggingface.md](huggingface.md))

**HuggingFace import / export** — [huggingface.md](huggingface.md) (`load_huggingface` / `save_huggingface` + `HfTokenizer`)

**Large model on 16 GB (HostSGD)**

```python
model.set_prefer_flash_attention(True)
model.set_prefer_muon(False)
model.set_prefer_host_sgd(True)
model.enable_cuda_train()
model.set_activation_checkpoint_mode(S.ActivationCheckpointMode.Full)
```

---

## Mid-level ops (custom loops)

Optional surface for custom host training / research — high-level `train()` remains the default path.

```python
logits = model.forward(ids)                          # vocab x seq Matrix
probs = S.Softmax.apply(logits)
loss = S.CrossEntropy.loss(probs, target_one_hot)

grads = S.LanguageModelGradients.zeros_from(model)
cache = S.LanguageModelCache()
loss = model.accumulate_example(example, grads, cache)  # host Softmax+CE bwd
grads.scale_in_place(1.0 / batch_size)
model.apply_gradients(grads)                         # host Adam step

# or:
loss = model.train_step(dataset.examples)            # Python helper

attn = S.CausalSelfAttention.create(64, 4, 128, kv_head_count=2)
ffn = S.FeedForward.create_with_intermediate_size(64, 96)
opt = S.Spulse(learning_rate=1e-2)                   # host SPULSE
file = S.safetensors_load("w.safetensors")
```

| Type / API | Notes |
| ---------- | ----- |
| `Matrix` | `shape`, `to_list` / `from_list`, `to_numpy` / `from_numpy` (numpy optional), `gemm` / `multiply` / … |
| `Softmax` / `SiLU` / `ReLU` / `CrossEntropy` / `MSE` | Activations + losses |
| `UniformInit` | Symmetry-breaking fills |
| `Adam` / `AdamState` / `SGD` / `MuonState` | Host optimizers (`model.optimizer` is the LM Adam) |
| `Spulse` / `SpulseState` | Host dual-horizon SPULSE (`CudaSpulse::updateHost`) |
| `Embedding` / `Dense` / `Dropout` / `MeanPool` / `RMSNorm` / `RotaryEmbedding` | Layers |
| `CausalSelfAttention` / `FeedForward` | Full fwd/bwd (+ caches); GQA / sparse knobs on attention |
| `TransformerBlock` | Block + `.attention` / `.feed_forward` accessors |
| `LanguageModelExample` / `LanguageModelGradients` / `LanguageModelCache` | Step buffers |
| `forward` / `example_loss` / `accumulate_example` / `apply_gradients` / `train_step` | Custom LM loop |
| `train_chunks` / `LanguageModelChunkSource` | Streaming JSONL epochs |
| `safetensors_load` / `safetensors_save` / `SafeTensorsFile` | Low-level weight I/O |
| `Sequential` / `ClassificationDataset` | Small classifier stack (embed→pool→MLP) |
| `token_embedding` / `final_norm` / `block(i)` / `lm_head_weight` / `output_projection` | Inspect / mutate weights |

`accumulate_example` / `apply_gradients` / `train_step` / host `Spulse.update` are **host** paths. Packed CUDA train stays on `enable_cuda_train` + `train()`. `enable_cuda()` still accelerates `forward` / `generate`.

Examples: [`custom_train_loop.py`](../examples/python/custom_train_loop.py), [`custom_layers_demo.py`](../examples/python/custom_layers_demo.py).

## Not exposed in Python (yet)

| C++ | Status |
| --- | ------ |
| Direct `CudaMatrix` / device Muon/Adam objects | Prefer host `Matrix` + high-level CUDA train |
| Arrow chunk reader | C++ only |

Tokenizer I/O (`.sbpe`) is exposed — see `BPETokenizer` above.
