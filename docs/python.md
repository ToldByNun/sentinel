# Python API

Package name on PyPI: **`sentinel-lm`**. Import name: **`sentinel`**. Version: **`0.1.0`**.

```python
import sentinel as S
S.cuda_available()
S.__version__
```

Bindings live in [`python/sentinel/_core.cpp`](../python/sentinel/_core.cpp), [`bindings_ops.cpp`](../python/sentinel/bindings_ops.cpp), [`bindings_data.cpp`](../python/sentinel/bindings_data.cpp) and are re-exported from [`python/sentinel/__init__.py`](../python/sentinel/__init__.py).

HuggingFace checkpoint import / export: **[huggingface.md](huggingface.md)**.

On Windows, importing `sentinel` adds CUDA `bin\x64` / `bin` via `os.add_dll_directory` so `cublasLt` loads. Set `CUDA_PATH` if the toolkit is non-default.

---

## Module

| Name | Type | Description |
| ---- | ---- | ----------- |
| `cuda_available()` | `() -> bool` | True if a CUDA device is usable |
| `__version__` | `str` | Binding / core version (`"0.1.0"`) |
| `safetensors_load` / `safetensors_save` / `is_safetensors_file` | helpers | Low-level SafeTensors I/O |
| `CLASS_CPP` / `CLASS_JSON` / `CLASS_PYTHON` / `CLASS_COUNT` | `int` | Labels for `ClassificationDataset` |
| `Matrix`, `Adam`, `Softmax`, … | classes | Mid-level ops — see [Mid-level ops](#mid-level-ops-custom-loops) |

---

## Enums

### `ActivationCheckpointMode`

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

Convention: keep the tokenizer next to weights as `{stem}.sbpe` (e.g. `run.safetensors` → `run.sbpe`).

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
| `is_loaded` | True after a successful `load` |

Unsupported: WordPiece / Unigram / Metaspace (Llama-2 SentencePiece). Full import guide: [huggingface.md](huggingface.md).

---

## `LanguageModelDataset`

In-memory next-token examples. For streaming JSONL / Arrow epochs use [`LanguageModelChunkSource`](#languagemodelchunksource-streaming) + `train_chunks`.

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
| `from_token_ids` (static) | `(token_ids, vocabulary_size, build_one_hot=True) -> LanguageModelDataset` | Build from a single id sequence |
| `make_one_hot_sequence` (static) | `(target_token_ids, vocabulary_size) -> Matrix` | Host one-hot targets |
| `size` | `int` (ro) | Number of examples |
| `total_prediction_count` | `int` (ro) | Sum of next-token positions |
| `vocabulary_size` | `int` (rw) | Set by `build` |
| `examples` | `list[LanguageModelExample]` (rw) | Underlying examples |
| `__len__` / `__getitem__` | | Index into `examples` |

For JSONL → list of strings without streaming, use [`examples/python/train_jsonl.py`](../examples/python/train_jsonl.py) (accepts `text` / `content` / `problem_statement`).

---

## `LanguageModelChunkSource` (streaming)

First-class Python streaming for `.jsonl` or HF Arrow dirs (via C++ `TextRowReader`). JSONL string fields: **`problem_statement`**, **`text`**, or **`content`** (same as `JsonlLoader`).

```python
source = S.LanguageModelChunkSource(
    "corpus.jsonl",
    maximum_text_characters=0,
    maximum_token_count=512,
    chunk_example_count=64,
    train_ratio=0.9,
    seed=42,
    test_reservoir_cap=256,
)
sample = source.prepare_tokenizer_sample(2000)
tok = S.BPETokenizer()
tok.train(sample, vocab_size=8000)
source.set_tokenizer(tok)          # BPETokenizer only; must outlive the source
source.materialize()
model.train_chunks(source, epochs=2, batch_size=32, gradient_accumulation_steps=4)

# Or iterate chunks yourself (host custom loop):
for chunk in source.iter_train_chunks():
    model.train_step(chunk.examples)
```

Runnable example: [`examples/python/train_chunks.py`](../examples/python/train_chunks.py).

| Method / property | Notes |
| ----------------- | ----- |
| ctor | `path`, `maximum_text_characters=0`, `maximum_token_count=0`, `chunk_example_count=64`, `train_ratio=0.9`, `seed=42`, `test_reservoir_cap=256` |
| `set_tokenizer` | **`BPETokenizer` only** (keep alive for the source lifetime) |
| `prepare_tokenizer_sample` | First N train-hash texts; rewinds (releases GIL) |
| `materialize` / `prepare_test_reservoir` | One corpus pass → cached token ids + test reservoir (releases GIL) |
| `rewind_train` | Start a new epoch (length-sorts materialized examples) |
| `sort_train_by_length` | Stable short→long sort |
| `fill_train_dataset` / `train_dataset()` | Copy all train examples into a `LanguageModelDataset` |
| `next_train_chunk(out)` | Fill `out`; `False` when exhausted |
| `take_train_chunk()` | `(dataset\|None, more)` — Pythonic next-chunk |
| `iter_train_chunks(rewind=True)` / `__iter__` | Yield `LanguageModelDataset` chunks for one epoch |
| `is_train_row(row_index)` | Train vs test partition for a corpus row index |
| `test_dataset` | Reservoir after materialize |
| `file_path`, `chunk_example_count`, `train_ratio` | Config mirrors |
| `train_example_count`, `train_prediction_count`, `is_materialized` | After materialize |

Also: `JsonlLoader.load(path, maximum_rows=50)`, `try_parse_line`, `source_to_label`, and `CorpusRow` (`text`, `source`) for JSONL inspection.

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

`load_huggingface` expects a **local** directory (`config.json` + weights). Hub repo ids are not resolved inside the binding — use a local path, or resolve first (see [huggingface.md](huggingface.md); `examples/python/finetune_hf.py` uses `huggingface_hub` when installed).

Or size / load from a native **`sentinel-model`** JSON/YAML config:

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
}, load_weights=True, base_directory="")
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
| `LanguageModel.from_config` | path / dict / `SentinelModelConfig`; kwargs `load_weights=True`, `base_directory=""` |
| `LanguageModel.load_sentinel_model(path, load_weights=True)` | From file/dir |
| `LanguageModel.from_sentinel_config(config, base_directory="", load_weights=True)` | From config object |
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
| `sync_device` | `() -> None` | Re-upload host weights if mirror active |
| `cuda_enabled` / `cuda_train_enabled` | `bool` (ro) | |
| `set_prefer_flash_attention` | `(enabled: bool) -> None` | |
| `set_prefer_mixed_precision` | `(enabled: bool) -> None` | FP16 GEMMs + loss scale when large |
| `set_prefer_muon` | `(enabled: bool) -> None` | Muon on hidden 2D weights |
| `set_muon_ns_steps` | `(steps: int) -> None` | Newton–Schulz steps for Muon |
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
| `set_logit_chunk_rows` | `(rows: int) -> None` | Chunked CE / LM-head |
| `set_tie_embedding` | `(enabled: bool) -> None` | Default on |
| `parameter_count` | `int` (ro) | Trainable elements (tied head counted once) |
| `intermediate_size` / `rope_theta` / `use_bias` / `kv_head_count` | (ro) | Arch mirrors |
| `tie_embedding` / `max_position` / `block_count` | (ro) | |
| `optimizer` | `Adam` (ro ref) | Host Adam attached to the LM |
| `token_embedding` / `final_norm` / `lm_head_weight` / `output_projection` | (ro refs) | Inspect / mutate weights |
| `token_embedding_state` / `final_norm_gamma_state` | `AdamState` (ro refs) | |
| `block(i)` | `TransformerBlock` | Reference to transformer block `i` |

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
model.train_chunks(source, epochs=2, batch_size=32, gradient_accumulation_steps=4)
loss = model.average_loss(train)
cont = model.generate(prompt_ids, new_token_count=32, temperature=0.9, top_k=20, seed=7)
```

| Method | Signature | Notes |
| ------ | --------- | ----- |
| `train` | `(train, epochs=1, batch_size=32, gradient_accumulation_steps=1, log_every_epochs=1, test=None) -> None` | In-memory dataset |
| `train_chunks` | `(source, epochs=1, log_every_epochs=1, batch_size=32, gradient_accumulation_steps=4) -> None` | Streaming `LanguageModelChunkSource` (default accum **4**; releases GIL) |
| `forward` | `(token_ids) -> Matrix` | Logits `vocab × seq` |
| `example_loss` / `average_loss` | example or dataset → `float` | |
| `accumulate_example` / `apply_gradients` / `train_step` | Host custom-loop step API | See [Mid-level ops](#mid-level-ops-custom-loops) |
| `generate` | `(prompt_token_ids, new_token_count, temperature=1.0, top_k=40, seed=42) -> list[int]` | Returns **new** tokens only. `temperature <= 0` → greedy |

### Checkpoints

| Method | Signature | Notes |
| ------ | --------- | ----- |
| `save_checkpoint` | `(path, include_optimizer=True) -> None` | Native `.snlm` |
| `load_checkpoint` | `(path) -> None` | `.snlm` or `.safetensors` |
| `save_safetensors` | `(path) -> None` | Weights + arch metadata (Sentinel tensor names) |
| `load_safetensors` | `(path) -> None` | Architecture must already match |
| `load_huggingface` | `(path, learning_rate=3e-4) -> LanguageModel` | Static: local HF dir → sized model + weights |
| `save_huggingface` | `(path, model_type="llama", tokenizer_source_directory="", weight_format="safetensors") -> None` | Export Transformers-compatible dir. `weight_format`: `"safetensors"` \| `"bin"` \| `"both"` |
| `from_config` / `load_sentinel_model` / `from_sentinel_config` | native `sentinel-model` JSON/YAML (+ optional weights) | |
| `sentinel_config` / `save_sentinel_config` | snapshot / write native config | |

Rebuild a model with the **same** dims before `load_safetensors`. Tokenizer is separate (`BPETokenizer` / `HfTokenizer`).

### Probe

```python
tok_s = model.probe_cuda_packed_train_tokens_per_second(512, warmup_steps=3, timed_steps=8)
model.probe_cuda_train_step_profile(512, warmup_steps=2, timed_steps=4)
```

Requires `enable_cuda_train`. Unset `SENTINEL_PHASE_TRACE` when quoting tok/s.

---

## Typical recipes

| Script | Role |
| ------ | ---- |
| [`train_tiny.py`](../examples/python/train_tiny.py) | Toy corpus → train → checkpoint |
| [`generate.py`](../examples/python/generate.py) | Load / train toy → sample |
| [`train_from_config.py`](../examples/python/train_from_config.py) | [`configs/tiny.json`](../examples/configs/tiny.json) / [`tiny.yaml`](../examples/configs/tiny.yaml) / [`base-768.json`](../examples/configs/base-768.json) |
| [`custom_train_loop.py`](../examples/python/custom_train_loop.py) | Host `accumulate_example` / `apply_gradients` |
| [`custom_layers_demo.py`](../examples/python/custom_layers_demo.py) | Attention / FFN / Spulse / SafeTensors |
| [`train_jsonl.py`](../examples/python/train_jsonl.py) | In-memory JSONL train |
| [`finetune_hf.py`](../examples/python/finetune_hf.py) | HF import → JSONL fine-tune → `save_huggingface` |

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

Optional surface for custom host training / research — high-level `train()` / `train_chunks()` remain the default path.

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

### `Matrix`

| API | Notes |
| --- | ----- |
| ctor `(rows, cols, fill=0)` | |
| `rows` / `cols` / `shape` / `empty` | |
| `at` / `set` / `fill` / `resize` / `ensure_size` | |
| `to_list` / `from_list` | Row-major flat floats |
| `to_numpy` / `from_numpy` | Optional numpy (added in `__init__.py`) |
| statics | `zeros_like`, `transpose`, `add`, `subtract`, `scale`, `multiply` (± transpose flags), `multiply_elementwise`, `add_in_place`, `scale_in_place`, `gemm` |

### Activations / losses / init

| Type | API |
| ---- | --- |
| `Softmax` | `apply`, `apply_into` (column-wise) |
| `SiLU` | `apply_into`, `derivative_into` |
| `ReLU` | `apply`, `apply_into`, `derivative`, `derivative_into` |
| `CrossEntropy` | `loss`, `gradient` (mean Softmax+CE) |
| `MSE` | `loss`, `gradient` |
| `UniformInit` | `matrix(rows, cols, scale, seed)`, `fill` |

### Optimizers

| Type | Notes |
| ---- | ----- |
| `Adam` | ctor `(lr, beta1=0.9, beta2=0.999, eps=1e-8)`; `step`, `update`, `update_selected_rows`; fields `learning_rate` / betas / `time_step` |
| `AdamState` | `first_moment`, `second_moment`; `zeros_like` |
| `SGD` | `(learning_rate)`; `update` |
| `MuonState` | `momentum`; `zeros_like` |
| `Spulse` | Host stepper (`CudaSpulse::updateHost`). ctor knobs: `learning_rate`, `momentum_beta`, `fast_beta`, `slow_beta`, `epsilon`, `scale_min`/`max`, `weight_decay`, `coverage`, `host_lightweight`, `momentum_storage`, `int8_block_size`. `step` / `update(param, state, grad, gradient_scale=1)` |
| `SpulseState` | `momentum`, `energy_fast`/`slow`, `scale`; `zeros_like` / `ensure` / `clear` |

Packed CUDA train stays on `enable_cuda_train` + `train()` / `train_chunks()`. Host `Spulse.update` does not drive the device path — use `set_prefer_spulse` for that.

### Layers

| Type | Notes |
| ---- | ----- |
| `Embedding` | `(vocab_size, embedding_dim)` or from `Matrix`; `forward` / `backward` |
| `Dense` | `(weight, bias)`; `forward` → `W @ x + b` |
| `Dropout` | `(drop_rate=0.3, seed=7)`; `forward` / `backward` |
| `MeanPool` | sequence mean pool |
| `RMSNorm` | `(embedding_dim, epsilon=1e-5)`; `forward`/`backward` with `RMSNormCache` |
| `RotaryEmbedding` | `(head_dimension, maximum_position_count, base=10000)`; rotate ± inverse in place |
| `CausalSelfAttention` | `create(embed, heads, max_pos, seed=11, window_size=-1, global_token_count=0, rope_base=10000, kv_head_count=-1)`; fwd/bwd + cache |
| `FeedForward` | `create` / `create_with_intermediate_size` / `default_intermediate_size`; SwiGLU fwd/bwd |
| `TransformerBlock` | ctor with GQA / intermediate / rope / bias; `.attention` / `.feed_forward` / norms / weight accessors; `apply_gradients` |
| caches / grads | `*Cache`, `TransformerBlockGradients`, `LanguageModelCache`, `LanguageModelGradients` |

### LM step buffers

| Type | Notes |
| ---- | ----- |
| `LanguageModelExample` | `input_token_ids`, `target_token_ids`, `target_one_hot` |
| `LanguageModelGradients` | `zeros_from(model)`, `zero_in_place`, `add_in_place`, `scale_in_place`, `block_gradients` / `block_count` |
| `LanguageModelCache` | Scratch for `accumulate_example` |

### SafeTensors helpers

| API | Notes |
| --- | ----- |
| `safetensors_load(path) -> SafeTensorsFile` | F32/BF16/F16 → host F32 |
| `safetensors_save(path, file)` | F32 write |
| `is_safetensors_file(path)` | |
| `SafeTensorsFile` | `metadata`, `tensor_names`, `has_tensor`, `get_tensor`, `set_tensor`, `put_matrix` |

### Classifier stack

| Type | Notes |
| ---- | ----- |
| `ClassificationExample` | `token_ids`, `target`, `label` |
| `ClassificationDataset` | `build` / `build_labeled` / `make_one_hot` / `infer_label` |
| `Sequential` | Embed→MeanPool→Dense→ReLU→Dropout→Dense→Softmax; `train` / `predict_class` / `accuracy` |
| `CLASS_*` | Module ints for the three demo labels |

Examples: [`custom_train_loop.py`](../examples/python/custom_train_loop.py), [`custom_layers_demo.py`](../examples/python/custom_layers_demo.py).

---

## Not exposed in Python (yet)

| C++ | Status |
| --- | ------ |
| Direct `CudaMatrix` / device Muon/Adam objects | Prefer host `Matrix` + high-level CUDA train |
| `ArrowChunkReader` (raw) | Used internally by `LanguageModelChunkSource`; no direct binding |
| `HuggingFace::resolveModelDirectory` / Hub download | C++ `IO/HuggingFaceResolve.hpp`; example falls back to `huggingface_hub` |
| `HuggingFace::loadConfig` / weight-map helpers | C++ only; use `load_huggingface` / `save_huggingface` |
| `PytorchStateDict` | C++ only (used by HF import/export) |
| Smoke demos on `LanguageModel` | Harness env gates only |

Tokenizer I/O (`.sbpe`) and HF `tokenizer.json` are exposed — see `BPETokenizer` / `HfTokenizer` above.
