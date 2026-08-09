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
| `safetensors_load` / `safetensors_save` / `is_safetensors_file` | functions | Low-level SafeTensors I/O |
| `CLASS_CPP` / `CLASS_JSON` / `CLASS_PYTHON` / `CLASS_COUNT` | `int` | Labels for `ClassificationDataset` |
| `Matrix`, `Adam`, `Softmax`, … | classes | Mid-level ops — see [below](#mid-level-ops) |

Full public export list: `sentinel.__all__`.

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
| `is_loaded` | `bool` (ro) |
| `ignore_merges` | Llama-3-style whole-piece vocab hits |

Unsupported: WordPiece / Unigram / Metaspace (Llama-2 SentencePiece). Full import guide: [huggingface.md](huggingface.md).

---

## Data: datasets, JSONL, streaming

### `LanguageModelDataset`

In-memory next-token examples. For large corpora prefer [`LanguageModelChunkSource`](#languagemodelchunksource) + `train_chunks`.

```python
data = S.LanguageModelDataset.build(
    texts,
    tok,  # BPETokenizer or HfTokenizer
    maximum_token_count=48,
    build_one_hot=False,
)
n = data.size
positions = data.total_prediction_count
example = data.examples[0]   # or data[0]
```

| Member | Signature | Notes |
| ------ | --------- | ----- |
| `build` (static) | `(texts, tokenizer, maximum_token_count=0, build_one_hot=False) -> LanguageModelDataset` | Skips sequences shorter than 2 tokens; truncates if `maximum_token_count > 0` |
| `from_token_ids` (static) | `(token_ids, vocabulary_size, build_one_hot=True) -> LanguageModelExample` | Single shifted example |
| `make_one_hot_sequence` (static) | `(target_token_ids, vocabulary_size) -> Matrix` | |
| `size` / `__len__` | `int` (ro) | Number of examples |
| `total_prediction_count` | `int` (ro) | Sum of next-token positions |
| `vocabulary_size` | `int` (rw) | Set by `build` |
| `examples` | `list[LanguageModelExample]` | Mutable reference |
| `__getitem__` | `int -> LanguageModelExample` | |

### `LanguageModelExample` / `LanguageModelGradients` / `LanguageModelCache`

| Type | Fields / API |
| ---- | ------------ |
| `LanguageModelExample` | `input_token_ids`, `target_token_ids`, `target_one_hot` |
| `LanguageModelCache` | Opaque host forward cache (default ctor) |
| `LanguageModelGradients` | `zeros_from(model)`, `zero_in_place`, `add_in_place`, `scale_in_place`; fields `token_embedding`, `final_norm_gamma`, `projection_weight`, `projection_bias`, `block_gradients(i)`, `block_count` |

### `JsonlLoader` / `CorpusRow`

Native JSONL reader used by streaming (expects a **`problem_statement`** string field — SERA-style). For ad-hoc demos with `text` / `content` aliases, see [`examples/python/train_jsonl.py`](../examples/python/train_jsonl.py).

| API | Notes |
| --- | ----- |
| `CorpusRow` | `text`, `source` |
| `JsonlLoader.load(path, maximum_rows=50)` | `≤0` = no row limit |
| `JsonlLoader.try_parse_line(line)` | `CorpusRow` or `None` |
| `JsonlLoader.source_to_label(source)` | Map source string → class id |

### `LanguageModelChunkSource`

Streaming epochs over `.jsonl` or HF Arrow directories (via `createTextRowReader`). Materializes token ids once, then yields train chunks.

```python
source = S.LanguageModelChunkSource(
    "corpus.jsonl",
    maximum_text_characters=0,
    maximum_token_count=512,
    chunk_example_count=32,
    train_ratio=0.9,
    seed=42,
    test_reservoir_cap=256,
)
sample = source.prepare_tokenizer_sample(2000)
tok.train(sample, vocab_size=8000)
source.set_tokenizer(tok)   # tokenizer must outlive the source
source.materialize()
model.train_chunks(source, epochs=2, batch_size=32, gradient_accumulation_steps=4)
test = source.test_dataset
```

| Method / property | Notes |
| ----------------- | ----- |
| ctor | `path`, `maximum_text_characters=0`, `maximum_token_count=0`, `chunk_example_count=64`, `train_ratio=0.9`, `seed=42`, `test_reservoir_cap=256` |
| `set_tokenizer` | `BPETokenizer` (must outlive source) |
| `prepare_tokenizer_sample` | First N train-hash texts; rewinds |
| `materialize` | One corpus pass → cached token ids |
| `prepare_test_reservoir` | Build/refresh test reservoir |
| `rewind_train` / `next_train_chunk(out)` | Epoch iteration (`next_train_chunk` → `False` when exhausted) |
| `sort_train_by_length` | Length sort for packing |
| `fill_train_dataset(out)` | Copy all train examples into a dataset |
| `test_dataset` | Reservoir after materialize |
| `file_path`, `chunk_example_count`, `train_ratio` | Config mirrors |
| `train_example_count`, `train_prediction_count`, `is_materialized` | After materialize |

Prefer `model.train_chunks(source, …)` over manual chunk loops when using the packed CUDA train path.

### `ClassificationDataset` / `Sequential`

Small host classifier stack (not the causal LM): embed → mean-pool → Dense → ReLU → Dropout → Dense → Softmax.

| API | Notes |
| --- | ----- |
| `CLASS_CPP` / `CLASS_JSON` / `CLASS_PYTHON` / `CLASS_COUNT` | Module-level label constants |
| `ClassificationExample` | `token_ids`, `target`, `label` |
| `ClassificationDataset.make_one_hot` / `infer_label` / `build` / `build_labeled` | |
| `Sequential(layer1, layer2, optimizer, drop_rate=0.3)` | Two `Dense` layers + `Adam` |
| `Sequential.forward` / `train` / `predict_class` / `accuracy` | Overloads for matrix or embed+pool+dataset |

---

## `SentinelModelConfig`

Native `format: "sentinel-model"` JSON/YAML (field names match safetensors metadata). Flat YAML only (no nested maps/lists).

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
| `from_sentinel_config` | From a `SentinelModelConfig` object |
| `sentinel_config` / `save_sentinel_config` | Snapshot / write |

Smoke: `SENTINEL_MODEL_CONFIG_SMOKE=1`. Examples: [`examples/configs/`](../examples/configs/), [`train_from_config.py`](../examples/python/train_from_config.py).

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

Or size / load from a native **`sentinel-model`** JSON/YAML config:

```python
model = S.LanguageModel.from_config("examples/configs/tiny.json")
# dict / SentinelModelConfig also work
model.save_sentinel_config("out/model.yaml")
```

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
| `set_prefer_mixed_precision` | `(enabled: bool) -> None` | FP16 GEMMs when CUDA train is on |
| `set_prefer_muon` | `(enabled: bool) -> None` | Muon on hidden 2D weights |
| `set_muon_ns_steps` | `(steps: int) -> None` | Newton–Schulz steps |
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
| `set_logit_chunk_rows` | `(rows: int) -> None` | Chunked CE / LM-head |
| `set_tie_embedding` | `(enabled: bool) -> None` | Default on |
| `tie_embedding` | `bool` (ro) | |
| `apply_vram_pack_budget` | `(free_fraction=0.70, safety_reserve_bytes=…) -> None` | Recompute pack from free VRAM |
| `parameter_count` | `int` (ro) | Trainable elements (tied head counted once) |
| `intermediate_size` / `rope_theta` / `use_bias` / `kv_head_count` | ro | Arch mirrors |
| `max_position` / `block_count` | `int` (ro) | |
| `optimizer` | `Adam` (ro ref) | Host Adam used by `apply_gradients` |
| `token_embedding` / `final_norm` / `output_projection` | layer refs | Inspect / mutate |
| `lm_head_weight` | `Matrix` (ro ref) | Embedding when tied |
| `token_embedding_state` / `final_norm_gamma_state` | `AdamState` | |
| `block(i)` | `TransformerBlock` | |

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
| `train_chunks` | `(source, epochs=1, log_every_epochs=1, batch_size=32, gradient_accumulation_steps=4) -> None` | Streaming `LanguageModelChunkSource` |
| `forward` | `(token_ids) -> Matrix` | Logits `vocab × seq` (CUDA mirror when enabled) |
| `example_loss` / `average_loss` | example or dataset → `float` | |
| `accumulate_example` / `apply_gradients` / `train_step` | Host custom-loop step API | See [Mid-level ops](#mid-level-ops) |
| `generate` | `(prompt_token_ids, new_token_count, temperature=1.0, top_k=40, seed=42) -> list[int]` | Returns **new** tokens only. `temperature <= 0` → greedy |

### Checkpoints

| Method | Signature | Notes |
| ------ | --------- | ----- |
| `save_checkpoint` | `(path, include_optimizer=True) -> None` | Native `.snlm` |
| `load_checkpoint` | `(path) -> None` | `.snlm` or `.safetensors` |
| `save_safetensors` | `(path) -> None` | Weights + arch metadata (Sentinel tensor names) |
| `load_safetensors` | `(path) -> None` | Architecture must already match |
| `load_huggingface` | `(path, learning_rate=3e-4) -> LanguageModel` | Static: HF dir → sized model + weights; see [huggingface.md](huggingface.md) |
| `save_huggingface` | `(path, model_type="llama", tokenizer_source_directory="", weight_format="safetensors") -> None` | Export Transformers-compatible dir. `weight_format`: `"safetensors"` \| `"bin"` \| `"both"` |
| `from_config` / `load_sentinel_model` / `from_sentinel_config` | → `LanguageModel` | Native `sentinel-model` JSON/YAML (+ optional weights) |
| `sentinel_config` / `save_sentinel_config` | snapshot / write native config | |

Rebuild a model with the **same** dims before `load_safetensors`. Tokenizer is separate (`BPETokenizer` / `HfTokenizer`).

### Probe

```python
tok_s = model.probe_cuda_packed_train_tokens_per_second(512, warmup_steps=3, timed_steps=4)
model.probe_cuda_train_step_profile(512, warmup_steps=2, timed_steps=4)
```

Requires `enable_cuda_train`. Unset `SENTINEL_PHASE_TRACE` when quoting tok/s.

---

## Mid-level ops

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
loss = model.train_step(dataset.examples)            # Python helper in __init__.py

attn = S.CausalSelfAttention.create(64, 4, 128, kv_head_count=2)
ffn = S.FeedForward.create_with_intermediate_size(64, 96)
opt = S.Spulse(learning_rate=1e-2)                   # host SPULSE
file = S.safetensors_load("w.safetensors")
```

### `Matrix`

Host float matrix (`rows` × `cols`, row-major).

| API | Notes |
| --- | ----- |
| ctor / `from_list` / `to_list` | Flat row-major floats |
| `to_numpy` / `from_numpy` | Optional numpy helpers (added in `__init__.py`) |
| `rows`, `cols`, `shape`, `empty`, `at`, `set`, `fill`, `resize`, `ensure_size` | |
| static `zeros_like`, `zero_in_place`, `transpose`, `add`, `subtract`, `scale`, `multiply`, `multiply_elementwise`, `add_in_place`, `scale_in_place`, `gemm` | |

### Activations / losses / init

| Type | API |
| ---- | --- |
| `Softmax` | `apply`, `apply_into` (column-wise) |
| `SiLU` | `apply_into`, `derivative_into` |
| `ReLU` | `apply`, `apply_into`, `derivative`, `derivative_into` |
| `CrossEntropy` | `loss`, `gradient` (one-hot targets) |
| `MSE` | `loss`, `gradient` |
| `UniformInit` | Symmetric fill helpers |

### Host optimizers

| Type | Notes |
| ---- | ----- |
| `Adam` / `AdamState` | `step`, `update`; `model.optimizer` is the LM Adam |
| `SGD` | `update(parameter, gradient)` |
| `MuonState` | Host Muon moment buffer |
| `Spulse` / `SpulseState` | Host dual-horizon SPULSE (`CudaSpulse::updateHost`) |

### Layers

| Type | Notes |
| ---- | ----- |
| `Embedding` | `forward` / backward into weight grad |
| `Dense` | `z = W @ x + b` |
| `Dropout` / `MeanPool` | fwd/bwd |
| `RMSNorm` / `RMSNormCache` | |
| `RotaryEmbedding` | apply / backward helpers |
| `CausalSelfAttention` / `CausalSelfAttentionCache` | `create(…, kv_head_count=…)`; GQA + sparse knobs |
| `FeedForward` / `FeedForwardCache` | `create` / `create_with_intermediate_size` |
| `TransformerBlock` / caches / `TransformerBlockGradients` | `.attention` / `.feed_forward` accessors |

### SafeTensors helpers

| API | Notes |
| --- | ----- |
| `safetensors_load(path) -> SafeTensorsFile` | F32/BF16/F16 → host F32 |
| `safetensors_save(path, file)` | Writes F32 |
| `is_safetensors_file(path)` | |
| `SafeTensorsFile` | `metadata`, `tensor_names`, `has_tensor`, `get_tensor`, `set_tensor`, `put_matrix` |

`LanguageModel.save_safetensors` / `load_safetensors` remain the high-level checkpoint path.

`accumulate_example` / `apply_gradients` / `train_step` / host `Spulse.update` are **host** paths. Packed CUDA train stays on `enable_cuda_train` + `train()` / `train_chunks()`. `enable_cuda()` still accelerates `forward` / `generate`.

Examples: [`custom_train_loop.py`](../examples/python/custom_train_loop.py), [`custom_layers_demo.py`](../examples/python/custom_layers_demo.py).

---

## Typical recipes

**Tiny CPU/GPU toy** — [`examples/python/train_tiny.py`](../examples/python/train_tiny.py)

**From JSON/YAML config** — [`examples/python/train_from_config.py`](../examples/python/train_from_config.py) + [`examples/configs/`](../examples/configs/)

**Custom host train loop** — [`examples/python/custom_train_loop.py`](../examples/python/custom_train_loop.py)

**Layer / SafeTensors / Spulse ops** — [`examples/python/custom_layers_demo.py`](../examples/python/custom_layers_demo.py)

**JSONL (in-memory)** — [`examples/python/train_jsonl.py`](../examples/python/train_jsonl.py)

**Streaming chunks** — `LanguageModelChunkSource` + `train_chunks` (see above)

**HuggingFace fine-tune + export** — [`examples/python/finetune_hf.py`](../examples/python/finetune_hf.py) (details: [huggingface.md](huggingface.md))

**Large model on 16 GB (HostSGD)**

```python
model.set_prefer_flash_attention(True)
model.set_prefer_muon(False)
model.set_prefer_host_sgd(True)
model.enable_cuda_train()
model.set_activation_checkpoint_mode(S.ActivationCheckpointMode.Full)
```

---

## Not exposed in Python (yet)

| C++ | Status |
| --- | ------ |
| Direct `CudaMatrix` / device Muon/Adam/SPULSE objects | Prefer host `Matrix` + high-level CUDA train |
| `HuggingFace::loadConfig` / `loadMappedWeights` / `resolveModelDirectory` | C++ only; Python uses `load_huggingface` / `HfTokenizer` / Hub via `finetune_hf.py` |
| `ArrowChunkReader` as a standalone type | Used internally by `LanguageModelChunkSource` |

Tokenizer I/O (`.sbpe`) and streaming (`LanguageModelChunkSource`) **are** exposed — see above.
