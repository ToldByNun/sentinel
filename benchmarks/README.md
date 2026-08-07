# Paper benchmarks

Shared code lives in [`_lib/`](_lib/) so folder names (`deepspeed/`, `sentinel/`, …) **cannot shadow** installed packages.

Workload ([`_lib/paper_config.py`](_lib/paper_config.py)): vocab 32k · **256×16** tokens/step · Adam `3e-4` · FP16 · warmup 2 / timed 4.

## Profiles

| Profile | Meaning |
| ------- | ------- |
| **fair** | Normal GPU-resident training (may OOM at large sizes) |
| **feat** | Memory-efficient offload path |

| Lib | fair | feat |
| --- | ---- | ---- |
| Sentinel | resident Adam | FP16w + host grads + host SGD |
| PyTorch | vanilla CUDA module | **Linux:** FSDP 2-rank FULL_SHARD + CPUOffload; **Windows:** FP16 GPU weights + host Adam (FSDP AV) |
| DeepSpeed | `deepspeed.initialize` ZeRO-2 | ZeRO-3 + CPU offload (**Linux/WSL**; native Windows → `na`, AV) |
| FSDP | FULL_SHARD via **2 ranks / 1 GPU** (Linux; Windows unsupported) | + CPUOffload |

## Run

```bash
pip install torch
# DeepSpeed Windows install (wheel — sdist builds usually fail):
powershell benchmarks/install_deepspeed_windows.ps1

python benchmarks/sentinel/benchmark.py
python benchmarks/pytorch/benchmark.py
python benchmarks/deepspeed/benchmark.py
python benchmarks/fsdp/benchmark.py

python benchmarks/deepspeed/100M.py fair
```

**Windows notes**

| Stack | Status |
| ----- | ------ |
| Sentinel | OK |
| PyTorch feat | FP16 GPU weights + host Adam (~7 GiB @ 1B; FSDP AVs) |
| FSDP | unsupported (AV) — Linux only |
| DeepSpeed | install via wheel OK; **training AVs** on WDDM/Blackwell → `status=na`; use WSL2/Linux for paper cells |

## Indicative Sentinel throughput (RTX 5070 Ti 16 GB)

Desktop WDDM / pack budget matter — re-measure after train-path changes. Headline numbers also live in the root [README](../README.md#performance-indicative).

| Shape | Setup | ~tok/s | Probe |
| ----- | ----- | ------ | ----- |
| ~60M (8×768) | Selective ckpt, FP16 AMP, int8 Adam, flash, auto pack | ~18–20k | harness / suite |
| ~97M (12×768) | ckpt **Off**, int8 Adam, flash, pack ~3840–4096 | ~25–26k | `runScale100M` / Muon probe table |
| ~97M Selective | same + Selective ckpt | ~11–13k | Attn recompute cost |
| ~4B (34×3072, H=48) | SBAO **HostFusedHalfSgd**, Full ckpt, seq 512, pack 8 | ~950–1000 | [`_lib/probe_4b_safe.py`](_lib/probe_4b_safe.py) |

```bash
# Unset SENTINEL_PHASE_TRACE for real tok/s
python benchmarks/_lib/probe_4b_safe.py
```

### Adam vs Muon (12×768)

| Optimizer | ckpt | ~tok/s |
| --------- | ---- | ------ |
| Adam | Off | ~25–26k |
| Adam | Selective | ~11–13k |
| Muon | Off / Sel | lower (Newton–Schulz on hidden weights) |

Muon adds NS cost — do not read a Muon probe as an Adam regression.

### 4B notes

- VRAM ~12.4 GiB; host holds FP32 masters (~7–8 GiB); no Adam `m`/`v` on HostSGD
- Warm multi-run means are noisy on WDDM; quote means, not single outliers
- SBAO Auto picks GpuInt8Adam when VRAM fits, else HostFusedHalfAdam, else HostFusedHalfSgd
