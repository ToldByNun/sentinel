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
| PyTorch | vanilla CUDA module | FSDP **2-rank** FULL_SHARD + CPUOffload |
| DeepSpeed | `deepspeed.initialize` ZeRO-2 | ZeRO-3 + CPU offload |
| FSDP | FULL_SHARD via **2 ranks / 1 GPU** (rejects NO_SHARD) | + CPUOffload |

## Run

```bash
pip install torch deepspeed   # for non-sentinel stacks

python benchmarks/sentinel/benchmark.py
python benchmarks/pytorch/benchmark.py
python benchmarks/deepspeed/benchmark.py
python benchmarks/fsdp/benchmark.py

python benchmarks/deepspeed/100M.py fair
```
