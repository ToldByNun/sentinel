"""DeepSpeed paper runner — real deepspeed.initialize (not a path-shadowed stub).

fair: ZeRO-2, params/opt on GPU
feat: ZeRO-3 + CPU offload of params + optimizer

Requires: pip install deepspeed
Launch-style env is set for single-GPU (WORLD_SIZE=1).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import gpu_used_mib, hardware, host_rss_mib, is_oom, print_report, progress
from paper_config import (
    LEARNING_RATE,
    PACK_EXAMPLES,
    SEQ,
    TIMED_STEPS,
    VOCAB,
    WARMUP_STEPS,
    ProfileName,
    resolve,
)
from torch_model import PaperLM, make_batch, param_count


def _import_deepspeed():
    """Import the real DeepSpeed package; fail clearly if shadowed or missing."""
    # Ensure benchmarks/ (parent of _lib) is NOT preferred over site-packages for 'deepspeed'.
    bench_root = str(ROOT.parent)
    sys.path[:] = [p for p in sys.path if os.path.abspath(p) != os.path.abspath(bench_root)]

    try:
        import deepspeed
    except ImportError as ex:
        raise ImportError(
            "DeepSpeed is not installed. Install with: pip install deepspeed"
        ) from ex

    if not hasattr(deepspeed, "initialize"):
        raise RuntimeError(
            f"Imported wrong module named deepspeed from {getattr(deepspeed, '__file__', deepspeed)!r}. "
            "Remove/rename benchmarks folders that shadow the package, or fix PYTHONPATH."
        )
    return deepspeed


def _ds_config(*, memory_efficient: bool) -> dict:
    zero: dict = {
        "stage": 3 if memory_efficient else 2,
        "contiguous_gradients": True,
        "overlap_comm": True,
        "reduce_scatter": True,
    }
    if memory_efficient:
        zero["offload_param"] = {"device": "cpu", "pin_memory": True}
        zero["offload_optimizer"] = {"device": "cpu", "pin_memory": True}
        zero["stage3_param_persistence_threshold"] = 0
    return {
        "train_micro_batch_size_per_gpu": PACK_EXAMPLES,
        "gradient_accumulation_steps": 1,
        "steps_per_print": 10_000_000,
        "optimizer": {
            "type": "Adam",
            "params": {"lr": LEARNING_RATE, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.0},
        },
        "fp16": {"enabled": True, "loss_scale": 0, "initial_scale_power": 16},
        "zero_optimization": zero,
        "wall_clock_breakdown": False,
    }


def _setup_single_gpu_dist(deepspeed) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29522")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    # DeepSpeed/torch distributed on Windows typically needs gloo.
    backend = "nccl" if sys.platform.startswith("linux") else "gloo"
    progress(f"deepspeed.init_distributed backend={backend}")
    deepspeed.init_distributed(dist_backend=backend)


def run(model_id: str, profile: ProfileName = "fair") -> int:
    import torch

    spec, train = resolve(model_id, profile)
    progress("collecting hardware")
    hw = hardware()
    mode = "feat/ZeRO-3+CPU-offload" if train.memory_efficient else "fair/ZeRO-2"
    print(f"=== DeepSpeed {spec.label}  profile={profile} ({mode}) ===", flush=True)
    for k, v in hw.items():
        print(f"{k}: {v}", flush=True)
    print(
        f"config: Adam lr={LEARNING_RATE} fp16 micro_batch={PACK_EXAMPLES} seq={SEQ} "
        f"tokens/step={SEQ * PACK_EXAMPLES} grad_ckpt={train.gradient_checkpointing}",
        flush=True,
    )

    tok_s = peak_vram = host_ram = step_ms = None
    status = "fail"
    try:
        deepspeed = _import_deepspeed()
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")

        _setup_single_gpu_dist(deepspeed)

        progress("constructing PaperLM (CPU) then deepspeed.initialize")
        model = PaperLM(
            VOCAB,
            spec.embedding_dim,
            spec.block_count,
            spec.head_count,
            spec.maximum_position_count,
            train.gradient_checkpointing,
        )
        progress(f"model ready params~{param_count(model) / 1e6:.1f}M")

        ds_cfg = _ds_config(memory_efficient=train.memory_efficient)
        progress(f"deepspeed.initialize ({mode})")
        engine, _, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=[p for p in model.parameters() if p.requires_grad],
            config=ds_cfg,
        )
        device = engine.device

        peak_vram = gpu_used_mib() or 0.0
        host_ram = host_rss_mib() or 0.0

        def one_step() -> None:
            x, y = make_batch(VOCAB, PACK_EXAMPLES, SEQ, device)
            loss = engine(x, y)
            engine.backward(loss)
            engine.step()

        for i in range(WARMUP_STEPS):
            progress(f"warmup {i + 1}/{WARMUP_STEPS}")
            one_step()
            torch.cuda.synchronize()
            peak_vram = max(peak_vram, gpu_used_mib() or 0.0)
            host_ram = max(host_ram, host_rss_mib() or 0.0)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for i in range(TIMED_STEPS):
            progress(f"timed {i + 1}/{TIMED_STEPS}")
            one_step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        tokens = SEQ * PACK_EXAMPLES * TIMED_STEPS
        tok_s = tokens / elapsed if elapsed > 0 else 0.0
        step_ms = (elapsed / TIMED_STEPS) * 1000.0
        peak_vram = max(peak_vram, gpu_used_mib() or 0.0)
        host_ram = max(host_ram, host_rss_mib() or 0.0)
        status = "success"
        progress(f"finished OK tok/s={tok_s:.0f}")
    except Exception as ex:
        status = "oom" if is_oom(ex) else "fail"
        peak_vram = max(peak_vram or 0.0, gpu_used_mib() or 0.0) if peak_vram is not None else gpu_used_mib()
        host_ram = max(host_ram or 0.0, host_rss_mib() or 0.0) if host_ram is not None else host_rss_mib()
        progress(f"{status}: {ex}")
        print(f"error: {ex}", flush=True)
    finally:
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            pass

    print_report(
        framework="deepspeed",
        label=spec.label,
        profile=profile,
        hw=hw,
        tok_s=tok_s,
        peak_vram=peak_vram,
        host_ram=host_ram,
        step_ms=step_ms,
        status=status,
    )
    return 0 if status == "success" else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_id", choices=sorted(__import__("paper_config").MODELS))
    parser.add_argument("--profile", choices=("fair", "feat"), default="fair")
    args = parser.parse_args(argv)
    return run(args.model_id, args.profile)


if __name__ == "__main__":
    raise SystemExit(main())
