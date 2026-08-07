"""PyTorch paper runner — idiomatic single-GPU training.

fair: vanilla nn.Module on CUDA, Adam, autocast FP16, gradient checkpointing, SDPA
feat: Linux → FSDP FULL_SHARD + CPUOffload (2 ranks / 1 GPU)
      Windows → host Adam offload (FSDP AVs on this WDDM/Blackwell stack)
"""

from __future__ import annotations

import argparse
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


def _run_fair(spec, train, hw) -> tuple:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    device = torch.device("cuda")

    progress("fair: vanilla PyTorch module on CUDA")
    model = PaperLM(
        VOCAB,
        spec.embedding_dim,
        spec.block_count,
        spec.head_count,
        spec.maximum_position_count,
        train.gradient_checkpointing,
    ).to(device)
    progress(f"model ready params~{param_count(model) / 1e6:.1f}M")
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler("cuda", enabled=train.fp16_amp)

    peak_vram = gpu_used_mib() or 0.0
    host_ram = host_rss_mib() or 0.0

    def one_step() -> None:
        x, y = make_batch(VOCAB, PACK_EXAMPLES, SEQ, device)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=train.fp16_amp):
            loss = model(x, y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    model.train()
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
    return tok_s, peak_vram, host_ram, step_ms


def _run_feat_host_adam(spec, train) -> tuple:
    """Windows-safe feat: FP16 GPU weights + grads/Adam on CPU (Sentinel-like).

    FSDP AVs on this WDDM/Blackwell stack. An FP32 module on GPU made peak VRAM
    look like ``fair`` (~weights+grads in FP32). Mirror Sentinel feat residency:
    FP16 weights on device, FP32 masters + Adam moments on host.
    """
    import torch
    import torch.nn as nn

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()

    progress("feat: FP16 GPU weights + host Adam (Windows — no FSDP)")
    model = (
        PaperLM(
            VOCAB,
            spec.embedding_dim,
            spec.block_count,
            spec.head_count,
            spec.maximum_position_count,
            train.gradient_checkpointing,
        )
        .to(device)
        .half()
    )
    progress(f"model ready params~{param_count(model) / 1e6:.1f}M dtype=fp16")

    gpu_params = [p for p in model.parameters() if p.requires_grad]
    cpu_masters = [nn.Parameter(p.detach().float().cpu()) for p in gpu_params]
    opt = torch.optim.Adam(cpu_masters, lr=LEARNING_RATE)
    # GradScaler.unscale_ requires FP32 .grad; FP16 weights produce FP16 grads.
    # Loss scaling is optional for this throughput/VRAM bench — skip scaler.

    peak_vram = gpu_used_mib() or 0.0
    host_ram = host_rss_mib() or 0.0

    def _peak() -> float:
        alloc = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
        return max(gpu_used_mib() or 0.0, alloc)

    def one_step() -> None:
        x, y = make_batch(VOCAB, PACK_EXAMPLES, SEQ, device)
        for p in gpu_params:
            p.grad = None
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            loss = model(x, y)
        loss.float().backward()
        for master, p in zip(cpu_masters, gpu_params):
            if p.grad is None:
                master.grad = None
            else:
                master.grad = p.grad.detach().float().cpu()
                p.grad = None
        opt.step()
        opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            for master, p in zip(cpu_masters, gpu_params):
                p.copy_(master.to(device=p.device, dtype=p.dtype, non_blocking=True))
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    model.train()
    for i in range(WARMUP_STEPS):
        progress(f"warmup {i + 1}/{WARMUP_STEPS}")
        one_step()
        peak_vram = max(peak_vram, _peak())
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
    peak_vram = max(peak_vram, _peak())
    host_ram = max(host_ram, host_rss_mib() or 0.0)
    return tok_s, peak_vram, host_ram, step_ms


def _run_feat(spec, train, hw) -> tuple:
    """Memory-efficient PyTorch path."""
    if sys.platform.startswith("win"):
        return _run_feat_host_adam(spec, train)

    from fsdp_launch import run_fsdp_multiprocess

    progress("feat: spawn FSDP FULL_SHARD + CPUOffload (world_size=2, 1 GPU)")
    result = run_fsdp_multiprocess(
        vocab=VOCAB,
        embedding_dim=spec.embedding_dim,
        block_count=spec.block_count,
        head_count=spec.head_count,
        max_pos=spec.maximum_position_count,
        gradient_checkpointing=train.gradient_checkpointing,
        memory_efficient=True,
        master_port="29511",
        require_full_shard=False,
    )
    if result.status != "success":
        raise RuntimeError(result.error or f"FSDP feat failed ({result.status})")
    progress(f"feat sharding={result.sharding}")
    return result.tok_s, result.peak_vram, result.host_ram, result.step_ms


def run(model_id: str, profile: ProfileName = "fair") -> int:
    spec, train = resolve(model_id, profile)
    progress("collecting hardware")
    hw = hardware()
    if train.memory_efficient:
        mode = (
            "feat/host-Adam-offload"
            if sys.platform.startswith("win")
            else "feat/FSDP+CPUOffload"
        )
    else:
        mode = "fair/vanilla CUDA"
    print(f"=== PyTorch {spec.label}  profile={profile} ({mode}) ===", flush=True)
    for k, v in hw.items():
        print(f"{k}: {v}", flush=True)
    print(
        f"config: Adam lr={LEARNING_RATE} seq={SEQ} pack={PACK_EXAMPLES} "
        f"tokens/step={SEQ * PACK_EXAMPLES} grad_ckpt={train.gradient_checkpointing}",
        flush=True,
    )

    tok_s = peak_vram = host_ram = step_ms = None
    status = "fail"
    try:
        if train.memory_efficient:
            tok_s, peak_vram, host_ram, step_ms = _run_feat(spec, train, hw)
        else:
            tok_s, peak_vram, host_ram, step_ms = _run_fair(spec, train, hw)
        status = "success"
        progress(f"finished OK tok/s={tok_s:.0f}")
    except Exception as ex:
        status = "oom" if is_oom(ex) else "fail"
        peak_vram = max(peak_vram or 0.0, gpu_used_mib() or 0.0) if peak_vram is not None else gpu_used_mib()
        host_ram = max(host_ram or 0.0, host_rss_mib() or 0.0) if host_ram is not None else host_rss_mib()
        progress(f"{status}: {ex}")
        print(f"error: {ex}", flush=True)

    print_report(
        framework="pytorch",
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
