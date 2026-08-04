"""Shared multi-process FSDP launcher.

PyTorch silently downgrades FULL_SHARD → NO_SHARD when world_size==1.
Paper benches therefore always run with world_size>=2 (2 ranks on one GPU).
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from typing import Any

from paper_config import LEARNING_RATE, PACK_EXAMPLES, SEQ, TIMED_STEPS, WARMUP_STEPS

FSDP_WORLD_SIZE = 2


@dataclass
class FsdpResult:
    status: str
    tok_s: float | None = None
    peak_vram: float | None = None
    host_ram: float | None = None
    step_ms: float | None = None
    error: str | None = None
    sharding: str | None = None


def _rank0(msg: str) -> None:
    import torch.distributed as dist

    if not dist.is_initialized() or dist.get_rank() == 0:
        print(f"[progress] {msg}", flush=True)


def fsdp_worker(
    rank: int,
    world_size: int,
    vocab: int,
    embedding_dim: int,
    block_count: int,
    head_count: int,
    max_pos: int,
    gradient_checkpointing: bool,
    memory_efficient: bool,
    master_port: str,
    result_list: Any,
) -> None:
    import torch
    import torch.distributed as dist
    from torch.distributed.fsdp import CPUOffload
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import MixedPrecision, ShardingStrategy

    from common import gpu_used_mib, host_rss_mib, is_oom
    from torch_model import Block, PaperLM, make_batch, param_count

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = master_port
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    backend = "nccl" if sys.platform.startswith("linux") else "gloo"
    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    if dist.get_world_size() < 2:
        raise RuntimeError("FSDP paper bench requires world_size>=2 for FULL_SHARD")

    per_rank_batch = max(1, PACK_EXAMPLES // world_size)
    status = "fail"
    tok_s = peak_vram = host_ram = step_ms = None
    sharding_name = None
    err = None

    try:
        _rank0(
            f"FSDP world_size={world_size} backend={backend} "
            f"per_rank_batch={per_rank_batch} (global pack={per_rank_batch * world_size})"
        )
        model = PaperLM(
            vocab,
            embedding_dim,
            block_count,
            head_count,
            max_pos,
            gradient_checkpointing,
        )
        if rank == 0:
            _rank0(f"model ready params~{param_count(model) / 1e6:.1f}M")

        mp = MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
        )
        cpu_offload = CPUOffload(offload_params=True) if memory_efficient else None
        try:
            from torch.distributed.fsdp.wrap import ModuleWrapPolicy

            auto_wrap: Any = ModuleWrapPolicy({Block})
        except Exception:
            auto_wrap = None

        model = FSDP(
            model,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=mp,
            cpu_offload=cpu_offload,
            auto_wrap_policy=auto_wrap,
            device_id=torch.cuda.current_device(),
            use_orig_params=True,
        )
        sharding_name = str(getattr(model, "sharding_strategy", "unknown"))
        if "NO_SHARD" in sharding_name.upper():
            raise RuntimeError(
                f"FSDP fell back to {sharding_name} (world_size={dist.get_world_size()}). "
                "FULL_SHARD required for this paper bench."
            )
        _rank0(f"FSDP sharding_strategy={sharding_name}")

        opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        device = torch.device("cuda")
        peak_vram = gpu_used_mib() or 0.0
        host_ram = host_rss_mib() or 0.0

        def one_step() -> None:
            x, y = make_batch(vocab, per_rank_batch, SEQ, device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                loss = model(x, y)
            loss.backward()
            opt.step()

        for i in range(WARMUP_STEPS):
            _rank0(f"warmup {i + 1}/{WARMUP_STEPS}")
            one_step()
            torch.cuda.synchronize()
            dist.barrier()
            if rank == 0:
                peak_vram = max(peak_vram, gpu_used_mib() or 0.0)
                host_ram = max(host_ram, host_rss_mib() or 0.0)

        dist.barrier()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for i in range(TIMED_STEPS):
            _rank0(f"timed {i + 1}/{TIMED_STEPS}")
            one_step()
        torch.cuda.synchronize()
        dist.barrier()
        elapsed = time.perf_counter() - t0

        tokens = SEQ * per_rank_batch * world_size * TIMED_STEPS
        tok_s = tokens / elapsed if elapsed > 0 else 0.0
        step_ms = (elapsed / TIMED_STEPS) * 1000.0
        if rank == 0:
            peak_vram = max(peak_vram, gpu_used_mib() or 0.0)
            host_ram = max(host_ram, host_rss_mib() or 0.0)
        status = "success"
        _rank0(f"finished OK tok/s={tok_s:.0f} sharding={sharding_name}")
    except Exception as ex:
        status = "oom" if is_oom(ex) else "fail"
        err = str(ex)
        if rank == 0:
            peak_vram = max(peak_vram or 0.0, gpu_used_mib() or 0.0) if peak_vram is not None else gpu_used_mib()
            host_ram = max(host_ram or 0.0, host_rss_mib() or 0.0) if host_ram is not None else host_rss_mib()
            print(f"[progress] {status}: {ex}", flush=True)
            print(f"error: {ex}", flush=True)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

    if rank == 0:
        result_list.append(
            {
                "status": status,
                "tok_s": tok_s,
                "peak_vram": peak_vram,
                "host_ram": host_ram,
                "step_ms": step_ms,
                "error": err,
                "sharding": sharding_name,
            }
        )


def run_fsdp_multiprocess(
    *,
    vocab: int,
    embedding_dim: int,
    block_count: int,
    head_count: int,
    max_pos: int,
    gradient_checkpointing: bool,
    memory_efficient: bool,
    master_port: str = "29533",
) -> FsdpResult:
    import torch
    import torch.multiprocessing as mp

    if not torch.cuda.is_available():
        return FsdpResult(status="fail", error="CUDA unavailable")

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    manager = mp.Manager()
    result_list = manager.list()
    mp.spawn(
        fsdp_worker,
        args=(
            FSDP_WORLD_SIZE,
            vocab,
            embedding_dim,
            block_count,
            head_count,
            max_pos,
            gradient_checkpointing,
            memory_efficient,
            master_port,
            result_list,
        ),
        nprocs=FSDP_WORLD_SIZE,
        join=True,
    )
    if not result_list:
        return FsdpResult(status="fail", error="FSDP worker returned no result")
    raw = dict(result_list[0])
    return FsdpResult(**raw)
