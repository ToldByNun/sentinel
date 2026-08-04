"""FSDP paper runner — real FULL_SHARD via 2-process group (single GPU).

world_size=1 makes PyTorch silently switch to NO_SHARD; this runner always
spawns world_size=2 on cuda:0 so sharding is genuine.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import hardware, print_report, progress
from fsdp_launch import FSDP_WORLD_SIZE, run_fsdp_multiprocess
from paper_config import (
    LEARNING_RATE,
    PACK_EXAMPLES,
    SEQ,
    VOCAB,
    ProfileName,
    resolve,
)


def run(model_id: str, profile: ProfileName = "fair") -> int:
    spec, train = resolve(model_id, profile)
    progress("collecting hardware")
    hw = hardware()
    mode = "feat/FULL_SHARD+CPUOffload" if train.memory_efficient else "fair/FULL_SHARD"
    print(f"=== FSDP {spec.label}  profile={profile} ({mode}) ===", flush=True)
    for k, v in hw.items():
        print(f"{k}: {v}", flush=True)
    print(
        f"config: Adam lr={LEARNING_RATE} fp16 seq={SEQ} global_pack={PACK_EXAMPLES} "
        f"fsdp_world_size={FSDP_WORLD_SIZE} (2 ranks / 1 GPU — required for FULL_SHARD) "
        f"grad_ckpt={train.gradient_checkpointing}",
        flush=True,
    )

    progress(f"spawn {FSDP_WORLD_SIZE} FSDP ranks on cuda:0")
    result = run_fsdp_multiprocess(
        vocab=VOCAB,
        embedding_dim=spec.embedding_dim,
        block_count=spec.block_count,
        head_count=spec.head_count,
        max_pos=spec.maximum_position_count,
        gradient_checkpointing=train.gradient_checkpointing,
        memory_efficient=train.memory_efficient,
        master_port="29533",
    )
    if result.error and result.status != "success":
        print(f"error: {result.error}", flush=True)
    if result.sharding:
        progress(f"confirmed sharding_strategy={result.sharding}")

    print_report(
        framework="fsdp",
        label=spec.label,
        profile=profile,
        hw=hw,
        tok_s=result.tok_s,
        peak_vram=result.peak_vram,
        host_ram=result.host_ram,
        step_ms=result.step_ms,
        status=result.status,
    )
    return 0 if result.status == "success" else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_id", choices=sorted(__import__("paper_config").MODELS))
    parser.add_argument("--profile", choices=("fair", "feat"), default="fair")
    args = parser.parse_args(argv)
    return run(args.model_id, args.profile)


if __name__ == "__main__":
    raise SystemExit(main())
