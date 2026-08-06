"""Mid-scale BAO probe — Auto policy vs forced HostFusedHalfAdam (~8×768 / ~12×768).

Shows that BAO Auto picks GpuInt8Adam when VRAM fits (faster resident path), and
HostFusedHalfAdam when forced / VRAM-tight.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import sentinel as S

from common import gpu_used_mib, hardware, progress

VOCAB = 4096
EMBED = 768
HEADS = 12
POS = 512
SEQ = 256
LR = 3e-4
WARMUP = 2  # allow CUDA graph capture warmups on Auto/GpuInt8
TIMED = 4


def run_once(label: str, blocks: int, mode: str) -> float:
    progress(f"{label}: construct L={blocks} d={EMBED}")
    model = S.LanguageModel(VOCAB, EMBED, POS, LR, blocks, HEADS)
    print(f"params~{model.parameter_count / 1e6:.1f}M", flush=True)
    model.enable_cuda()
    model.set_prefer_flash_attention(True)
    model.set_prefer_muon(False)
    if mode == "auto":
        model.set_prefer_bao(True)
    elif mode == "host_adam":
        model.set_bao_mode(S.BaoMode.HostFusedHalfAdam)
    elif mode == "cpu_adam":
        model.set_prefer_cpu_adam_offload(True)
    else:
        raise ValueError(mode)
    model.enable_cuda_train()
    # GpuInt8 Auto: keep CUDA graphs. Host FP16w paths: graphs off.
    use_graphs = mode == "auto" and model.bao_mode_resolved == S.BaoMode.GpuInt8Adam
    model.set_prefer_train_graph(use_graphs)
    print(
        f"{label}: bao_resolved={model.bao_mode_resolved} max_pack={model.max_packed_columns} "
        f"graphs={'on' if use_graphs else 'off'} VRAM~{gpu_used_mib():.0f} MiB",
        flush=True,
    )
    t0 = time.perf_counter()
    tok_s = float(model.probe_cuda_packed_train_tokens_per_second(SEQ, WARMUP, TIMED))
    wall = time.perf_counter() - t0
    pack = max(1, model.max_packed_columns // SEQ)
    if tok_s <= 0 and wall > 0:
        tok_s = (SEQ * pack * TIMED) / wall
    print(
        f"RESULT {label} tok/s={tok_s:.0f} pack={pack} bao={model.bao_mode_resolved} "
        f"VRAM~{gpu_used_mib():.0f} MiB",
        flush=True,
    )
    return tok_s


def main() -> int:
    parser = argparse.ArgumentParser(description="BAO mid-scale probe")
    parser.add_argument("--blocks", type=int, default=8, choices=(8, 12))
    parser.add_argument(
        "--mode",
        default="both",
        choices=("auto", "host_adam", "cpu_adam", "both"),
        help="auto=BAO policy; host_adam=force HostFusedHalfAdam; both=compare Auto vs host_adam",
    )
    args = parser.parse_args()

    if not S.cuda_available():
        print("CUDA unavailable", flush=True)
        return 1

    print("hardware:", hardware(), flush=True)
    if args.mode == "both":
        auto_tps = run_once("auto", args.blocks, "auto")
        host_tps = run_once("host_adam", args.blocks, "host_adam")
        print(
            f"COMPARE Auto={auto_tps:.0f} vs HostFusedHalfAdam={host_tps:.0f} tok/s "
            f"(Auto should match or beat host when GpuInt8 fits)",
            flush=True,
        )
        return 0 if auto_tps > 0 and host_tps > 0 else 1

    return 0 if run_once(args.mode, args.blocks, args.mode) > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
