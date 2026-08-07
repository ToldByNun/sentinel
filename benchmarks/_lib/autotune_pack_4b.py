"""L2 lever — pack-budget autotune for the 4B host-SGD (feat) train path.

Sweeps ``max_packed_columns`` and reports tok/s for each setting so you can pick
the pack budget that best fills the GPU without OOM. More tokens/step amortize
per-step host-SGD + launch overhead, but cost VRAM — this finds the peak.

Usage (on a CUDA machine):

    python benchmarks/_lib/autotune_pack_4b.py                # sweep around auto
    python benchmarks/_lib/autotune_pack_4b.py --seq 256      # match 4B_PoC feat
    python benchmarks/_lib/autotune_pack_4b.py --columns 4096,6144,8192,10240

Combine with the other levers via env, e.g.:

    SENTINEL_CKPT_KEEP_FREE_MIB=1024 SENTINEL_MAX_ASYNC_HOST_UPDATE=6 \
        python benchmarks/_lib/autotune_pack_4b.py --seq 256

Quote tok/s with SENTINEL_PHASE_TRACE unset (the trace adds syncs).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import sentinel as S

from common import gpu_used_mib, hardware, progress

VOCAB = 32000
EMBED = 3072
BLOCKS = 34
HEADS = 48
POS = 2048
LR = 3e-4


def build_model() -> "S.LanguageModel":
    model = S.LanguageModel(VOCAB, EMBED, POS, LR, BLOCKS, HEADS)
    model.enable_cuda()
    model.set_prefer_flash_attention(True)
    model.set_prefer_muon(False)
    model.set_prefer_host_sgd(True)
    model.enable_cuda_train()
    model.set_prefer_train_graph(False)
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq", type=int, default=256, help="sequence length (4B_PoC feat uses 256)")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--timed", type=int, default=4)
    parser.add_argument(
        "--columns",
        type=str,
        default="",
        help="comma-separated max_packed_columns to try; default sweeps multiples of the auto budget",
    )
    args = parser.parse_args()

    if not S.cuda_available():
        print("CUDA unavailable", flush=True)
        return 1

    print("hardware:", hardware(), flush=True)
    progress("constructing ~4B model (CPU)")
    model = build_model()
    params_b = model.parameter_count / 1e9
    auto_cols = int(model.max_packed_columns)
    print(f"params~{params_b:.2f}B  sbao={model.sbao_mode_resolved}  auto max_packed_columns={auto_cols}", flush=True)

    if args.columns.strip():
        candidates = [int(c) for c in args.columns.split(",") if c.strip()]
    else:
        # Multiples of the auto budget, snapped to a whole number of sequences.
        factors = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
        candidates = sorted({max(args.seq, (int(auto_cols * f) // args.seq) * args.seq) for f in factors})

    print(f"seq={args.seq}  candidates(columns)={candidates}", flush=True)

    best = (0.0, 0, 0)  # tok/s, actual_columns, pack
    for requested in candidates:
        model.set_max_packed_columns(requested)
        actual = int(model.max_packed_columns)
        pack = max(1, actual // args.seq)
        # Warm discard (cold WDDM first pass is noisy), then the timed probe.
        _ = float(model.probe_cuda_packed_train_tokens_per_second(args.seq, 2, 2))
        try:
            tok_s = float(model.probe_cuda_packed_train_tokens_per_second(args.seq, args.warmup, args.timed))
        except Exception as exc:  # noqa: BLE001 - a candidate may OOM; keep sweeping
            print(f"  columns_req={requested:6d} actual={actual:6d} pack={pack:4d} -> FAILED ({exc})", flush=True)
            continue
        vram = gpu_used_mib()
        print(
            f"  columns_req={requested:6d} actual={actual:6d} pack={pack:4d} "
            f"tokens/step={args.seq * pack:6d} tok/s={tok_s:8.0f} VRAM~{vram:.0f} MiB",
            flush=True,
        )
        if tok_s > best[0]:
            best = (tok_s, actual, pack)

    if best[0] > 0:
        print(
            f"BEST tok/s={best[0]:.0f} at max_packed_columns={best[1]} (pack={best[2]}, seq={args.seq})",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
