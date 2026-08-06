"""Quick 4B feat probe — Full ckpt + fixed pack (no auto-sweep)."""

from __future__ import annotations

import sys
import time
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
SEQ = 512
LR = 3e-4
WARMUP = 3
TIMED = 4


def main() -> int:
    if not S.cuda_available():
        print("CUDA unavailable", flush=True)
        return 1
    hw = hardware()
    print("hardware:", hw, flush=True)
    progress("constructing ~4B model (CPU)")
    model = S.LanguageModel(VOCAB, EMBED, POS, LR, BLOCKS, HEADS)
    print(f"params~{model.parameter_count / 1e9:.2f}B", flush=True)
    model.enable_cuda()
    model.set_prefer_flash_attention(True)
    model.set_prefer_muon(False)
    # Host-SGD before enable_cuda_train — avoids HostAdam pack-budget thrash (free≈0 → minCols).
    progress("host-SGD (Full ckpt + alloc-only pack, no microstep sweep)")
    model.set_prefer_host_sgd(True)
    model.enable_cuda_train()
    model.set_prefer_train_graph(False)
    print(f"max_packed_columns={model.max_packed_columns}", flush=True)
    print(f"bao={model.bao_mode_resolved} VRAM used~{gpu_used_mib():.0f} MiB before probe", flush=True)
    # Discard pass: pin freelist + boost clocks before timed window (cold WDDM first-pass is noisy).
    _ = float(model.probe_cuda_packed_train_tokens_per_second(SEQ, 1, 1))
    t0 = time.perf_counter()
    tok_s = float(model.probe_cuda_packed_train_tokens_per_second(SEQ, WARMUP, TIMED))
    wall = time.perf_counter() - t0
    pack = max(1, model.max_packed_columns // SEQ)
    tokens = SEQ * pack * TIMED
    if tok_s <= 0 and wall > 0:
        tok_s = tokens / wall
    print(
        f"RESULT tok/s={tok_s:.0f} pack={pack} seq={SEQ} tokens/step={SEQ * pack} "
        f"VRAM~{gpu_used_mib():.0f} MiB wall={wall:.1f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
