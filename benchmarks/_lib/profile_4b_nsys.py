"""One-step 4B host-SGD profile target for nsys (warmup outside profile window)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import sentinel as S

VOCAB, EMBED, BLOCKS, HEADS, POS, SEQ, LR = 32000, 3072, 34, 48, 2048, 512, 3e-4


def main() -> int:
    if not S.cuda_available():
        print("CUDA unavailable", flush=True)
        return 1
    model = S.LanguageModel(VOCAB, EMBED, POS, LR, BLOCKS, HEADS)
    model.enable_cuda()
    model.set_prefer_flash_attention(True)
    model.set_prefer_muon(False)
    model.set_prefer_cpu_adam_offload(True)
    model.enable_cuda_train()
    model.set_prefer_host_sgd(True)
    model.set_prefer_train_graph(False)
    print(f"max_packed_columns={model.max_packed_columns}", flush=True)
    # Warmup outside nsys capture if launched with --capture-range; here always 1+1.
    _ = model.probe_cuda_packed_train_tokens_per_second(SEQ, 1, 1)
    print("profile step done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
