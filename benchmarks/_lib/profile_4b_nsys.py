"""4B host-SGD nsys target: setup+warmup outside capture, one timed step inside.

Use:
  nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop ...
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import sentinel as S

try:
    from cuda.bindings import runtime as cudart  # type: ignore
except Exception:
    cudart = None

VOCAB, EMBED, BLOCKS, HEADS, POS, SEQ, LR = 32000, 3072, 34, 48, 2048, 512, 3e-4


def _profiler_start() -> None:
    if cudart is not None:
        cudart.cudaProfilerStart()
        return
    # Fallback: ctypes cuda.dll
    import ctypes

    lib = ctypes.CDLL("cudart64_13.dll")
    lib.cudaProfilerStart()


def _profiler_stop() -> None:
    if cudart is not None:
        cudart.cudaProfilerStop()
        return
    import ctypes

    lib = ctypes.CDLL("cudart64_13.dll")
    lib.cudaProfilerStop()


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

    print("warmup outside capture", flush=True)
    _ = model.probe_cuda_packed_train_tokens_per_second(SEQ, 1, 0)

    print("capture start", flush=True)
    _profiler_start()
    try:
        _ = model.probe_cuda_packed_train_tokens_per_second(SEQ, 0, 1)
    finally:
        _profiler_stop()
    print("capture stop / profile step done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
