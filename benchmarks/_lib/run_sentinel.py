"""Sentinel paper runner.

fair = normal GPU-resident (weights+grads+opt+acts on device)
feat = memory-efficient offload (FP16 GPU weights; grads/opt/masters on host)
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


def run(model_id: str, profile: ProfileName = "fair") -> int:
    spec, train = resolve(model_id, profile)
    progress("collecting hardware")
    hw = hardware()
    mode = "feat/offload" if train.memory_efficient else "fair/resident"
    print(f"=== Sentinel {spec.label}  profile={train.name} ({mode}) ===", flush=True)
    for k, v in hw.items():
        print(f"{k}: {v}", flush=True)
    if train.memory_efficient:
        print(
            "layout: GPU=FP16 weights + ckpt activations | CPU=grads + optimizer/masters (host SGD)",
            flush=True,
        )
    else:
        print(
            "layout: GPU=weights + grads + activations + optimizer (normal full-train)",
            flush=True,
        )
    print(
        f"config: lr={LEARNING_RATE} amp={train.fp16_amp} flash={train.flash_attention} "
        f"ckpt={train.sentinel_ckpt} seq={SEQ} pack={PACK_EXAMPLES} "
        f"tokens/step={SEQ * PACK_EXAMPLES} warmup={WARMUP_STEPS} timed={TIMED_STEPS}",
        flush=True,
    )

    tok_s = peak_vram = host_ram = step_ms = None
    status = "fail"
    try:
        progress("checking CUDA")
        if not S.cuda_available():
            raise RuntimeError("CUDA unavailable")

        ckpt = getattr(S.ActivationCheckpointMode, train.sentinel_ckpt)
        progress(
            f"constructing vocab={VOCAB} d={spec.embedding_dim} L={spec.block_count} "
            f"H={spec.head_count} pos={spec.maximum_position_count}"
        )
        model = S.LanguageModel(
            VOCAB,
            spec.embedding_dim,
            spec.maximum_position_count,
            LEARNING_RATE,
            spec.block_count,
            spec.head_count,
        )
        progress(f"model ready params~{model.parameter_count / 1e6:.1f}M")
        progress("enable_cuda")
        model.enable_cuda()
        model.set_prefer_flash_attention(train.flash_attention)
        model.set_prefer_muon(False)
        model.set_max_packed_columns(SEQ * PACK_EXAMPLES)

        if train.memory_efficient:
            progress("feat: enable FP16 GPU weights + host grads + host SGD")
            model.set_prefer_cpu_adam_offload(True)
            model.enable_cuda_train()
            model.set_prefer_host_sgd(True)
            model.set_activation_checkpoint_mode(S.ActivationCheckpointMode.Full)
        else:
            progress("fair: enable resident Adam (no int8 / no host offload)")
            model.set_prefer_cpu_adam_offload(False)
            model.set_prefer_int8_adam_moments(False)
            model.enable_cuda_train()
            model.set_activation_checkpoint_mode(ckpt)

        model.set_prefer_train_graph(False)
        progress(f"max_packed_columns={model.max_packed_columns}")

        peak_vram = gpu_used_mib() or 0.0
        host_ram = host_rss_mib() or 0.0
        progress(f"before probe: VRAM~{peak_vram:.0f} MiB  host~{host_ram:.0f} MiB")
        progress(f"probe seq={SEQ} warmup={WARMUP_STEPS} timed={TIMED_STEPS}")

        t0 = time.perf_counter()
        tok_s = float(
            model.probe_cuda_packed_train_tokens_per_second(SEQ, WARMUP_STEPS, TIMED_STEPS)
        )
        wall = time.perf_counter() - t0
        # Prefer probe tok/s; if it returns 0, derive from wall + known token budget.
        tokens = float(SEQ * PACK_EXAMPLES * TIMED_STEPS)
        if tok_s <= 0.0 and wall > 0.0:
            tok_s = tokens / wall
            progress(f"probe returned 0; using wall tok/s={tok_s:.0f}")
        if tok_s > 0.0:
            step_ms = (SEQ * PACK_EXAMPLES / tok_s) * 1000.0

        peak_vram = max(peak_vram, gpu_used_mib() or 0.0)
        host_ram = max(host_ram, host_rss_mib() or 0.0)
        status = "success"
        progress(f"finished OK tok/s={tok_s:.0f} step_ms={step_ms:.2f}")
    except Exception as ex:
        status = "oom" if is_oom(ex) else "fail"
        peak_vram = max(peak_vram or 0.0, gpu_used_mib() or 0.0) if peak_vram is not None else gpu_used_mib()
        host_ram = max(host_ram or 0.0, host_rss_mib() or 0.0) if host_ram is not None else host_rss_mib()
        progress(f"{status}: {ex}")
        print(f"error: {ex}", flush=True)

    print_report(
        framework="sentinel",
        label=spec.label,
        profile=train.name,
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
