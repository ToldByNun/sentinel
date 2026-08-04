"""Shared paper benchmark shapes + train profiles (1:1 across libs/sizes).

Profiles (same meaning for every size):

  fair — *normal* full-train residency
         GPU: weights + gradients + activations + optimizer
         (may OOM at 1.5B/4B — that is a valid paper cell)

  feat — *memory-efficient* offload path (the former 4B PoC layout)
         GPU: FP16 weights + activations (checkpointed)
         CPU: gradients + optimizer / masters
         Sentinel: host-SGD offload | DeepSpeed: ZeRO-Offload | FSDP: CPU offload
         PyTorch single-GPU: no equivalent → status na

OOM / na are valid paper cells.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ProfileName = Literal["fair", "feat"]

# Fixed workload for both profiles (tokens/step = SEQ * PACK_EXAMPLES).
SEQ = 256
PACK_EXAMPLES = 16  # 4096 tokens / step
WARMUP_STEPS = 2
TIMED_STEPS = 4
LEARNING_RATE = 3e-4
VOCAB = 32000


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    label: str
    embedding_dim: int
    block_count: int
    head_count: int
    maximum_position_count: int


MODELS: dict[str, ModelSpec] = {
    "100m": ModelSpec("100m", "~100M", 768, 12, 12, 512),
    "500m": ModelSpec("500m", "~500M", 1280, 24, 20, 1024),
    "1b": ModelSpec("1b", "~1B", 2048, 24, 32, 2048),
    "1_5b": ModelSpec("1_5b", "~1.5B", 2560, 28, 40, 2048),
    "4b_poc": ModelSpec("4b_poc", "~4B PoC", 3072, 34, 48, 2048),
}


@dataclass(frozen=True)
class TrainProfile:
    name: ProfileName
    # True = memory-efficient offload; False = normal GPU-resident
    memory_efficient: bool
    optimizer: str
    fp16_amp: bool
    gradient_checkpointing: bool
    flash_attention: bool
    # Sentinel ckpt: Selective for fair (keep some acts); Full for feat offload
    sentinel_ckpt: str


PROFILES: dict[ProfileName, TrainProfile] = {
    "fair": TrainProfile(
        name="fair",
        memory_efficient=False,
        optimizer="adam",
        fp16_amp=True,
        gradient_checkpointing=True,
        flash_attention=True,
        sentinel_ckpt="Selective",
    ),
    "feat": TrainProfile(
        name="feat",
        memory_efficient=True,
        optimizer="adam",  # DeepSpeed/FSDP; Sentinel feat uses host-SGD masters
        fp16_amp=True,
        gradient_checkpointing=True,
        flash_attention=True,
        sentinel_ckpt="Full",
    ),
}


def resolve(model_id: str, profile: ProfileName = "fair") -> tuple[ModelSpec, TrainProfile]:
    if model_id not in MODELS:
        raise KeyError(f"unknown model_id={model_id!r}; choose from {list(MODELS)}")
    if profile not in PROFILES:
        raise KeyError(f"unknown profile={profile!r}; choose from {list(PROFILES)}")
    return MODELS[model_id], PROFILES[profile]
