"""Sentinel — C++/CUDA full-train causal LM (pip: sentinel-lm).

High-level: LanguageModel.train / generate / checkpoints.
Mid-level: Matrix, layers (Attention/FFN/…), losses, Adam/SGD/Spulse,
accumulate_example / apply_gradients, streaming ChunkSource, SafeTensors I/O.

API docs: https://github.com/ToldByNun/sentinel/blob/main/docs/python.md
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Sequence, Union


def _add_cuda_dll_directories() -> None:
    """Python 3.8+ on Windows ignores PATH for extension DLL deps; add CUDA bins."""
    if sys.platform != "win32":
        return
    roots: list[str] = []
    for key in ("CUDA_PATH", "CUDA_HOME"):
        value = os.environ.get(key)
        if value:
            roots.append(value)
    versioned = [
        v
        for k, v in os.environ.items()
        if k.startswith("CUDA_PATH_V") and v
    ]
    roots.extend(versioned)
    if not roots:
        default = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
        if os.path.isdir(default):
            try:
                versions = sorted(
                    (d for d in os.listdir(default) if d.startswith("v")),
                    reverse=True,
                )
            except OSError:
                versions = []
            for ver in versions:
                roots.append(os.path.join(default, ver))
                break
    seen: set[str] = set()
    for root in roots:
        for sub in ("bin\\x64", "bin"):
            path = os.path.join(root, sub)
            if path in seen or not os.path.isdir(path):
                continue
            seen.add(path)
            try:
                os.add_dll_directory(path)
            except (OSError, AttributeError):
                pass


_add_cuda_dll_directories()

from ._core import (  # noqa: E402
    CLASS_COUNT,
    CLASS_CPP,
    CLASS_JSON,
    CLASS_PYTHON,
    ActivationCheckpointMode,
    Adam,
    AdamState,
    BPETokenizer,
    CausalSelfAttention,
    CausalSelfAttentionCache,
    ClassificationDataset,
    ClassificationExample,
    CorpusRow,
    CrossEntropy,
    Dense,
    Dropout,
    Embedding,
    FeedForward,
    FeedForwardCache,
    HfTokenizer,
    JsonlLoader,
    LanguageModel,
    LanguageModelCache,
    LanguageModelChunkSource,
    LanguageModelDataset,
    LanguageModelExample,
    LanguageModelGradients,
    MSE,
    Matrix,
    MeanPool,
    MuonState,
    RMSNorm,
    RMSNormCache,
    ReLU,
    RotaryEmbedding,
    SGD,
    SafeTensorsFile,
    SbaoMode,
    SentinelModelConfig,
    Sequential,
    SiLU,
    Softmax,
    Spulse,
    SpulseCoverage,
    SpulseMomentumStorage,
    SpulseState,
    TransformerBlock,
    TransformerBlockCache,
    TransformerBlockGradients,
    UniformInit,
    cuda_available,
    is_safetensors_file,
    safetensors_load,
    safetensors_save,
    __version__ as _core_version,
)


def _matrix_to_numpy(self: Matrix):
    """Copy into a numpy float32 array shaped (rows, cols). Requires numpy."""
    import numpy as np

    return np.asarray(self.to_list(), dtype=np.float32).reshape(self.rows, self.cols)


def _matrix_from_numpy(cls, array) -> Matrix:
    """Build a Matrix from an array-like (rows, cols) float buffer. Requires numpy."""
    import numpy as np

    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("Matrix.from_numpy expects a 2D array")
    rows, cols = int(arr.shape[0]), int(arr.shape[1])
    return cls.from_list(rows, cols, arr.reshape(-1).tolist())


Matrix.to_numpy = _matrix_to_numpy  # type: ignore[attr-defined]
Matrix.from_numpy = classmethod(_matrix_from_numpy)  # type: ignore[attr-defined]


def _language_model_from_config(
    source: Union[str, dict[str, Any], SentinelModelConfig],
    *,
    load_weights: bool = True,
    base_directory: str = "",
) -> LanguageModel:
    """Build a LanguageModel from a path, dict, or SentinelModelConfig."""
    if isinstance(source, SentinelModelConfig):
        return LanguageModel.from_sentinel_config(source, base_directory, load_weights)
    if isinstance(source, dict):
        config = SentinelModelConfig.parse_json(json.dumps(source))
        return LanguageModel.from_sentinel_config(config, base_directory, load_weights)
    if isinstance(source, str):
        return LanguageModel.load_sentinel_model(source, load_weights)
    raise TypeError(
        "LanguageModel.from_config expects a path str, dict, or SentinelModelConfig"
    )


LanguageModel.from_config = staticmethod(_language_model_from_config)  # type: ignore[attr-defined]


def _language_model_train_step(
    self: LanguageModel,
    examples: Sequence[LanguageModelExample],
    *,
    grad_scale: float | None = None,
) -> float:
    """Host microstep: accumulate examples → mean grads → Adam."""
    if not examples:
        return 0.0
    grads = LanguageModelGradients.zeros_from(self)
    cache = LanguageModelCache()
    total = 0.0
    for example in examples:
        total += self.accumulate_example(example, grads, cache)
    scale = (1.0 / float(len(examples))) if grad_scale is None else float(grad_scale)
    grads.scale_in_place(scale)
    self.apply_gradients(grads)
    return total / float(len(examples))


LanguageModel.train_step = _language_model_train_step  # type: ignore[attr-defined]

__all__ = [
    "CLASS_COUNT",
    "CLASS_CPP",
    "CLASS_JSON",
    "CLASS_PYTHON",
    "ActivationCheckpointMode",
    "Adam",
    "AdamState",
    "BPETokenizer",
    "CausalSelfAttention",
    "CausalSelfAttentionCache",
    "ClassificationDataset",
    "ClassificationExample",
    "CorpusRow",
    "CrossEntropy",
    "Dense",
    "Dropout",
    "Embedding",
    "FeedForward",
    "FeedForwardCache",
    "HfTokenizer",
    "JsonlLoader",
    "LanguageModel",
    "LanguageModelCache",
    "LanguageModelChunkSource",
    "LanguageModelDataset",
    "LanguageModelExample",
    "LanguageModelGradients",
    "MSE",
    "Matrix",
    "MeanPool",
    "MuonState",
    "RMSNorm",
    "RMSNormCache",
    "ReLU",
    "RotaryEmbedding",
    "SGD",
    "SafeTensorsFile",
    "SbaoMode",
    "SentinelModelConfig",
    "Sequential",
    "SiLU",
    "Softmax",
    "Spulse",
    "SpulseCoverage",
    "SpulseMomentumStorage",
    "SpulseState",
    "TransformerBlock",
    "TransformerBlockCache",
    "TransformerBlockGradients",
    "UniformInit",
    "cuda_available",
    "is_safetensors_file",
    "safetensors_load",
    "safetensors_save",
    "__version__",
]

__version__ = _core_version
