"""Sentinel — C++/CUDA full-train causal LM (pip: sentinel-lm).

Public surface: BPETokenizer, HfTokenizer, LanguageModelDataset, LanguageModel,
ActivationCheckpointMode, SbaoMode, SpulseCoverage, cuda_available.

API docs: https://github.com/ToldByNun/sentinel/blob/main/docs/python.md
"""

from __future__ import annotations

import os
import sys


def _add_cuda_dll_directories() -> None:
    """Python 3.8+ on Windows ignores PATH for extension DLL deps; add CUDA bins."""
    if sys.platform != "win32":
        return
    roots: list[str] = []
    for key in ("CUDA_PATH", "CUDA_HOME"):
        value = os.environ.get(key)
        if value:
            roots.append(value)
    # CUDA 13+ often keeps cublasLt under bin\\x64; older layouts use bin.
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
    ActivationCheckpointMode,
    SbaoMode,
    SpulseCoverage,
    BPETokenizer,
    HfTokenizer,
    LanguageModel,
    LanguageModelDataset,
    cuda_available,
    __version__ as _core_version,
)

__all__ = [
    "ActivationCheckpointMode",
    "SbaoMode",
    "SpulseCoverage",
    "BPETokenizer",
    "HfTokenizer",
    "LanguageModel",
    "LanguageModelDataset",
    "cuda_available",
    "__version__",
]

__version__ = _core_version
