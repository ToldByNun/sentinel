"""Shared hardware / metric helpers for paper benches."""

from __future__ import annotations

import os
import platform
import re
import subprocess
import sys


def progress(msg: str) -> None:
    print(f"[progress] {msg}", flush=True)


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return ""


def hardware() -> dict[str, str]:
    name = driver = ""
    vram_mib = None
    smi = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    if smi.strip():
        parts = [p.strip() for p in smi.strip().splitlines()[0].split(",")]
        name = parts[0] if parts else ""
        try:
            vram_mib = int(float(parts[1])) if len(parts) > 1 else None
        except ValueError:
            vram_mib = None
        driver = parts[2] if len(parts) > 2 else ""
    cuda = ""
    m = re.search(r"release\s+([\d.]+)", _run(["nvcc", "--version"]))
    if m:
        cuda = m.group(1)
    cpu = platform.processor() or ""
    if sys.platform == "win32":
        out = _run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "(Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name)",
            ]
        )
        if out.strip():
            cpu = out.strip()
    elif sys.platform.startswith("linux"):
        try:
            for line in open("/proc/cpuinfo", encoding="utf-8", errors="replace"):
                if line.lower().startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
        except OSError:
            pass
    ram_mib = None
    if sys.platform == "win32":
        out = _run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "[int]((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1MB)",
            ]
        )
        try:
            ram_mib = int(out.strip())
        except ValueError:
            ram_mib = None
    elif sys.platform.startswith("linux"):
        try:
            for line in open("/proc/meminfo", encoding="utf-8"):
                if line.startswith("MemTotal:"):
                    ram_mib = int(line.split()[1]) // 1024
                    break
        except (OSError, ValueError):
            pass
    return {
        "GPU": name,
        "VRAM": f"{vram_mib} MiB" if vram_mib is not None else "",
        "CPU": cpu,
        "RAM": f"{ram_mib} MiB" if ram_mib is not None else "",
        "CUDA Version": cuda or driver,
    }


def gpu_used_mib() -> float | None:
    out = _run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
    try:
        return float(out.strip().splitlines()[0].strip().replace(",", "."))
    except (ValueError, IndexError):
        return None


def host_rss_mib() -> float | None:
    if sys.platform == "win32":
        out = _run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                f"(Get-Process -Id {os.getpid()}).WorkingSet64 / 1MB",
            ]
        )
        try:
            return float(out.strip().replace(",", "."))
        except ValueError:
            return None
    try:
        with open(f"/proc/{os.getpid()}/status", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        return None
    return None


def is_oom(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return (
        "out of memory" in msg
        or "oom" in msg
        or "cudaerror memory" in msg.replace(" ", "")
        or "cuda out of memory" in msg
    )


def print_report(
    *,
    framework: str,
    label: str,
    profile: str,
    hw: dict[str, str],
    tok_s: float | None,
    peak_vram: float | None,
    host_ram: float | None,
    step_ms: float | None,
    status: str,
) -> None:
    print("---", flush=True)
    print(f"framework: {framework}", flush=True)
    print(f"model: {label}", flush=True)
    print(f"profile: {profile}", flush=True)
    print(f"tok/s: {tok_s:.0f}" if tok_s is not None else "tok/s: n/a", flush=True)
    print(f"peak VRAM: {peak_vram:.0f} MiB" if peak_vram is not None else "peak VRAM: n/a", flush=True)
    print(f"host RAM: {host_ram:.0f} MiB" if host_ram is not None else "host RAM: n/a", flush=True)
    print(f"step time: {step_ms:.2f} ms" if step_ms is not None else "step time: n/a", flush=True)
    print(f"status: {status}", flush=True)
    for k, v in hw.items():
        print(f"{k}: {v}", flush=True)
