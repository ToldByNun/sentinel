"""Paper bench entry: fsdp / 500M — shared code in benchmarks/_lib."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_lib"))
from run_fsdp import run

if __name__ == "__main__":
    profile = "fair"
    if len(sys.argv) > 1 and sys.argv[1] in ("fair", "feat"):
        profile = sys.argv[1]
    raise SystemExit(run("500m", profile))
