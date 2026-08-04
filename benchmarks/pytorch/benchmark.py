"""Run all pytorch paper sizes: fair table, then feat table."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_lib"))

from paper_config import MODELS
from run_pytorch import run

ORDER = ("100m", "500m", "1b", "1_5b", "4b_poc")


def main() -> int:
    failures = 0
    for profile in ("fair", "feat"):
        print(f"\n======== PYTORCH PROFILE={profile} ========", flush=True)
        for model_id in ORDER:
            label = MODELS[model_id].label
            print(f"\n----- {label} / {profile} -----", flush=True)
            code = run(model_id, profile)
            if code != 0:
                failures += 1
                print(f"[runner] {label} {profile} -> exit {code} (continue)", flush=True)
    print(f"\n======== PYTORCH DONE failures={failures} ========", flush=True)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
