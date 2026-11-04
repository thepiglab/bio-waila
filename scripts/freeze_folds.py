#!/usr/bin/env python3
"""Freeze the leave-dataset-group-out folds (training_plan.md §1.2) from the plan, before any trainer runs.

The folds are balanced on the planned images per dataset family (plan/selection.jsonl), so they don't depend on how much
has been downloaded. Trainers otherwise create plan/folds.json from whatever manifest they see first (e.g. the pilot).

    uv run python scripts/freeze_folds.py            # → plan/folds.json (refuses to overwrite; --force rebuilds)
"""
import json
import sys

from bio_waila.paths import PLAN
from bio_waila.train.folds import balance


def main(k=5, seed=0):
    out = PLAN / "folds.json"
    if out.exists() and "--force" not in sys.argv:
        raise SystemExit(f"{out} exists: the folds are frozen (pass --force to rebuild)")
    sizes = {}
    for line in open(PLAN / "selection.jsonl"):
        r = json.loads(line)
        sizes[str(r["group"])] = sizes.get(str(r["group"]), 0) + int(r["plan_images"])
    folds = balance(sizes, k, seed)
    out.write_text(json.dumps({"k": k, "seed": seed, "folds": folds}, indent=1, sort_keys=True))
    load = [sum(n for g, n in sizes.items() if folds[g] == f) for f in range(k)]
    print(f"{len(folds)} groups → {out}; planned images per fold: {load}")


if __name__ == "__main__":
    main()
