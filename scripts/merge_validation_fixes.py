#!/usr/bin/env python3
"""Merge the ```json (registry_fixes format) and ```csv (overrides rows) blocks of validation/discovery reports into
plan/registry_fixes.json and plan/overrides.csv. Idempotent; prints label values that are not in the label space.

  uv run python scripts/merge_validation_fixes.py docs/data/validation/V4.md [more.md ...]
"""
import csv
import json
import re
import sys
from pathlib import Path

PLAN = Path(__file__).resolve().parents[1] / "plan"


def main(paths):
    F = json.loads((PLAN / "registry_fixes.json").read_text())
    F.setdefault("force_include", {})
    existing = set(tuple(r) for r in csv.reader(open(PLAN / "overrides.csv")))
    new_rows, n = [], 0
    for p in paths:
        md = Path(p).read_text()
        for b in re.findall(r"```json\s*(.*?)```", md, re.S):
            try:
                j = json.loads(b)
            except Exception as ex:
                print(f"{p}: unparsable json block ({ex})")
                continue
            if not isinstance(j, dict):
                continue
            for k in ("drop", "route", "labels", "exclude_paths", "force_include"):
                for i, v in (j.get(k) or {}).items():
                    if k == "labels":
                        F["labels"].setdefault(i, {}).update(v)
                    else:
                        F[k][i] = v
                    n += 1
        for b in re.findall(r"```csv\s*(.*?)```", md, re.S):
            for r in csv.reader(b.strip().splitlines()):
                if len(r) == 5 and r[0] != "dataset_id" and tuple(r) not in existing:
                    new_rows.append(r)
                    existing.add(tuple(r))
    (PLAN / "registry_fixes.json").write_text(json.dumps(F, indent=1, ensure_ascii=False))
    with open(PLAN / "overrides.csv", "a", newline="") as f:
        csv.writer(f).writerows(new_rows)
    sp = json.loads((PLAN / "label_space.json").read_text())
    bad = [(i, f, v) for i, fl in F["labels"].items() for f, t in fl.items() if isinstance(t, dict)
           for v in t.get("values", []) if f in sp and v not in sp[f]["classes"]]
    print(f"merged {n} fixes and {len(new_rows)} override rows; totals: " +
          ", ".join(f"{k} {len(F[k])}" for k in ("drop", "route", "labels", "exclude_paths", "force_include")))
    if bad:
        print("label values not (yet) in label_space.json:", bad)


if __name__ == "__main__":
    main(sys.argv[1:])
