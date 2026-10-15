#!/usr/bin/env python3
"""mmbu-public → eval manifest in the training label format (same normalizers as training).

Fields a row leaves empty were not annotated, so Pass 2 would not ask for them: they become mode "masked" and are
skipped when scoring. Domain and body part are multi-label in MMBU (e.g. "Chest|Lung"), so they use mode "multi".
Use this to (a) set confidence thresholds for autolabel.py and (b) score the trained classifier. Per the challenge
notes, keep the public split for evaluation only — never train on it.

  uv run waila-eval-set            # → plan/eval_public.parquet
"""
import ast
import json
from pathlib import Path

import pandas as pd

from bio_waila.data.vocab import CANON, FIELDS

from bio_waila.paths import PLAN, PUBLIC
COLS = {"modality": "metadata_modality", "submodality": "metadata_submodality", "body_part": "metadata_body_part",
        "stain": "metadata_stain", "specimen": "metadata_specimen", "domain": "metadata_medical_domain"}


def split_vals(field, v):
    if field == "domain":
        items = ast.literal_eval(v) if isinstance(v, str) and v.startswith("[") else [v]
        return [x.strip() for it in items for x in str(it).split(",")]
    return [x for x in str(v).split("|")]


def build(public=None, plan=None, write=True) -> pd.DataFrame:
    """metadata.jsonl under `public` → eval manifest (training label format); written to <plan>/eval_public.parquet."""
    public, plan = Path(public or PUBLIC), Path(plan or PLAN)
    df = pd.read_json(public / "metadata.jsonl", lines=True)
    out = []
    for _, r in df.iterrows():
        rec = {"image": str(r.file_name), "dataset_id": "mmbu-public", "split": "eval",  # image path relative to WAILA_PUBLIC
               "task": str(r.task).split("|")[0].strip(), "unique_id": r.unique_id}
        for f in FIELDS:
            vals = sorted({c for c in (CANON[f](x) for x in split_vals(f, r[COLS[f]])) if c})
            mode = "masked" if not vals else ("multi" if f in ("domain", "body_part") or len(vals) > 1 else "hard")
            rec[f"{f}_mode"], rec[f"{f}_values"], rec[f"{f}_source"], rec[f"{f}_conf"] = mode, vals, "mmbu", 1.0
        out.append(rec)
    ev = pd.DataFrame(out)
    if write:
        plan.mkdir(parents=True, exist_ok=True)
        ev.to_parquet(plan / "eval_public.parquet", index=False)
    return ev


def main():
    ev = build()
    print(f"{len(ev):,} eval rows →", PLAN / "eval_public.parquet")
    print(json.dumps({f: ev[f"{f}_mode"].value_counts().to_dict() for f in FIELDS}, indent=1))


if __name__ == "__main__":
    main()
