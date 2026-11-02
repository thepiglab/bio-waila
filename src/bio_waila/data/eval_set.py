#!/usr/bin/env python3
"""mmbu-public → eval manifest in the training label format (same normalizers as training).

Fields a row leaves empty were not annotated, so Pass 2 would not ask for them: they become mode "masked" and are
skipped when scoring. Domain and body part are multi-label in MMBU (e.g. "Chest|Lung"), so they use mode "multi".
Use this to (a) set confidence thresholds for autolabel.py and (b) score the trained classifier. Per the challenge
notes, keep the public split for evaluation only — never train on it.

  uv run waila-eval-set            # → plan/eval_public.parquet
"""
import ast
import hashlib
import json
from pathlib import Path

import pandas as pd

from bio_waila.data.vocab import CANON, FIELDS

from bio_waila.paths import PLAN, PUBLIC, PUBLIC_REVISION
COLS = {"modality": "metadata_modality", "submodality": "metadata_submodality", "body_part": "metadata_body_part",
        "stain": "metadata_stain", "specimen": "metadata_specimen", "domain": "metadata_medical_domain"}


def split_vals(field, v):
    if field == "domain":
        items = ast.literal_eval(v) if isinstance(v, str) and v.startswith("[") else [v]
        return [x.strip() for it in items for x in str(it).split(",")]
    return [x for x in str(v).split("|")]


def local_revision(public) -> str | None:
    """The dataset commit a `hf download --local-dir` copy came from, read from the metadata huggingface_hub keeps beside
    it (None for a copy made any other way)."""
    try:
        return (Path(public) / ".cache/huggingface/download/metadata.jsonl.metadata").read_text().split()[0]
    except (OSError, IndexError):
        return None


def image_sha1(path) -> str | None:
    """First 16 hex digits of the SHA-1 of an image file's bytes (None if the file is missing)."""
    try:
        return hashlib.sha1(Path(path).read_bytes()).hexdigest()[:16]
    except OSError:
        return None


def recorded_revision(plan=None) -> str | None:
    """The commit <plan>/eval_public.parquet was built from (None for a manifest built before revisions were recorded)."""
    try:
        return json.loads((Path(plan or PLAN) / "eval_public.json").read_text()).get("revision")
    except (OSError, ValueError):
        return None


def build(public=None, plan=None, write=True, revision=PUBLIC_REVISION) -> pd.DataFrame:
    """metadata.jsonl under `public` → eval manifest (training label format); written to <plan>/eval_public.parquet, with
    the dataset commit in <plan>/eval_public.json. A copy from another commit than the pinned `revision` is refused."""
    public, plan = Path(public or PUBLIC), Path(plan or PLAN)
    rev = local_revision(public)
    if revision and rev and not rev.startswith(revision[:12]):
        raise SystemExit(f"{public} holds mmbu-public@{rev[:12]}, not the pinned {revision[:12]}: re-sync it with "
                         f"`hf download mmbu/mmbu-public --repo-type dataset --revision {revision} --local-dir {public}` "
                         "(into an empty folder), or set WAILA_PUBLIC_REVISION")
    df = pd.read_json(public / "metadata.jsonl", lines=True)
    out = []
    for _, r in df.iterrows():
        rec = {"image": str(r.file_name), "dataset_id": "mmbu-public", "split": "eval",  # image path relative to WAILA_PUBLIC
               "image_sha1": image_sha1(public / str(r.file_name)),  # releases change pixels under the same file name
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
        (plan / "eval_public.json").write_text(json.dumps({"revision": rev, "rows": len(ev)}) + "\n")
    return ev


def main():
    ev = build()
    print(f"{len(ev):,} eval rows →", PLAN / "eval_public.parquet")
    print(json.dumps({f: ev[f"{f}_mode"].value_counts().to_dict() for f in FIELDS}, indent=1))


if __name__ == "__main__":
    main()
