#!/usr/bin/env python3
"""Summarise a (pilot) prepare run: which selected datasets produced no images and why, which datasets have label
conflicts (registry value vs what the files say, per field), and the most common error lines in the job logs.

  uv run --no-sync python scripts/pilot_report.py                       # data/prepared_pilot + logs/pilot_*
  uv run --no-sync python scripts/pilot_report.py --prep data/prepared --logs 'logs/prepare_*'
Writes <prep>/pilot_report.md and prints it.
"""
import argparse
import glob
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from bio_waila.data.vocab import FIELDS
from bio_waila.paths import DATA, PLAN

ERR = re.compile(r"error|traceback|exception|manual|failed|timed? ?out|http \d{3}|not supported|unsupported|no images", re.I)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prep", default=str(DATA / "prepared_pilot"))
    ap.add_argument("--logs", default="logs/pilot_*")
    ap.add_argument("--min-images", type=int, default=10, help="also list datasets with fewer images than this")
    a = ap.parse_args()
    prep = Path(a.prep)
    df = pd.read_parquet(prep / "manifest.parquet")
    sel = [json.loads(l) for l in open(PLAN / "selection.jsonl")]
    ids = [s["id"] for s in sel]
    counts = df["dataset_id"].value_counts()
    logs = {p: Path(p).read_text(errors="replace").splitlines() for p in sorted(glob.glob(a.logs))}
    lines = [l for ls in logs.values() for l in ls]

    def reason(ds):
        hits = [l.strip() for l in lines if ds in l and ERR.search(l)]
        return hits[-1][:220] if hits else "(no error line mentions it)"

    L = [f"# Pilot report: {prep}", "", f"{len(df):,} images from {counts.size} of {len(ids)} selected datasets.", ""]
    missing = [i for i in ids if i not in counts]
    L += [f"## Selected datasets with no images ({len(missing)})", ""]
    L += [f"- `{i}`: {reason(i)}" for i in missing] or ["- none"]
    low = [(i, int(counts[i])) for i in ids if i in counts and counts[i] < a.min_images]
    L += ["", f"## Datasets with fewer than {a.min_images} images ({len(low)})", ""]
    L += [f"- `{i}`: {n} images" for i, n in low] or ["- none"]

    L += ["", "## Label conflicts by dataset and field", "",
          "A conflict means the file names / metadata strongly suggest a different value than the registry label.", "",
          "| dataset | field | conflicts / images | registry label | suggested (source:value) | example image |", "|---|---|---:|---|---|---|"]
    rows = []
    for f in FIELDS:
        c = df[df[f + "_conflict"].fillna("") != ""]
        for ds, g in c.groupby("dataset_id"):
            sug = Counter(g[f + "_conflict"]).most_common(2)
            vals = g[f + "_values"].iloc[0]
            rows.append((len(g), ds, f, int(counts[ds]), ", ".join(map(str, list(vals)[:3])),
                         "; ".join(f"{s} ×{n}" for s, n in sug), str(g["image"].iloc[0])[-70:]))
    for n, ds, f, tot, vals, sug, ex in sorted(rows, reverse=True)[:60]:
        L.append(f"| {ds} | {f} | {n} / {tot} | {vals} | {sug} | `{ex}` |")

    errs = Counter(re.sub(r"\d+", "N", l.strip())[:160] for l in lines if ERR.search(l))
    L += ["", f"## Most common error lines across {len(logs)} log files", ""]
    L += [f"- ×{n}: `{t}`" for t, n in errs.most_common(25)] or ["- none"]
    text = "\n".join(L) + "\n"
    (prep / "pilot_report.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
