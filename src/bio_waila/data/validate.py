"""Validate the data pipeline one dataset at a time on a laptop, inside the size-capped cache (data/cache.py).

For each dataset (from plan/selection.jsonl):
  1. wait until the cache is under its cap (default 30 GB), then download a pilot sample (--pilot images, ≤ --max-gb)
  2. metalabel profile → map → apply  (per-sample labels from metadata; per-worker work dir so workers don't collide)
  3. prepare                          (decode, sample, ≤ 512 px JPEG, attach labels, drop near-duplicates of mmbu-public)
  4. write reports/<id>.json: download outcome, metadata columns and their top values, which columns were trusted for
     which field, values needing review, per-field label modes and sources, conflicts, and a few sample images with
     their labels for a visual check
  5. delete raw/<id> (unless --keep-raw); the small prepared sample and the report stay

  uv run waila-validate --only cystods,papila --worker w1
  uv run waila-validate --shard 0/5 --worker w1 --pilot 100 --max-gb 3
  uv run waila-validate --summary            # one table over every report in the cache
"""
import argparse
import json
import random
import time
import traceback
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from bio_waila.data import download as dl
from bio_waila.data import metalabel as ml
from bio_waila.data import prepare as pp
from bio_waila.data.cache import CacheGuard
from bio_waila.data.vocab import FIELDS
from bio_waila.paths import CACHE, PLAN, PUBLIC


def plan_rows():
    return {r["id"]: r for r in (json.loads(l) for l in open(PLAN / "selection.jsonl"))}


def summarize_labels(df):
    out = {}
    for f in FIELDS:
        if f + "_mode" not in df:
            continue
        modes = df[f + "_mode"].value_counts(normalize=True).round(3).to_dict()
        src = df.loc[df[f + "_mode"].isin(["hard", "multi"]), f + "_source"].value_counts().to_dict()
        vals = Counter(v for vs in df[f + "_values"] for v in vs).most_common(6)
        out[f] = {"modes": modes, "sources": src, "top_values": vals,
                  "conflicts": int((df.get(f + "_conflict", pd.Series(dtype=str)).fillna("") != "").sum())}
    return out


def validate_one(row, guard: CacheGuard, args):
    rid = row["id"]
    work = guard.root / "work" / args.worker
    raw = guard.root / "raw"
    rep = {"id": rid, "method": row["method"], "ref": row["ref"], "registry_labels": row["labels"],
           "license_group": row.get("license_group", "permissive"), "license": row.get("license", ""),
           "worker": args.worker, "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if not guard.wait_for_room(timeout_s=args.wait_s):
        rep["status"] = "deferred: cache over cap"
        return rep
    guard.update(rid, status="downloading", worker=args.worker)
    row = {**row, "plan_images": min(row["plan_images"], args.pilot)}
    dargs = SimpleNamespace(retry=args.retry, dry_run=False, max_gb=args.max_gb, seed=0)
    st = dl.run_one(row, raw, dargs)
    rep["download"] = {k: st.get(k) for k in ("state", "method", "ref", "want", "n_media", "bytes", "strategy", "notes", "listed_files", "listed_bytes")}
    if st.get("state") not in ("done", "partial"):
        rep["status"] = f"download {st.get('state')}"
        if not args.keep_raw:
            guard.remove_raw(rid)
        guard.update(rid, status=rep["status"], raw_bytes=0)
        return rep
    # metalabel on this one dataset, into the worker's own folder
    margs = SimpleNamespace(raw=str(raw), out=str(work), only=rid, shard="0/1")
    ml.cmd_profile(margs)
    ml.cmd_map(margs)
    ml.cmd_apply(margs)
    mdir = work / "metalabel" / rid
    if (mdir / "profile.json").exists():
        prof = json.loads((mdir / "profile.json").read_text())
        rep["metadata"] = {"n_samples": prof["n_samples"], "tables": prof["tables"],
                           "columns": {c: {"non_null": v["non_null"], "n_distinct": v["n_distinct"], "skipped": v["skipped"],
                                           "top": list(v["top"].items())[:8]} for c, v in prof["columns"].items()}}
    if (mdir / "colmaps.json").exists():
        cm = json.loads((mdir / "colmaps.json").read_text())
        rep["trusted_columns"] = {col: {f: dict(list(vals.items())[:10]) for f, vals in fm.items()} for col, fm in cm.items()}
    rv = work / "metalabel" / "review.csv"
    if rv.exists() and rv.stat().st_size > 0:
        r = pd.read_csv(rv)
        rep["review_values"] = r[r.dataset_id == rid].head(30).to_dict("records") if len(r) else []
    # prepare (images + labels) into the worker folder
    space = json.loads((PLAN / "label_space.json").read_text())
    pub = pp.public_hashes(PUBLIC, guard.root / "_public_dhash.npy")
    pst = pp.prepare_dataset(row, raw, work, set(space["body_part"]["classes"]), pub, 0)
    rep["prepare"] = pst
    man = work / "manifest" / f"{rid}.parquet"
    if man.exists():
        df = pd.read_parquet(man)
        rep["n_images"] = len(df)
        rep["labels"] = summarize_labels(df)
        idx = random.Random(0).sample(range(len(df)), min(args.samples, len(df)))
        rep["samples"] = [{"image": str(work / df.at[i, "image"]), "source": df.at[i, "source"],
                           **{f: list(df.at[i, f + "_values"]) for f in FIELDS},
                           "evidence": {f: df.at[i, f + "_evidence"] for f in FIELDS if df.at[i, f + "_evidence"]}} for i in idx]
    rep["status"] = "validated" if rep.get("n_images") else "no images prepared"
    freed = 0 if args.keep_raw else guard.remove_raw(rid)
    rep["raw_bytes_freed"] = freed
    guard.update(rid, status=rep["status"], n_images=rep.get("n_images", 0), raw_bytes=0 if not args.keep_raw else freed,
                 worker=args.worker)
    return rep


def summary(guard: CacheGuard):
    rows = []
    for p in sorted((guard.root / "reports").glob("*.json")):
        r = json.loads(p.read_text())
        lab = r.get("labels", {})
        hard = {f: round(sum(v for k, v in lab.get(f, {}).get("modes", {}).items() if k in ("hard", "multi")), 2) for f in FIELDS}
        auto = {f: sum(v for k, v in lab.get(f, {}).get("sources", {}).items() if k != "registry") for f in FIELDS}
        rows.append({"id": r["id"], "status": r.get("status"), "license_group": r.get("license_group", "permissive"),
                     "method": r.get("method"), "n_images": r.get("n_images", 0),
                     **{f"hard_{f}": hard[f] for f in FIELDS}, **{f"auto_{f}": auto[f] for f in FIELDS},
                     "conflicts": sum(lab.get(f, {}).get("conflicts", 0) for f in FIELDS),
                     "review": len(r.get("review_values", []) or []), "worker": r.get("worker")})
    df = pd.DataFrame(rows)
    df.to_csv(guard.root / "reports" / "summary.csv", index=False)
    print(df["status"].value_counts().to_string() if len(df) else "no reports")
    print(f"{len(df)} reports → {guard.root / 'reports' / 'summary.csv'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default="")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--worker", default="w0", help="name of this worker; its metalabel/prepare outputs go to work/<worker>")
    ap.add_argument("--pilot", type=int, default=100, help="images per dataset")
    ap.add_argument("--max-gb", type=float, default=3.0, help="per-dataset download budget")
    ap.add_argument("--samples", type=int, default=8, help="sample images listed in each report for visual checks")
    ap.add_argument("--keep-raw", action="store_true")
    ap.add_argument("--retry", action="store_true")
    ap.add_argument("--redo", action="store_true", help="re-validate datasets that already have a report")
    ap.add_argument("--wait-s", type=int, default=3600)
    ap.add_argument("--root", default=str(CACHE))
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    guard = CacheGuard(Path(args.root))
    if args.summary:
        return summary(guard)
    rows = list(plan_rows().values())
    if args.only:
        keep = set(args.only.split(","))
        rows = [r for r in rows if r["id"] in keep]
    i, n = map(int, args.shard.split("/"))
    rows = rows[i::n]
    for k, row in enumerate(rows):
        out = guard.root / "reports" / f"{row['id']}.json"
        if out.exists() and not args.redo:
            continue
        try:
            rep = validate_one(row, guard, args)
        except Exception as ex:
            rep = {"id": row["id"], "status": f"error: {type(ex).__name__}: {ex}", "trace": traceback.format_exc(limit=4)}
            guard.remove_raw(row["id"]) if not args.keep_raw else None
            guard.update(row["id"], status="error", worker=args.worker)
        out.write_text(json.dumps(rep, indent=1, default=str))
        print(f"[{k + 1}/{len(rows)}] {row['id']:44s} {rep.get('status')}  images={rep.get('n_images', 0)}  "
              f"cache={guard.size() / 2**30:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
