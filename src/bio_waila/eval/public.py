"""Scoring on the public MMBU split (mmbu-public: EVALUATION ONLY, never trained on), shared by every trainer and `waila-eval`.

`load_public()` gives the eval manifest (plan/eval_public.parquet, built from mmbu-public/metadata.jsonl if absent).
`score_public()` returns the shared per-field metrics grouped by MMBU task (all rows share one dataset id, so `task`
is the macro-averaging key; pooled numbers are reported too) plus the context score C, a stand-in for Pass 2:

  C(image) = mean over the image's annotated (non-masked) fields of [top-1 class ∈ the field's target set]
  C        = mean of C(image) over images with at least one annotated field
             (single- and multi-label fields are scored alike: top-1 in the target set; candidate sets count)

C is also split by the image's MMBU modality (first `modality` target) and by task. Every report is logged in
plan/public_looks.jsonl: the project records each look at the public split.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from bio_waila.eval.metrics import all_metrics
from bio_waila.paths import PLAN, PUBLIC
from bio_waila.train.labels import MASKED


class PublicMissing(RuntimeError):
    pass


def load_public(build_if_missing=True, parquet=None):
    """(eval manifest, image root). Raises PublicMissing (with the expected location) if mmbu-public is not there."""
    from bio_waila.data import eval_set
    pq = Path(parquet) if parquet else PLAN / "eval_public.parquet"
    if not pq.exists():
        if parquet is None and build_if_missing and (PUBLIC / "metadata.jsonl").exists():
            df = eval_set.build(PUBLIC, PLAN)
        else:
            raise PublicMissing(f"public eval set not found: {pq} is missing and {PUBLIC / 'metadata.jsonl'} does not exist "
                                "(mmbu-public is expected next to the repo; set WAILA_PUBLIC to its location, then run "
                                "`uv run waila-eval-set`)")
    else:
        df = pd.read_parquet(pq)
    if not PUBLIC.exists():
        raise PublicMissing(f"mmbu-public images not found at {PUBLIC} (set WAILA_PUBLIC to override)")
    return df, PUBLIC


def public_enabled(cfg) -> bool:
    return bool((cfg.get("eval") or {}).get("public", True))


def try_load_public(cfg, parquet=None, limit=None):
    """(df, root) for a trainer's final evaluation, or None (with a one-line warning) if disabled or unavailable."""
    if not public_enabled(cfg):
        return None
    try:
        df, root = load_public(parquet=parquet)
    except PublicMissing as e:
        print(f"[public] skipped: {e}", flush=True)
        return None
    return (subsample(df, limit) if limit else df), root


def subsample(df, n, seed=0):
    return df.sample(n=min(n, len(df)), random_state=seed).sort_index().reset_index(drop=True) if n and n < len(df) else df


def _modality(df):
    return df["modality_values"].map(lambda v: str(v[0]) if v is not None and len(v) else "unannotated").to_numpy()


def _hits(logits, modes, tgts, space):
    top, hit, ann = {}, {}, {}
    for f in space.fields:
        t = logits[f].float().argmax(-1).cpu().numpy()
        y = tgts[f].cpu().numpy() > 0
        top[f], hit[f] = t, y[np.arange(len(t)), t]
        ann[f] = modes[f].cpu().numpy() != MASKED
    return top, hit, ann


def score_public(logits, modes, tgts, df, space, temps=None):
    """Metrics by task and pooled, context score C and its splits. `df` rows align with the logits (needs task, modality_values)."""
    top, hit, ann = _hits(logits, modes, tgts, space)
    n_ann = sum(ann[f].astype(int) for f in space.fields)
    n_ok = sum((hit[f] & ann[f]).astype(int) for f in space.fields)
    scored = n_ann > 0
    c_img = np.where(scored, n_ok / np.maximum(n_ann, 1), np.nan)

    def group(keys):
        keys = np.asarray(keys)
        return {str(k): {"C": float(c_img[(keys == k) & scored].mean()), "n": int((keys == k).sum())}
                for k in sorted(set(keys)) if ((keys == k) & scored).any()}
    fa = {f: {"acc": float(hit[f][ann[f]].mean()), "n": int(ann[f].sum())} for f in space.fields if ann[f].any()}
    task = df["task"].to_numpy()
    return {"C": float(c_img[scored].mean()) if scored.any() else float("nan"), "n_images": int(len(df)),
            "n_scored": int(scored.sum()), "C_by_modality": group(_modality(df)), "C_by_task": group(task),
            "field_acc": fa, "mean_field_acc": float(np.mean([v["acc"] for v in fa.values()])) if fa else float("nan"),
            "metrics_by_task": all_metrics(logits, modes, tgts, task, space, temps),
            "metrics_pooled": all_metrics(logits, modes, tgts, np.zeros(len(df), int), space, temps)}


def predictions_table(logits, modes, tgts, df, space):
    """One row per image: unique_id, image, task, and per field the top-1 label, the target labels and a correct flag (NA if masked)."""
    top, hit, ann = _hits(logits, modes, tgts, space)
    out = df[["unique_id", "image", "task"]].reset_index(drop=True).copy()
    for f in space.fields:
        names = space.classes[f]
        out[f"{f}_pred"] = [names[i] for i in top[f]]
        y = tgts[f].cpu().numpy() > 0
        out[f"{f}_target"] = [[names[j] for j in np.nonzero(r)[0]] for r in y]
        out[f"{f}_correct"] = pd.array([bool(h) if a else None for h, a in zip(hit[f], ann[f])], dtype="boolean")
    return out


def write_public_report(out_dir, run_name, result, predictions_df):
    """out_dir/public_metrics.json + public_predictions.parquet, and one line appended to plan/public_looks.jsonl."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "public_metrics.json").write_text(json.dumps(result, indent=1, default=float))
    predictions_df.to_parquet(out_dir / "public_predictions.parquet", index=False)
    PLAN.mkdir(parents=True, exist_ok=True)
    with open(PLAN / "public_looks.jsonl", "a") as fh:
        fh.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "run": run_name, "out": str(out_dir), "C": result["C"],
                             "mean_field_acc": result["mean_field_acc"], "n_images": result["n_images"]}, default=float) + "\n")
    return out_dir / "public_metrics.json"


def evaluate_public(logits, modes, tgts, df, space, temps, out_dir, run_name):
    """Score, write the report, print one line; returns the result dict (what trainers put in report['public'])."""
    res = score_public(logits, modes, tgts, df, space, temps)
    write_public_report(out_dir, run_name, res, predictions_table(logits, modes, tgts, df, space))
    print(f"[public] {run_name}: C={res['C']:.4f} mean_field_acc={res['mean_field_acc']:.4f} n={res['n_images']}", flush=True)
    return res
