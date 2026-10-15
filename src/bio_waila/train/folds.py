"""Leave-dataset-group-out folds. Every image inherits its dataset's labels, so splitting by image measures dataset
recognition; folds are built from the manifest's `group` column (dataset family) and frozen to plan/folds.json once.

    from bio_waila.train.folds import load_or_make_folds, split
    folds = load_or_make_folds(df)            # {group: fold}
    tr, ho = split(df, folds, fold=0)         # train on folds != 0, evaluate on fold 0
"""
import json
from pathlib import Path

import pandas as pd

from bio_waila.paths import PLAN


def make_folds(df: pd.DataFrame, k: int = 5, seed: int = 0) -> dict:
    """Greedy balance: biggest groups first, each into the fold with the fewest images (ties broken by seed)."""
    import random
    sizes = df.groupby("group").size().sort_values(ascending=False)
    rng = random.Random(seed)
    load = [0] * k
    out = {}
    for g, n in sizes.items():
        m = min(load)
        f = rng.choice([i for i, x in enumerate(load) if x == m])
        out[str(g)] = f
        load[f] += int(n)
    return out


def load_or_make_folds(df: pd.DataFrame, k: int = 5, path: Path | None = None, seed: int = 0) -> dict:
    """Read plan/folds.json; groups not in it (new datasets) are added greedily and the file is rewritten."""
    path = Path(path or PLAN / "folds.json")
    folds = json.loads(path.read_text())["folds"] if path.exists() else {}
    missing = df[~df["group"].astype(str).isin(folds)]
    if len(missing):
        load = [0] * k
        for g, f in folds.items():
            load[f] += int((df["group"].astype(str) == g).sum())
        for g, n in missing.groupby("group").size().sort_values(ascending=False).items():
            f = load.index(min(load))
            folds[str(g)] = f
            load[f] += int(n)
        path.write_text(json.dumps({"k": k, "seed": seed, "folds": folds}, indent=1, sort_keys=True))
    return folds


def split(df: pd.DataFrame, folds: dict, fold: int):
    f = df["group"].astype(str).map(folds)
    return df[f != fold].reset_index(drop=True), df[f == fold].reset_index(drop=True)
