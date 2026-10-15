"""Load the prepared training manifest with the dataset filters every trainer should use.

The one switch most runs care about is the licence layer: `nc` rows (non-commercial, tier B datasets) are excluded
unless asked for, so a run on the permissive set and a run with the NC layer differ only by `include_nc`.

    from bio_waila.data.manifest import load_manifest
    df = load_manifest(include_nc=False)                     # permissive only (default)
    df = load_manifest(include_nc=True, splits=("train",))   # add the NC layer, training split only

Configs (configs/data.yaml) carry the same keys, so experiments record which layer they used.
"""
import json
from pathlib import Path

import pandas as pd

from bio_waila.paths import PLAN, PREP


def load_manifest(path: Path | str | None = None, include_nc: bool = False, splits=("train", "val"),
                  exclude_datasets=(), only_datasets=None) -> pd.DataFrame:
    df = pd.read_parquet(path or PREP / "manifest.parquet")
    if "license_group" not in df:  # manifests built before the tag existed: recover it from the plan
        groups = {r["id"]: r.get("license_group", "permissive") for r in map(json.loads, open(PLAN / "selection.jsonl"))}
        df["license_group"] = df["dataset_id"].map(groups).fillna("permissive")
    keep = df["split"].isin(splits)
    if not include_nc:
        keep &= df["license_group"] != "nc"
    if exclude_datasets:
        keep &= ~df["dataset_id"].isin(set(exclude_datasets))
    if only_datasets is not None:
        keep &= df["dataset_id"].isin(set(only_datasets))
    return df[keep].reset_index(drop=True)


def license_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Images and datasets per licence group (for logging at the start of a run)."""
    return df.groupby("license_group").agg(images=("image", "size"), datasets=("dataset_id", "nunique"))
