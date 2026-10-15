"""Which images go into which batch, reproducibly.

Sampling distribution (`sampler.mode`):
  hierarchical  (default) pick a submodality at its planned share^tau, then a dataset within it ∝ n^power, then an
                image uniformly. Keeps the benchmark mix that waila-select planned, and spreads every label over as many
                datasets as possible (fights "label = dataset fingerprint" shortcuts; we evaluate on held-out datasets).
  dataset_sqrt  each dataset gets mass ∝ min(n, cap)^power (the old default; double-balances on top of the plan)
  uniform       every image equally (the manifest's own mix)

Reproducibility: the *global* sequence of (row, augmentation seed) is drawn once per epoch from the run seed and does
not depend on the number of GPUs. Each global batch is a fixed block of that sequence; rank r takes its slice of every
block. So a 1-GPU and a 4-GPU run with the same global batch see the same examples in the same global batches, with the
same augmentations (per-sample seeds), and train on the same data stream. (Gradients still differ slightly: each rank
averages over its own contributing rows before DDP averages the ranks.)
"""
import numpy as np
import torch


def stratum(row) -> str:
    """The label a row is stratified on: its submodality when exactly known, else '(other)'."""
    m, v = row.get("submodality_mode"), row.get("submodality_values")
    v = [] if v is None else list(v)
    return str(v[0]) if m in ("hard", "multi") and len(v) == 1 else "(other)"


def sample_weights(df, mode="hierarchical", tau=0.7, power=0.5, cap=20000, class_mass=None) -> np.ndarray:
    """Per-row sampling probabilities (sum 1). class_mass: optional {stratum: target share} (default: the manifest's
    own per-stratum counts, i.e. what waila-select planned)."""
    n_ds = df.groupby("dataset_id")["image"].transform("size").to_numpy().astype(float)
    if mode == "uniform":
        w = np.ones(len(df))
    elif mode == "dataset_sqrt":
        w = np.minimum(n_ds, cap) ** power / n_ds
    elif mode == "hierarchical":
        st = np.array([stratum(r) for r in df[["submodality_mode", "submodality_values"]].to_dict("records")])
        d = df.assign(_st=st)
        target = class_mass or d.groupby("_st").size().to_dict()
        cls = {s: float(target.get(s, 0.0)) ** tau for s in np.unique(st)}            # tempered class shares
        total = sum(cls.values()) or 1.0
        n_dc = d.groupby(["_st", "dataset_id"])["image"].transform("size").to_numpy().astype(float)
        pair_mass = np.minimum(n_dc, cap) ** power                                      # dataset within class
        per_class = d.assign(_m=pair_mass).drop_duplicates(["_st", "dataset_id"]).groupby("_st")["_m"].sum().to_dict()
        # P(row) = P(class) · P(dataset | class) · 1 / n(class, dataset)
        w = np.array([cls[s] / total for s in st]) * pair_mass / np.array([per_class[s] for s in st]) / n_dc
    else:
        raise ValueError(mode)
    w = np.nan_to_num(w, nan=0.0)
    return w / w.sum()


def epoch_order(weights: np.ndarray, n: int, seed: int, epoch: int):
    """(rows, seeds) for one epoch: n draws with replacement and one augmentation seed per draw."""
    g = np.random.default_rng([seed, epoch])
    rows = g.choice(len(weights), size=n, replace=True, p=weights)
    seeds = g.integers(0, 2**31 - 1, size=n)
    return rows, seeds


class RankSlicedBatches(torch.utils.data.Sampler):
    """Yields this rank's list of positions for each global batch of the epoch sequence (drop_last semantics)."""

    def __init__(self, n_items, global_batch, rank=0, world=1):
        assert global_batch % world == 0, f"global batch {global_batch} not divisible by {world} GPUs"
        self.n, self.gb, self.rank, self.world = n_items, global_batch, rank, world
        self.pb = global_batch // world

    def __len__(self):
        return self.n // self.gb

    def __iter__(self):
        for b in range(len(self)):
            start = b * self.gb + self.rank * self.pb
            yield list(range(start, start + self.pb))
