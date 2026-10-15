"""Image datasets and samplers over the prepared manifest (images are ≤512 px JPEGs written by waila-prepare).

Augmentation follows docs/experiments/training_plan.md: crops, flips, small rotations, blur, brightness/contrast and
JPEG / resampling noise (to kill compression shortcuts). No hue/saturation jitter and no grayscale: colour *is* the
label for stains and many submodalities.
"""
import io
import math
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter

from bio_waila.train.labels import LabelSpace


class JpegNoise:
    def __init__(self, p=0.3, q=(40, 95)):
        self.p, self.q = p, q

    def __call__(self, im):
        if random.random() > self.p:
            return im
        b = io.BytesIO()
        im.save(b, "JPEG", quality=random.randint(*self.q))
        b.seek(0)
        return Image.open(b).convert("RGB")


class DownUp:
    def __init__(self, p=0.2, scale=(0.4, 0.9)):
        self.p, self.scale = p, scale

    def __call__(self, im):
        if random.random() > self.p:
            return im
        w, h = im.size
        s = random.uniform(*self.scale)
        return im.resize((max(8, int(w * s)), max(8, int(h * s))), Image.BILINEAR).resize((w, h), Image.BILINEAR)


class Blur:
    def __init__(self, p=0.2, r=(0.1, 1.5)):
        self.p, self.r = p, r

    def __call__(self, im):
        return im.filter(ImageFilter.GaussianBlur(random.uniform(*self.r))) if random.random() < self.p else im


def image_transform(size: int, train: bool, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    """SigLIP-style preprocessing: squash to size x size (as the SigLIP processor does), normalise to [-1, 1]."""
    from torchvision import transforms as T
    if train:
        return T.Compose([
            T.RandomResizedCrop(size, scale=(0.7, 1.0), ratio=(0.8, 1.25), antialias=True),
            T.RandomHorizontalFlip(), T.RandomVerticalFlip(p=0.2), T.RandomRotation(10),
            T.ColorJitter(brightness=0.2, contrast=0.2),  # no hue / saturation
            Blur(), DownUp(), JpegNoise(),
            T.ToTensor(), T.Normalize(mean, std)])
    return T.Compose([T.Resize((size, size), antialias=True), T.ToTensor(), T.Normalize(mean, std)])


class ManifestImages(torch.utils.data.Dataset):
    """Returns (pixel tensor, {field: mode code}, {field: target vector}, row index)."""

    def __init__(self, df, root: Path, space: LabelSpace, transform):
        self.df, self.root, self.space, self.tf = df.reset_index(drop=True), Path(root), space, transform
        self.enc = [space.encode(r) for r in self.df.to_dict("records")]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        try:
            im = Image.open(self.root / self.df.at[i, "image"]).convert("RGB")
        except Exception:
            im = Image.new("RGB", (64, 64))  # unreadable file: keeps the batch shape; all fields masked below
            return self.tf(im), {f: 0 for f in self.space.fields}, {f: np.zeros(self.space.n(f), np.float32) for f in self.space.fields}, i
        e = self.enc[i]
        return self.tf(im), {f: e[f][0] for f in self.space.fields}, {f: e[f][1] for f in self.space.fields}, i


class SeededOrder(torch.utils.data.Dataset):
    """Item i = base[rows[i]], with python/numpy/torch RNGs seeded from seeds[i] first, so the augmentation of every draw
    is fixed by the run seed regardless of GPU count, worker count or which worker loads it (sampling.py)."""

    def __init__(self, base, rows, seeds):
        self.base, self.rows, self.seeds = base, rows, seeds

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        s = int(self.seeds[i])
        random.seed(s)
        np.random.seed(s % 2**32)
        torch.manual_seed(s)
        return self.base[int(self.rows[i])]


def to_device(batch, device):
    px, modes, tgts, idx = batch
    nb = device.type == "cuda"
    return (px.to(device, non_blocking=nb), {f: v.to(device, non_blocking=nb) for f, v in modes.items()},
            {f: v.to(device, non_blocking=nb) for f, v in tgts.items()}, idx)


def collate(batch):
    px = torch.stack([b[0] for b in batch])
    fields = batch[0][1].keys()
    modes = {f: torch.tensor([b[1][f] for b in batch], dtype=torch.long) for f in fields}
    tgts = {f: torch.from_numpy(np.stack([b[2][f] for b in batch])) for f in fields}
    idx = torch.tensor([b[3] for b in batch])
    return px, modes, tgts, idx


def balanced_weights(df, cap: int = 20000, power: float = 0.5) -> np.ndarray:
    """Per-image sampling weights: each dataset gets mass ∝ min(n, cap)^power (square-root up-weighting of small
    datasets, a hard cap on huge ones), spread evenly over its images."""
    n = df.groupby("dataset_id")["image"].transform("size").to_numpy().astype(float)
    mass = np.minimum(n, cap) ** power
    return mass / n


def make_sampler(df, epoch_size: int | None = None, cap: int = 20000, power: float = 0.5, seed: int = 0):
    w = balanced_weights(df, cap, power)
    g = torch.Generator().manual_seed(seed)
    return torch.utils.data.WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double),
                                                  num_samples=int(epoch_size or len(df)), replacement=True, generator=g)


def steps_per_epoch(n_samples: int, batch: int, world: int = 1) -> int:
    return max(1, math.ceil(n_samples / (batch * world)))
