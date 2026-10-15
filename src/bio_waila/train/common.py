"""Small pieces every trainer uses: config loading, schedules, EMA, logging, seeding."""
import json
import math
import os
import random
import re
import time
from pathlib import Path

import numpy as np
import torch


def load_config(path, overrides=()):
    """YAML with ${ENV} / ${ENV:-default} expansion, plus dotted `key=value` overrides (values parsed as YAML)."""
    import yaml
    text = Path(path).read_text()
    text = re.sub(r"\$\{(\w+)(?::-([^}]*))?\}", lambda m: os.environ.get(m.group(1), m.group(2) or ""), text)
    cfg = yaml.safe_load(text) or {}
    for o in overrides:
        k, v = o.split("=", 1)
        d = cfg
        *head, last = k.split(".")
        for h in head:
            d = d.setdefault(h, {})
        d[last] = yaml.safe_load(v)
    return cfg


def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def cosine_lr(step, total, warmup, base=1.0, final=0.0):
    if step < warmup:
        return base * (step + 1) / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup)
    return final + (base - final) * 0.5 * (1 + math.cos(math.pi * min(1.0, t)))


class EMA:
    """Exponential moving average of the trainable parameters (evaluate the EMA weights)."""

    def __init__(self, model, decay=0.9998):
        self.decay = decay
        self.shadow = {n: p.detach().clone().float() for n, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self, model):
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(p.detach().float(), alpha=1 - self.decay)
            elif p.requires_grad:  # newly unfrozen (stage B): start tracking
                self.shadow[n] = p.detach().clone().float()

    @torch.no_grad()
    def swap(self, model):
        """Swap model weights with the EMA weights (call twice to swap back)."""
        for n, p in model.named_parameters():
            if n in self.shadow:
                tmp = p.detach().clone()
                p.copy_(self.shadow[n].to(p.dtype))
                self.shadow[n] = tmp.float()


class JsonlLog:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.t0 = time.time()

    def __call__(self, **kv):
        kv = {"t": round(time.time() - self.t0, 1), **kv}
        with open(self.path, "a") as f:
            f.write(json.dumps(kv, default=float) + "\n")
        print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in kv.items()), flush=True)


def save_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1, default=float))
