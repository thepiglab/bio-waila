"""Per-field metrics on held-out datasets, shared by CLM and Jev evaluation.

Every number is computed per dataset and then macro-averaged over datasets (`*_macro_ds`), because every image
inherits its dataset's labels: a pooled image-level average is dominated by a few large datasets.

  single-label field  acc = top-1 is the label (HARD rows) / top-1 is inside the set (CAND rows, reported apart)
  multi-label field   f1 = set F1 of {top-1} ∪ {p ≥ thr} (at most `max_k`) vs the targets; top1 = top-1 ∈ targets
  ECE                 15-bin expected calibration error of the top-1 probability on HARD rows (single-label fields)
"""
from collections import defaultdict

import numpy as np
import torch

from bio_waila.train.labels import CAND, HARD, MULTI

MAX_K = {"body_part": 4, "domain": 3}


def ece(conf, correct, bins=15):
    conf, correct = np.asarray(conf), np.asarray(correct, float)
    if not len(conf):
        return float("nan")
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(e)


def _macro(values, ds):
    by = defaultdict(list)
    for v, d in zip(values, ds):
        by[d].append(v)
    return float(np.mean([np.mean(v) for v in by.values()])) if by else float("nan")


def field_metrics(logits: torch.Tensor, modes: torch.Tensor, tgt: torch.Tensor, ds, multilabel: bool, f: str,
                  temperature: float = 1.0, thr: float = 0.5):
    logits = logits.float() / temperature
    modes, tgt, ds = modes.cpu().numpy(), tgt.cpu().numpy() > 0, np.asarray(ds)
    out = {}
    if not multilabel:
        p = torch.softmax(logits, -1).cpu().numpy()
        top = p.argmax(-1)
        hit = tgt[np.arange(len(top)), top]
        h = modes == HARD
        c = (modes == CAND) | (modes == MULTI)
        out.update(n_hard=int(h.sum()), n_cand=int(c.sum()))
        if h.any():
            out.update(acc=float(hit[h].mean()), acc_macro_ds=_macro(hit[h], ds[h]), ece=ece(p.max(-1)[h], hit[h]))
        if c.any():
            out.update(acc_cand=float(hit[c].mean()), acc_cand_macro_ds=_macro(hit[c], ds[c]))
    else:
        p = torch.sigmoid(logits).cpu().numpy()
        top = p.argmax(-1)
        hit = tgt[np.arange(len(top)), top]
        k = MAX_K.get(f, 4)
        pos = (modes == HARD) | (modes == MULTI)
        out.update(n_hard=int(pos.sum()), n_cand=int((modes == CAND).sum()))
        if pos.any():
            f1s = []
            for i in np.nonzero(pos)[0]:
                order = np.argsort(-p[i])[:k]
                pred = {order[0]} | {j for j in order if p[i, j] >= thr}
                true = set(np.nonzero(tgt[i])[0])
                tp = len(pred & true)
                f1s.append(0.0 if tp == 0 else 2 * tp / (len(pred) + len(true)))
            f1s = np.array(f1s)
            out.update(f1=float(f1s.mean()), f1_macro_ds=_macro(f1s, ds[pos]), top1=float(hit[pos].mean()),
                       acc_macro_ds=_macro(hit[pos], ds[pos]))
        cm = modes == CAND
        if cm.any():
            out.update(acc_cand=float(hit[cm].mean()), acc_cand_macro_ds=_macro(hit[cm], ds[cm]))
    return out


def all_metrics(logits: dict, modes: dict, tgts: dict, ds, space, temperatures: dict | None = None):
    res = {}
    for f in space.fields:
        res[f] = field_metrics(logits[f], modes[f], tgts[f], ds, space.multilabel[f], f,
                               temperature=(temperatures or {}).get(f, 1.0))
    keyvals = [r["acc_macro_ds"] for r in res.values() if "acc_macro_ds" in r]
    res["mean_acc_macro_ds"] = float(np.mean(keyvals)) if keyvals else float("nan")
    return res


def fit_temperature(logits: torch.Tensor, modes: torch.Tensor, tgt: torch.Tensor, multilabel: bool, iters=200) -> float:
    """Scalar temperature minimising NLL on HARD rows (single-label) / BCE on HARD+MULTI rows (multi-label)."""
    sel = (modes == HARD) if not multilabel else ((modes == HARD) | (modes == MULTI))
    if sel.sum() < 20:
        return 1.0
    x, y = logits[sel].float().detach(), tgt[sel].float()
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=iters)

    def closure():
        opt.zero_grad()
        z = x / log_t.exp()
        loss = (-(y * torch.log_softmax(z, -1)).sum(-1).mean() if not multilabel
                else torch.nn.functional.binary_cross_entropy_with_logits(z, y))
        loss.backward()
        return loss
    opt.step(closure)
    return float(log_t.exp().clamp(0.05, 20))
