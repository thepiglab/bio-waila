"""Mode-aware losses for one field (see labels.py for the mode codes).

Single-label fields (softmax over the vocabulary):
  HARD  cross-entropy with label smoothing
  CAND  partial-label loss  -log Σ_{c∈S} p(c)       (a "multi" set in a single-label field arrives here as CAND)
Multi-label fields (independent sigmoids):
  HARD/MULTI  BCE, positives = targets, with ASL-style focusing on negatives (gamma_neg) and margin clipping
  CAND        at least one candidate is true: -log(1 - Π_{c∈S}(1-p_c)); classes outside S are negatives
MASKED rows contribute nothing. Returns (mean loss over contributing rows, number of contributing rows).

Options: `smooth_dist` [C, C] sends the label-smoothing mass to similar classes (similarity.py) instead of uniformly;
`cand_weight` scales candidate-set rows (ramped from 0 to 1 early in training: clean hard labels first).

Contrastive terms (similarity.py for the targets):
  soft_infonce        in-batch InfoNCE between states and the batch's label embeddings with *graded* targets
                      (T_ij = label similarity of rows i and j), both directions
  cross_field_infonce image-image InfoNCE whose targets average the per-field similarities over the fields both images
                      have labels for (same modality + body part, different submodality → partial credit)
"""
import torch
import torch.nn.functional as F

from bio_waila.train.labels import CAND, HARD, MASKED, MULTI


def softmax_field_loss(logits, modes, tgt, smoothing=0.1, smooth_dist=None, cand_weight=1.0):
    logp = F.log_softmax(logits.float(), -1)
    loss = torch.zeros(len(logits), device=logits.device)
    hard = modes == HARD
    if hard.any():
        y = tgt[hard]
        k = logits.shape[-1]
        if smooth_dist is not None:
            y = y * (1 - smoothing) + smoothing * (y @ smooth_dist.to(y.device))
        else:
            y = y * (1 - smoothing) + smoothing / k
        loss[hard] = -(y * logp[hard]).sum(-1)
    cand = (modes == CAND) | (modes == MULTI)
    if cand.any():
        m = tgt[cand] > 0
        loss[cand] = cand_weight * -torch.logsumexp(logp[cand].masked_fill(~m, float("-inf")), -1)
    keep = modes != MASKED
    return loss[keep].mean() if keep.any() else logits.sum() * 0, int(keep.sum())


def sigmoid_field_loss(logits, modes, tgt, gamma_neg=2.0, clip=0.05, cand_weight=1.0):
    x = logits.float()
    p = torch.sigmoid(x)
    loss = torch.zeros(len(x), device=x.device)
    pos_rows = (modes == HARD) | (modes == MULTI)
    if pos_rows.any():
        y = tgt[pos_rows]
        pr = p[pos_rows]
        pn = (pr - clip).clamp(min=0)  # probability shifting for negatives (ASL)
        lp = y * F.logsigmoid(x[pos_rows])
        ln = (1 - y) * torch.log((1 - pn).clamp(min=1e-8)) * pn.pow(gamma_neg)
        loss[pos_rows] = -(lp + ln).sum(-1)
    cand = modes == CAND
    if cand.any():
        m = tgt[cand] > 0
        pc = p[cand]
        none_true = torch.where(m, 1 - pc, torch.ones_like(pc)).prod(-1).clamp(max=1 - 1e-6)
        lpos = -torch.log1p(-none_true)
        pn = (pc - clip).clamp(min=0)
        lneg = -(torch.where(m, torch.zeros_like(pc), torch.log((1 - pn).clamp(min=1e-8)) * pn.pow(gamma_neg))).sum(-1)
        loss[cand] = cand_weight * (lpos + lneg)
    keep = modes != MASKED
    return loss[keep].mean() if keep.any() else logits.sum() * 0, int(keep.sum())


def field_loss(logits, modes, tgt, multilabel: bool, smoothing=0.1, gamma_neg=2.0, smooth_dist=None, cand_weight=1.0):
    if multilabel:
        return sigmoid_field_loss(logits, modes, tgt, gamma_neg=gamma_neg, cand_weight=cand_weight)
    return softmax_field_loss(logits, modes, tgt, smoothing=smoothing, smooth_dist=smooth_dist, cand_weight=cand_weight)


def multi_field_loss(logits: dict, modes: dict, tgts: dict, space, weights: dict | None = None, smooth_dists=None, **kw):
    """Sum of per-field losses (each a mean over its contributing rows), and a dict of the parts for logging."""
    total, parts = 0.0, {}
    for f in space.fields:
        l, n = field_loss(logits[f], modes[f], tgts[f], space.multilabel[f],
                          smooth_dist=(smooth_dists or {}).get(f), **kw)
        if n:
            total = total + (weights or {}).get(f, 1.0) * l
            parts[f] = float(l.detach())
    return total, parts


def _soft_ce(logits, T):
    """Cross-entropy of softmax(logits) against row-normalised targets T (rows with no target mass are skipped)."""
    keep = T.sum(-1) > 0
    if not keep.any():
        return logits.sum() * 0
    p = T[keep] / T[keep].sum(-1, keepdim=True)
    return -(p * F.log_softmax(logits[keep].float(), -1)).sum(-1).mean()


def soft_infonce(state_z, action_z, T, scale, gamma=1.0):
    """Bidirectional in-batch InfoNCE with graded targets. state_z [B,D], action_z [B,D] (row j's label embedding),
    T [B,B] target similarity (T_ii = 1). Binary InfoNCE is the special case T = identity."""
    logits = scale * state_z @ action_z.t()
    T = T.pow(gamma)
    return (_soft_ce(logits, T) + _soft_ce(logits.t(), T.t())) / 2


def cross_field_targets(tgts: dict, modes: dict, S: dict, weights: dict | None = None):
    """[B,B] image-image targets: weighted mean over fields of the per-field soft similarity, counting only fields
    labelled (not masked) for both images; zero where they share no labelled field; diagonal zero (self)."""
    from bio_waila.train.similarity import soft_targets
    num, den = 0.0, 0.0
    for f, y in tgts.items():
        known = (modes[f] != MASKED).float()
        both = known[:, None] * known[None, :]
        w = (weights or {}).get(f, 1.0)
        num = num + w * both * soft_targets(y, S[f])
        den = den + w * both
    T = num / torch.as_tensor(den).clamp(min=1e-8)
    return T.fill_diagonal_(0)


def cross_field_infonce(emb_z, tgts: dict, modes: dict, S: dict, scale, weights=None, gamma=2.0):
    """Image-image InfoNCE (X-CLR / supervised-contrastive style) with cross-field graded targets. gamma > 1 sharpens
    the targets so near-identical label sets dominate over the many images that share only a modality."""
    T = cross_field_targets(tgts, modes, S, weights).pow(gamma)
    logits = (scale * emb_z @ emb_z.t()).float()
    logits = logits.masked_fill(torch.eye(len(emb_z), dtype=torch.bool, device=emb_z.device), -1e4)  # self: excluded
    return _soft_ce(logits, T)
