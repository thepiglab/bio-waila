"""Jev-style decision model on Gemma 4 (prototype: google/gemma-4-E4B-it): image + one field's question + lettered
options → calibrated probabilities over the options from the next-token distribution (docs/models/jev_methods.md,
decision_index_2026-10.md §8).

  readout   P(letter) = Σ_{tokens t that decode to that letter} softmax(logits_last)[t]   (Cygnet / deck31b trick)
  options   fields with ≤ k_max classes show all of them (shuffled in training, canonical at evaluation); bigger
            fields (body part: 99) show a sampled subset that always contains the gold label(s); evaluation runs
            a two-round tournament (chunks of k_max, then the chunk finalists)
  formulations  F2 independent: no context.  F3 progressive: earlier fields' answers written into the prompt
            ("Known: modality = x-ray imaging; ..."), with context dropout in training. Order: domain last.
  training  LoRA on the language model only; the vision tower is frozen (optionally unfrozen at a lower LR)

Backends share one method, `letter_logits(images, prompts) → [B, 26] log-probabilities over A..Z`:
  GemmaBackend  the real model      FakeBackend  a tiny trainable stand-in for offline tests
"""
import hashlib
import random

import numpy as np
import torch
import torch.nn as nn

from bio_waila.train.labels import CAND, HARD, MASKED, MULTI

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
ORDER = ["modality", "submodality", "body_part", "stain", "specimen", "domain"]
QUESTION = {
    "modality": "Which imaging modality produced this image?",
    "submodality": "Which imaging technique (submodality) is this?",
    "body_part": "Which body part, organ, tissue or cell type is shown?",
    "stain": "Which stain or labelling was used?",
    "specimen": "What is the specimen (species or sample type)?",
    "domain": "Which medical specialty does this image belong to?",
}


# ── prompts and options ──────────────────────────────────────────────────────────────────────────────────────────
def choose_options(n_classes, vec, mode, k_max=20, rng=None, train=True, prefer=None):
    """Indices of the classes shown as options. Always includes every positive / candidate class (truncated to
    k_max), fills with `prefer` (hard negatives, e.g. a CLM teacher's top-k) and then random negatives."""
    rng = rng or random
    if n_classes <= k_max:
        opts = list(range(n_classes))
    else:
        pos = [int(i) for i in np.nonzero(vec)[0]][:k_max] if mode != MASKED else []
        rest = [i for i in (prefer or []) if i not in pos]
        neg = [i for i in range(n_classes) if i not in pos and i not in rest]
        rng.shuffle(neg)
        opts = (pos + rest + neg)[:k_max]
    if train:
        rng.shuffle(opts)
    return opts


def build_prompt(field, option_names, context=None):
    lines = [f"Field: {field.replace('_', ' ')}."]
    if context:
        lines.append("Known: " + "; ".join(f"{k.replace('_', ' ')} = {v}" for k, v in context.items()) + ".")
    lines.append(QUESTION[field])
    lines += [f"{LETTERS[i]}. {name}" for i, name in enumerate(option_names)]
    lines.append("Answer with the letter of the best option only.")
    return "\n".join(lines)


def option_target(vec, mode, opts):
    """(mode for the loss, target over the shown options)."""
    t = np.array([vec[i] for i in opts], np.float32)
    if mode == MASKED or t.sum() == 0:
        return MASKED, t
    if mode in (HARD, MULTI) and t.sum() == 1:
        return HARD, t
    if mode == MULTI:
        return MULTI, t / t.sum()   # several true classes: spread the mass (multi-label fields)
    return CAND, t


def gold_context(row_enc, space, upto_field, drop_p=0.0, rng=None):
    """F3 context: earlier fields' gold answers (hard / multi only), each dropped with probability drop_p."""
    rng = rng or random
    ctx = {}
    for f in ORDER:
        if f == upto_field:
            break
        if f not in space.fields:
            continue
        mode, vec = row_enc[f]
        if mode in (HARD, MULTI) and rng.random() >= drop_p:
            ctx[f] = ", ".join(space.classes[f][i] for i in np.nonzero(vec)[0])
    return ctx


def letter_loss(logp, modes, tgts, smoothing=0.0):
    """logp [B, k] (log-probs over the shown options), modes [B], tgts [B, k]."""
    loss = torch.zeros(len(logp), device=logp.device)
    valid = logp > -1e3                              # padded option slots carry a large negative (finite) log-prob
    soft = (modes == HARD) | (modes == MULTI)
    if soft.any():
        y = tgts[soft]
        if smoothing:
            v = valid[soft].float()
            y = y * (1 - smoothing) + smoothing * v / v.sum(-1, keepdim=True)
        loss[soft] = -(y * logp[soft]).sum(-1)
    cand = modes == CAND
    if cand.any():
        loss[cand] = -torch.logsumexp(logp[cand].masked_fill((tgts[cand] <= 0) | ~valid[cand], float("-inf")), -1)
    keep = modes != MASKED
    return loss[keep].mean() if keep.any() else logp.sum() * 0


# ── backends ─────────────────────────────────────────────────────────────────────────────────────────────────────
def letter_token_groups(tokenizer):
    """{letter index: [token ids that decode (stripped) to that letter]} over the whole vocabulary."""
    groups = {i: [] for i in range(len(LETTERS))}
    vocab = tokenizer.get_vocab()
    for tok, tid in vocab.items():
        s = tokenizer.convert_tokens_to_string([tok]).strip()
        if len(s) == 1 and s in LETTERS:
            groups[LETTERS.index(s)].append(tid)
    return groups


class GemmaBackend(nn.Module):
    def __init__(self, name, lora=None, max_soft_tokens=280, dtype="bfloat16", train_vision=False,
                 gradient_checkpointing=True):
        super().__init__()
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.proc = AutoProcessor.from_pretrained(name)
        self.proc.tokenizer.padding_side = "left"
        self.model = AutoModelForImageTextToText.from_pretrained(name, dtype=getattr(torch, dtype))
        for p in self.model.parameters():
            p.requires_grad = False
        if lora:
            from peft import LoraConfig, get_peft_model
            cfg = LoraConfig(r=lora.get("r", 16), lora_alpha=lora.get("alpha", 32), lora_dropout=lora.get("dropout", 0.05),
                             target_modules=lora.get("target_modules",
                                                     r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"))
            self.model = get_peft_model(self.model, cfg)
        if train_vision:
            for n, p in self.model.named_parameters():
                if "vision" in n:
                    p.requires_grad = True
        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            if hasattr(self.model, "enable_input_require_grads"):
                self.model.enable_input_require_grads()
        groups = letter_token_groups(self.proc.tokenizer)
        V = max(max(g) for g in groups.values() if g) + 1
        member = torch.zeros(len(LETTERS), V, dtype=torch.bool)
        for i, ids in groups.items():
            member[i, ids] = True
        self.register_buffer("member", member, persistent=False)
        self.budget = max_soft_tokens

    def _chat(self, prompt):
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        try:
            return self.proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        except TypeError:
            return self.proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)

    def letter_logits(self, images, prompts):
        dev = next(self.model.parameters()).device
        inputs = self.proc(text=[self._chat(p) for p in prompts], images=[[im] for im in images], padding=True,
                           return_tensors="pt", max_soft_tokens=self.budget).to(dev)
        try:
            out = self.model(**inputs, logits_to_keep=1)
        except TypeError:
            out = self.model(**inputs)
        logp = torch.log_softmax(out.logits[:, -1].float(), -1)            # [B, V]
        m = self.member[:, : logp.shape[-1]]
        return torch.logsumexp(logp.unsqueeze(1).masked_fill(~m.unsqueeze(0), float("-inf")), -1)  # [B, 26]


class FakeBackend(nn.Module):
    """Offline stand-in: image mean colour + a hash of each option line → letter logits. Trainable, deterministic."""

    def __init__(self, dim=32):
        super().__init__()
        self.img = nn.Linear(3, dim)
        self.opt = nn.Embedding(4096, dim)

    def letter_logits(self, images, prompts):
        x = torch.tensor(np.stack([np.asarray(im.convert("RGB").resize((4, 4)), np.float32).reshape(-1, 3).mean(0) / 255
                                   for im in images])).to(self.img.weight.device)
        h = self.img(x)                                                    # [B, dim]
        out = torch.full((len(prompts), len(LETTERS)), -1e4, device=h.device)
        for b, p in enumerate(prompts):
            lines = [line for line in p.split("\n") if len(line) > 2 and line[1] == "." and line[0] in LETTERS]
            for line in lines:
                tid = int(hashlib.md5((p.split("\n")[0] + line[3:]).encode()).hexdigest()[:6], 16) % 4096
                out[b, LETTERS.index(line[0])] = (self.opt.weight[tid] * h[b]).sum()
        return torch.log_softmax(out, -1)


def make_backend(cfg):
    if cfg.get("kind", "gemma") == "fake":
        return FakeBackend()
    return GemmaBackend(cfg["name"], lora=cfg.get("lora"), max_soft_tokens=cfg.get("max_soft_tokens", 280),
                        dtype=cfg.get("dtype", "bfloat16"), train_vision=cfg.get("train_vision", False),
                        gradient_checkpointing=cfg.get("gradient_checkpointing", True))


# ── inference over the whole vocabulary of a field ───────────────────────────────────────────────────────────────
@torch.no_grad()
def field_logprobs(backend, images, field, space, contexts=None, k_max=20, finalists_per_chunk=4):
    """[B, n_classes] log-probabilities over every class of `field` (tournament when n_classes > k_max)."""
    C = space.n(field)
    names = space.classes[field]
    B = len(images)
    ctx = contexts or [None] * B

    def run(option_sets):
        prompts = [build_prompt(field, [names[i] for i in opts], ctx[b]) for b, opts in enumerate(option_sets)]
        lp = backend.letter_logits(images, prompts).float().cpu()
        return [torch.log_softmax(lp[b, : len(opts)], -1) for b, opts in enumerate(option_sets)]

    out = torch.full((B, C), -30.0)
    if C <= k_max:
        res = run([list(range(C))] * B)
        for b in range(B):
            out[b] = res[b]
        return out
    chunks = [list(range(s, min(C, s + k_max))) for s in range(0, C, k_max)]
    finalists = [[] for _ in range(B)]
    for ch in chunks:
        res = run([ch] * B)
        for b in range(B):
            out[b, ch] = res[b] - 10.0                     # below any finalist, but keeps the within-chunk order
            top = torch.topk(res[b], min(finalists_per_chunk, len(ch))).indices.tolist()
            finalists[b] += [ch[i] for i in top]
    res = run([f[:k_max] for f in finalists])
    for b in range(B):
        out[b, finalists[b][:k_max]] = res[b]
    return out
