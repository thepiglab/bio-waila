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
# Training-only rewordings (randomised serialisation, as in Baseten's mammoth-1 recipe); evaluation uses QUESTION.
PARAPHRASES = {
    "modality": ["What kind of imaging produced this picture?", "Name the imaging modality of this image."],
    "submodality": ["Which specific acquisition technique was used?", "Name the imaging technique shown here."],
    "body_part": ["What anatomy, organ, tissue or cell type is visible?", "Which body region or tissue does this show?"],
    "stain": ["Which stain, dye or label was applied to the sample?", "Name the staining or labelling method."],
    "specimen": ["Which species or sample type is this specimen?", "What organism or sample is imaged?"],
    "domain": ["Which clinical or research specialty would use this image?", "Name the medical specialty of this image."],
}
# The "I don't know" option (mammoth-1 §2.5): its target is 1 when no shown option is correct. Always the last letter.
ABSTAIN = "none of these options"


# ── prompts and options ──────────────────────────────────────────────────────────────────────────────────────────
def choose_options(n_classes, vec, mode, k_max=20, rng=None, train=True, prefer=None, absent_p=0.0, prefer_p=1.0):
    """Indices of the classes shown as options. Always includes every positive / candidate class (truncated to
    k_max), fills with `prefer` (hard negatives, e.g. a CLM teacher's top-k; used with probability prefer_p) and then
    random negatives. With probability absent_p (training, abstain on) the positives are left out instead, so the
    correct answer is the abstain option."""
    rng = rng or random
    pos = [int(i) for i in np.nonzero(vec)[0]] if mode != MASKED else []
    if pos and absent_p and rng.random() < absent_p:
        opts = [i for i in range(n_classes) if i not in pos]
        rng.shuffle(opts)
        opts = opts[:k_max]
    elif n_classes <= k_max:
        opts = list(range(n_classes))
    else:
        pos = pos[:k_max]
        rest = [i for i in (prefer or []) if i not in pos] if rng.random() < prefer_p else []
        neg = [i for i in range(n_classes) if i not in pos and i not in rest]
        rng.shuffle(neg)
        opts = (pos + rest + neg)[:k_max]
    if train:
        rng.shuffle(opts)
    return opts


def build_prompt(field, option_names, context=None, abstain=False, rng=None):
    """`rng` (training) picks a random question wording; None keeps the canonical one."""
    lines = [f"Field: {field.replace('_', ' ')}."]
    if context:
        lines.append("Known: " + "; ".join(f"{k.replace('_', ' ')} = {v}" for k, v in context.items()) + ".")
    lines.append(rng.choice([QUESTION[field]] + PARAPHRASES.get(field, [])) if rng else QUESTION[field])
    names = list(option_names) + ([ABSTAIN] if abstain else [])
    lines += [f"{LETTERS[i]}. {name}" for i, name in enumerate(names)]
    lines.append("Answer with the letter of the best option only.")
    return "\n".join(lines)


def option_target(vec, mode, opts, abstain=False):
    """(mode for the loss, target over the shown options [+ the abstain slot])."""
    t = np.array([vec[i] for i in opts], np.float32)
    if abstain:
        t = np.append(t, np.float32(0))
        if mode != MASKED and t.sum() == 0:      # nothing correct is shown → "none of these options"
            t[-1] = 1
            return HARD, t
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


def letter_loss(logp, modes, tgts, smoothing=0.0, brier=0.0):
    """logp [B, k] (log-probs over the shown options), modes [B], tgts [B, k]. `brier` adds λ·Brier to the CE
    (mammoth-1 uses λ = 1); for candidate rows the Brier term is on the candidate set's total probability."""
    loss = torch.zeros(len(logp), device=logp.device)
    valid = logp > -1e3                              # padded option slots carry a large negative (finite) log-prob
    soft = (modes == HARD) | (modes == MULTI)
    if soft.any():
        y = tgts[soft]
        if smoothing:
            v = valid[soft].float()
            y = y * (1 - smoothing) + smoothing * v / v.sum(-1, keepdim=True)
        loss[soft] = -(y * logp[soft]).sum(-1)
        if brier:
            loss[soft] = loss[soft] + brier * ((logp[soft].exp() - tgts[soft]) ** 2 * valid[soft]).sum(-1)
    cand = modes == CAND
    if cand.any():
        lse = torch.logsumexp(logp[cand].masked_fill((tgts[cand] <= 0) | ~valid[cand], float("-inf")), -1)
        loss[cand] = -lse + (brier * (1 - lse.exp()) ** 2 if brier else 0)
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
    """Letter-readout backend. Despite the name it takes any HF image-text-to-text repo (`name`); `adapter` is an optional
    LoRA repo/dir merged on top of `name` (inference), `device_map="auto"` places the model on the visible GPU(s), and
    `max_soft_tokens` is only passed to Gemma processors (the knob does not exist elsewhere)."""

    def __init__(self, name, lora=None, max_soft_tokens=280, dtype="bfloat16", train_vision=False,
                 gradient_checkpointing=True, adapter=None, trust_remote_code=False, device_map=None, full_ft=False):
        super().__init__()
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.proc = AutoProcessor.from_pretrained(name, trust_remote_code=trust_remote_code)
        self.proc.tokenizer.padding_side = "left"
        kw = {"device_map": device_map} if device_map else {}
        self.model = AutoModelForImageTextToText.from_pretrained(name, dtype=getattr(torch, dtype),
                                                                 trust_remote_code=trust_remote_code, **kw)
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter).merge_and_unload()
        for p in self.model.parameters():
            p.requires_grad = False
        if lora:
            from peft import LoraConfig, get_peft_model
            cfg = LoraConfig(r=lora.get("r", 16), lora_alpha=lora.get("alpha", 32), lora_dropout=lora.get("dropout", 0.05),
                             target_modules=lora.get("target_modules",
                                                     r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"))
            self.model = get_peft_model(self.model, cfg)
        elif full_ft:                    # mammoth-1 style: every language-model weight, tower frozen (fp32 master weights)
            for n, p in self.model.named_parameters():
                if "language_model" in n:
                    p.requires_grad = True
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
        self.budget = max_soft_tokens if type(self.proc).__name__.lower().startswith("gemma") else None

    def _chat(self, prompt):
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        try:
            return self.proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        except TypeError:
            return self.proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)

    def letter_logits(self, images, prompts):
        dev = next(self.model.parameters()).device
        kw = {"max_soft_tokens": self.budget} if self.budget else {}
        inputs = self.proc(text=[self._chat(p) for p in prompts], images=[[im] for im in images], padding=True,
                           return_tensors="pt", **kw).to(dev)
        try:
            out = self.model(**inputs, logits_to_keep=1)
        except TypeError:
            out = self.model(**inputs)
        return self._letters(out.logits[:, -1])

    def _letters(self, logits):
        logp = torch.log_softmax(logits.float(), -1)                       # [N, V]
        m = self.member[:, : logp.shape[-1]].to(logp.device)
        if m.shape[1] < logp.shape[-1]:                # member stops at the largest letter id, not at the vocab size
            m = torch.nn.functional.pad(m, (0, logp.shape[-1] - m.shape[1]))
        return torch.logsumexp(logp.unsqueeze(1).masked_fill(~m.unsqueeze(0), float("-inf")), -1)  # [N, 26]

    def packed_letter_logits(self, images, prompt_lists, bidirectional=False):
        """F4: every prompt of an image in one sequence over the shared image prefix (models/jev_packed.py).
        Returns [Σ len(prompt_lists[i]), 26], in prompt order. With bidirectional=False it equals letter_logits
        on the flattened prompts."""
        from bio_waila.models.jev_packed import collate_packed, pack, packed_readout_logits
        dev = next(self.model.parameters()).device
        kw = {"max_soft_tokens": self.budget} if self.budget else {}
        flat = [(im, p) for im, ps in zip(images, prompt_lists) for p in ps]
        inputs = self.proc(text=[self._chat(p) for _, p in flat], images=[[im] for im, _ in flat], padding=True,
                           return_tensors="pt", **kw)
        img_id = self.model.config.image_token_id
        rows = [r[m.bool()].tolist() for r, m in zip(inputs["input_ids"], inputs["attention_mask"])]
        packed, first, i = [], [], 0
        for ps in prompt_lists:
            seqs = rows[i:i + len(ps)]
            img_end = max(t for t, x in enumerate(seqs[0]) if x == img_id) + 1
            packed.append(pack(seqs, min_prefix=img_end))
            first.append(i)
            i += len(ps)
        ids, block, pos, _, ro = collate_packed(packed, pad_id=self.proc.tokenizer.pad_token_id or 0)
        vis = {k: inputs[k][first].to(dev) for k in ("pixel_values", "image_position_ids") if k in inputs}
        base = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
        logits = packed_readout_logits(base, ids.to(dev), block.to(dev), pos.to(dev), ro,
                                       bidirectional=bidirectional, **vis)
        return self._letters(logits)


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

    def packed_letter_logits(self, images, prompt_lists, bidirectional=False):
        return self.letter_logits([im for im, ps in zip(images, prompt_lists) for _ in ps],
                                  [p for ps in prompt_lists for p in ps])


def make_backend(cfg):
    if cfg.get("kind", "gemma") == "fake":
        return FakeBackend()
    return GemmaBackend(cfg["name"], lora=cfg.get("lora"), max_soft_tokens=cfg.get("max_soft_tokens", 280),
                        dtype=cfg.get("dtype", "bfloat16"), train_vision=cfg.get("train_vision", False),
                        gradient_checkpointing=cfg.get("gradient_checkpointing", True), adapter=cfg.get("adapter"),
                        trust_remote_code=cfg.get("trust_remote_code", False), device_map=cfg.get("device_map"),
                        full_ft=cfg.get("full_ft", False))


# ── inference over the whole vocabulary of a field ───────────────────────────────────────────────────────────────
def tournament(B, C, k_max=20, finalists_per_chunk=4):
    """Generator form of the tournament: yields a list of calls (each: one option set per image), receives the
    list of their results (per call, per image, a log-prob tensor over that set's options), returns [B, C].

    C <= k_max is a single call. Otherwise round 1 scores chunks of k_max consecutive classes; the top
    `finalists_per_chunk` of every chunk (shrunk so all finalists fit in one set) meet in a final round. Finalists
    carry their final-round log-probs; everyone else keeps the within-chunk order, 10 nats below any finalist."""
    out = torch.full((B, C), -30.0)
    if C <= k_max:
        (res,) = yield [[list(range(C))] * B]
        for b in range(B):
            out[b] = res[b]
        return out
    chunks = [list(range(s, min(C, s + k_max))) for s in range(0, C, k_max)]
    keep = max(1, min(finalists_per_chunk, k_max // len(chunks)))
    finalists = [[] for _ in range(B)]
    results = yield [[ch] * B for ch in chunks]
    for ch, res in zip(chunks, results):
        for b in range(B):
            out[b, ch] = res[b] - 10.0                     # below any finalist, but keeps the within-chunk order
            top = torch.topk(res[b], min(keep, len(ch))).indices.tolist()
            finalists[b] += [ch[i] for i in top]
    sets = [f[:k_max] for f in finalists]
    (res,) = yield [sets]
    for b in range(B):
        out[b, sets[b]] = res[b]
    return out


def tournament_logprobs(run, B, C, k_max=20, finalists_per_chunk=4):
    """[B, C] log-probabilities over C classes; `run(option_sets)` scores one option set per image."""
    g = tournament(B, C, k_max, finalists_per_chunk)
    calls = next(g)
    while True:
        try:
            calls = g.send([run(c) for c in calls])
        except StopIteration as e:
            return e.value


@torch.no_grad()
def field_logprobs(backend, images, field, space, contexts=None, k_max=20, finalists_per_chunk=4, abstain=False):
    """[B, n_classes] log-probabilities over every class of `field` (tournament when n_classes > k_max). With
    `abstain` the prompt carries the "none of these options" letter; class scores keep the mass it took (not
    renormalised), so a chunk that holds no right answer scores lower across the board."""
    names = space.classes[field]
    ctx = contexts or [None] * len(images)

    def run(option_sets):
        prompts = [build_prompt(field, [names[i] for i in opts], ctx[b], abstain=abstain)
                   for b, opts in enumerate(option_sets)]
        lp = backend.letter_logits(images, prompts).float().cpu()
        return [torch.log_softmax(lp[b, : len(opts) + abstain], -1)[: len(opts)] for b, opts in enumerate(option_sets)]

    return tournament_logprobs(run, len(images), space.n(field), k_max, finalists_per_chunk)


@torch.no_grad()
def packed_fields_logprobs(backend, images, fields, space, k_max=20, finalists_per_chunk=4, abstain=False,
                           bidirectional=False):
    """F4 evaluation: {field: [B, n_classes]} with every field's tournament run in lockstep, each round's questions
    of all fields packed into one sequence per image."""
    B = len(images)
    gens = {f: tournament(B, space.n(f), k_max, finalists_per_chunk) for f in fields}
    pending = {f: next(g) for f, g in gens.items()}
    out = {}
    while pending:
        qs = [(f, c) for f, calls in pending.items() for c in range(len(calls))]   # questions asked of every image
        prompts = [[build_prompt(f, [space.classes[f][i] for i in pending[f][c][b]], abstain=abstain) for f, c in qs]
                   for b in range(B)]
        lp = backend.packed_letter_logits(images, prompts, bidirectional=bidirectional).float().cpu()
        lp = lp.view(B, len(qs), -1)
        res = {f: [[None] * B for _ in pending[f]] for f in pending}
        for q, (f, c) in enumerate(qs):
            for b in range(B):
                n = len(pending[f][c][b])
                res[f][c][b] = torch.log_softmax(lp[b, q, : n + abstain], -1)[:n]
        nxt = {}
        for f in pending:
            try:
                nxt[f] = gens[f].send(res[f])
            except StopIteration as e:
                out[f] = e.value
        pending = nxt
    return out
