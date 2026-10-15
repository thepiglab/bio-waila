"""CLM-LLM: the original CLM recipe (Contrastive-LM/CLM: frozen LLM, pooled embeddings, two small MLP heads) on a
non-Chinese multimodal LLM instead of Qwen3-8B.

  state  = frozen MLLM embedding of (image + a summarising prompt), computed once per image and cached
  action = frozen MLLM embedding of each label's text (a few prompt variants per label), cached once
  score  = scale_f * cos(state_head_f(state), action_head(action_c)) + bias_f

CLM uses one state head over "context + question"; we embed each image once with a field-agnostic prompt and give
each field its own state head (six MLPs), so the encoder runs once per image instead of six times. The action head is
shared by all fields (label texts carry the field name: "submodality: fundus photography").

Encoders (`encoder.kind`):
  gemma4      Gemma 4 (e.g. google/gemma-4-E4B-it) via transformers; hidden state of layer `layer`, pooled `pool`
  fake        deterministic random features for tests

Readout (docs/models/clm_llm_vs_encoder.md): frozen-MLLM probes are best from a *middle* layer (≈50-70% depth) and
from the mean over image tokens, not the last token of the last layer; the default state readout is therefore
`layer: 0.6, pool: image_mean+last` (image-token mean concatenated with the last prompt token). Label texts have no
image tokens and use `label_pool: last`. Expect this arm to trail SigLIP 2 (MIEB linear probe: SigLIP-so400m 84.6
vs E5-V 74.5); it is a cheap side arm, not the primary.
"""
import hashlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

STATE_PROMPT = ("Look at this biomedical image: imaging modality and technique, anatomy, stain, specimen and medical "
                "specialty. Summarize the image in one word:")
LABEL_PROMPT = "{field}: {text}\nSummarize the above in one word:"


class FakeEncoder:
    """Tests: features that depend on image colour (so they are learnable) and on text (deterministic hash)."""
    dim = 64

    def embed_images(self, images, prompt=STATE_PROMPT):
        out = []
        for im in images:
            a = np.asarray(im.convert("RGB").resize((8, 8)), np.float32).reshape(-1, 3).mean(0) / 255
            g = np.random.default_rng(int(a.sum() * 1000) % 2**31)
            out.append(np.concatenate([np.tile(a, 8), g.normal(0, 0.05, self.dim - 24)]))
        return np.stack(out).astype(np.float32)

    def embed_texts(self, texts):
        return np.stack([np.random.default_rng(int(hashlib.md5(t.encode()).hexdigest()[:8], 16)).normal(0, 1, self.dim)
                         for t in texts]).astype(np.float32)


class Gemma4Encoder:
    """Frozen Gemma 4 as an embedding model. Left padding, so the last position is the last real token."""

    def __init__(self, name, layer=0.6, pool="image_mean+last", label_pool="last", max_soft_tokens=280,
                 dtype="bfloat16", device=None):
        from transformers import AutoModel, AutoProcessor
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.proc = AutoProcessor.from_pretrained(name)
        self.proc.tokenizer.padding_side = "left"
        self.model = AutoModel.from_pretrained(name, dtype=getattr(torch, dtype)).to(self.device).eval()
        self.layer, self.pool, self.label_pool, self.budget = layer, pool, label_pool, max_soft_tokens
        cfg = self.model.config
        self.image_token_id = getattr(cfg, "image_token_id", None) or 258_880

    def _chat(self, content):
        return self.proc.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True,
                                             tokenize=False)

    @torch.no_grad()
    def _pool(self, inputs, pool):
        out = self.model(**inputs, output_hidden_states=True)
        hs = out.hidden_states                                          # embeddings + one per layer
        li = self.layer if isinstance(self.layer, int) else int(round(self.layer * (len(hs) - 1)))
        h = hs[li].float()                                              # [B, T, D]
        am = inputs["attention_mask"].unsqueeze(-1).float()
        parts = []
        for p in pool.split("+"):
            if p == "last":
                parts.append(h[:, -1])                                  # left padding: the last real token
            elif p == "mean":
                parts.append((h * am).sum(1) / am.sum(1).clamp(min=1))
            elif p == "image_mean":
                m = (inputs["input_ids"] == self.image_token_id).unsqueeze(-1).float()
                parts.append((h * m).sum(1) / m.sum(1).clamp(min=1))
            else:
                raise ValueError(p)
        return torch.cat(parts, -1).cpu().numpy()

    def embed_images(self, images, prompt=STATE_PROMPT):
        texts = [self._chat([{"type": "image"}, {"type": "text", "text": prompt}])] * len(images)
        inputs = self.proc(text=texts, images=[[im.convert("RGB")] for im in images], padding=True, return_tensors="pt",
                           max_soft_tokens=self.budget).to(self.device)
        return self._pool(inputs, self.pool)

    def embed_texts(self, texts):
        chats = [self._chat([{"type": "text", "text": t}]) for t in texts]
        inputs = self.proc(text=chats, padding=True, return_tensors="pt").to(self.device)
        return self._pool(inputs, self.label_pool)


def make_encoder(cfg: dict):
    kind = cfg.get("kind", "gemma4")
    if kind == "fake":
        return FakeEncoder()
    if kind == "gemma4":
        return Gemma4Encoder(cfg["name"], layer=cfg.get("layer", 0.6), pool=cfg.get("pool", "image_mean+last"),
                             label_pool=cfg.get("label_pool", "last"), max_soft_tokens=cfg.get("max_soft_tokens", 280),
                             dtype=cfg.get("dtype", "bfloat16"))
    raise ValueError(f"unknown encoder kind {kind}")


def label_texts(space, field, templates=None):
    """[class][k] texts for the action side: the field name plus each prompt variant."""
    return [[LABEL_PROMPT.format(field=field.replace("_", " "), text=p) for p in cls] for cls in space.prompts(field, templates)]


# ── heads: the official CLM architecture ─────────────────────────────────────────────────────────────────────────
# make_head is adapted from Contrastive-LM/CLM src/clm/heads.py (Apache-2.0, https://github.com/Contrastive-LM/CLM):
# hidden -> width -> ... -> proj MLP with optional LayerNorm / residual hidden blocks. Defaults follow CLM's `choice`
# task (width 1536, depth 3, LayerNorm on, proj 512). Module names match CLM's, so `clm_checkpoint()` below writes
# files that CLM's own HeadPair can load (for the shared-state variant).
def make_head(width: int, depth: int = 3, proj: int = 512, activation: str = "gelu", layernorm: bool = True,
              residual: bool = False, hidden: int = 4096):
    act = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}[activation]

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Linear(hidden, width)
            self.hidden = nn.ModuleList(nn.Linear(width, width) for _ in range(depth - 2))
            self.norms = nn.ModuleList((nn.LayerNorm(width) if layernorm else nn.Identity()) for _ in range(depth - 2))
            self.out = nn.Linear(width, proj)
            self.act = act()
            self.residual = residual

        def forward(self, x):
            x = self.act(self.inp(x))
            for lin, nrm in zip(self.hidden, self.norms):
                h = self.act(nrm(lin(x)))
                x = x + h if self.residual else h
            return self.out(x)

    return Head()


def group_masked_infonce(state_z, action_z, codes, scale):
    """CLM's `_clm_loss` (train/finetune.py): bidirectional in-batch InfoNCE over (state, its action) pairs, with
    entries that share a code (the same label) masked out as false negatives. state_z/action_z are normalised."""
    logits = scale * state_z @ action_z.t()
    labels = torch.arange(len(state_z), device=state_z.device)
    same = codes.unsqueeze(0) == codes.unsqueeze(1)
    same.fill_diagonal_(False)
    fwd = F.cross_entropy(logits.masked_fill(same, float("-inf")), labels)
    bwd = F.cross_entropy(logits.t().masked_fill(same, float("-inf")), labels)
    return (fwd + bwd) / 2


class CLMHeads(nn.Module):
    """Per-field state heads (default) or one shared state head (`shared_state`, closest to CLM: one state head for
    every question, fields told apart only through the label texts), and one shared action head."""

    def __init__(self, space, d_in, label_emb: dict, width=1536, d_out=512, depth=3, layernorm=True, residual=False,
                 activation="gelu", shared_state=False):
        super().__init__()
        self.space = space
        d_label = next(iter(label_emb.values())).shape[-1]
        self.cfg = dict(width=width, depth=depth, projection_dim=d_out, activation=activation, layernorm=layernorm,
                        residual=residual, hidden_size=d_in, label_hidden_size=d_label, shared_state=shared_state)
        mk = lambda h: make_head(width, depth, d_out, activation, layernorm, residual, hidden=h)  # noqa: E731
        self.shared_state = shared_state
        self.state = nn.ModuleDict({"all": mk(d_in)} if shared_state else {f: mk(d_in) for f in space.fields})
        self.action = mk(d_label)
        for f in space.fields:
            self.register_buffer(f"lab_{f}", label_emb[f].float(), persistent=True)
        self.log_scale = nn.ParameterDict({f: nn.Parameter(torch.tensor(float(np.log(1 / 0.07)))) for f in space.fields})
        self.bias = nn.ParameterDict({f: nn.Parameter(torch.tensor(-5.0 if space.multilabel[f] else 0.0)) for f in space.fields})

    def state_head(self, f):
        return self.state["all" if self.shared_state else f]

    def actions(self, f):
        lab = getattr(self, f"lab_{f}")                                  # [C, K, D]
        if self.training:                                                # one random prompt variant per class
            k = torch.randint(0, lab.shape[1], (lab.shape[0],), device=lab.device)
            x = lab[torch.arange(lab.shape[0], device=lab.device), k]
        else:
            x = lab.mean(1)
        return F.normalize(self.action(x), dim=-1)

    def forward(self, state_emb, fields=None):
        out = {}
        for f in fields or self.space.fields:
            s = F.normalize(self.state_head(f)(state_emb), dim=-1)
            out[f] = self.log_scale[f].exp().clamp(max=100) * s @ self.actions(f).t() + self.bias[f]
        return out

    def infonce_loss(self, state_emb, modes, tgts, S=None, gamma=1.0, cross_field=0.0, cross_gamma=2.0):
        """CLM-style in-batch InfoNCE, with *graded* targets when S (similarity.label_similarity) is given.

        Per field: every labelled row (hard, multi-label set or candidate set) is paired with its label embedding (the
        mean of its labels' action embeddings); target T_ij = label similarity of rows i and j (taxonomy siblings,
        same anatomical family, overlapping label sets get partial credit). S=None gives CLM's binary group-masked loss.
        cross_field > 0 adds an image-image term whose targets average similarity over all fields both rows label."""
        from bio_waila.train.labels import MASKED
        from bio_waila.train.losses import cross_field_infonce, soft_infonce
        from bio_waila.train.similarity import soft_targets
        total, parts = 0.0, {}
        for f in self.space.fields:
            m = modes[f] != MASKED
            if m.sum() < 2:
                continue
            y = tgts[f][m].float()
            s = F.normalize(self.state_head(f)(state_emb[m]), dim=-1)
            a = F.normalize((y / y.sum(-1, keepdim=True)) @ self.actions(f), dim=-1)     # row j's label embedding
            scale = self.log_scale[f].exp().clamp(max=100)
            if S is None:                                                                 # CLM's original loss
                cls = y.argmax(-1)
                keep = y.sum(-1) == 1
                if keep.sum() < 2:
                    continue
                l = group_masked_infonce(s[keep], a[keep], cls[keep], scale)
            else:
                l = soft_infonce(s, a, soft_targets(y, S[f]), scale, gamma=gamma)
            total = total + l
            parts[f] = float(l.detach())
        if cross_field and S is not None:
            joint = F.normalize(torch.stack([F.normalize(self.state_head(f)(state_emb), dim=-1)
                                             for f in self.space.fields]).mean(0), dim=-1)
            l = cross_field_infonce(joint, tgts, modes, S, self.log_scale[self.space.fields[0]].exp().clamp(max=100),
                                    gamma=cross_gamma)
            total = total + cross_field * l
            parts["cross_field"] = float(l.detach())
        return total, parts

    def clm_checkpoint(self, field=None):
        """A dict in CLM's checkpoint format (state_head, action_head, logit_scale, cfg) for one field's state head
        (or the shared one)."""
        f = field or self.space.fields[0]
        return {"state_head": self.state_head(f).state_dict(), "action_head": self.action.state_dict(),
                "logit_scale": self.log_scale[f].detach().cpu(), "cfg": dict(self.cfg), "projection_dim": self.cfg["projection_dim"]}
