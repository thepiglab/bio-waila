"""CLM-vision: a SigLIP / SigLIP 2 image tower with one attention-pooling head per field and label vectors initialised
from the text tower (a cosine classifier whose weights start as SigLIP's own zero-shot classifier).

    logits_f = scale_f * cos(head_f(tokens), w_f[c]) + bias_f

- head_f starts as a copy of the backbone's MAP pooling head, w_f[c] as the normalised mean text embedding of class
  c's prompts, scale/bias as the backbone's logit_scale/logit_bias. So before training the model *is* SigLIP
  zero-shot, which makes WiSE-FT (interpolating trained and initial weights) exact.
- FLYP-style prompt jitter: in training, w_f[c] is perturbed by (one random prompt embedding - the prompt mean).
- Fixed-resolution SigLIP 2 checkpoints (e.g. google/siglip2-so400m-patch16-384) load as SiglipModel. NaFlex
  checkpoints (Siglip2Model) need patchified inputs with attention masks; not wired yet (see training_plan.md §2.1).
"""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


def load_backbone(cfg: dict):
    """cfg: {name: hf id} or {from_config: {...SiglipConfig kwargs...}} (tests)."""
    import transformers as t
    if cfg.get("from_config"):
        return t.SiglipModel(t.SiglipConfig(**cfg["from_config"]))
    m = t.AutoModel.from_pretrained(cfg["name"], torch_dtype=getattr(torch, cfg.get("dtype", "float32")))
    if type(m).__name__.startswith("Siglip2"):
        raise NotImplementedError("NaFlex Siglip2Model checkpoints are not wired yet; use a fixed-resolution "
                                  "SigLIP 2 checkpoint such as google/siglip2-so400m-patch16-384")
    if cfg.get("vision_override"):
        # another tower of the same shape (e.g. Moondream 3.1's, waila-convert-moondream); the text tower and the MAP
        # head stay SigLIP's, so the model no longer starts exactly at SigLIP zero-shot: give stage A more epochs
        from safetensors.torch import load_file
        from bio_waila.models.moondream_vision import load_into
        load_into(m.vision_model, load_file(cfg["vision_override"]), eps=cfg.get("vision_layernorm_eps", 1e-5))
    return m


@torch.no_grad()
def encode_texts(backbone, tokenizer, texts, batch=256, device="cpu", max_length=64):
    """Normalised text-tower embeddings (SigLIP was trained with padding to max_length=64)."""
    out = []
    backbone = backbone.to(device)
    for i in range(0, len(texts), batch):
        tok = tokenizer(texts[i:i + batch], padding="max_length", truncation=True, max_length=max_length, return_tensors="pt").to(device)
        f = backbone.get_text_features(input_ids=tok["input_ids"])
        f = f.pooler_output if hasattr(f, "pooler_output") else f
        out.append(F.normalize(f.float(), dim=-1).cpu())
    return torch.cat(out)


class SiglipCLM(nn.Module):
    def __init__(self, backbone, space, prompt_emb: dict):
        """prompt_emb: field → tensor [n_classes, K, D] of normalised prompt embeddings."""
        super().__init__()
        self.space = space
        self.trunk = backbone.vision_model
        self.heads = nn.ModuleDict({f: copy.deepcopy(self.trunk.head) for f in space.fields})
        if hasattr(self.trunk, "use_head"):
            self.trunk.use_head = False  # the per-field heads replace it
        self.trunk.head = nn.Identity()
        self.w = nn.ParameterDict()
        for f in space.fields:
            p = prompt_emb[f].float()
            self.register_buffer(f"p_{f}", p, persistent=True)
            w0 = F.normalize(p.mean(1), dim=-1)
            self.register_buffer(f"w0_{f}", w0.clone(), persistent=True)
            self.w[f] = nn.Parameter(w0.clone())
        s0 = float(backbone.logit_scale.detach().exp().clamp(max=100)) if hasattr(backbone, "logit_scale") else 10.0
        b0 = float(backbone.logit_bias.detach()) if getattr(backbone, "logit_bias", None) is not None else -10.0
        self.log_scale = nn.ParameterDict({f: nn.Parameter(torch.tensor(float(torch.log(torch.tensor(s0))))) for f in space.fields})
        self.bias = nn.ParameterDict({f: nn.Parameter(torch.tensor(b0 if space.multilabel[f] else 0.0)) for f in space.fields})
        self.prompt_jitter = True

    # ── forward ──
    def tokens(self, px):
        o = self.trunk(pixel_values=px)
        return o.last_hidden_state

    def label_vectors(self, f):
        w = self.w[f]
        if self.training and self.prompt_jitter:
            p = getattr(self, f"p_{f}")
            k = torch.randint(0, p.shape[1], (p.shape[0],), device=p.device)
            w = w + (p[torch.arange(p.shape[0], device=p.device), k] - p.mean(1))
        return F.normalize(w, dim=-1)

    def forward(self, px, fields=None, return_emb=False):
        h = self.tokens(px)
        out, emb = {}, {}
        for f in fields or self.space.fields:
            q = self.heads[f](h)
            q = F.normalize(q.float(), dim=-1)
            emb[f] = q
            out[f] = self.log_scale[f].exp().clamp(max=100) * q @ self.label_vectors(f).float().t() + self.bias[f]
        return (out, emb) if return_emb else out

    # ── training helpers ──
    def label_pull(self, strength: dict | float = 1.0):
        """L2 pull of label vectors to their text initialisation (stronger for rare labels: pass a per-class weight)."""
        reg = 0.0
        for f in self.space.fields:
            d = (self.w[f] - getattr(self, f"w0_{f}")).pow(2).sum(-1)
            s = strength[f] if isinstance(strength, dict) else strength
            reg = reg + (d * s).sum()
        return reg

    def set_tower_trainable(self, on: bool):
        for p in self.trunk.parameters():
            p.requires_grad = on

    def param_groups(self, lr, head_lr, llrd=0.8, wd=0.05):
        """Layer-wise LR decay over the trunk; heads, label vectors, scales at head_lr. No decay on norms/biases."""
        layers = list(self.trunk.encoder.layers)
        L = len(layers)
        depth = {}
        for n, _ in self.trunk.embeddings.named_parameters():
            depth[f"trunk.embeddings.{n}"] = 0
        for i, layer in enumerate(layers):
            for n, _ in layer.named_parameters():
                depth[f"trunk.encoder.layers.{i}.{n}"] = i + 1
        groups = {}
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            nd = p.ndim < 2 or n.endswith(".bias") or "norm" in n or n.startswith(("log_scale", "bias."))
            if n.startswith("trunk."):
                d = depth.get(n, L + 1)
                g_lr = lr * llrd ** (L + 1 - d)
            else:
                g_lr = head_lr
            key = (round(g_lr, 12), nd)
            groups.setdefault(key, {"params": [], "lr": g_lr, "weight_decay": 0.0 if nd else wd})["params"].append(p)
        for g in groups.values():
            g["base_lr"] = g["lr"]
        return list(groups.values())


def build(cfg: dict, space, device="cpu", encode_fn=None):
    """Backbone + label prompt embeddings → SiglipCLM. encode_fn(texts) → [N, D] lets tests skip the tokenizer."""
    backbone = load_backbone(cfg["backbone"])
    if encode_fn is None and cfg["backbone"].get("from_config"):  # tests: hash characters into token ids, no tokenizer
        V = backbone.config.text_config.vocab_size
        L = backbone.config.text_config.max_position_embeddings

        def encode_fn(texts):
            ids = torch.tensor([[(ord(c) * 31 + i) % V for i, c in enumerate(t[:L].ljust(L))] for t in texts])
            with torch.no_grad():
                f = backbone.get_text_features(input_ids=ids)
            return F.normalize((f.pooler_output if hasattr(f, "pooler_output") else f).float(), dim=-1)
    if encode_fn is None:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(cfg["backbone"]["name"])
        encode_fn = lambda texts: encode_texts(backbone, tok, texts, device=device)  # noqa: E731
    prompt_emb = {}
    for f in space.fields:
        pr = space.prompts(f)
        k = len(pr[0])
        e = encode_fn([t for cls in pr for t in cls])
        prompt_emb[f] = e.view(len(pr), k, -1)
    return SiglipCLM(backbone, space, prompt_emb)


@torch.no_grad()
def wise_ft(model, zero_state: dict, alpha: float):
    """In-place: θ = (1-α) θ_zero-shot + α θ_fine-tuned, for every floating tensor present in both."""
    cur = model.state_dict()
    for k, v in cur.items():
        if k in zero_state and v.is_floating_point():
            v.copy_((1 - alpha) * zero_state[k].to(v.device, v.dtype) + alpha * v)
    return model
