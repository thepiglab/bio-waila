"""Moondream 3.1's vision tower as a SigLIP vision model, for a CLM-vision bake-off arm.

docs/models/moondream31_for_clm_jev.md: the 3.1 tower is SigLIP 2 So400m/14-384 fine-tuned inside the VLM (all 333
vision tensors are BF16; only the LM's MoE experts are FP8). It maps one-to-one onto transformers' SiglipVisionModel
(qkv split by rows, patch linear reshaped to a conv kernel); it has no MAP head and no text tower, so the CLM keeps
SigLIP 2's head and text tower and re-aligns them in the head warm-up stage. Moondream's LayerNorm eps is 1e-5.

    uv run waila-convert-moondream --out $WAILA_WORK/models/moondream31_vision.safetensors
then set `model.backbone.vision_override` to that file (configs/clm_moondream31_vision.yaml).

Only the vision tensors are fetched (HTTP range reads over the safetensors header and slices), never the 9B file.
"""
import argparse
import json
import os
import re
import struct
from pathlib import Path

import requests
import torch

URL = "https://huggingface.co/moondream/moondream3.1-9B-A2B/resolve/main/model.safetensors"
PREFIX = "vision_encoder.encoder.model.visual."
DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}


def _get(url, start, end, headers):
    r = requests.get(url, headers={**headers, "Range": f"bytes={start}-{end}"}, timeout=300, allow_redirects=True)
    r.raise_for_status()
    return r.content


def fetch_vision_tensors(url=URL, prefix=PREFIX):
    """{name without prefix: tensor} for every tensor under `prefix`, via range requests."""
    h = {"Authorization": f"Bearer {os.environ['HF_TOKEN']}"} if os.environ.get("HF_TOKEN") else {}
    n = struct.unpack("<Q", _get(url, 0, 7, h))[0]
    header = json.loads(_get(url, 8, 8 + n - 1, h))
    base = 8 + n
    out = {}
    for name, meta in header.items():
        if name == "__metadata__" or not name.startswith(prefix):
            continue
        a, b = meta["data_offsets"]
        buf = bytearray(_get(url, base + a, base + b - 1, h))
        out[name[len(prefix):]] = torch.frombuffer(buf, dtype=DTYPES[meta["dtype"]]).reshape(meta["shape"]).clone()
    return out


def map_to_siglip(t: dict) -> dict:
    """Moondream visual.* names → SiglipVisionTransformer state-dict keys (no head)."""
    sd = {}
    w = t["patch_embed.linear.weight"]
    d = w.shape[0]
    p = int(round((w.shape[1] / 3) ** 0.5))
    sd["embeddings.patch_embedding.weight"] = w.reshape(d, 3, p, p)
    sd["embeddings.patch_embedding.bias"] = t["patch_embed.linear.bias"]
    sd["embeddings.position_embedding.weight"] = t["pos_embed"][0]
    sd["post_layernorm.weight"], sd["post_layernorm.bias"] = t["norm.weight"], t["norm.bias"]
    blocks = sorted({int(m.group(1)) for k in t if (m := re.match(r"blocks\.(\d+)\.", k))})
    for i in blocks:
        b, o = f"blocks.{i}.", f"encoder.layers.{i}."
        for src, dst in (("norm1", "layer_norm1"), ("norm2", "layer_norm2"), ("attn.proj", "self_attn.out_proj"),
                         ("mlp.fc1", "mlp.fc1"), ("mlp.fc2", "mlp.fc2")):
            for s in ("weight", "bias"):
                sd[f"{o}{dst}.{s}"] = t[f"{b}{src}.{s}"]
        for s in ("weight", "bias"):
            q, k, v = t[f"{b}attn.qkv.{s}"].chunk(3, dim=0)
            sd[f"{o}self_attn.q_proj.{s}"], sd[f"{o}self_attn.k_proj.{s}"], sd[f"{o}self_attn.v_proj.{s}"] = q, k, v
    return sd


def load_into(vision_model, sd: dict, eps=1e-5):
    """Load mapped weights into a SiglipVisionTransformer; everything but the MAP head must match exactly."""
    sd = {k: v.to(next(vision_model.parameters()).dtype) for k, v in sd.items()}
    missing, unexpected = vision_model.load_state_dict(sd, strict=False)
    bad = [k for k in missing if not k.startswith("head.")]
    if unexpected or bad:
        raise ValueError(f"Moondream→SigLIP mapping mismatch: missing {bad[:5]}, unexpected {list(unexpected)[:5]}")
    for m in vision_model.modules():
        if isinstance(m, torch.nn.LayerNorm):
            m.eps = eps
    return vision_model


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--url", default=URL)
    a = ap.parse_args(argv)
    from safetensors.torch import save_file
    t = fetch_vision_tensors(a.url)
    print(f"fetched {len(t)} vision tensors ({sum(v.numel() for v in t.values()) / 1e6:.0f}M params)")
    sd = {k: v.contiguous() for k, v in map_to_siglip(t).items()}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    save_file(sd, a.out, metadata={"source": a.url, "mapping": "bio_waila.models.moondream_vision.map_to_siglip"})
    print(f"wrote {a.out}: {len(sd)} tensors")


if __name__ == "__main__":
    main()
