"""Check F4 (packed, shared image prefix) against unpacked F2 on the real Gemma-4 processor and chat template.

    uv run python scripts/check_f4_equivalence.py --tiny                         # random tiny weights, real processor
    uv run python scripts/check_f4_equivalence.py --model google/gemma-4-E4B-it --dtype float32 --images DIR

For every image it asks all six Pass-2 fields (first k_max classes of each, plus the abstain letter) both ways,
then reports the largest letter log-prob difference, argmax agreement and the token saving. With --dtype float32
the difference should be float noise (< 1e-3); in bfloat16 expect ~1e-2 from kernel order alone. `--bidirectional`
shows how far the untrained bidirectional mask moves the readout (it should move: that variant needs training).
"""
import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

from bio_waila.models.jev_gemma import ORDER, GemmaBackend, build_prompt


def tiny_from(name):
    """A random 4-layer Gemma 4 with the real vocabulary and special-token ids of `name` (no weights downloaded)."""
    from transformers import AutoConfig, Gemma4Config, Gemma4ForConditionalGeneration
    real = AutoConfig.from_pretrained(name)
    t = dict(vocab_size=real.text_config.vocab_size, vocab_size_per_layer_input=real.text_config.vocab_size,
             hidden_size=32, intermediate_size=64, num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1,
             head_dim=16, global_head_dim=16, hidden_size_per_layer_input=8, num_kv_shared_layers=2, sliding_window=64,
             layer_types=["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
             pad_token_id=real.text_config.pad_token_id, final_logit_softcapping=30.0)
    v = dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
             head_dim=16)
    c = Gemma4Config(text_config=t, vision_config=v, image_token_id=real.image_token_id,
                     boi_token_id=real.boi_token_id, eoi_token_id=real.eoi_token_id)
    c._attn_implementation = "sdpa"
    torch.manual_seed(0)
    return Gemma4ForConditionalGeneration(c).eval()


def demo_images(n):
    return [Image.new("RGB", (96 + 32 * i, 80), (40 * i % 255, 120, 200 - 30 * i % 200)) for i in range(n)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="google/gemma-4-E4B-it")
    ap.add_argument("--tiny", action="store_true", help="random tiny weights; only the processor is downloaded")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--images", default=None, help="directory of images (default: synthetic)")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--k-max", type=int, default=20)
    ap.add_argument("--bidirectional", action="store_true")
    ap.add_argument("--device-map", default=None)
    a = ap.parse_args(argv)

    if a.tiny:
        import transformers
        model = tiny_from(a.model)
        transformers.AutoModelForImageTextToText.from_pretrained = lambda *x, **k: model
    be = GemmaBackend(a.model, dtype=a.dtype, gradient_checkpointing=False, device_map=a.device_map).eval()
    if a.images:
        paths = sorted(p for p in Path(a.images).iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))[: a.n]
        ims = [Image.open(p).convert("RGB") for p in paths]
    else:
        ims = demo_images(a.n)
    from bio_waila.train.labels import LabelSpace
    try:
        space = LabelSpace.load(None)
        fields = [f for f in ORDER if f in space.fields]
        opts = {f: space.classes[f][: a.k_max] for f in fields}
    except Exception:                                           # no plan/label_space.json here: stand-in options
        fields = ORDER
        opts = {f: [f"{f} option {i}" for i in range(6)] for f in fields}
    prompts = [[build_prompt(f, opts[f], abstain=True) for f in fields] for _ in ims]

    with torch.no_grad():
        ref = be.letter_logits([im for im in ims for _ in fields], [p for ps in prompts for p in ps])
        got = be.packed_letter_logits(ims, prompts, bidirectional=a.bidirectional)
    n = [len(opts[f]) + 1 for f in fields] * len(ims)
    diffs, agree = [], 0
    for q, k in enumerate(n):
        r, g = torch.log_softmax(ref[q, :k].float(), -1), torch.log_softmax(got[q, :k].float(), -1)
        diffs.append((r - g).abs().max().item())
        agree += int(r.argmax() == g.argmax())
    from bio_waila.models.jev_packed import pack
    kw = {"max_soft_tokens": be.budget} if be.budget else {}
    unpacked = packed = 0
    for im, ps in zip(ims, prompts):
        x = be.proc(text=[be._chat(p) for p in ps], images=[[im]] * len(ps), padding=True, return_tensors="pt", **kw)
        rows = [r[m.bool()].tolist() for r, m in zip(x["input_ids"], x["attention_mask"])]
        unpacked += sum(map(len, rows))
        packed += len(pack(rows)[0])
    print(f"questions {len(n)}  max |Δ log p| {max(diffs):.2e}  mean {sum(diffs) / len(diffs):.2e}  "
          f"argmax agree {agree}/{len(n)}")
    print(f"tokens: unpacked {unpacked}, packed {packed} ({unpacked / packed:.2f}x fewer)")
    tol = 1e-3 if a.dtype == "float32" else 5e-2
    ok = a.bidirectional or max(diffs) < tol
    print("OK" if ok else f"MISMATCH (tolerance {tol})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
