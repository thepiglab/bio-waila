#!/usr/bin/env python3
"""Auto-labelling layer 3: model-based pseudo-labels on the closed vocabulary (runs on GPU, on the cluster).

Layers 1–2 (prepare.py) leave some fields as `candidates` (dataset lists several values) or `masked` (no label).
This script resolves as many as it can *without* lowering label quality:

  scores   Zero-shot CLIP-style scoring of every image against every class name of every head, with
           BiomedCLIP (Microsoft, MIT) and/or SigLIP2 (Google, Apache-2.0). Saved as per-head probability matrices.
  vlm      Jev-style option scoring for the rows zero-shot cannot settle: a VLM (default Gemma-4-E4B-it, ungated,
           Apache-2.0) reads the image once per option and we compare the log-likelihood of each option's text.
           Options are the dataset's candidate set, or zero-shot's top-k for masked fields. This is a decision model
           over a closed option list: no free generation, no parsing.
  calibrate  Per head, pick the confidence threshold whose precision is ≥ --precision on registry-hard rows of the
           held-out *validation datasets*. mmbu-public (plan/eval_public.parquet) accuracy is reported, never fit on.
  apply    Accept a pseudo-label only when
             • candidates: the top option is inside the candidate set, clears the threshold, and the VLM (if run)
               agrees → mode hard, source zeroshot / vlm / zeroshot+vlm
             • masked:     both zero-shot and VLM agree and both clear their thresholds (stricter: no registry prior)
           Stain is only filled for microscopy/pathology images; domain is never pseudo-labelled (it describes the
           source dataset, not something visible). Registry-hard labels are never changed.
  Later rounds: train the Pass-2 classifier, write its probabilities in the same .npz format, and pass them with
  --extra-scores to add it as a third voter (self-training).

  uv run waila-autolabel scores    --prep $PREP --encoders biomedclip,siglip2
  uv run waila-autolabel scores    --prep $PREP --encoders biomedclip,siglip2 --eval     # same for mmbu-public
  uv run waila-autolabel calibrate --prep $PREP
  uv run waila-autolabel vlm       --prep $PREP --max-rows 200000
  uv run waila-autolabel apply     --prep $PREP          # → $PREP/manifest_autolabel.parquet + autolabel_report.md
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from bio_waila.data.vocab import FIELDS, STAINED_MODALITIES

from bio_waila.paths import PLAN, PREP
HEADS = [f for f in FIELDS if f != "domain"]
PROMPTS = {
    "modality": ["a {} image", "a biomedical image acquired with {}"],
    "submodality": ["a {} image", "a medical image: {}"],
    "body_part": ["a medical image of the {}", "an image showing the {}"],
    "stain": ["a microscopy image stained with {}", "{} stained cells or tissue"],
    "specimen": ["a biomedical image of a {} specimen", "an image of {} cells or tissue"],
}
QUESTIONS = {
    "modality": "What imaging modality produced this image?",
    "submodality": "What imaging submodality is this?",
    "body_part": "What body part or tissue does this image show?",
    "stain": "What stain was used for this image?",
    "specimen": "What specimen or organism is imaged?",
}
ENCODERS = {"biomedclip": "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
            "siglip2": "google/siglip2-so400m-patch14-384"}


def space():
    return json.loads((PLAN / "label_space.json").read_text())


def load_rows(prep: Path, eval_: bool):
    if eval_:
        from bio_waila.paths import PUBLIC
        df = pd.read_parquet(PLAN / "eval_public.parquet")
        df["path"] = df["image"].map(lambda p: str(PUBLIC / p))
    else:
        df = pd.read_parquet(prep / "manifest.parquet")
        df["path"] = df["image"].map(lambda p: str(prep / p))
    return df.reset_index(drop=True)


# ── zero-shot scores ────────────────────────────────────────────────────────────────────────────────────────────
class Encoder:
    def __init__(self, name, device):
        import torch
        self.name, self.device, self.torch = name, device, torch
        if name == "biomedclip":
            import open_clip
            self.model, self.preprocess = open_clip.create_model_from_pretrained(ENCODERS[name])
            self.tok = open_clip.get_tokenizer(ENCODERS[name])
            self.model.to(device).eval()
        else:
            from transformers import AutoModel, AutoProcessor
            self.model = AutoModel.from_pretrained(ENCODERS[name], torch_dtype=torch.float16).to(device).eval()
            self.proc = AutoProcessor.from_pretrained(ENCODERS[name])

    def text(self, texts):
        t = self.torch
        with t.no_grad():
            if self.name == "biomedclip":
                z = self.model.encode_text(self.tok(texts, context_length=256).to(self.device))
            else:
                inp = self.proc(text=texts, padding="max_length", max_length=64, return_tensors="pt").to(self.device)
                z = self.model.get_text_features(**inp)
        return t.nn.functional.normalize(z.float(), dim=-1)

    def images(self, pil):
        t = self.torch
        with t.no_grad():
            if self.name == "biomedclip":
                x = t.stack([self.preprocess(im) for im in pil]).to(self.device)
                z = self.model.encode_image(x)
            else:
                inp = self.proc(images=pil, return_tensors="pt").to(self.device)
                z = self.model.get_image_features(pixel_values=inp["pixel_values"].half())
        return t.nn.functional.normalize(z.float(), dim=-1)

    def scale(self):
        m = self.model
        return float(m.logit_scale.exp()) if hasattr(m, "logit_scale") else 100.0


def cmd_scores(args):
    import torch
    from PIL import Image
    sp = space()
    df = load_rows(Path(args.prep), args.eval)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.prep) / "_autolabel"
    out.mkdir(parents=True, exist_ok=True)
    for name in args.encoders.split(","):
        enc = Encoder(name, dev)
        T = {}
        for f in HEADS:  # class embedding = mean of prompt-template embeddings
            cls = sp[f]["classes"]
            embs = torch.stack([enc.text([p.format(c) for c in cls]) for p in PROMPTS[f]]).mean(0)
            T[f] = torch.nn.functional.normalize(embs, dim=-1)
        probs = {f: np.zeros((len(df), len(sp[f]["classes"])), dtype=np.float16) for f in HEADS}
        for s in range(0, len(df), args.batch):
            ims = []
            for p in df.path[s:s + args.batch]:
                try:
                    ims.append(Image.open(p).convert("RGB"))
                except Exception:
                    ims.append(Image.new("RGB", (224, 224)))
            z = enc.images(ims)
            for f in HEADS:
                logits = enc.scale() * z @ T[f].T
                probs[f][s:s + len(ims)] = logits.softmax(-1).cpu().numpy().astype(np.float16)
            if s // args.batch % 50 == 0:
                print(f"{name}: {s + len(ims):,}/{len(df):,}", flush=True)
        tag = "eval" if args.eval else "train"
        np.savez_compressed(out / f"zs_{name}_{tag}.npz", **probs)
        print("saved", out / f"zs_{name}_{tag}.npz")


def ensemble(prep: Path, tag, extra=None):
    """Average log-probs of every saved zero-shot encoder (+ optional extra voter) → per-head probs."""
    files = sorted((prep / "_autolabel").glob(f"zs_*_{tag}.npz")) + ([Path(extra)] if extra else [])
    if not files:
        raise SystemExit(f"no zero-shot scores for {tag}; run `scores` first")
    zs = [np.load(f) for f in files]
    out = {}
    for f in HEADS:
        lp = np.mean([np.log(z[f].astype(np.float32) + 1e-6) for z in zs], 0)
        e = np.exp(lp - lp.max(1, keepdims=True))
        out[f] = e / e.sum(1, keepdims=True)
    return out


# ── calibration ─────────────────────────────────────────────────────────────────────────────────────────────────
def restricted_top(p, cls, allowed):
    idx = [cls.index(v) for v in allowed if v in cls]
    if not idx:
        return None, 0.0
    q = p[idx] / max(p[idx].sum(), 1e-9)
    j = int(q.argmax())
    return cls[idx[j]], float(q[j])


def cmd_calibrate(args):
    prep = Path(args.prep)
    sp = space()
    rows = []
    # thresholds are fit on held-out *validation datasets* (registry-hard labels); mmbu-public is reported, never fit on
    for tag, df in (("eval", load_rows(prep, True)), ("train", load_rows(prep, False))):
        P = ensemble(prep, tag)
        if tag == "train":
            keep = (df["split"] == "val").values
        for f in HEADS:
            cls = sp[f]["classes"]
            m = df[f"{f}_mode"].isin(["hard", "multi"]) & ((df[f"{f}_source"].isin(["registry", "mmbu"])))
            if tag == "train":
                m = m & keep
            for i in np.where(m)[0][:200_000]:
                gold = set(df.at[i, f"{f}_values"])
                pred = cls[int(P[f][i].argmax())]
                rows.append((f, tag, float(P[f][i].max()), pred in gold))
    cal = pd.DataFrame(rows, columns=["field", "tag", "conf", "ok"])
    th, rep = {}, ["| Head | rows | top-1 acc | threshold (precision ≥ %.2f) | coverage at threshold | acc on mmbu-public |" % args.precision,
                   "|---|---:|---:|---:|---:|---:|"]
    for f in HEADS:
        c_all = cal[cal.field == f]
        c = c_all[c_all.tag == "train"].sort_values("conf", ascending=False)  # fit on validation datasets only
        if c.empty:
            continue
        prec = c.ok.cumsum() / np.arange(1, len(c) + 1)
        okk = np.where(prec.values >= args.precision)[0]
        t = float(c.conf.values[okk[-1]]) if len(okk) else 1.01
        th[f] = t
        ev = c_all[c_all.tag == "eval"]
        rep.append(f"| {f} | {len(c):,} | {c.ok.mean():.3f} | {t:.3f} | {(c.conf >= t).mean():.1%} | "
                   f"{ev.ok.mean() if len(ev) else float('nan'):.3f} |")
    (prep / "_autolabel" / "thresholds.json").write_text(json.dumps(th, indent=1))
    (prep / "_autolabel" / "calibration.md").write_text("\n".join(rep) + "\n")
    print("\n".join(rep))


# ── VLM option scoring (Jev-style) ──────────────────────────────────────────────────────────────────────────────
def cmd_vlm(args):
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor
    prep = Path(args.prep)
    sp = space()
    df = load_rows(prep, False)
    P = ensemble(prep, "train")
    proc = AutoProcessor.from_pretrained(args.vlm)
    model = AutoModelForImageTextToText.from_pretrained(args.vlm, torch_dtype=torch.bfloat16, device_map="auto").eval()
    out_p = prep / "_autolabel" / "vlm.jsonl"
    done = {json.loads(l)["i"] for l in open(out_p)} if out_p.exists() else set()
    todo = [i for i in range(len(df)) if i not in done and any(df.at[i, f"{f}_mode"] in ("candidates", "masked") for f in HEADS)]
    todo = todo[: args.max_rows]

    def option_scores(img, question, options):
        """Mean log-prob of each option's tokens after the question: one forward pass per option batch."""
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question + " Answer with the label only."}]}]
        prompt = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        texts = [prompt + o for o in options]
        enc = proc(images=[img] * len(options), text=texts, return_tensors="pt", padding=True).to(model.device)
        n_prompt = len(proc(images=[img], text=[prompt], return_tensors="pt")["input_ids"][0])
        with torch.no_grad():
            logits = model(**enc).logits.float()
        lp = logits[:, :-1].log_softmax(-1).gather(-1, enc["input_ids"][:, 1:, None])[..., 0]
        mask = enc["attention_mask"][:, 1:].clone()
        mask[:, : n_prompt - 1] = 0  # score only the option tokens (assumes right padding)
        return ((lp * mask).sum(1) / mask.sum(1).clamp(min=1)).cpu().numpy()

    with open(out_p, "a") as f:
        for k, i in enumerate(todo):
            img = Image.open(df.at[i, "path"]).convert("RGB")
            res = {"i": int(i)}
            for h in HEADS:
                mode = df.at[i, f"{h}_mode"]
                if mode not in ("candidates", "masked"):
                    continue
                if h == "stain" and not set(df.at[i, "modality_values"]) & STAINED_MODALITIES:
                    continue
                cls = sp[h]["classes"]
                opts = list(df.at[i, f"{h}_values"]) if mode == "candidates" else [cls[j] for j in np.argsort(-P[h][i])[: args.topk]]
                if len(opts) < 2:
                    continue
                s = option_scores(img, QUESTIONS[h], opts)
                q = np.exp(s - s.max())
                q /= q.sum()
                res[h] = {"options": opts, "probs": [round(float(x), 4) for x in q]}
            f.write(json.dumps(res) + "\n")
            if k % 200 == 0:
                print(f"vlm {k:,}/{len(todo):,}", flush=True)


# ── apply ───────────────────────────────────────────────────────────────────────────────────────────────────────
def cmd_apply(args):
    prep = Path(args.prep)
    sp = space()
    df = load_rows(prep, False)
    P = ensemble(prep, "train", args.extra_scores)
    th = json.loads((prep / "_autolabel" / "thresholds.json").read_text())
    vlm = {}
    vp = prep / "_autolabel" / "vlm.jsonl"
    if vp.exists():
        for line in open(vp):
            j = json.loads(line)
            vlm[j["i"]] = j
    gained = {h: {"candidates": 0, "masked": 0} for h in HEADS}
    for h in HEADS:
        cls = sp[h]["classes"]
        t = th.get(h, 1.01)
        for col in ("mode", "values", "source", "conf"):
            df[f"{h}_{col}_orig"] = df[f"{h}_{col}"]
        for i in np.where(df[f"{h}_mode"].isin(["candidates", "masked"]))[0]:
            mode = df.at[i, f"{h}_mode"]
            if h == "stain" and not set(df.at[i, "modality_values"]) & STAINED_MODALITIES:
                continue
            allowed = list(df.at[i, f"{h}_values"]) if mode == "candidates" else cls
            zs_v, zs_c = restricted_top(P[h][i], cls, allowed)
            v = vlm.get(int(i), {}).get(h)
            vl_v, vl_c = (v["options"][int(np.argmax(v["probs"]))], max(v["probs"])) if v else (None, 0.0)
            new = None
            if mode == "candidates":
                if zs_v and zs_c >= t and (vl_v is None or vl_v == zs_v):
                    new = (zs_v, "zeroshot+vlm" if vl_v else "zeroshot", zs_c)
                elif vl_v and vl_c >= args.vlm_threshold and (zs_v == vl_v or zs_c < 0.5):
                    new = (vl_v, "vlm", vl_c)
            elif zs_v and vl_v == zs_v and zs_c >= t and vl_c >= args.vlm_threshold:
                new = (zs_v, "zeroshot+vlm", min(zs_c, vl_c))
            if new:
                df.at[i, f"{h}_mode"], df.at[i, f"{h}_values"] = "hard", [new[0]]
                df.at[i, f"{h}_source"], df.at[i, f"{h}_conf"] = new[1], round(float(new[2]), 4)
                gained[h][mode] += 1
    df.drop(columns=["path"]).to_parquet(prep / "manifest_autolabel.parquet", index=False)
    L = ["# Auto-label layer 3 report\n", f"{len(df):,} images. Thresholds from calibration.md (precision ≥ {args.precision}).\n",
         "| Head | hard before | hard after | resolved from candidates | filled from masked |", "|---|---:|---:|---:|---:|"]
    for h in HEADS:
        b, a = (df[f"{h}_mode_orig"] == "hard").mean(), (df[f"{h}_mode"] == "hard").mean()
        L.append(f"| {h} | {b:.1%} | {a:.1%} | {gained[h]['candidates']:,} | {gained[h]['masked']:,} |")
    (prep / "autolabel_report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["scores", "calibrate", "vlm", "apply"])
    ap.add_argument("--prep", required=True, help="prepare.py output dir")
    ap.add_argument("--encoders", default="biomedclip,siglip2")
    ap.add_argument("--eval", action="store_true", help="scores: score mmbu-public instead of the training manifest")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--precision", type=float, default=0.95)
    ap.add_argument("--vlm", default="google/gemma-4-E4B-it")
    ap.add_argument("--vlm-threshold", type=float, default=0.8)
    ap.add_argument("--topk", type=int, default=6)
    ap.add_argument("--max-rows", type=int, default=10**9)
    ap.add_argument("--extra-scores", default=None, help="npz of classifier probs (same format) for self-training rounds")
    args = ap.parse_args()
    {"scores": cmd_scores, "calibrate": cmd_calibrate, "vlm": cmd_vlm, "apply": cmd_apply}[args.cmd](args)


if __name__ == "__main__":
    main()
