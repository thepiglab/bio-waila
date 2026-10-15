"""CLM-LLM teacher, in two steps (docs/models/clm_methods.md; training_plan.md §2.1).

  1. embed:  run the frozen MLLM once over every manifest image (and mmbu-public, for evaluation) and over the label
             texts; write float16 shards. GPU-bound; shard it across GPUs/jobs.
       uv run waila-train-clm-llm embed --config configs/clm_llm_gemma4.yaml --shard 0/4
  2. train:  train the state/action heads on the cached embeddings (minutes on one GPU, also fine on CPU).
       uv run waila-train-clm-llm train --config configs/clm_llm_gemma4.yaml --fold 0

Cache layout (<cache>): images_<i>of<n>.npy + images_<i>of<n>.parquet (row → manifest image path), labels.pt.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from bio_waila.data.manifest import load_manifest
from bio_waila.eval.metrics import all_metrics, fit_temperature
from bio_waila.models.clm_llm import CLMHeads, label_texts, make_encoder
from bio_waila.paths import PLAN, PREP, PUBLIC
from bio_waila.train.common import JsonlLog, load_config, save_json, seed_all
from bio_waila.train.folds import load_or_make_folds, split
from bio_waila.train.labels import LabelSpace
from bio_waila.train.losses import multi_field_loss


def _cache_dir(cfg):
    return Path(cfg.get("cache") or "runs/clm_llm_cache")


def cmd_embed(cfg, shard, limit=None):
    from PIL import Image
    i, n = map(int, shard.split("/"))
    cache = _cache_dir(cfg)
    cache.mkdir(parents=True, exist_ok=True)
    enc = make_encoder(cfg["encoder"])
    space = LabelSpace.load(cfg["data"].get("label_space"))
    if i == 0 and not (cache / "labels.pt").exists():
        lab = {}
        for f in space.fields:
            texts = label_texts(space, f)
            k = len(texts[0])
            lab[f] = torch.from_numpy(enc.embed_texts([t for cls in texts for t in cls])).view(len(texts), k, -1).half()
        torch.save(lab, cache / "labels.pt")
    d = cfg["data"]
    df = load_manifest(d.get("manifest") or None, include_nc=True, splits=("train", "val"))[["image"]]
    df["root"] = str(d.get("image_root") or PREP)
    pub = Path(d.get("eval_public") or PLAN / "eval_public.parquet")
    if pub.exists():
        p = pd.read_parquet(pub)[["image"]]
        p["root"] = str(d.get("public_root") or PUBLIC)
        df = pd.concat([df, p], ignore_index=True)
    df = df.iloc[i::n].reset_index(drop=True)
    if limit:
        df = df.head(limit)
    bs = cfg.get("embed_batch", 16)
    embs = []
    for s in range(0, len(df), bs):
        ims = []
        for r in df.iloc[s:s + bs].itertuples():
            try:
                ims.append(Image.open(Path(r.root) / r.image).convert("RGB"))
            except Exception:
                ims.append(Image.new("RGB", (64, 64)))
        embs.append(enc.embed_images(ims).astype(np.float16))
        if s // bs % 50 == 0:
            print(f"shard {shard}: {s + len(ims)}/{len(df)}", flush=True)
    np.save(cache / f"images_{i}of{n}.npy", np.concatenate(embs) if embs else np.zeros((0, 1), np.float16))
    df[["image", "root"]].to_parquet(cache / f"images_{i}of{n}.parquet")


def load_cache(cfg):
    cache = _cache_dir(cfg)
    X, keys = [], []
    for p in sorted(cache.glob("images_*of*.npy")):
        X.append(np.load(p))
        keys.append(pd.read_parquet(p.with_suffix(".parquet")))
    keys = pd.concat(keys, ignore_index=True)
    keys["k"] = keys["root"] + "|" + keys["image"]
    return np.concatenate(X), dict(zip(keys["k"], range(len(keys)))), torch.load(cache / "labels.pt")


def tensors(df, root, X, pos, space):
    idx = [pos.get(f"{root}|{im}", -1) for im in df["image"]]
    keep = np.array(idx) >= 0
    df = df[keep].reset_index(drop=True)
    x = torch.from_numpy(X[np.array(idx)[keep]].astype(np.float32))
    enc = [space.encode(r) for r in df.to_dict("records")]
    modes = {f: torch.tensor([e[f][0] for e in enc]) for f in space.fields}
    tgts = {f: torch.from_numpy(np.stack([e[f][1] for e in enc])) if len(enc) else torch.zeros(0, space.n(f)) for f in space.fields}
    return df, x, modes, tgts


@torch.no_grad()
def predict(heads, x, bs=8192):
    heads.eval()
    outs = [heads(x[i:i + bs]) for i in range(0, len(x), bs)]
    return {f: torch.cat([o[f] for o in outs]) for f in heads.space.fields} if outs else {}


def cmd_train(cfg, fold):
    tc, d = cfg["train"], cfg["data"]
    seed_all(tc.get("seed", 0))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(cfg.get("out", "runs/clm_llm")) / f"fold{fold}"
    log = JsonlLog(out / "log.jsonl")
    space = LabelSpace.load(d.get("label_space"))
    X, pos, lab = load_cache(cfg)
    root = str(d.get("image_root") or PREP)
    df = load_manifest(d.get("manifest") or None, include_nc=d.get("include_nc", False), splits=("train", "val"),
                       exclude_datasets=d.get("exclude_datasets", []))
    tr_all, val = df[df["split"] == "train"], df[df["split"] == "val"].reset_index(drop=True)
    folds = load_or_make_folds(tr_all, k=d.get("folds_k", 5), path=d.get("folds_path"))
    train, heldout = split(tr_all, folds, fold)
    train, xtr, mtr, ttr = tensors(train, root, X, pos, space)
    sets = {name: tensors(s, root, X, pos, space) for name, s in (("val", val), ("heldout", heldout)) if len(s)}
    log(event="data", train=len(train), **{k: len(v[0]) for k, v in sets.items()}, dim=int(X.shape[1]))

    hc = cfg.get("heads", {})
    heads = CLMHeads(space, X.shape[1], {f: lab[f].float() for f in space.fields}, width=hc.get("width", 1536),
                     d_out=hc.get("d_out", 512), depth=hc.get("depth", 3), layernorm=hc.get("layernorm", True),
                     residual=hc.get("residual", False), shared_state=hc.get("shared_state", False)).to(dev)
    opt = torch.optim.AdamW(heads.parameters(), lr=tc.get("lr", 5e-4), weight_decay=tc.get("wd", 0.0))
    from bio_waila.train.sampling import epoch_order, sample_weights
    from bio_waila.train.similarity import label_similarity, smoothing_distribution
    sc = cfg.get("sampler", {})
    w = sample_weights(train, sc.get("mode", "hierarchical"), tau=sc.get("tau", 0.7), power=sc.get("power", 0.5),
                       cap=sc.get("cap", 20000))
    S = label_similarity(space, sibling=tc.get("sibling_sim", 0.3), family=tc.get("family_sim", 0.5))
    smooth = {f: smoothing_distribution(S[f]).to(dev) for f in space.fields} if tc.get("smoothing_mode") == "hierarchy" else None
    bs, epochs = tc.get("global_batch", tc.get("batch", 2048)), tc.get("epochs", 20)
    per_epoch = max(1, (tc.get("epoch_size") or len(train)) // bs)
    total = epochs * per_epoch
    ramp = int(tc.get("cand_ramp_frac", 0.05) * total)
    # CLM's schedule: OneCycle (10% warm-up, cosine decay), early stopping on the selection set (patience)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=tc.get("lr", 5e-4), total_steps=total, pct_start=0.1,
                                                anneal_strategy="cos")
    loss_kind = tc.get("loss", "softce")         # softce: softmax over each field's whole vocabulary (mode-aware)
    best, best_state, bad = -1.0, None, 0        # infonce: CLM's group-masked in-batch InfoNCE (+ softce for the rest)
    step = 0
    for ep in range(epochs):
        heads.train()
        order, _ = epoch_order(w, per_epoch * bs, tc.get("seed", 0), ep)        # reproducible sample stream
        for it in range(per_epoch):
            b = torch.from_numpy(order[it * bs:(it + 1) * bs])
            xb = xtr[b].to(dev)
            mb, tb = {f: mtr[f][b].to(dev) for f in space.fields}, {f: ttr[f][b].to(dev) for f in space.fields}
            cw = min(1.0, step / ramp) if ramp else 1.0                          # candidate-set rows ramp in
            loss, parts = multi_field_loss(heads(xb), mb, tb, space, smoothing=tc.get("label_smoothing", 0.1),
                                           gamma_neg=tc.get("gamma_neg", 2.0), smooth_dists=smooth, cand_weight=cw)
            if loss_kind == "infonce":
                l2, p2 = heads.infonce_loss(xb, mb, tb, S=S if tc.get("soft_targets", True) else None,
                                            gamma=tc.get("target_gamma", 1.0), cross_field=tc.get("cross_field_weight", 0.0),
                                            cross_gamma=tc.get("cross_field_gamma", 2.0))
                loss = tc.get("infonce_weight", 1.0) * l2 + tc.get("softce_weight", 0.5) * loss
                parts = {**{f"{k}_ce": v for k, v in parts.items()}, **{f"{k}_nce": v for k, v in p2.items()}}
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
        sel = sets.get("val") or sets.get("heldout")
        if sel:
            sdf, sx, sm, st = sel
            res = all_metrics({k: v.cpu() for k, v in predict(heads, sx.to(dev)).items()}, sm, st, sdf["dataset_id"].to_numpy(), space)
            log(event="eval", epoch=ep, loss=float(loss.detach()), mean_acc_macro_ds=res["mean_acc_macro_ds"], **parts)
            if res["mean_acc_macro_ds"] > best:
                best, bad = res["mean_acc_macro_ds"], 0
                best_state = {k: v.detach().clone() for k, v in heads.state_dict().items()}
            else:
                bad += 1
                if bad >= tc.get("patience", 5):
                    log(event="early_stop", epoch=ep, best=best)
                    break
    if best_state is not None:
        heads.load_state_dict(best_state)
    temps, report = {}, {"fold": fold}
    sel = sets.get("val") or sets.get("heldout")
    if sel:
        L = {k: v.cpu() for k, v in predict(heads, sel[1].to(dev)).items()}
        temps = {f: fit_temperature(L[f], sel[2][f], sel[3][f], space.multilabel[f]) for f in space.fields}
    for name, (sdf, sx, sm, st) in sets.items():
        L = {k: v.cpu() for k, v in predict(heads, sx.to(dev)).items()}
        report[name] = all_metrics(L, sm, st, sdf["dataset_id"].to_numpy(), space, temps)
        log(event="eval", stage="final", set=name, mean_acc_macro_ds=report[name]["mean_acc_macro_ds"])
    report["temperatures"] = temps
    torch.save({"state": heads.state_dict(), "config": cfg, "temperatures": temps, "fields": space.fields,
                "classes": space.classes, "dim": int(X.shape[1]), "head_cfg": heads.cfg}, out / "final.pt")
    if heads.shared_state:  # also in CLM's own format (loadable by clm.heads.HeadPair)
        torch.save(heads.clm_checkpoint(), out / "clm_heads.pt")
    save_json(out / "metrics.json", report)
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["embed", "train"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args(argv)
    cfg = load_config(a.config, a.set)
    if a.cmd == "embed":
        return cmd_embed(cfg, a.shard, a.limit)
    return cmd_train(cfg, a.fold if a.fold is not None else cfg["data"].get("fold", 0))


if __name__ == "__main__":
    main()
