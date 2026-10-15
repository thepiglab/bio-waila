"""Train the Jev-style teacher (Gemma-4-E4B + LoRA, letter readout).

    uv run waila-train-jev --config configs/jev_gemma4_e4b.yaml --fold 0
    accelerate launch -m bio_waila.train.train_jev --config configs/jev_gemma4_e4b.yaml --fold 0      # multi-GPU
    uv run waila-train-jev --config configs/jev_gemma4_e4b.yaml --set train.formulation=F3        # progressive
    uv run waila-train-jev --config configs/jev_gemma4_e4b.yaml --set train.epochs=0              # stock model only

One training example is (image, field): the fields of each image are expanded into separate prompts (F2 = the
independent formulation; it is mathematically the packed-F2 of experiment_plan.md at higher cost). F3 adds the
earlier fields' gold answers as context with dropout. Evaluation runs every class of every field (tournament for
fields with more than k_max classes) on up to `eval_per_dataset` images per held-out/val dataset, fits per-field
temperatures on val, and reports with the shared metrics.
"""
import argparse
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from bio_waila.data.manifest import load_manifest
from bio_waila.eval.metrics import all_metrics, fit_temperature
from bio_waila.models.jev_gemma import (ORDER, build_prompt, choose_options, field_logprobs, gold_context, letter_loss,
                                    make_backend, option_target)
from bio_waila.paths import PREP
from bio_waila.train.common import JsonlLog, cosine_lr, load_config, save_json, seed_all
from bio_waila.train.folds import load_or_make_folds, split
from bio_waila.train.labels import MASKED, LabelSpace


class JevExamples(torch.utils.data.Dataset):
    """A seed-fixed stream of (row, field) prompts. Rows follow the hierarchical sampler (sampling.py); the field of
    each draw is chosen inversely to how often fields are labelled (stain applies to ~18% of rows, so it is not starved);
    every draw has its own RNG (field, options, context dropout), so the stream is identical on any number of GPUs or
    workers."""

    def __init__(self, df, root, space, cfg, n, seed=0, sampler_cfg=None):
        from bio_waila.train.sampling import epoch_order, sample_weights
        self.df, self.root, self.space, self.cfg, self.n = df.reset_index(drop=True), Path(root), space, cfg, n
        self.enc = [space.encode(r) for r in self.df.to_dict("records")]
        sc = sampler_cfg or {}
        w = sample_weights(self.df, sc.get("mode", "hierarchical"), tau=sc.get("tau", 0.7), power=sc.get("power", 0.5),
                           cap=sc.get("cap", 20000))
        self.rows, self.seeds = epoch_order(w, n, seed, 0)
        avail = {f: max(1e-3, float(np.mean([e[f][0] != MASKED for e in self.enc]))) for f in space.fields}
        self.field_w = {f: 1.0 / a for f, a in avail.items()}
        self.max_side = cfg.get("max_side", 512)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        r = int(self.rows[i])
        rng = random.Random(int(self.seeds[i]))
        e = self.enc[r]
        fields = [f for f in self.space.fields if e[f][0] != MASKED] or [self.space.fields[0]]
        f = rng.choices(fields, weights=[self.field_w[x] for x in fields])[0]
        mode, vec = e[f]
        opts = choose_options(self.space.n(f), vec, mode, k_max=self.cfg.get("k_max", 20), rng=rng, train=True)
        lmode, tgt = option_target(vec, mode, opts)
        ctx = (gold_context(e, self.space, f, self.cfg.get("context_dropout", 0.3), rng)
               if self.cfg.get("formulation", "F2") == "F3" else None)
        prompt = build_prompt(f, [self.space.classes[f][j] for j in opts], ctx)
        try:
            im = Image.open(self.root / self.df.at[r, "image"]).convert("RGB")
            im.thumbnail((self.max_side, self.max_side))
        except Exception:
            im, lmode = Image.new("RGB", (64, 64)), MASKED
        k = self.cfg.get("k_max", 20)
        pad = np.zeros(k, np.float32)
        pad[: len(tgt)] = tgt
        return im, prompt, len(opts), lmode, pad


def collate(b):
    return [x[0] for x in b], [x[1] for x in b], torch.tensor([x[2] for x in b]), torch.tensor([x[3] for x in b]), \
        torch.from_numpy(np.stack([x[4] for x in b]))


def eval_subset(df, per_dataset, seed=0):
    if not len(df):
        return df
    df = df.sample(frac=1.0, random_state=seed)
    return df[df.groupby("dataset_id").cumcount() < per_dataset].reset_index(drop=True)


@torch.no_grad()
def evaluate(backend, df, root, space, cfg, temps=None, bs=8):
    """Full-vocabulary log-probs for every field (F3 uses the model's own earlier predictions as context)."""
    if not len(df):
        return None, None
    backend.eval()
    enc = [space.encode(r) for r in df.to_dict("records")]
    L = {f: [] for f in space.fields}
    for s in range(0, len(df), bs):
        ims = []
        for p in df["image"].iloc[s:s + bs]:
            im = Image.open(Path(root) / p).convert("RGB")
            im.thumbnail((cfg.get("max_side", 512),) * 2)
            ims.append(im)
        ctx = [dict() for _ in ims]
        for f in [f for f in ORDER if f in space.fields]:
            lp = field_logprobs(backend, ims, f, space, ctx if cfg.get("formulation", "F2") == "F3" else None,
                                k_max=cfg.get("k_max", 20))
            L[f].append(lp)
            for b in range(len(ims)):
                ctx[b][f] = space.classes[f][int(lp[b].argmax())]
    L = {f: torch.cat(v) for f, v in L.items()}
    M = {f: torch.tensor([e[f][0] for e in enc]) for f in space.fields}
    T = {f: torch.from_numpy(np.stack([e[f][1] for e in enc])) for f in space.fields}
    return all_metrics(L, M, T, df["dataset_id"].to_numpy(), space, temps), (L, M, T)


def public_eval(backend, space, cfg, temps, out, public=None, limit=None, bs=8):
    """Score the backend on mmbu-public (eval only; every row) and write the report; None if unavailable."""
    from bio_waila.eval import public as pub
    public = public or pub.try_load_public(cfg, limit=limit)
    if public is None:
        return None
    df, root = public
    _, (L, M, T) = evaluate(backend, df, root, space, cfg["train"], temps, bs=bs)
    return pub.evaluate_public(L, M, T, df, space, temps, out, str(out))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args(argv)
    cfg = load_config(a.config, a.set)
    tc, d = cfg["train"], cfg["data"]
    from accelerate import Accelerator
    acc = Accelerator(mixed_precision="no", gradient_accumulation_steps=tc.get("grad_accum", 1))  # weights already bf16
    seed_all(tc.get("seed", 0))
    fold = a.fold if a.fold is not None else d.get("fold", 0)
    out = Path(a.out or cfg.get("out", "runs/jev_e4b")) / f"{tc.get('formulation', 'F2')}_fold{fold}"
    log = JsonlLog(out / "log.jsonl") if acc.is_main_process else (lambda **k: None)

    space = LabelSpace.load(d.get("label_space"))
    df = load_manifest(d.get("manifest") or None, include_nc=d.get("include_nc", False), splits=("train", "val"),
                       exclude_datasets=d.get("exclude_datasets", []))
    root = Path(d.get("image_root") or PREP)
    tr_all, val = df[df["split"] == "train"], df[df["split"] == "val"]
    folds = load_or_make_folds(tr_all, k=d.get("folds_k", 5), path=d.get("folds_path"))
    train, heldout = split(tr_all, folds, fold)
    ev = {"val": eval_subset(val, tc.get("eval_per_dataset", 20)), "heldout": eval_subset(heldout, tc.get("eval_per_dataset", 20))}
    log(event="data", train=len(train), **{k: len(v) for k, v in ev.items()})

    if acc.is_main_process:
        save_json(out / "config.json", cfg)         # lets `waila-eval --run` rebuild the model
    backend = make_backend(cfg["model"])
    trainable = [p for p in backend.parameters() if p.requires_grad]
    if tc.get("epochs", 1) and trainable:
        from bio_waila.train.sampling import RankSlicedBatches
        gb = tc.get("global_batch") or tc.get("batch", 8) * acc.num_processes     # per optimiser micro-step
        n = int(tc.get("examples_per_epoch", 200000)) * tc.get("epochs", 1) // gb * gb
        ds = JevExamples(train, root, space, tc, n, seed=tc.get("seed", 0), sampler_cfg=cfg.get("sampler"))
        loader = torch.utils.data.DataLoader(ds, num_workers=tc.get("workers", 4), collate_fn=collate,
                                             batch_sampler=RankSlicedBatches(n, gb, acc.process_index, acc.num_processes))
        opt = torch.optim.AdamW(trainable, lr=tc.get("lr", 1e-4), weight_decay=tc.get("wd", 0.0))
        backend, opt = acc.prepare(backend, opt)
        log(event="sampler", global_batch=gb, per_gpu=gb // acc.num_processes, grad_accum=tc.get("grad_accum", 1), examples=n)
        total = len(loader) // tc.get("grad_accum", 1)
        step = 0
        backend.train()
        for it, (ims, prompts, nopt, modes, tgts) in enumerate(loader):
            with acc.accumulate(backend):
                lp = acc.unwrap_model(backend).letter_logits(ims, prompts)        # [B, 26]
                k = tgts.shape[1]
                lp = lp[:, :k].masked_fill(torch.arange(k, device=lp.device)[None] >= nopt[:, None].to(lp.device), -1e4)
                lp = torch.log_softmax(lp, -1)
                loss = letter_loss(lp, modes.to(lp.device), tgts.to(lp.device), smoothing=tc.get("label_smoothing", 0.0))
                acc.backward(loss)
                if acc.sync_gradients:
                    for g in opt.param_groups:
                        g["lr"] = tc.get("lr", 1e-4) * cosine_lr(step, total, int(tc.get("warmup_frac", 0.03) * total))
                    acc.clip_grad_norm_(trainable, tc.get("grad_clip", 1.0))
                    step += 1
                opt.step()
                opt.zero_grad(set_to_none=True)
            if it % tc.get("log_every", 20) == 0:
                log(event="train", step=step, loss=float(loss.detach()))
            if tc.get("max_steps") and step >= tc["max_steps"]:
                break
        backend = acc.unwrap_model(backend)
        if acc.is_main_process and hasattr(backend, "model") and hasattr(backend.model, "save_pretrained"):
            backend.model.save_pretrained(out / "adapter")
        elif acc.is_main_process:
            torch.save(backend.state_dict(), out / "backend.pt")

    report, temps = {"fold": fold, "formulation": tc.get("formulation", "F2")}, {}
    if acc.is_main_process:
        res, raw = evaluate(backend, ev["val"], root, space, tc)
        if raw:
            L, M, T = raw
            temps = {f: fit_temperature(L[f], M[f], T[f], space.multilabel[f]) for f in space.fields}
        for name, sdf in ev.items():
            res, _ = evaluate(backend, sdf, root, space, tc, temps)
            if res:
                report[name] = res
                log(event="eval", set=name, mean_acc_macro_ds=res["mean_acc_macro_ds"])
        report["temperatures"] = temps
        try:
            pr = public_eval(backend, space, cfg, temps, out)
        except Exception as e:  # never lose a trained model to the public look
            pr = None
            log(event="public_failed", error=repr(e)[:300])
        if pr is not None:
            report["public"] = pr
            log(event="eval", set="public", C=pr["C"], mean_field_acc=pr["mean_field_acc"])
        save_json(out / "metrics.json", report)
    return report


if __name__ == "__main__":
    main()
