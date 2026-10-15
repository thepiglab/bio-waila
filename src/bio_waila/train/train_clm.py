"""Train the CLM-vision teacher (SigLIP 2 + per-field heads + text-initialised label vectors).

    uv run waila-train-clm --config configs/clm_siglip2.yaml --fold 0
    accelerate launch -m bio_waila.train.train_clm --config configs/clm_siglip2.yaml --fold 0     # multi-GPU
    uv run waila-train-clm --config configs/clm_siglip2.yaml --set train.epochs_b=1 --limit 2000   # smoke run

Stages (docs/experiments/training_plan.md §2.1):
  0  zero-shot evaluation (= the B1-sig baseline) and a copy of the initial weights for WiSE-FT
  A  head warm-up: tower frozen; per-field heads, label vectors and scales trained
  B  full fine-tune with layer-wise LR decay, cosine schedule, EMA, FLYP-style prompt jitter
  C  WiSE-FT sweep and per-field temperature, both chosen on the validation datasets (split == "val");
     the held-out fold of the training datasets is the reported number

Outputs in <out>: log.jsonl, zero_shot.pt, last.pt (EMA weights), final.pt (best WiSE-FT α), metrics.json.
"""
import argparse
from pathlib import Path

import torch

from bio_waila.data.manifest import load_manifest
from bio_waila.eval.metrics import all_metrics, fit_temperature
from bio_waila.models import clm_siglip
from bio_waila.paths import PREP
from bio_waila.train.common import EMA, JsonlLog, cosine_lr, load_config, save_json, seed_all
from bio_waila.train.data import ManifestImages, SeededOrder, collate, image_transform, to_device
from bio_waila.train.losses import cross_field_infonce
from bio_waila.train.sampling import RankSlicedBatches, epoch_order, sample_weights
from bio_waila.train.similarity import label_similarity, smoothing_distribution
from bio_waila.train.folds import load_or_make_folds, split
from bio_waila.train.labels import LabelSpace
from bio_waila.train.losses import multi_field_loss


def datasets_for(cfg, fold, limit=None):
    d = cfg["data"]
    df = load_manifest(d.get("manifest") or None, include_nc=d.get("include_nc", False), splits=("train", "val"),
                       exclude_datasets=d.get("exclude_datasets", []))
    if limit:
        df = df[df.groupby("dataset_id").cumcount() < max(1, limit // df["dataset_id"].nunique())]
    tr_all, val = df[df["split"] == "train"], df[df["split"] == "val"]
    folds = load_or_make_folds(tr_all, k=d.get("folds_k", 5), path=d.get("folds_path"))
    train, heldout = split(tr_all, folds, fold)
    if d.get("train_on_all"):  # final teacher: no held-out fold
        train, heldout = tr_all.reset_index(drop=True), heldout.iloc[:0]
    return train, heldout, val.reset_index(drop=True)


@torch.no_grad()
def predict(model, loader, acc, fields):
    model.eval()
    L, M, T, I = {f: [] for f in fields}, {f: [] for f in fields}, {f: [] for f in fields}, []
    for px, modes, tgts, idx in loader:
        with acc.autocast():
            out = model(px)
        out, modes, tgts, idx = acc.gather_for_metrics((out, modes, tgts, idx))
        for f in fields:
            L[f].append(out[f].float().cpu())
            M[f].append(modes[f].cpu())
            T[f].append(tgts[f].cpu())
        I.append(idx.cpu())
    cat = lambda d: {f: torch.cat(v) for f, v in d.items()}  # noqa: E731
    return cat(L), cat(M), cat(T), torch.cat(I)


def evaluate(model, loader, df, acc, space, temps=None):
    if loader is None or len(df) == 0:
        return None, None
    L, M, T, I = predict(model, loader, acc, space.fields)
    ds = df["dataset_id"].to_numpy()[I.numpy()]
    return all_metrics(L, M, T, ds, space, temps), (L, M, T)


def public_eval(pm, acc, space, cfg, temps, out, public=None, limit=None, bs=None):
    """Score the final model on mmbu-public (eval only) and write the report next to the run; None if unavailable."""
    from bio_waila.eval import public as pub
    public = public or pub.try_load_public(cfg, limit=limit)
    if public is None:
        return None
    df, root = public
    tc, size = cfg["train"], cfg["model"].get("image_size", 384)
    bs = bs or tc.get("batch", 64)
    ldr = acc.prepare(torch.utils.data.DataLoader(ManifestImages(df, root, space, image_transform(size, False)), batch_size=bs,
                                                  num_workers=tc.get("workers", 0), collate_fn=collate))
    L, M, T, I = predict(pm, ldr, acc, space.fields)
    if not acc.is_main_process:
        return None
    return pub.evaluate_public(L, M, T, df.iloc[I.numpy()].reset_index(drop=True), space, temps, out, str(out))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. train.lr=1e-5")
    ap.add_argument("--limit", type=int, default=None, help="rows (spread over datasets) for a smoke run")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.set)
    from accelerate import Accelerator
    tc, mc = cfg["train"], cfg["model"]
    acc = Accelerator(mixed_precision=tc.get("precision", "bf16") if torch.cuda.is_available() else "no")
    seed_all(tc.get("seed", 0))
    fold = args.fold if args.fold is not None else cfg["data"].get("fold", 0)
    out = Path(args.out or cfg.get("out", "runs/clm_siglip2")) / f"fold{fold}"
    log = JsonlLog(out / "log.jsonl") if acc.is_main_process else (lambda **k: None)

    space = LabelSpace.load(cfg["data"].get("label_space"))
    train_df, heldout_df, val_df = datasets_for(cfg, fold, args.limit)
    root = Path(cfg["data"].get("image_root") or PREP)
    size = mc.get("image_size", 384)
    log(event="data", train=len(train_df), heldout=len(heldout_df), val=len(val_df),
        train_datasets=train_df["dataset_id"].nunique(), fold=fold)

    model = clm_siglip.build(mc, space, device=acc.device)
    zero_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if acc.is_main_process:
        out.mkdir(parents=True, exist_ok=True)
        torch.save(zero_state, out / "zero_shot.pt")

    # global batch: the unit of reproducibility (same examples per optimiser step on 1 or 4 GPUs)
    gb = tc.get("global_batch") or tc.get("batch", 256) * acc.num_processes
    assert gb % acc.num_processes == 0, f"global_batch {gb} must be divisible by {acc.num_processes} GPUs"
    bs, workers = gb // acc.num_processes, tc.get("workers", 8)
    mk_eval = lambda df: torch.utils.data.DataLoader(  # noqa: E731
        ManifestImages(df, root, space, image_transform(size, False)), batch_size=bs, num_workers=workers,
        collate_fn=collate) if len(df) else None
    ho_loader, val_loader = mk_eval(heldout_df), mk_eval(val_df)
    ho_loader, val_loader = (acc.prepare(x) if x is not None else None for x in (ho_loader, val_loader))
    epoch_size = (tc.get("epoch_size") or len(train_df)) // gb * gb
    train_base = ManifestImages(train_df, root, space, image_transform(size, True))
    sc = cfg.get("sampler", {})
    weights = sample_weights(train_df, sc.get("mode", "hierarchical"), tau=sc.get("tau", 0.7),
                             power=sc.get("power", 0.5), cap=sc.get("cap", 20000))

    def epoch_loader(epoch):
        """This rank's slice of the seed-fixed global sequence for one epoch (independent of the GPU count)."""
        rows, seeds = epoch_order(weights, epoch_size, tc.get("seed", 0), epoch)
        return torch.utils.data.DataLoader(SeededOrder(train_base, rows, seeds), num_workers=workers, collate_fn=collate,
                                           batch_sampler=RankSlicedBatches(epoch_size, gb, acc.process_index, acc.num_processes),
                                           pin_memory=torch.cuda.is_available())
    S = label_similarity(space, sibling=tc.get("sibling_sim", 0.3), family=tc.get("family_sim", 0.5))
    smooth = ({f: smoothing_distribution(S[f]).to(acc.device) for f in space.fields}
              if tc.get("smoothing_mode") == "hierarchy" else None)
    S_dev = {f: v.to(acc.device) for f, v in S.items()}
    log(event="sampler", mode=sc.get("mode", "hierarchical"), global_batch=gb, per_gpu=bs, epoch_size=epoch_size)

    # stage 0: zero-shot
    m0 = acc.prepare(model)
    for name, ldr, df in (("val", val_loader, val_df), ("heldout", ho_loader, heldout_df)):
        res, _ = evaluate(m0, ldr, df, acc, space)
        if res:
            log(event="eval", stage="zero_shot", set=name, mean_acc_macro_ds=res["mean_acc_macro_ds"],
                **{f"{f}_acc": res[f].get("acc_macro_ds", float("nan")) for f in space.fields})
    model = acc.unwrap_model(m0)

    loss_kw = dict(smoothing=tc.get("label_smoothing", 0.1), gamma_neg=tc.get("gamma_neg", 2.0), smooth_dists=smooth)
    ema = None
    step_global, epoch_global = 0, 0
    steps_all = (tc.get("epochs_a", 1) + tc.get("epochs_b", 8)) * (epoch_size // gb)
    ramp = int(tc.get("cand_ramp_frac", 0.05) * steps_all)                       # candidate-set rows ramp in
    xw = tc.get("cross_field_weight", 0.0)
    for stage, epochs in (("A", tc.get("epochs_a", 1)), ("B", tc.get("epochs_b", 8))):
        if not epochs:
            continue
        model.set_tower_trainable(stage == "B")
        lr, head_lr = (0.0, tc.get("head_lr", 1e-3)) if stage == "A" else (tc.get("lr", 3e-5), tc.get("head_lr", 1e-3))
        groups = model.param_groups(lr, head_lr, llrd=tc.get("llrd", 0.8), wd=tc.get("wd", 0.05))
        groups = [g for g in groups if g["lr"] > 0]
        opt = torch.optim.AdamW(groups, betas=(0.9, 0.999))
        acc.free_memory()
        pm, opt = acc.prepare(model, opt)
        if stage == "B":
            ema = EMA(acc.unwrap_model(pm), decay=tc.get("ema", 0.9998))
        total = epochs * (epoch_size // gb)
        warm = int(tc.get("warmup_frac", 0.05) * total)
        step = 0
        for ep in range(epochs):
            pm.train()
            for batch in epoch_loader(epoch_global):
                px, modes, tgts, _ = to_device(batch, acc.device)
                scale = cosine_lr(step, total, warm)
                for g in opt.param_groups:
                    g["lr"] = g["base_lr"] * scale
                with acc.autocast():
                    logits, emb = pm(px, return_emb=True)
                cw = min(1.0, step_global / ramp) if ramp else 1.0
                loss, parts = multi_field_loss(logits, modes, tgts, space, weights=tc.get("field_weights"),
                                               cand_weight=cw, **loss_kw)
                if xw:  # image-image term with cross-field graded targets (similarity.py)
                    joint = torch.nn.functional.normalize(torch.stack(list(emb.values())).mean(0), dim=-1)
                    lx = cross_field_infonce(joint, tgts, modes, S_dev, tc.get("cross_field_scale", 10.0),
                                             gamma=tc.get("cross_field_gamma", 2.0))
                    loss = loss + xw * lx
                    parts["cross_field"] = float(lx.detach())
                if tc.get("label_pull", 0):
                    loss = loss + tc["label_pull"] * acc.unwrap_model(pm).label_pull()
                acc.backward(loss)
                if tc.get("grad_clip", 1.0):
                    acc.clip_grad_norm_(pm.parameters(), tc.get("grad_clip", 1.0))
                opt.step()
                opt.zero_grad(set_to_none=True)
                if ema is not None:
                    ema.update(acc.unwrap_model(pm))
                if step % tc.get("log_every", 50) == 0:
                    log(event="train", stage=stage, epoch=ep, step=step, lr_scale=scale, loss=float(loss.detach()), **parts)
                step += 1
                step_global += 1
                if tc.get("max_steps") and step >= tc["max_steps"]:
                    break
            epoch_global += 1
            if ema is not None:
                ema.swap(acc.unwrap_model(pm))
            res, _ = evaluate(pm, val_loader, val_df, acc, space) if val_loader is not None else evaluate(pm, ho_loader, heldout_df, acc, space)
            if ema is not None:
                ema.swap(acc.unwrap_model(pm))
            if res:
                log(event="eval", stage=stage, epoch=ep, set="val" if val_loader is not None else "heldout",
                    mean_acc_macro_ds=res["mean_acc_macro_ds"])
        model = acc.unwrap_model(pm)

    # stage C: EMA weights → WiSE-FT α and temperatures on val, report on held-out
    if ema is not None:
        ema.swap(model)
    ft_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if acc.is_main_process:
        torch.save(ft_state, out / "last.pt")
    sel_loader, sel_df = (val_loader, val_df) if val_loader is not None else (ho_loader, heldout_df)
    best = (-1.0, 1.0)
    for a in tc.get("wise_alphas", [0.3, 0.5, 0.7, 1.0]):
        model.load_state_dict(ft_state)
        clm_siglip.wise_ft(model, zero_state, a)
        res, _ = evaluate(acc.prepare(model), sel_loader, sel_df, acc, space)
        model = acc.unwrap_model(model)
        if res:
            log(event="wise_ft", alpha=a, mean_acc_macro_ds=res["mean_acc_macro_ds"])
            if res["mean_acc_macro_ds"] > best[0]:
                best = (res["mean_acc_macro_ds"], a)
    model.load_state_dict(ft_state)
    clm_siglip.wise_ft(model, zero_state, best[1])
    pm = acc.prepare(model)
    temps = {}
    _, raw = evaluate(pm, sel_loader, sel_df, acc, space)
    if raw:
        L, M, T = raw
        temps = {f: fit_temperature(L[f], M[f], T[f], space.multilabel[f]) for f in space.fields}
    report = {"fold": fold, "wise_alpha": best[1], "temperatures": temps}
    for name, ldr, df in (("val", val_loader, val_df), ("heldout", ho_loader, heldout_df)):
        res, _ = evaluate(pm, ldr, df, acc, space, temps)
        if res:
            report[name] = res
            log(event="eval", stage="final", set=name, mean_acc_macro_ds=res["mean_acc_macro_ds"])
    if acc.is_main_process:
        torch.save({"state": acc.unwrap_model(pm).state_dict(), "config": cfg, "temperatures": temps,
                    "wise_alpha": best[1], "fields": space.fields, "classes": space.classes}, out / "final.pt")
    try:
        pr = public_eval(pm, acc, space, cfg, temps, out)
    except Exception as e:  # never lose a trained model to the public look
        pr = None
        log(event="public_failed", error=repr(e)[:300])
    if pr is not None:
        report["public"] = pr
        log(event="eval", stage="final", set="public", C=pr["C"], mean_field_acc=pr["mean_field_acc"])
    if acc.is_main_process:
        save_json(out / "metrics.json", report)
    return report


if __name__ == "__main__":
    main()
