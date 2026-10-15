"""Score a trained run on mmbu-public (eval only): rebuild the model from the run's saved checkpoint and config.

    uv run waila-eval --run runs/clm_siglip2/fold0 [--limit 200] [--out DIR] [--config CFG.yaml]

Writes public_metrics.json and public_predictions.parquet (see eval/public.py) next to the run and logs the look in
plan/public_looks.jsonl. Families (detected from the run directory):
  CLM-vision  final.pt (state + config + temperatures)                       rebuilt from the config's backbone
  CLM-LLM     final.pt (head state + config) + the embedding cache           needs mmbu-public embedded by `embed`
  Jev         adapter/ or backend.pt + config.json (or --config) + metrics.json   reloads the LoRA adapter into the backend
"""
import argparse
import json
from pathlib import Path

import torch

from bio_waila.eval import public as pub
from bio_waila.train.common import load_config
from bio_waila.train.labels import LabelSpace


class NotSupported(RuntimeError):
    pass


def detect(run: Path) -> str:
    if (run / "final.pt").exists():
        ck = torch.load(run / "final.pt", map_location="cpu", weights_only=False)
        return "clm_llm" if "head_cfg" in ck else "clm"
    if (run / "adapter").exists() or (run / "backend.pt").exists():
        return "jev"
    raise NotSupported(f"{run} has no final.pt, adapter/ or backend.pt: not a finished run")


def space_from(cfg, ck=None):
    """The run's label space: the saved classes (the checkpoint's heads are indexed by them), flags from label_space.json."""
    classes = ck["classes"] if ck else None
    try:
        sp = LabelSpace.load(cfg["data"].get("label_space"))
        if classes is None or {f: sp.classes[f] for f in classes} == classes:
            return sp
        flags = sp.multilabel
    except (OSError, KeyError):
        flags = {}
    if classes is None:
        raise NotSupported("label space unavailable")
    return LabelSpace(list(ck["fields"]), classes, {f: flags.get(f, f in ("body_part", "domain")) for f in ck["fields"]})


def eval_clm(run, out, limit, cfg_path):
    from accelerate import Accelerator
    from bio_waila.models import clm_siglip
    from bio_waila.train.train_clm import public_eval
    ck = torch.load(run / "final.pt", map_location="cpu", weights_only=False)
    cfg = load_config(cfg_path) if cfg_path else ck["config"]
    acc = Accelerator(mixed_precision=cfg["train"].get("precision", "bf16") if torch.cuda.is_available() else "no")
    space = space_from(cfg, ck)
    model = clm_siglip.build(cfg["model"], space, device=acc.device)
    model.load_state_dict(ck["state"])
    got = pub.try_load_public({}, limit=limit)
    if got is None:
        raise NotSupported("mmbu-public is not available")
    return public_eval(acc.prepare(model), acc, space, cfg, ck.get("temperatures") or {}, out, public=got)


def eval_clm_llm(run, out, limit, cfg_path):
    from bio_waila.models.clm_llm import CLMHeads
    from bio_waila.train.train_clm_llm import load_cache, load_public_cached, public_eval
    ck = torch.load(run / "final.pt", map_location="cpu", weights_only=False)
    cfg = load_config(cfg_path) if cfg_path else ck["config"]
    space = space_from(cfg, ck)
    try:
        X, pos, lab = load_cache(cfg)
    except (OSError, ValueError) as e:
        raise NotSupported(f"the embedding cache ({cfg.get('cache')}) is not readable: {e}") from e
    hc = cfg.get("heads", {})
    heads = CLMHeads(space, ck["dim"], {f: lab[f].float() for f in space.fields}, width=hc.get("width", 1536),
                     d_out=hc.get("d_out", 512), depth=hc.get("depth", 3), layernorm=hc.get("layernorm", True),
                     residual=hc.get("residual", False), shared_state=hc.get("shared_state", False))
    heads.load_state_dict(ck["state"])
    got = load_public_cached({"data": cfg["data"]}, limit)
    if got is None:
        raise NotSupported("mmbu-public is not available")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return public_eval(heads.to(dev), cfg, space, X, pos, ck.get("temperatures") or {}, out, dev, public=got)


def eval_jev(run, out, limit, cfg_path):
    from bio_waila.models.jev_gemma import make_backend
    from bio_waila.train.train_jev import public_eval
    cp = Path(cfg_path) if cfg_path else run / "config.json"
    if not cp.exists():
        raise NotSupported(f"Jev runs saved before config.json existed need --config (looked for {cp})")
    cfg = load_config(cp) if cp.suffix in (".yaml", ".yml") else json.loads(cp.read_text())
    space = space_from(cfg)
    backend = make_backend(cfg["model"])
    if (run / "adapter").exists():
        from peft import load_peft_weights, set_peft_model_state_dict
        set_peft_model_state_dict(backend.model, load_peft_weights(str(run / "adapter")))
    else:
        backend.load_state_dict(torch.load(run / "backend.pt", map_location="cpu"))
    temps = json.loads((run / "metrics.json").read_text()).get("temperatures", {}) if (run / "metrics.json").exists() else {}
    got = pub.try_load_public({}, limit=limit)
    if got is None:
        raise NotSupported("mmbu-public is not available")
    return public_eval(backend, space, cfg, temps, out, public=got)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run directory (e.g. runs/clm_siglip2/fold0)")
    ap.add_argument("--limit", type=int, default=None, help="random subsample of N public images for a quick check")
    ap.add_argument("--out", default=None, help="report directory (default: the run directory)")
    ap.add_argument("--config", default=None, help="override the config saved with the run")
    a = ap.parse_args(argv)
    run = Path(a.run)
    out = Path(a.out) if a.out else run
    try:
        res = {"clm": eval_clm, "clm_llm": eval_clm_llm, "jev": eval_jev}[detect(run)](run, out, a.limit, a.config)
    except NotSupported as e:
        print(f"not supported yet: {e}")
        return None
    if res is not None:
        print(f"C={res['C']:.4f}  mean_field_acc={res['mean_field_acc']:.4f}  images={res['n_images']}  → {out}/public_metrics.json")
    return res


if __name__ == "__main__":
    main()
