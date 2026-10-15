"""Offline tests of the training code: label encoding, losses, metrics, folds, and tiny end-to-end runs of the
CLM-vision, CLM-LLM and Jev trainers (random tiny models, synthetic images; no downloads, CPU only)."""
import json

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from bio_waila.train.labels import CAND, HARD, MASKED, MULTI, LabelSpace  # noqa: E402

SPACE = {
    "modality": {"classes": ["x-ray", "microscopy", "endoscopy"], "multilabel": False},
    "submodality": {"classes": ["chest x-ray", "h&e", "colonoscopy", "fundus"], "multilabel": False},
    "body_part": {"classes": ["chest", "lung", "colon", "eye", "skin"], "multilabel": True},
    "stain": {"classes": ["h&e", "ihc"], "multilabel": False},
    "specimen": {"classes": ["human", "mouse"], "multilabel": False},
    "domain": {"classes": ["radiology", "pathology", "gastroenterology"], "multilabel": True},
}
KINDS = [  # (modality, submodality, body parts, stain, domain, colour)
    ("x-ray", "chest x-ray", ["chest", "lung"], None, ["radiology"], (40, 40, 40)),
    ("microscopy", "h&e", ["colon"], "h&e", ["pathology"], (200, 80, 160)),
    ("endoscopy", "colonoscopy", ["colon"], None, ["gastroenterology"], (180, 60, 40)),
]


@pytest.fixture
def world(tmp_path, monkeypatch):
    from PIL import Image
    (tmp_path / "plan").mkdir()
    (tmp_path / "plan" / "label_space.json").write_text(json.dumps(SPACE))
    rows, rng = [], np.random.default_rng(0)
    for d in range(9):
        mod, sub, bps, stain, dom, col = KINDS[d % 3]
        ds = f"ds{d}"
        (tmp_path / "prep" / "images" / ds).mkdir(parents=True)
        for i in range(6):
            a = np.clip(np.array(col) + rng.normal(0, 25, (48, 48, 3)), 0, 255).astype(np.uint8)
            Image.fromarray(a).save(tmp_path / "prep" / "images" / ds / f"{i}.jpg")
            r = {"image": f"images/{ds}/{i}.jpg", "dataset_id": ds, "group": ds, "split": "val" if d >= 7 else "train",
                 "license_group": "permissive"}
            for f, v, m in (("modality", [mod], "hard"), ("submodality", [sub], "hard"), ("body_part", bps, "multi"),
                            ("stain", [stain] if stain else [], "hard" if stain else "masked"),
                            ("specimen", ["human", "mouse"], "candidates"), ("domain", dom, "multi")):
                r[f"{f}_mode"], r[f"{f}_values"] = m, v
            rows.append(r)
    pd.DataFrame(rows).to_parquet(tmp_path / "prep" / "manifest.parquet")
    monkeypatch.setenv("WAILA_PREP", str(tmp_path / "prep"))
    return tmp_path


def tiny_siglip_cfg(tmp_path):
    return {
        "out": str(tmp_path / "runs"),
        "data": {"manifest": str(tmp_path / "prep" / "manifest.parquet"), "image_root": str(tmp_path / "prep"),
                 "label_space": str(tmp_path / "plan" / "label_space.json"), "folds_path": str(tmp_path / "plan" / "folds.json"),
                 "folds_k": 3},
        "model": {"image_size": 32, "backbone": {"from_config": {
            "text_config": dict(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=4,
                                vocab_size=128, max_position_embeddings=16),
            "vision_config": dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                                  image_size=32, patch_size=8)}}},
        "train": {"batch": 8, "workers": 0, "epoch_size": 32, "epochs_a": 1, "epochs_b": 1, "lr": 1e-4, "head_lr": 1e-2,
                  "wise_alphas": [0.5, 1.0], "log_every": 1, "precision": "no"},
    }


def test_label_encoding_modes(world):
    sp = LabelSpace.load(world / "plan" / "label_space.json")
    e = sp.encode({"modality_mode": "hard", "modality_values": ["x-ray"], "submodality_mode": "multi",
                   "submodality_values": ["chest x-ray", "fundus"], "body_part_mode": "multi", "body_part_values": ["chest", "lung"],
                   "stain_mode": "hard", "stain_values": ["unknown-stain"], "specimen_mode": "candidates",
                   "specimen_values": ["human"], "domain_mode": "masked", "domain_values": []})
    assert e["modality"][0] == HARD and e["modality"][1].tolist() == [1, 0, 0]
    assert e["submodality"][0] == CAND        # "multi" in a single-label field = candidate set
    assert e["body_part"][0] == MULTI and e["body_part"][1].sum() == 2
    assert e["stain"][0] == MASKED            # value outside the label space
    assert e["specimen"][0] == HARD           # candidate set of one
    assert e["domain"][0] == MASKED


def test_losses_respect_modes():
    from bio_waila.train.losses import sigmoid_field_loss, softmax_field_loss
    logits = torch.tensor([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 0.0]], requires_grad=True)
    modes = torch.tensor([HARD, CAND, MASKED])
    tgt = torch.tensor([[1.0, 0, 0], [1.0, 1.0, 0], [0, 0, 1.0]])
    l, n = softmax_field_loss(logits, modes, tgt, smoothing=0.0)
    assert n == 2 and l.item() < 0.05      # both rows already right (row 2: top-1 inside the candidate set)
    l.backward()
    assert logits.grad[2].abs().sum() == 0  # masked row: no gradient
    lm, n = sigmoid_field_loss(torch.tensor([[4.0, 4.0, -4.0]]), torch.tensor([MULTI]), torch.tensor([[1.0, 1.0, 0.0]]))
    lw, _ = sigmoid_field_loss(torch.tensor([[-4.0, -4.0, 4.0]]), torch.tensor([MULTI]), torch.tensor([[1.0, 1.0, 0.0]]))
    assert lm < lw


def test_metrics_macro_over_datasets():
    from bio_waila.eval.metrics import field_metrics
    logits = torch.tensor([[3.0, 0.0]] * 9 + [[0.0, 3.0]])
    modes = torch.tensor([HARD] * 10)
    tgt = torch.tensor([[1.0, 0.0]] * 10)
    ds = ["big"] * 9 + ["small"]
    m = field_metrics(logits, modes, tgt, ds, False, "modality")
    assert abs(m["acc"] - 0.9) < 1e-6 and abs(m["acc_macro_ds"] - 0.5) < 1e-6


def test_folds_keep_groups_together(world):
    from bio_waila.train.folds import load_or_make_folds, split
    df = pd.read_parquet(world / "prep" / "manifest.parquet")
    folds = load_or_make_folds(df, k=3, path=world / "plan" / "folds.json")
    tr, ho = split(df, folds, 0)
    assert not set(tr["group"]) & set(ho["group"]) and len(ho) > 0
    assert load_or_make_folds(df, k=3, path=world / "plan" / "folds.json") == folds  # frozen on disk


def test_clm_siglip_zero_shot_equals_backbone(world):
    """Before training, each field head is a copy of SigLIP's pooling head: the CLM's logits are SigLIP zero-shot."""
    import torch.nn.functional as F
    from bio_waila.models import clm_siglip
    cfg = tiny_siglip_cfg(world)
    sp = LabelSpace.load(cfg["data"]["label_space"])
    torch.manual_seed(0)
    m = clm_siglip.build(cfg["model"], sp).eval()
    torch.manual_seed(0)
    bb = clm_siglip.load_backbone(cfg["model"]["backbone"]).eval()
    px = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        img = bb.get_image_features(pixel_values=px).pooler_output
        ref = F.normalize(img, dim=-1) @ m.w["modality"].t() * bb.logit_scale.exp()
        got = m(px)["modality"]
    assert torch.allclose(got, ref, atol=1e-4)


def test_train_clm_end_to_end(world):
    import yaml
    from bio_waila.train import train_clm
    p = world / "clm.yaml"
    p.write_text(yaml.safe_dump(tiny_siglip_cfg(world)))
    rep = train_clm.main(["--config", str(p), "--fold", "0"])
    out = world / "runs" / "fold0"
    assert (out / "final.pt").exists() and (out / "zero_shot.pt").exists() and (out / "metrics.json").exists()
    assert "val" in rep and "heldout" in rep and rep["wise_alpha"] in (0.5, 1.0)
    assert set(rep["temperatures"]) == set(SPACE)


def test_train_clm_llm_embed_and_train(world):
    import yaml
    from bio_waila.train import train_clm_llm
    cfg = {"out": str(world / "runs_llm"), "cache": str(world / "cache_llm"),
           "data": {"manifest": str(world / "prep" / "manifest.parquet"), "image_root": str(world / "prep"),
                    "label_space": str(world / "plan" / "label_space.json"), "folds_path": str(world / "plan" / "folds.json"),
                    "folds_k": 3, "eval_public": str(world / "none.parquet")},
           "encoder": {"kind": "fake"}, "heads": {"width": 64, "d_out": 32},
           "train": {"batch": 16, "epochs": 30, "epoch_size": 64, "lr": 3e-3, "patience": 30}}
    p = world / "llm.yaml"
    p.write_text(yaml.safe_dump(cfg))
    for s in ("0/2", "1/2"):
        train_clm_llm.main(["embed", "--config", str(p), "--shard", s])
    rep = train_clm_llm.main(["train", "--config", str(p), "--fold", "0"])
    assert (world / "runs_llm" / "fold0" / "final.pt").exists()
    # colour separates the three kinds; val datasets (ds7, ds8) are kinds the training folds always contain
    assert rep["val"]["modality"]["acc_macro_ds"] > 0.6


def test_jev_options_prompt_and_loss():
    import random as _r
    from bio_waila.models.jev_gemma import build_prompt, choose_options, letter_loss, option_target
    vec = np.zeros(99, np.float32)
    vec[[5, 42]] = 1
    opts = choose_options(99, vec, MULTI, k_max=20, rng=_r.Random(0), train=True)
    assert len(opts) == 20 and {5, 42} <= set(opts)
    mode, t = option_target(vec, MULTI, opts)
    assert mode == MULTI and abs(t.sum() - 1) < 1e-6
    p = build_prompt("body_part", ["chest", "lung"], {"modality": "x-ray"})
    assert "A. chest" in p and "B. lung" in p and "Known: modality = x-ray." in p
    logp = torch.log_softmax(torch.tensor([[2.0, 0.0, -1e4], [0.0, 2.0, -1e4]]), -1)
    l = letter_loss(logp, torch.tensor([HARD, CAND]), torch.tensor([[1.0, 0, 0], [1.0, 1.0, 0]]), smoothing=0.1)
    assert torch.isfinite(l) and l.item() < 0.5


def test_jev_tournament_covers_every_class():
    from PIL import Image
    from bio_waila.models.jev_gemma import FakeBackend, field_logprobs
    sp = LabelSpace(["body_part"], {"body_part": [f"organ{i}" for i in range(45)]}, {"body_part": True})
    lp = field_logprobs(FakeBackend(), [Image.new("RGB", (8, 8), (200, 10, 10))] * 2, "body_part", sp, k_max=20)
    assert lp.shape == (2, 45) and torch.isfinite(lp).all()
    assert (lp.max(-1).values > -10).all()      # finalists carry real log-probs


def test_letter_token_groups_on_a_tiny_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    from bio_waila.models.jev_gemma import letter_token_groups
    vocab = {"[UNK]": 0, "A": 1, "▁A": 2, "B": 3, "chest": 4, "AB": 5}
    tk = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    g = letter_token_groups(PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="[UNK]"))
    assert 1 in g[0] and 3 in g[1] and 5 not in sum(g.values(), [])


def test_train_jev_fake_backend_end_to_end(world):
    import yaml
    from bio_waila.train import train_jev
    cfg = {"out": str(world / "runs_jev"),
           "data": {"manifest": str(world / "prep" / "manifest.parquet"), "image_root": str(world / "prep"),
                    "label_space": str(world / "plan" / "label_space.json"), "folds_path": str(world / "plan" / "folds.json"),
                    "folds_k": 3},
           "model": {"kind": "fake"},
           "train": {"batch": 8, "workers": 0, "epochs": 1, "examples_per_epoch": 160, "lr": 5e-2, "k_max": 3,
                     "formulation": "F3", "eval_per_dataset": 3, "log_every": 5}}
    p = world / "jev.yaml"
    p.write_text(yaml.safe_dump(cfg))
    rep = train_jev.main(["--config", str(p), "--fold", "0"])
    assert "val" in rep and "heldout" in rep and set(rep["temperatures"]) == set(SPACE)
    assert (world / "runs_jev" / "F3_fold0" / "metrics.json").exists()


def test_clm_heads_match_official_format_and_infonce():
    """Heads follow Contrastive-LM/CLM's make_head; the shared-state checkpoint has CLM's keys; InfoNCE runs."""
    from bio_waila.models.clm_llm import CLMHeads, group_masked_infonce
    sp = LabelSpace(["modality"], {"modality": ["a", "b", "c"]}, {"modality": False})
    h = CLMHeads(sp, 16, {"modality": torch.randn(3, 2, 8)}, width=32, d_out=8, depth=3, shared_state=True)
    ck = h.clm_checkpoint()
    assert set(ck) >= {"state_head", "action_head", "logit_scale", "cfg"}
    assert {"inp.weight", "hidden.0.weight", "norms.0.weight", "out.weight"} <= set(ck["state_head"])
    x = torch.randn(6, 16)
    modes = {"modality": torch.tensor([HARD] * 6)}
    tg = {"modality": torch.eye(3)[[0, 1, 2, 0, 1, 2]]}
    l, parts = h.infonce_loss(x, modes, tg)
    assert torch.isfinite(l) and "modality" in parts
    z = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)
    assert torch.isfinite(group_masked_infonce(z, z, torch.tensor([0, 0, 1, 2]), 10.0))


def test_train_clm_llm_infonce_shared_state(world):
    import yaml
    from bio_waila.train import train_clm_llm
    cfg = {"out": str(world / "runs_llm2"), "cache": str(world / "cache_llm2"),
           "data": {"manifest": str(world / "prep" / "manifest.parquet"), "image_root": str(world / "prep"),
                    "label_space": str(world / "plan" / "label_space.json"), "folds_path": str(world / "plan" / "folds.json"),
                    "folds_k": 3, "eval_public": str(world / "none.parquet")},
           "encoder": {"kind": "fake"}, "heads": {"width": 64, "d_out": 32, "shared_state": True},
           "train": {"batch": 16, "epochs": 5, "epoch_size": 64, "lr": 3e-3, "loss": "infonce"}}
    p = world / "llm2.yaml"
    p.write_text(yaml.safe_dump(cfg))
    train_clm_llm.main(["embed", "--config", str(p)])
    train_clm_llm.main(["train", "--config", str(p), "--fold", "0"])
    assert (world / "runs_llm2" / "fold0" / "clm_heads.pt").exists()


def test_moondream_vision_mapping_loads_strictly():
    """Synthetic Moondream-named tensors map onto a SigLIP vision tower with every non-head key matched."""
    import transformers as t
    from bio_waila.models.moondream_vision import load_into, map_to_siglip
    vc = dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4, image_size=28, patch_size=14)
    vm = t.SiglipVisionModel(t.SiglipVisionConfig(**vc))  # transformers 5: the tower itself (= SiglipModel.vision_model)
    md = {"patch_embed.linear.weight": torch.randn(32, 3 * 14 * 14), "patch_embed.linear.bias": torch.randn(32),
          "pos_embed": torch.randn(1, 4, 32), "norm.weight": torch.randn(32), "norm.bias": torch.randn(32)}
    for i in range(2):
        b = f"blocks.{i}."
        md.update({b + "norm1.weight": torch.randn(32), b + "norm1.bias": torch.randn(32), b + "norm2.weight": torch.randn(32),
                   b + "norm2.bias": torch.randn(32), b + "attn.qkv.weight": torch.randn(96, 32), b + "attn.qkv.bias": torch.randn(96),
                   b + "attn.proj.weight": torch.randn(32, 32), b + "attn.proj.bias": torch.randn(32),
                   b + "mlp.fc1.weight": torch.randn(64, 32), b + "mlp.fc1.bias": torch.randn(64),
                   b + "mlp.fc2.weight": torch.randn(32, 64), b + "mlp.fc2.bias": torch.randn(32)})
    load_into(vm, map_to_siglip(md))
    assert torch.equal(vm.encoder.layers[1].self_attn.k_proj.weight, md["blocks.1.attn.qkv.weight"][32:64])
    assert vm.post_layernorm.eps == 1e-5


# ── batch composition, reproducibility and graded targets ─────────────────────────────────────────────────────────
def test_global_batches_identical_across_gpu_counts():
    from bio_waila.train.sampling import RankSlicedBatches, epoch_order
    w = np.ones(500) / 500
    rows1, seeds1 = epoch_order(w, 96, seed=7, epoch=3)
    rows2, seeds2 = epoch_order(w, 96, seed=7, epoch=3)
    assert (rows1 == rows2).all() and (seeds1 == seeds2).all()
    one = list(RankSlicedBatches(96, 32, 0, 1))
    four = [list(RankSlicedBatches(96, 32, r, 4)) for r in range(4)]
    for b in range(len(one)):  # global batch b: the 4 ranks together hold exactly the 1-GPU batch
        assert sorted(sum((four[r][b] for r in range(4)), [])) == sorted(one[b])


def test_seeded_order_fixes_augmentation(world):
    from bio_waila.train.data import ManifestImages, SeededOrder, image_transform
    sp = LabelSpace.load(world / "plan" / "label_space.json")
    df = pd.read_parquet(world / "prep" / "manifest.parquet")
    base = ManifestImages(df, world / "prep", sp, image_transform(32, True))
    a = SeededOrder(base, [3, 3, 5], [11, 12, 11])
    x0, x1, x2 = a[0][0], a[1][0], a[2][0]
    assert torch.equal(x0, SeededOrder(base, [3], [11])[0][0])     # same draw, same augmentation, any access order
    assert not torch.equal(x0, x1)                                   # different seed, different augmentation


def test_hierarchical_sampler_keeps_class_mix_and_spreads_datasets():
    from bio_waila.train.sampling import sample_weights
    rows = []
    for d, (sub, n) in enumerate([("a", 1000), ("a", 10), ("b", 100)]):
        rows += [{"dataset_id": f"d{d}", "image": f"{d}_{i}", "submodality_mode": "hard", "submodality_values": [sub]} for i in range(n)]
    df = pd.DataFrame(rows)
    u, h = sample_weights(df, "uniform"), sample_weights(df, "hierarchical", tau=1.0)
    share = lambda w, ds: w[df.dataset_id.isin(ds)].sum()  # noqa: E731
    assert abs(share(h, ["d0", "d1"]) - share(u, ["d0", "d1"])) < 1e-9   # class share unchanged
    assert share(h, ["d1"]) > 5 * share(u, ["d1"])                        # small dataset of class a seen far more


def test_graded_label_similarity_and_targets():
    from bio_waila.train.similarity import label_similarity, smoothing_distribution, soft_targets
    sp = LabelSpace(["submodality", "body_part"],
                    {"submodality": ["chest x-ray", "digital radiography", "brightfield microscopy"],
                     "body_part": ["retina", "optic disc", "colon"]}, {"submodality": False, "body_part": True})
    S = label_similarity(sp, sibling=0.3, family=0.5)
    assert S["submodality"][0, 1] == 0.3 and S["submodality"][0, 2] == 0.0      # x-ray siblings vs microscopy
    assert S["body_part"][0, 1] == 0.5 and S["body_part"][0, 2] == 0.0          # eye family vs colon
    y = torch.tensor([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
    T = soft_targets(y, S["submodality"])
    assert abs(T[0, 0] - 1) < 1e-5 and 0 < T[0, 1] < 1 and T[0, 2] == 0
    yb = torch.tensor([[1.0, 1.0, 0], [1.0, 0, 0], [0, 0, 1.0]])                  # {retina, optic disc} vs {retina}
    Tb = soft_targets(yb, S["body_part"])
    assert Tb[0, 1] > 0.8 and Tb[0, 2] == 0
    D = smoothing_distribution(S["submodality"])
    assert torch.allclose(D.sum(-1), torch.ones(3)) and D[0, 1] > D[0, 2] and D[0, 0] == 0


def test_soft_infonce_reduces_to_binary_and_cross_field_partial_credit():
    from bio_waila.models.clm_llm import group_masked_infonce
    from bio_waila.train.losses import cross_field_targets, soft_infonce
    torch.manual_seed(0)
    s = torch.nn.functional.normalize(torch.randn(5, 8), dim=-1)
    a = torch.nn.functional.normalize(torch.randn(5, 8), dim=-1)
    assert torch.allclose(soft_infonce(s, a, torch.eye(5), 10.0), group_masked_infonce(s, a, torch.arange(5), 10.0), atol=1e-5)
    S = {"modality": torch.eye(2), "body_part": torch.eye(2)}
    tg = {"modality": torch.tensor([[1.0, 0], [1.0, 0], [1.0, 0]]), "body_part": torch.tensor([[1.0, 0], [1.0, 0], [0, 1.0]])}
    md = {f: torch.tensor([HARD] * 3) for f in tg}
    T = cross_field_targets(tg, md, S)
    assert T[0, 1] == 1.0 and T[0, 2] == 0.5 and T[0, 0] == 0                    # same modality only → half credit


def test_clm_llm_soft_infonce_with_cross_field(world):
    from bio_waila.models.clm_llm import CLMHeads
    from bio_waila.train.similarity import label_similarity
    sp = LabelSpace.load(world / "plan" / "label_space.json")
    lab = {f: torch.randn(sp.n(f), 2, 16) for f in sp.fields}
    h = CLMHeads(sp, 16, lab, width=32, d_out=8)
    df = pd.read_parquet(world / "prep" / "manifest.parquet").head(12)
    enc = [sp.encode(r) for r in df.to_dict("records")]
    modes = {f: torch.tensor([e[f][0] for e in enc]) for f in sp.fields}
    tg = {f: torch.from_numpy(np.stack([e[f][1] for e in enc])) for f in sp.fields}
    l, parts = h.infonce_loss(torch.randn(12, 16), modes, tg, S=label_similarity(sp), cross_field=0.5)
    assert torch.isfinite(l) and "cross_field" in parts and "body_part" in parts   # multi-label rows now included
    l.backward()
