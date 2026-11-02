"""bio_waila.eval.zoo: registry, Pass-2 prompt/JSON parsing, label mapping, option tournament, leaderboard, and an end-to-end
`run` with fake backends on a tiny fake mmbu-public (offline, CPU, no downloads)."""
import json
import re

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
yaml = pytest.importorskip("yaml")

from conftest import CLASSES  # noqa: E402
from bio_waila.eval import public as pub  # noqa: E402
from bio_waila.eval import zoo  # noqa: E402
from bio_waila.models import jev_gemma  # noqa: E402
from bio_waila.train.labels import LabelSpace  # noqa: E402
from test_public import _tiny_public_manifest, fake_public  # noqa: E402,F401
from test_train import KINDS, SPACE, world  # noqa: E402,F401


def small_space():
    return LabelSpace(list(CLASSES), {f: list(v) for f, v in CLASSES.items()},
                      {f: f in ("body_part", "domain") for f in CLASSES})


# ── registry ─────────────────────────────────────────────────────────────────────────────────────────────────────
def test_default_registry_is_valid():
    es = zoo.load_registry()
    by = {e.slug: e for e in es}
    assert len(by) == len(es) >= 25
    for e in es:
        assert e.gpus == 1 and e.mem_gb in (32, 48, 64, 128) and (e.enabled or e.reason)
        assert not zoo.BANNED.search(e.hf_id)
    assert {e.mode for e in es} == {"generative", "decision"}
    assert by["dec_ajev_gemma4_12b_lora5"].base == "google/gemma-4-12B-it"
    assert by["dec_liquid_d1_3b"].readout == "d1" and by["dec_liquid_d1_omni_600m"].trust_remote_code
    for slug in ("gen_llama32_11b_vision", "gen_phi4_multimodal", "dec_winnow_12b", "dec_rune_26b_a4b",
                 "dec_blink_v0_3_26b_a4b_nvfp4"):
        assert not by[slug].enabled
    assert by["gen_medgemma_27b_it"].enabled and by["dec_quyet_1_0_large"].mem_gb == 128
    # only Gemma and Liquid models run (user, 2026-11-01)
    assert {e.hf_id.split("/")[0] for e in es if e.enabled and not e.backend} == {"google", "LiquidAI"}


def _reg(tmp_path, *models):
    p = tmp_path / "zoo.yaml"
    p.write_text(yaml.safe_dump({"models": list(models)}))
    return p


@pytest.mark.parametrize("bad,msg", [
    ({"slug": "a", "hf_id": "x/y", "mode": "oracle"}, "mode"),
    ({"slug": "a b", "hf_id": "x/y", "mode": "decision"}, "slug"),
    ({"slug": "a", "hf_id": "noorg", "mode": "decision"}, "hf_id"),
    ({"slug": "a", "hf_id": "x/y", "mode": "decision", "enabled": False}, "reason"),
    ({"slug": "a", "hf_id": "Qwen/Qwen3-VL-8B", "mode": "generative"}, "Chinese"),
    ({"slug": "a", "hf_id": "x/lora", "base": "Qwen/Qwen3-8B", "mode": "decision"}, "Chinese"),
    ({"slug": "a", "hf_id": "x/y", "mode": "generative", "base": "x/z"}, "base"),
    ({"slug": "a", "hf_id": "x/y", "mode": "decision", "k_max": 40}, "k_max"),
    ({"slug": "a", "hf_id": "x/y", "mode": "decision", "bogus": 1}, "unknown keys"),
])
def test_registry_rejects(tmp_path, bad, msg):
    with pytest.raises(ValueError, match=msg):
        zoo.load_registry(_reg(tmp_path, bad))


def test_registry_duplicate_slug(tmp_path):
    m = {"slug": "a", "hf_id": "x/y", "mode": "decision"}
    with pytest.raises(ValueError, match="duplicate"):
        zoo.load_registry(_reg(tmp_path, m, dict(m)))


def test_cli_list_porcelain(capsys):
    assert zoo.main(["list", "--porcelain"]) == 0
    rows = [ln.split("\t") for ln in capsys.readouterr().out.splitlines()]
    assert len(rows) == sum(e.enabled and not e.backend for e in zoo.load_registry()) and all(len(r) == 3 for r in rows)
    zoo.main(["list"])
    assert "disabled:" in capsys.readouterr().out


# ── prompt and parsing ───────────────────────────────────────────────────────────────────────────────────────────
def test_prompt_lists_requested_fields_only():
    p = zoo.pass2_prompt(["domain", "modality"])
    assert p.count("modality, medical domain") == 2 and "body part" not in p and "stain" not in p
    assert p.startswith("Look at the image provided and identify the following metadata: modality, medical domain.")
    assert '{"predicted_metadata": {"<field>": "..."}, "perception_rationale": "..."}' in p
    assert "body part, modality, stain, submodality, specimen, medical domain" in zoo.pass2_prompt(zoo.FIELDS)


GOOD = {"predicted_metadata": {"body part": "Chest", "medical domain": "Radiology"}, "perception_rationale": "dark lung fields"}


@pytest.mark.parametrize("text", [
    json.dumps(GOOD),
    "```json\n" + json.dumps(GOOD, indent=2) + "\n```",
    "Sure! Here is the JSON:\n```\n" + json.dumps(GOOD) + "\n```\nHope this helps {really}.",
    json.dumps(GOOD)[:-1].replace('"}}', '",}}') + "}",                    # trailing comma
    json.dumps(GOOD).replace('"', "“", 0) + " trailing text",
])
def test_parse_output_variants(text):
    v, rat, ok = zoo.parse_output(text)
    assert ok and v == {"body_part": "Chest", "domain": "Radiology"} and rat == "dark lung fields"


def test_parse_output_odd_shapes():
    v, _, ok = zoo.parse_output('{"Body_Part": ["Chest", "Lung"], "Medical_Domain": "x", "modality": null}')   # flat, key variants
    assert ok and v == {"body_part": ["Chest", "Lung"], "domain": "x", "modality": None}
    v, _, ok = zoo.parse_output("{'predicted_metadata': {'modality': 'ct'}}")                                     # single quotes
    assert ok and v == {"modality": "ct"}
    v, _, ok = zoo.parse_output('{"noise": {"a": 1}} then {"predicted_metadata": {"stain": "h&e"}}')            # skips non-metadata objects
    assert ok and v == {"stain": "h&e"}
    v, _, ok = zoo.parse_output('{"predicted_metadata": {"modality": "ct scan", "body part": "chest", "stain": "unkn')  # truncated
    assert not ok and v == {"modality": "ct scan", "body_part": "chest"}
    assert zoo.parse_output("I cannot help with that.") == ({}, "", False)
    assert zoo.parse_output("") == ({}, "", False)


# ── mapping onto the label space ─────────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("field,value,cls,how", [
    ("modality", "computed tomography", "computed tomography", "exact"),
    ("modality", "Computed Tomography (CT)", "computed tomography", "exact"),
    ("modality", "CT scan", "computed tomography", "alias"),
    ("modality", "MRI", "magnetic resonance imaging", "alias"),
    ("modality", "X-Ray", "x-ray imaging", "exact"),
    ("modality", "ultrasound", "ultrasound imaging", "exact"),
    ("modality", "Histopathology", "digital pathology", "alias"),
    ("stain", "H&E stain", "h&e", "exact"),
    ("stain", "Hematoxylin and eosin", "h&e", "exact"),  # via the shared canonical normalizer
    ("specimen", "Homo sapiens", "human", "exact"),
    ("body_part", "Lungs", "lung", "exact"),
    ("body_part", "kidny", "kidney", "fuzzy"),
    ("body_part", "Chest, Lung", "chest", "exact"),
    ("body_part", ["Liver", "Kidney"], "liver", "exact"),
    ("body_part", "unknown | brain", "brain", "exact"),
    ("domain", ["Radiology"], "radiology", "exact"),
    ("submodality", "total internal reflection", None, "unmapped"),
    # compound answers: split only when the whole string does not map; the first mappable piece is the answer
    ("body_part", "chest/thorax", "chest", "exact"),
    ("body_part", "Liver or Kidney", "liver", "exact"),
    ("domain", "Urology/Radiology", "urology", "exact"),
    ("modality", "banana/smoothie", None, "unmapped"),
])
def test_mapper(field, value, cls, how):
    sp = small_space()
    c, h = zoo.Mapper(sp).map(field, value)
    assert (c, h) == (cls, how)


def test_mapper_unknown_unmapped_missing():
    m = zoo.Mapper(small_space())
    assert m.map("stain", "unknown") == (None, "unknown") and m.map("stain", "N/A") == (None, "unknown")
    assert m.map("stain", ["unknown", "none of the above"]) == (None, "unknown")
    assert m.map("modality", "banana smoothie") == (None, "unmapped")
    assert m.map("modality", None) == (None, "missing") and m.map("modality", []) == (None, "missing")
    assert m.map("modality", {"a": 1})[0] is None


def test_mapper_map_all_keeps_every_piece_in_order():
    m = zoo.Mapper(small_space())
    assert m.map_all("body_part", "chest/abdomen") == ["chest", "abdomen"]
    assert m.map_all("body_part", ["Liver", "unknown", "Kidney"]) == ["liver", "kidney"]
    assert m.map_all("body_part", "lung / lungs") == ["lung"] and m.map_all("body_part", None) == []
    sp = small_space()
    assert zoo._map_field(m, sp, "body_part", "chest/abdomen") == ([sp.index["body_part"]["chest"], sp.index["body_part"]["abdomen"]], "exact")
    assert zoo._map_field(m, sp, "modality", "MRI/CT") == ([sp.index["modality"]["magnetic resonance imaging"]], "alias")  # single-label: first


def test_onehot_logits_multilabel_answer_keeps_primary_on_top():
    tgt = torch.tensor([[1., 1., 0.]])
    L = zoo.onehot_logits([[1, 0]], tgt, 3)
    assert int(L.argmax(-1)[0]) == 1 and torch.sigmoid(L[0])[0] > 0.99 and torch.sigmoid(L[0])[2] < 0.01


def test_onehot_logits_unmapped_is_never_correct():
    tgt = torch.tensor([[1., 0, 0], [0, 1., 0], [1., 1., 1.]])
    L = zoo.onehot_logits([None, 1, None], tgt, 3)
    top = L.argmax(-1)
    assert tgt[0, top[0]] == 0 and tgt[1, top[1]] == 1       # unmapped row 0 lands on a non-target class
    assert torch.sigmoid(L[0]).max() < 0.01                  # and cannot be read as a multi-label positive
    assert torch.sigmoid(L[1])[1] > 0.99 and torch.sigmoid(L[1])[0] < 0.01


# ── option chunking / tournament ─────────────────────────────────────────────────────────────────────────────────
def _scorer(targets, calls):
    def run(option_sets):
        calls.append(option_sets)
        assert all(len(o) <= 26 for o in option_sets)
        return [torch.log_softmax(torch.tensor([5.0 if i == targets[b] else 0.0 for i in o]), -1)
                for b, o in enumerate(option_sets)]
    return run


@pytest.mark.parametrize("C", [5, 26, 27, 99])
def test_tournament_finds_target_in_any_chunk(C):
    targets = [0, C - 1, C // 2, min(25, C - 1)]
    calls = []
    out = jev_gemma.tournament_logprobs(_scorer(targets, calls), len(targets), C, k_max=26)
    assert out.argmax(-1).tolist() == targets
    assert len(calls) == (1 if C <= 26 else -(-C // 26) + 1)         # one call, or one per chunk plus the final round
    assert torch.isfinite(out).all()
    if C > 26:   # losers of round 1 stay below every finalist
        top = out[1].argmax()
        assert out[1, top] > -1 and (out[1] < out[1, top]).sum() == C - 1


def test_tournament_shrinks_finalists_to_fit():
    # 30 chunks of 2 → 15 chunk winners must fit in k_max=2? they cannot: finalists per chunk shrinks to 1 and set is truncated
    calls = []
    jev_gemma.tournament_logprobs(_scorer([3], calls), 1, 8, k_max=2)
    assert max(len(o) for c in calls for o in c) <= 2


class LetterFake:
    """A 'model' that puts all letter mass on the option whose text is TARGET[b] (prompts are the real Jev prompts)."""

    def __init__(self, targets):
        self.targets = targets
        self.calls = 0

    def letter_logits(self, images, prompts):
        self.calls += 1
        out = torch.full((len(prompts), 26), -20.0)
        for b, p in enumerate(prompts):
            for line in p.split("\n"):
                m = re.match(r"^([A-Z])\. (.*)$", line)
                if m:
                    out[b, jev_gemma.LETTERS.index(m.group(1))] = 5.0 if m.group(2) == self.targets[b] else 0.0
        return torch.log_softmax(out, -1)


def test_field_logprobs_with_many_options():
    names = [f"part {i}" for i in range(99)]
    sp = LabelSpace(["body_part"], {"body_part": names}, {"body_part": True})
    fake = LetterFake(["part 98", "part 3", "part 40"])
    lp = jev_gemma.field_logprobs(fake, [None] * 3, "body_part", sp, k_max=26)
    assert lp.argmax(-1).tolist() == [98, 3, 40] and fake.calls == 5


# ── leaderboard ──────────────────────────────────────────────────────────────────────────────────────────────────
def _fake_metrics(root, slug, C, mode="generative", **z):
    d = root / slug
    d.mkdir(parents=True)
    m = {"C": C, "mean_field_acc": C / 2, "n_images": 10,
         "field_acc": {"modality": {"acc": C, "n": 10}, "stain": {"acc": 0.1, "n": 4}},
         "zoo": {"slug": slug, "mode": mode, "hf_id": f"o/{slug}", "params_b": 4.3, "sec_per_image": 1.5,
                 "unmapped_rate": 0.2 if mode == "generative" else None, **z}}
    (d / "public_metrics.json").write_text(json.dumps(m))


def test_leaderboard_aggregates(tmp_path):
    _fake_metrics(tmp_path, "low", 0.2)
    _fake_metrics(tmp_path, "high", 0.7, mode="decision")
    (tmp_path / "trained").mkdir()
    (tmp_path / "trained" / "public_metrics.json").write_text(json.dumps({"C": 0.99}))      # not a zoo run: ignored
    df = zoo.leaderboard(tmp_path)
    assert list(df.slug) == ["high", "low"] and df.C.iloc[0] == 0.7
    back = pd.read_csv(tmp_path / "leaderboard.csv")
    assert list(back.columns[:8]) == ["slug", "mode", "hf_id", "params_b", "n", "unscored", "C", "mean_field_acc"]
    assert back.acc_modality.tolist() == [0.7, 0.2] and np.isnan(back.acc_body_part[0]) and np.isnan(back.unmapped_rate[0])
    md = (tmp_path / "leaderboard.md").read_text()
    assert md.index("| high ") < md.index("| low ") and "0.700" in md and "–" in md
    assert len(zoo.leaderboard(tmp_path / "empty")) == 0


# ── end to end with fake backends ────────────────────────────────────────────────────────────────────────────────
def _idx(im):
    return int(round(np.asarray(im)[:, :, 0].mean() / 60))        # fake_public paints image i with red = 60 i


class FakeGenerator:
    n_params = 1.5

    def __init__(self):
        self.prompts = []

    def generate(self, images, prompts):
        outs = []
        for im, p in zip(images, prompts):
            self.prompts.append(p)
            i = _idx(im)
            if i == 0:
                outs.append(json.dumps({"predicted_metadata": {"body part": "Chest", "modality": "X-Ray",
                                                               "submodality": "chest x-ray", "medical domain": "Radiology"},
                                        "perception_rationale": "lung fields"}))
            elif i == 1:
                outs.append('```json\n{"predicted_metadata": {"body part": "Colon, Rectum", "modality": "microscopy", '
                            '"stain": "unknown", "submodality": "H&E", "medical domain": ["Pathology"]}, "perception_rationale": "x"}\n```')
            elif i == 2:
                outs.append("I'm sorry, I can't help with that.")
            else:
                outs.append('Here you go: {"predicted_metadata": {"body_part": "lung", "modality": "endoscopy", '
                            '"submodality": "Chest X-ray", "medical_domain": "radiology",}} bye')
        return outs


@pytest.fixture
def zoo_world(world, fake_public, monkeypatch):
    from bio_waila.train import labels
    _tiny_public_manifest(world, fake_public)
    monkeypatch.setattr(labels, "PLAN", world / "plan")
    return world


def entry(slug="fake", mode="generative", **kw):
    return zoo.Entry(slug=slug, hf_id="fake/model", mode=mode, **kw).validate()


def test_run_generative_end_to_end(zoo_world):
    gen = FakeGenerator()
    res = zoo.run_model(entry(), backend=gen, out_root=zoo_world / "zoo", batch=3)
    # only the annotated fields are requested, in the official order
    assert any(p.endswith("Requested metadata fields for this example: body part, modality, stain, submodality, medical domain")
               for p in gen.prompts)
    assert all("specimen" not in p.split("identify the following metadata:")[1].split(".")[0] for p in gen.prompts)
    # img0 4/4, img1 4/5 (stain unknown), img2 0/4 (unparseable), img3 3/4 (wrong modality; trailing comma + key variants)
    assert res["C"] == pytest.approx((1 + 0.8 + 0 + 0.75) / 4)
    z = res["zoo"]
    assert z["mode"] == "generative" and z["params_b"] == 1.5 and z["unmapped_rate"] == pytest.approx(4 / 17)
    assert z["unknown_rate"] == pytest.approx(1 / 17) and z["json_ok_rate"] == pytest.approx(0.75)
    assert z["mapping"]["modality"]["parse_failed"] == 1 and z["mapping"]["stain"]["unknown"] == 1
    out = zoo_world / "zoo" / "fake"
    raw = [json.loads(x) for x in (out / "raw_outputs.jsonl").read_text().splitlines()]
    assert len(raw) == 4 and raw[2]["json_ok"] is False and "sorry" in raw[2]["raw"] and raw[1]["mapped"]["body_part"]["class"] == "colon"
    pr = pd.read_parquet(out / "public_predictions.parquet")
    assert list(pr.modality_correct) == [True, True, False, False] and pr.stain_map[1] == "unknown"
    assert json.loads((out / "public_metrics.json").read_text())["zoo"]["slug"] == "fake"
    looks = [json.loads(x) for x in (pub.PLAN / "public_looks.jsonl").read_text().splitlines()]
    assert looks[-1]["run"] == "zoo:fake" and looks[-1]["C"] == pytest.approx(res["C"])
    df = zoo.leaderboard(zoo_world / "zoo")
    assert df.slug.tolist() == ["fake"] and df.unmapped_rate[0] == pytest.approx(4 / 17)


def test_rescore_follows_a_new_public_revision(zoo_world):
    zoo.run_model(entry("dec", "decision", k_max=12), backend=FakeDecider(), out_root=zoo_world / "zoo")
    df, root = pub.load_public()
    new = df[df.unique_id != "u0"].reset_index(drop=True)              # u0 dropped from the current set
    new.loc[new.unique_id == "u1", "image"] = "images/9.jpg"           # u1's image replaced: its old answer no longer applies
    i3 = int(new.index[new.unique_id == "u3"][0])
    new.at[i3, "modality_values"] = [next(c for c in SPACE["modality"]["classes"] if c != KINDS[0][0])]  # relabelled
    new = pd.concat([new, new[new.unique_id == "u2"].assign(unique_id="u4")], ignore_index=True)        # a row never seen
    got = zoo.rescore(None, zoo_world / "zoo", public=(new, root))
    n3 = sum(new.at[i3, f"{f}_mode"] != "masked" for f in SPACE)
    assert got["dec"][0] == pytest.approx(1.0) and got["dec"][1] == pytest.approx((1 + (n3 - 1) / n3) / 2)
    out = zoo_world / "zoo" / "dec"
    m = json.loads((out / "public_metrics.json").read_text())
    assert m["n_images"] == 2 and m["zoo"]["n_unscored"] == 2 and m["metrics_by_task"] is None
    assert json.loads((out / "public_metrics.before_rescore.json").read_text())["n_images"] == 4
    assert list(pd.read_parquet(out / "public_predictions.parquet").unique_id) == ["u2", "u3"]
    assert zoo.leaderboard(zoo_world / "zoo").unscored.tolist() == [2]


def test_rescore_finds_images_changed_under_the_same_name(zoo_world, tmp_path):
    import shutil
    from PIL import Image
    from bio_waila.data import eval_set
    zoo.run_model(entry("dec", "decision", k_max=12), backend=FakeDecider(), out_root=zoo_world / "zoo")
    df, root = pub.load_public()
    prev = tmp_path / "prev"                                         # the copy the run was scored on
    shutil.copytree(root, prev)
    Image.new("RGB", (32, 32), (9, 9, 9)).save(root / "images" / "2.jpg")   # the new release redraws u2's image
    new = df.assign(image_sha1=[eval_set.image_sha1(root / im) for im in df.image])
    zoo.rescore(None, zoo_world / "zoo", public=(new, root), previous=prev)
    m = json.loads((zoo_world / "zoo" / "dec" / "public_metrics.json").read_text())
    assert m["n_images"] == 3 and m["zoo"]["n_unscored"] == 1
    pr = pd.read_parquet(zoo_world / "zoo" / "dec" / "public_predictions.parquet")
    assert "u2" not in set(pr.unique_id) and pr.image_sha1.notna().all()   # later rescores compare these hashes directly


def test_rescore_remaps_saved_answers(zoo_world):
    res = zoo.run_model(entry(), backend=FakeGenerator(), out_root=zoo_world / "zoo", batch=3)
    out = zoo_world / "zoo" / "fake"
    assert zoo.rescore(None, zoo_world / "zoo") == {"fake": (pytest.approx(res["C"]), pytest.approx(res["C"]))}  # same mapper
    pr = pd.read_parquet(out / "public_predictions.parquet")
    pr.loc[3, "modality_raw"] = json.dumps(f"banana/{pr.modality_target[3][0]}")  # img3's wrong modality, now a compound answer
    pr.to_parquet(out / "public_predictions.parquet", index=False)
    got = zoo.rescore(["fake"], zoo_world / "zoo")
    assert got["fake"][1] == pytest.approx((1 + 0.8 + 0 + 1) / 4)
    m = json.loads((out / "public_metrics.json").read_text())
    assert m["zoo"]["rescored"] and m["zoo"]["mapping"]["modality"]["parse_failed"] == 1 and m["zoo"]["json_ok_rate"] == pytest.approx(0.75)
    pr = pd.read_parquet(out / "public_predictions.parquet")
    assert bool(pr.modality_correct[3]) and pr.modality_map[3] == "exact" and pr.stain_map[1] == "unknown"


class FakeDecider:
    n_params = 0.5

    def __init__(self):
        self.k_maxes = []

    def logprobs(self, images, field, space, k_max=None):
        self.k_maxes.append(k_max)
        L = torch.full((len(images), space.n(field)), -5.0)
        for b, im in enumerate(images):
            mod, sub, bps, stain, dom, _ = KINDS[_idx(im) % 3]
            truth = {"modality": mod, "submodality": sub, "body_part": bps[0], "stain": stain or "h&e",
                     "specimen": "human", "domain": dom[0]}[field]
            L[b, space.index[field][truth]] = -0.1
        return torch.log_softmax(L, -1)


def test_run_decision_end_to_end_and_limit(zoo_world):
    dec = FakeDecider()
    res = zoo.run_model(entry("dec", "decision", k_max=12), backend=dec, out_root=zoo_world / "zoo")
    assert res["C"] == pytest.approx(1.0) and res["zoo"]["unmapped_rate"] is None and set(dec.k_maxes) == {12}
    assert res["zoo"]["mode"] == "decision" and res["n_images"] == 4
    pr = pd.read_parquet(zoo_world / "zoo" / "dec" / "public_predictions.parquet")
    assert (pr.modality_prob > 0.9).all() and not (zoo_world / "zoo" / "dec" / "raw_outputs.jsonl").exists()
    small = zoo.run_model(entry("dec2", "decision"), backend=FakeDecider(), out_root=zoo_world / "zoo", limit=2)
    assert small["n_images"] == 2 and small["zoo"]["limit"] == 2


def test_cli_run_uses_loader(zoo_world, monkeypatch):
    reg = _reg(zoo_world, {"slug": "fake", "hf_id": "fake/model", "mode": "decision"},
               {"slug": "off", "hf_id": "fake/off", "mode": "decision", "enabled": False, "reason": "no"})
    monkeypatch.setattr(zoo, "load_backend", lambda e: FakeDecider())
    monkeypatch.setattr(zoo, "RUNS", zoo_world / "runs")
    assert zoo.main(["--registry", str(reg), "run", "--model", "fake", "--limit", "3"]) == 0
    assert (zoo_world / "runs" / "zoo" / "fake" / "public_metrics.json").exists()
    with pytest.raises(SystemExit, match="disabled"):
        zoo.main(["--registry", str(reg), "run", "--model", "off"])
    with pytest.raises(SystemExit, match="unknown model"):
        zoo.main(["--registry", str(reg), "run", "--model", "nope"])
    assert zoo.main(["--registry", str(reg), "leaderboard"]) == 0


def test_prefetch_reports_failures_and_continues(zoo_world, capsys):
    reg = _reg(zoo_world, {"slug": "a", "hf_id": "g/gated", "mode": "generative"},
               {"slug": "b", "hf_id": "o/lora", "base": "o/base", "mode": "decision"},
               {"slug": "c", "hf_id": "o/base", "mode": "generative"},
               {"slug": "d", "hf_id": "o/off", "mode": "decision", "enabled": False, "reason": "x"})
    seen = []

    def fetch(repo, token):
        seen.append(repo)
        return ("gated", "accept the terms") if repo == "g/gated" else ("ok", "")
    assert zoo.prefetch(None, reg, fetch=fetch) == 0
    out = capsys.readouterr().out
    assert seen == ["g/gated", "o/base", "o/lora"] and "gated" in out and "accept the terms" in out and "2/4 ok" not in out
    assert "3/4 ok" in out and "1 FAILED" in out and "eval set: 4 rows" in out
    assert zoo.prefetch(["a"], reg, strict=True, fetch=fetch) == 1


# ── Slurm chain: lanes and purge ─────────────────────────────────────────────────────────────────────────────────
def test_lanes_group_shared_repos_and_deal_round_robin(tmp_path, capsys):
    reg = _reg(tmp_path, {"slug": "a", "hf_id": "x/one", "mode": "generative"},
               {"slug": "b", "hf_id": "x/one", "mode": "decision"},
               {"slug": "c", "hf_id": "o/lora", "base": "o/base", "mode": "decision"},
               {"slug": "d", "hf_id": "o/base", "mode": "generative"},
               {"slug": "e", "hf_id": "e/e", "mode": "generative"},
               {"slug": "f", "hf_id": "f/f", "mode": "decision"},
               {"slug": "g", "hf_id": "g/g", "mode": "decision", "enabled": False, "reason": "x"},
               {"slug": "q", "hf_id": "q/lora", "base": "e/e", "mode": "decision", "enabled": False, "reason": "x"})
    plan = [(lane, group, e.slug) for lane, group, e in zoo.lanes(None, 2, reg)]
    assert plan == [(0, 0, "a"), (0, 0, "b"), (1, 1, "c"), (1, 1, "d"), (0, 2, "e"), (1, 3, "f")]
    # the given order is the priority order; a model whose repos touch two groups merges them
    assert [(lane, group, e.slug) for lane, group, e in zoo.lanes(["e", "b", "a"], 4, reg)] == [(0, 0, "e"), (1, 1, "b"), (1, 1, "a")]
    (tmp_path / "two").mkdir()
    reg2 = _reg(tmp_path / "two", {"slug": "p", "hf_id": "p/p", "mode": "generative"},
                {"slug": "r", "hf_id": "r/r", "mode": "generative"}, {"slug": "s", "hf_id": "r/r", "base": "p/p", "mode": "decision"})
    assert {(lane, group) for lane, group, _ in zoo.lanes(None, 4, reg2)} == {(0, 0)}
    with pytest.raises(SystemExit, match="disabled"):
        zoo.lanes(["a", "g"], 2, reg)
    with pytest.raises(SystemExit, match="unknown"):
        zoo.lanes(["nope"], 2, reg)
    assert zoo.main(["--registry", str(reg), "lanes", "--lanes", "2", "--models", "f,a"]) == 0
    assert capsys.readouterr().out.splitlines() == ["0\t0\tf\t1\t64", "1\t1\ta\t1\t64"]


def _cached(cache, repo, payload):
    """A minimal Hugging Face cache entry: blob, refs/main and a snapshot symlink."""
    import hashlib
    root = cache / f"models--{repo.replace('/', '--')}"
    commit, blob = hashlib.sha1(repo.encode()).hexdigest(), hashlib.sha1(repo.encode() + payload).hexdigest()
    (root / "blobs").mkdir(parents=True)
    (root / "blobs" / blob).write_bytes(payload)
    (root / "refs").mkdir()
    (root / "refs" / "main").write_text(commit)
    (root / "snapshots" / commit).mkdir(parents=True)
    (root / "snapshots" / commit / "model.safetensors").symlink_to(f"../../blobs/{blob}")
    return root


def test_purge_deletes_only_those_models(tmp_path, capsys):
    reg = _reg(tmp_path, {"slug": "a", "hf_id": "x/one", "mode": "generative"},
               {"slug": "b", "hf_id": "o/lora", "base": "o/base", "mode": "decision"},
               {"slug": "c", "hf_id": "k/keep", "mode": "generative"})
    cache = tmp_path / "hub"
    gone = [_cached(cache, r, b"w" * 4096) for r in ("x/one", "o/lora", "o/base")]
    kept = _cached(cache, "k/keep", b"k" * 4096)
    assert zoo.purge(["a", "b"], reg, cache_dir=cache) == 3 * 4096
    assert not any(p.exists() for p in gone) and (kept / "snapshots").exists()
    assert "freed" in capsys.readouterr().out
    assert zoo.purge(["a"], reg, cache_dir=cache) == 0  # nothing left to delete
    with pytest.raises(SystemExit, match="unknown"):
        zoo.purge(["nope"], reg, cache_dir=cache)


# ── System One backend (respanai/span-02) ────────────────────────────────────────────────────────────────────────
import base64  # noqa: E402
import io  # noqa: E402
import threading  # noqa: E402
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer  # noqa: E402

from PIL import Image  # noqa: E402

from bio_waila.models import zoo_models  # noqa: E402


class _FakeSystemOne:
    """In-process System One server: answers every choice question from the image colour (as FakeDecider), records requests."""

    def __init__(self, fail_first=0, bad_field=None, health=200, api_key=None):
        self.requests, self.auth, self.fail_first, self.bad_field, self.health, self.api_key = [], [], fail_first, bad_field, health, api_key
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, obj):
                b = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)

            def do_GET(self):
                self._send(outer.health, {"ok": True}) if self.path == "/health" else self._send(404, {})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.auth.append(self.headers.get("Authorization"))
                if self.path != "/v1/systemone":
                    return self._send(404, {"error": {"type": "not_found", "message": "no"}})
                outer.requests.append(body)
                if outer.fail_first > 0:
                    outer.fail_first -= 1
                    return self._send(529, {"error": {"type": "overloaded", "message": "busy"}})
                if outer.bad_field:
                    return self._send(400, {"error": {"type": "invalid_request", "message": "bad question"}})
                im = Image.open(io.BytesIO(base64.b64decode(body["images"][0].split(",", 1)[1])))
                i = _idx(im)
                mod, sub, bps, stain, dom, _ = KINDS[i % 3]
                truth = {"modality": mod, "submodality": sub, "body_part": bps[0], "stain": stain or "h&e",
                         "specimen": "human", "domain": dom[0]}
                answers = {}
                for f, q in body["questions"].items():
                    keys = list(q["criteria"])
                    omit = [k for k in keys if k != truth[f]][-1]       # one wrong class is omitted: must get the floor
                    probs = {k: 0.01 for k in keys if k != omit}
                    probs[truth[f]] = 0.9
                    answers[f] = {"type": "choice", "choice": max(probs, key=probs.get), "confidence": 0.9, "probabilities": probs}
                self._send(200, {"answers": answers, "usage": {}})

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def close(self):
        self.srv.shutdown()


@pytest.fixture
def fake_s1():
    made = []

    def make(**kw):
        made.append(_FakeSystemOne(**kw))
        return made[-1]
    yield make
    for m in made:
        m.close()


def _s1(srv, **kw):
    return zoo_models.SystemOneScorer({"base_url": srv.url, "backoff": 0, "timeout": 10, **kw})


def _img(i):
    return Image.new("RGB", (32, 32), (60 * i, 10, 10))


def test_systemone_request_and_logits(zoo_world, fake_s1, monkeypatch):
    srv = fake_s1()
    monkeypatch.setenv("SPAN_API_KEY", "secret")
    sc = _s1(srv)
    space = LabelSpace.load()
    rows = sc.score_rows([_img(0)], [["modality", "body_part"]], space)
    assert len(srv.requests) == 1                                   # one request carries both fields
    body = srv.requests[0]
    assert set(body["questions"]) == {"modality", "body_part"} and len(body["images"]) == 1
    assert body["images"][0].startswith("data:image/jpeg;base64,") and body["reasoning"] == {"mode": "auto"}
    assert body["state"] == {"task": "biomedical image metadata", "image": "Image 1"}
    q = body["questions"]["modality"]
    assert q["type"] == "choice" and "imaging modality of Image 1" in q["instructions"]
    assert list(q["criteria"]) == space.classes["modality"] and len(q["criteria"]) <= 679
    assert srv.auth == ["Bearer secret"]
    lp = rows[0]["modality"]
    assert lp.shape == (space.n("modality"),) and lp.exp().sum().item() == pytest.approx(1.0)
    top = int(lp.argmax())
    assert space.classes["modality"][top] == KINDS[0][0]
    assert lp.min() < lp.max() - 8 and torch.isfinite(lp).all()    # the omitted class sits at the floor
    one = sc.logprobs([_img(0), _img(1)], "domain", space)
    assert one.shape == (2, space.n("domain"))


def test_systemone_no_auth_header_without_key(zoo_world, fake_s1, monkeypatch):
    monkeypatch.delenv("SPAN_API_KEY", raising=False)
    srv = fake_s1()
    _s1(srv).logprobs([_img(0)], "modality", LabelSpace.load())
    assert srv.auth == [None]


def test_systemone_retries_529_then_succeeds(zoo_world, fake_s1):
    srv = fake_s1(fail_first=2)
    out = _s1(srv).logprobs([_img(0)], "modality", LabelSpace.load())
    assert len(srv.requests) == 3 and out.shape[0] == 1
    srv = fake_s1(fail_first=99)
    with pytest.raises(zoo_models.SystemOneError, match="529 overloaded"):
        _s1(srv, retries=2).logprobs([_img(0)], "modality", LabelSpace.load())
    assert len(srv.requests) == 3


def test_systemone_error_body_is_not_retried(zoo_world, fake_s1):
    srv = fake_s1(bad_field=True)
    with pytest.raises(zoo_models.SystemOneError, match="bad question"):
        _s1(srv).logprobs([_img(0)], "modality", LabelSpace.load())
    assert len(srv.requests) == 1


def test_systemone_health(fake_s1):
    with pytest.raises(zoo_models.SystemOneError, match="HTTP 503"):
        _s1(fake_s1(health=503))
    _s1(fake_s1(health=404))                                          # hosted endpoints may not serve /health
    with pytest.raises(zoo_models.SystemOneError, match="not reachable"):
        zoo_models.SystemOneScorer({"base_url": "http://127.0.0.1:1", "timeout": 2})
    with pytest.raises(ValueError, match="unknown systemone options"):
        zoo_models.SystemOneScorer({"bogus": 1}, check=False)


def test_systemone_registry_entry():
    e = {x.slug: x for x in zoo.load_registry()}["dec_span_02"]
    assert e.backend == "systemone" and not e.enabled and e.hf_id == "respanai/span-02" and "sm_120" in e.notes
    with pytest.raises(ValueError, match="systemone"):
        zoo.Entry(slug="a", hf_id="x/y", mode="generative", backend="systemone").validate()
    with pytest.raises(ValueError, match="options"):
        zoo.Entry(slug="a", hf_id="x/y", mode="decision", options={"a": 1}).validate()


def test_run_systemone_end_to_end_via_cli(zoo_world, fake_s1, monkeypatch, capsys):
    srv = fake_s1()
    reg = _reg(zoo_world, {"slug": "span", "hf_id": "respanai/span-02", "mode": "decision", "backend": "systemone",
                           "enabled": False, "reason": "needs a server", "options": {"base_url": "http://127.0.0.1:1"}})
    monkeypatch.setattr(zoo, "RUNS", zoo_world / "runs")
    with pytest.raises(SystemExit, match="disabled"):
        zoo.main(["--registry", str(reg), "run", "--model", "span"])
    plain = zoo_world / "plain.yaml"
    plain.write_text(yaml.safe_dump({"models": [{"slug": "x", "hf_id": "a/b", "mode": "decision"}]}))
    with pytest.raises(SystemExit, match="only applies"):
        zoo.main(["--registry", str(plain), "run", "--model", "x", "--base-url", srv.url])
    assert zoo.main(["--registry", str(reg), "run", "--model", "span", "--base-url", srv.url]) == 0
    out = zoo_world / "runs" / "zoo" / "span"
    m = json.loads((out / "public_metrics.json").read_text())
    assert m["C"] == pytest.approx(1.0) and m["zoo"]["backend"] == "systemone" and m["zoo"]["unmapped_rate"] is None
    assert len(srv.requests) == 4                                      # one request per image, all its annotated fields
    assert all(1 <= len(r["questions"]) <= 6 for r in srv.requests)
    pr = pd.read_parquet(out / "public_predictions.parquet")
    assert (pr.modality_prob > 0.5).all()
    assert zoo.main(["--registry", str(reg), "run", "--model", "span", "--enable", "--base-url", srv.url, "--limit", "2",
                     "--out", str(zoo_world / "o2")]) == 0
    capsys.readouterr()
    zoo.main(["--registry", str(reg), "list", "--porcelain"])
    assert capsys.readouterr().out == ""                               # backend entries are never in the Slurm loop


def test_prefetch_systemone_downloads_whole_repo(zoo_world):
    reg = _reg(zoo_world, {"slug": "span", "hf_id": "respanai/span-02", "mode": "decision", "backend": "systemone",
                           "enabled": False, "reason": "x"},
               {"slug": "g", "hf_id": "o/g", "mode": "generative"})
    calls = {}

    def fetch(repo, token, **kw):
        calls[repo] = kw
        return "ok", ""
    assert zoo.prefetch(["span", "g"], reg, fetch=fetch) == 0
    assert calls == {"respanai/span-02": {"ignore": []}, "o/g": {}}
    calls.clear()
    zoo.prefetch(None, reg, fetch=fetch)
    assert list(calls) == ["o/g"]                                      # disabled: only fetched when named


class FakeD1Model:
    """Stands in for a d1 checkpoint: `system_one_batch` over (None, {field: choice question}, [image]) requests."""

    def __init__(self):
        self.requests = []

    def system_one_batch(self, reqs):
        self.requests += reqs
        out = []
        for _, qs, (im,) in reqs:
            mod, sub, bps, stain, dom, _ = KINDS[_idx(im) % 3]
            truth = {"modality": mod, "submodality": sub, "body_part": bps[0], "stain": stain or "h&e",
                     "specimen": "human", "domain": dom[0]}
            ans = {}
            for name, q in qs.items():
                f = {v: k for k, v in jev_gemma.QUESTION.items()}[q["instructions"]]
                names = list(q["criteria"])
                pr = {n: (0.9 if n == truth[f] else 0.1 / max(1, len(names) - 1)) for n in names}
                ans[name] = {"choice": truth[f], "confidence": 0.9, "probabilities": pr}
            out.append({"answers": ans})
        return out


def _d1(model):
    from bio_waila.models.zoo_models import D1Scorer
    sc = D1Scorer.__new__(D1Scorer)
    sc.model, sc.n_params = model, 1.9
    return sc


def test_d1_asks_all_fields_in_one_request_with_every_class(zoo_world):
    m = FakeD1Model()
    res = zoo.run_model(entry("d1", "decision", readout="d1"), backend=_d1(m), out_root=zoo_world / "zoo")
    assert res["C"] == pytest.approx(1.0) and res["zoo"]["k_max"] is None
    assert len(m.requests) == res["n_images"]                       # one request per image, every annotated field in it
    space = LabelSpace.load()
    for _, qs, _ in m.requests:
        assert len(qs) > 1 and all(len(q["criteria"]) == space.n(f) for f, q in qs.items())   # no tournament


def test_d1_with_k_max_falls_back_to_tournaments(zoo_world):
    m = FakeD1Model()
    zoo.run_model(entry("d1k", "decision", readout="d1", k_max=12), backend=_d1(m), out_root=zoo_world / "zoo")
    assert all(len(qs) == 1 and len(next(iter(qs.values()))["criteria"]) <= 12 for _, qs, _ in m.requests)
