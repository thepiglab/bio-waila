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
from test_train import KINDS, world  # noqa: E402,F401


def small_space():
    return LabelSpace(list(CLASSES), {f: list(v) for f, v in CLASSES.items()},
                      {f: f in ("body_part", "domain") for f in CLASSES})


# ── registry ─────────────────────────────────────────────────────────────────────────────────────────────────────
def test_default_registry_is_valid():
    es = zoo.load_registry()
    by = {e.slug: e for e in es}
    assert len(by) == len(es) >= 25
    for e in es:
        assert e.gpus == 1 and e.mem_gb in (64, 128) and (e.enabled or e.reason)
        assert not zoo.BANNED.search(e.hf_id)
    assert {e.mode for e in es} == {"generative", "decision"}
    assert by["dec_ajev_gemma4_12b_lora5"].base == "google/gemma-4-12B-it"
    assert by["dec_liquid_d1_3b"].readout == "d1" and by["dec_liquid_d1_omni_600m"].trust_remote_code
    for slug in ("gen_llama32_11b_vision", "gen_phi4_multimodal", "dec_winnow_12b", "dec_rune_26b_a4b",
                 "dec_blink_v0_3_26b_a4b_nvfp4"):
        assert not by[slug].enabled
    assert by["gen_medgemma_27b_it"].enabled and by["dec_quyet_1_0_large"].mem_gb == 128


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
    assert len(rows) == sum(e.enabled for e in zoo.load_registry()) and all(len(r) == 3 for r in rows)
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
    assert list(back.columns[:7]) == ["slug", "mode", "hf_id", "params_b", "n", "C", "mean_field_acc"]
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
