"""Backends for the model zoo (eval/zoo.py): off-the-shelf models used zero-shot, GPU only.

  HFGenerator   any AutoModelForImageTextToText model: image + prompt → greedy-decoded text   (mode `generative`)
  LetterScorer  lettered-option readout via GemmaBackend (stock models, or a LoRA adapter on a base)   (mode `decision`)
  D1Scorer      Liquid d1 decision models (`model.system_one`, one `choice` question per field)       (mode `decision`)
  SystemOneScorer  HTTP client for a TypeSafe System One server (respanai/span-02): one request per image, all fields  (mode `decision`)

Decision scorers share `logprobs(images, field, space, k_max) -> [B, n_classes]`; classes beyond k_max options per call go
through the two-round tournament in jev_gemma.tournament_logprobs. Every class here has `n_params` (billions).
"""
import base64
import io
import time
from concurrent.futures import ThreadPoolExecutor

import torch

from bio_waila.models.jev_gemma import QUESTION, GemmaBackend, field_logprobs, tournament_logprobs


def _n_params(model) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e9


class HFGenerator:
    def __init__(self, hf_id, dtype="bfloat16", trust_remote_code=False, max_new_tokens=384):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.proc = AutoProcessor.from_pretrained(hf_id, trust_remote_code=trust_remote_code)
        tok = getattr(self.proc, "tokenizer", None)
        if tok is not None:
            tok.padding_side = "left"
        self.model = AutoModelForImageTextToText.from_pretrained(
            hf_id, dtype=getattr(torch, dtype), device_map="auto", trust_remote_code=trust_remote_code).eval()
        self.max_new_tokens = max_new_tokens
        self.n_params = _n_params(self.model)

    def _chat(self, prompt):
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        try:
            return self.proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        except TypeError:
            return self.proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)

    @torch.no_grad()
    def _generate(self, images, prompts):
        inputs = self.proc(text=[self._chat(p) for p in prompts], images=[[im] for im in images], padding=True,
                           return_tensors="pt").to(self.model.device, dtype=self.model.dtype)
        out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        return self.proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    def generate(self, images, prompts) -> list[str]:
        """Greedy decodes one answer per (image, prompt); a batch the processor cannot pad falls back to one by one."""
        try:
            return [t.strip() for t in self._generate(images, prompts)]
        except Exception as e:  # noqa: BLE001  (processors differ in what they batch; never lose the run to it)
            if len(images) == 1:
                raise
            print(f"[zoo] batched generate failed ({type(e).__name__}: {str(e)[:120]}); retrying one by one", flush=True)
            return [self._generate([im], [p])[0].strip() for im, p in zip(images, prompts)]


class LetterScorer:
    """Letter readout through GemmaBackend. `base` set: hf_id is a LoRA adapter merged onto `base`."""

    def __init__(self, hf_id, base=None, dtype="bfloat16", trust_remote_code=False, max_soft_tokens=280):
        self.backend = GemmaBackend(base or hf_id, max_soft_tokens=max_soft_tokens, dtype=dtype, gradient_checkpointing=False,
                                    adapter=hf_id if base else None, trust_remote_code=trust_remote_code, device_map="auto")
        self.backend.eval()
        self.n_params = _n_params(self.backend.model)

    @torch.no_grad()
    def logprobs(self, images, field, space, k_max=26):
        return field_logprobs(self.backend, images, field, space, k_max=k_max)


class D1Scorer:
    """Liquid d1 (3B and omni-600M share one API, per their model cards):
    `model.system_one_batch([(state, {name: {"type": "choice", "instructions", "criteria": {option: description}}}, [image])])`
    → `{"answers": {name: {"choice", "confidence", "probabilities"}}}`; the image is the whole state (state=None)."""

    def __init__(self, hf_id, dtype="bfloat16", trust_remote_code=True):
        from transformers import AutoModel
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModel.from_pretrained(hf_id, trust_remote_code=trust_remote_code, dtype=getattr(torch, dtype)).to(dev).eval()
        self.n_params = _n_params(self.model)

    @staticmethod
    def _question(field, names):
        return {"type": "choice", "instructions": QUESTION[field], "criteria": {n: n for n in names}}

    @staticmethod
    def _probs(ans, names):
        """Per-option probabilities from one `choice` answer (dict by name or list in option order); one-hot on `choice` if absent."""
        pr = ans.get("probabilities")
        if isinstance(pr, dict):
            p = torch.tensor([float(pr.get(n, 0.0)) for n in names])
        elif isinstance(pr, (list, tuple)) and len(pr) == len(names):
            p = torch.tensor([float(x) for x in pr])
        else:
            p = torch.zeros(len(names))
            if ans.get("choice") in names:
                p[names.index(ans["choice"])] = 1.0
        return torch.log(p.clamp_min(1e-6) / p.sum().clamp_min(1e-6))

    @torch.no_grad()
    def logprobs(self, images, field, space, k_max=None):
        names = space.classes[field]

        def run(option_sets):
            reqs = [(None, {"q": self._question(field, [names[i] for i in opts])}, [im]) for im, opts in zip(images, option_sets)]
            res = self.model.system_one_batch(reqs)
            res = [res] if isinstance(res, dict) else list(res)
            return [self._probs(r["answers"]["q"], [names[i] for i in opts]) for r, opts in zip(res, option_sets)]

        return tournament_logprobs(run, len(images), len(names), k_max or len(names))


class SystemOneError(RuntimeError):
    pass


class SystemOneScorer:
    """Client for a TypeSafe System One compatible server (respanai/span-02: self-hosted Rust/CUDA engine on an sm_120 GPU, or a
    hosted API). `POST {base_url}{endpoint}` with {"state", "questions": {field: choice question}, "images": [data URI],
    "reasoning": {"mode", "budget_ms"}} → {"answers": {field: {"choice", "confidence", "probabilities"}}}. One request per image
    carries every annotated field as its own `choice` question (the server accepts 1-679 options, so no tournament).
    `options` (registry `options:` / CLI): base_url, endpoint, api_key_env, model, reasoning, budget_ms, concurrency, timeout,
    retries, health. The bearer token is read from the env var `api_key_env` and only sent when it is set."""

    DEFAULTS = {"base_url": "http://127.0.0.1:8000", "endpoint": "/v1/systemone", "api_key_env": "SPAN_API_KEY", "model": None,
                "reasoning": "auto", "budget_ms": None, "concurrency": 8, "timeout": 300.0, "retries": 6, "backoff": 2.0,
                "health": "/health"}
    RETRY_STATUS = {408, 425, 429, 500, 502, 503, 504, 529}
    PHRASE = {"modality": "imaging modality", "domain": "medical domain"}
    FLOOR = 1e-6
    n_params = 26.0  # span-02 = Gemma-4-26B-A4B fine-tune (26B total, ~4B active)

    def __init__(self, options=None, session=None, check=True):
        import os
        import requests
        o = {**self.DEFAULTS, **{k: v for k, v in (options or {}).items() if v is not None}}
        unknown = set(o) - set(self.DEFAULTS)
        if unknown:
            raise ValueError(f"unknown systemone options {sorted(unknown)}")
        self.o = o
        self.url = o["base_url"].rstrip("/") + "/" + o["endpoint"].lstrip("/")
        self.session = session or requests.Session()
        key = os.environ.get(o["api_key_env"] or "")
        self.headers = {"Content-Type": "application/json", **({"Authorization": f"Bearer {key}"} if key else {})}
        if check:
            self.health()

    def health(self):
        """GET {base_url}/health must answer 200 (a 404, as on a hosted endpoint, only warns); any other failure raises."""
        if not self.o["health"]:
            return
        url = self.o["base_url"].rstrip("/") + self.o["health"]
        try:
            r = self.session.get(url, headers=self.headers, timeout=min(30.0, self.o["timeout"]))
        except Exception as e:  # noqa: BLE001
            raise SystemOneError(f"System One server not reachable at {url}: {type(e).__name__}: {e}") from e
        if r.status_code == 404:
            print(f"[zoo] {url} -> 404 (hosted endpoint?); skipping the health check", flush=True)
        elif r.status_code != 200:
            raise SystemOneError(f"System One health check {url} -> HTTP {r.status_code}: {r.text[:200]}")

    @staticmethod
    def _data_uri(im) -> str:
        buf = io.BytesIO()
        im.convert("RGB").save(buf, format="JPEG", quality=90)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

    def _body(self, image, fields, space) -> dict:
        questions = {f: {"type": "choice", "instructions": f"What is the {self.PHRASE.get(f, f.replace('_', ' '))} of Image 1?",
                         "criteria": {n: n for n in space.classes[f]}} for f in fields}
        reasoning = {"mode": self.o["reasoning"], **({"budget_ms": int(self.o["budget_ms"])} if self.o["budget_ms"] else {})}
        body = {"state": {"task": "biomedical image metadata", "image": "Image 1"}, "questions": questions,
                "images": [self._data_uri(image)], "reasoning": reasoning}
        if self.o["model"]:
            body["model"] = self.o["model"]
        return body

    def _post(self, body) -> dict:
        import requests
        last = ""
        for attempt in range(int(self.o["retries"]) + 1):
            if attempt:
                time.sleep(float(self.o["backoff"]) * 2 ** (attempt - 1) if self.o["backoff"] else 0)
            try:
                r = self.session.post(self.url, json=body, headers=self.headers, timeout=self.o["timeout"])
            except (requests.ConnectionError, requests.Timeout) as e:
                last = f"{type(e).__name__}: {e}"
                continue
            if r.status_code == 200:
                try:
                    return r.json()
                except ValueError as e:
                    raise SystemOneError(f"non-JSON 200 response from {self.url}: {r.text[:200]}") from e
            try:
                err = (r.json().get("error") or {})
                last = f"HTTP {r.status_code} {err.get('type', '')}: {err.get('message', r.text[:200])}"
            except (ValueError, AttributeError):
                last = f"HTTP {r.status_code}: {r.text[:200]}"
            if r.status_code not in self.RETRY_STATUS:
                raise SystemOneError(f"System One request failed ({last})")
        raise SystemOneError(f"System One request failed after {self.o['retries']} retries ({last})")

    def _score_one(self, image, fields, space) -> dict:
        res = self._post(self._body(image, fields, space))
        answers = res.get("answers") or {}
        out = {}
        for f in fields:
            ans = answers.get(f)
            if not isinstance(ans, dict) or "error" in ans:
                raise SystemOneError(f"no usable answer for field {f!r}: {str(ans if ans is not None else res)[:200]}")
            out[f] = self._probs(ans, space.classes[f])
        return out

    @classmethod
    def _probs(cls, ans, names):
        """Log-probs over `names` from one choice answer; classes the server did not return get the floor, then renormalized."""
        pr = ans.get("probabilities")
        if isinstance(pr, dict):
            p = torch.tensor([max(float(pr.get(n, 0.0)), cls.FLOOR) for n in names])
        else:
            p = torch.full((len(names),), cls.FLOOR)
            if ans.get("choice") in names:
                p[names.index(ans["choice"])] = 1.0
        return torch.log(p / p.sum())

    def score_rows(self, images, fields_per_image, space) -> list[dict]:
        """One request per image (thread pool): [{field: log-probs [n_classes]}] with the fields asked for each image."""
        with ThreadPoolExecutor(max_workers=max(1, int(self.o["concurrency"]))) as ex:
            return list(ex.map(lambda a: self._score_one(a[0], a[1], space), zip(images, fields_per_image)))

    def logprobs(self, images, field, space, k_max=None):
        rows = self.score_rows(images, [[field]] * len(images), space)
        return torch.stack([r[field] for r in rows])


def load_backend(entry):
    """The backend for a zoo entry (eval/zoo.py Entry)."""
    if entry.backend == "systemone":
        return SystemOneScorer(entry.options)
    if entry.mode == "generative":
        return HFGenerator(entry.hf_id, entry.dtype, entry.trust_remote_code, entry.max_new_tokens)
    if entry.readout == "d1":
        return D1Scorer(entry.hf_id, entry.dtype, entry.trust_remote_code)
    return LetterScorer(entry.hf_id, entry.base, entry.dtype, entry.trust_remote_code)
