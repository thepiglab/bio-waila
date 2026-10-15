"""Backends for the model zoo (eval/zoo.py): off-the-shelf models used zero-shot, GPU only.

  HFGenerator   any AutoModelForImageTextToText model: image + prompt → greedy-decoded text   (mode `generative`)
  LetterScorer  lettered-option readout via GemmaBackend (stock models, or a LoRA adapter on a base)   (mode `decision`)
  D1Scorer      Liquid d1 decision models (`model.system_one`, one `choice` question per field)       (mode `decision`)

Decision scorers share `logprobs(images, field, space, k_max) -> [B, n_classes]`; classes beyond k_max options per call go
through the two-round tournament in jev_gemma.tournament_logprobs. Every class here has `n_params` (billions).
"""
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


def load_backend(entry):
    """The backend for a zoo entry (eval/zoo.py Entry)."""
    if entry.mode == "generative":
        return HFGenerator(entry.hf_id, entry.dtype, entry.trust_remote_code, entry.max_new_tokens)
    if entry.readout == "d1":
        return D1Scorer(entry.hf_id, entry.dtype, entry.trust_remote_code)
    return LetterScorer(entry.hf_id, entry.base, entry.dtype, entry.trust_remote_code)
