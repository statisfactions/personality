"""HF-based logprob helpers for Likert surveys and binary-choice scoring.

Replaces the Ollama /api/generate + /api/chat path previously used by
run_ipip300.py, run_hexaco.py, and validate_protocol.py's Röttger/Likert tests.

Design notes
- Likert surveys (IPIP-300, HEXACO-100) use bare-text prompts — no chat
  template — to preserve the week 1–6 measurement behaviour. See to_try.md
  §15 for the bookmark on whether that choice is load-bearing.
- Binary choice uses the chat template by default (matches the old Ollama
  /api/chat path), exposed via use_chat_template=False for bare-text ablations.
- Digit / letter tokens may or may not carry a leading space depending on the
  tokenizer and the prompt's trailing character. We sum probability across
  the no-space and leading-space variants for whichever is single-token.
"""

import math
from typing import Dict, Iterable, List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


MODELS: Dict[str, str] = {
    # Small cohort (weeks 1–6).
    "Gemma":    "google/gemma-3-4b-it",
    "Llama":    "meta-llama/Llama-3.2-3B-Instruct",
    "Phi4":     "microsoft/Phi-4-mini-instruct",
    "Qwen":     "Qwen/Qwen2.5-3B-Instruct",
    # Lowercase aliases (validate_protocol.py convention).
    "gemma3":   "google/gemma-3-4b-it",
    "llama3.2": "meta-llama/Llama-3.2-3B-Instruct",
    "phi4":     "microsoft/Phi-4-mini-instruct",
    "qwen2.5":  "Qwen/Qwen2.5-3B-Instruct",
    # Phase-1 larger cohort (SAE-covered).
    "Gemma12":  "google/gemma-3-12b-it",
    "Gemma27":  "google/gemma-3-27b-it",
    "Llama8":   "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen7":    "Qwen/Qwen2.5-7B-Instruct",
}


def resolve(name_or_repo: str) -> str:
    """Short name → HF repo; unknown strings pass through."""
    return MODELS.get(name_or_repo, name_or_repo)


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(name_or_repo: str, device: str = None, dtype=None):
    device = device or pick_device()
    dtype = dtype if dtype is not None else torch.bfloat16
    repo = resolve(name_or_repo)
    tok = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForCausalLM.from_pretrained(repo, dtype=dtype, device_map=device)
    model.eval()
    return model, tok, device


def _single_token_variants(tok, s: str, space_variant: bool = True) -> List[int]:
    """Return all single-token IDs for `s` among {s, " "+s} (or just {s})."""
    candidates = [s, " " + s] if space_variant else [s]
    ids = []
    for c in candidates:
        enc = tok(c, add_special_tokens=False).input_ids
        if len(enc) == 1:
            ids.append(enc[0])
    return ids


def _token_ids_map(tok, labels: Iterable[str]) -> Dict[str, List[int]]:
    out = {}
    for label in labels:
        variants = _single_token_variants(tok, label)
        if not variants:
            raise ValueError(
                f"Label {label!r} has no single-token encoding in "
                f"{tok.__class__.__name__}"
            )
        out[label] = variants
    return out


def _final_position_logits(model, tok, text: str, device: str) -> torch.Tensor:
    inputs = tok(text, return_tensors="pt", add_special_tokens=True).to(device)
    with torch.no_grad():
        out = model(**inputs, use_cache=False)
    return out.logits[0, -1, :].float()


def _prob_per_label(logits: torch.Tensor, ids_map: Dict[str, List[int]]) -> Dict[str, float]:
    """Renormalized softmax over the given labels, summing variant tokens per label."""
    flat_ids = []
    label_for = []
    for label, ids in ids_map.items():
        for tid in ids:
            flat_ids.append(tid)
            label_for.append(label)
    selected = logits[torch.tensor(flat_ids, device=logits.device)]
    probs = torch.softmax(selected, dim=-1).tolist()
    result = {label: 0.0 for label in ids_map}
    for p, label in zip(probs, label_for):
        result[label] += p
    return result


def likert_distribution(
    model,
    tok,
    prompt: str,
    device: str,
    digits: Tuple[str, ...] = ("1", "2", "3", "4", "5"),
    use_chat_template: bool = True,
    system_content: str = "",
) -> Tuple[Dict[str, float], str, float]:
    """Distribution over Likert digits at the final-token position.

    With use_chat_template=True (default), the prompt is wrapped as a single
    user turn (optionally with a system turn) via tok.apply_chat_template with
    add_generation_prompt=True. This matches the weeks 1–6 Ollama
    /api/generate path which (with raw=False, the default for non-Qwen3
    models) applied the chat template server-side. Set use_chat_template=False
    for the bare-text ablation flagged in to_try.md §15.

    Returns (dist, argmax, entropy_nats).
    """
    ids_map = _token_ids_map(tok, digits)
    if use_chat_template:
        messages = []
        if system_content:
            messages.append({"role": "system", "content": system_content})
        messages.append({"role": "user", "content": prompt})
        text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tok(text, return_tensors="pt", add_special_tokens=False).to(device)
        with torch.no_grad():
            out = model(**inputs, use_cache=False)
        logits = out.logits[0, -1, :].float()
    else:
        logits = _final_position_logits(model, tok, prompt, device)
    dist = _prob_per_label(logits, ids_map)
    argmax = max(dist, key=dist.get)
    h = -sum(p * math.log(p) for p in dist.values() if p > 0)
    return dist, argmax, h


def bc_logodds(
    model,
    tok,
    user_prompt: str,
    device: str,
    a: str = "A",
    b: str = "B",
    use_chat_template: bool = True,
    system_content: str = "",
) -> Tuple[float, float, float]:
    """log(p(A) / p(B)) at the final-token position.

    With use_chat_template=True (default), wraps the prompt as a single user
    turn (optionally with a system turn) and calls tok.apply_chat_template with
    add_generation_prompt=True. Matches the old Ollama /api/chat behaviour.

    Returns (logit_a - logit_b, logit_a, logit_b). The returned value is a
    log-odds ratio over the two specific token IDs, *not* normalized across
    the full vocabulary — same convention as the old bc_diff in validate_protocol.
    """
    ids_map = _token_ids_map(tok, [a, b])
    if use_chat_template:
        messages = []
        if system_content:
            messages.append({"role": "system", "content": system_content})
        messages.append({"role": "user", "content": user_prompt})
        text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tok(text, return_tensors="pt", add_special_tokens=False).to(device)
    else:
        inputs = tok(user_prompt, return_tensors="pt", add_special_tokens=True).to(device)
    with torch.no_grad():
        out = model(**inputs, use_cache=False)
    logits = out.logits[0, -1, :].float()

    # Take the max logit across variants for each label; summing logits would
    # double-count. For BC we report raw logit difference, not probability.
    def _label_logit(label):
        return max(logits[tid].item() for tid in ids_map[label])

    la = _label_logit(a)
    lb = _label_logit(b)
    return la - lb, la, lb


def free_text(
    model,
    tok,
    user_prompt: str,
    device: str,
    max_new_tokens: int = 60,
    use_chat_template: bool = True,
    system_content: str = "",
) -> str:
    """Greedy free-text generation. Chat template on by default."""
    if use_chat_template:
        messages = []
        if system_content:
            messages.append({"role": "system", "content": system_content})
        messages.append({"role": "user", "content": user_prompt})
        text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tok(text, return_tensors="pt", add_special_tokens=False).to(device)
    else:
        inputs = tok(user_prompt, return_tensors="pt", add_special_tokens=True).to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    prompt_len = inputs["input_ids"].shape[1]
    generated = out[0, prompt_len:]
    return tok.decode(generated, skip_special_tokens=True).strip()
