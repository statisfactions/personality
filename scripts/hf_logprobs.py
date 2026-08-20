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
    # W12 §6 cohort scale-up (M5 Max 128GB).
    "Qwen32":   "Qwen/Qwen2.5-32B-Instruct",
    "Gemma4":   "google/gemma-4-31B-it",         # needs transformers >= 5.5
    "Gemma4MoE": "google/gemma-4-26B-A4B-it",    # MoE, MPS untested
    "Qwen36":   "Qwen/Qwen3.6-35B-A3B",          # MoE, thinking-mode default-on
    # W13 §8.2 outlier models (axis-of-the-models hypothesis test).
    "Aya":         "CohereLabs/aya-expanse-8b",            # multilingual SFT, 23 langs
    "FalconMamba": "tiiuae/falcon-mamba-7b-instruct",      # pure SSM, no attention
    # Mr. Chatterbox (tventurella/mr_chatterbox_model) was the third §8.2 outlier
    # candidate — a 340M nanochat GPT trained from scratch on 1837–1899 British
    # text. Struck: the repo ships only the checkpoint, not the custom 32768-vocab
    # BPE it was trained with, so it's unrunnable as published (token IDs would be
    # meaningless). Not a loader-effort problem.
    # W15 §3 base-vs-instruct (training-stage) checkpoints — probe with bare text
    # (format held constant across stages so the weights are the only variable).
    "Qwen7Base":  "Qwen/Qwen2.5-7B",                  # base for Qwen7 (Instruct)
    "Llama8Base": "meta-llama/Llama-3.1-8B",          # base for Llama8 (Instruct)
    "Gemma12Base": "google/gemma-3-12b-pt",           # base for Gemma12 (it)
    # OLMo-2 7B: clean base -> SFT -> DPO -> Instruct(=RLVR) ladder, same tokenizer
    # across stages (AllenAI, Apache-2.0, ungated; verified by W15 §3 agent).
    "Olmo2Base":  "allenai/OLMo-2-1124-7B",
    "Olmo2SFT":   "allenai/OLMo-2-1124-7B-SFT",
    "Olmo2DPO":   "allenai/OLMo-2-1124-7B-DPO",
    "Olmo2Inst":  "allenai/OLMo-2-1124-7B-Instruct",  # RLVR endpoint
    # Zephyr SFT-vs-DPO minimal pair (Mistral-7B base; no RLVR confound).
    "ZephyrSFT":  "alignment-handbook/zephyr-7b-sft-full",
    "ZephyrDPO":  "alignment-handbook/zephyr-7b-dpo-full",
}


def resolve(name_or_repo: str) -> str:
    """Short name → HF repo; unknown strings pass through."""
    return MODELS.get(name_or_repo, name_or_repo)


# Canonical display names: {family}{version}-{size}, SIZE ALWAYS EXPLICIT.
# The short names hide scale confusingly ("qwen2.5" is the 3B; "Qwen7" the 7B),
# so use display() for all charts/tables/reports. Nicknames in comments.
DISPLAY = {
    "llama3.2": "Llama3.2-3B",  "Llama":   "Llama3.2-3B",
    "Llama8":   "Llama3.1-8B",
    "qwen2.5":  "Qwen2.5-3B",   "Qwen":    "Qwen2.5-3B",
    "Qwen7":    "Qwen2.5-7B",
    "Qwen32":   "Qwen2.5-32B",
    "gemma3":   "Gemma3-4B",    "Gemma":   "Gemma3-4B",
    "Gemma12":  "Gemma3-12B",
    "Gemma27":  "Gemma3-27B",
    "Gemma4":   "Gemma4-31B",
    "Gemma4MoE": "Gemma4-26B-A4B",
    "phi4":     "Phi4-3.8B",    "Phi4":    "Phi4-3.8B",   # Phi-4-mini
    "Aya":      "Aya-8B",                                  # aya-expanse-8b
    "FalconMamba": "FalconMamba-7B",
    "Qwen36":   "Qwen3.6-35B-A3B",
    "Qwen7Base": "Qwen2.5-7B-base",
    "Llama8Base": "Llama3.1-8B-base",
    "Gemma12Base": "Gemma3-12B-base",
    "Olmo2Base": "OLMo2-7B-base", "Olmo2SFT": "OLMo2-7B-SFT",
    "Olmo2DPO": "OLMo2-7B-DPO",   "Olmo2Inst": "OLMo2-7B-RLVR",
    "ZephyrSFT": "Zephyr-7B-SFT", "ZephyrDPO": "Zephyr-7B-DPO",
}


def display(name: str) -> str:
    """Short name → canonical {family}{version}-{size} label; unknown → as-is."""
    return DISPLAY.get(name, name)


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# Chat templates for pre-template-convention instruct models, so every
# instrument arm sees them as chat models (otherwise: silently bare in the
# self arm, ValueError in enact/represent). Format per the model's own
# deployment docs (FastChat conversation registry for Vicuna v1.5).
_FALLBACK_TEMPLATES = {
    "vicuna": (
        "{% if messages[0]['role'] != 'system' %}"
        "{{ 'A chat between a curious user and an artificial intelligence "
        "assistant. The assistant gives helpful, detailed, and polite "
        "answers to the user\\'s questions. ' }}{% endif %}"
        "{% for message in messages %}"
        "{% if message['role'] == 'system' %}{{ message['content'] + ' ' }}"
        "{% elif message['role'] == 'user' %}"
        "{{ 'USER: ' + message['content'] + ' ' }}"
        "{% elif message['role'] == 'assistant' %}"
        "{{ 'ASSISTANT: ' + message['content'] + '</s>' }}"
        "{% endif %}{% endfor %}"
        "{% if add_generation_prompt %}{{ 'ASSISTANT:' }}{% endif %}"),
}


# trust_remote_code allowlist: ONLY manifest-flagged models, ONLY at the
# pinned revision (trusting a repo without a pin trusts all future commits).
# Populated from instruments/cohort100_manifest.json at import.
def _remote_code_pins():
    import json
    import os
    p = os.path.join(os.path.dirname(__file__), "..",
                     "instruments", "cohort100_manifest.json")
    try:
        man = json.load(open(p))["models"]
        return {m["repo"]: m["revision"] for m in man
                if "remote_code" in m.get("flags", []) and m.get("revision")}
    except Exception:
        return {}


REMOTE_CODE_PINS = _remote_code_pins()


def load_model(name_or_repo: str, device: str | None = None, dtype=None):
    device = device or pick_device()
    dtype = dtype if dtype is not None else torch.bfloat16
    repo = resolve(name_or_repo)
    rc = {}
    if repo in REMOTE_CODE_PINS:
        rc = {"trust_remote_code": True, "revision": REMOTE_CODE_PINS[repo]}
        print(f"[hf_logprobs] remote code enabled for {repo} "
              f"@ {rc['revision']} (manifest pin)")
    tok = AutoTokenizer.from_pretrained(repo, **rc)
    if tok.chat_template is None:
        for key, tmpl in _FALLBACK_TEMPLATES.items():
            if key in repo.lower():
                tok.chat_template = tmpl
                print(f"[hf_logprobs] injected fallback chat template "
                      f"({key}) for {repo}")
                break
    model = AutoModelForCausalLM.from_pretrained(repo, dtype=dtype,
                                                 device_map=device, **rc)
    model.eval()
    return model, tok, device


def _single_token_variants(tok, s: str, space_variant: bool = True) -> List[int]:
    """Return all single-token IDs for `s` among {s, " "+s} (or just {s}).

    Slow SentencePiece tokenizers (e.g. LlamaTokenizer when fast conversion
    fails, as with Vicuna) encode bare digits as [prefix-space, digit] — two
    tokens — even though the digit IS a single vocab entry. Fall back to
    direct vocab lookup of s and "▁"+s (SP word-initial form)."""
    candidates = [s, " " + s] if space_variant else [s]
    ids = []
    for c in candidates:
        enc = tok(c, add_special_tokens=False).input_ids
        if len(enc) == 1:
            ids.append(enc[0])
    if not ids:
        unk = tok.unk_token_id
        for form in (s, "▁" + s):
            tid = tok.convert_tokens_to_ids(form)
            if tid is not None and tid != unk and tid >= 0:
                ids.append(tid)
    return list(dict.fromkeys(ids))


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
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False
        )
        inputs = tok(text, return_tensors="pt", add_special_tokens=False).to(device)
        with torch.no_grad():
            out = model(**inputs, use_cache=False)
        logits = out.logits[0, -1, :].float()
    else:
        logits = _final_position_logits(model, tok, prompt, device)
    dist = _prob_per_label(logits, ids_map)
    argmax = max(dist, key=lambda k: dist[k])
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
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False
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
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False
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
