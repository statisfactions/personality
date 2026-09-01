"""Model naming + cohort roster — the light, torch-free layer.

MODELS / DISPLAY / resolve / display moved verbatim from
scripts/hf_logprobs.py (refactor-core, 2026-09-01; scripts/hf_logprobs.py
is now a shim over pkit.extraction, which re-exports these). The roster
sets below consolidate conventions that lived in facet_slides_wide.py and
self_population_slides.py.
"""
import re
from typing import Dict


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


# ---------------------------------------------------------------------------
# Cohort roster conventions (SELF wide-n)
# ---------------------------------------------------------------------------

# Filename-level exclusion for aggregate loaders: training-stage rungs
# (Base/SFT/DPO), bare-prompt arms, think arms, shelved artifacts, smokes.
# (From facet_slides_wide.py.)
EXCLUDE = re.compile(r"Base|SFT|DPO|_bare|THINKOPEN|_think|_smoke|_PRERETRY")

# Always-think models: the forced-prefill SELF row is off-policy noise
# (Glimmer: prefill shape r=-0.19 with cohort vs think 0.84); loaders
# prefer the _think file when it exists (2026-08-24 swap, ledgered).
THINK_PREFER = {"meta-models/Muse-Glimmer-30B"}

# Population-roster drops (2026-08-24, self_population_slides.py):
#   internlm2_5-7b-chat — instrument-broken (adjective-invariance retest
#     r=0.11 with original: model, not harness);
#   falcon-7b-instruct — probation for flat-row leverage (spread 0.16;
#     LOO rotates ipsatized iPC4 by 1-|r|=0.90).
# R1 distills stay IN pending the statisfactions conversation.
DROP = {"internlm2_5-7b-chat", "falcon-7b-instruct"}


def manifest():
    """cohort100_manifest.json entries (list of dicts)."""
    import json

    from .paths import MANIFEST
    return json.load(open(MANIFEST))["models"]
