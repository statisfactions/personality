"""Compatibility shim — hf_logprobs moved to pkit.extraction (2026-09-01).

Everything (MODELS, DISPLAY, resolve, display, load_model,
likert_distribution, bc_logodds, free_text, forward_with_hidden_states,
the remote-code pins and shims) now lives in pkit/, split as:
  pkit.roster     — naming layer (torch-free)
  pkit.extraction — model loading + logprob readouts (re-exports roster names)

This shim keeps every existing `PYTHONPATH=scripts` import working
unchanged. New code should import pkit directly.
"""
import os
import sys

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from pkit.extraction import *  # noqa: F401,F403
from pkit.extraction import (  # noqa: F401  (names import-* can miss)
    MODELS, DISPLAY, display, resolve, pick_device, load_model,
    likert_distribution, bc_logodds, free_text, forward_with_hidden_states,
    REMOTE_CODE_PINS, _FALLBACK_TEMPLATES, _shim_transformers_4x_api,
    _single_token_variants, _token_ids_map, _final_position_logits,
    _prob_per_label)
