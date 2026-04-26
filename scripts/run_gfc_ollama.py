#!/usr/bin/env python3
"""Administer Okada et al. (2026) GFC Big Five inventory via Ollama.

Presents 30 desirability-matched forced-choice pairs to an LLM and collects
7-point bipolar graded responses plus logprobs over all 7 options.

Supports both local Ollama (default) and remote Orin server (--remote).
Supports persona-based administration for TIRT fitting (--personas).

Usage:
    # No persona (single respondent)
    python3 scripts/run_gfc_ollama.py --model llama3.2:3b

    # With PersonaChat personas from admin session (50 respondents × 30 pairs)
    python3 scripts/run_gfc_ollama.py --model llama3.2:3b --personas

    # Limit personas for testing
    python3 scripts/run_gfc_ollama.py --model llama3.2:3b --personas --max-personas 5

    # Remote Orin server
    export $(grep OLLAMA_API_KEY .env | tr -d "'")
    python3 scripts/run_gfc_ollama.py --model llama3.2:3b --remote --personas

    # Quick test (5 pairs, no persona)
    python3 scripts/run_gfc_ollama.py --model llama3.2:3b --pairs 5
"""

import argparse
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError

INSTRUMENT_PATH = "instruments/okada_gfc30.json"
ADMIN_SESSION_PATH = "admin_sessions/prod_run_01_external_rating.json"
GFC_TOKENS = ["1", "2", "3", "4", "5", "6", "7"]
MAX_ENTROPY = math.log(len(GFC_TOKENS))  # ln(7) ≈ 1.946

# Ollama endpoints
LOCAL_URL = "http://localhost:11434"
REMOTE_URL = "https://apollo.quocanmeomeo.io.vn"


# --- Prompt construction ---

GFC_INSTRUCTION = (
    "For the following pair of statements, indicate which one describes you "
    "more accurately and by how much using a 7-point bipolar scale:\n"
    "1: LEFT statement describes me much more accurately\n"
    "2: LEFT statement describes me moderately more accurately\n"
    "3: LEFT statement describes me slightly more accurately\n"
    "4: About the same\n"
    "5: RIGHT statement describes me slightly more accurately\n"
    "6: RIGHT statement describes me moderately more accurately\n"
    "7: RIGHT statement describes me much more accurately\n"
    "Return ONLY one integer (1-7). Do not include any other text.\n"
)

GFC_ITEM_TEMPLATE = (
    "++++\n"
    "LEFT: {left_text}\n"
    "||\n"
    "RIGHT: {right_text}\n"
    "++++\n"
)

# Okada Appendix F.1: persona preamble is the full preamble string
# (includes "YOU ARE THE RESPONDENT" and "Answer all questions AS THIS PERSON would")
# followed by the honest instruction block (Appendix F.2)
PERSONA_PREFIX_TEMPLATE = (
    '{persona_desc}\n'
    'You will complete a personality questionnaire. When completing this '
    'questionnaire, we would like you to be as honest as possible, that is, '
    'to present yourself as you really are.\n'
)

# Okada Appendix F.2 fake-good preamble (used with persona prefix)
PERSONA_PREFIX_FAKEGOOD_TEMPLATE = (
    '{persona_desc}\n'
    'You will complete a personality questionnaire. When completing this '
    'questionnaire, we would like you to answer the questions in order to '
    'give a really good impression of yourself, that is, to present '
    'yourself in the best possible light.\n'
)

# Neutral frames for model-default placement (no persona)
# "respondent": includes "YOU ARE THE RESPONDENT" role assignment + honest instruction
RESPONDENT_PREFIX = (
    'YOU ARE THE RESPONDENT. '
    'You will complete a personality questionnaire. When completing this '
    'questionnaire, we would like you to be as honest as possible, that is, '
    'to present yourself as you really are.\n'
)
# "bare": honest instruction only, no role assignment
BARE_PREFIX = (
    'You will complete a personality questionnaire. When completing this '
    'questionnaire, we would like you to be as honest as possible, that is, '
    'to present yourself as you really are.\n'
)


def build_prompt(pair, randomize_lr=True, use_raw=False, seed=None,
                 persona_desc=None, neutral_mode=None, fake_good=False):
    """Build GFC prompt for a single pair, optionally with persona prefix.

    Args:
        pair: Dict with 'left' and 'right' item dicts.
        randomize_lr: If True, randomly swap left/right assignment.
        use_raw: If True, use Qwen3 raw prompt format.
        seed: Random seed for left/right randomization (for reproducibility).
        persona_desc: If set, prepend persona description to the prompt.
        neutral_mode: "respondent" = honest instruction (no persona),
                      "bare" = just GFC instruction + item, no framing.
                      None = default (use persona_desc if provided).
        fake_good: If True, use Okada F.2 fake-good preamble instead of
                   honest. Only meaningful with persona_desc set.

    Returns:
        (prompt_text, swapped) where swapped is True if L/R were flipped.
    """
    rng = random.Random(seed)
    swapped = False
    left_text = pair["left"]["text"]
    right_text = pair["right"]["text"]

    if randomize_lr and rng.random() < 0.5:
        left_text, right_text = right_text, left_text
        swapped = True

    # Build prompt parts
    parts = []
    if persona_desc:
        tpl = PERSONA_PREFIX_FAKEGOOD_TEMPLATE if fake_good else PERSONA_PREFIX_TEMPLATE
        parts.append(tpl.format(persona_desc=persona_desc))
    elif neutral_mode == "respondent":
        parts.append(RESPONDENT_PREFIX)
    elif neutral_mode == "bare":
        parts.append(BARE_PREFIX)
    parts.append(GFC_INSTRUCTION)
    parts.append(GFC_ITEM_TEMPLATE.format(left_text=left_text,
                                          right_text=right_text))
    prompt = "".join(parts)

    if use_raw:
        prompt = (
            "<|im_start|>system\n"
            "You are a helpful assistant. /no_think<|im_end|>\n"
            "<|im_start|>user\n" + prompt +
            "<|im_end|>\n<|im_start|>assistant\n"
        )

    return prompt, swapped


def load_personas(session_path):
    """Extract unique persona descriptions from PsyBORGS admin session.

    Returns dict mapping persona_id (e.g. 'd1') to description string.
    """
    with open(session_path) as f:
        session = json.load(f)

    preambles = session.get("item_preambles", {})
    personas = {}
    for key, text in preambles.items():
        if key.endswith("-rg6"):  # take one framing per persona
            match = re.search(r'"(.+?)"', text)
            if match:
                desc_id = key.split("-")[0]
                personas[desc_id] = match.group(1)

    return dict(sorted(personas.items(), key=lambda x: int(x[0][1:])))


def load_synthetic_personas(path):
    """Load synthetic personas with known trait profiles.

    Generated by scripts/generate_trait_personas.py.
    Returns dict mapping persona_id (e.g. 's1') to preamble string.
    The preamble includes "YOU ARE THE RESPONDENT" framing (Okada F.1).
    """
    with open(path) as f:
        data = json.load(f)

    personas = {}
    for p in data["personas"]:
        personas[p["persona_id"]] = p["preamble"]

    return personas


# --- Ollama API (local, urllib) ---

def ollama_generate_local(model, prompt, top_logprobs=10, num_predict=1,
                          raw=False):
    """Call local Ollama /api/generate."""
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "raw": raw,
        "stream": False,
        "options": {"num_predict": num_predict, "temperature": 0},
        "logprobs": True,
        "top_logprobs": top_logprobs,
    }).encode()

    req = Request(
        f"{LOCAL_URL}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urlopen(req, timeout=300) as resp:
        return json.loads(resp.read())


# --- Ollama API (remote Orin, curl) ---

def ollama_generate_remote(model, prompt, api_key, top_logprobs=20,
                           timeout=120):
    """Call remote Orin server via curl (OpenAI-compatible endpoint)."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system",
             "content": ("You are completing a personality questionnaire. "
                         "Respond with ONLY a single integer (1-7).")},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "stream": False,
        "logprobs": True,
        "top_logprobs": top_logprobs,
    }
    payload_json = json.dumps(payload, separators=(",", ":"))

    cmd = [
        "curl", "-s",
        "-m", str(timeout),
        "-H", f"Authorization: Bearer {api_key}",
        "-H", "Content-Type: application/json",
        "-d", payload_json,
        f"{REMOTE_URL}/v1/chat/completions",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return None

    try:
        data = json.loads(result.stdout)
        choice = data["choices"][0]
        content = choice["message"]["content"].strip()
        logprobs_content = choice.get("logprobs", {}).get("content", [])
        # Convert to local Ollama format for unified processing
        return {
            "response": content,
            "logprobs": [
                {
                    "token": entry["token"],
                    "logprob": entry["logprob"],
                    "top_logprobs": entry.get("top_logprobs", []),
                }
                for entry in logprobs_content
            ] if logprobs_content else [],
        }
    except (json.JSONDecodeError, KeyError, IndexError):
        return None


# --- Logprob extraction ---

def extract_gfc_distribution(response):
    """Extract probability distribution over {1..7} from logprobs.

    Returns (dist, argmax_response, raw_logprobs) where:
        dist: dict mapping "1"-"7" to renormalized probabilities
        argmax_response: str, the most probable GFC token
        raw_logprobs: dict mapping "1"-"7" to raw log-probabilities
    """
    if response is None:
        return None, None, None

    generated = response.get("response", "").strip()
    logprobs_list = response.get("logprobs", [])
    if not logprobs_list:
        return None, generated, None

    # Find the right token position (skip thinking tokens if present)
    token_logprobs = None
    past_think = False
    for entry in logprobs_list:
        tok = entry.get("token", "")
        if tok == "</think>":
            past_think = True
            continue
        if not past_think and logprobs_list[0].get("token", "") == "<think>":
            continue
        candidates = entry.get("top_logprobs", [])
        has_gfc = any(c.get("token", "").strip() in GFC_TOKENS
                      for c in candidates)
        if has_gfc:
            token_logprobs = candidates
            break

    if token_logprobs is None:
        token_logprobs = logprobs_list[0].get("top_logprobs", [])

    # Build raw logprob map
    raw = {}
    for entry in token_logprobs:
        tok = entry.get("token", "").strip()
        if tok in GFC_TOKENS:
            raw[tok] = entry["logprob"]

    if not raw:
        return None, generated, None

    # Renormalize over GFC tokens
    max_lp = max(raw.values())
    exp_sum = sum(math.exp(lp - max_lp) for lp in raw.values())
    dist = {}
    for tok in GFC_TOKENS:
        if tok in raw:
            dist[tok] = math.exp(raw[tok] - max_lp) / exp_sum
        else:
            dist[tok] = 0.0

    argmax = max(raw, key=raw.get)

    return dist, argmax, raw


def entropy(dist):
    """Shannon entropy in nats."""
    if dist is None:
        return None
    return -sum(p * math.log(p) for p in dist.values() if p > 0)


def expected_value(dist):
    """Expected value of the 1-7 distribution."""
    if dist is None:
        return None
    return sum(int(k) * v for k, v in dist.items())


# --- Main ---

def load_instrument(path):
    """Load GFC instrument JSON."""
    with open(path) as f:
        return json.load(f)


def administer_one(pair, model, api_key, use_raw, num_predict,
                   top_logprobs, timeout, randomize_lr, seed,
                   persona_desc=None, remote=False, neutral_mode=None,
                   fake_good=False):
    """Administer a single GFC pair and return result dict."""
    prompt, swapped = build_prompt(
        pair,
        randomize_lr=randomize_lr,
        use_raw=use_raw,
        seed=seed,
        persona_desc=persona_desc,
        neutral_mode=neutral_mode,
        fake_good=fake_good,
    )

    if remote:
        response = ollama_generate_remote(
            model, prompt, api_key,
            top_logprobs=top_logprobs,
            timeout=timeout)
    else:
        response = ollama_generate_local(
            model, prompt, top_logprobs,
            num_predict=num_predict, raw=use_raw)

    dist, argmax_resp, raw_logprobs = extract_gfc_distribution(response)
    ev = expected_value(dist)
    h = entropy(dist)

    if swapped:
        actual_left = pair["right"]
        actual_right = pair["left"]
    else:
        actual_left = pair["left"]
        actual_right = pair["right"]

    return {
        "block": pair["block"],
        "left_trait": actual_left["trait"],
        "left_keying": actual_left["keying"],
        "left_text": actual_left["text"],
        "right_trait": actual_right["trait"],
        "right_keying": actual_right["keying"],
        "right_text": actual_right["text"],
        "swapped": swapped,
        "response_argmax": argmax_resp,
        "response_ev": round(ev, 4) if ev is not None else None,
        "response_entropy": round(h, 4) if h is not None else None,
        "distribution": dist,
        "raw_logprobs": raw_logprobs,
        "generated_text": response.get("response", "").strip()
                          if response else None,
    }


def print_summary(results):
    """Print summary statistics for a set of GFC results."""
    valid = [r for r in results if r["response_argmax"] is not None]
    print(f"Valid responses: {len(valid)}/{len(results)}")
    if not valid:
        return

    evs = [r["response_ev"] for r in valid if r["response_ev"] is not None]
    hs = [r["response_entropy"] for r in valid
          if r["response_entropy"] is not None]
    if evs:
        print(f"Mean EV: {sum(evs)/len(evs):.2f} (4.0 = no preference)")
    if hs:
        print(f"Mean entropy: {sum(hs)/len(hs):.3f} / {MAX_ENTROPY:.3f}")

    print()
    print("Trait preferences (mean endorsement, >4 = high trait):")
    for trait in ["A", "C", "E", "N", "O"]:
        all_endorsement = []
        for r in valid:
            if r["response_ev"] is None:
                continue
            if r["right_trait"] == trait:
                all_endorsement.append(r["response_ev"])
            if r["left_trait"] == trait:
                all_endorsement.append(8 - r["response_ev"])
        if all_endorsement:
            mean_end = sum(all_endorsement) / len(all_endorsement)
            print(f"  {trait}: {mean_end:.2f} (n={len(all_endorsement)})")


def main():
    parser = argparse.ArgumentParser(
        description="Administer Okada GFC-30 Big Five via Ollama")
    parser.add_argument("--model", required=True,
                        help="Ollama model name (e.g. llama3.2:3b)")
    parser.add_argument("--pairs", type=int, default=0,
                        help="Limit to first N pairs (0 = all 30)")
    parser.add_argument("--remote", action="store_true",
                        help="Use remote Orin server instead of local Ollama")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (auto-generated if omitted)")
    parser.add_argument("--top-logprobs", type=int, default=10,
                        help="Number of top logprobs to request")
    parser.add_argument("--no-randomize", action="store_true",
                        help="Don't randomize left/right assignment")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for L/R randomization")
    parser.add_argument("--timeout", type=int, default=120,
                        help="Request timeout in seconds")
    # Persona options
    parser.add_argument("--personas", action="store_true",
                        help="Administer under PersonaChat personas (50 respondents)")
    parser.add_argument("--synthetic-personas", type=str, default=None,
                        help="Path to synthetic persona JSON (from generate_trait_personas.py)")
    parser.add_argument("--max-personas", type=int, default=0,
                        help="Limit to first N personas (0 = all)")
    parser.add_argument("--admin-session", type=str,
                        default=ADMIN_SESSION_PATH,
                        help="Path to admin session JSON for persona extraction")
    parser.add_argument("--checkpoint-every", type=int, default=100,
                        help="Save checkpoint every N prompts (default 100)")
    parser.add_argument("--neutral", type=str, default=None,
                        choices=["respondent", "bare"],
                        help="Neutral mode: 'respondent' = honest instruction "
                             "(no persona), 'bare' = just GFC instruction + item")
    parser.add_argument("--fake-good", action="store_true",
                        help="Use Okada F.2 fake-good preamble instead of "
                             "honest. Only affects persona prompts.")
    args = parser.parse_args()

    instrument = load_instrument(INSTRUMENT_PATH)
    pairs = instrument["pairs"]
    if args.pairs > 0:
        pairs = pairs[:args.pairs]

    # Load personas if requested
    persona_list = [("none", None)]  # default: no persona
    persona_source = "no persona"
    if args.synthetic_personas:
        personas = load_synthetic_personas(args.synthetic_personas)
        persona_list = list(personas.items())
        persona_source = f"synthetic ({args.synthetic_personas})"
    elif args.personas:
        personas = load_personas(args.admin_session)
        persona_list = list(personas.items())
        persona_source = "PersonaChat"
    if args.max_personas > 0:
        persona_list = persona_list[:args.max_personas]

    total_prompts = len(persona_list) * len(pairs)
    print(f"Model:    {args.model}")
    print(f"Pairs:    {len(pairs)} / {instrument['n_pairs']}")
    print(f"Personas: {len(persona_list)} ({persona_source})")
    print(f"Total:    {total_prompts} prompts")
    print(f"Mode:     {'remote (Orin)' if args.remote else 'local Ollama'}")
    print()

    # Output path
    output_path = args.output
    if output_path is None:
        os.makedirs("results", exist_ok=True)
        model_slug = args.model.replace(":", "-").replace("/", "-")
        if args.synthetic_personas:
            suffix = "_synthetic-fakegood" if args.fake_good else "_synthetic"
        elif args.personas:
            suffix = "_personas-fakegood" if args.fake_good else "_personas"
        elif args.neutral:
            suffix = f"_neutral-{args.neutral}"
        else:
            suffix = ""
        output_path = f"results/{model_slug}_gfc30{suffix}.json"

    # Resume from checkpoint if it exists
    completed_keys = set()
    prior_results = []
    if os.path.exists(output_path):
        with open(output_path) as f:
            prior = json.load(f)
        prior_results = prior.get("results", [])
        for r in prior_results:
            completed_keys.add((r.get("persona_id", "none"), r["block"]))
        print(f"Resuming from checkpoint: {len(prior_results)}/{total_prompts} "
              f"already completed")
        print()

    # Connectivity check
    api_key = None
    if args.remote:
        api_key = os.environ.get("OLLAMA_API_KEY")
        if not api_key:
            print("ERROR: OLLAMA_API_KEY not set for remote mode.",
                  file=sys.stderr)
            sys.exit(1)
        cmd = ["curl", "-s", "-m", "10",
               "-H", f"Authorization: Bearer {api_key}",
               f"{REMOTE_URL}/api/tags"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Cannot reach Orin at {REMOTE_URL}", file=sys.stderr)
            sys.exit(1)
        print("Connected to Orin server.")
    else:
        try:
            urlopen(f"{LOCAL_URL}/api/tags", timeout=5)
        except URLError as e:
            print(f"Cannot reach Ollama at {LOCAL_URL}: {e}", file=sys.stderr)
            sys.exit(1)
        print("Connected to local Ollama.")

    # Detect thinking model (local only)
    use_raw = False
    num_predict = 1
    if not args.remote:
        probe = ollama_generate_local(args.model, "Say hello.",
                                      args.top_logprobs, num_predict=2)
        first_tok = (probe.get("logprobs", [{}])[0].get("token", "")
                     if probe.get("logprobs") else "")
        if first_tok == "<think>":
            use_raw = True
            num_predict = 10
            print("Detected thinking model — using /no_think with raw prompt")

    print()

    # Administer all persona × pair combinations
    results = list(prior_results)
    n_completed = len(prior_results)
    n_new = 0
    t_start = time.time()

    for persona_id, persona_desc in persona_list:
        persona_label = persona_id if persona_desc else "no-persona"
        persona_done = sum(1 for k in completed_keys if k[0] == persona_id)
        if persona_done == len(pairs):
            continue  # all pairs done for this persona

        if args.personas:
            print(f"--- Persona {persona_id} "
                  f"({persona_desc[:60]}...) ---")

        for pair in pairs:
            block = pair["block"]
            if (persona_id, block) in completed_keys:
                continue

            # Seed incorporates persona for different L/R per persona
            lr_seed = args.seed * 10000 + hash(persona_id) % 10000 + block

            result = administer_one(
                pair, args.model, api_key, use_raw, num_predict,
                args.top_logprobs, args.timeout,
                randomize_lr=not args.no_randomize,
                seed=lr_seed,
                persona_desc=persona_desc,
                remote=args.remote,
                neutral_mode=args.neutral,
                fake_good=args.fake_good,
            )
            result["persona_id"] = persona_id
            results.append(result)
            n_completed += 1
            n_new += 1

            # Progress
            argmax = result["response_argmax"]
            ev = result["response_ev"]
            status = (f"  [{n_completed}/{total_prompts}] "
                      f"B{block:02d} → {argmax}")
            if ev is not None:
                status += f"  EV={ev:.2f}"
            print(status)

            # Checkpoint
            if (args.checkpoint_every > 0
                    and n_new % args.checkpoint_every == 0):
                _save_output(output_path, args, pairs, persona_list, results)
                elapsed = time.time() - t_start
                rate = n_new / elapsed if elapsed > 0 else 0
                print(f"\n  Checkpoint: {n_completed}/{total_prompts} "
                      f"({rate:.1f} prompts/sec)\n")

    # Final save
    _save_output(output_path, args, pairs, persona_list, results)

    # Summary
    print()
    print("=" * 60)
    print_summary(results)
    elapsed = time.time() - t_start
    if n_new > 0:
        print(f"\nCompleted {n_new} new prompts in {elapsed:.0f}s "
              f"({n_new/elapsed:.1f}/sec)")
    print(f"Results saved to {output_path}")


def _save_output(path, args, pairs, persona_list, results):
    """Save results to JSON."""
    output = {
        "model": args.model,
        "instrument": "okada_gfc30",
        "n_pairs": len(pairs),
        "n_personas": len(persona_list),
        "has_personas": args.personas,
        "randomize_lr": not args.no_randomize,
        "seed": args.seed,
        "results": results,
    }
    with open(path, "w") as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    main()
