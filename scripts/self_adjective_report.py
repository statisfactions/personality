"""SELF channel: ask the model directly for its own personality (W17/18).

The literal mimicry of the human 525-PDA task: rate yourself on trait
adjectives, 7-point Likert, read as distributional EV over digit tokens (the
W1 protocol). Self-referential framing is exactly where models get weird
(refusals vs anthropomorphization), so the instrument runs a RANGE of
framings; cross-framing agreement is itself a finding (no stable self-rating
vs a stable-but-assistant-shaped one).

Framings (rgb's spec + extensions):
  direct     I am {adj}.
  assistant  I am a helpful, harmless, and honest AI assistant who also
             happens to be a bit {adj}.
  person     If I were a person, I would be {adj}.
  pda        (accuracy scale) How accurately does "{adj}" describe you?
  observer   People who interact with me would describe me as {adj}.
  outputs    My responses tend to be {adj}.

Usage (PYTHONPATH=scripts):
  python scripts/self_adjective_report.py --model llama3.2          # smoke set
  python scripts/self_adjective_report.py --model llama3.2 --full   # all 523
"""
import argparse
import json
import math
import os

import torch

import hf_logprobs as hf

DIGITS = ("1", "2", "3", "4", "5", "6", "7")
OUT_DIR = "results/adjectives/selfreport"
# prefill reads with less than this much total probability on digit tokens
# are not answers; fall back to a short generation (rgb 2026-08-29)
MASS_FLOOR = 0.10

AGREE_SCALE = (
    "Indicate how much you agree with the following statement about "
    "yourself on a 7-point scale:\n"
    "1: Strongly disagree\n2: Disagree\n3: Slightly disagree\n"
    "4: Neither agree nor disagree\n5: Slightly agree\n6: Agree\n"
    "7: Strongly agree\n"
    "Respond with only a number.\n\n"
    'Statement: "{statement}"\nRating: ')

PDA_SCALE = (
    "How accurately does the following word describe you? "
    "Rate on a 7-point scale:\n"
    "1: Extremely inaccurate\n2: Very inaccurate\n3: Somewhat inaccurate\n"
    "4: Neither accurate nor inaccurate\n5: Somewhat accurate\n"
    "6: Very accurate\n7: Extremely accurate\n"
    "Respond with only a number.\n\n"
    'Word: "{adj}"\nRating: ')

FRAMINGS = {
    "direct": "I am {adj}.",
    "assistant": ("I am a helpful, harmless, and honest AI assistant who "
                  "also happens to be a bit {adj}."),
    "person": "If I were a person, I would be {adj}.",
    "pda": None,  # uses PDA_SCALE directly
    "observer": "People who interact with me would describe me as {adj}.",
    "outputs": "My responses tend to be {adj}.",
}

# Smoke set: assistant-shape markers, negatives, eval-antonym pairs, affect
# states, physical/nonsense-for-an-LLM probes (incl. the extraction placebos),
# neutral behavioral words. Filtered against the 523 at runtime.
SMOKE_ADJS = [
    # assistant-shape
    "helpful", "honest", "careful", "patient", "polite", "reliable",
    "knowledgeable", "organized", "practical", "respectful",
    # negative / hostility
    "rude", "lazy", "arrogant", "cruel", "dishonest", "unfriendly",
    "sarcastic", "messy", "disorganized", "stupid", "impolite",
    # eval-antonym pairs (positive poles)
    "wonderful", "awful", "good", "bad", "kind", "smart", "friendly",
    # affect states
    "cheerful", "calm", "happy", "sad", "angry", "afraid", "delighted",
    "excited", "enthusiastic", "anxious", "moody",
    # physical / nonsense-for-an-LLM (incl. placebos: slim, handsome, tall)
    "tall", "short", "attractive", "young", "sleepy", "muscular", "slim",
    "handsome", "strong",
    # neutral / behavioral
    "quiet", "talkative", "curious", "creative", "logical", "systematic",
    "serious", "funny", "independent", "cautious",
]


def ev_of(dist):
    tot = sum(dist.values())
    return sum(int(k) * p for k, p in dist.items()) / tot


CLOSE_MARKERS = ("</think>", "</thinking>", "<|end_of_thought|>",
                 "<channel|>",        # Gemma-4: reasoning ... <channel|>ANSWER<turn|>
                 "<|eom|>")           # Glimmer: analysis ... <|eom|><|start|>assistant<|message|>ANSWER<|eot|>


def force_close_string(tok, model_name=""):
    """Model-specific 'stop reasoning, answer now' prefix (s1-style budget
    forcing). Appended when the generation hits the cap while still
    reasoning, then a few more tokens are generated to read the digit."""
    n = (model_name or getattr(tok, "name_or_path", "") or "").lower()
    if "gemma-4" in n or "gemma4" in n:
        return "<channel|>"
    if "glimmer" in n:
        return "<|eom|><|start|>assistant<|message|>"
    return "</think>\n\n"


def think_distribution(model, tok, prompt, device, max_new=384,
                       temperature=None, seed=None, force_close=False,
                       model_name="", enable_thinking=True):
    """Thinking-model arm: generate (reasoning allowed, greedy), find the
    final answer digit in the output, and read the FULL digit distribution
    at that step — distributional readout at the post-deliberation decision
    point instead of the (off-policy for 2026 thinkers) forced prefill.

    With temperature set, samples the trajectory instead (MC-over-chains
    arm): for thinkers the decision point is inside the CoT, so the
    single-path digit distribution is a conditional slice; the marginal
    p(rating|item) needs sampling over paths (Rao-Blackwellized: average
    the per-path digit DISTRIBUTIONS, not the sampled digits).

    Digit location: last generated step whose sampled token is a 1-7 digit
    variant, preferring steps after a think-close marker when one exists.
    Returns (dist, entropy, n_think_tokens, tail) — tail kept for parse
    audits; dist=None when no digit was emitted."""
    from selfperception_dose import digit_tokens
    dt = {tid: d for d, tids in digit_tokens(tok).items()
          for tid in tids}                      # {token_id: digit_str}
    # enable_thinking=True: hybrid templates default it OFF (Gemma4 —
    # its 2026-08-24 "think arm" was a silent prefill duplicate, n_think=0
    # for all 3138 items); Qwen3-family defaults ON; templates without the
    # variable ignore it.
    if tok.chat_template is None:          # base models: bare prompt
        s = prompt
    else:
        s = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                    tokenize=False, add_generation_prompt=True,
                                    enable_thinking=enable_thinking)
    ids = tok(s, add_special_tokens=tok.chat_template is None,
              return_tensors="pt").input_ids.to(device)
    if seed is not None:
        torch.manual_seed(seed)
    sample_kw = (dict(do_sample=True, temperature=temperature, top_p=0.95)
                 if temperature else dict(do_sample=False))
    out = model.generate(ids, max_new_tokens=max_new,
                         output_scores=True, return_dict_in_generate=True,
                         pad_token_id=tok.eos_token_id, **sample_kw)
    seq = out.sequences[0, ids.shape[1]:].tolist()
    text = tok.decode(seq, skip_special_tokens=False)
    eos_ids = set(getattr(model.generation_config, "eos_token_id", None) or [])
    if isinstance(model.generation_config.eos_token_id, int):
        eos_ids = {model.generation_config.eos_token_id}
    eos_ids.add(tok.eos_token_id)
    finished = len(seq) < max_new or seq[-1] in eos_ids
    close = max((text.rfind(m) for m in CLOSE_MARKERS), default=-1)
    meta = {"n_gen": len(seq), "finished": bool(finished),
            "closed": bool(close >= 0), "forced": False}
    scores = out.scores
    if force_close and not finished and close < 0:
        # budget forcing: end the reasoning ourselves and read the decision
        fc = force_close_string(tok, model_name)
        ids2 = torch.cat([ids, torch.tensor([seq], device=device),
                          tok(fc, add_special_tokens=False,
                              return_tensors="pt").input_ids.to(device)], 1)
        out2 = model.generate(ids2, max_new_tokens=8, do_sample=False,
                              output_scores=True, return_dict_in_generate=True,
                              pad_token_id=tok.eos_token_id)
        seq2 = out2.sequences[0, ids2.shape[1]:].tolist()
        text = text + fc + tok.decode(seq2, skip_special_tokens=False)
        hits2 = [i for i, t in enumerate(seq2) if t in dt]
        meta["forced"] = True
        if not hits2:
            return None, None, len(seq), text, meta
        step = hits2[0]
        logits = out2.scores[step][0].float()
        probs = torch.softmax(logits, dim=-1)
        dist = {}
        for tid, d in dt.items():
            dist[d] = dist.get(d, 0.0) + probs[tid].item()
        tot = sum(dist.values())
        dist = {k: v / tot for k, v in dist.items()}
        ent = -sum(p * math.log(p) for p in dist.values() if p > 0)
        return dist, ent, len(seq), text, meta
    hits = [i for i, t in enumerate(seq) if t in dt]
    if close >= 0:
        after = len(tok(text[:close], add_special_tokens=False).input_ids)
        post = [i for i in hits if i >= after]
        hits = post or hits
    if not hits:
        return None, None, len(seq), text, meta
    step = hits[-1] if close < 0 else hits[0]   # first digit AFTER a close; last mention when unclosed
    logits = scores[step][0].float()
    probs = torch.softmax(logits, dim=-1)
    dist = {}
    for tid, d in dt.items():
        dist[d] = dist.get(d, 0.0) + probs[tid].item()
    tot = sum(dist.values())
    dist = {k: v / tot for k, v in dist.items()}
    ent = -sum(p * math.log(p) for p in dist.values() if p > 0)
    return dist, ent, step, text, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--full", action="store_true",
                    help="all 523 adjectives instead of the smoke set")
    ap.add_argument("--think", action="store_true",
                    help="thinking arm: reason first, read digit dist at the "
                         "post-deliberation decision point")
    ap.add_argument("--think-mc", type=int, default=0, metavar="K",
                    help="MC-over-chains arm: K sampled trajectories per "
                         "item, per-path digit dist at each decision point "
                         "(marginal = mean of dists; implies thinking)")
    ap.add_argument("--mc-temp", type=float, default=0.6)
    ap.add_argument("--framings", nargs="+", default=None,
                    help="subset of framings to run (default: all six)")
    ap.add_argument("--max-new", type=int, default=384,
                    help="think arms: generation budget (tokens)")
    ap.add_argument("--force-close", action="store_true",
                    help="think arms: if the cap is hit mid-reasoning, append "
                         "the model's close marker and read the forced answer")
    ap.add_argument("--backfill", action="store_true",
                    help="if the output exists, load it and run ONLY the "
                         "adjectives it lacks (525 extension), then rewrite")
    args = ap.parse_args()

    # canonical 525 list (Inspirational/Insensitive reinstated 2026-08-14;
    # the human-side columns were reverse-coded in the deposit, now un-flipped)
    from extract_adjectives import load_adjectives
    all_adjs = sorted({a.lower() for a in load_adjectives()})
    if args.full:
        adjs = all_adjs
    else:
        adjs = [a for a in SMOKE_ADJS if a in all_adjs]
        dropped = [a for a in SMOKE_ADJS if a not in all_adjs]
        if dropped:
            print(f"not in 523, dropped: {dropped}")

    os.makedirs(OUT_DIR, exist_ok=True)
    tag = ("full" if args.full else "smoke") + ("_think" if args.think else "")
    if args.think_mc:
        tag = ("full" if args.full else "smoke") + f"_thinkmc{args.think_mc}"
    if args.force_close:
        tag += "_fc"
    if args.max_new != 384:
        tag += f"_b{args.max_new}"
    out = f"{OUT_DIR}/{args.model.replace('/', '_')}_self_{tag}.json"
    part = out + ".part"
    results = {}
    if os.path.exists(out):
        if not args.backfill:
            print(f"[skip] {out} exists")
            return
        results = json.load(open(out))["results"]
        missing = {f: [a for a in adjs if a not in results.get(f, {})]
                   for f in results}
        print(f"[backfill] {out}: missing per framing "
              f"{ {f: len(v) for f, v in missing.items()} }")
        if not any(missing.values()):
            print("[backfill] nothing to do")
            return
    if os.path.exists(part):
        results = json.load(open(part))["results"]
        print(f"resuming from {part} ({list(results)} done)")

    model, tok, device = hf.load_model(args.model, dtype=torch.bfloat16)
    run_framings = {f: t for f, t in FRAMINGS.items()
                    if args.framings is None or f in args.framings}
    for fname, template in run_framings.items():
        if fname in results and len(results[fname]) == len(adjs):
            continue
        results.setdefault(fname, {})
        for a in adjs:
            if a in results[fname]:
                continue
            if fname == "pda":
                prompt = PDA_SCALE.format(adj=a)
            else:
                prompt = AGREE_SCALE.format(statement=template.format(adj=a))
            # base models ship no chat template — probe them bare (W15 §3
            # convention); tuned models keep the templated path they were
            # measured with, so cohort numbers stay comparable
            if args.think_mc:
                samples = []
                import zlib
                for k in range(args.think_mc):
                    dist, ent, nthink, text, meta = think_distribution(
                        model, tok, prompt, device, max_new=args.max_new,
                        temperature=args.mc_temp,
                        seed=1000 * k + zlib.crc32(a.encode()) % 997,
                        force_close=args.force_close, model_name=args.model)
                    samples.append(
                        {"ev": ev_of(dist) if dist else None,
                         "entropy": ent, "dist": dist, "n_think": nthink,
                         **meta} | ({} if dist else {"tail": text[-200:]}))
                evs_ = [s["ev"] for s in samples if s["ev"] is not None]
                ents_ = [s["entropy"] for s in samples
                         if s["entropy"] is not None]
                mu = sum(evs_) / len(evs_) if evs_ else None
                results[fname][a] = {
                    "samples": samples,
                    "ev": mu,
                    "entropy": sum(ents_) / len(ents_) if ents_ else None,
                    "ev_path_sd": (math.sqrt(sum((e - mu) ** 2 for e in evs_)
                                             / (len(evs_) - 1))
                                   if len(evs_) > 1 else None)}
                if len(results[fname]) % 20 == 0:
                    with open(part, "w") as f:
                        json.dump({"model": args.model,
                                   "results": results}, f)
            elif args.think:
                dist, ent, nthink, text, meta = think_distribution(
                    model, tok, prompt, device, max_new=args.max_new,
                    force_close=args.force_close, model_name=args.model)
                if dist is None:
                    results[fname][a] = {"ev": None, "entropy": None,
                                         "n_think": nthink, "text": text,
                                         **meta}
                    continue
                results[fname][a] = {"ev": ev_of(dist), "entropy": ent,
                                     "dist": dist, "n_think": nthink,
                                     "text": text, **meta}
                if len(results[fname]) % 20 == 0:   # watchdog-reset insurance
                    with open(part, "w") as f:
                        json.dump({"model": args.model,
                                   "results": results}, f)
            else:
                dist, _, ent, mass = hf.likert_distribution(
                    model, tok, prompt, device, digits=DIGITS,
                    use_chat_template=tok.chat_template is not None,
                    return_mass=True)
                if mass >= MASS_FLOOR:
                    results[fname][a] = {"ev": ev_of(dist), "entropy": ent,
                                         "dist": dist, "digit_mass": mass,
                                         "read": "prefill"}
                else:
                    # the model isn't answering with a digit at the read
                    # position (base models, refusers, formatters): fall
                    # back to a short generation and read the first digit
                    d2, e2, step, text, meta = think_distribution(
                        model, tok, prompt, device, max_new=16,
                        enable_thinking=False, model_name=args.model)
                    if d2 is None:
                        results[fname][a] = {"ev": None, "entropy": None,
                                             "dist": None, "digit_mass": mass,
                                             "read": "none", "text": text}
                    else:
                        results[fname][a] = {"ev": ev_of(d2), "entropy": e2,
                                             "dist": d2, "digit_mass": mass,
                                             "read": "generated",
                                             "gen_step": step, "text": text}
        evs = [results[fname][a]["ev"] for a in adjs
               if results[fname][a]["ev"] is not None]
        ents = [results[fname][a]["entropy"] for a in adjs
                if results[fname][a]["entropy"] is not None]
        if not evs:
            print(f"{fname:>10}: no parsable ratings", flush=True)
            continue
        rated = [a for a in adjs if results[fname][a]["ev"] is not None]
        top = sorted(rated, key=lambda a: -results[fname][a]["ev"])[:4]
        bot = sorted(rated, key=lambda a: results[fname][a]["ev"])[:4]
        print(f"{fname:>10}: mean EV {sum(evs)/len(evs):.2f}  "
              f"mean H {sum(ents)/len(ents):.2f}  "
              f"top {top}  bottom {bot}", flush=True)
        with open(part, "w") as f:
            json.dump({"model": args.model, "results": results}, f)

    with open(out, "w") as f:
        import transformers
        json.dump({"model": args.model, "framings": FRAMINGS,
                   "transformers_version": transformers.__version__,
                   "agree_scale": AGREE_SCALE, "pda_scale": PDA_SCALE,
                   "adjectives": adjs, "results": results}, f, indent=1)
    os.remove(part) if os.path.exists(part) else None
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
