#!/usr/bin/env python3
"""W13 §8.3: compare the no-persona assistant self-rating across English and
Mandarin IPIP-NEO-120 for the cohort + Aya.

Both languages use the matched 120-item form (instruments/ipip120_english.json
and instruments/ipip120_mandarin.json), identical facet/reverse-key structure,
so per-trait means are directly comparable. The only manipulation is the
language of items + prompt scaffolding.

Reports per model:
  - English vs Mandarin trait means (EV) for N/E/O/A/C
  - per-trait Δ (Mandarin − English) and mean |Δ|
  - mean response entropy per language (the "does the model have an opinion"
    filter — high entropy ≈ near-uniform ≈ no coherent self-rating)

Also, as a side control (question 3): for any model with a bare-text IPIP-300
run cached, compute the chat-vs-raw trait Δ and compare its magnitude to the
English-vs-Mandarin Δ — is language a bigger perturbation than prompt format?

Usage: PYTHONPATH=scripts .venv/bin/python scripts/compare_en_zh_persona.py
"""

import json
from pathlib import Path

SURVEYS = Path("results/surveys")
MODELS = ["Qwen", "Qwen7", "Qwen32", "Aya", "Gemma", "Gemma12", "Gemma27",
          "Gemma4", "Phi4", "Llama", "Llama8"]
TRAITS_120 = ["IPIP120E-N", "IPIP120E-E", "IPIP120E-O", "IPIP120E-A", "IPIP120E-C"]
TRAITS_120_ZH = ["IPIP120M-N", "IPIP120M-E", "IPIP120M-O", "IPIP120M-A", "IPIP120M-C"]
TRAIT_LBL = ["N", "E", "O", "A", "C"]


def load_scales(path):
    if not path.exists():
        return None
    return json.load(open(path))["scale_scores"]


def ev_vector(scales, scale_ids):
    return [scales[s]["ev_mean"] for s in scale_ids]


def mean_entropy(scales, scale_ids):
    hs = [scales[s]["entropy_mean"] for s in scale_ids if scales[s].get("entropy_mean")]
    return sum(hs) / len(hs) if hs else None


def main():
    print(f"{'model':<10s} | {'lang':<4s} | {'N':>5s} {'E':>5s} {'O':>5s} {'A':>5s} {'C':>5s} | {'meanH':>6s}")
    print("-" * 72)
    summary = []
    for m in MODELS:
        en = load_scales(SURVEYS / f"{m}_ipip120_english.json")
        zh = load_scales(SURVEYS / f"{m}_ipip120_mandarin.json")
        if en is None or zh is None:
            print(f"{m:<10s} | MISSING ({'en' if en is None else ''}{'zh' if zh is None else ''})")
            continue
        en_ev = ev_vector(en, TRAITS_120)
        zh_ev = ev_vector(zh, TRAITS_120_ZH)
        en_h = mean_entropy(en, TRAITS_120)
        zh_h = mean_entropy(zh, TRAITS_120_ZH)
        deltas = [z - e for z, e in zip(zh_ev, en_ev)]
        mean_abs_delta = sum(abs(d) for d in deltas) / len(deltas)

        print(f"{m:<10s} | {'EN':<4s} | " + " ".join(f"{v:>5.2f}" for v in en_ev) + f" | {en_h:>6.3f}")
        print(f"{m:<10s} | {'ZH':<4s} | " + " ".join(f"{v:>5.2f}" for v in zh_ev) + f" | {zh_h:>6.3f}")
        print(f"{'':<10s} | {'Δ':<4s} | " + " ".join(f"{d:>+5.2f}" for d in deltas) +
              f" | mean|Δ|={mean_abs_delta:.3f}")
        print()
        summary.append({
            "model": m, "en_ev": en_ev, "zh_ev": zh_ev, "deltas": deltas,
            "mean_abs_delta": mean_abs_delta, "en_h": en_h, "zh_h": zh_h,
        })

    # Opinion filter: flag models whose mean entropy is high in either language.
    print("=" * 72)
    print("Opinion strength (mean entropy; >0.6 ≈ near-uniform / no coherent self-rating):")
    for s in summary:
        flag_en = " <- weak" if s["en_h"] > 0.6 else ""
        flag_zh = " <- weak" if s["zh_h"] > 0.6 else ""
        print(f"  {s['model']:<10s}: EN H={s['en_h']:.3f}{flag_en:>9s}   ZH H={s['zh_h']:.3f}{flag_zh:>9s}")

    print()
    print("EN↔ZH assistant-persona shift, models with an opinion in BOTH languages:")
    opinionated = [s for s in summary if s["en_h"] < 0.6 and s["zh_h"] < 0.6]
    for s in sorted(opinionated, key=lambda x: -x["mean_abs_delta"]):
        per_trait = "  ".join(f"{lbl}{d:+.2f}" for lbl, d in zip(TRAIT_LBL, s["deltas"]))
        print(f"  {s['model']:<10s}: mean|Δ|={s['mean_abs_delta']:.3f}   [{per_trait}]")
    if opinionated:
        avg = sum(s["mean_abs_delta"] for s in opinionated) / len(opinionated)
        print(f"  ---> cohort-of-opinionated mean|Δ| = {avg:.3f}")

    # Question 3 control: chat-vs-raw, for any model with a bare-text IPIP-300.
    print()
    print("=" * 72)
    print("Question 3 — is EN↔ZH bigger than chat↔raw? (models with bare-text IPIP-300)")
    for m in MODELS:
        chat = load_scales(SURVEYS / f"{m}_ipip300.json")
        raw = load_scales(SURVEYS / f"{m}_ipip300_baretext.json")
        if chat is None or raw is None:
            continue
        ids = ["IPIP300-NEU", "IPIP300-EXT", "IPIP300-OPE", "IPIP300-AGR", "IPIP300-CON"]
        c = ev_vector(chat, ids)
        r = ev_vector(raw, ids)
        fmt_delta = sum(abs(a - b) for a, b in zip(c, r)) / len(c)
        s = next((x for x in summary if x["model"] == m), None)
        lang_delta = s["mean_abs_delta"] if s else float("nan")
        print(f"  {m:<10s}: chat↔raw mean|Δ|={fmt_delta:.3f} (IPIP-300)  "
              f"vs EN↔ZH mean|Δ|={lang_delta:.3f} (IPIP-120)  "
              f"-> language is {lang_delta/fmt_delta:.1f}× format" if fmt_delta else "")


if __name__ == "__main__":
    main()
