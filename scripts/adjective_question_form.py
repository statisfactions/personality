#!/usr/bin/env python3
"""Sensitivity analysis on the QUESTION FORM before the full ToM run (W16 §1).

The semantic scale floors (76-79% single-pole) and ToM is better but still 54%
floored. Root cause: both are UNIPOLAR (1=different/inaccurate ... 7=same/accurate),
so *opposite* and *unrelated* both map to the floor. An instrument that measures
the full space should put antonyms low, unrelated in the MIDDLE, synonyms high —
i.e. a BIPOLAR scale. This sweeps forms and reports, for each:

  spread (on a random full-vocab sample): mean EV, std EV, %floor, %ceiling, entropy
  separation (on the 26-word reference set, via GROUPS):
      SYN  = within-pole (synonyms)        -- want HIGH
      UNR  = evaluative x neutral (Tall...) -- want MIDDLE (only a bipolar scale can)
      ANT  = pos_eval x neg_eval + warm x antagonism (antonyms) -- want LOW
  good instrument: large std, low floor+ceiling, and SYN > UNR > ANT with real gaps.

Usage: PYTHONPATH=scripts .venv/bin/python scripts/adjective_question_form.py --model Qwen7 --k 200
"""
import argparse

import numpy as np

from hf_logprobs import load_model
from adjective_introspection import ADJ, GROUPS
from adjective_judge_full import load_adjectives, digit_token_ids, batch_ev_entropy
from adjective_judge_symmetry import sample_pairs

VARIANTS = {
    "sem-uni": (  # baseline semantic (unipolar)
        "How similar in meaning are these two words, as descriptions of a person?\n"
        "Word 1: {a}\nWord 2: {b}\n"
        "Answer with one number from 1 to 7, where 1 = completely different in "
        "meaning and 7 = nearly the same in meaning.\nNumber:"),
    "sem-bi": (   # bipolar meaning: opposite / unrelated / same
        "How are these two words related, as descriptions of a person?\n"
        "Word 1: {a}\nWord 2: {b}\n"
        "Answer with one number from 1 to 7, where 1 = opposite in meaning, "
        "4 = unrelated, and 7 = nearly the same in meaning.\nNumber:"),
    "tom-acc": (  # baseline ToM (unipolar)
        "Consider a person who is very {a}.\n"
        "How accurately does the word \"{b}\" describe this person?\n"
        "Answer with one number from 1 to 7, where 1 = very inaccurate and "
        "7 = very accurate.\nNumber:"),
    "tom-bi": (   # bipolar ToM: opposite / unrelated / also true
        "Consider a person who is very {a}.\n"
        "Is \"{b}\" the opposite of this person, unrelated to them, or also true "
        "of them?\nAnswer with one number from 1 to 7, where 1 = the opposite, "
        "4 = unrelated, and 7 = also clearly true.\nNumber:"),
    "tom-likely": (  # dispositional probability
        "Consider a person who is very {a}.\n"
        "How likely is this person to also be {b}?\n"
        "Answer with one number from 1 to 7, where 1 = very unlikely and "
        "7 = very likely.\nNumber:"),
}

GI = {g: [ADJ.index(w) for w in GROUPS[g]] for g in GROUPS}


def block(mat, ga, gb):
    sub = mat[np.ix_(GI[ga], GI[gb])]
    return float(np.nanmean(sub[~np.isnan(sub)]))


def run(model, tok, device, template, pairs, words, digit_ids, bs=48):
    texts = [tok.apply_chat_template(
        [{"role": "user", "content": template.format(a=words[i].lower(), b=words[j].lower())}],
        tokenize=False, add_generation_prompt=True) for i, j in pairs]
    ev, H = [], []
    for s in range(0, len(texts), bs):
        e, h = batch_ev_entropy(model, tok, device, texts[s:s + bs], digit_ids)
        ev.append(e); H.append(h)
    return np.concatenate(ev), np.concatenate(H)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["Qwen7"])
    ap.add_argument("--forms", nargs="+", default=list(VARIANTS),
                    help="subset of question forms to sweep")
    ap.add_argument("--k", type=int, default=200)
    args = ap.parse_args()
    forms = {k: VARIANTS[k] for k in args.forms}

    full = load_adjectives()
    rand_pairs = sample_pairs(len(full), args.k)
    ref_pairs = [(i, j) for i in range(len(ADJ)) for j in range(i + 1, len(ADJ))]

    for short in args.models:
        model, tok, device = load_model(short)
        tok.padding_side = "left"
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        digit_ids = digit_token_ids(tok)

        print(f"\n===== {short}: question-form sweep ({args.k} random + 26-word ref) =====")
        print(f"{'form':<11}{'meanEV':>7}{'stdEV':>7}{'%flr':>6}{'%ceil':>6}{'meanH':>7}"
              f"  | {'SYN':>4} {'UNR':>4} {'ANT':>4}  {'spread':>7}")
        print("-" * 78)
        for name, tmpl in forms.items():
            ev, H = run(model, tok, device, tmpl, rand_pairs, full, digit_ids)
            rev, _ = run(model, tok, device, tmpl, ref_pairs, ADJ, digit_ids)
            M = np.full((len(ADJ), len(ADJ)), np.nan)
            for (i, j), v in zip(ref_pairs, rev):
                M[i, j] = M[j, i] = v
            syn = np.mean([block(M, g, g) for g in ("pos_eval", "neg_eval", "warm",
                                                    "antagonism", "distress", "intellect")])
            unr = np.mean([block(M, g, "neutral") for g in ("pos_eval", "neg_eval", "warm",
                                                            "intellect")])
            ant = np.mean([block(M, "pos_eval", "neg_eval"), block(M, "warm", "antagonism")])
            floor = 100 * np.mean(ev <= 1.1); ceil = 100 * np.mean(ev >= 6.9)
            print(f"{name:<11}{ev.mean():>7.2f}{ev.std():>7.2f}{floor:>5.0f}%{ceil:>5.0f}%"
                  f"{H.mean():>7.3f}  | {syn:>4.1f} {unr:>4.1f} {ant:>4.1f}  {syn-ant:>7.1f}")
        import gc, torch
        del model, tok; gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    print("\nwant: large stdEV, low %flr+%ceil, and SYN > UNR > ANT with UNR near the "
          "middle (only a bipolar scale can place 'unrelated' at ~4).")


if __name__ == "__main__":
    main()
