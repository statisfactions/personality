"""Task-conditional sharpness probe: is Llama's flatness (Gemma's peakedness)
constitutional or task-conditional? (W18 §4)

Three question classes, digit-answer format matching tom_likely:
  FACTUAL    — verifiable single-digit answer (sharp = correct behavior)
  IMPOSSIBLE — undeterminable (fair die roll): uniform = correct
  SUBJECTIVE — tom_likely Likert pairs, weak/no true association

Per model x class: mean digit entropy (nats, and as % of uniform), argmax
mass, argmax accuracy on FACTUAL. Finding (2026-07-12): every model is sharp
and accurate on FACTUAL and hedges on IMPOSSIBLE — the family difference in
JUDGE entropy is how the model CLASSIFIES the subjective question (gemma
files person-inference under "has an answer" at 8% of uniform; llama files
it under "undeterminable" at 56%), not a global entropy policy.

Usage: PYTHONPATH=scripts python scripts/entropy_probe.py
"""
import numpy as np
import torch

import hf_logprobs as hf

FACTUAL = [
    ("How many legs does a spider have?", "8"),
    ("How many days are there in a week?", "7"),
    ("What is 2 + 3?", "5"),
    ("How many sides does a triangle have?", "3"),
    ("What is 9 minus 5?", "4"),
    ("How many letters are in the word 'cat'?", "3"),
]
IMPOSSIBLE = [
    "I just rolled a fair six-sided die. What number did it show?",
    "I am thinking of a number between 1 and 6. What is it?",
    "A stranger picked their favorite number between 1 and 6. What is it?",
    "I shuffled cards numbered 1 to 6 and drew one. Which number did I draw?",
]
LIKERT = ("Consider a person who is very {a}.\n"
          "How likely is this person to also be {b}?\n"
          "Answer with one number from 1 to 7, where 1 = very unlikely and "
          "7 = very likely.\nNumber:")
SUBJECTIVE = [("organized", "friendly"), ("curious", "talkative"),
              ("tall", "cheerful"), ("quiet", "creative"),
              ("polite", "athletic"), ("thrifty", "imaginative")]


def probe(model, tok, prompt, device, digits):
    d, _, ent = hf.likert_distribution(model, tok, prompt, device, digits=digits)
    tot = sum(d.values())
    p = {k: v / tot for k, v in d.items()}
    top = max(p, key=lambda k: p[k])
    return ent, top, p[top]


def main():
    for name in ["gemma3", "llama3.2", "phi4", "qwen2.5"]:
        model, tok, device = hf.load_model(name, dtype=torch.bfloat16)
        rows = {}

        ents, accs, mass = [], [], []
        for q, ans in FACTUAL:
            e, top, pm = probe(model, tok, q + "\nAnswer with one number.\nNumber:",
                               device, tuple("123456789"))
            ents.append(e); accs.append(top == ans); mass.append(pm)
        rows["FACTUAL"] = (np.mean(ents), np.mean(mass), np.mean(accs))

        ents, mass = [], []
        for q in IMPOSSIBLE:
            e, top, pm = probe(model, tok, q + "\nAnswer with one number.\nNumber:",
                               device, tuple("123456"))
            ents.append(e); mass.append(pm)
        rows["IMPOSSIBLE"] = (np.mean(ents), np.mean(mass), None)

        ents, mass = [], []
        for a, b in SUBJECTIVE:
            e, top, pm = probe(model, tok, LIKERT.format(a=a, b=b),
                               device, tuple("1234567"))
            ents.append(e); mass.append(pm)
        rows["SUBJECTIVE"] = (np.mean(ents), np.mean(mass), None)

        umax = {"FACTUAL": np.log(9), "IMPOSSIBLE": np.log(6),
                "SUBJECTIVE": np.log(7)}
        print(f"\n=== {name} ===")
        for k, (e, pm, acc) in rows.items():
            extra = f"  acc={acc:.2f}" if acc is not None else ""
            print(f"  {k:11s} entropy {e:.3f} ({e/umax[k]:.0%} of uniform)  "
                  f"argmax-mass {pm:.2f}{extra}")
        del model
        torch.mps.empty_cache()


if __name__ == "__main__":
    main()
