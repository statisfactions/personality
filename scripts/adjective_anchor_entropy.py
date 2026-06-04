#!/usr/bin/env python3
"""Level-controlled anchor entropy: variance/evidence vs valence (W16 §5).

The directional asymmetry P(neg|pos) > P(pos|neg) (net of base rates and "very")
is robust, but its MECHANISM is open: valence/structural ("bad excludes good")
vs variance/evidence ("good people carry wider behavioral variance → less reliably
ascribable"). The clean, confound-free test of the variance account is the model's
OWN posterior width — the per-judgment ENTROPY of P(B | person is X) — NOT a verbal
"how predictable" probe (which measures the target word's denotation).

Prediction: variance/evidence => judgments about GOOD anchors are MORE uncertain
(higher entropy) than about BAD anchors; valence/structural => no entropy asymmetry.

Confound to control: positivity bias gives good anchors higher EV, and entropy is
an inverted-U in EV (peaks mid-scale, ~0 at the extremes), so a raw Hgood-vs-Hbad
gap could be pure level. So compare entropy at MATCHED EV (binned), and report the
EV-residual gap (entropy minus the pooled entropy-vs-EV curve).

Usage: PYTHONPATH=scripts .venv/bin/python scripts/adjective_anchor_entropy.py
"""
import json

import numpy as np

from adjective_introspection import GROUPS

LABELS = json.load(open("results/adjectives/escs_525pda_corr_raw.json"))["labels"]


def collect(short):
    z = np.load(f"results/adjectives/introspect_full/{short}_tom_likely_dir.npz",
                allow_pickle=True)
    adj = list(z["adjectives"]); B = z["B"].astype(float); H = z["Hent"].astype(float)
    ri = [adj.index(w) for w in LABELS]
    B = B[np.ix_(ri, ri)]; H = H[np.ix_(ri, ri)]
    idx = {w: k for k, w in enumerate(LABELS)}
    pos = [idx[w] for w in GROUPS["pos_eval"]]; neg = [idx[w] for w in GROUPS["neg_eval"]]
    ev, h, grp = [], [], []
    for name, anchors in (("good", pos), ("bad", neg)):
        for a in anchors:
            for b in range(len(LABELS)):
                if b == a or np.isnan(B[a, b]) or np.isnan(H[a, b]):
                    continue
                ev.append(B[a, b]); h.append(H[a, b]); grp.append(name)
    return np.array(ev), np.array(h), np.array(grp)


def main():
    for short in ("Qwen7", "Gemma12"):
        ev, h, grp = collect(short)
        g = grp == "good"; b = grp == "bad"
        # pooled entropy-vs-EV curve (0.5-wide bins), residual control
        edges = np.arange(1, 7.001, 0.5)
        bi = np.digitize(ev, edges)
        pred = np.empty_like(h)
        for k in np.unique(bi):
            m = bi == k
            pred[m] = h[m].mean()
        resid = h - pred
        print(f"\n===== {short} =====")
        print(f"  raw entropy:           Hgood {h[g].mean():.3f}  Hbad {h[b].mean():.3f}"
              f"   gap {h[g].mean()-h[b].mean():+.3f}")
        print(f"  EV-controlled residual: good {resid[g].mean():+.3f}  bad {resid[b].mean():+.3f}"
              f"   gap {resid[g].mean()-resid[b].mean():+.3f}")
        print(f"  EV means:               good {ev[g].mean():.2f}  bad {ev[b].mean():.2f}")
        print(f"  {'EV bin':<12}{'n_good':>7}{'n_bad':>7}{'Hgood':>8}{'Hbad':>8}{'gap':>8}")
        for k in sorted(np.unique(bi)):
            m = bi == k
            hg = h[m & g]; hb = h[m & b]
            if len(hg) >= 8 and len(hb) >= 8:    # overlap region with power
                lo = edges[k-1]
                print(f"  [{lo:.1f}-{lo+0.5:.1f}){'':<4}{len(hg):>7}{len(hb):>7}"
                      f"{hg.mean():>8.3f}{hb.mean():>8.3f}{hg.mean()-hb.mean():>+8.3f}")
        print("  (variance/evidence => good>bad within bins & positive residual gap; "
              "valence => ~0)")


if __name__ == "__main__":
    main()
