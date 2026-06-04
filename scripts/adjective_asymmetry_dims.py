#!/usr/bin/env python3
"""Is valence THE dimension of the tom_likely asymmetry? Secondary dims? (W16 §5)

The directional asymmetry D[A,B] = P(B|A) - P(A|B) is an ANTISYMMETRIC flow.
Helmholtz-Hodge: D = gradient(potential) + curl. The gradient part is
G[A,B] = p[A] - p[B] for a per-word potential p (a transitive ranking); the
least-squares p is the row-mean of D. The residual R = D - G is the curl
(non-transitive cycles).

  - gradient fraction high  => the asymmetry is essentially a 1-D ranking
  - p correlates with valence => that 1-D is valence (primary dimension)
  - residual-of-p-after-valence + curl => SECONDARY dimensions live here

Valence proxies (independent of the judgment asymmetry): the representation's
evaluative axis (rep eval-PC1). Also report the judgment desirability c[X] =
mean ascription rate (col-mean of B) for triangulation (note c enters p=r-c, so
that correlation is partly mechanical).

Usage: PYTHONPATH=scripts .venv/bin/python scripts/adjective_asymmetry_dims.py
"""
import json
import os

import numpy as np

from adjective_geom import model_matrix
from adjective_introspection import GROUPS
from hf_logprobs import MODELS

LABELS = json.load(open("results/adjectives/escs_525pda_corr_raw.json"))["labels"]


def eval_axis():
    """Representation evaluative axis (rep PC1), sign-fixed +=good. Use Qwen7's."""
    base = f"results/adjectives/acts/{MODELS['Qwen7'].replace('/', '_')}__pers"
    p = f"{base}_bare.pt" if os.path.exists(f"{base}_bare.pt") else f"{base}.pt"
    _, C, _, _ = model_matrix(p, np.array(LABELS))
    w, V = np.linalg.eigh(C); v = V[:, np.argsort(np.abs(w))[::-1][0]]
    pos = [LABELS.index(x) for x in GROUPS["pos_eval"]]
    if np.mean(v[pos]) < 0:
        v = -v
    return v


def main():
    val = eval_axis()
    pc2 = None
    for short in ("Qwen7", "Gemma12"):
        z = np.load(f"results/adjectives/introspect_full/{short}_tom_likely_dir.npz",
                    allow_pickle=True)
        adj = list(z["adjectives"]); B = z["B"].astype(float)
        ri = [adj.index(w) for w in LABELS]; B = B[np.ix_(ri, ri)]
        n = len(LABELS); off = ~np.eye(n, dtype=bool)
        D = np.where(off, B - B.T, np.nan)
        p = np.nanmean(D, axis=1)                       # potential
        G = p[:, None] - p[None, :]
        dv, gv = D[off], G[off]
        frac_grad = 1 - np.nansum((dv - gv) ** 2) / np.nansum(dv ** 2)
        c = np.nanmean(B, axis=0)                        # desirability (col mean)
        def corr(a, b):
            m = ~(np.isnan(a) | np.isnan(b)); return float(np.corrcoef(a[m], b[m])[0, 1])
        r_val = corr(p, val); r_des = corr(p, c)
        # residual potential after removing valence (linear)
        m = ~np.isnan(p)
        beta = np.polyfit(val[m], p[m], 1)
        p_res = p - np.polyval(beta, val)
        frac_val = r_val ** 2                            # share of p variance ~ valence
        print(f"\n===== {short} =====")
        print(f"  gradient (transitive-ranking) fraction of asymmetry: {frac_grad:.3f}"
              f"   (curl/cyclic = {1-frac_grad:.3f})")
        print(f"  potential p ~ valence (rep eval-PC1):  r = {r_val:+.3f}  (r^2 {frac_val:.2f})")
        print(f"  potential p ~ desirability (col-mean): r = {r_des:+.3f}  (partly mechanical)")
        order = np.argsort(p)
        print(f"  most EXCLUSIONARY anchors (low p, exclude their opposites):")
        print(f"    {', '.join(LABELS[i] for i in order[:10])}")
        print(f"  most TOLERANT anchors (high p):")
        print(f"    {', '.join(LABELS[i] for i in order[::-1][:10])}")
        ro = np.argsort(p_res)
        print(f"  SECONDARY (residual potential after valence) — low end:")
        print(f"    {', '.join(LABELS[i] for i in ro[:10])}")
        print(f"  SECONDARY — high end:")
        print(f"    {', '.join(LABELS[i] for i in ro[::-1][:10])}")


if __name__ == "__main__":
    main()
