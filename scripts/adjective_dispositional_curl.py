#!/usr/bin/env python3
"""(a) prevalence-vs-valence on a dispositional subset + (b) SVD the curl (W16 §5).

(a) The full 525-PDA set is heterogeneous (physical/demographic/status/acute-state/
markedness words inflate the prevalence gradient AND the cycles). Re-run the
Hodge split on a hand-curated DISPOSITIONAL subset and see whether valence rises
relative to prevalence in the gradient.

(b) Name the secondary structure systematically: the curl R = D - grad is a real
antisymmetric (divergence-free) matrix; its SVD pairs singular vectors into
ROTATIONAL PLANES. Show the dominant planes' extreme words and correlate each axis
with valence (rep eval-PC1) and prevalence (ascription base rate).

Usage: PYTHONPATH=scripts .venv/bin/python scripts/adjective_dispositional_curl.py
"""
import json
import os

import numpy as np

from adjective_geom import model_matrix
from adjective_introspection import GROUPS
from hf_logprobs import MODELS

LABELS = json.load(open("results/adjectives/escs_525pda_corr_raw.json"))["labels"]

# Hand-curated non-dispositional words to drop (judgment call; the big offenders are
# unambiguous: physical, demographic, status, acute states, markedness/default).
DROP = {
 # physical / appearance / body / strength / health
 "Muscular","Athletic","Tall","Short","Chubby","Skinny","Tiny","Slim","Thin","Big",
 "Little","Fat","Slender","Handsome","Good-looking","Pretty","Beautiful","Gorgeous",
 "Cute","Attractive","Unattractive","Glamorous","Sexy","Seductive","Graceful",
 "Ungraceful","Clean","Strong","Weak","Healthy","Sickly","Alcoholic",
 # age / sex / demographic
 "Elderly","Old","Young","Youthful","Middle-aged","Middle-class","Feminine","Masculine",
 "Left-handed",
 # disability / medical-cognitive
 "Blind","Disabled","Handicapped","Retarded","Senile",
 # socioeconomic / status
 "Rich","Poor","Wealthy","Prosperous","Well-to-do","Homeless","Employed","Unemployed",
 "Famous","Well-known","Popular","Unpopular","Prominent","Distinguished",
 # acute physical/emotional states ("also be" reads as "become")
 "Awake","Tired","Exhausted","Sleepy","Busy","Rushed","Bored","Surprised","Embarrassed",
 "Ashamed","Humiliated","Heartbroken","Delighted","Disappointed","Frustrated","Annoyed",
 "Irritated","Upset","Worried","Scared","Afraid","Furious","Grumpy","Troubled","Disturbed",
 "Confused","Preoccupied","Encouraged","Laughing","Excited","Pleased","Glad","Satisfied",
 "Grateful","Thankful","Appreciated","Relaxed","Comfortable","Uncomfortable","Alone",
 "Lonely","Lonesome",
 # markedness / default
 "Normal","Abnormal","Ordinary","Average","Familiar","Natural","Plain","Unusual","Strange",
 "Weird",
}


def valence_axis(keep_idx):
    base = f"results/adjectives/acts/{MODELS['Qwen7'].replace('/', '_')}__pers"
    p = f"{base}_bare.pt" if os.path.exists(f"{base}_bare.pt") else f"{base}.pt"
    _, C, _, _ = model_matrix(p, np.array(LABELS))
    C = C[np.ix_(keep_idx, keep_idx)]
    w, V = np.linalg.eigh(C); v = V[:, np.argsort(np.abs(w))[::-1][0]]
    sub = [LABELS[i] for i in keep_idx]
    pos = [sub.index(x) for x in GROUPS["pos_eval"] if x in sub]
    if np.mean(v[pos]) < 0:
        v = -v
    return v


def main():
    keep_idx = [i for i, w in enumerate(LABELS) if w not in DROP]
    sub = [LABELS[i] for i in keep_idx]
    print(f"dispositional subset: {len(sub)} of {len(LABELS)} words "
          f"({len(LABELS)-len(sub)} dropped)")
    val = valence_axis(keep_idx)

    for short in ("Qwen7", "Gemma12"):
        z = np.load(f"results/adjectives/introspect_full/{short}_tom_likely_dir.npz",
                    allow_pickle=True)
        adj = list(z["adjectives"]); B = z["B"].astype(float)
        ri = [adj.index(w) for w in sub]; B = B[np.ix_(ri, ri)]
        n = len(sub)
        D = np.nan_to_num(B - B.T, nan=0.0); np.fill_diagonal(D, 0.0)
        p = D.mean(axis=1); G = p[:, None] - p[None, :]
        off = ~np.eye(n, dtype=bool)
        frac_grad = 1 - (((D - G)[off]) ** 2).sum() / ((D[off]) ** 2).sum()
        c = np.nanmean(B, axis=0)
        def corr(a, b): return float(np.corrcoef(a, b)[0, 1])
        print(f"\n===== {short} (dispositional subset) =====")
        print(f"  (a) gradient fraction {frac_grad:.3f}   "
              f"potential~valence r={corr(p, val):+.3f} (r^2 {corr(p,val)**2:.2f})   "
              f"potential~prevalence r={corr(p, c):+.3f}")

        # (b) SVD the curl R (divergence-free); singular vectors pair into planes
        R = D - G; np.fill_diagonal(R, 0.0)
        U, S, Vt = np.linalg.svd(R)
        sub_arr = np.array(sub)
        def ends(vec, k=7):
            o = np.argsort(vec)
            return (", ".join(sub_arr[o[::-1][:k]]), ", ".join(sub_arr[o[:k]]))
        print(f"  (b) curl singular values (paired -> planes): "
              f"{', '.join(f'{s:.0f}' for s in S[:6])}")
        for plane in range(2):
            for ax in (2 * plane, 2 * plane + 1):
                u = U[:, ax]
                hi, lo = ends(u)
                rv, rc = corr(u, val), corr(u, c)
                print(f"    plane{plane+1} axis{ax%2+1} (val r={rv:+.2f} prev r={rc:+.2f})")
                print(f"      +  {hi}")
                print(f"      -  {lo}")


if __name__ == "__main__":
    main()
