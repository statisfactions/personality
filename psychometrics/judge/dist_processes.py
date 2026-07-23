#!/usr/bin/env python3
"""Response-PROCESS products from the full JUDGE raw distributions (Report I §7).

`convert_full_dists.py` recomputes the two moments (B, Hent) and a shape census from
`dists`. This script goes further — it interrogates the 7-category distributions *as
response processes*, producing things the two moments cannot express:

  proc_content.csv    per model: the decomposition of a rater's total rating
                      variability H(marginal) into content (mutual information the
                      adjective pair carries about the rating) + within-judgment
                      hedging (mean per-cell entropy). MI = H(marginal) − mean Hent.
                      Also decisiveness (mean max-category prob), endpoint mass, and
                      the fraction of "coin-flip" cells (balanced 1-vs-7).  + marginal.
  proc_signature.csv  per model × EV-bin: the mean 7-category distribution of judgments
                      whose expected rating falls in that bin — the "response-process
                      signature". A graded rater slides a unimodal peak 1→7; a bimodal
                      rater forks mass to the endpoints at mid-EV; a flat/digit-prior
                      rater emits a fixed shape regardless of EV.
  proc_modepairs.csv  per model: census of the (lower, higher) mode pair for every
                      genuinely multi-modal cell (≥2 prominent local maxima) — rgb's
                      "commit-vs-hedge vs no-vs-yes" distinction, made a distribution.
  proc_endpoint_examples.csv  balanced 1-vs-7 "coin-flip" exemplars per relevant model
                      (the ill-posed category-mismatch cells rgb flagged).

Source: data/dists_full/judge_dists/*_tom_likely_dists_full.npz  (gitignored).
All outputs are small per-model summaries; the heavy npz stay on disk. Diagonal NaN.
"""
import glob, os
import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
SRC = os.path.join(HERE, "data", "dists_full", "judge_dists")
OUT = os.path.join(HERE, "data")
CANON = {"gemma3": "Gemma", "llama3.2": "Llama", "phi4": "Phi4", "qwen2.5": "Qwen"}
K = np.arange(1, 8, dtype=np.float64)
EV_EDGES = np.arange(1.0, 7.0001, 0.25)          # 24 EV bins for the signature
EV_CENT = (EV_EDGES[:-1] + EV_EDGES[1:]) / 2


def ent(P, axis=-1):
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.where(P > 0, P * np.log(P), 0.0).sum(axis)


def local_maxima(P, thr=0.15):
    """Boolean (N,7): category k is a prominent local max (mass>=thr, > neighbours)."""
    left = np.ones_like(P, bool); right = np.ones_like(P, bool)
    left[:, 1:] = P[:, 1:] > P[:, :-1]
    right[:, :-1] = P[:, :-1] > P[:, 1:]          # strict; ties are not maxima on the right
    return (P >= thr) & left & right


def main():
    from convert_full_dists import META  # reuse canonical display/family map
    paths = sorted(p for p in glob.glob(os.path.join(SRC, "*_tom_likely_dists_full.npz"))
                   if not os.path.basename(p).startswith("._"))
    assert paths, f"no npz in {SRC}"
    order_key = lambda s: (META[s][1], META[s][2])

    content, signature, modepairs, examples = [], [], [], []
    for p in paths:
        stem = os.path.basename(p).replace("_tom_likely_dists_full.npz", "")
        s = CANON.get(stem, stem)
        disp, fam, _ = META[s]
        z = np.load(p, allow_pickle=True)
        D = np.asarray(z["dists"], dtype=np.float64)      # (523,523,7) diag NaN
        adj = [str(a) for a in z["adjectives"]]
        n = D.shape[0]
        off = ~np.eye(n, dtype=bool)
        P = D[off]                                        # (273006, 7)
        P = P[~np.isnan(P).any(1)]                        # drop any stray NaN rows
        ev = P @ K

        # ---- content decomposition (response-process information budget) ----
        marg = P.mean(0)
        H_marg = float(ent(marg))
        mean_within = float(ent(P, 1).mean())            # = mean Hent
        mi = H_marg - mean_within                        # mutual info pair;rating (nats)
        row = {
            "model": s, "display": disp, "family": fam,
            "H_marginal": H_marg, "mean_within_entropy": mean_within,
            "MI_nats": mi, "content_frac": mi / H_marg if H_marg > 0 else np.nan,
            "maxprob_mean": float(P.max(1).mean()),
            "endpoint_mass_mean": float((P[:, 0] + P[:, 6]).mean()),
            "frac_coinflip": float(np.mean(np.minimum(P[:, 0], P[:, 6]) >= 0.25)),
        }
        row.update({f"marg{c}": float(marg[c - 1]) for c in range(1, 8)})
        content.append(row)

        # ---- response-process signature: mean pmf per EV bin ----
        b = np.clip(np.digitize(ev, EV_EDGES) - 1, 0, len(EV_CENT) - 1)
        for bi in range(len(EV_CENT)):
            m = b == bi
            if not m.any():
                continue
            mp = P[m].mean(0)
            for c in range(1, 8):
                signature.append((s, disp, round(float(EV_CENT[bi]), 3), c,
                                  float(mp[c - 1]), int(m.sum())))

        # ---- mode-pair census over genuinely multi-modal cells ----
        lm = local_maxima(P)
        nlm = lm.sum(1)
        sel = nlm >= 2
        if sel.any():
            Psel = P[sel]; lmsel = lm[sel]
            masked = np.where(lmsel, Psel, -1.0)
            top2 = np.argsort(-masked, 1)[:, :2]          # two biggest local maxima (0-based cat)
            lo = np.minimum(top2[:, 0], top2[:, 1]) + 1
            hi = np.maximum(top2[:, 0], top2[:, 1]) + 1
            pairs, counts = np.unique(np.stack([lo, hi], 1), axis=0, return_counts=True)
            nb = int(sel.sum())
            for (a, c), ct in zip(pairs, counts):
                modepairs.append((s, disp, int(a), int(c), int(ct), ct / nb, nb))

        # ---- balanced endpoint "coin-flip" exemplars (ill-posed cells) ----
        # rank off-diag cells by min(p1,p7); recover their (i,j) adjectives.
        oi, oj = np.where(off)
        keep = ~np.isnan(D[off]).any(1)
        oi, oj = oi[keep], oj[keep]
        bal = np.minimum(P[:, 0], P[:, 6])
        if bal.max() >= 0.30:                             # only models that actually do this
            top = np.argsort(-bal)[:6]
            for t in top:
                if bal[t] < 0.30:
                    continue
                examples.append({
                    "model": s, "display": disp, "adj_i": adj[oi[t]], "adj_j": adj[oj[t]],
                    "ev": round(float(ev[t]), 3), "min_end": round(float(bal[t]), 3),
                    **{f"p{c}": round(float(P[t, c - 1]), 4) for c in range(1, 8)}})

    ordf = {s: k for k, s in enumerate(sorted({r["model"] for r in content}, key=order_key))}
    def sort_(df): return df.assign(_o=df["model"].map(ordf)).sort_values("_o").drop(columns="_o")

    sort_(pd.DataFrame(content)).to_csv(os.path.join(OUT, "proc_content.csv"), index=False)
    sort_(pd.DataFrame(signature, columns=["model", "display", "ev_bin", "cat", "mean_mass", "n_cells"])
          ).to_csv(os.path.join(OUT, "proc_signature.csv"), index=False)
    sort_(pd.DataFrame(modepairs, columns=["model", "display", "lo", "hi", "n", "frac", "n_bimodal"])
          ).to_csv(os.path.join(OUT, "proc_modepairs.csv"), index=False)
    pd.DataFrame(examples).to_csv(os.path.join(OUT, "proc_endpoint_examples.csv"), index=False)
    print(f"{len(content)} models -> proc_content / proc_signature / proc_modepairs / proc_endpoint_examples")


if __name__ == "__main__":
    main()
