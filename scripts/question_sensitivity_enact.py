"""ENACT question-selection sensitivity (2026-08-22, rgb's request).

The persona vectors are built from 60 rollouts = 12 advice-register
questions x 5 sys templates. How much of each vector — and of the
downstream structure (adjacency, effdim, human-congruence) — depends on
which questions were asked? All from saved per-rollout acts; no GPU.

Per model (mid stored layer):
  A. question split-half (6/6) cosine of persona vectors vs random 30/30
     rollout split (noise floor at matched n)
  B. facet comparison: single-question vectors' cross-question agreement
     vs single-template vectors' cross-template agreement
  C. leave-one-question-out jackknife (worst question)
  D. structure: cross-half adjacency r, effdim (PR) full vs halves,
     44-block human-congruence per half

Usage: PYTHONPATH=scripts python scripts/question_sensitivity_enact.py \
           [--models llama3.2 qwen2.5 ...]
Out:   stdout + results/persona_vectors/question_sensitivity.json
"""
import argparse
import json

import numpy as np
import torch

import adjective_facet_cohort as afc
from extract_persona_vectors import QUESTIONS

PV = "results/persona_vectors"
ALL = ["llama3.2", "qwen2.5", "gemma3", "phi4", "Llama8", "Qwen7",
       "Aya", "Gemma12", "Qwen32", "Gemma27"]
N_SPLITS = 50


def unit(X):
    return X / np.maximum(np.linalg.norm(X, axis=-1, keepdims=True), 1e-12)


def vectors_from(A, rows):
    """A: dict cond -> (n_roll, H) mid-layer acts. rows: rollout indices
    (per cond, same design so indices align). Returns dict cond -> vec
    (persona mean minus grand mean over the same rows)."""
    means = {c: A[c][rows].mean(0) for c in A}
    g = np.mean(list(means.values()), axis=0)
    return {c: means[c] - g for c in means}


def adjacency(V, conds):
    U = unit(np.stack([V[c] for c in conds]))
    S = U @ U.T
    np.fill_diagonal(S, 0)
    return S


def effdim(V, conds):
    X = np.stack([V[c] for c in conds])
    X = X - X.mean(0)
    w = np.linalg.eigvalsh(X @ X.T)
    w = w[w > 0]
    return w.sum() ** 2 / (w ** 2).sum()


def block_grid(S, idxs):
    kb = len(idxs)
    B = np.zeros((kb, kb))
    for i, ii in enumerate(idxs):
        for j, jj in enumerate(idxs):
            sub = S[np.ix_(ii, jj)]
            B[i, j] = (sub[~np.eye(len(ii), dtype=bool)].mean()
                       if i == j else sub.mean())
    return B


def human_r(S, idxs, Hg):
    g = block_grid(afc.zscore_offdiag(S), idxs)
    m = ~np.eye(g.shape[0], dtype=bool)
    return np.corrcoef(g[m], Hg[m])[0, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=ALL)
    args = ap.parse_args()

    h = json.load(open("results/adjectives/escs_525pda_corr_raw.json"))
    labels = [l.lower() for l in h["labels"]]
    tc = json.load(open("instruments/trait_blocks_44.json"))
    clusters = sorted(tc["pool"], key=lambda c: (c["branch"], -c["coh"]))
    Hm = np.array(h["correlation_matrix"], float)
    np.fill_diagonal(Hm, 0)
    hidx = [[labels.index(m) for m in c["members"]] for c in clusters]
    Hg = block_grid(afc.zscore_offdiag(Hm), hidx)

    rng = np.random.default_rng(0)
    out = {}
    for model in args.models:
        print(f"\n=== {model} ===", flush=True)
        b = torch.load(f"{PV}/{model}_pda.pt", map_location="cpu",
                       weights_only=False)
        texts = json.load(open(f"{PV}/{model}_pda_texts.json"))
        layers = b["acts_layers"]
        mid = len(layers) // 2
        conds = [c for c in labels if c in b["acts"]]
        A = {c: np.asarray(b["acts"][c])[:, mid, :].astype(np.float64)
             for c in conds}
        del b
        # rollout index -> (question, sys); design identical across conds,
        # but verify against one cond's texts and require full 60 rows
        conds = [c for c in conds if len(A[c]) == 60]
        t0 = texts[conds[0]]
        q_of = np.array([QUESTIONS.index(r["question"]) for r in t0])
        sys_of = np.array([hash(r["sys"]) for r in t0])
        sys_ids = {s: i for i, s in enumerate(dict.fromkeys(sys_of))}
        sys_of = np.array([sys_ids[s] for s in sys_of])
        n_q, n_s = len(set(q_of)), len(set(sys_of))
        print(f"  {len(conds)} personas, {n_q} questions x {n_s} templates, "
              f"mid layer {layers[mid]}", flush=True)

        V_full = vectors_from(A, np.arange(60))
        S_full = adjacency(V_full, conds)
        ed_full = effdim(V_full, conds)
        hr_full = human_r(S_full, hidx, Hg)

        # A: question split-half vs random split-half
        cos_q, cos_r, adj_r, ed_half, hr_half = [], [], [], [], []
        for s in range(N_SPLITS):
            qs = rng.permutation(n_q)
            ra = np.isin(q_of, qs[:n_q // 2])
            Va = vectors_from(A, np.where(ra)[0])
            Vb = vectors_from(A, np.where(~ra)[0])
            cos_q.append(np.mean([float(unit(Va[c]) @ unit(Vb[c]))
                                  for c in conds]))
            Sa, Sb = adjacency(Va, conds), adjacency(Vb, conds)
            m = ~np.eye(len(conds), dtype=bool)
            adj_r.append(np.corrcoef(Sa[m], Sb[m])[0, 1])
            if s < 10:
                ed_half.append((effdim(Va, conds) + effdim(Vb, conds)) / 2)
                hr_half.append((human_r(Sa, hidx, Hg)
                                + human_r(Sb, hidx, Hg)) / 2)
            perm = rng.permutation(60)
            Vc = vectors_from(A, perm[:30])
            Vd = vectors_from(A, perm[30:])
            cos_r.append(np.mean([float(unit(Vc[c]) @ unit(Vd[c]))
                                  for c in conds]))

        # B: facet comparison
        qv = [vectors_from(A, np.where(q_of == q)[0]) for q in range(n_q)]
        sv = [vectors_from(A, np.where(sys_of == s)[0]) for s in range(n_s)]

        def cross(vlist):
            cc = []
            for i in range(len(vlist)):
                for j in range(i + 1, len(vlist)):
                    cc.append(np.mean([float(unit(vlist[i][c])
                                             @ unit(vlist[j][c]))
                                       for c in conds]))
            return np.array(cc)
        xq, xs = cross(qv), cross(sv)

        # C: jackknife
        jk = []
        for q in range(n_q):
            Vj = vectors_from(A, np.where(q_of != q)[0])
            jk.append(np.mean([float(unit(Vj[c]) @ unit(V_full[c]))
                               for c in conds]))
        jk = np.array(jk)
        worst = int(np.argmin(jk))

        row = {
            "cos_q_split": float(np.mean(cos_q)),
            "cos_rand_split": float(np.mean(cos_r)),
            "cross_question": float(xq.mean()),
            "cross_template": float(xs.mean()),
            "jackknife_min": float(jk.min()),
            "worst_question": QUESTIONS[worst][:60],
            "adj_r_half": float(np.mean(adj_r)),
            "effdim_full": float(ed_full),
            "effdim_half": float(np.mean(ed_half)),
            "human_r_full": float(hr_full),
            "human_r_half": float(np.mean(hr_half)),
        }
        out[model] = row
        print(f"  A split-half cos: question {row['cos_q_split']:.3f} vs "
              f"random {row['cos_rand_split']:.3f}", flush=True)
        print(f"  B facet agreement: cross-question {row['cross_question']:.3f}"
              f"  cross-template {row['cross_template']:.3f}", flush=True)
        print(f"  C jackknife min cos {row['jackknife_min']:.3f} "
              f"(worst: {row['worst_question']})", flush=True)
        print(f"  D adjacency half-r {row['adj_r_half']:.3f}  effdim "
              f"{row['effdim_full']:.1f} -> half {row['effdim_half']:.1f}  "
              f"human-r {row['human_r_full']:.3f} -> half "
              f"{row['human_r_half']:.3f}", flush=True)
        with open(f"{PV}/question_sensitivity.json", "w") as fp:
            json.dump(out, fp, indent=1)

    print(f"\nwrote {PV}/question_sensitivity.json")


if __name__ == "__main__":
    main()
