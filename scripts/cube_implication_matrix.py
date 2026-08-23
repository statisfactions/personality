"""Persona->self-report implication matrix from the 270-cube (2026-08-23).

Han et al.'s RQ3 cross-effects (inject persona X, read self-reported Y)
form an asymmetric implication matrix B[injected, reported]. The W12 cube
computed exactly this (cross_correlation, 5x5, continuous dosed z's) and
analyzed only the diagonal. This script assembles the off-diagonal
structure and asks WHICH channel it matches: the trait-judgment geometry
(JUDGE tom_likely), the representation geometry (REPRESENT), or human
Big-Five inter-scale correlations.

Common basis: the cube's own marker-pole double difference. For a channel
matrix S (523x523, adjective space) the trait-level projection is
  P[i,j] = mean_{p in pole(i), q in pole(j)} sign_i sign_j S[p,q]
using the same Goldberg marker high/low pole sets the cube scored with.

Registered prediction (to_try 2026-08-23): implication ~ JUDGE >> REPRESENT.

Usage: PYTHONPATH=scripts python scripts/cube_implication_matrix.py
"""
import glob
import itertools
import json

import numpy as np

import adjective_facet_cohort as afc
from generate_trait_personas import MARKERS, TRAITS

MODELS = ["Gemma", "Gemma12", "Gemma27", "Gemma4", "Llama", "Llama8",
          "Phi4", "Qwen", "Qwen7", "Qwen32"]
FORMS = ["", "_ipip_raw", "_ipip_reflowed"]
CONDS = ["", "_fake_good", "_fake_good_fgpfx"]


def implication_from_file(path):
    d = json.load(open(path))
    Z = np.array([[p["z_scores"][t] for t in TRAITS]
                  for p in d["persona_data"]])
    S = np.array([[p["scored_trait"][t] for t in TRAITS]
                  for p in d["persona_data"]])
    ok = ~np.isnan(S).any(1)
    Z, S = Z[ok], S[ok]
    M = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            M[i, j] = np.corrcoef(Z[:, i], S[:, j])[0, 1]
    return M


def pole_project(S, labels):
    """Project an adjective-space matrix to trait space via marker poles."""
    idx = {}
    for t in TRAITS:
        for pole, sign in (("high", 1), ("low", -1)):
            for m in MARKERS[t][pole]:
                if m.lower() in labels:
                    idx.setdefault(t, []).append((labels.index(m.lower()), sign))
    P = np.zeros((5, 5))
    for i, ti in enumerate(TRAITS):
        for j, tj in enumerate(TRAITS):
            vals = [si * sj * S[a, b] for a, si in idx[ti]
                    for b, sj in idx[tj] if a != b]
            P[i, j] = np.mean(vals)
    cov = {t: len(idx.get(t, [])) for t in TRAITS}
    return P, cov


def offdiag(M):
    return M[~np.eye(5, dtype=bool)]


def main():
    h = json.load(open("results/adjectives/escs_525pda_corr_raw.json"))
    labels = [l.lower() for l in h["labels"]]
    Hm = np.array(h["correlation_matrix"], float)
    np.fill_diagonal(Hm, 0)
    cz = np.load(afc.CACHE)

    HUMAN, covH = pole_project(Hm, labels)
    JUDGE, _ = pole_project(cz["JUDGE"], labels)
    REPR, _ = pole_project(cz["REPRESENT"], labels)
    ENACT, _ = pole_project(cz["ENACT"], labels)
    print("marker coverage in 523:", covH)
    print("\nchannel projections (rows/cols = A C E N O):")
    for nm, M in [("HUMAN", HUMAN), ("JUDGE", JUDGE), ("REPRESENT", REPR),
                  ("ENACT", ENACT)]:
        print(f"  {nm}:\n" + "\n".join("    " + " ".join(f"{v:6.2f}" for v in r)
                                       for r in M))

    # assemble cube implication matrices
    cube = {}
    for m, f, c in itertools.product(MODELS, FORMS, CONDS):
        p = f"results/persona/persona_instrument_response_{m}{f}{c}.json"
        if glob.glob(p):
            cube[(m, f or "_markers", c or "_honest")] = implication_from_file(p)
    print(f"\ncube cells loaded: {len(cube)}")

    honest = [M for (m, f, c), M in cube.items() if c == "_honest"]
    Mh = np.mean(honest, axis=0)
    print("\nmean HONEST implication matrix (rows=injected, cols=reported):")
    print("\n".join("    " + " ".join(f"{v:6.2f}" for v in r) for r in Mh))

    def cmp(M, name):
        rows = []
        for ch_name, C in [("HUMAN", HUMAN), ("JUDGE", JUDGE),
                           ("REPRESENT", REPR), ("ENACT", ENACT)]:
            sym = (M + M.T) / 2
            r_off = np.corrcoef(offdiag(sym), offdiag((C + C.T) / 2))[0, 1]
            rows.append(f"{ch_name} {r_off:.2f}")
        print(f"  {name:28s} off-diag r vs: " + "  ".join(rows))

    print("\nimplication-vs-channel pattern match (off-diagonal, symmetrized):")
    cmp(Mh, "cohort mean (honest)")
    for c, lbl in [("_fake_good", "FG-suffix"), ("_fake_good_fgpfx", "FG-prefix")]:
        Ms = [M for (m, f, cc), M in cube.items() if cc == c]
        cmp(np.mean(Ms, axis=0), f"cohort mean ({lbl})")

    print("\nper-model (honest, markers form):")
    for m in MODELS:
        M = cube.get((m, "_markers", "_honest"))
        if M is not None:
            cmp(M, m)

    # asymmetry check: is the implication matrix directional like JUDGE?
    asym = np.linalg.norm(Mh - Mh.T) / np.linalg.norm(Mh)
    print(f"\nhonest-mean asymmetry |M-M'|/|M| = {asym:.2f}")

    out = {"traits": TRAITS,
           "human": HUMAN.tolist(), "judge": JUDGE.tolist(),
           "represent": REPR.tolist(), "enact": ENACT.tolist(),
           "cube": {"|".join(k): v.tolist() for k, v in cube.items()}}
    with open("results/persona/cube_implication_matrix.json", "w") as fp:
        json.dump(out, fp, indent=1)
    print("wrote results/persona/cube_implication_matrix.json")


if __name__ == "__main__":
    main()
