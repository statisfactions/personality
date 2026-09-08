"""OLS gain model over the SELF tensor (2026-09-07, P13/P14 ledger arc).

    likert_mfa = lambda_mf + beta_mf * t_ma + eps
    t_m  = framing-average of per-(m,f)-centered profiles (trait template)
    beta_mf = C_f . t / t . t  decomposes as  r_mf x amp_mf:
      r_mf   = corr(frame shape, template)   -> ROTATION (direction change)
      amp_mf = ||frame shape|| / ||template|| -> GAIN (expressiveness)

Also scores each t_m on the elevation-invariant factors (Ten Berge
ruler, as in fig_big5_clouds) for the implied-trait-score table.

Usage: .venv/bin/python scripts/gain_model.py
Out:   results/adjectives/gain_model.json + gain_model.csv
"""
import json

import numpy as np
import pandas as pd

import pkit

labels = pkit.load.adjectives()
FR = pkit.load.FRAMINGS

# elevation-invariant ruler (same construction as fig_big5_clouds v9)
M, lab = pkit.load.load_human()
lab_l = [l.lower() for l in lab]
hix = [lab_l.index(a) for a in labels]
Mh = M[:, hix]
w, v = pkit.axes.eig_axes(Mh, kmax=5)
L = pkit.axes.varimax(v * np.sqrt(w))
u = np.ones(len(labels)) / np.sqrt(len(labels))
Lp = L - np.outer(u, u @ L)
Uo, sv, Vt = np.linalg.svd(Lp, full_matrices=False)
Q = Uo @ Vt
MARKERS = {"A": "kind-hearted", "E": "exciting", "N": "troubled",
           "C": "thorough", "O": "intelligent"}
axmap = {}
for name, m_ in MARKERS.items():
    i = labels.index(m_)
    k = int(np.abs(Q[i]).argmax())
    if Q[i, k] < 0:
        Q[:, k] = -Q[:, k]
    axmap[name] = k
h_t = (Mh - Mh.mean(1, keepdims=True))        # human templates (single frame)
h_scores = h_t @ Q
h_mu, h_sd = h_scores.mean(0), h_scores.std(0)

rows, out = [], {}
for repo, p in sorted(pkit.load.self_paths("cohort").items()):
    d = json.load(open(p))["results"]
    try:
        X = np.array([[d[f][a]["ev"] for a in labels] for f in FR])
    except (KeyError, TypeError):
        continue
    nm = repo.split("/")[-1]
    lam = X.mean(1)
    C = X - lam[:, None]
    t = C.mean(0)
    nt = np.linalg.norm(t)
    if nt < 1e-9:
        continue
    beta = C @ t / (nt ** 2)
    r = np.array([np.corrcoef(C[f], t)[0, 1] for f in range(len(FR))])
    amp = np.linalg.norm(C, axis=1) / nt
    # two conventions, both vs matching human standardization:
    #   amplitude-in (raw view): template as-is — model amplitude deficit
    #     shows as negative scores on normative-positive axes;
    #   direction (shape view): unit-normalized template — matches the cloud.
    tz = ((t @ Q) - h_mu) / h_sd
    hdir = h_t / np.linalg.norm(h_t, axis=1, keepdims=True)
    hd_s = hdir @ Q
    tzd = (((t / nt) @ Q) - hd_s.mean(0)) / hd_s.std(0)
    out[nm] = {"lambda": dict(zip(FR, np.round(lam, 3))),
               "beta": dict(zip(FR, np.round(beta, 3))),
               "r": dict(zip(FR, np.round(r, 3))),
               "amp": dict(zip(FR, np.round(amp, 3))),
               "trait_scores_EIF_ampin": {a: round(float(tz[axmap[a]]), 2)
                                          for a in "AENCO"},
               "trait_scores_EIF_dir": {a: round(float(tzd[axmap[a]]), 2)
                                        for a in "AENCO"},
               "template_norm": round(float(nt), 2)}
    for f in range(len(FR)):
        rows.append(dict(model=nm, framing=FR[f], lam=lam[f], beta=beta[f],
                         r=r[f], amp=amp[f]))
df = pd.DataFrame(rows)
df.to_csv("results/adjectives/gain_model.csv", index=False)
json.dump(out, open("results/adjectives/gain_model.json", "w"), indent=1)
print(f"wrote gain_model.json/.csv ({len(out)} models)")

# cohort summary: is the assistant's low beta rotation or gain?
g = df.groupby("framing")[["beta", "r", "amp"]].mean().reindex(FR)
print("\ncohort means by framing:")
print(g.round(2).to_string())
