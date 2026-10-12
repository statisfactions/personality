"""Is each channel's off-diagonal strength an artifact of its member count?
(rgb 2026-10-11; redoes the 2026-09-22 saturation check at the 525 level, all
five channels, both summary stages.)

Members are subsampled without replacement (10 draws per n): respondents for
HUMAN/SELF, per-model matrices for REPRESENT/JUDGE/ENACT. For each draw the
level-removed and the level & top removed 525 matrices are rebuilt with the
adopted recipe (pkit.channels.level_removed / residual) and summarized by the
off-diagonal SD and mean |entry|.

The two member types fail in opposite directions, so the null differs:
  matrix channels: the cohort mean averages members, so noise SHRINKS with n;
    a flat curve means the grid is signal, not under-averaged noise.
  respondent channels: item correlations from few respondents are noisy, so
    strength GROWS as n falls; the floor is the column-permutation null
    (items shuffled independently across respondents) at the same n.
Usage: PYTHONPATH=scripts .venv/bin/python scripts/offdiag_strength_vs_n.py
"""
import numpy as np
import pkit
from pkit import channels as chn

labels = pkit.load.adjectives()
mem = {c: chn.members(c, labels) for c in chn.CHANNELS}
off = ~np.eye(len(labels), dtype=bool)
rng = np.random.default_rng(0)
DRAWS = 10


def stats(M):
    v = np.asarray(M, float)[off]
    return v.std(), np.abs(v).mean()


def grid_for(c):
    n = len(mem[c][1])
    if mem[c][0] == "respondents":
        g = [12, 24, 49] + ([100, 200, 400] if n > 49 else [])
    else:
        g = [4, 8] + ([12, 24] if n > 24 else [])
    return sorted({k for k in g if k < n} | {n})


def run(c, k, permute=False):
    kind, X = mem[c][0], mem[c][1]
    out = []
    for _ in range(DRAWS if k < len(X) else 1):
        idx = rng.choice(len(X), k, replace=False)
        Xs = np.asarray(X)[idx] if kind == "respondents" else [X[i] for i in idx]
        if permute:
            Xs = np.column_stack([rng.permutation(Xs[:, j]) for j in range(Xs.shape[1])])
        out.append(stats(chn.level_removed(c, kind, Xs)) + stats(chn.residual(c, kind, Xs)))
    return np.array(out).mean(0)


print(f"{'channel':10s} {'n':>5s} | {'level SD':>8s} {'level |r|':>9s} | {'L&T SD':>7s} {'L&T |r|':>8s} | null (respondent channels): level SD, L&T SD")
for c in chn.CHANNELS:
    for k in grid_for(c):
        a = run(c, k)
        line = f"{c:10s} {k:5d} | {a[0]:8.3f} {a[1]:9.3f} | {a[2]:7.3f} {a[3]:8.3f}"
        if mem[c][0] == "respondents":
            z = run(c, k, permute=True); line += f" | {z[0]:.3f}  {z[2]:.3f}"
        print(line, flush=True)
