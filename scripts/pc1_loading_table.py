"""Is "top removed" removing the same thing in every channel? (rgb 2026-10-10)

For each channel's LEVEL-REMOVED cohort matrix (pkit.channels, the summary
figures' first row), take the component that measures.remove_pc1 removes
(largest |eigenvalue|), sign it so "kind" loads positive, and report:
  * its share of the positive spectrum and its split-half stability
    (|cos| between PC1s of member halves, 20 splits);
  * the 5x5 matrix of signed cosines between channels' PC1 loading vectors;
  * the top positive / negative loading adjectives per channel;
  * the most channel-specific adjectives: the largest residuals after
    regressing the channel's loadings on the mean of the other four.
Writes a markdown table to ../personality-paper/_data/table_pc1_loadings.md.
Usage: PYTHONPATH=scripts .venv/bin/python scripts/pc1_loading_table.py
"""
import numpy as np
import pkit
from pkit import channels as chn

labels = pkit.load.adjectives(); n = len(labels); ki = labels.index("kind")
CH = chn.CHANNELS
mats, mem = chn.channel_matrices(labels, with_members=True)
LV = chn.channel_level_removed(labels, mem)
rng = np.random.default_rng(0)


def pc1(M):
    A = np.array(M, float); np.fill_diagonal(A, 0.0)
    w, V = np.linalg.eigh(A); k = np.argmax(np.abs(w)); v = V[:, k]
    return v * np.sign(v[ki]), w[k], w[k] / w[w > 0].sum()


P, share, lam = {}, {}, {}
for c in CH:
    P[c], lam[c], share[c] = pc1(LV[c])

stab = {}
for c in CH:
    kind, X = mem[c][0], mem[c][1]; N = X.shape[0]; cs = []
    for _ in range(20):
        p = rng.permutation(N); a, b = p[: N // 2], p[N // 2:]
        va = pc1(chn.level_removed(c, kind, X[a]))[0]; vb = pc1(chn.level_removed(c, kind, X[b]))[0]
        cs.append(abs(va @ vb))
    stab[c] = float(np.mean(cs))

C = np.array([[P[a] @ P[b] for b in CH] for a in CH])


def words(v, k=6, sign=1):
    o = np.argsort(sign * v)[::-1][:k]
    return ", ".join(labels[i] for i in o)


uniq = {}
for c in CH:
    others = np.mean([P[o] for o in CH if o != c], 0)
    Q = np.c_[np.ones(n), others]; res = P[c] - Q @ np.linalg.lstsq(Q, P[c], rcond=None)[0]
    uniq[c] = (res, float(np.corrcoef(P[c], others)[0, 1]))

out = []
out.append("Table: First principal component of each channel's level-removed similarity matrix (525 adjectives). "
           "Share = eigenvalue share of the positive spectrum; stability = mean |cos| between PC1s of random member halves; "
           "cosines are signed with *kind* loading positive. {#tbl-pc1-cos}\n")
out.append("| | " + " | ".join(CH) + " | Share | Stability |")
out.append("|---|" + "---:|" * (len(CH) + 2))
for i, a in enumerate(CH):
    cells = [("—" if i == j else f"{C[i, j]:.2f}") for j in range(len(CH))]
    out.append(f"| {a} | " + " | ".join(cells) + f" | {100 * share[a]:.0f}% | {stab[a]:.2f} |")
out.append("")
out.append("Table: Highest- and lowest-loading adjectives on each channel's PC1, and the most channel-specific adjectives "
           "(largest residuals after regressing the channel's loadings on the mean of the other four channels' loadings; "
           "r = that regression's correlation). {#tbl-pc1-words}\n")
out.append("| Channel | Top positive | Top negative | Channel-specific (+) | Channel-specific (−) | r with others |")
out.append("|---|---|---|---|---|---:|")
for c in CH:
    res, r = uniq[c]
    out.append(f"| {c} | {words(P[c])} | {words(P[c], sign=-1)} | {words(res, 5)} | {words(res, 5, -1)} | {r:.2f} |")
md = "\n".join(out) + "\n"
dst = "../personality-paper/_data/table_pc1_loadings.md"
open(dst, "w").write(md)
print(md); print("wrote", dst)
print("eigenvalue sign check (all should be positive):", {c: round(float(lam[c]), 2) for c in CH})
