"""Centering before top-component removal: none vs grand-mean (ours) vs Gower double-centering (rgb 2026-09-17). Usage: .venv/bin/python scripts/centering_check.py"""
import sys, numpy as np
sys.argv = ["x", "--perms", "0", "--halves", "1"]
src = open("scripts/channel_similarity_matrix.py").read().split("# ---- congruence matrices")[0]
exec(src)   # builds mats, CH, labels, blockify, r_off, measures, n
def grand(M): B = M.copy(); m = ~np.eye(len(M), dtype=bool); B[m] -= B[m].mean(); return B
def double(M):                    # Gower / classical-MDS double centering of the zero-diagonal matrix
    J = np.eye(len(M)) - np.ones((len(M), len(M))) / len(M); return J @ M @ J
def top_rm(M): w, V = np.linalg.eigh(M); k = np.argmax(np.abs(w)); return M - w[k] * np.outer(V[:, k], V[:, k])
variants = {"no centering": lambda M: top_rm(M), "grand-mean centering (ours)": lambda M: top_rm(grand(M)), "double centering (Gower)": lambda M: top_rm(double(M))}
uni = np.ones(n) / np.sqrt(n)
Hfull = mats["HUMAN"]; wh, Vh = np.linalg.eigh(grand(Hfull)); evax = Vh[:, np.argmax(np.abs(wh))]
for name, f in variants.items():
    G = {c: blockify(f(mats[c])) for c in CH}
    print(f"\n{name}: 44-block congruence after top removal")
    print(f"{'':10s}" + "".join(f"{c:>10s}" for c in CH) + "   | removed axis |cos| with uniform / human-eval")
    for a in CH:
        pre = {"no centering": mats[a], "grand-mean centering (ours)": grand(mats[a]), "double centering (Gower)": double(mats[a])}[name]
        w, V = np.linalg.eigh(pre); k = np.argmax(np.abs(w)); v = V[:, k]
        print(f"{a:10s}" + "".join(f"{r_off(G[a], G[b]):10.2f}" for b in CH) + f"   | {abs(v @ uni):.2f} / {abs(v @ evax):.2f}")
