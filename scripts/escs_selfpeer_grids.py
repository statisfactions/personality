"""Self vs peer vs cross-source human grids (ESCS S/P-I) + SDV desirability
check (W17 systematic-distortion test, Mini-Marker subset).

Three human grids on the ~40 Mini-Marker adjectives:
  SELF  - corr across 699 self-respondents
  PEER  - corr across targets of per-target peer-mean ratings (2-3 peers)
  CROSS - corr(self_a, peermean_b), symmetrized: the validity-filtered
          structure (covariance surviving a vantage-point change is
          substance; within-source-only structure is shared narrative /
          method variance)

Then: which grid does each model channel (ENACT/JUDGE/REPRESENT, plus the
525-PDA self grid as reference) match best? Narrative-prior prediction:
ENACT ~ SELF >= PEER > CROSS. Directional only (~40 words, ~780 cells).

SDV check: mean desirability-in-others (DESIR1-100, N=701) vs our
pos_eval/neg_eval valence proxy on the overlap; high r licenses the proxy
for the full 523.

Usage:
  PYTHONPATH=scripts .venv/bin/python scripts/escs_selfpeer_grids.py --model llama3.2
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pyreadstat
from scipy.stats import spearmanr

import four_grid_compare as fg
from adjective_introspection import GROUPS

SPI_DIR = Path("data/escs_selfpeer")
SDV_POR = Path("data/escs_sdv/SDV-1.por")
OUT = Path("results/persona_vectors")

# label fixes: ESCS typos / formatting -> standard lowercase forms
FIX = {"cooperatative": "cooperative", "unintellignet": "unintelligent",
       "good_looking": "good-looking"}


def adjective_items(df, meta):
    out = {}
    for c in df.columns:
        l = (meta.column_names_to_labels.get(c) or c).strip()
        if c != "ID" and " " not in l and l[0].isupper():
            k = FIX.get(l.lower(), l.lower())
            out[c] = k
    return out


def pairwise_corr(A, B=None):
    """Pairwise-complete Pearson columns(A) x columns(B or A)."""
    B = A if B is None else B
    n_a, n_b = A.shape[1], B.shape[1]
    M = np.full((n_a, n_b), np.nan)
    for i in range(n_a):
        for j in range(n_b):
            ok = np.isfinite(A[:, i]) & np.isfinite(B[:, j])
            if ok.sum() > 30:
                M[i, j] = np.corrcoef(A[ok, i], B[ok, j])[0, 1]
    return M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3.2")
    args = ap.parse_args()

    # --- self / peer data ---------------------------------------------------
    sdf, smeta = pyreadstat.read_por(str(SPI_DIR / "SelfPeer_Self-1.por"))
    pdf, pmeta = pyreadstat.read_por(str(SPI_DIR / "SelfPeer_Peers-1.por"))
    s_items = adjective_items(sdf, smeta)
    p_items = adjective_items(pdf, pmeta)
    words = sorted(set(s_items.values()) & set(p_items.values()))
    s_cols = {v: k for k, v in s_items.items()}
    p_cols = {v: k for k, v in p_items.items()}

    peer_mean = pdf.groupby("ID")[[p_cols[w] for w in words]].mean()
    self_df = sdf.set_index("ID")[[s_cols[w] for w in words]]
    joined = self_df.join(peer_mean, how="inner", lsuffix="_s", rsuffix="_p")
    S_all = self_df.to_numpy(float)
    P_all = peer_mean.to_numpy(float)
    Sj = joined.iloc[:, :len(words)].to_numpy(float)
    Pj = joined.iloc[:, len(words):].to_numpy(float)
    print(f"{len(words)} adjectives | self n={len(self_df)} "
          f"targets-with-peers n={len(joined)}")

    SELF = pairwise_corr(S_all)
    PEER = pairwise_corr(P_all)
    X = pairwise_corr(Sj, Pj)
    CROSS = (X + X.T) / 2
    diag_validity = np.nanmean(np.diag(X))
    print(f"mean self-peer agreement (same adjective): r={diag_validity:.3f}")

    # --- model channels on the overlap with the 523 set ----------------------
    H525, labels = fg.load_human()
    lab_low = [l.lower() for l in labels]
    present = [w for w in words if w in lab_low]
    print(f"overlap with 523-set: {len(present)} of {len(words)} "
          f"(missing: {[w for w in words if w not in lab_low]})")
    widx = [words.index(w) for w in present]
    gidx = [lab_low.index(w) for w in present]

    R, _ = fg.load_represent(args.model, labels)
    J = fg.load_judge(args.model, labels)
    E = fg.load_enact(args.model, labels)
    human_grids = {"SELF": SELF[np.ix_(widx, widx)],
                   "PEER": PEER[np.ix_(widx, widx)],
                   "CROSS": CROSS[np.ix_(widx, widx)],
                   "PDA525": H525[np.ix_(gidx, gidx)]}
    model_grids = {"ENACT": E[np.ix_(gidx, gidx)],
                   "JUDGE": J[np.ix_(gidx, gidx)],
                   "REPRESENT": R[np.ix_(gidx, gidx)]}

    iu = np.triu_indices(len(present), k=1)
    result = {"n_words": len(present), "self_peer_validity": diag_validity,
              "matrix": {}}
    print(f"\nSpearman, model channel x human vantage ({len(iu[0])} cells):")
    print(f"{'':>10s}" + "".join(f"{h:>9s}" for h in human_grids))
    for mname, M in model_grids.items():
        row = {}
        for hname, Hg in human_grids.items():
            x, y = M[iu], Hg[iu]
            ok = np.isfinite(x) & np.isfinite(y)
            row[hname] = float(spearmanr(x[ok], y[ok]).statistic)
        result["matrix"][mname] = row
        print(f"{mname:>10s}" + "".join(f"{row[h]:>9.3f}" for h in human_grids))
    # human-human reference
    hh = {}
    for a in ("SELF", "PEER"):
        x, y = human_grids[a][iu], human_grids["CROSS"][iu]
        ok = np.isfinite(x) & np.isfinite(y)
        hh[f"{a}|CROSS"] = float(spearmanr(x[ok], y[ok]).statistic)
    x, y = human_grids["SELF"][iu], human_grids["PEER"][iu]
    ok = np.isfinite(x) & np.isfinite(y)
    hh["SELF|PEER"] = float(spearmanr(x[ok], y[ok]).statistic)
    result["human_human"] = hh
    print("human-human:", "  ".join(f"{k}={v:.3f}" for k, v in hh.items()))

    # --- SDV desirability vs valence proxy -----------------------------------
    ddf, dmeta = pyreadstat.read_por(str(SDV_POR))
    des_cols = [(c, (dmeta.column_names_to_labels.get(c) or "").strip().lower())
                for c in ddf.columns if c.upper().startswith("DESIR")]
    des = {l: float(np.nanmean(ddf[c].to_numpy(float))) for c, l in des_cols}
    idx = {l: i for i, l in enumerate(labels)}
    pos_r = [idx[w] for w in GROUPS["pos_eval"]]
    neg_r = [idx[w] for w in GROUPS["neg_eval"]]
    pairs = []
    for l, d in des.items():
        if l in lab_low:
            i = lab_low.index(l)
            v = np.nanmean(H525[i, pos_r]) - np.nanmean(H525[i, neg_r])
            pairs.append((l, d, v))
    dv = np.array([(d, v) for _, d, v in pairs])
    r_proxy = float(np.corrcoef(dv[:, 0], dv[:, 1])[0, 1])
    rs_proxy = float(spearmanr(dv[:, 0], dv[:, 1]).statistic)
    result["sdv"] = {"n_overlap": len(pairs), "pearson_vs_proxy": r_proxy,
                     "spearman_vs_proxy": rs_proxy,
                     "desirability_means": des}
    print(f"\nSDV desirability vs valence proxy: n={len(pairs)} overlap, "
          f"pearson={r_proxy:.3f} spearman={rs_proxy:.3f}")

    out = OUT / f"selfpeer_{args.model}.json"
    json.dump(result, open(out, "w"), indent=1)
    np.savez(OUT / "escs_selfpeer_grids.npz", words=np.array(words),
             SELF=SELF, PEER=PEER, CROSS=CROSS)
    print(f"saved {out} and escs_selfpeer_grids.npz")


if __name__ == "__main__":
    main()
