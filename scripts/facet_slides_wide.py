"""Wide-cohort (n~50) version of the five-space facet slides.

SELF: model-mean profiles from every completed wide-n + standing instruct
self-report (bases, SFT/DPO rungs, think arms, and shelved artifacts
excluded; deduped by resolved repo). REPRESENT: cohort mean of per-model
z-scored mid-layer cosine grids over all captured __pers acts, per-model
massive-dim winsorization computed from the acts themselves. JUDGE /
ENACT: unchanged cohort-12 / cohort-10 channels (wide-n did not capture
them), labeled as such.

Usage: PYTHONPATH=scripts python scripts/facet_slides_wide.py
Out:   results/persona_vectors/figs/slides_wide/
"""
import glob
import json
import os
import re
from pathlib import Path

import numpy as np
import torch

import adjective_facet_cohort as afc
import facet_slides as fs
import hf_logprobs as hf

OUT = Path("results/persona_vectors/figs/slides_wide")
SELF_DIR = "results/adjectives/selfreport"
ACTS_DIR = "results/adjectives/acts"
EXCLUDE = re.compile(r"Base|SFT|DPO|_bare|THINKOPEN|_think|_smoke|_PRERETRY")

# Always-think models: the forced-prefill SELF row is off-policy noise
# (Glimmer: prefill shape r=-0.19 with cohort vs think 0.84); prefer the
# think-arm file when it exists (2026-08-24 swap, ledgered 2026-08-22).
THINK_PREFER = {"meta-models/Muse-Glimmer-30B"}


def selfreport_path(repo, default_path):
    if repo in THINK_PREFER:
        p = f"{SELF_DIR}/{repo.replace('/', '_')}_self_full_think.json"
        if os.path.exists(p):
            return p
    return default_path


def collect_self(labels):
    by_repo = {}
    for p in glob.glob(f"{SELF_DIR}/*_self_full.json"):
        name = os.path.basename(p).replace("_self_full.json", "")
        if EXCLUDE.search(name):
            continue
        repo = hf.resolve(name) if name in hf.MODELS else name.replace("_", "/", 1)
        by_repo.setdefault(repo, p)
    resp, models = [], []
    for repo, p in sorted(by_repo.items()):
        d = json.load(open(selfreport_path(repo, p)))["results"]
        try:
            resp.append(np.mean([[d[f][a]["ev"] for a in labels]
                                 for f in afc.FRAMINGS], axis=0))
            models.append(repo)
        except (KeyError, TypeError):
            print(f"  [self skip] {repo} (missing framing/adjective)")
    print(f"SELF: {len(models)} models")
    return np.array(resp)


def collect_represent(labels):
    acc = []
    used = 0
    for p in sorted(glob.glob(f"{ACTS_DIR}/*__pers.pt")):
        if EXCLUDE.search(os.path.basename(p)):
            continue
        try:
            b = torch.load(p, map_location="cpu", weights_only=False)
            adj = [str(a).lower() for a in b["adjectives"]]
            idx = []
            for l in labels:
                if l not in adj:
                    raise KeyError(l)
                idx.append(adj.index(l))
            A = np.asarray(b["acts"])
            mid = (A.shape[1] - 1) // 2
            X = A[idx, mid, :].astype(np.float64)
            ma = np.abs(X).mean(0)
            med = np.median(ma)
            massive = np.where(ma >= 20 * med)[0]
            keep = np.setdiff1d(np.arange(X.shape[1]), massive)
            std = X.std(0)
            cap = std[keep].max() if len(keep) else std.max()
            for d_ in massive:
                if std[d_] > cap:
                    X[:, d_] *= cap / std[d_]
            Xc = X - X.mean(0)
            Xn = Xc / np.linalg.norm(Xc, axis=1, keepdims=True)
            S = Xn @ Xn.T
            np.fill_diagonal(S, 0)
            acc.append(afc.zscore_offdiag(S))
            used += 1
        except Exception as e:
            print(f"  [repr skip] {os.path.basename(p)}: {type(e).__name__}")
    print(f"REPRESENT: {used} models")
    return np.mean(acc, axis=0), used


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    h = json.load(open("results/adjectives/escs_525pda_corr_raw.json"))
    labels = [l.lower() for l in h["labels"]]
    tc = json.load(open("instruments/trait_blocks_44.json"))
    clusters = sorted(tc["pool"], key=lambda c: (c["branch"], -c["coh"]))
    H0 = np.array(h["correlation_matrix"], float)
    Hm = H0.copy()
    np.fill_diagonal(Hm, 0)

    R = collect_self(labels)
    S = np.corrcoef(R.T)
    np.fill_diagonal(S, 0)
    n_self = R.shape[0]

    REP, n_rep = collect_represent(labels)

    cz = np.load(afc.CACHE)
    mats = {"HUMAN": afc.zscore_offdiag(Hm), "SELF": afc.zscore_offdiag(S),
            "REPRESENT": REP, "JUDGE": cz["JUDGE"], "ENACT": cz["ENACT"]}

    def block(M):
        k = len(clusters)
        B = np.zeros((k, k))
        for i, ci in enumerate(clusters):
            ii = [labels.index(m) for m in ci["members"]]
            for j, cj in enumerate(clusters):
                jj = [labels.index(m) for m in cj["members"]]
                sub = M[np.ix_(ii, jj)]
                B[i, j] = (sub[~np.eye(len(ii), dtype=bool)].mean()
                           if i == j else sub.mean())
        return B

    names = []
    for c in clusters:
        hidx = [labels.index(m) for m in c["members"]]
        sub = H0[np.ix_(hidx, hidx)]
        names.append(c["members"][int(np.argmax(sub.mean(1)))])
    branches = [c["branch"] for c in clusters]

    grids = {ch: block(M) for ch, M in mats.items()}
    grids_p = {ch: block(afc.zscore_offdiag(afc.remove_pc1(M)))
               for ch, M in mats.items()}

    # wide-cohort captions
    fs.BEATS["SELF"]["sub"] = (
        f"Same construction with models as respondents: n = {n_self} "
        "deployed instruct models (wide-n cohort + standing; 6 framings "
        "averaged) — the correlation estimate is now full-rank at block "
        "level. Raw congruence is a desirability freebie; the top-"
        "component-removed number is the honest one.")
    fs.BEATS["REPRESENT"]["sub"] = (
        f"Residual-stream cosine between adjective activations (pers "
        f"framing, mid layer), {n_rep}-model cohort mean (wide-n capture). "
        "More structure than SELF, but beyond the shared evaluative axis "
        "it organizes traits its own way.")
    fs.BEATS["JUDGE"]["sub"] += "  [cohort-12 channel — not wide-n captured]"
    fs.BEATS["ENACT"]["sub"] += "  [cohort-10 channel — not wide-n captured]"

    fs.OUT = OUT
    for ch in ["HUMAN", "SELF", "REPRESENT", "JUDGE", "ENACT"]:
        fig = fs.slide(ch, grids, grids_p, names, branches)
        stem = OUT / f"slide{fs.BEATS[ch]['num']}_{ch.lower()}"
        fig.write_html(f"{stem}.html")
        fig.write_image(f"{stem}.png", scale=2)
        print(f"saved {stem}.png")
    fig = fs.summary_slide(grids_p)
    fig.write_html(OUT / "slide6_summary.html")
    fig.write_image(OUT / "slide6_summary.png", scale=2)
    for ch in ["SELF", "REPRESENT", "JUDGE", "ENACT"]:
        print(f"  {ch:9s} raw r={fs.congruence(grids, ch):.3f}  "
              f"pc1-removed r={fs.congruence(grids_p, ch):.3f}")


if __name__ == "__main__":
    main()
