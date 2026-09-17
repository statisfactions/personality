"""The five channels as 525-adjective similarity matrices, built from the
current artifacts with the adopted cooking (2026-09-17 centralization).

Every channel comes in two forms: the per-respondent / per-model MEMBERS
(what split-half reliability is computed over) and the COHORT matrix
(zero-diagonal, raw correlation-like units in [-1, 1]):

    HUMAN      700 ESCS respondents x 525 ratings      -> Pearson item matrix
    SELF       core instruct models x 525 framing-mean EVs (within-model
               SD >= 0.5)                               -> Pearson item matrix
    REPRESENT  per-model column-centered cosine of mid-layer activations,
               massive dims winsorized (cached in represent_model_grids.npz
               key `cos`), core models only            -> mean matrix
    JUDGE      per-model tom_likely EV matrix -> pairs-only potential ->
               level pinned to medP=.5 -> implied phi, CLIPPED to [-1, 1]
                                                        -> mean matrix
    ENACT      per-model persona-vector cosine, meta massive dims winsorized
                                                        -> mean matrix

Top-component removal (the paper's second row): subtract the off-diagonal
mean (grand-mean centering), then the largest eigencomponent. See
scripts/centering_check.py for why grand-mean and not none / Gower.
"""
import glob
import json
import os

import numpy as np

from . import cooking, facets, load, measures, paths, roster

CHANNELS = ["HUMAN", "SELF", "REPRESENT", "JUDGE", "ENACT"]
CORE_SD = 0.50
PHI_CLIP = 1.0
REPRESENT_CACHE = "results/adjectives/represent_model_grids.npz"


def _zero_diag(S):
    S = np.array(S, float)
    np.fill_diagonal(S, 0.0)
    return S


def center(M):
    """Grand-mean centering: subtract the off-diagonal mean (units kept)."""
    B = np.array(M, float)
    m = ~np.eye(len(B), dtype=bool)
    B[m] -= B[m].mean()
    return B


def top_removed(M):
    """Grand-mean centering, then remove the largest eigencomponent."""
    return measures.remove_pc1(center(M))


# ---- members ---------------------------------------------------------------

def human_members(labels=None):
    """Respondents x 525 raw ratings, aligned to the canonical label order."""
    labels = labels or load.adjectives()
    Hm, hlab = load.load_human()
    hlab = [l.lower() for l in hlab]
    return Hm[:, [hlab.index(l) for l in labels]]


def self_members(labels=None, core_sd=CORE_SD):
    """Core instruct models x 525 framing-mean EVs; returns (X, names)."""
    R = load.self_matrix(which="cohort", labels=labels)
    keep = R.values.std(1) >= core_sd
    return R.values[keep], list(R.index[keep])


def represent_members(labels=None, core_sd=CORE_SD):
    """Per-model raw cosine matrices for the core models; returns (grids, names)."""
    labels = labels or load.adjectives()
    z = np.load(REPRESENT_CACHE, allow_pickle=True)
    names = [str(x) for x in z["names"]]
    keep = []
    for i, n in enumerate(names):
        try:
            d = json.load(open(load._self_file(n.replace("_", "/", 1))))["results"]
            X = np.array([[d[f][a]["ev"] for a in labels] for f in load.FRAMINGS])
            if X.mean(0).std() >= core_sd:
                keep.append(i)
        except Exception:
            pass
    return z["cos"][keep].astype(np.float32), [names[i] for i in keep]


def judge_members(labels=None, phi_clip=PHI_CLIP):
    """Per-model clipped implied-phi matrices (adopted cooking); returns (mats, names)."""
    labels = labels or load.adjectives()
    mats, names = [], []
    for p in sorted(glob.glob("results/adjectives/introspect_full/*_tom_likely_dir.npz")):
        zj = np.load(p, allow_pickle=True)
        ja = [str(a).lower() for a in zj["adjectives"]]
        idx = [ja.index(l) for l in labels]
        B = np.asarray(zj["B"], float)[np.ix_(idx, idx)]
        psi = cooking.pairs_potential(B)
        P = np.clip(np.exp(psi - np.median(psi) + np.log(0.5)), 0.01, 0.99)
        phi = np.clip(cooking.implied_phi(cooking.EV2P(B), P), -phi_clip, phi_clip)
        mats.append(_zero_diag(phi))
        names.append(os.path.basename(p).split("_tom_likely")[0])
    return np.array(mats), names


def enact_members(labels=None):
    """Per-model persona-vector cosine matrices; returns (mats, names)."""
    labels = labels or load.adjectives()
    mats, names = [], []
    for p in sorted(glob.glob("results/persona_vectors/enact_mid/*.npz")):
        m = os.path.basename(p).replace(".npz", "")
        ze = np.load(p, allow_pickle=True)
        ea = [str(a).lower() for a in ze["adjectives"]]
        E = np.asarray(ze["dir"], np.float64)[[ea.index(l) for l in labels]]
        meta = json.load(open(f"results/persona_vectors/{m}_pda_meta.json"))
        Xe = measures.cos_sim(measures.winsorize(E, np.asarray(meta["massive_dims"], int)))
        mats.append(_zero_diag(Xe))
        names.append(m)
    return np.array(mats), names


def members(channel, labels=None):
    """(kind, X, names): kind is 'respondents' (rows -> correlate) or
    'matrices' (per-model matrices -> average)."""
    labels = labels or load.adjectives()
    if channel == "HUMAN":
        return "respondents", human_members(labels), None
    if channel == "SELF":
        X, names = self_members(labels)
        return "respondents", X, names
    fn = {"REPRESENT": represent_members, "JUDGE": judge_members, "ENACT": enact_members}[channel]
    X, names = fn(labels)
    return "matrices", X, names


def cohort_matrix(kind, X):
    """Cohort 525 matrix from members: correlate respondents, or average matrices."""
    if kind == "respondents":
        return _zero_diag(np.corrcoef(np.asarray(X, float).T))
    return np.asarray(X, np.float64).mean(0)


def channel_matrices(labels=None, with_members=False):
    """Dict of the five cohort matrices (zero-diagonal, raw units).
    with_members=True also returns {channel: (kind, X, names)}."""
    labels = labels or load.adjectives()
    mem = {c: members(c, labels) for c in CHANNELS}
    mats = {c: cohort_matrix(mem[c][0], mem[c][1]) for c in CHANNELS}
    return (mats, mem) if with_members else mats


def blockify(M, which="blocks44"):
    return facets.block(M, facets.clusters(which)).values


def congruence(A, B):
    """Off-diagonal Pearson r between two square matrices."""
    m = ~np.eye(len(A), dtype=bool)
    return float(np.corrcoef(np.asarray(A)[m], np.asarray(B)[m])[0, 1])
