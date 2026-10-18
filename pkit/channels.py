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

Level removed (the summary figures' first row since 2026-10-10) and top
removal (the second row; recipe adopted 2026-09-22, to_try.md
"RECIPE ADOPTED"): remove level where it lives (LEVEL below: HUMAN/SELF
row-center the respondent x item design matrix and re-correlate; REPRESENT/
ENACT nothing, their column-centered cosines are already centered; JUDGE
double-center phi), then remove the largest eigencomponent. The former
grand-mean version survives only as top_removed(), a LEGACY comparison.
"""
import glob
import json
import os

import numpy as np

from . import cooking, facets, load, measures, paths, roster

CHANNELS = ["HUMAN", "SELF", "REPRESENT", "JUDGE", "ENACT"]
CORE_SD = 0.50
PHI_CLIP = 1.0
REPRESENT_CACHE = str(paths.ADJ / "represent_model_grids.npz")  # root-relative (callable from ../personality-paper)


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
    """LEGACY (pre-2026-09-22): grand-mean centering, then remove the largest
    eigencomponent. Superseded by residual()/channel_residuals(); kept for
    the comparison rows and for reproducing earlier numbers."""
    return measures.remove_pc1(center(M))


def double_center(S):
    """Gower double centering J S J (J = I - 11'/n), zero diagonal."""
    S = np.asarray(S, float); n = len(S)
    Jm = np.eye(n) - np.ones((n, n)) / n
    return _zero_diag(Jm @ S @ Jm)


# Adopted level-removal recipe (rgb, 2026-09-22): remove each channel's
# LEVEL where it lives. HUMAN/SELF: row-center the respondent x item design
# matrix (center-only ipsatization), then correlate items -- so the grid and
# the factor analysis are one object. REPRESENT/ENACT: nothing -- column-
# centered cosines are already centered (matrix centering is inert, r .999).
# JUDGE: double-center phi, which carries a level and is not a correlation
# matrix. The residual then removes the largest eigencomponent (desirability).
LEVEL = {"HUMAN": "rows", "SELF": "rows", "REPRESENT": "none", "ENACT": "none", "JUDGE": "double"}


def level_removed(channel, kind, X):
    """Cohort 525 matrix with the channel's level removed (see LEVEL)."""
    how = LEVEL[channel]
    if how == "rows":
        assert kind == "respondents"
        X = np.asarray(X, float)
        return cohort_matrix(kind, X - X.mean(1, keepdims=True))
    S = cohort_matrix(kind, X)
    return double_center(S) if how == "double" else S


def residual(channel, kind, X):
    """Level removed, then the largest eigencomponent removed."""
    return _zero_diag(measures.remove_pc1(level_removed(channel, kind, X)))


def channel_level_removed(labels=None, mem=None):
    """Dict of the five level-removed cohort matrices (the summary figures'
    first row, rgb 2026-10-10; the raw native coefficients stay available
    from channel_matrices for the SELF section's messiness)."""
    labels = labels or load.adjectives()
    mem = mem or {c: members(c, labels) for c in CHANNELS}
    return {c: level_removed(c, mem[c][0], mem[c][1]) for c in CHANNELS}


def channel_residuals(labels=None, mem=None):
    """Dict of the five residual matrices (adopted recipe)."""
    labels = labels or load.adjectives()
    mem = mem or {c: members(c, labels) for c in CHANNELS}
    return {c: residual(c, mem[c][0], mem[c][1]) for c in CHANNELS}


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
    """Per-model raw cosine matrices for the core models; returns (grids, names).
    Core = the SELF core roster (self_members), matched on the repo tail.
    (2026-10-10 fix: the former per-model SELF-file lookup by repo id failed
    for the deep-cohort models stored under short names, and a bare except
    dropped 10 core models silently — REPRESENT was n=40, should be 50.)"""
    labels = labels or load.adjectives()
    z = np.load(REPRESENT_CACHE, allow_pickle=True)
    names = [str(x) for x in z["names"]]
    core = set(self_members(labels, core_sd)[1])
    keep = [i for i, n in enumerate(names) if n.split("_", 1)[-1] in core]
    return z["cos"][keep].astype(np.float32), [names[i] for i in keep]


def judge_members(labels=None, phi_clip=PHI_CLIP):
    """Per-model clipped implied-phi matrices (adopted cooking); returns (mats, names)."""
    labels = labels or load.adjectives()
    mats, names = [], []
    for p in sorted(glob.glob(str(paths.INTROSPECT / "*_tom_likely_dir.npz"))):
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
    for p in sorted(glob.glob(str(paths.ENACT_MID / "*.npz"))):
        m = os.path.basename(p).replace(".npz", "")
        ze = np.load(p, allow_pickle=True)
        ea = [str(a).lower() for a in ze["adjectives"]]
        E = np.asarray(ze["dir"], np.float64)[[ea.index(l) for l in labels]]
        meta = json.load(open(paths.RESULTS / "persona_vectors" / f"{m}_pda_meta.json"))
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


# ---- paper-facing summaries (data only; presentation lives in the paper) ----

SHARE_DIAG = {"HUMAN": "ones", "SELF": "ones", "REPRESENT": "ones", "JUDGE": "zero", "ENACT": "ones"}


def spectrum_share(w, diag="ones"):
    """Eigenvalue shares from the ZERO-diagonal spectrum w (descending order
    not required). diag="ones": unit diagonal restored (w + 1), i.e. the
    textbook percent of item variance; valid for the correlation/cosine
    channels, whose ones-diagonal matrices are PSD. diag="zero": share of the
    positive part of the zero-diagonal spectrum (JUDGE: double-centered phi
    is not a correlation matrix, so it has no item-variance reading).
    (rgb 2026-10-11: zero-diagonal shares were being read as percent of
    variance; HUMAN PC1 .31 zero-diag vs .16 ones-diag.)"""
    w = np.asarray(w, float)
    if diag == "ones":
        w1 = w + 1.0
        return w1 / w1[w1 > 0].sum()
    return w / w[w > 0].sum()


def top_component(M, labels, ref="kind", diag="ones"):
    """The component measures.remove_pc1 removes (largest |eigenvalue| of the
    zero-diagonal matrix; the diagonal shifts eigenvalues uniformly, so the
    loadings do not depend on it), signed so `ref` loads positive.
    Returns (loadings, zero-diagonal eigenvalue, share per spectrum_share(diag);
    pass diag=SHARE_DIAG[channel])."""
    A = np.array(M, float); np.fill_diagonal(A, 0.0)
    w, V = np.linalg.eigh(A); k = np.argmax(np.abs(w)); v = V[:, k]
    return v * np.sign(v[labels.index(ref)]), float(w[k]), float(spectrum_share(w, diag)[k])


def congruence_table(which="level", labels=None, mem=None):
    """5x5 DataFrame of off-diagonal Pearson congruence on the full 525
    matrices. which: "raw" (native coefficients), "level" (level removed),
    "top" (level & top removed)."""
    import pandas as pd
    labels = labels or load.adjectives()
    mem = mem or {c: members(c, labels) for c in CHANNELS}
    M = {"raw": lambda: {c: cohort_matrix(mem[c][0], mem[c][1]) for c in CHANNELS},
         "level": lambda: channel_level_removed(labels, mem),
         "top": lambda: channel_residuals(labels, mem)}[which]()
    return pd.DataFrame([[congruence(M[a], M[b]) if a != b else 1.0 for b in CHANNELS] for a in CHANNELS],
                        index=CHANNELS, columns=CHANNELS)


def pc1_summary(labels=None, mem=None, n_splits=20, k_words=6, k_unique=5, seed=0):
    """Is top removal removing the same thing in every channel? Computed on the
    level-removed 525 matrices. Returns a dict:
      cos        5x5 DataFrame, signed cosines between channels' PC1 loadings
      loadings   DataFrame adjectives x channels
      share      Series, PC1 share (spectrum_share, diag=SHARE_DIAG: percent of
                 item variance; JUDGE positive spectrum)
      stability  Series, mean |cos| between PC1s of random member halves
      r_others   Series, r of each channel's loadings with the mean of the other four
      top_pos / top_neg          {channel: [adjectives]}, highest / lowest loadings
      unique_pos / unique_neg    {channel: [adjectives]}, largest +/- residuals after
                                 regressing on the mean of the other four channels
    """
    import pandas as pd
    labels = labels or load.adjectives(); n = len(labels)
    mem = mem or {c: members(c, labels) for c in CHANNELS}
    LV = channel_level_removed(labels, mem)
    P, share = {}, {}
    for c in CHANNELS:
        P[c], _, share[c] = top_component(LV[c], labels, diag=SHARE_DIAG[c])
    rng = np.random.default_rng(seed); stab = {}
    for c in CHANNELS:
        kind, X = mem[c][0], mem[c][1]; N = X.shape[0]; cs = []
        for _ in range(n_splits):
            p = rng.permutation(N); a, b = p[: N // 2], p[N // 2:]
            cs.append(abs(top_component(level_removed(c, kind, X[a]), labels)[0]
                          @ top_component(level_removed(c, kind, X[b]), labels)[0]))
        stab[c] = float(np.mean(cs))
    top = lambda v, k, s: [labels[i] for i in np.argsort(s * v)[::-1][:k]]
    out = dict(cos=pd.DataFrame([[float(P[a] @ P[b]) for b in CHANNELS] for a in CHANNELS], index=CHANNELS, columns=CHANNELS),
               loadings=pd.DataFrame(P, index=labels), share=pd.Series(share), stability=pd.Series(stab),
               r_others={}, top_pos={}, top_neg={}, unique_pos={}, unique_neg={})
    for c in CHANNELS:
        others = np.mean([P[o] for o in CHANNELS if o != c], 0)
        Q = np.c_[np.ones(n), others]; res = P[c] - Q @ np.linalg.lstsq(Q, P[c], rcond=None)[0]
        out["r_others"][c] = float(np.corrcoef(P[c], others)[0, 1])
        out["top_pos"][c], out["top_neg"][c] = top(P[c], k_words, 1), top(P[c], k_words, -1)
        out["unique_pos"][c], out["unique_neg"][c] = top(res, k_unique, 1), top(res, k_unique, -1)
    out["r_others"] = pd.Series(out["r_others"])
    return out


def level_removal_summary(labels=None, mem=None, channels=("HUMAN", "SELF", "JUDGE")):
    """What level removal does, per channel (rgb 2026-10-18: harmless for
    HUMAN, removes nonsense for SELF, a design choice for JUDGE). DataFrame
    indexed by channel:
      n                      members (respondents / models)
      method                 LEVEL recipe ("rows" = row-center, "double" = Gower)
      rowmean_var_frac       share of item variance due to respondent means
                             (respondent channels only)
      mean_offdiag_raw/level mean off-diagonal entry before / after
      cos_rawpc1_ones        |cos| of the raw top component with the all-ones
                             (level) direction
      cos_rawpc1_levelpc1    |cos| of raw vs level-removed top component
      pc1_share_raw/level    top-component share (spectrum_share, SHARE_DIAG)
      congruence             off-diagonal r, raw vs level removed (blind to a
                             uniform shift -- do not use it alone)
    REPRESENT and ENACT have no level step (LEVEL "none")."""
    import pandas as pd
    labels = labels or load.adjectives(); n = len(labels)
    mem = mem or {c: members(c, labels) for c in CHANNELS}
    off = ~np.eye(n, dtype=bool); one = np.ones(n) / np.sqrt(n); rows = {}
    for c in channels:
        kind, X = mem[c][0], np.asarray(mem[c][1], float)
        raw, lvl = cohort_matrix(kind, X), level_removed(c, kind, X)
        v_raw, _, sh_raw = top_component(raw, labels, diag=SHARE_DIAG[c])
        v_lvl, _, sh_lvl = top_component(lvl, labels, diag=SHARE_DIAG[c])
        r = dict(n=len(X), method=LEVEL[c], rowmean_var_frac=np.nan,
                 mean_offdiag_raw=raw[off].mean(), mean_offdiag_level=lvl[off].mean(),
                 cos_rawpc1_ones=abs(v_raw @ one), cos_rawpc1_levelpc1=abs(v_raw @ v_lvl),
                 pc1_share_raw=sh_raw, pc1_share_level=sh_lvl, congruence=congruence(raw, lvl))
        if kind == "respondents":
            Xc = X - X.mean(0)
            r["rowmean_var_frac"] = n * Xc.mean(1).var() / Xc.var(0).sum()
        rows[c] = r
    return pd.DataFrame(rows).T


def mantel_p(A, B, perms=1000, seed=0):
    """One-sided Mantel permutation p for off-diagonal congruence: relabel B's
    rows and columns together. Floor is 1 / (perms + 1)."""
    rng = np.random.default_rng(seed)
    obs = congruence(A, B); k = len(A); m = ~np.eye(k, dtype=bool); a = np.asarray(A)[m]
    hits = sum(np.corrcoef(a, np.asarray(B)[np.ix_(p, p)][m])[0, 1] >= obs
               for p in (rng.permutation(k) for _ in range(perms)))
    return (hits + 1) / (perms + 1)
