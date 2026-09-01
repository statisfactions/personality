"""Loaders: results files -> tidy pandas objects.

Model identifiers: every loader accepts a deep-cohort short name ("phi4",
"Qwen7"), or a full HF repo id ("microsoft/phi-4"); SELF files are
resolved by trying the short-name file then the repo-underscore form.

Faithfulness note: self_paths()/self_matrix() replicate the committed
slide-script conventions exactly (glob order, setdefault dedupe, EXCLUDE,
THINK_PREFER, framing-mean) so pkit numbers == slide numbers.
"""
import glob
import json
import os

import numpy as np
import pandas as pd

from . import paths, roster

FRAMINGS = ["direct", "assistant", "person", "pda", "observer", "outputs"]

# --- human 525-PDA deposit repairs (from adjective_corr_cluster.py) -------
# The two columns are SWAPPED with each other (2026-08-14 2nd-pass
# diagnosis): the only alphabetically out-of-order pair in the file —
# transposed labels on alphabetical data, a one-slot clerical swap in
# deposit assembly. Fix: swap the labels back.
DENY_LABELS = set()
SWAPPED_LABEL_PAIRS = [("Inspirational", "Insensitive")]


def adjectives():
    """Canonical 525 PDA adjective order (lowercased human-matrix labels)."""
    return [l.lower() for l in json.load(open(paths.HUMAN_CORR))["labels"]]


# ---------------------------------------------------------------------- SELF

def _self_file(model, arm="auto"):
    """Resolve a model identifier to its selfreport JSON path.
    arm: "auto" (THINK_PREFER-aware), "prefill", "think"."""
    repo = roster.resolve(model)
    suffixes = {"prefill": ["_self_full.json"],
                "think": ["_self_full_think.json"],
                "auto": (["_self_full_think.json", "_self_full.json"]
                         if repo in roster.THINK_PREFER
                         else ["_self_full.json"])}[arm]
    for s in suffixes:
        for stem in (str(model), repo.replace("/", "_")):
            p = paths.SELF_DIR / f"{stem}{s}"
            if p.exists():
                return p
    raise FileNotFoundError(f"no selfreport file for {model!r} (arm={arm})")


def load_self(model, arm="auto", dist=False):
    """One model's SELF instrument as a tidy DataFrame:
    (framing, adjective, ev, entropy[, p1..p7])."""
    d = json.load(open(_self_file(model, arm)))
    rows = []
    for f, block in d["results"].items():
        for a, r in block.items():
            if not isinstance(r, dict) or "ev" not in r:
                continue
            row = {"framing": f, "adjective": a,
                   "ev": r["ev"], "entropy": r.get("entropy")}
            if dist and isinstance(r.get("dist"), dict):
                row.update({f"p{k}": v for k, v in r["dist"].items()})
            rows.append(row)
    return pd.DataFrame(rows)


def self_paths(which="population"):
    """repo -> selfreport path for the wide cohort.

    which: "cohort" (EXCLUDE-filtered, THINK_PREFER-swapped) or
    "population" (also applies roster.DROP). Replicates the slide scripts:
    glob order + setdefault dedupe, then think-prefer override."""
    by_repo = {}
    for p in glob.glob(str(paths.SELF_DIR / "*_self_full.json")):
        name = os.path.basename(p).replace("_self_full.json", "")
        if roster.EXCLUDE.search(name):
            continue
        repo = (roster.resolve(name) if name in roster.MODELS
                else name.replace("_", "/", 1))
        by_repo.setdefault(repo, p)
    for repo in roster.THINK_PREFER:  # always-think rows may exist only as _think
        tp = paths.SELF_DIR / f"{repo.replace('/', '_')}_self_full_think.json"
        if tp.exists():
            by_repo[repo] = str(tp)
    if which == "population":
        by_repo = {r: p for r, p in by_repo.items()
                   if r.split("/")[-1] not in roster.DROP}
    return by_repo


def self_matrix(framings=None, which="population", labels=None):
    """models x adjectives DataFrame of framing-mean EVs (the slides' R).
    Index = repo tail (model short id)."""
    labels = labels if labels is not None else adjectives()
    framings = framings if framings is not None else FRAMINGS
    rows, idx = [], []
    for repo, p in sorted(self_paths(which).items()):
        d = json.load(open(p))["results"]
        try:
            rows.append(np.mean([[d[f][a]["ev"] for a in labels]
                                 for f in framings], axis=0))
            idx.append(repo.split("/")[-1])
        except (KeyError, TypeError):
            continue
    return pd.DataFrame(rows, index=idx, columns=list(labels))


# --------------------------------------------------------------------- JUDGE

def load_judge(model):
    """JUDGE tom_likely matrices. B[a, b] = EV of "given a, how likely b"
    (row = condition a, col = target b); Hent = per-pair entropy.
    Returns dict(B=DataFrame, Hent=DataFrame, meta=dict)."""
    z = np.load(paths.INTROSPECT / f"{model}_tom_likely_dir.npz",
                allow_pickle=True)
    adj = [str(a).lower() for a in z["adjectives"]]
    out = {"B": pd.DataFrame(np.asarray(z["B"], float), index=adj, columns=adj),
           "Hent": pd.DataFrame(np.asarray(z["Hent"], float),
                                index=adj, columns=adj)}
    jp = paths.INTROSPECT / f"{model}_tom_likely_dir.json"
    out["meta"] = json.load(open(jp)) if jp.exists() else {}
    return out


def base_rate(model):
    """Direct-prevalence instrument EVs as a Series (adjective -> ev)."""
    d = json.load(open(paths.INTROSPECT / f"{model}_base_rate.json"))["results"]
    return pd.Series({a: r["ev"] for a, r in d.items()}, name="ev")


def base_rate_fit(model):
    """Saved joint-LS fit (judge_base_rate_fit.py output npz) as dict of
    Series/DataFrame: l, psi, d, phi."""
    z = np.load(paths.INTROSPECT / f"{model}_base_rate_fit.npz",
                allow_pickle=True)
    adj = [str(a) for a in z["adjectives"]]
    return {"l": pd.Series(np.asarray(z["l"], float), index=adj),
            "psi": pd.Series(np.asarray(z["psi"], float), index=adj),
            "d": pd.Series(np.asarray(z["d"], float), index=adj),
            "phi": pd.DataFrame(np.asarray(z["phi"], float),
                                index=adj, columns=adj)}


# --------------------------------------------------------------------- ENACT

def load_enact(model):
    """Per-model ENACT persona vectors (enact_mid split-file convention).
    Returns dict(dir=DataFrame adjectives x hidden, axis=..., grand=...)."""
    z = np.load(paths.ENACT_MID / f"{model}.npz", allow_pickle=True)
    adj = [str(a).lower() for a in z["adjectives"]]
    out = {"dir": pd.DataFrame(np.asarray(z["dir"], float), index=adj)}
    for k in z.files:
        if k not in ("dir", "adjectives"):
            out[k] = np.asarray(z[k])
    return out


# --------------------------------------------------------------------- HUMAN

def human_corr():
    """Human 525 x 525 raw correlation matrix (escs_525pda_corr_raw.json)."""
    h = json.load(open(paths.HUMAN_CORR))
    lab = [l.lower() for l in h["labels"]]
    return pd.DataFrame(np.array(h["correlation_matrix"], float),
                        index=lab, columns=lab)


def load_human(ipsatize=False):
    """Raw 525-PDA respondent matrix (respondents x 525) from the .por
    deposit, with the label-swap repair and column-mean NaN imputation.
    Needs data/ (gitignored) and pyreadstat. Returns (M, labels)."""
    import pyreadstat
    df, meta = pyreadstat.read_por(str(paths.HUMAN_POR))
    deny = {c for c in df.columns
            if (meta.column_names_to_labels.get(c) or c) in DENY_LABELS}
    cols = [c for c in df.columns if c.upper() != "ID" and c not in deny]
    A = df[cols].astype(float)
    lab = {c: (meta.column_names_to_labels.get(c) or c) for c in cols}
    for x, y in SWAPPED_LABEL_PAIRS:
        cx = [c for c in cols if lab[c] == x]
        cy = [c for c in cols if lab[c] == y]
        if cx and cy:
            A[[cx[0], cy[0]]] = A[[cy[0], cx[0]]].values
    M = A.values
    M = np.where(np.isnan(M), np.nanmean(M, axis=0, keepdims=True), M)
    if ipsatize:
        from .measures import ipsatize as _ips
        M = _ips(M)
    return M, [lab[c] for c in cols]
