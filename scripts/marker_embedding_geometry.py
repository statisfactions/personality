#!/usr/bin/env python3
"""Is Big Five trait geometry already in the token-embedding table? (W13 §3.10)

rgb's logit-lens prediction (§3.9 follow-up): if the facet/trait geometry our
models recover "lives so close to the textual level," it should be present in
the static input-embedding matrix (W_E) for SINGLE-TOKEN markers — no forward
pass, no composition. The layer-0 sweep on multi-token IPIP *items* found the
geometry weak at layer 0 and built up with depth — but that is a sentence-
composition result (items are multi-token, mean-pooled, diluted by template
tokens), not a test of whether the per-token lexical geometry exists.

This runs the pure test: read W_E directly, look up single-token Big Five
adjective markers, and ask whether same-trait markers cluster (within-vs-across
cosine, nearest-neighbor purity, 5-cluster purity) and whether meandiff trait
directions are near-orthogonal (as Big Five traits are in humans: within-facet
0.42 vs across −0.01).

Two marker sets, both transcribed from the originals:
  - Saucier (1994) Mini-Markers, Table 2 (40 items; sign of varimax loading
    gives the pole).
  - Goldberg (1992) transparent bipolar markers, Appendix B (35 pairs).

Reads only the embedding tensor from the cached safetensors (fast, low memory;
fine for the 32B). Usage:
    .venv/bin/python scripts/marker_embedding_geometry.py
"""

import glob
import json
import os

import numpy as np
from safetensors import safe_open
from transformers import AutoTokenizer

import sys
sys.path.insert(0, "scripts")
from hf_logprobs import MODELS

HUMAN_PATH = "instruments/ipip300_human_facet_correlations.json"
OUT_PATH = "results/facets/marker_embedding_geometry.json"
HF_HUB = os.path.expanduser("~/.cache/huggingface/hub")

# Marker entry: (word, trait, pole)  pole=+1 -> high-trait pole, -1 -> low.
# For N (from Emotional-Stability scales) +1 = high neuroticism (unstable).
SAUCIER = [
    # Factor I -> E
    ("talkative", "E", 1), ("extroverted", "E", 1), ("bold", "E", 1), ("energetic", "E", 1),
    ("shy", "E", -1), ("quiet", "E", -1), ("bashful", "E", -1), ("withdrawn", "E", -1),
    # Factor II -> A
    ("sympathetic", "A", 1), ("warm", "A", 1), ("kind", "A", 1), ("cooperative", "A", 1),
    ("cold", "A", -1), ("unsympathetic", "A", -1), ("rude", "A", -1), ("harsh", "A", -1),
    # Factor III -> C
    ("organized", "C", 1), ("efficient", "C", 1), ("systematic", "C", 1), ("practical", "C", 1),
    ("disorganized", "C", -1), ("sloppy", "C", -1), ("inefficient", "C", -1), ("careless", "C", -1),
    # Factor IV (Emotional Stability) -> N  (high-N = unstable pole)
    ("moody", "N", 1), ("jealous", "N", 1), ("temperamental", "N", 1), ("envious", "N", 1),
    ("touchy", "N", 1), ("fretful", "N", 1), ("unenvious", "N", -1), ("relaxed", "N", -1),
    # Factor V -> O
    ("creative", "O", 1), ("imaginative", "O", 1), ("philosophical", "O", 1), ("intellectual", "O", 1),
    ("complex", "O", 1), ("deep", "O", 1), ("uncreative", "O", -1), ("unintellectual", "O", -1),
]

# Goldberg (1992) Appendix B transparent bipolar pairs. Right pole = high score.
# For N the right pole is calm/stable = LOW neuroticism, so the LEFT pole is +1.
_GB_PAIRS = {
    "E": [("introverted", "extraverted"), ("unenergetic", "energetic"), ("silent", "talkative"),
          ("timid", "bold"), ("inactive", "active"), ("unassertive", "assertive"),
          ("unadventurous", "adventurous")],
    "A": [("cold", "warm"), ("unkind", "kind"), ("uncooperative", "cooperative"),
          ("selfish", "unselfish"), ("disagreeable", "agreeable"), ("distrustful", "trustful"),
          ("stingy", "generous")],
    "C": [("disorganized", "organized"), ("irresponsible", "responsible"),
          ("negligent", "conscientious"), ("impractical", "practical"), ("careless", "thorough"),
          ("lazy", "hardworking"), ("extravagant", "thrifty")],
    # N: left pole (first) is high-neuroticism.
    "N": [("angry", "calm"), ("tense", "relaxed"), ("nervous", "at ease"),
          ("envious", "not envious"), ("unstable", "stable"), ("discontented", "contented"),
          ("emotional", "unemotional")],
    "O": [("unintelligent", "intelligent"), ("unanalytical", "analytical"),
          ("unreflective", "reflective"), ("uninquisitive", "curious"),
          ("unimaginative", "imaginative"), ("uncreative", "creative"),
          ("unsophisticated", "sophisticated")],
}
GOLDBERG = []
for tr, pairs in _GB_PAIRS.items():
    for lo, hi in pairs:
        pole_hi = 1 if tr != "N" else -1   # E/A/C/O right pole = high-trait; N right = low-N
        GOLDBERG.append((hi, tr, pole_hi))
        GOLDBERG.append((lo, tr, -pole_hi))

TRAITS = ["E", "A", "C", "N", "O"]
MARKER_SETS = {"saucier": SAUCIER, "goldberg": GOLDBERG}


def snapshot_dir(repo):
    d = os.path.join(HF_HUB, "models--" + repo.replace("/", "--"), "snapshots")
    subs = sorted(glob.glob(d + "/*"))
    return subs[-1] if subs else None


def load_embedding_matrix(repo):
    """Read only the input-embedding tensor from cached safetensors."""
    s = snapshot_dir(repo)
    if s is None:
        return None
    idx = os.path.join(s, "model.safetensors.index.json")

    def is_input_embed(k):
        kl = k.lower()
        return ("embed_tokens.weight" in kl or "backbone.embeddings.weight" in kl
                or kl.endswith("model.embedding.weight")) and "vision" not in kl

    if os.path.exists(idx):
        wm = json.load(open(idx))["weight_map"]
        cands = [k for k in wm if is_input_embed(k)]
        if not cands:
            return None
        key = cands[0]
        shard = os.path.join(s, wm[key])
        with safe_open(shard, framework="pt") as f:
            return f.get_tensor(key).float().numpy(), key
    else:
        st = sorted(glob.glob(s + "/*.safetensors"))
        if not st:
            return None
        for path in st:
            with safe_open(path, framework="pt") as f:
                cands = [k for k in f.keys() if is_input_embed(k)]
                if cands:
                    return f.get_tensor(cands[0]).float().numpy(), cands[0]
    return None


def single_token_id(tok, word):
    """Return the token id if ' word' encodes to exactly one token, else None."""
    for variant in (" " + word, word):
        ids = tok.encode(variant, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0]
    return None


def unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def analyze(W, tok, markers):
    vecs, labels, poles, kept, missed = [], [], [], [], []
    for word, tr, pole in markers:
        tid = single_token_id(tok, word)
        if tid is None or tid >= W.shape[0]:
            missed.append(word)
            continue
        vecs.append(unit(W[tid]))
        labels.append(tr)
        poles.append(pole)
        kept.append(word)
    if len(vecs) < 10:
        return None
    V = np.vstack(vecs)
    cos = V @ V.T
    n = len(vecs)

    # pole-agnostic clustering: do same-trait words group?
    within, across = [], []
    for i in range(n):
        for j in range(i + 1, n):
            (within if labels[i] == labels[j] else across).append(cos[i, j])
    nn_right = 0
    for i in range(n):
        sims = cos[i].copy(); sims[i] = -np.inf
        if labels[int(np.argmax(sims))] == labels[i]:
            nn_right += 1

    # 5-cluster hierarchical purity
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    dist = 1 - cos; np.fill_diagonal(dist, 0); dist = (dist + dist.T) / 2
    Z = linkage(squareform(dist, checks=False), method="average")
    cl = fcluster(Z, t=5, criterion="maxclust")
    from collections import defaultdict
    c2t = defaultdict(list)
    for c, t in zip(cl, labels):
        c2t[int(c)].append(t)
    purity = sum(max((g.count(t) for t in set(g))) for g in c2t.values()) / n

    # meandiff trait directions -> 5x5; near-orthogonality = clean trait axes
    dirs, present = [], []
    for tr in TRAITS:
        hi = [V[i] for i in range(n) if labels[i] == tr and poles[i] == 1]
        lo = [V[i] for i in range(n) if labels[i] == tr and poles[i] == -1]
        if not hi or not lo:
            continue
        dirs.append(unit(np.mean(hi, axis=0) - np.mean(lo, axis=0)))
        present.append(tr)
    Dm = np.vstack(dirs) if dirs else np.zeros((0, V.shape[1]))
    dcos = (Dm @ Dm.T) if len(dirs) else np.zeros((0, 0))
    off = [abs(dcos[i, j]) for i in range(len(dirs)) for j in range(i + 1, len(dirs))]

    return {
        "n_markers_kept": n, "missed": missed,
        "within_mean": float(np.mean(within)), "across_mean": float(np.mean(across)),
        "within_minus_across": float(np.mean(within) - np.mean(across)),
        "nn_purity": nn_right / n, "cluster5_purity": float(purity),
        "trait_dir_offdiag_abs_mean": float(np.mean(off)) if off else float("nan"),
        "traits_with_dir": present,
        "trait_dir_cos": dcos.tolist(),
    }


def main():
    results = {"sets": {}, "models": {}}
    name_to_repo = {n: MODELS[n] for n in [
        "Gemma", "Gemma12", "Gemma27", "Gemma4", "Llama", "Llama8",
        "Phi4", "Qwen", "Qwen7", "Qwen32", "Aya", "FalconMamba"] if n in MODELS}

    for mset in MARKER_SETS:
        results["sets"][mset] = [w for w, _, _ in MARKER_SETS[mset]]

    print(f"{'model':<12}{'set':<10}{'kept':>5}{'within':>8}{'across':>8}"
          f"{'w-a':>7}{'NNpur':>7}{'5clpur':>8}{'dir|off|':>9}")
    print("-" * 74)
    for name, repo in name_to_repo.items():
        try:
            tok = AutoTokenizer.from_pretrained(repo)
        except Exception as e:
            print(f"{name}: tokenizer load failed ({e})"); continue
        loaded = load_embedding_matrix(repo)
        if loaded is None:
            print(f"{name}: embedding tensor not found"); continue
        W, key = loaded
        results["models"].setdefault(name, {"repo": repo, "embed_key": key,
                                            "vocab": int(W.shape[0]),
                                            "hidden": int(W.shape[1]), "sets": {}})
        for mset, markers in MARKER_SETS.items():
            res = analyze(W, tok, markers)
            if res is None:
                print(f"{name} {mset}: too few single-token markers"); continue
            results["models"][name]["sets"][mset] = res
            print(f"{name:<12}{mset:<10}{res['n_markers_kept']:>5}"
                  f"{res['within_mean']:>8.3f}{res['across_mean']:>8.3f}"
                  f"{res['within_minus_across']:>7.3f}{res['nn_purity']:>7.2f}"
                  f"{res['cluster5_purity']:>8.2f}{res['trait_dir_offdiag_abs_mean']:>9.3f}")
        del W

    # cohort means
    print("-" * 74)
    for mset in MARKER_SETS:
        rows = [m["sets"][mset] for m in results["models"].values() if mset in m["sets"]]
        if not rows:
            continue
        print(f"{'COHORT':<12}{mset:<10}{'':>5}"
              f"{np.mean([r['within_mean'] for r in rows]):>8.3f}"
              f"{np.mean([r['across_mean'] for r in rows]):>8.3f}"
              f"{np.mean([r['within_minus_across'] for r in rows]):>7.3f}"
              f"{np.mean([r['nn_purity'] for r in rows]):>7.2f}"
              f"{np.mean([r['cluster5_purity'] for r in rows]):>8.2f}"
              f"{np.nanmean([r['trait_dir_offdiag_abs_mean'] for r in rows]):>9.3f}")

    print("\nHuman reference: Big Five traits near-orthogonal (within-facet cos")
    print("0.42 vs across -0.01). NN-purity/5cl-purity ~1.0 and small within-minus-")
    print("across would mean trait geometry is NOT yet separated in W_E; high")
    print("purity + within>>across means it IS already in the token embeddings.")

    os.makedirs("results/facets", exist_ok=True)
    json.dump(results, open(OUT_PATH, "w"), indent=2)
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
