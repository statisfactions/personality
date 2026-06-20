"""What ARE the cloud axes? Correlate REPRESENT/ENACT/HUMAN PC1,PC2 against
human-derived construct scores (W17), to test the eyeballed interpretations
(Gemma ENACT=competence x warmth, Llama=dominance x valence, Qwen=competence x
extraversion; REPRESENT=intensity x valence).

Construct score per adjective = mean human correlation with a seed set (seeds
filtered to those present in 523-PDA). Bipolar constructs use pos-minus-neg.
Also valence-residualized versions (regress out valence) to see what a construct
adds BEYOND good/bad. Report corr(PC1,construct), corr(PC2,construct), and
in-plane R=sqrt(r1^2+r2^2) (PC1_|_PC2). NOTE: when lambda1~lambda2 (Gemma) the
PC1/PC2 split is rotationally degenerate -> trust in-plane R, not the split.
"""
import numpy as np
import four_grid_compare as fg
from adjective_introspection import GROUPS
from hf_logprobs import display

SEEDS = {
    "warmth":     ["warm", "affectionate", "kind", "gentle", "loving",
                   "sympathetic", "caring", "tender", "sweet"],
    "dominance":  ["bold", "assertive", "aggressive", "strong", "dominant",
                   "forceful", "fearless", "outspoken"],
    "competence": ["competent", "intelligent", "capable", "skilled",
                   "efficient", "smart", "clever", "brilliant", "skillful"],
    "extraversion": ["outgoing", "talkative", "sociable", "extraverted",
                     "lively", "enthusiastic", "social"],
    "arousal":    ["energetic", "active", "lively", "excitable", "passionate",
                   "restless", "intense"],
}


def main():
    H, labels = fg.load_human()
    low = [l.lower() for l in labels]
    idx = {l: i for i, l in enumerate(low)}
    Hn = np.nan_to_num(H, nan=0.0)

    pos = [idx[w.lower()] for w in GROUPS["pos_eval"]]
    neg = [idx[w.lower()] for w in GROUPS["neg_eval"]]
    valence = np.array([np.nanmean(H[i, pos]) - np.nanmean(H[i, neg])
                        for i in range(len(low))])

    def score(seeds):
        s = [idx[w] for w in seeds if w in idx]
        return Hn[:, s].mean(1), len(s)

    constructs = {"valence": valence, "extremity": np.abs(valence)}
    for name, seeds in SEEDS.items():
        sc, n = score(seeds)
        constructs[name] = sc
        if n < 3:
            print(f"  !! {name}: only {n} seeds present")
    # valence-residualized (what each adds beyond good/bad)
    v = valence - valence.mean()
    for name in list(SEEDS):
        c = constructs[name] - constructs[name].mean()
        constructs[name + "~v"] = c - (c @ v) / (v @ v) * v

    def mds(C, diag):
        C = C.astype(float).copy()
        iu = ~np.eye(len(C), dtype=bool)
        C[np.isnan(C) & iu] = np.nanmean(C[iu]); np.fill_diagonal(C, diag)
        n = len(C); P = np.eye(n) - 1 / n; G = P @ C @ P
        w, V = np.linalg.eigh((G + G.T) / 2); o = np.argsort(w)[::-1]
        coords = V[:, o[:2]] * np.sqrt(np.clip(w[o[:2]], 0, None))
        if np.corrcoef(coords[:, 0], valence)[0, 1] < 0:
            coords[:, 0] *= -1
        return coords, w[o[0]] / w[o[1]]

    def report(tag, C, diag):
        xy, ratio = mds(C, diag)
        deg = " [DEGENERATE PC1/PC2 — trust in-plane R]" if ratio < 1.4 else ""
        print(f"\n=== {tag}  (λ1/λ2={ratio:.1f}){deg}")
        print(f"{'construct':>14s} {'r·PC1':>7s} {'r·PC2':>7s} {'in-plane R':>11s}")
        rowdat = []
        for cn, cs in constructs.items():
            r1 = np.corrcoef(xy[:, 0], cs)[0, 1]
            r2 = np.corrcoef(xy[:, 1], cs)[0, 1]
            rowdat.append((cn, r1, r2, (r1**2 + r2**2)**0.5))
        for cn, r1, r2, R in sorted(rowdat, key=lambda t: -t[3]):
            if R > 0.25 or cn in ("valence",):
                print(f"{cn:>14s} {r1:+7.2f} {r2:+7.2f} {R:11.2f}")

    report("HUMAN", H, 1.0)
    for m in ["gemma3", "llama3.2", "qwen2.5"]:
        report(f"{display(m)} REPRESENT", fg.load_represent(m, labels)[0], 1.0)
        report(f"{display(m)} ENACT", fg.load_enact(m, labels, ablate=True), 1.0)


if __name__ == "__main__":
    main()
