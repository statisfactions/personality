"""ValuePortrait generation-behavior rescoring: does within-scenario contrast
scoring rescue the structure that arXiv:2509.10078 reports as absent?

Their method: score(c) = mean log P(r|scenario) over responses tagged c
(|r|>0.3), pooled ACROSS scenarios — length/fluency/scenario confounds swamp
construct variance, and the VP labels are SIGNED continuous correlations
(their formula states no sign handling).

Our method: p_r = log P(response|scenario) per token, z-scored WITHIN each
scenario's 5 candidates (ipsatized preference); construct profile = corr(p,
signed label column) over all 520 responses.

Readouts per model:
  A. split-half reliability (100 random scenario splits) of the 15-construct
     profile, theirs vs ours — the stat their paper doesn't report.
  B. construct profile under both scorings.
  C. their eta^2-style construct clustering on |r|>0.3 primary tags.
Models: gemma3 (their Gemma3-4B), Qwen7 (their Qwen2.5-7B), llama3.2,
qwen2.5, phi4. Saves per-response logprobs to results/vp_rescore/<m>.json.
"""
import json
import os

import numpy as np
import torch

import hf_logprobs as hf

SP = "dependencies"  # scripts/fetch_dependencies.sh clones ValuePortrait here
OUT = "results/vp_rescore"
MODELS = ["gemma3", "Qwen7", "llama3.2", "qwen2.5", "phi4"]
PVQ = ["Benevolence", "Universalism", "Self_Direction", "Stimulation",
       "Hedonism", "Achievement", "Power", "Security", "Conformity",
       "Tradition"]
BFI = ["Extraversion", "Agreeableness", "Conscientiousness", "Neuroticism",
       "Openness"]
CONSTRUCTS = PVQ + BFI


def load_vp():
    d = json.load(open(f"{SP}/ValuePortrait/data/query-response-tagged/"
                       "query-response-tagged.json"))
    rows = []
    for qi, e in enumerate(d):
        c = e["content"]
        scen = ((c.get("title") or "") + "\n\n" + (c.get("text") or "")).strip()
        for o in e["outputs"]:
            lab = {k: 0.0 for k in CONSTRUCTS}
            for k, r in o.get("correlations", []):
                if k in lab:
                    lab[k] = r
            for k, r in o.get("bfi_correlations", []):
                if k in lab:
                    lab[k] = r
            rows.append(dict(q=qi, scen=scen, resp=o["content"],
                             lab=[lab[k] for k in CONSTRUCTS]))
    return rows


@torch.no_grad()
def resp_logprob(model, tok, device, scen, resp):
    msgs = [{"role": "user", "content": scen}]
    pre_s = tok.apply_chat_template(msgs, tokenize=False,
                                    add_generation_prompt=True)
    full_s = tok.apply_chat_template(
        msgs + [{"role": "assistant", "content": resp}],
        tokenize=False, add_generation_prompt=False)
    pre = tok(pre_s, add_special_tokens=False).input_ids
    full = tok(full_s, add_special_tokens=False).input_ids
    ids = torch.tensor([full], device=device)
    logits = model(ids).logits[0].float().log_softmax(-1)
    npre = len(pre)
    tgt = ids[0, npre:]
    lp = logits[npre - 1:-1].gather(1, tgt.unsqueeze(1)).squeeze(1)
    return lp.sum().item(), len(tgt)


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = load_vp()
    print(f"{len(rows)} responses / {len(set(r['q'] for r in rows))} scenarios")
    L = np.array([r["lab"] for r in rows])            # (520, 15) signed
    Q = np.array([r["q"] for r in rows])

    for m in MODELS:
        f = f"{OUT}/{m}.json"
        if os.path.exists(f):
            print(f"[skip] {f}")
            continue
        model, tok, device = hf.load_model(m, dtype=torch.bfloat16)
        lps, lens = [], []
        for i, r in enumerate(rows):
            lp, nt = resp_logprob(model, tok, device, r["scen"], r["resp"])
            lps.append(lp)
            lens.append(nt)
            if (i + 1) % 100 == 0:
                print(f"  {m} {i+1}/{len(rows)}", flush=True)
        json.dump(dict(model=m, logprob=lps, ntok=lens,
                       constructs=CONSTRUCTS),
                  open(f, "w"))
        del model
        torch.mps.empty_cache()
        print(f"wrote {f}")

    # ---------- analysis ----------
    rng = np.random.default_rng(7)
    qids = np.unique(Q)
    print("\n=== analysis (100-split split-half reliability of 15-construct "
          "profile) ===")
    for m in MODELS:
        f = f"{OUT}/{m}.json"
        if not os.path.exists(f):
            continue
        d = json.load(open(f))
        lp = np.array(d["logprob"], float)
        nt = np.array(d["ntok"], float)
        ppt = lp / nt                                  # per-token logprob

        # ours: within-scenario z of per-token logprob
        p = np.zeros(len(lp))
        for q in qids:
            s = Q == q
            p[s] = (ppt[s] - ppt[s].mean()) / (ppt[s].std() + 1e-9)

        def profile_ours(mask):
            return np.array([np.corrcoef(p[mask], L[mask, c])[0, 1]
                             for c in range(len(CONSTRUCTS))])

        # theirs: mean raw logprob over |r|>0.3 tagged responses, unsigned
        def profile_theirs(mask):
            out = []
            for c in range(len(CONSTRUCTS)):
                sel = mask & (np.abs(L[:, c]) > 0.3)
                out.append(lp[sel].mean() if sel.sum() else np.nan)
            return np.array(out)

        def splithalf(fn):
            rs = []
            for _ in range(100):
                qs = rng.permutation(qids)
                m1 = np.isin(Q, qs[: len(qs) // 2])
                a, b = fn(m1), fn(~m1)
                ok = np.isfinite(a) & np.isfinite(b)
                rs.append(np.corrcoef(a[ok], b[ok])[0, 1])
            return np.mean(rs)

        prof = profile_ours(np.ones(len(lp), bool))
        top = np.argsort(-np.abs(prof))[:5]
        print(f"\n{m}: split-half OURS {splithalf(profile_ours):+.2f}   "
              f"THEIRS {splithalf(profile_theirs):+.2f}")
        print("  profile (ours, top |r|): " +
              ", ".join(f"{CONSTRUCTS[c]} {prof[c]:+.2f}" for c in top))
    np.save(f"{OUT}/labels.npy", L)


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Extended analysis (run after the main scoring pass; pure CPU):
#   PYTHONPATH=scripts python scripts/vp_rescore.py --analyze
# Adds: desirability-partialled (label-PC1-removed) split-half; cross-method
# Spearman vs our IPIP-300 ev_mean Big5. Findings 2026-07-25 in W18 §7.
# ---------------------------------------------------------------------------
def analyze():
    from scipy.stats import spearmanr
    L = np.load(f"{OUT}/labels.npy")
    Lz = (L - L.mean(0)) / (L.std(0) + 1e-9)
    Q = np.repeat(np.arange(104), 5)
    rng = np.random.default_rng(7)
    u, s, vt = np.linalg.svd(Lz, full_matrices=False)
    pc1 = vt[0]
    dscore = Lz @ pc1
    print("label PC1 (desirability axis), var explained "
          f"{s[0]**2/np.sum(s**2):.2f}")

    def pref(m):
        dd = json.load(open(f"{OUT}/{m}.json"))
        lp = np.array(dd["logprob"]) / np.array(dd["ntok"])
        p = np.zeros(len(lp))
        for q in range(104):
            sq = Q == q
            p[sq] = (lp[sq] - lp[sq].mean()) / (lp[sq].std() + 1e-9)
        return p

    def profile(p, mask, Lm):
        return np.array([np.corrcoef(p[mask], Lm[mask, c])[0, 1]
                         for c in range(Lm.shape[1])])

    def splithalf(p, Lm):
        rs = []
        for _ in range(100):
            qs = rng.permutation(104)
            m1 = np.isin(Q, qs[:52])
            rs.append(np.corrcoef(profile(p, m1, Lm),
                                  profile(p, ~m1, Lm))[0, 1])
        return np.mean(rs)

    Lres = Lz - np.outer(dscore, pc1)
    KEY = {"Extraversion": "IPIP300-EXT", "Agreeableness": "IPIP300-AGR",
           "Conscientiousness": "IPIP300-CON", "Neuroticism": "IPIP300-NEU",
           "Openness": "IPIP300-OPE"}
    SURVEY = {"gemma3": "Gemma", "llama3.2": "Llama", "qwen2.5": "Qwen",
              "phi4": "Phi4", "Qwen7": "Qwen7"}
    for m in MODELS:
        if not os.path.exists(f"{OUT}/{m}.json"):
            continue
        p = pref(m)
        beta = np.dot(p, dscore) / np.dot(dscore, dscore)
        p_res = p - beta * dscore
        line = (f"{m:9s} split-half raw {splithalf(p, Lz):+.2f}  "
                f"desirability-partialled {splithalf(p_res, Lres):+.2f}")
        sf = SURVEY.get(m)
        if sf and os.path.exists(f"results/surveys/{sf}_ipip300.json"):
            ss = json.load(open(f"results/surveys/{sf}_ipip300.json"))["scale_scores"]
            gv = [np.corrcoef(p, Lz[:, CONSTRUCTS.index(c)])[0, 1] for c in BFI]
            qv = [ss[KEY[c]]["ev_mean"] for c in BFI]
            line += f"  cross-method(BFI5) rho {spearmanr(qv, gv)[0]:+.2f}"
        print(line)


if __name__ == "__main__" and "--analyze" in os.sys.argv:
    analyze()
