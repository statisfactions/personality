#!/usr/bin/env python3
"""How close is each LLM's facet geometry to the EMBEDDING geometry? (W13 §3.9 follow-up)

§3.9 asked "does the LLM beat the encoder at recovering the *human* matrix"
(answer: no, excess ~= 0). This asks the more direct thing rgb wanted: how close
is the LLM's own 30x30 facet matrix to the *encoder's* facet matrix, compared to
how close it is to humans, and compared to how close LLMs are to each other?

If the geometry is item semantics, LLM<->encoder should be TIGHTER than
LLM<->human, and ~ as tight as LLM<->LLM (the encoder is "just another model of
the same items"). The LLM-specific residual = (1 - LLM<->encoder r).

Reads only the two cached matrices, no model loads:
  results/facets/embedding_facet_baseline.json  (encoder matrices + r_to_each_model)
  results/facets/ipip_facet_cluster.json         (each model's canonical meandiff matrix)

Honest comparison uses keyed-diff/raw for encoders (§3.9: PC1 is content for
encoders, projecting it out costs r); the model side is the canonical
meandiff-itempc1 (proj) instrument throughout. dwulff is the contaminated upper
reference (fine-tuned on the empirical target), shown for contrast only.

Usage: PYTHONPATH=scripts .venv/bin/python scripts/llm_vs_embedding_geometry.py
"""

import json

import numpy as np

EMB_PATH = "results/facets/embedding_facet_baseline.json"
MODEL_PATH = "results/facets/ipip_facet_cluster.json"
ENC_MODE = "keyed-diff/raw"  # honest encoder extraction (§3.9)
HONEST = ["mpnet-base", "bge-large"]


def upper_tri(M):
    n = M.shape[0]
    return np.array([M[i, j] for i in range(n) for j in range(i + 1, n)])


def matrix_r(A, B):
    return float(np.corrcoef(upper_tri(A), upper_tri(B))[0, 1])


def main():
    emb = json.load(open(EMB_PATH))
    mdl = json.load(open(MODEL_PATH))

    model_mats = {m: np.array(mdl[m]["cosine_matrix"], float) for m in mdl}
    names = list(model_mats)

    # cross-model agreement: each model's mean r to the OTHER models (LLM<->LLM
    # ceiling -- "what does another model of the same items look like").
    llm_llm = {}
    for a in names:
        rs = [matrix_r(model_mats[a], model_mats[b]) for b in names if b != a]
        llm_llm[a] = float(np.mean(rs))

    r_human = emb["model_r_to_human"]                       # LLM <-> human
    enc_human = {e: emb["encoders"][e]["modes"][ENC_MODE]["r_to_human"]
                 for e in emb["encoders"]}

    # LLM <-> encoder (direct matrix r), per honest encoder + dwulff ref
    def r_to_enc(enc):
        return emb["encoders"][enc]["modes"][ENC_MODE]["r_to_each_model"]

    print(f"Facet-matrix similarity (upper-tri Pearson r). Encoder mode = {ENC_MODE}.")
    print(f"LLM<->emb columns: how close the LLM's facet geometry is to the encoder's.\n")
    hdr = (f"{'model':<12}{'mpnet':>8}{'bge':>8}{'<-emb avg':>10}"
           f"{'human':>8}{'LLM-LLM':>9}{'emb-hum':>9}")
    print(hdr); print("-" * len(hdr))

    emb_avg_all, human_all, llmllm_all = [], [], []
    for m in names:
        rm = r_to_enc("mpnet-base")[m]
        rb = r_to_enc("bge-large")[m]
        eavg = np.mean([rm, rb])
        print(f"{m:<12}{rm:>8.3f}{rb:>8.3f}{eavg:>10.3f}"
              f"{r_human[m]:>8.3f}{llm_llm[m]:>9.3f}{'':>9}")
        emb_avg_all.append(eavg); human_all.append(r_human[m]); llmllm_all.append(llm_llm[m])
    print("-" * len(hdr))
    print(f"{'COHORT':<12}{np.mean([r_to_enc('mpnet-base')[m] for m in names]):>8.3f}"
          f"{np.mean([r_to_enc('bge-large')[m] for m in names]):>8.3f}"
          f"{np.mean(emb_avg_all):>10.3f}{np.mean(human_all):>8.3f}"
          f"{np.mean(llmllm_all):>9.3f}")

    print(f"\nencoder<->human (honest, {ENC_MODE}):  "
          + "  ".join(f"{e} {enc_human[e]:+.3f}" for e in HONEST))
    dw = emb["encoders"]["dwulff"]["modes"][ENC_MODE]
    print(f"dwulff (CONTAMINATED) <->human {dw['r_to_human']:+.3f}; "
          f"<->cohort-mean-model {dw['r_to_cohort_mean_model']:+.3f}; "
          f"mean <->each-model {np.mean([dw['r_to_each_model'][m] for m in names]):+.3f}")

    print("\n=== the comparison rgb asked for ===")
    print(f"  cohort mean LLM<->encoder (honest)  {np.mean(emb_avg_all):+.3f}")
    print(f"  cohort mean LLM<->human             {np.mean(human_all):+.3f}")
    print(f"  cohort mean LLM<->LLM (cross-model) {np.mean(llmllm_all):+.3f}")
    print(f"  -> LLM is closer to the encoder than to humans by "
          f"{np.mean(emb_avg_all) - np.mean(human_all):+.3f}")
    print(f"  -> encoder vs another-LLM as a neighbor: "
          f"{np.mean(emb_avg_all):+.3f} vs {np.mean(llmllm_all):+.3f} "
          f"(gap {np.mean(emb_avg_all) - np.mean(llmllm_all):+.3f})")


if __name__ == "__main__":
    main()
