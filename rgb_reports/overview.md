# rgb_reports — overview & index

Worklogs for the distributional-logprobs / RepE / forced-choice track (rgb). The
numbered "weeks" are **work-phases, not calendar weeks** — the label count has run
ahead of the calendar because several calendar weeks absorbed two phases each.

**Calendar check (as of 2026-06-01):** the first measurement commit (`ba18e89`,
distributional Ollama logprobs) landed **2026-03-23**, so we are **~10–11 calendar
weeks in** — yet the labels read "Week 13." Four calendar weeks doubled up on
phases: wk1 = W1+W2, wk2 = W3+W4, wk8 = W10+W11, wk9 = W12+W13(start). The
W10/W11 pair is the sharpest compression (both in a 3-day burst, May 11–13).
Treat the labels as stable identifiers, not week numbers.

| Label | Report | Dates (git) | Cal wk | One-line summary (~25 words) |
|---|---|---|---|---|
| W1 | `report.md` | 03-22 → 03-28 | 1 | Distributional logprob surveys (IPIP-300, HEXACO-100) via Ollama; ICC reliability across 4 phrasings; mixture model splitting variance into shared-assistant + unique + noise. Argmax is the mask, the distribution is the signal. |
| W2 | `report_week2.md` | 03-29 | 1 | RepE trait vectors from 300 contrast pairs; PCA-PC1-is-a-norm-artifact discovery; format-invariant period-token protocol; first steering attempt fails. Pivot from self-report to scenario-based measurement. |
| W3 | `report_week3.md` | 03-31 → 04-04 | 2 | Five-test validation suite across 4 models (layer/framing sensitivity, RepE-vs-Likert, Röttger, cross-model transfer); the three-construct dissociation — representation ≠ preference ≠ free-text. |
| W4 | `report_week4_steering.md` | 04-05 | 2 | Backprop-optimized, norm-constrained steering vectors; the knowledge-action gap (LDA classifies 100% but won't steer); three steering objectives compared; cross-method 5×5 correlation matrix. |
| W5 | `report_week5_meandiff.md` | 04-11 → 04-12 | 3 | Mean-diff (Sofroniew-style) vs LDA extraction; stratified audit-aware holdout; position-bias and chat-template confounds surfaced; FC→BC rename (our pipeline is binary choice, not forced choice). |
| W6 | `report_week6.md` | 04-18 → 04-21 | 4 | LDA→logistic-regression swap (Σ⁻¹-noise in the n≪d regime); facet-level structure shows LLM representation does *not* follow HEXACO factors; contrast-vs-disposition reframing. |
| W7 | `report_week7.md` | 04-26 → 05-02 | 5 | Full battery on a larger cohort (Gemma-12B, Llama-8B, Qwen-7B); HF inference port; BC↔RepE sign-flip at scale; character preserves but reliability is family-bound; symbolic-vs-associative theory born. |
| W8 | `report_week8.md` | 05-02 → 05-03 | 6 | Natural-persona prereg pilot. "Rep is vocabulary-bound, reasoning is vocabulary-free" — the W7 +0.144 Likert-over-rep gap is largely a Goldberg-vocabulary-coupling artifact; reflow ablation. |
| W9 | `report_week9.md` | 05-10 → 05-21 | 7 | Single-direction representations and anisotropy — contrast extraction was hiding pre-norm anisotropy (PC1≈norm); human facet-correlation comparison (N=145,388); Cheerfulness/Liberalism case studies. |
| W10 | `report_week10.md` | 05-11 → 05-13 | 8 | Ran statisfactions's GFC/TIRT pipeline as the third readout across the 7-model cohort × 3 persona forms; TIRT recovers persona z's at lower magnitude than Rep or Likert. |
| W11 | `report_week11.md` | 05-13 | 8 | Built a desirability-matched graded-forced-choice instrument (IPIP-NEO-GFC-60) via constrained MIP with cohort raters; TIRT is structurally SDR-immune; surprise negative Phase-D recovery result. |
| W12 | `report_week12.md` | 05-16 → 05-23 | 9 | TIRT loading diagnostic (cohort loadings pinned at the prior mean; per-item relative loadings show clean assistant-shape); the 270-cube (3 method × 3 form × 3 condition × 10 model); axis-of-the-models reframe. |
| W13 | `report_week13.md` | 05-24 → 06-01 | 9–11 | Axis-of-the-models execution (Aya + Falcon Mamba land on the cohort facet axis); Mandarin IPIP-120 cross-language persona; embedding-baseline scoop → adjective geometry: intensity-over-valence, Big Five a thin overlay. |
| W14 | `report_week14.md` | 06-01 | 11 | Is the adjective Big Five real? Three over-extraction routes (rotation stability, bass-ackwards, respondent bootstrap) + a Kaiser/SPSS varimax fix — no stable 6th factor, model collapses to a 2-factor evaluative core. §2 reconciles that with the r≈0.56 IPIP facet match: it's a metric difference (relational match high & stimulus-invariant; dimensional Big-Five recovery low). §3 localizes the disagreement to Openness (valence-neutral) + the non-dispositional tail. §4: the core is two near-orthogonal valence poles (not an intensity factor); shared PC1 carries ~70% of the match. |
| W15 | `report_week15.md` | 06-01 → … | 11 | Representation vs introspection: the behavioral bridge. Every model 3B→32B *represents* Wonderful≈Awful as merged but *judges* them opposite (sign-flip onto the human antonym value) — localized read/write dissociation; symbolic overrides associative; not size-gated. §2: judgment is near-PSD (two coherent geometries); persona/ToM judgment also splits (resolves the semantics-vs-ToM confound) and *overshoots* humans with an evaluative halo that scales with size. §3: weights vs context (Qwen + OLMo-2 ladder) — halo weight-resident (chat≈bare); only robust cross-family claim is that the **representational merge is pretrained-constant** (flat all stages, both families); the behavioral split is family-dependent (Qwen base voices it + entropy collapses; OLMo builds it across SFT→DPO→RLVR, entropy ~flat). Single-model conclusion revised 4×. |
| W16 | `report_week16.md` | 06-03 → … | 11–12 | How far back does the Wonderful≈Awful merge go? Regress over model classes (LLM → encoders → static GloVe/komninos → human): every distributional-geometry model merges the eval-antonyms (antonym-z>0 from 2014 static vectors up); only LLM-judgment and human self-report split them (−0.49 / −0.53, identical). The merge is a property of the distributional hypothesis, not the architecture — merge magnitude isn't even era-monotonic (LLM-repr merges hardest, 2016 word2vec harder than 2023 encoder). §2 reasons the rest (LSTM merges, HMM/Brown-cluster merges harder/total, CRF is a category error). The write side is the only thing that escapes the regress — and it's the new thing. §4 methods aside: tested unifying the paired+unpaired denoise to one IPR-gated `meandiff-adaptive` rule — costs −0.010 cohort / −0.08 FalconMamba (keeping PC1 hurts low-anisotropy models), so the fixed-for-paired / adaptive-for-unpaired split stays canonical; principled because the paired contrast makes PC1 never-the-signal. |
| W17 | `report_week17.md` | 06-30 → … | 14 | ENACT is a linear image of REPRESENT (llama3.2, same activation space): per-adjective alignment (retrieval ~40%), held-out ridge R²=0.72 compressing 45→10 effdims onto the human-matched evaluative core; the 64%-out-of-span part is a rotation, not an intent space; the unpredictable residual is reliable but personality-free. Causal test: mapped ê=Wr steers as well as or better than the recorded persona vector (the residual does no behavioral work — functional denoising); read vectors steer *topic* not *conduct* at ~¼ potency. Zero-rollout persona-vector recipe. §8: qwen2.5 replicates everything structural; family parameter is the rotation size — qwen keeps 62% of ENACT in the read span (llama 36%) and its raw read vectors steer conduct at near-enact potency. |

## Supporting documents (not week-numbered)

| File | What it is |
|---|---|
| `synthesis.md` | **Digestible state-of-the-project narrative** (for statisfactions re-onboarding) — the whole arc by theme, glossing false starts. Start here. |
| `methodology.md` | Full instrument/script reference with flags and output paths. |
| `result_summary.md` | Topline numbers from every instrument across all 4 original models. |
| `representation_vector_methods.md` | The five single-direction extraction methods (W9 reference). |
| `cross_method_correlations.md` | 5×5 correlation matrix across methods (Likert, BC, RepE). |
| `scenario_audit.md` | Per-pair signal audit; construct-heterogeneity-within-traits finding. |
| `paper_outline.md` | Crystallization outline for the rgb/statisfactions write-up. |
| `to_try.md` | Long-form backlog with rationale. |
| `bibliography.md` | Annotated reading list, ordered by influence on the project. |
| `lit_review_scenario_personality.md` | Scenario-based personality measurement literature. |
| `lit_review_steering_vectors.md` | Gradient-based steering vector literature. |
| `week5_discussion.md` | Condensed reading-group write-up of Week 5. |
