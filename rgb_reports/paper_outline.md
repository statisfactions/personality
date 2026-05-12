# Paper outline — working draft

**Status:** scaffold for the rgb + statisfactions conversation about what to write.
Captures the structure I see in the existing W7–W9 work plus the GFC/TIRT track
statisfactions wired in at `1b4bbd7`. Pre-results for the GFC line; final
shape of §3 of the paper depends on what the trajectory plot looks like
after the 21 HF inferences land.

Last updated: 2026-05-11.

## Working title

**"Three independent readouts of personality in LLMs: representation,
Likert ratings, and forced-choice Thurstonian IRT. All recover human
Big Five structure; apparent method-superiority is mostly vocabulary
coupling."**

(Title is too long but captures the argument; we can tighten later.)

## TL;DR

Across a 7-model cohort (3B–12B; Gemma / Llama / Phi4 / Qwen families)
under matched persona-induction conditions, three measurement tracks
recover human Big Five facet structure independently:

1. **Residual-stream representation** via mean-difference contrast +
   neutral-PC projection (`meandiff-pcs`).
2. **Likert self-report** scored on IPIP-NEO behavioral items.
3. **Forced-choice Thurstonian IRT** on the Okada GFC-30 instrument.

Model facet cosine matrices match the human inter-facet correlation
matrix (N=145,388 from Johnson IPIP-NEO-300) at cohort r=+0.564 under
contrast extraction — direct evidence that models pick up the
empirical Big Five facet covariance from training data, consistently
across 4 architectures and 4× scale range.

The apparent W7 finding that Likert recovers persona z's much better
than Rep (+0.144 gap on Qwen 7B) decomposes substantially into
**vocabulary coupling**: the gap shrinks to ~+0.05 (cohort) when
persona description, rating target, and direction extraction are all
matched on IPIP-form vocabulary instead of Goldberg adjectives. TIRT
serves as a third structurally-clean readout — it eliminates 2 of
the 3 vocabulary couplings by construction (no rating target via
forced choice, no direction extraction via θ estimation), and is
predicted to show a step function with two flat segments across the
W8 trajectory conditions where Rep and Likert vary continuously.

Methodologically, single-direction extraction (without contrast)
fails on the cohort due to residual-stream anisotropy: pairwise
cosine ≈+1.0 between all inputs in absolute space. A matched-format
baseline (`single-ipip-mean`) partially rescues this but
heterogeneously across the cohort — Gemma family + Qwen 7B suffer
incomplete anisotropy cancellation due to extreme norm magnitudes
or extreme alignment. Contrast subtraction is the simplest universal
method.

## Story arc

1. **The puzzle (W7 §11.5.10):** Likert ratings recover sampled persona
   z's at cohort r ≈ +0.89 vs Rep at +0.74 — a +0.14 gap that looked
   like an "instruments measure what activations don't" finding.

2. **The decomposition (W8 §3–§6):** the gap mostly evaporates as
   vocabulary couplings are stripped. Three couplings: persona
   description / rating target / direction extraction. Each contributes
   ~+0.03 to the apparent gap; together ~+0.10. Cohort matched gap is
   +0.052 (raw IPIP) or +0.075 (reflowed IPIP). Per-model variation is
   substantial; Llama 8B is a counterexample where Rep beats Likert.

3. **The third readout (W10, statisfactions):** TIRT on Okada GFC-30 is
   structurally immune to 2 of 3 couplings. Predicted to show a step
   function with two flat segments at the IPIP-raw and IPIP-reflowed
   x-positions while Rep and Likert vary continuously across §3/§4/§5.
   If that prediction holds, vocabulary coupling is confirmed as the
   right diagnosis across a third independent measurement.

4. **The validation anchor (W9 §7):** model facet cosine matrices match
   the human IPIP-NEO-300 facet correlation matrix at cohort r=+0.564.
   This isn't just three transformer architectures agreeing with each
   other — they agree with humans at substantial magnitude, suggesting
   the cross-architecture preservation reflects training-data structure
   rather than just shared architectural priors.

5. **Methods caveats (W9 §1+§7.3):** anisotropy dominates absolute-space
   geometry. Contrast extraction is the simplest method that recovers
   structure universally. Single-direction methods need careful
   baseline matching; even then they fail on high-anisotropy models
   (Gemma family) due to residue effects.

## Headline findings (with key numbers)

| Finding | Number | Source |
|---|---|---|
| Cohort model-vs-human facet correlation (meandiff-pcs) | r = **+0.564** | W9 §7 |
| All cohort models ≥ this human alignment | r ≥ +0.443 (Phi4 floor) | W9 §7 |
| Cross-architecture preservation (meandiff-pcs) | r = +0.940 | W7 §8.4, W8 §9 |
| Within-trait / across-trait cosine ratio (meandiff-pcs) | 5.3× cohort mean | W8 §9 |
| Human within / across trait ratio (N=145k) | 19× | W9 §7 |
| Vocabulary coupling gap (raw) | +0.052 cohort | W8 §6 |
| Vocabulary coupling gap (reflowed) | +0.075 cohort | W8 §8 |
| W7 apparent gap (single Qwen7) | +0.144 | W7 §11.5.10 |
| Cohort raw item-item cosine in absolute space | +0.999+ at most layers | W9 §3 |
| Anisotropy-residue ↔ human-alignment Spearman ρ | −0.93 | W9 §7.3 |

## Section structure (rough)

### Abstract (target: 200 words)

State the three-readout framing, the human-alignment headline, the
vocabulary-coupling decomposition, and the methodological contribution
(anisotropy as the universal obstacle to single-direction extraction).

### 1. Introduction

- Motivation: Personality measurement in LLMs across three competing
  methodological traditions: psychometric self-report (Serapio-Garcia
  et al. 2023), representation engineering (Zou et al. 2023, Sofroniew
  et al. 2026), and forced-choice IRT (Okada et al. 2026).
- Each tradition reports robust findings. Are they measuring the same
  thing?
- This paper: yes, with substantial agreement (cohort r vs human ≈
  +0.56), but apparent method-superiority claims reflect vocabulary
  coupling more than fundamental method differences.

### 2. Methods

- **Cohort:** 7 cohort models, all bf16 HF inference: Gemma 3 4B / 12B,
  Llama 3.2 3B / 3.1 8B, Phi-4-mini, Qwen 2.5 3B / 7B. M5 Max 128 GB.
- **Instruments:** IPIP-NEO-300 (30 facets × 5 traits), Goldberg
  adjective markers, Okada GFC-30 forced-choice instrument.
- **Persona induction:** sampled z ∈ ℝ⁵ persona vectors, mapped to
  three description forms (Goldberg markers, IPIP-raw, IPIP-reflowed
  via Sonnet paraphrase).
- **Readout 1 — Representation:** mean-difference contrast with
  neutral-PC projection at ~2/3 depth. Token aggregation is
  mean-all-tokens-skip-BOS on chat-wrapped items. See
  `representation_vector_methods.md`.
- **Readout 2 — Likert:** distributional logprobs over {1,...,5} on
  chat-wrapped IPIP items with persona-as-system-message.
- **Readout 3 — TIRT:** Okada GFC-30 graded forced-choice with
  Thurstonian IRT estimation; Stan model `tirt_okada_indep.stan`.
- **Human reference:** Johnson IPIP-NEO-300 raw data via
  NeuroQuestAi/ipip-neo-data, N=145,388, scored via standard
  Goldberg/Johnson 1999 key.

### 3. Results

- **3.1. Three readouts agree on persona z-recovery** — the W8 §6
  cohort table for Rep and Likert; statisfactions's TIRT line as
  the third row. Figure: the W8 trajectory plot with three lines.
  Pending pipeline output.

- **3.2. Models recover human facet covariance** — the W9 §7
  comparison table (per-model × per-method r vs human matrix).
  Figure: `ipip_facet_vs_human_dashboard.html` (8-panel side-by-side).

- **3.3. Cross-architecture preservation is human-aligned** — the
  W8 §9 cohort r=+0.94 reframes as the human-aligned axis (since
  same-axis r=+0.56 model-human, +0.94 model-model). Figure:
  `ipip_facet_dashboard.html` cross-model heatmaps.

- **3.4. Vocabulary coupling decomposition** — W8 §3–§6 narrative.
  Three couplings, each ~+0.03; together ~+0.10 of the apparent gap
  on Qwen7. TIRT confirms by structural immunity (3.1 figure).

- **3.5. Per-model heterogeneity** — Phi4 outlier on rep extraction,
  Llama 8B as pro-rep counterexample, Qwen family "indifferent" gap.
  Reflects post-training recipe differences. Figure: per-model gap
  plot (`persona_w8_per_model_gap.html`).

### 4. Discussion

- Three different measurement traditions agree on the headline; this
  is itself a methodological contribution (suggests they're not
  measuring fundamentally different constructs).
- Vocabulary coupling as a confound: applies whenever you measure a
  representation against a specific lexical instrument (most prior
  work). Even the "behavioral" claim of FC measurement carries
  persona-side vocabulary coupling.
- Training-data origin: models converge on human facet structure
  because they trained on human-generated text encoding that
  structure. Doesn't entail anthropomorphic interpretation — the
  model represents the *concept* of personality, not necessarily
  has a personality (rgb's "representation isn't intention" stance,
  cf. `memory/user_representation_not_intention.md`).
- Open: read/write dissociation (W4 finding) means representation
  reads don't entail behavioral writes. Persona induction here
  may exploit one channel without confirming the other.

### 5. Methods supplement (or appendix)

- The five extraction methods (W9 §1) — anisotropy diagnostic,
  why contrast works, why single-direction needs matched baseline.
- The Gemma anisotropy-magnitude failure mode and Qwen7 anisotropy-
  alignment failure mode (W9 §7.3).
- TIRT identification issues — the {A,C} sign-flip basin and whole-
  vector reflection on binary forced choice (ecb-reports W11).
- Token aggregation heterogeneity (HEXACO mean-response vs IPIP
  mean-all-skip0) and what we know about layer sensitivity.

### 6. Limitations

- Cohort is 4 architectures × 2 size points each (3B and 7-12B); no
  models above 12B, no base models, no frontier models. Larger or
  differently-trained models could behave qualitatively differently.
- All persona induction is via written description; we don't measure
  whether the model would behave consistently in extended interaction
  (the "read/write gap").
- Human reference is a self-report convenience sample (N=145k US
  volunteers), not a representative population sample. The trait
  structure should be robust to this; absolute means / variances
  should not.
- Single-direction extraction failure on Gemma family + Qwen 7B may
  be a methodological artifact rather than a substantive finding —
  the anisotropy-projection variant in W10 next-steps would test this.
- TIRT recovery on open models is structurally lower than Likert
  (statisfactions reports |r|≈0.20–0.35 for graded forced-choice vs
  Rep/Likert ~0.70). This is a real methodological caveat — see
  `ecb-reports/week08_pool_scaleup_27b_70b_72b.md`.

### 7. Future work

- Per-facet rep direction × persona facet z-recovery (chunking-
  granularity hypothesis test, W8 §9 / W9 §7.4).
- Anisotropy-projection variant of single-direction (W9 §11.1).
- Read/write reconnection: does representation-induced persona persist
  through extended generation? (Connects to Wu et al. 2026 "Knowing
  without Acting" and our W4 backprop-steering work.)
- Trait-conflict dilemma instrument (open since W3, listed in
  `to_try.md`).

## Figures (pointers to existing artifacts)

- **F1. W8 trajectory plot (Rep / Likert / TIRT × 5 conditions × 7
  cohort models)**, `results/persona_w8_trajectory.html`. The TIRT
  step function is the central methodological argument. Awaiting
  GFC pipeline output (task `b4ng0i6d7`, ETA ~1 hour).
- **F2. Model facet cosine matrices vs human (8-panel)**,
  `results/facets/ipip_facet_vs_human_dashboard.html`. The headline
  empirical figure. Already built.
- **F3. Five-method comparison dashboard**,
  `results/facets/ipip_facet_method_dashboard.html`. Methods supplement.
- **F4. Per-model raw → reflow gap**,
  `results/persona/persona_w8_per_model_gap.html`. Heterogeneity story.
- **F5. Cross-model cosine-matrix correlation heatmap (meandiff-pcs vs
  single-ipip-mean)** — from the method dashboard. Methods supplement.

## What's drafted vs missing

**Drafted (in repo as `rgb_reports/`):**

- W7 §8.4 cross-stimulus / cross-model analysis (`report_week7.md`)
- W7 §11.5.10 first persona-prereg (`report_week7.md`)
- W8 vocabulary coupling decomposition + reflow ablation
  (`report_week8.md`, ~9000 words)
- W8 §9 facet cluster geometry (`report_week8.md`)
- W9 §1 five-extraction method ablation (`report_week9.md`)
- W9 §7 human-alignment comparison (`report_week9.md`)
- W9 §7.3 anisotropy-residue diagnostic (`report_week9.md`)
- Methods catalog (`representation_vector_methods.md`)
- Bibliography (`bibliography.md`)

**Drafted in `ecb-reports/`** (statisfactions, TIRT track):

- Okada graded TIRT scale-up to 27B/70B/72B (`week08_pool_scaleup_...`)
- Okada binary TIRT comparison (`week11_okada_binary_tirt.md`)
- Okada recovery diagnosis (`week07_okada_recovery_diagnosis.md`)
- Pooled replication + Stan model (`week07_okada_pooled_replication.md`)
- Haiku Stan replication (`week07_okada_replication_haiku_stan.md`)

**Missing for paper crystallization:**

- 21 GFC HF inferences + TIRT fits (pipeline running; ETA ~1 hour
  inference + ~minutes/model for fits)
- W8 trajectory plot with TIRT line populated
- Unified abstract + intro tying both tracks together
- Decision on whether F1 (trajectory plot) or F2 (human alignment)
  is the lead figure — they tell different sides of the story
- Methodology supplement structure (separate file or appendix in paper?)
- Authorship / contribution statement; venue target

## Open questions for the rgb + statisfactions chat

1. **Venue / scope.** Methods paper at NeurIPS / ICLR workshop?
   Empirical paper at a psychometric venue (J. Personality & SocPsy,
   Psychometrika)? Position paper at TMLR? The three-readout framing
   plays differently in each.

2. **TIRT track integration.** statisfactions has substantial work on
   Okada replication + scale-up + binary TIRT identification issues.
   How much of this is paper-relevant vs supplement-relevant?
   Specifically: do the {A,C} sign-flip / whole-vector-reflection
   identification problems deserve a §6.2 or do they live in the
   methods supplement?

3. **Single-direction story scope.** The W9 §1 anisotropy/methods
   finding is methodologically interesting but secondary to the
   three-readout / human-alignment story. Include in main text or
   supplement only?

4. **Cohort coverage.** We have 3B–12B; statisfactions also has TIRT
   numbers on Haiku 4.5, Gemma 3 27B, Llama 3.3 70B, Qwen 2.5 72B.
   Do we extend rgb's Rep/Likert track to those models for full
   matched coverage, or keep tracks at different cohort sizes?

5. **The chunking-granularity hypothesis (to_try #18).** Open since
   W7. W9 §7 partially addresses it (Big Five facets ARE the model-
   natural chunks at r=+0.56). Worth a Phase B / follow-up paper or
   does it belong in the discussion?

6. **Read/write dissociation.** W4 backprop-steering work showed a
   substantial knowledge-action gap. Does this paper address it
   (briefly, in discussion) or hold it for a follow-up?

7. **Lead figure decision.** F1 (trajectory plot with three lines) is
   the methodological hero. F2 (model vs human dashboard) is the
   empirical headline. The intro framing depends on which we lead with.

8. **Authorship + naming convention.** statisfactions's work has been
   in `ecb-reports/`; rgb's in `rgb_reports/`. For the paper we'd
   want a single source-of-truth working draft — separate file? Shared
   doc? Branch?

## Status

Pipeline running (background task `b4ng0i6d7`, currently step 5/21 at
~3 min/step). R/rstan installed for stage 2 TIRT fits. Methods +
data + report material in place for the four core results sections.
Once GFC numbers land we have everything needed to start drafting
the paper text.
