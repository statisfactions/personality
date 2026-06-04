# The state of the rgb track — a digestible synthesis

**For:** statisfactions, getting back in after grading.
**Written:** 2026-06-03.
**Scope:** the whole rgb (distributional-logprobs / RepE / forced-choice) arc,
organized by theme rather than by week, glossing the false starts. The
week-by-week index is `overview.md`; the paper plan is `paper_outline.md` (last
touched ~W10, so it predates everything in §4–§6 below). This doc is the bridge
from "where we were when you got busy" to "where we are now."

---

## TL;DR — if you read one paragraph

We measure LLM personality through three readouts — what the model **represents**
(residual-stream geometry), what it **prefers** (Likert / binary choice), and
what it **writes** (free text). The throughline of the whole project is that
these three disagree, and the disagreements are informative. The biggest recent
result: models **represent** evaluative antonyms (Wonderful ≈ Awful) as *merged*
neighbors but **judge** them as opposite — a clean, universal read/write
dissociation at the lexical level. And the merge turns out not to be a transformer
artifact or a scale effect: it is already complete in 2014-era static word
vectors. It is a property of the distributional hypothesis itself. What is *new*
in LLMs is the **write** side — the capacity to hold a symbolic distinction that
overrides the associative merge when the model is asked to judge. That reframes
our contribution: the geometry everyone (Wulff, Milano) is now publishing is the
old, inherited part; the override is ours.

---

## 1. The founding move: distribution > argmax

The first thing that made this track its own thing (vs. the Serapio-Garcia
self-report approach) was reading the **full logprob distribution** over a
Likert response, not the argmax. The selected answer is a mask; the distribution
is the signal. Two models can both "answer 4" while one is razor-peaked and the
other near-uniform — and that difference is the personality-relevant quantity.

- Gemma / Qwen are peaked (digit-entropy ≈ 0.15); Llama is near-uniform (≈ 1.4).
- This is why entropy keeps showing up as a load-bearing variable later (it is a
  detectability / confidence axis, not just noise).

## 2. The assistant shape

Every model, scored as itself, lands **low-Neuroticism, high-Agreeableness /
Conscientiousness**. In Big Five space the HHH (helpful-honest-harmless) assistant
persona is roughly **rank-1**: E–C correlate at r ≈ 0.93. HEXACO partially resists
this collapse because Honesty-Humility maps directly onto HHH and pulls out as its
own axis. Practical consequence: a lot of apparent "trait structure" in a tuned
model is one evaluative assistant axis wearing five names. This recurs as the
"evaluative halo" in §5.

## 3. Three readouts that disagree (the spine)

| Readout | What it captures | How we get it |
|---|---|---|
| **Representation** | what the model *encodes* about a trait/word | hidden states at ~2/3 depth, denoised. **Paired** data (facet/persona contrast pairs): `meandiff-itempc1` — `unit(project_out(mean(fwd) − mean(rev), item top-1 PC))`. **Unpaired** data (single adjectives): `adaptive_denoise` — center, and remove PC1 *only* when it's a concentrated rogue dim (IPR < 10 ≈ Gemma's massive activations), else keep it. |
| **Preference** | what the model *picks* | distributional logprobs over Likert / binary choice |
| **Free text / judgment** | what the model *does* | generated text; or a rating task it has to reason through |

The **three-construct dissociation** (W3) is the finding that these don't agree —
and the read/write work (§5) is the sharpest case of the disagreement, localized
to a single matrix cell.

Two methods lessons worth carrying:

- **PC1 is a norm artifact — but only sometimes, and the real lesson is about the
  signal, not the artifact.** In pre-norm transformers the top PC of *raw* hidden
  states correlates r ≈ 1.0 with activation norm and carries zero trait info — never
  raw PCA. Anisotropy (all inputs cosine ≈ +1 in absolute space) is the universal
  obstacle, but how you remove it depends on whether the trait signal is *defined by
  a contrast*. With **paired** data the signal lives in `fwd − rev` by construction,
  structurally off the dominant (norm) axis; the contrast also largely subtracts the
  *shared* anisotropy offset (not all of it — fwd/rev norms differ, leaving a
  second-order residual), and projecting out item-PC1 (≈ that axis, from the raw
  cloud) cleans the residual. This is *always* safe — item-PC1 is content-free and
  the signal is the contrast, so removal can't touch trait variance — but it's
  marginal (+0.007 r); the contrast did the real work. With **unpaired** single
  words there's no contrast, so the trait signal just *is* an axis of the centered
  cloud — possibly the dominant one — and you can't blindly strip PC1 without risking
  the signal itself. So you keep it, *unless* it's an identifiable artifact (Gemma's
  massive activations, ~1e5, surviving centering as a concentrated spike). That
  regime-test — remove PC1 only when its inverse participation ratio flags a spike —
  is what makes the adjective pipeline *adaptive* rather than fixed. The deep point:
  paired data separates signal from the dominant axis for free; unpaired data
  doesn't, so removal has to be conditional.
- **Format-invariant measurement.** Reading at the period token gives r = 1.000
  stability across response formats (a causal-attention guarantee). Useful when
  you need the measurement not to depend on prompt scaffolding.

## 4. The human-alignment anchor — and the scoop

The empirical headline that survived everything: across the cohort (Gemma /
Llama / Phi4 / Qwen, 3B–12B), each model's 30×30 IPIP **facet cosine matrix
matches the human facet correlation matrix** (Johnson N≈145k) at cohort
r ≈ **+0.56**. Models pick up the empirical Big Five covariance from training
text, consistently across 4 architectures and a 4× scale range.

Then the **vocabulary-coupling** decomposition (W8): the apparent "Likert
recovers personas better than representation does" gap (+0.144 on one Qwen)
mostly evaporates (~+0.05 cohort) once persona description, rating target, and
extraction vocabulary are all matched. The methods aren't measuring different
things; the gap was lexical coupling. **Your TIRT readout slots in exactly here**
as the third, structurally-cleaner line — it removes 2 of the 3 couplings by
construction (W10–W12).

**The scoop (W13).** Wulff & Mata (2025, *Nat. Hum. Behav.*) and Milano et al.
(2025) published that the human factor structure is recoverable from item *text*
alone via sentence embeddings. That partially took our headline. Our response
reframed the contribution and is the seed of everything below: build the embedding
baseline ourselves, and ask **where the model deviates from it**, because the
deviation — not the match — is the part that is about *this kind of model*.

The embedding baseline result: the model's facet geometry does **not** beat an
encoder's. The structure lives in the item semantics both consume, not in anything
autoregressive-or-alignment-specific. The encoder is "just another model of the
same items," and the LLM sits closer to the encoder than to humans. So the
*geometry* is inherited; we went looking for what isn't.

## 5. The read/write gap, made behavioral (W14–W15) — the capstone

**Is the adjective Big Five even real (W14)?** Three over-extraction diagnostics
(rotation stability, bass-ackwards trees, respondent bootstrap) + a Kaiser/SPSS
varimax fix: there is **no stable 6th factor**, and the model collapses to a
**2-factor evaluative core** — two *near-orthogonal valence poles*, not an
intensity factor. Reconciled with the r ≈ 0.56 match in W14 §2: it's a **metric**
difference. *Relationally* (matrix-correlation of pairwise similarities) the Big
Five is present and stimulus-invariant; *dimensionally* (factor congruence) it is
weak-to-absent, because factor extraction is variance-weighted and the model's
variance concentrates on evaluation. "Thin Big Five overlay" = low-variance-but-
present, not missing.

**The behavioral bridge (W15).** The geometry above is a fact about a
*representation*. Does the model *act* on it? Take 26 pole-spanning adjectives
(Wonderful/Amazing/… vs Awful/Terrible/…, plus warmth, antagonism, distress,
intellect, neutral). Compare three corners on the same words:

- **Representation** (residual-stream cosine): puts pos-eval × neg-eval **above**
  its own mean — Wonderful ≈ Awful, *merged*.
- **Judgment** (Likert-logprob similarity rating, valence-neutral anchor): puts
  the same pair **below** the mean — *split*, at or past the human antonym value.

Every model 3B→32B (Qwen 3/7/32B, Gemma 4/12B) shows this **sign-flip, not
attenuation**. It is **not size-gated** (complete at 3B); family matters more
than size (both Gemmas *overshoot* the human split). The divergence is **localized**
— representation and judgment agree everywhere except the evaluative pole-merge,
i.e. the representation's one signature error is exactly the cell that doesn't
reach behavior.

Mechanism probes (W15 §2–§3):

- It's not coherent-rep-vs-noisy-judgment: the judgment matrix is **near-PSD**
  (two self-consistent geometries disagreeing on one block).
- It's not a semantics-vs-disposition confound: re-run as a Theory-of-Mind task
  ("consider a person who is very *Wonderful*; how accurately does *Awful*
  describe them?") and judgment **still splits**. ToM also *overshoots* the human
  split with an **evaluative halo that scales with model size** — the assistant
  shape (§2) reappearing at the lexical level.
- **Weights vs context, base vs instruct:** the representational merge is a
  **pretrained constant** — flat across all stages of both Qwen (Base→Instruct)
  and OLMo-2 (Base→SFT→DPO→RLVR), bare-extracted. The behavioral split is
  **family-dependent**: Qwen's base already voices it and tuning just collapses
  entropy (confidence); OLMo builds it cumulatively across post-training while
  staying high-entropy. Single-model conclusions got revised 4× by the OLMo
  ladder — so the only claim that survives across families is the
  representation-level one. *(rgb's standing caution — "single-model conclusions
  often get revised" — earned its keep here.)*

Interpretation, in one line: **the merge is associative** (distributional twins,
same emphatic contexts, living in the residual stream); the model **also holds a
symbolic valence distinction** and deploys it the moment it has to judge. So the
external 2/3-depth geometry is a *worse witness to the model's evaluative
competence than its own judgment*. (This is the behavioral teeth on rgb's
long-standing "representation isn't intention" stance.)

## 6. NEW (W16): how far back does the merge go?

Reading-group question this week: the merge is in current transformer encoders
too — how much further back does it stretch? Down to LSTMs? HMMs?

We walked the regress with the **same merge statistic on the same 26 adjectives**,
newest → oldest: LLM-judgment, LLM-representation, transformer encoders
(bge-large, mpnet), and **static word vectors** (GloVe 6B/840B, 2014; komninos),
against the human antonym reference.

**Result** (`scripts/adjective_regress.py`, fig `results/adjectives/regress/regress.png`;
antonym-z = pos-eval × neg-eval block similarity, z-scored within each matrix —
above 0 = merged, below 0 = split):

| stratum | antonym-z | verdict |
|---|---:|---|
| LLM-repr (Qwen-7B residual stream) | **+1.58** | merged (most of all) |
| komninos (dependency word2vec, 2016) | +1.18 | merged |
| glove.6B (static, 2014) | +0.71 | merged |
| mpnet-base (encoder, 2021) | +0.51 | merged |
| glove.840B (static, 2014) | +0.24 | merged |
| bge-large (encoder, 2023) | +0.16 | merged |
| **LLM-judge (Qwen-7B rating)** | **−0.49** | **split** |
| **human (525-PDA self-report)** | **−0.53** | **split** |

The merge is **complete in 2014 static GloVe vectors** and persists unbroken
through encoders and the LLM's resting representation. Only the LLM's **judgment**
and human self-report push the antonyms apart — and they land on top of each other
(−0.49 vs −0.53). The whole 26-word geometry agrees everywhere (synonyms group,
different dimensions separate) *except* this one antonym cell — exactly the W15
read/write cell. And merge magnitude is **not** era-monotonic: the LLM representation
merges *hardest*, a 2016 word2vec merges harder than a 2023 encoder. The
distributional objective predicts the *sign*; modernity predicts nothing. (Full
table + the LSTM/HMM/CRF reasoning in `report_week16.md`.)

The point this makes: **the good/bad merge is a property of the distributional
hypothesis, not of transformers, depth, or scale.** Antonyms-are-distributional-
neighbors is the oldest known failure mode of vector-space semantics (documented
continuously from LSA/HAL in the 1990s through word2vec/GloVe; the counter-fitting
literature exists precisely to undo it). A class-based generative model (an HMM /
Brown-cluster LM) would merge them *harder*, into the same induced slot; a CRF is
a category error here (supervised sequence labeler, induces no word geometry on
its own). So the regress bottoms out not at an architecture but at **"learn meaning
from co-occurrence."**

And that sharpens the whole project: a GloVe vector is **all read, no write** — it
*cannot* un-merge the antonyms because it has no behavioral channel. The LLM can.
**The override is the new thing**, and it is the thing our read/write work measures.

## 7. The unifying mechanism — symbolic vs associative

One frame ties §3–§6 together (rgb's, since W7):

- **Associative** route: the aggregated residual stream. Co-occurrence statistics,
  distributional twins, the evaluative merge. Inherited, pretrained, ancient.
- **Symbolic** route: per-item judgment. Negation, antonymy, the discrete
  good ≠ bad call. Deployed in the Likert / ToM / write tasks; overrides the
  associative merge.

Likert recovers more than projection because it is a **symbolic per-adjective
judgment**; representation aggregates the **associative residual**. The read/write
gap is the symbolic route overriding the associative one, and W16 says the
associative substrate is as old as distributional semantics while the symbolic
override is what tuning shapes.

## 8. Where your track meets this, and what's open

- **Your GFC/TIRT is the third readout.** It is already wired in (W10–W12) as the
  structurally-cleanest persona-recovery line, and it is the natural "preference"
  corner that is immune to the vocabulary coupling that contaminates Likert. The
  read/write framing gives it a sharper role: TIRT is a *write*-adjacent measure
  (forced behavior under desirability matching), so it should pattern with
  judgment, not with representation — a prediction worth checking on the adjective
  antonyms directly.
- **Trait-conflict dilemma instrument** — open since W3; the ceiling-effect
  breaker; Thurstonian-scorable scenario forced-choice. Still the most paper-shaped
  open item.
- **Read/write reconnection** — does a representation-induced persona persist
  through extended generation? Connects your persona work to the W4 steering
  results and the W15 override.
- **Regress, write side:** does an HMM/Brown-cluster LM merge antonyms (it should,
  harder)? A small from-scratch PPMI-SVD demo would close the bottom of §6
  conclusively. Cheap; not yet built.

## Key outputs to look at (in order)

| # | Artifact | What it shows |
|---|---|---|
| 1 | `results/facets/reconcile_facet_adjective.png` | the r≈0.56 match is relational, not dimensional (W14 §2) |
| 2 | `results/adjectives/introspection_vs_representation.png` | the read/write sign-flip — represents-merged, judges-split (W15 §1) |
| 3 | `results/adjectives/training_stage_*.png` | merge is pretrained-constant; behavioral split is family-dependent (W15 §3) |
| 4 | `results/adjectives/regress/regress.png` | the merge regresses to 2014 static vectors — distributional, not architectural (W16) |
| 5 | `results/facets/ipip_facet_vs_human_dashboard.html` | the original human-alignment headline (W9 §7) |

Reports, if you want the long form: `report_week14.md` and `report_week15.md` are
the freshest and most self-contained; `report_week7.md` is where the
symbolic-vs-associative frame is born; `paper_outline.md` is the (now partial)
paper plan to update together.
