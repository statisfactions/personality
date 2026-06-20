# Methods note: generation-based persona-vector extraction (W17 track)

Reference recipe for the write-side extraction program: what Lu et al. and Chen
et al. actually did, what we adapt, and why. Implementation:
`scripts/extract_persona_vectors.py`.

## Sources

- **Lu, Gallagher, Michala, Fish & Lindsey (2026), "The Assistant Axis:
  Situating and Stabilizing the Default Persona of Language Models."**
  arXiv:2601.10387. Vs-mean extraction; the assistant axis.
- **Chen, Arditi, Sleight, Evans & Lindsey (2025), "Persona Vectors:
  Monitoring and Controlling Character Traits in Language Models."**
  arXiv:2507.21509. Pos-vs-neg extraction. Code: github.com/safety-research/persona_vectors.

## Lu et al.'s exact procedure (from the paper)

- **Personas:** 275 roles (iteratively developed with Claude Sonnet 4),
  **5 system prompts per role**, **240 extraction questions** "designed to
  differentiate responses based on expressed characteristics" → up to 1,200
  rollouts per role (all prompt × question combinations).
- **Quality filter:** LLM judge classifies each response as fully / somewhat /
  not role-playing; a role is retained only with ≥10 acceptable samples.
- **Activations:** "the mean **post-MLP residual stream** activations at **all
  response tokens**" (prompt tokens excluded). In HF terms this is
  `output_hidden_states` (the post-block residual).
- **Layer:** middle layer unless otherwise specified.
- **Role vector:** mean activation per role; population standardized by
  "subtracting the mean vector across roles."
- **Assistant axis:** "We subtracted the mean of all fully role-playing role
  vectors from the mean default Assistant activation":
  `a = mean(default assistant) − mean(role vectors)`.
  Validation: cos(a, PC1 of role vectors) > 0.71 at the middle layer of every
  model.
- **Steering scale:** vectors scaled "with respect to the average post-MLP
  residual stream norm (measured on lmsys-chat-1m) at that layer." No other
  anisotropy handling — the contrast (subtracting the role-population mean)
  is what removes the shared high-norm component.
- **Models:** Gemma-2 27B, Qwen-3 32B (reasoning off), Llama-3.3 70B. Note
  **no Gemma-3** — our cohort is the stress test for the massive-activation
  families.

## Chen et al.'s variant (persona vectors)

Positive vs negative system prompts per trait ("You are an evil assistant"
style, ~5 generated pairs), evaluation questions, judge-scored trait
expression as the inclusion filter, and the direction = mean difference of
**response-token-averaged** activations between pos and neg conditions
(`response_avg_diff` is what the paper uses; prompt-avg and prompt-last
variants exist in the repo but were not preferred). Layer ~20 on
Qwen2.5-7B-Instruct (≈ middle). Steering validated at coef ≈ 2 (inference)
and 5 (training-time).

**Why vs-mean is our primary and pos-neg the comparison:** pos-neg
differencing *defines* the direction as (pos − neg) and therefore cannot
detect whether the two poles are antipodal in the model's geometry. The
W15/W16 merge result says evaluative antonyms are *not* antipodal on the read
side — whether they are on the write side is precisely our question. Vs-mean
extraction leaves the angle between d_wonderful and d_awful free to be
measured. The Chen-style vector is recoverable as d_pos − d_neg afterward, so
nothing is lost.

## Our adaptation (525-PDA write-side extraction)

| Component | Lu et al. | Ours |
|---|---|---|
| Personas | 275 hand-built roles | 525-PDA adjectives: "You are someone who is {adj}…" |
| Sys variants / role | 5 | 5 templates (smoke: 2), `SYS_TEMPLATES` |
| Questions | 240 | 12-question generic bank (smoke: 6); same bank for every adjective so the persona prompt is the only varying factor |
| Rollouts | ≤1,200/role | sampled (T=0.8, top-p 0.95), budget-scaled |
| Filter | LLM judge (roleplay quality) | lexical-leakage flag (cheap first pass; judge upgrade if needed) |
| Tokens | all response tokens | same (`--skip-first` available for sink-token insurance) |
| Layer | middle | all layers saved; middle for headline diagnostics |
| Direction | role mean − mean of role means | same, fp64 |
| Assistant axis | default mean − role-mean centroid | same |
| Steering scale | avg residual norm (lmsys) | TBD at steering stage |

### Precision protocol

- bf16 forward (fp16 NaNs on Gemma-3), but hidden states cast to **fp32 on
  device before token averaging**. bf16 carries ~3 significant digits; the
  trait component is ~0.15% of activation norm (W2), i.e. at/below bf16
  rounding for Gemma-scale (1e5) activations — a bf16-accumulated mean
  destroys it. The same fix is applied to the shared
  `extract_meandiff_vectors.hidden_states_for_text` (all downstream
  re-extractions inherit it).
- MPS has no fp64, so cross-rollout means, grand mean, directions, and all
  diagnostics accumulate in **fp64 on CPU** (numpy). Per-rollout activations
  stored fp32.

### Design decisions specific to our questions

1. **Eval-antonym poles extracted separately** (wonderful-person and
   awful-person each vs the role centroid). The angle between them is the
   write-side merge test: ≈ +1 → the merge propagates to the control/persona
   layer (the model can't *enact* a distinction it can judge); ≈ −1 → the
   write pathway is where the symbolic split lives.
2. **Physical adjectives as a placebo arm**, not just a filter. A slim person
   does not write differently; the word "slim" has rich associates anyway. If
   placebo directions are as strong/stable as trait directions, the method is
   reading lexical semantics out of the prompt, not enactment — the write-side
   interpretation would collapse. Keep ~a dozen in every run; exclude from
   factor analyses.
3. **Questions must not invite self-description.** Roleplay narration ("As
   someone who keeps fit…") lets unenactable adjectives fake a direction. The
   bank is everyday advice/action prompts; the leakage flag
   (`\b{adj}\w{0,4}\b` in the response) catches explicit narration; if leak
   rates are high, escalate to judge filtering (Chen-style).
   **Caveat (scaled-30 run):** the regex mostly catches idioms, not
   narration — "honest" 47% is "to be honest", "tall" 43% is "tall order".
   Tighten or judge-score before using it as an exclusion filter.
4. **Bootstrap coherence instead of one split-half.** Resample each
   adjective's rollouts B=200 times (grand mean fixed), cosine of each
   bootstrap direction to the full-data direction. Gives a per-adjective
   stability distribution comparable across the cohort, and is the
   **enactability score**: physical placebos should sit at the floor.
   Prediction: evaluative core highly coherent, physical items unstable,
   informative middle.
5. **Standing Gemma checks** (from W2/W5): (a) corr(per-rollout projection on
   d, per-rollout activation norm) ≈ 0; (b) top-10 dims' share of ||d||² small
   (massive-activation channels behave like biases and should cancel in the
   contrast — verify, don't assume); (c) `--skip-first` sensitivity for
   position-norm drift.

   **Outcome (Gemma-3-4B smoke, 2026-06-11): the contrast does NOT cancel
   Gemma's massive channels.** Dim 443 carries mean |act| ≈ 30,000 (~2,500×
   the median dim) and the massive channels are condition-dependent:
   |norm_r| hit 1.00 and single directions put 81% of ||d||² in 10 dims.
   Zeroing the top ~16 dims restores the trait geometry (wonderful|awful
   −0.55, talkative|quiet −0.44) and Llama-like norm_r. The script now
   auto-flags dims ≥20× median mean|act| at the mid layer and reports all
   pairwise/axis diagnostics raw *and* ablated. Notably, Gemma's
   default-vs-roleplay (assistant axis) contrast lives almost entirely IN
   the massive channels — after ablation the axis flattens (cos ≈ 0.3 max).
   Possible mechanism finding (massive channels as persona/mode signal),
   connects to the chat-template-gating thread (to_try §11); treat Gemma
   axis claims with care. Follow-up decomposition: dim 443 is primarily a
   default-vs-persona mode flag (default ~+2000 over every persona), the
   massive dims also track response length within condition (r ≈ ±0.3), but
   the massive subspace is NOT valence-dead (wonderful|awful −0.81 within
   those 16 dims alone).

6. **Caricature, and why we didn't soften the prompts (Llama-3B scaled-30
   analysis, 2026-06-12).** The model enacts traits as screenplay tropes
   (handsome: 33 smirks + 15 winks / 60 rollouts; dishonest: "whistles
   innocently"; quiet: 88% *pauses, trails off*) — rgb's worry was that the
   extreme prompt wording ("deeply X", "to your core") extracts trite
   behavior. Tested via the recorded per-rollout sys template (T1/T3 mild
   vs T2/T4/T5 extreme), the extremity wording turns out not to be the
   lever: (a) extreme/mild displacement ratio 1.02 — the bare adjective
   already saturates the persona shift; (b) theater rate 43% mild vs 47%
   extreme — the trite register comes from roleplay framing per se (and/or
   3B capacity), not intensity words; (c) genuine direction rotation is
   small: template-split cosine 0.841 vs random-split noise floor 0.910
   (~0.07 real). So softer wording buys no nuance here, and the cohort run
   proceeds unchanged. The construct caveat stands: these are *performed
   trait schema* directions (stereotype register, valence-saturated), not
   graded dispositions — W6's contrast-vs-disposition lesson on the write
   side. Per-model re-checks are free (sys is recorded in every texts.json);
   the scale question — do bigger models shed the trope register? — rides
   along with the cohort. A naturalistic-induction comparison (trait
   embedded in biographical preambles, W7 §11.5.9-style) is the designed
   follow-up if the performance/disposition axes need separating.

7. **Geometry can establish presence, never absence — absence claims must
   ground in judged text.** Cautionary tale (2026-06-11): post-ablation
   Gemma geometry showed no attractiveness halo on `handsome` (valence-
   aligned cos ≈ 0.0 vs Llama's +0.18), suggesting "Gemma doesn't write the
   halo." Text-grounded check (`scripts/judge_rollout_tone.py`, tone 1–7 by
   logprob EV, wonderful/awful as range anchors) showed the OPPOSITE:
   Gemma's handsome rollouts are +0.92 above slim (vs Llama's +0.30) and
   both models put handsome ~+0.5 above their persona-mean tone. The
   cleaned mid-layer cosine was a false negative — the behavioral halo is
   present in both models, and on Gemma it isn't visible in post-ablation
   condition-mean directions (in the zeroed channels, at other depths, or
   distributed). Standing protocol: every "model lacks X" claim gets a
   judge-scored text check before it's reported.
6. **Comparability across channels:** the write-side factor structure must be
   computed on the post-filter adjective subset, and the W14 read-side, W16
   judgment-side, and human (raw 525-PDA data) structures recomputed on the
   same subset before comparing columns.

## The four-grid comparison (analysis spec, agreed 2026-06-12)

Four 523² similarity structures over the same adjective set, one per channel
(terminology going forward — retiring "read/write", which collided once
enactment became a third model channel):

| grid | channel | cell (X,Y) |
|---|---|---|
| HUMAN | ground truth | human self-report co-occurrence (raw 525-PDA correlations) |
| REPRESENT | representation | residual-stream cosine (W14 adjective geometry) |
| JUDGE | judgment | tom_likely — the model's explicit guess at HUMAN (W16 §5 full matrix, symmetrized) |
| ENACT | enactment | persona-vector direction cosine (W17, mid layer) |

**Design decisions:**
- **Raw matrices, no ipsatization.** The construct is "the structure each
  channel actually produces" — HUMAN included. Ipsatizing the human side
  only would (a) make the pipeline asymmetric, (b) mechanically induce
  negative correlations (row-centering artifact) and delete the general
  factor while silently taking a side in the GFP style-vs-substance debate,
  (c) destroy the caricature yardstick (ENACT's evaluative saturation is
  measured *relative to* the human halo). Response biases are
  multi-directional (acquiescence inflates uniformly, SDR inflates the eval
  block, careless responding attenuates) — no single correction fixes them,
  so we don't pretend one does. C&C ipsatized human available as a
  literature footnote only.
- **Spearman primary, Pearson secondary**, over the common upper triangle.
  Rank correlation is invariant to the channels' different monotone scaling
  histories (tom_likely EV→symmetrize→percentile, etc.).
- **General-factor sensitivity = symmetric robustness row**: remove each
  grid's own first principal component from itself (same operation, all
  four), re-correlate residuals. Answers "is the match just the shared
  evaluative axis" without ipsative artifacts.
- **Eval-block vs off-block breakdown** as the third view — the
  eval-antonym block is where REPRESENT diverges and ENACT's saturation
  should show.
- **ENACT cleaning policy as robustness, not choice**: raw and
  massive-dim-ablated cosines both reported; leak-flagged adjectives
  in/out likewise.

**Registered prediction (before computing):** ENACT correlates highest with
JUDGE (enactment ≈ applied theory-of-mind); ENACT↔HUMAN below JUDGE↔HUMAN;
the ENACT−JUDGE residual is the performance register (eval block amplified,
trope cells inflated). Falsifier: ENACT tracking REPRESENT off the
hand-picked antonym cells would overturn symbolic-overrides-associative
where it matters most.

**First results (Llama-3.2, 2026-06-12, `four_grid_compare.py`):** registered
prediction falsified in the interesting direction — ENACT is the BEST
human-matcher (Spearman 0.747 vs JUDGE 0.604, REPRESENT 0.437), monotone in
how behavioral the channel is; survives symmetric PC1-removal (0.435, still
highest), double-centering (0.758), and all ENACT cleaning variants (±0.005).
Caricature signature where predicted: ENACT antonym block at 6th percentile
vs human 31st — performance over-separates the eval poles. Caveat: JUDGE is
weak on Llama specifically (near-uniform logprobs; antonym block 54th);
ENACT-beats-JUDGE needs Qwen7/Gemma12 confirmation. REPRESENT|JUDGE 0.304
matches W16's ~0.35 (implementation cross-check).

**Vector-level REPRESENT→ENACT transformation
(`represent_enact_geometry.py`, layers 14 & 19):** not a rotation. (a)
In-place overlap minimal: direct per-adjective cosine ≈ 0.18, per-adjective
norms uncorrelated (r ≈ 0.05); top-k subspaces nearly disjoint (principal
angles median ~75°, min ~50°) — the W2-4 read/write subspace orthogonality
at population scale. (b) After the BEST rotation the shapes do match
substantially (Procrustes corr 0.70–0.73 — same relational geometry,
different room). (c) Significant shear on top: within matched top-k PC
coords, general linear beats orthogonal by ΔR² ≈ +0.20–0.31. (d) The shear
is spectrally interpretable: ENACT concentrates ~46% of variance in its top
3 axes vs REPRESENT's ~21% — the write channel amplifies a low-dimensional
(evaluative) core and attenuates the tail. Caricature as anisotropic gain,
the linear-algebra form of the 6th-vs-31st antonym exaggeration.

**Density asymmetry / Anna Karenina (2026-06-12).** Raw similarity: positive
adjectives are denser than negative in HUMAN/JUDGE/ENACT (Unkelbach's
density hypothesis) — but double-centering flips the sign in ALL four grids
(HUMAN −0.49 SD, JUDGE −0.25, ENACT −0.17, REPRESENT −0.15): the positive
density is purely a margin/halo effect with no residual structure, while
negative attributes carry genuine residual covariation. Effective dims
agree (ENACT pos 8.0 vs neg 9.8; REPRESENT 25.5 vs 32.4). Precise Anna
Karenina: happy attributes alike in one undifferentiated way; unhappy ones
different in their own ways — and humans differentiate badness MORE than
the model does in residual terms.

**Negative syndromes (`viz_neg_syndromes.py`).** k-means (k=5) on the
bottom-30%-valence adjectives in Llama's ENACT space recovers, unsupervised:
menace/madness (insane/frightening/evil), fear (scared/anxious/nervous),
hostility (hostile/rude/mean), incompetence (stupid/careless/unreliable),
distress (sad/lonely/depressed) — approximately HiTOP-adjacent structure,
including the fear-vs-distress internalizing split. Human-grid clustering of
the same words carves similar regions (ARI 0.21; rosters more convincing
than the index). Caveats: silhouette ~0.18 (overlapping regions), one model.
Figures: `figs/spectrum_*`, `figs/clouds_*`, `figs/neg_syndromes_*`.

**Interpretive thesis (rgb): ENACT samples the model's *character* prior,
not its *person* prior — "enact tells a better story; humans don't fully
cooperate with the narrative."** Unifies the W17 results: dimension
compression (characters < people), antonym over-separation (heroes vs
villains), stock-syndrome clusters, trope register; ENACT↔HUMAN 0.75 =
fiction is a good-but-sharpened model of human structure. Matches Lu et
al.'s "amalgamation of character archetypes from pretraining." Two designed
tests:
1. **FICTION as a fifth grid** — adjective co-occurrence over fictional
   character descriptions (e.g., CMU Movie Summary personas); predict
   ENACT ≈ FICTION > HUMAN on compression/antonym stats, FICTION between
   ENACT and HUMAN on grid correlations.
2. **Documentary framing** — re-run scaled-30 with person-prior induction
   ("ordinary real person, not a performance"); watch compression ratio,
   antonym percentile, trope rate move toward human values. If they do,
   caricature is prior-selection, not 3B capability.

**Qwen-2.5 replication (2026-06-13) — revises the Llama headline.** Qwen has
a SHARP JUDGE channel (antonym block 12th pct, vs Llama's mushy 54th), so it
tests whether ENACT-beats-JUDGE was a JUDGE-quality artifact. It was.
  - HUMAN match, raw Spearman: JUDGE 0.638 ≈ ENACT 0.617 > REPRESENT 0.512.
  - HUMAN match, double-centered: JUDGE 0.746 > ENACT 0.626 > REPRESENT 0.526.
  So the registered prediction (ENACT≈JUDGE) is ~right on a good-JUDGE model;
  Llama's "enacted > declarative knowledge" does NOT survive. Caution stands
  that double-centering is the construct-relevant view (interaction structure,
  prevalence stripped).
  - Double-centering BOOSTS JUDGE hugely on both models (Llama 0.604→0.702,
    Qwen 0.638→0.746) but barely moves ENACT (0.617→0.626): JUDGE carries
    strong per-adjective prevalence marginals; ENACT's match is already in the
    raw pairwise geometry. Different structure, not just different strength.
  - Leak-filtered ENACT rises (0.617→0.662) but on 31% of cells (both
    adjectives non-leaky) — confounded subset, not a clean "denoised" number.

**Compression = assistant-axis collapse (the robust, strengthened headline).**
The REPRESENT→ENACT dimension compression replicates and is far STRONGER on
Qwen: effective dim (participation ratio) REPRESENT 44.9 → ENACT **2.9**
(15× vs Llama's 4.2×; Llama was 43→10). Survives massive-dim ablation
(2.9→3.6, PC1 0.58→0.51 — not a Qwen-massive-dim artifact). Critically:
**cos(ENACT PC1, assistant_axis) = 0.811** — enactment projects the adjective
space mostly onto the Lu et al. assistant axis. This unifies the track with
its origin: the caricature/compression IS the assistant-shape rank-1 collapse
(W1) seen on the write side. Enacting a persona ≈ sliding along the assistant
axis plus a 2-3-dim residual. Vector-level on Qwen: shear gap larger
(+0.36–0.50), Procrustes 0.57, principal angles 49/68° — same NOT-a-rotation
story as Llama, more extreme.

**Caricature (antonym over-separation) is robust:** ENACT antonym block at
6th pct on BOTH models vs human 31st. Qwen JUDGE also over-splits (12th).

**Effective dim across ALL four grids (PR of similarity-matrix eigenvalues,
PSD Grams raw / JUDGE double-centered as it's not a true Gram).** The
collapse is SPECIFIC to enactment, not a general behavioral-channel property:
  | grid | Llama | Qwen |
  |---|---|---|
  | REPRESENT | 71 | 42 |
  | HUMAN | 27 | 27 |
  | JUDGE | 56 | 23 (neg mass .29) |
  | ENACT | 9.8 | 3.9 |
  - HUMAN is high-dim (27, both models), in the REPRESENT neighborhood, NOT
    the ENACT one — but BELOW REPRESENT (model representation over-resolves
    vs real human structure; some is anisotropy/noise).
  - JUDGE is high-dim too (spectral neighbor of HUMAN), NOT collapsed like
    ENACT. ENACT is the lone outlier. So judgment preserves human-like
    dimensionality; only enactment crushes it onto the assistant axis. This
    is WHY JUDGE matches HUMAN best in the four-grid (same shape) while ENACT
    over-separates within a collapsed shape. The two channels fail oppositely.
  - Caveats: JUDGE not PSD (neg eigenvalue mass .10 Llama / .29 Qwen, softer
    number); Llama JUDGE 56 partly noise-spread (near-uniform logprobs),
    Qwen's 23 more trustworthy. Qualitative verdict (JUDGE 2-6x ENACT)
    robust. Similarity-matrix PR ≈ vector-SVD PR (REPRESENT ~42-45 Qwen,
    ENACT ~3-7) — pipelines agree.

**Induction register gradient (llama3.2 smoke, 8 adj, 2026-06-14).** Tests
whether the trite/theatrical register is driven by the *performance framing*
vs trait induction per se. Three registers (`--induction`): performance
("roleplay as / stay in character"), subtle ("a bit X, respond naturally,
not a performance"), plain ("You are X"). Result:
  | register | theater | ‖dir‖ | drift-from-default | cos(W,A) |
  |---|---|---|---|---|
  | performance | 0.32 | 2.03 | 3.35 | −0.50 |
  | subtle | 0.03 | 0.87 | 2.44 | −0.59 |
  | plain | 0.07 | 0.79 | 1.09 | −0.22 |
  - **Theater is framing-driven, not magnitude-driven**: removing the
    roleplay words kills stage directions (0.32→0.03) while magnitude only
    drops 57% — clean dissociation. (Theater metric: SINGLE-asterisk regex;
    `**markdown bold**` from the default assistant's lists is a false
    positive — bit me once.)
  - **3B DID follow "a bit X"**: monotone drift performance>subtle>plain;
    subtle still induces 73% of performance's drift. Allays the
    small-model-can't-modulate worry.
  - **plain ≈ default assistant**: bare attribution is largely ignored
    (drift 1.09, markdown lists, no trait) — needs *some* behavioral framing.
  - **Subtle is rotated, not just attenuated** (cos 0.54 vs performance, mag
    0.44). The performance "theater" component is a sign-flipping shared
    subspace, not a single vector: mean(perf−subtle)≈0 (cancels) but SVD of
    the differences gives PC1 evr 0.41 vs null 0.14 — real shared structure.
    However it's NOT more concentrated than the signal (signal PC1 0.36–0.52),
    and its PC1 is orthogonal to the assistant axis (cos 0.00). So you can't
    peel off a low-rank theater axis — re-extract under subtle for clean
    assistant-drift. (Methodological note: the mean-projection test was
    misleading — sign-flipping structure cancels in the mean; use SVD vs null.
    compare_induction.py fixed accordingly.) CAVEAT: 8 noisy adj, needs
    scaled-30. Bigger-model (Llama8) confirmation in progress.
  Implication for the assistant-drift goal: subtle framing is the working
  register — substantial induction, no performance contamination.

**Induction-frame comparison for the assistant-DRIFT instrument (2026-06-14).**
The drift instrument ("how far does the deployed assistant move toward X") is a
SEPARATE deliverable from the persona-vector/four-grid object (which is the
Lu/Chen *performance* register and is what the cohort extracts). It needs ONE
committed frame. Registers (`--induction`): performance, subtle, plain,
assistant (HHH), subtleA (human-anchor "real individual/person/human"),
subtleB (no-anchor). Findings:

- **Theater = persona-performance framing, graded by persistence cues, not by
  intensity.** Per-template (scaled-30, corrected single-* regex): "stay in
  character no matter what" 0.50 > "to your core" 0.33; mild "respond as this
  person" 0.26. Moderate-roleplay 0.34 vs extreme 0.42 — de-amplifying halves
  theater but even mildest roleplay is ~9× the naturalistic registers (0.03).
  You only ELIMINATE theater by dropping persona-performance framing.
- **Scale (llama3.2-3B vs Llama-3.1-8B):** theater-drop and drift gradient
  (performance>subtle>plain) replicate. The 3B can't use the TERSE plain frame
  (treats "You are X" as ≈default assistant, mushy antonyms −0.22); the 8B
  induces sharply from it (−0.60). Both handle the elaborate subtle frame fine.
  So small models need explicit behavioral scaffolding, not less of it.
- **Three-frame head-to-head (llama3.2, anti-assistant/pro-HHH/normalcy/eval
  traits):**
  | frame | anti-HHH traits | theater | cos→asst (on-axis) |
  |---|---|---|---|
  | no-anchor | **CHOKES** (dishonest→"I'm an LLM") | 0.08 | mid |
  | human "not an assistant" | expresses well | **0.30** (theater returns!) | lowest (off-axis) |
  | HHH-assistant | expresses well, no choke | 0.16 | **highest** |
  - no-anchor reverts to honest-LLM on cruel/dishonest (anti-harmless/honest;
    lazy is fine — not a moral violation). Confirmed in text.
  - **Corrected prediction:** the HHH frame does NOT choke on anti-HHH traits —
    "HHH assistant who's a bit dishonest" induces dishonest/cruel/rude fine
    (rude strongest, 1.80). The HHH preamble is not a guard against a trait
    nudge — mild alignment note in itself.
  - The "not an assistant" cue specifically re-triggers theater (0.30) +
    pulls off-axis; the gentler "normal person" wording didn't (0.03).
  - HHH pro-HHH drift ≈ 0 (0.98) — correct: the assistant already IS
    kind/honest/careful. Validity check: zero drift where there should be none.
  - **Verdict: HHH-assistant is best for drift** (on-axis, no choke, low
    theater, correct null-drift). Human-anchor reserved for un-assistant-
    constrained trait expression if ever needed (hybrid available). NOT yet
    committed — user deciding 2026-06-15.
- **Frame-word collision (subtle register bug):** "normal person…ordinary…
  nothing unusual…naturally" contains 4 words that ARE 525-PDA traits
  (normal/ordinary/natural/unusual) + adjacent (abnormal/strange/weird/
  extraordinary). subtleA/subtleB/HHH avoid it ("genuine" is a trait — kept
  out of subtleA). Smoke sets had no normalcy traits, so prior results clean.

## Steering capstone — pre-registered trait clusters

For the causal capstone (does ENACT steer behavior, REPRESENT not), steer at
the SYNONYM-CLUSTER grain — finer than Big5 (Big5 domains collapse to valence
in these models, so a domain-level steering vector mostly steers good/bad),
coarser than single adjectives (per-adjective ENACT residual is small + noisy
after centroid removal; pooling a synonym neighborhood reinforces the shared
trait-residual). Clusters are PRE-REGISTERED to avoid ad-hoc selection.

Procedure (`scripts/build_trait_clusters.py` → `instruments/trait_clusters.json`,
deterministic, run once):
1. HUMAN 525-PDA correlation matrix (model-independent → one canonical set).
2. Distance = 1−r (POLE-RESPECTING — antonyms r<0 → dist>1 → separate clusters;
   the eval-antonym merge can't put wonderful with awful, as REPRESENT cosine
   would).
3. **Ward** linkage (average/complete chain — average gave median-3 + 76-blob).
4. Fine cut k=65 (~8/cluster); harvest 5–12-member clusters with within-cluster
   mean r > 0.25 → 35-cluster candidate POOL.
5. Coarse cut k=8 → empirical higher-level BRANCHES (not an imposed taxonomy).
6. Best-coherence cluster per branch + forced evaluation cluster (valence
   control) → LOCKED selection of 8.

Selected 8 (one per branch, coh 0.27–0.50): evaluation* (valence control),
creativity (O), extraversion (E), distress (N), antagonism (low A), rationality,
disorganization (low C), reserve (low E). Full Big5 coverage + valence control.
Quirk: "left-handed" sits in the disorganization cluster (525-PDA spurious
corr) — harmless, flag if it matters. The evaluation-vs-specific-trait contrast
is the key test: can the model steer a SPECIFIC trait, or only valence?

Steering ladder per cluster (escalate only on failure): (A) observational
cluster-mean ENACT direction — no learning, no scarcity (5-9 adj × 60 rollouts
pooled); (B) backprop-δ on the cluster with leave-one-adjective-out validation
(norm-constrained = regularized; LOAO proves trait- not word-learning). Readouts judge-free: BC (channel-matched)
+ Likert (cross-channel; the BC-moves/Likert-doesn't asymmetry IS the read/write
result). REPRESENT directions as the negative control (should not steer BC).

### Gemma two-axis ENACT: denoising robustness + why the massive channels aren't trait signal

**The finding** (PC1-vs-PC2 / general-factor shape, `viz_pc12_shape.py`): every
model's ablated ENACT collapses onto one dominant axis (λ1/λ2 ≈ 2.8–4.1) EXCEPT
Gemma-3, which spreads over two comparable axes (λ1/λ2 ≈ 1.2). REPRESENT (λ1/λ2
≈ 1.2, diffuse) and JUDGE (≈ 2.7, ≈ HUMAN 3.0) shapes are model-invariant; only
ENACT's varies, and Gemma is the outlier.

**Robust to denoising method** (sweep, gemma3 vs qwen2.5 control): Gemma ENACT
λ1/λ2 = 1.2 under zero-top-1-var, zero-top-{4,16,64}, per-dim z-score, AND the
default massive-dim ablation — flat across all. The ONLY value showing single-
axis is **raw (2.8)**, where the artifact is still in. Qwen (no massive
channels) barely moves (4.0 raw → 3.0 aggressive), staying well clear of 1.2.
So Gemma-two-axis is a property, not an ablation artifact. Caveat: absolute
λ1/λ2 are denoising-dependent (z-scoring compresses everyone); the robust claim
is the ordering + Gemma's ≈1.2 plateau, NOT the precise ratios.

**Why we treat raw as the wrong reading — the massive channels aren't trait
signal** (so removing them is correcting, not cherry-picking):
1. Known massive-activation / attention-sink channels (Gemma-family quirk; lit:
   Sun et al.). dim 443 ≈30k, ~2500× median dim.
2. **Near-constant across adjectives**: the 6 massive dims have CV ≈ 0.076
   across the 523 adjectives (mean |act| ≈5300, ~7.5% variation) vs 1.07 for
   typical dims — 14× less relative variation. A dim ~equal for every adjective
   can't discriminate traits; in a cosine matrix it's a constant offset
   (anisotropy), not content.
3. W17 smoke diagnostics: dim 443 = default-vs-persona MODE flag (+~2000 on
   default over every persona) + tracks response length (r≈0.3); with massive
   dims in, geometry is degenerate (norm_r ±1.00, 81% var in 10 dims); ablating
   restores sensible trait geometry.
4. The handsome-halo false negative: raw Gemma geometry gave a WRONG (sign-
   flipped) answer that ablated geometry + text-judge corrected.
5. Convergent denoisers: per-dim z-score (rescales ALL dims, doesn't target the
   massive ones) agrees with zeroing them (both → 1.2) → the channels act via
   magnitude, the signature of a scale artifact, not a content direction.

### Planned re-extraction

All activation data that feeds final analyses gets re-extracted under the
fp32-before-mean fix (the historical caches were bf16-averaged). Expected to
matter most for Gemma-3 family; small models / fp32-dominant paths should
reproduce.
