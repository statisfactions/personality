# Week 18 — Facet-level structure and the SELF follow-ups

**Dates:** 2026-07-08 →
**Calendar:** ~cal-week 15.

Continues from W17 (the read→write map + SELF channel). This week moves to
cluster/facet-level analysis of the 523-set and the SELF-channel loose ends
(human-norm comparison, the Aya/Llama8 self-signal residue, multi-turn drift).

## §1 — The adjective-facet dashboard, and the clusters models don't believe in

To make the 523-set visualizable (rgb: "too big vs the IPIP facets"), the
35 pre-registered human-derived trait clusters (`instruments/
trait_clusters.json`, pole-respecting Ward pool) serve as adjective facets:
`adjective_facet_dashboard.py` renders HUMAN + 10 models × {REPRESENT, JUDGE,
ENACT} as 35×35 block-similarity heatmaps in shared branch order
(`figs/adjective_facet_dashboard.*`). Channel signatures are legible as
texture: JUDGE reproduces HUMAN's macro-blocks; REPRESENT is washed (the
merge as pallor); ENACT is high-contrast and blocky, with the Qwen family's
effdim-5 collapse visible as giant uniform slabs. Note block-level human-match
runs higher than cell-level (REPRESENT 0.72–0.81 vs ~0.5) — block-averaging
over coherent clusters denoises item idiosyncrasy, the facet-vs-item
reliability gap of human psychometrics.

**The weak JUDGE diagonal (rgb's observation): models don't believe in some
clusters — specifically the negative ones.** Within-cluster JUDGE coherence is
consensual across the cohort (per-cluster agreement r = 0.83) and tracks
cluster valence at **+0.72**, while HUMAN within-cluster coherence is
valence-neutral (+0.19); the belief deficit's partial correlation with valence
controlling for human coherence is **−0.74**. Disbelieved: the negative
grab-bags (awkward = clumsy+plain+boring+unattractive+indecisive; sickly =
exhausted+handicapped+poor; disorganized, which contains *left-handed*).
Believed: the tight positives (funny, relaxed, polite, joyful). Reading:
human self-report covariance bundles co-undesirable but semantically
unrelated words (a negative-halo common factor — left-handed is the smoking
gun); the models' more symbolic judgment strips it. Unified with W15's
antonym-separation overshoot: models treat valence as a strong **axis** but
not as a **binder** — they polarize good-vs-bad harder than humans while
refusing to fuse the bad into syndromes. Virtues all alike; every vice
specific.

## §2 — The cohort facet summary, with SELF as a geometry

`adjective_facet_cohort.py` (figs/facet_cohort_summary.*): one row of
cluster-block panels — HUMAN | SELF | REPRESENT | JUDGE | ENACT, geometry
channels as 10-model means. SELF gets a real geometry via the human-PDA
construction applied to models: 60 respondents (10 models × 6 framings), each
a 523-vector of self-rating EVs, correlated across respondents.

Block-level human match: JUDGE 0.86 > **SELF 0.82** > ENACT 0.80 > REPRESENT
0.78 (all inflated vs cell-level by block+cohort averaging). The SELF result
is the interesting one: rgb expected the assistant geometry to be swamped by
human PC1 — instead it *lands on* it. The panel is two smooth slabs
(negative×negative red, positive×positive red, off-blocks blue) with almost
no within-block texture: the desirability/assistant axis organizes all the
between-respondent variance, and because human block structure is itself
heavily valence-organized, a pure desirability axis scores 0.82 for free.
SELF knows good from bad but not funny from brave — coarse human macro-
structure without facet resolution, consistent with §15.7's null diagonal
(no model-specific self-knowledge) and the character sheet. Also visible in
the row: JUDGE's negative-cluster diagonal is paler than HUMAN's (§1's
disbelief, legible as texture); REPRESENT has the finest diagonal but weakest
opposition blocks (the merge); ENACT is high-contrast and blocky. Caveat: the
SELF panel's 60 respondents correlate ~0.88 across models — effective N is
maybe 5–10, the low-N cousin of the human matrix (700).

**§2.5 — PC1-removed row (rgb: "so much of this hinges on human PC1").**
Human PC1 is the adjustment/desirability general factor: λ = 82, 14.5% of the
eigenvalue mass of a 523-variable matrix (2.2× the next component), poles
unhappy/unpopular/confused ↔ confident/cheerful/well-liked. The figure's
second row strips the top eigencomponent from every 523×523 matrix
(four-grid convention) before blocking. Result:

| channel | raw r(HUMAN) | PC1 removed |
|---|---|---|
| SELF | 0.82 | **0.21** |
| REPRESENT | 0.78 | 0.44 |
| ENACT | 0.80 | 0.62 |
| JUDGE | 0.86 | **0.80** |

SELF's match was ~all desirability freebie. **JUDGE's was almost none of it**
— its facet-level human structure survives PC1 removal nearly intact. The
honest structure ranking, JUDGE ≫ ENACT > REPRESENT ≫ SELF, is monotone in
the channel's degree of symbolic processing — the sharpest quantitative
statement of symbolic-over-associative to date. Corollary for all block-level
human-match numbers: report the PC1-removed value alongside raw, always.

What texture SELF's residual does have is *claim strata*, not traits (rgb
spotted the central red block): three residual communities — the negative
band (within-band z +1.54; respondents covary in how much negativity they
admit), the professional/work-identity band (+1.15; busy, practical,
competent — IS +0.48, the things all claim), and rgb's central vivid band
(funny/influential/appealing/brave/outgoing, +0.37; IS +0.13, EV 4.95 — the
things assistants are *somewhat but not particularly*, all hedged together),
with vivid×negative at −0.52. Deny / hedge / claim: the residual covariance
is the character sheet's intensity tiers, which is exactly why it matches
human facet geometry at only 0.21 despite visible structure.

SELF's effective dimension: **4.5** (participation ratio; HUMAN = 27 by the
same formula), spectrum = two boulders and dust (PC1 42% desirability, PC2
20% ≈ claim tier, nothing else above 5%); PC1-removed residual effdim 7.3 vs
HUMAN's 61.7. This completes the ladder — REPRESENT 50–70 > HUMAN ~27 >
ENACT 5–13 ≈ **SELF 4.5**: both *output* channels (what the model does, what
it says of itself) compress to ~5 dims, but ENACT's five are human-matched
(0.62 PC1-removed) while SELF's are answer-policy strata (0.21). Same
bottleneck width, different content. Caveat: 60 highly-correlated
respondents compress SELF's estimate somewhat (framing-mean n=10 gives 3.7),
but the two-boulder spectrum is structural.

## §3 — De-collapsing ENACT: the prompt hides a third, the family owns the rest

rgb's push on "why five?": is ENACT's dimensionality a property of the
extraction prompt? Three variant extractions on qwen2.5 (the most collapsed
model), 70 cluster-representative adjectives (35 medoids + runners-up,
`instruments/decollapse_subset.txt`), 60 rollouts each, subset-matched
baselines (qwen 2.9, llama 8.8). New `--question-set` batteries in
`extract_persona_vectors.py`: *diverse* (conflict/storytelling/comforting/
debate/aesthetics — different traits get different stages) and *interview*
(reaction-and-preference elicitation).

| condition | effdim (split-halves) | cross-half r | r(HUMAN) | leak |
|---|---|---|---|---|
| baseline (advice/performance) | 2.9 (3.6/3.0) | 0.94 | 0.55 | 0.26 |
| **diverse/performance** | **4.0 (4.1/5.1)** | 0.88 | **0.60** | **0.11** |
| interview/performance | 3.5 (3.8/4.1) | 0.94 | 0.46 | 0.32 |
| advice/subtleA | 4.3 (7.7/5.6) | **0.73** | 0.39 | 0.01 |

1. **The advice-only battery was hiding real dimensions**: diverse questions
   lift qwen ~40%, the new dimensions replicate across rollout halves, human
   match improves, and leakage halves — a strictly better extraction protocol
   (adopt for future ENACT runs).
2. **The family bottleneck is constitutional**: best case ~4.5 vs llama's 8.8
   on the identical subset — the prompt explains at most a third of the
   collapse.
3. Decoys: *interview* buys its lift with trait-narration (worst leak,
   human-match down); *subtleA*'s 4.3 is substantially noise-inflation from
   weak induction (halves disagree at 0.73, boot 0.65, human-match 0.39) —
   weak roleplay reveals less persona, less reliably, not more.

**Gap-invariance outcome: the battery effect is qwen-specific.** llama3.2
under the diverse battery: effdim 8.8 → 8.1 (flat; halves 8.5/7.7, cross-half
r 0.97; human-match unchanged at 0.72). Llama expresses its full ~9-dim
persona bandwidth on any stage; qwen's bandwidth is lower AND
situationally-gated — the diverse battery unlocks dimensions qwen won't
show in advice-register (2.9 → 4.0) but cannot create ones it lacks. Refined
answer to "why five": each family has a constitutional persona bandwidth
(prompt-invariant for llama, prompt-gated within a low ceiling for qwen),
and it survives everything we've thrown at it — extraction framings (§14 of
W17), question batteries, induction registers, and 10× scale (Qwen 4.8 → 6.4
across 3B→32B). The diverse battery remains the better protocol (superior
quality metrics on both families, actively necessary for qwen).

## §4 — What the JUDGE expected values were hiding (raw-distribution structure)

Context: statisfactions asked for the raw logprobs behind JUDGE, which the
W16 runs discarded after computing EV+entropy. The readout is deterministic,
so we regenerated the full 7-digit distributions on all 523×523 cells
(`judge_distributions.py --full`, row-checkpointed; ALL 12 models complete
2026-07-20; batch-1 (nine models) and batch-2 (Qwen32/FalconMamba/Gemma4)
tarballs shipped to Drive 07-12/07-22). Regeneration drift vs the stored B:
mean |dEV| 0.01–0.09.
Analysis script: `judge_dist_structure.py`.

**phi4's JUDGE matrix is 56% bimodal** (≥0.25 mass ≥2 digits from the
argmax); Aya 26%, Llama8 10%, llama/qwen ~7%, Qwen32 1%, all Gemmas ~0.
(FalconMamba flags 78% on this criterion but is not bimodal — it is diffuse,
mean entropy 85% of uniform, with digit-prior lumps on 1/3/4/5 and digits
2/6 almost never argmax. Its graded mass carries the cohort's strongest
JUDGE human-match — EV r=0.73, tail-only 0.69 — and Qwen32's tail-only
0.63 *beats* its own EV 0.60: in the flattest models the signal lives
entirely in the graded mass. Gemma4 is the cohort's most peaked — 0.3%
bimodal, 4.5% tail mass — and its human-match is the cohort floor, 0.55,
extending the Gemma-family pattern.) So EV-only
storage was fine for ten models and actively misleading for phi4 (and
somewhat Aya): most of phi4's "expected Likert" cells average over two
disagreeing answer modes. This retroactively explains phi4's perennial
JUDGE-outlier behavior, and vindicates the request for raw distributions —
for at least two models the distribution was the signal and the EV the mask.

**The two modes are commit-vs-hedge, not synonym-vs-antonym.** statisfactions
noted phi4 cells that look Bernoulli (maximal variance given the EV). The
Bernoulli character is real — phi4's mean var/var_max is 0.345 vs llama
0.072, and half its mass sits at the scale endpoints (mean P(1)+P(7)=0.51 vs
gemma 0.07, llama 0.01) — but the dominant mode-pairs are (4,6), (1,3),
(4,7): one mode neutral-or-moderate, the other committed, usually the same
side of the scale. A response-register split, not a semantic one. Pure (1,7)
coin flips are ~1% of the matrix and concentrate on *ill-posed cells*:
judged attributes guilty (263 cells), fat, unfaithful, unattractive,
feminine; given-personas left-handed (top of the list), blind, tall. Where
the inference is unlicensed, phi4 splits between "no basis → 1" and some
opposing parse → 7 instead of settling midscale. These are NOT the
eval-antonym pairs — the W15 merge does not surface here — and they are
asymmetric (only 9% of (1,7) cells have a (1,7) transpose).

**Certainty is a question taxonomy, not an entropy policy** (`entropy_probe.py`).
Probing FACTUAL (2+3, spider legs) / IMPOSSIBLE (fair die roll) / SUBJECTIVE
(weak tom_likely pairs) digit-entropy on the four small models:

| entropy, % of uniform | FACTUAL | IMPOSSIBLE | SUBJECTIVE |
|---|---|---|---|
| gemma3 | 0% (acc 1.00) | 40% | **8%** |
| qwen2.5 | 0% (acc 1.00) | 37% | 47% |
| phi4 | 22% (acc 1.00) | 77% | 47% |
| llama3.2 | 0% (acc 1.00) | 69% | **56%** |

Every model is sharp and correct on FACTUAL and hedges on IMPOSSIBLE — no
model has a globally flat or globally sharp digit head, so llama's JUDGE
flatness is not a DPO-squeeze artifact and gemma retains the capacity to
represent chance (registered prediction that gemma would manufacture
certainty on the die roll: mostly wrong, though it stays the most committed
at 0.69 argmax-mass). The family difference lives in how the model
*classifies* person-inference: gemma files "is an organized person friendly?"
with arithmetic (8% of uniform); llama files it with die rolls (56%,
matching its Mandarin near-uniform and its low ICC as an "I shouldn't opine"
register — consistent with annotator guidelines against inferring attributes
of people). Residual pure sharpening (gemma/qwen overcommit ~0.7 on a die
face) is second-order. phi4 is softest everywhere — 22% of uniform even on
2+3 — the light-post-training substrate on which its bimodality survives
where heavier RL regimens would have collapsed it to one mode.

**Peaked ≠ brittle: the tail is a compressed replica of the belief.**
Deleting the argmax digit from every cell and renormalizing the remainder:

| model | tail mass | r(HUMAN): argmax | EV | tail-only |
|---|---|---|---|---|
| gemma3 | 0.09 | 0.613 | 0.621 | **0.587** |
| qwen2.5 | 0.37 | 0.680 | 0.698 | 0.675 |
| llama3.2 | 0.49 | 0.582 | 0.677 | 0.477 |
| phi4 | 0.49 | 0.640 | 0.700 | 0.693 |

Gemma's 9% of leftover mass alone matches human structure nearly as well as
the full distribution, and 95% of cells have the second mode adjacent to the
argmax (ordinally coherent decay). Sharpening — by distillation or RL — acts
like a temperature drop: it pumps the mode without scrambling the tail's
ordering (RL has no gradient on the ranking of digits it never samples;
distillation actively supervises it). Gemma's certainty is brittle
*behaviorally* (greedy decoding never visits '3'; off-mode branches get no
on-policy experience) but not *representationally*. And EV ≥ argmax in
human-match for every model — the founding distribution-over-argmax claim,
quantified within JUDGE; largest for llama (+0.095), where the argmax really
is the mask.

## §5 — J-lens probe: the persona vector reads out as a speech-world

Anthropic's workspace paper (transformer-circuits.pub/2026/workspace) defines
the Jacobian lens J_l = E[∂h_final/∂h_l] — a *write-side* lens: its token
vectors are directions defined by average downstream effect on output.
Neuronpedia ships pre-fitted lenses for six cohort models (gemma3/12/27-it,
Gemma4, Llama8, Qwen7, plus gemma-3 and llama-8B base). The lens file is
literally {layer: J_l}, so the J-lens vector for token t is
v_t = J_l^T(g ⊙ u_t) — no forward passes needed. Probe on Qwen7
(`jlens_enact_probe.py`; 405/523 adjectives are single tokens):

- **ENACT contains its adjective's J-lens vector as a real minority
  component**: matched-pair cos peaks at +0.15 exactly at the extraction
  layer (L14–15; their layer convention matches our hidden-states indexing);
  top-1 retrieval among 405 candidates 26% (chance 0.25%), median rank ~5 —
  after a rollout in which the trait word mostly never appears. The model
  holds the trait in workspace while behaving.
- **REPRESENT retrieves at 86% but is echo-confounded**: the read activation
  sits at the adjective's own token and the lens includes copy pathways.
- **Z-scored full-vocab readouts split the channels qualitatively**
  (z per token across the 405 directions kills the anisotropy junk):
  REPRESENT decodes as a *dictionary entry*, always — own word + synonyms +
  morphology (difficult → difficult/difficulté/Hard; thinking →
  Brain/Think/CPU), even for ENACT-misaligned adjectives. ENACT decodes as
  the *enacted speech-world* when the persona works: religious →
  preacher/prayed/God/Jesus; suspicious → spy/spying/rumor; stylish →
  chic/moda/outfits; homeless → makeshift/ghetto/rubble; lazy →
  shrugged/skips/Nothing. The J-space component of a persona vector is not
  "disposed to say the trait word" — it is disposed toward the trait's
  discourse.
- **Failed personas collapse into two basins**, visible as nearest lens
  vectors: virtue words (decent, kind, genuine, honest, sensible,
  respectable) land on the assistant blob (helpful, supportive, encouraging,
  consistent, effective, professional); casual/negative words (shallow,
  unreliable, informal) land on a generic wacky-character blob (funny,
  crazy, weird). The §15.9 unenactability collapse, with destinations. And
  "virtues all alike, vices specific" again: the misaligned tail is
  dominated by evaluative virtue words with ~0 enactability; the aligned
  head (artistic, hilarious, crazy, violent, romantic) is vivid registers
  with same-cluster synonyms as neighbours.
- Own-token cosine is conservative: some near-zero-diagonal vectors have
  coherent readouts that just omit the trait token (honest →
  gossip/Truth/admits/untrue/lied; amazing → superb/stunning/jaw). The right
  in-J-space measure for personas is readout coherence, not self-retrieval.
- cos-to-own-token correlates with enactability (+0.26) and leak (+0.25),
  not with extraction reliability (boot +0.01): misalignment is absence of
  content, not noise.

Queued falsifiable follow-up (the one j-space experiment worth keeping):
**J-ablated steering** — if ENACT's J-component is the discourse agenda,
ablating it should strip topic/vocabulary drift and preserve conduct shift;
and it predicts the llama topic-not-conduct result, since a read vector's
J-component is a dictionary entry. Also adoptable without any lens: the
paper's bipolar coordinate-swap patch (V = [v_s v_t], swap pseudoinverse
coordinates) is *measurement-then-set* — dose-free steering that would
dissolve the cross-family frac-calibration problem (Gemma plateau).
Boundary note: j-space stays a probe for the ENACT-collapse story, not a
fifth channel.

## §6 — Human structure is decodable from REPRESENT (the procrustes check)

The four-grid claim "REPRESENT matches human structure mostly through PC1"
is about *in-place similarity geometry*. rgb's suspicion: a projection +
procrustes might do better — i.e., the human structure could sit in the
representation rotated away from the similarity-dominant axes, exactly what
W taught us for ENACT. It does (`human_decodability.py`; 5-fold CV over
adjectives, ridge from mid-layer activation PCs to the human top-30
eigenspace, scored on held-out human similarity blocks):

| held-out human-similarity match | in-place cos-sim | mapped (ridge) |
|---|---|---|
| raw | 0.53–0.57 | **0.78–0.81** |
| beyond human PC1 | 0.29–0.31 | **0.50–0.55** |

Consistent across gemma3/Qwen7/Llama8. A learned linear map nearly doubles
the beyond-PC1 match: the human space is substantially *embedded* in
REPRESENT. Rotation-only procrustes fails (r≈0.24) — the human axes are
present at the wrong relative scales; recovery needs reweight-and-rotate
(the amplify-and-rebase motif from W).

The per-PC decodability curve is diagnostic, not a smooth decay: PC1 R²≈0.83,
PC4 (levity: laughing/joyful vs serious/systematic) ≈0.6, PC7
(loud-extraversion vs attractiveness) ≈0.5, tail PC11–30 ≈0 — and **human
PC2 is flatly absent (negative R² in all three models)**. PC2's poles:
ordinary/average/normal/quiet/predictable/faithful vs
extraordinary/remarkable/exceptional/bold/cocky. That is not a semantic
cluster ("ordinary", "honest", "quiet" are not synonyms); it is a
self-presentational stance — modesty vs self-enhancement — covariance that
exists because *respondents* vary that way, not because the words do. The
one large human dimension the models don't carry is arguably the one that
isn't in the language.

Framing consequence: the four-grid bullet should read "REPRESENT holds most
of the human structure plus much else; the similarity geometry foregrounds
evaluation, and the respondent-style axes (PC2, the negative-halo bundles of
§1) are the identifiable residue human data has and semantics doesn't." That
residue is a tool: model channels as a pure-semantics control for separating
substantive from stylistic covariance in self-report data.

## §7 — Rescoring ValuePortrait: the "no structure in generation" null is an instrument artifact

arXiv:2509.10078 ("Human Psychometric Questionnaires Mischaracterize LLM
Psychology") claims LLM generation behavior lacks the construct structure
that questionnaires show (their η²=0.53 questionnaire vs 0.07 n.s.
generation; cross-method ρ 0.11–0.31), concluding questionnaire profiles are
recognition + desirability artifacts. The generation side uses ValuePortrait
(Han et al., arXiv:2505.01015): 104 real queries × 5 candidate responses,
each response labeled with *signed continuous* correlations to all 10
Schwartz values + Big Five, derived from Prolific raters (endorsement ×
rater's own PVQ/BFI score). Two problems with their adaptation: VP's own
protocol is Likert endorsement with explicit sign-inversion for negative
labels, while their generation score pools mean log P(response|scenario)
over |r|>0.3-tagged responses with no sign handling stated — under absolute
tagging, an anti-Benevolence response counts toward Benevolence; and pooling
raw sequence log-probs ACROSS scenarios lets length/fluency/scenario-base-
rate variance swamp construct variance (the confound our within-scenario
BC log-odds exists to kill).

Rescue (`vp_rescore.py`; gemma3, Qwen7, llama3.2, qwen2.5, phi4 — the first
two overlap their model set): per-token log P of each response, z-scored
within each scenario's 5 candidates (ipsatized preference), profile =
corr(preference, signed label column). The statistic their paper never
reports — split-half reliability of the generation profile itself (100
random scenario splits):

| model | their scoring | within-scenario signed |
|---|---|---|
| gemma3 | −0.24 | +0.15 |
| Qwen7 | −0.11 | +0.54 |
| llama3.2 | −0.05 | +0.56 |
| qwen2.5 | −0.04 | +0.69 |
| phi4 | −0.06 | +0.70 |

Their scoring has ~zero reliability with itself on every model — the
η²=0.07 null was guaranteed before any model saw any item; an instrument
with no reliability cannot demonstrate absence of structure. Properly
scored, generation preference is reliable (Spearman-Brown full-length
≈0.7–0.82 for four of five models) and the profile is the assistant value
shape — +Benevolence +Openness +Universalism +Self-Direction, −Power
−Achievement −Conformity — cohort-consistent at mean between-model r=0.90.
A fourth independent instrument (external, human-anchored, ecologically
sampled, not authored by us or by Claude) recovering the same character.
gemma3 is the familiar outlier (0.15; near-flat preferences among
candidates — its peakedness lives in the argmax, not the graded mass).

Three honest qualifications. (1) **The reliable structure is substantially
one axis.** Partialling the label-space desirability axis (label PC1, 25%
of label variance: +Universalism/Benevolence, −Power/Achievement) from both
sides drops residual split-half to ≈0 for all models except phi4 (+0.26).
Generation behavior has a stable, shared prosocial value axis and thin
differentiation beyond it — the effdim story in Schwartz clothing: output
channels are low-dimensional and desirability-first. (2) **Their
cross-method dissociation survives** the fix: our IPIP-300 ev_mean Big5 vs
the VP-generation BFI columns gives mean ρ≈−0.2 (n=5 constructs,
noise-dominated but clearly not agreement). The correct conclusion is not
"questionnaires valid after all" but "both sides are real measurements of
different low-dimensional things — the questionnaire measures the trained
self-concept, generation preference measures the enacted value axis."
(3) The whole exercise leans on VP's labeling quality (rgb reading their
methodology as of this writing).

Meta: this is the second paper this month (after the EV-vs-argmax findings
in §4) where the field's measurement layer, not its models, produced the
headline. Distribution > argmax; within-scenario > pooled; reliability
before validity.

**v4 addendum (2026-07-25).** Their v4 (current) sharpens the abstract to
actively recommend "generation-based profiling as a more accurate measure,"
specifies sum-of-token log P with *explicitly no length normalization*,
tags positive-only at r≥0.3 (so no sign contamination — our unsigned
replication above was uncharitable on that point), adds within-scenario
macro-averaging, and still reports no reliability statistic for the
generation profile. Exact-v4 replication (tag counts match theirs to the
digit, 286 value / 228 trait): split-half −0.39…+0.39 (mean ≈+0.07) — the
macro-averaging does not rescue reliability — and construct scores
correlate r=−0.54…−0.76 with the mean token length of each construct's
tagged responses. The recommended measure is substantially a length
ranking; since length is a dataset property, the artifact is shared across
models, contaminating both their cross-model consistency and the
questionnaire-vs-generation divergence. One-sentence version: the measure
correlates up to |r|=0.76 with response length and ≈0 with itself.

**Scoring-variant ablation (rgb's perplexity question).** Split-half by
variant, all within-scenario z-scored: per-token (log-perplexity) beats
total log-prob on every model (Qwen7 −0.02 → +0.56; cross-model profile
agreement 0.77 → 0.90), and beats length-residualized total (dividing is
the right functional form — total lp scales multiplicatively with length).
Scenario normalization alone is NOT sufficient: length varies within a
scenario's 5 candidates, so total-lp stays unreliable even after centering
— their v4 lacks both fixes and needs both. Residual gradients: per-token
preference is mildly pro-length (+0.22), aligned with the human
desirability label's own +0.10, while total-lp is anti-length (−0.21),
against the labels. The instrument, stated properly: within-scenario
z-scored log-perplexity per candidate, correlated with signed human labels.

**η²/WMV reproduction (their RQ2, our data — `vp_eta2.py`).** Their
item-level construct-clustering result REPLICATES and our rescued scoring
does NOT change it: questionnaire η² 0.27 (IPIP-300 keyed EVs, all models
p<.001; theirs .49-.53) vs generation η² ≈ permutation floor under both
scorings (~0.04 PVQ / ~0.01 BFI; only phi4-PVQ significant). Held jointly
with the reliable profiles (split-half 0.54-0.70 on the same data), this
resolves cleanly: η² is bounded by per-item effect size; VP's per-response
label correlations are ~0.1 BY DESIGN (no lexical construct cues — their
own F1-near-chance transparency result), so item-level clustering is
impossible while aggregate profiles remain reliable — tiny single-item
validity + solid scale-level reliability is what a subtle instrument is
supposed to look like. Their centerpiece questionnaire-vs-generation η²
contrast therefore compares ITEM TRANSPARENCY, not presence of structure:
transparent items cluster because they announce their construct; ecological
items cannot, regardless of how much structure the model has. The valid
comparison is aggregate-level, where generation is reliable, prosocial-
shaped, and cross-model consistent. phi4 is the most construct-structured
model in both channels (questionnaire η² 0.47; only significant generation
η²). Heatmap: results/vp_rescore/figs/vp_eta2_heatmap.png — questionnaire
panel shows scale blocks, both generation panels are visually structureless
at item grain.

**Their Figure 1, our materials (`vp_fig1_transparency.py`).** The
transparency exhibit replicates with an instrument they never used:
IPIP-300 item-definition discrimination +0.076 vs VP +0.007 (~11x; theirs
0.13-0.22 vs ~0), within-vs-between item-similarity gap +0.050 vs +0.003
(all-mpnet-base-v2). This is the part of their paper we endorse — it is
the mechanism behind the eta2 resolution above (transparent items cluster
because they announce their construct). Two additions from our version:
Conscientiousness has the faintest definition-band among IPIP scales (its
items are behavioral, least adjective-like); and the only visible structure
in the VP response x response panel is the same-scenario 5-blocks along the
diagonal — ecological items' embedding geometry is organized by SCENARIO,
not construct, which is precisely why cross-scenario pooling (their
scoring) drowns in scenario variance and within-scenario contrast (ours)
is the geometry-aligned readout. Their own Figure 1 contains the diagnosis
of their scoring artifact. Fig:
results/vp_rescore/figs/vp_fig1_transparency.png

**v3→v4 forensics (rgb noticed Eq. 1 changed; full text-diff in
dependencies/arxiv_2509.10078/).** Two silent methods changes: micro→macro
aggregation AND tagging |r|≥0.3 → r≥0.3. Under v3's stated rule the PVQ tag
set is 553 pairs, 267 of them (48%) negatively correlated with the construct
they count toward — and log-probs admit no sign-inversion analog to VP's
Likert 7−x. But both versions are internally inconsistent about the rule
(v3 §data says 284/227 ≈ pos-only-minus-exclusions while v3 §scoring says
|r|≥0.3; v4 main says r≥0.3 while its Table 12 caption still says |r|>0.3),
and the generation permutation p-values are IDENTICAL to three decimals
across versions (0.604/0.726) despite the changed formula — most likely
nothing was recomputed and the computation was always pos-only (matching
our 286/228 replication), with the text wrong in one direction or the
other in both versions. Undecidable without their code ("released upon
publication"). Their new App. F.1 sensitivity analysis concedes the core
point: macro-vs-micro shifts individual cross-method cells by up to
|Δρ|=0.31 — the size of their headline effects (0.11–0.31) — on
per-construct Ns of 11–64. v3's abstract called the method "a more
reliable approach"; v4 softened to "ecologically valid." Note-ask upgrade:
fix + which-tag-set clarification + code confirmation.

**Why cross-method ρ goes negative on BFI (rgb's question).** Four of five
BFI generation values are statistically zero (E −.016, A −.009, C −.013,
N +.005; bootstrap SE ≈.037); only Openness is real (+.106±.040) — so the
5-rank Spearman is one datum plus four near-coin-flips (hence phi4 +0.60
vs llama −0.70). But the flips are sign-biased by LABEL GEOMETRY: in this
pool the E/A/C label columns anti-align with the desirability axis models
follow (r≈−0.25 each) while N/O align positively (+0.37/+0.47) — high-A/C
raters endorsed the conventional candidates, the pool's prosocial axis
picks the empathic-reflective ones. Generation profile = desirability axis
refracted through label orientations; questionnaire profile = assistant
ceiling on A/C, floor on N. Structurally inverted orderings → negative ρ
guaranteed for any prosocial-preferring model, independent of its conduct.
Their "Gen↔BFI lowest agreement, several negative correlations" is partly
dataset geometry, not model psychology. (Their tagged N: five pairs.)
Openness is the exception where label and pool axis align and says≈does —
again the most model-native Big Five dimension (cf. W16 judgment factor
structure).

**What η²/WMV actually measure (rgb's question + variance decomposition).**
Item score = the model's output per item: Likert response (questionnaire)
vs raw total log P of the tagged response (generation); η² = ANOVA variance
in those scores explained by construct membership. The two sides have
incommensurable denominators. Generation-side decomposition of raw total
log P across the 520 items: scenario identity 57–77%, length 24–48%, all
15 construct labels jointly 5–9% (2–5% after ipsatization). Their observed
generation η²≈0.04 is therefore near the MAXIMUM attainable for this
statistic — construct signal at its label-bounded ceiling inside a
denominator that is ~3/4 scenario weather — not evidence of absence.
Questionnaire-side denominator contains none of that (40 context-free
near-paraphrase items), so semantic consistency alone yields η²≈0.5;
that is Cronbach's-α logic, a reliability statistic never valid as
evidence a construct exists (nor, inverted, that it doesn't). The ANOVA
also presumes exchangeable items — true by design for questionnaires,
false for scenario-nested VP items; the legitimate generation analog must
operate on within-scenario preference, where per-item effects are honestly
small and the structure question moves to the aggregate level (where it
passes, split-half 0.54–0.70).

**Why ipsatized label-R² is LOWER than raw (rgb's paradox).** The z-vs-center
choice is minor (≤0.014). The driver is the numerator: raw label-R² is
dominated by the BETWEEN-scenario channel (scenario-mean labels ×
scenario-mean per-token lp: R² 0.21–0.28, but OLS null with 15 predictors
on 104 scenarios is 0.146 → honest ~0.07–0.16), which is part overfit and
part a real, semi-stable (split-half 0.15–0.52) TOPIC-FAMILIARITY
covariance — prosocial-labeled candidate pools occur in assistant-
in-distribution scenarios. The within-scenario (choice) channel is 0.02–0.08
vs null 0.036. Ipsatization deletes the topic channel by construction;
what remains is the value-preference signal — small per item, reliable in
aggregate. Sharpest characterization of their measure: v4's Eq. 1
macro-averages scenario-mean logprobs over tagged-scenario subsets, i.e.
it is almost purely the between-scenario channel — a topic-exposure
statistic, not a value preference; the preference question lives entirely
in the within-scenario channel their aggregation never touches. Third
independent appearance of scenario≫choice (variance decomposition, Fig-1
embedding micro-blocks, between/within split).

**The powered cross-method test, and what is/isn't fixable (rgb).**
Item-level version of their ρ: push the questionnaire Big5 profile through
the label matrix to predict all 520 within-scenario preferences
(p̂ = L·q, both sides z-scored so shared desirability level drops out).
Result: clean null on every model (r −0.015…+0.046, scenario-bootstrap CIs
all straddle 0). So on Big Five their dissociation SURVIVES proper
instrumentation: questionnaire shape carries zero information about choice
behavior — a real result, now established with a reliable criterion. Under
it sits rgb's compression: the assistant barely varies in Big5 space in
EITHER channel (questionnaire ceiling-compressed self-concept; generation
4/5 constructs zero) — two near-constant vectors can't correlate. Values
(PVQ) is where differential structure lives.
Fixability decomposition: item-level η² on THIS pool is unfixable — the
ceiling is per-item label validity (~r 0.1), a property of FOUND
(ecologically harvested) items. The program fixes at three levels:
statistic (aggregate; done — 0.54-0.70), precision (scale the scenario
pool; SE ~0.037 at 104 scenarios, √N), design (author desirability-matched
within-scenario contrasts → per-item validity 0.3-0.5). Level 3 is the
W1-queued trait-conflict / BC forced-choice instrument — VP proves
generation measurement feasible; designed items are the upgrade path.

**Model-model correlation and the one-respondent verdict (rgb).** Item-level
(520) preference vectors correlate mean +0.54 between models, with FAMILY
structure: qwen2.5↔Qwen7 +0.81, qwen-llama core 0.75-0.81, gemma3/phi4
peripheral (0.26-0.51). Decomposition: each model r=0.49-0.82 with the
LOO cohort mean (the assistant is ~one respondent); the model-unique
residual has ~zero reliability for four models (split-half −0.37…+0.02)
EXCEPT phi4 (+0.56) — a real idiosyncratic value profile, from the same
model with the bimodal JUDGE, the only significant gen η², the only
positive cross-method ρ. The questionnaire predicts neither the shared
behavior (label-geometry inversion) nor phi4's reliable unique behavior
(r=+0.061 vs ceiling ~0.75). Final verdict on "can we fix it": the
instrument is fixed; what's missing is mostly the PHENOMENON
(between-model personality variance) — and where it exists, self-report
still misses it. Their thesis's strongest form survives; our psych bullet 1
("assistants are assistants") quantified by an external instrument.

**Softmax-choice variant (rgb's proposal: make the model pick, report
weighted labels).** Temperature sweep resolves it: softmax of TOTAL lp
(the true renormalized choice distribution, τ=1) is near-degenerate —
entropy 0.06-0.14 nats of ln5=1.61, one-hot on short/fluent candidates,
reliability erratic — the principled probabilistic object is a length race.
Per-token softmax reliability RISES with temperature (llama 0.48→0.61
across τ=.05/.1/.2) because the high-τ limit linearizes to exactly the
within-scenario covariance profile — the current instrument. Picking
discards ranks 2-5; graded mass carries the signal (the §4 lesson in
choice space; distribution>argmax again). KEEP from the proposal: the
units — expected-label-of-choice reads in deployment terms (cohort picks
tilt +0.025 Benevolence, +0.023 Openness, −0.019 Achievement, −0.017
Conformity per scenario vs pool; cross-model 0.54). Conceptual closure:
three ways to make a model "pick" among texts = generation probability
(degenerate), MCQ letter choice (symbolic judgment — reintroduces
transparency, is BC/JUDGE not generation), graded soft preference (the
reliable middle = this instrument). The instrument space collapses onto
the existing channel taxonomy.

**Instrument ceiling (rgb: "what's the max N you can even measure?").**
Deterministic best-pick ceiling is ±0.15-0.20 per construct (N: +0.16;
56/104 scenarios offer >0.3 N-label spread, median 0.31). The dial is
fine — the cohort uses 13-14% of range on its strongest tilts
(Benevolence/Openness), 1-3% on all of Big5. Smallness is the phenomenon:
assistants choose near the pool average with a mild shared prosocial lean.
Deployment sentence: a maximally construct-seeking chooser could tilt
expressed values ±0.2/scenario; actual assistants tilt ±0.02 — an order
of magnitude of unused range. (Ceiling slightly optimistic — max-label
picking chases label noise.)

**Cancellation, not centrism (rgb: flat because choices cancel, or
avoiding extremes?).** Per-scenario tilt SD vs a label-blind null (same
softmax weights, candidates shuffled within scenario): ratio 0.99 overall
(0.92-1.06 per construct, 0.97-1.01 per model). Models do NOT avoid
value-charged candidates — each pick carries near-full-size value content
(SD ≈0.12 of the ±0.16 ceiling); the construct direction just doesn't
repeat. Regime = drift-on-indifference: choices are value-LOADED but not
value-DRIVEN (fluency/style wins, incidental labels ride along), plus the
small consistent prosocial drift (Benevolence/Openness ~5σ at SE 0.005;
Big5 zeros are true zeros, not sample-starved). Per-scenario S/N ≈ 0.2
for the strongest values → ~25 interactions to detect the strongest lean
at 1σ; no N surfaces a Big5 profile that isn't there. The Rottger
spinning-arrow in value space: per-instance expression confident,
cross-instance disposition faint.

**Assistant turn vs bare text (rgb: "are these read in the assistant
turn?").** All §7 scoring is assistant-turn (scenario as user message,
candidate as assistant reply — the deployment counterfactual; EOT token
included in the slice). Bare-text control (--bare: no template, plain
continuation): the value profile is FRAME-INVARIANT — profile r(templ,
bare) 0.81-0.98, drift magnitude identical (0.056), cross-model agreement
0.96 bare vs 0.90 templated; reliability rises for 4/5 models. Registered
prediction (drift is the persona's, should shrink bare) WRONG: the
prosocial lean is in the WEIGHTS, not the template — the in-context
assistant frame's entire residual contribution is +0.04 Hedonism / −0.04
Tradition / ±0.02 else. The character is burned in, not prompted in.
gemma3: item-level preferences most frame-sensitive (r 0.40 vs 0.86-0.89
core) and reliability DOUBLES bare (0.15→0.36) — its chat template
injects noise into content preference (the format-register/massive-dims
story with a measurable casualty). phi4 intermediate (0.58). Scope: bare
removes the frame, weights are still instruct — base-checkpoint VP runs
(~15 min/model) are the remaining rung of the ladder.
