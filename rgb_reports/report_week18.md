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
