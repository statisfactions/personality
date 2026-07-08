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
