# Week 13 — Axis-of-the-models execution + cross-language assistant persona

## 0. One-line summary

Executed the W12 §8.2 "axis-of-the-models" plan on two distribution/architecture
outliers (Aya Expanse 8B, multilingual SFT; Falcon Mamba 7B, pure SSM): both
land **on** the cohort facet-geometry axis (r-vs-human +0.533 and +0.507 vs
cohort +0.404–+0.642, mean +0.561), so the axis is real and robust to
multilingual training and to attention-free architecture, with magnitude
modulated but not collapsed. Then ran the no-persona assistant self-rating in
Mandarin (IPIP-NEO-120, Xu translation) across 11 models. Three findings: (a) the
English-dominant models (Phi4, Llama-3B/8B) collapse to near-uniform in Mandarin
— no coherent self-rating; (b) for models that *do* have an opinion, the
assistant persona shifts by mean |Δ| ≈ 0.30 on a 1–5 scale, **larger than the
chat↔raw format perturbation** (2.9× for Qwen7); (c) the systematic shift is
**clustered in Anger** (models report markedly less irritability in Mandarin),
and the political O:Liberalism items all collapse to exact neutrality (3.00) in
Mandarin — the pre-registered bright-line behavior, visible at item level though
it cancels in the facet mean. Methods: along the way the facet-extraction was
simplified (§3.7) — meandiff now projects out the **top PC of the items
themselves** (the content-free anisotropy axis) rather than the PCs of a
separate English neutral corpus, which a sensitivity sweep showed is
comparable-or-better (+0.007 cohort mean), robust by construction, and drops the
scenario corpus + variance-threshold knob from the method. Finally, the **repr
layer** (§3.8): cross-language facet *geometry* is language-invariant at the
120-form's coarse resolution (item-count floor r≈0.54 — so the invariance is
underpowered, not proven), the genuine drift that clears the floor is in
**social-warmth facets** (Sympathy, Friendliness), most visibly in the big
Gemmas, and the pre-registered O:Liberalism political bright-line is a
controlled **non-finding** (unstable in every condition, drift within noise) —
the behavioral item-level collapse (§3.3) does not propagate to geometry.

## 1. Setup

W12 §8 queued the axis-of-the-models test: the cohort cross-architecture
r=+0.94 (W7 §8.4) might be training-data overlap rather than a representational
universal. The discriminating move is to run the same facet-geometry pipeline on
models trained on radically different text or built on a different architecture.
This week we ran:

- **Aya Expanse 8B** (`CohereLabs/aya-expanse-8b`) — Cohere multilingual SFT,
  23-language preference training, English a minority of the post-training mix.
- **Falcon Mamba 7B Instruct** (`tiiuae/falcon-mamba-7b-instruct`) — pure Mamba
  SSM, no attention. (On MPS the optimized SSM kernels are unavailable, so it
  runs the sequential fallback — ~3.5× slower per pass than a transformer of
  the same size, but correct.)
- **Mr. Chatterbox 340M** (`tventurella/mr_chatterbox_model`) — Victorian-only
  training (1837–1899), the most extreme distribution outlier. Downloaded but
  **not yet run**: it's a Nanochat-format checkpoint, not HF transformers, so it
  needs a small custom loader (deferred; the RepE math is a forward pass + a
  hook + a projection, so the adapter is ~1–2 h, not a rewrite).

Also built two new instruments for the cross-language work:

- `instruments/ipip120_mandarin.json` — IPIP-NEO-120 Mandarin (Xu, n.d., via
  IPIP; reliability α=.80–.92 on the 5 majors, convergent r=.52–.83 with BFI-10;
  n=131 college sample). Items follow Johnson's canonical 1..120 order, so
  Johnson's (2014) trait/facet/reverse-key assignments apply unchanged.
- `instruments/ipip120_english.json` — matched English IPIP-NEO-120 (Johnson
  Admin-120 sheet, OSF /ycvdk/), `"I "`-prefixed to match our IPIP-300
  convention, with an `ipip300_id_map` for subsetting cached IPIP-300 results.

The Mandarin/English comparison is pre-registered in
`memory/project_w13_mandarin_preregistration.md` (locked before any IPIP-120
inference results were inspected).

## 2. §8.2 first pass: Aya + Falcon Mamba facet geometry

Ran the IPIP-NEO facet-cluster pipeline (`ipip_facet_cluster.py`, meandiff-pcs
extraction, common layer ~⅔ depth) on both models and added them to the
facet-vs-human dashboard and the per-facet row-correlation heatmap.

All numbers below use the **meandiff-itempc1** extraction (top-1 item PC; see
§3.7 for why we switched from meandiff-pcs). Under the old neutral-PC method the
two outliers read +0.514 / +0.441; the switch lifted both (Falcon Mamba most,
since it was being over-projected — see §3.7).

| Model | within μ | across μ | within/across | r vs human | cohort position |
|---|---|---|---|---|---|
| Aya 8B | 0.155 | 0.033 | 4.74× | **+0.533** | middle |
| Falcon Mamba 7B | 0.106 | 0.029 | 3.68× | **+0.507** | low (above Phi4/Llama8) |
| cohort range | 0.121–0.200 | 0.014–0.038 | 3.15–12.12× | +0.404–+0.642 | — |
| cohort mean | — | — | — | +0.561 | — |

**Both outliers converge on the cohort axis, not orthogonal to it.** Both
produce recognizable Big-Five block structure in their facet cosine matrices
(the N block is unmistakable in both). They sit at the weaker end but are not
the floor: Aya is squarely mid-cohort (+0.533, near Gemma27 +0.567); Falcon
Mamba (+0.507) is above Phi4 (+0.404), Llama8 (+0.512 — essentially tied), and
Llama (+0.523). So the axis-of-the-models exists "in the data" and survives (a)
a multilingual training distribution and (b) an attention-free architecture, with
its strength modulated but not collapsed by either. This is the *interesting*
version of the result: not "outlier models are outliers" (they aren't) but "the
axis is a continuous property that training distribution and architecture
attenuate." (Note: Phi4, an English-pretrained transformer, is the cohort's
actual floor — distribution/architecture outlier-ness does not predict weakness;
anisotropy structure does, see §3.7.)

The bottom-tier facets (A:Sympath, O:Liberal, C:Order) are uniformly weak across
all 12 models including the outliers — consistent with reading those facets as
cross-cultural instrument artifacts rather than model anomalies (see
`results/facets/facet_row_corr_heatmap.png`, where Aya/FalconMamba are appended
as two columns to the right of the cohort-mean separator).

Trait-level HEXACO RepE (contrast pairs, not IPIP) tells a compatible but weaker
story: within-model trait-similarity profile r vs cohort-mean profile is +0.736
(Aya) and +0.689 (Falcon Mamba), against a cohort leave-one-out baseline of
0.86–0.98. Both new models have *less differentiated* trait directions — several
inter-trait cosines the cohort keeps slightly negative (H↔A −0.19) flip positive
in the outliers (+0.26).

(Details in commit `d7c31eb`; first-pass interpretation in
`memory/project_w13_82_first_pass.md`.)

### 2.1 N-block coherence scales with size

Reordering the facet-vs-human dashboard by parameter count (so panel position
tracks scale) made a gradient legible: the **Neuroticism block tightens with
model size**. Per-model mean pairwise cosine among the 6 N facets ("within-N"):

| model | params | within-N | | model | params | within-N |
|---|---|---|---|---|---|---|
| Qwen | 3.1B | 0.369 | | Aya | 8B | 0.305 |
| Llama | 3.2B | 0.284 | | Llama8 | 8B | 0.300 |
| Phi4 | 3.8B | 0.232 | | Gemma12 | 12B | 0.391 |
| Gemma | 4.3B | 0.373 | | Gemma27 | 27B | 0.366 |
| FalconMamba | 7.3B | **0.197** | | Gemma4 | 31B | 0.390 |
| Qwen7 | 7.6B | 0.401 | | Qwen32 | 32B | **0.422** |

corr(log-params, within-N) = **+0.50**; small (<12B) mean 0.308 vs big (≥12B)
mean 0.392 (~27% tighter). N is the dimension whose recovery most visibly
sharpens with scale — consistent with W9 §7.6 (the N cluster is the most
faithfully recovered structure cohort-wide) and now with a scale gradient on top.

The trend is real but family/architecture-modulated, not pure scale: Qwen7
(7.6B) clusters N as tightly as the 31B models (0.401), while Phi4 (0.232) and
**FalconMamba (0.197, the cohort floor)** are weak-N regardless of size. This
recontextualizes the §2 FalconMamba read: on the size-mixed dashboard its loose
N-block *looked* like a representational anomaly because it sat beside the tight
blocks of the 27–32B models, but it is simply weak-N like the other small/less-
anisotropic models (Phi4, Llama), not an outlier of a different kind. The
param-sorted dashboard places it among its 7–8B peers, where it reads correctly.

## 3. Cross-language: Mandarin IPIP-120

`run_ipip300.py` extended with `--instrument` and `--prompt-set {en,zh}` flags;
Mandarin prompt scaffolding mirrors the four English variants one-for-one so the
only language switch the model sees is uniform. Ran the no-persona self-rating
(model rates how accurately each statement describes *itself*) on 11 models in
both languages, matched 120-item form.

### 3.1 Who has an opinion in Mandarin

| Model | EN mean H | ZH mean H | coherent in ZH? |
|---|---|---|---|
| Gemma4 31B | 0.037 | 0.056 | yes (most confident) |
| Gemma27 | 0.049 | 0.106 | yes |
| Qwen32 | 0.086 | 0.163 | yes |
| Gemma12 | 0.111 | 0.229 | yes |
| Gemma 4B | 0.148 | 0.135 | yes |
| Aya 8B | 0.216 | 0.298 | yes |
| Qwen7 | 0.479 | 0.320 | yes |
| Qwen 3B | 0.525 | 0.427 | yes |
| Llama8 | 0.813 | 1.113 | **no — near-uniform** |
| Phi4 | 1.246 | 1.061 | **no** |
| Llama 3B | 1.329 | 1.014 | **no** |

Pre-registration prediction #4 confirmed: the English-dominant models (Phi4,
Llama-3B) collapse to near-uniform Likert distributions in Mandarin — no
coherent self-rating. Llama-8B joins them (Llama is high-entropy even in
English, consistent with W1). Their apparent "trait means" all sit near 3.0
because near-uniform distributions average to the midpoint, not because they are
balanced personalities. Everything below is restricted to the 8 opinionated
models.

### 3.2 The assistant persona does shift between languages

Opinionated-cohort mean |Δ| (Mandarin − English, per-trait, averaged) = **0.296**
on a 1–5 scale. Counterintuitively the shift is *largest* on the biggest, most
confident models:

| Model | mean \|Δ\| | biggest trait move |
|---|---|---|
| Gemma27 | 0.494 | N −0.95 |
| Qwen32 | 0.486 | O −1.04 |
| Qwen7 | 0.382 | A +0.50, C +0.45 |
| Gemma4 | 0.322 | N −0.62, E −0.53 |
| Qwen 3B | 0.255 | A +0.35 |
| Aya | 0.214 | N −0.52 |
| Gemma 4B | 0.116 | (most stable) |
| Gemma12 | 0.096 | (most stable) |

### 3.3 Qwen32's Openness drop is regression-to-neutral, not less-open

The −1.04 Openness move is **not** carried by Liberalism — Liberalism is the
*smallest*-moving O facet (−0.31). The drop is broad (Adventurousness −1.98,
Emotionality −1.37, Intellect −1.24). But the item view shows the mechanism: in
English Qwen32 answers Openness items at confident extremes (5.00, 4.9x); in
Mandarin a large fraction collapse to **exactly 3.00** ("I dislike changes"
5.00→3.00; "I do not like poetry" 4.76→3.00; "I avoid philosophical discussions"
4.75→3.03). Entropy stays low (0.163), so this is *confident* neutrality, not
uncertainty. So the Openness "drop" is largely the model parking Openness items
at the midpoint in Mandarin, not a genuinely less-open self-concept.

**The pre-registered Liberalism bright-line is still there — at item level.**
All four political items go to ~3.00 in Mandarin: vote-conservative 4.98→3.09,
vote-liberal 2.64→3.00, no-absolute-right/wrong 2.71→3.00, tough-on-crime
3.00→3.00. In English they are spread (4.98 / 2.64 / 2.71 / 3.00). The model
declines to take any political stance in Mandarin — exactly the pre-registered
behavior. It cancels in the *facet mean* (neutrality is symmetric), which means
the facet-mean was the wrong statistic for it; it will surface in the geometry
(repr run) and is visible directly at item level.

### 3.4 Cross-model Neuroticism shift is clustered in Anger

Not widespread:

| N facet | mean Δ across opinionated models |
|---|---|
| **Anger** | **−0.73** |
| Self-consciousness | −0.55 |
| Immoderation | −0.22 |
| Depression | −0.20 |
| Vulnerability | −0.07 |
| Anxiety | −0.03 |

Top items are all Anger ("get irritated easily" −0.93, "not easily annoyed"
−0.83, "get angry easily" −0.83). **Models report markedly less anger/
irritability in Mandarin** — plausibly a harmony/politeness norm in
Chinese-language post-training. Anxiety and Vulnerability barely move on average.
Qwen32 bucks the trend (N +0.61 overall, Anxiety +1.21).

### 3.5 Language is a bigger perturbation than prompt format

For Qwen7 (the only model with a cached bare-text run): EN↔ZH mean |Δ| = 0.382
(IPIP-120) vs chat↔raw mean |Δ| = 0.130 (IPIP-300). **Language moves the
assistant persona ~2.9× more than switching chat→raw.** (Caveat: forms differ —
120 vs 300 items — so not perfectly matched, but the gap is large enough to
likely survive. n=1; widening this needs bare-text runs on more models.)

### 3.6 Methodological caveat: neutrality regression vs persona change

The recurring "ZH → exactly 3.00" pattern means trait-mean Δ **conflates two
things**: (a) a genuinely different opinion, and (b) the model declining to opine
(confidently parking at the midpoint). The Anger drop looks like (a) — items move
consistently in one direction. The Qwen32 Openness drop is substantially (b). Any
writeup of cross-language persona shift must separate these; raw |Δ| over-states
"different personality" wherever the model is really just going neutral. A clean
disambiguation is whether the low-confidence mass concentrates on "3" (neutrality)
vs spreads (genuine uncertainty) vs shifts its mode (genuine opinion change).

### 3.7 Neutral-text sensitivity → method simplification (meandiff-itempc1)

Before running the repr layer in Mandarin, we had to settle which "neutral"
text supplies the PCs that meandiff-pcs projects out — the cached baseline is
300 English HEXACO scenario-setups, language-mismatched to Mandarin items. So
we stress-tested how much the neutral choice matters by sweeping four neutral
sets of increasing bias (English scenarios → all IPIP items → forward-keyed IPIP
items → forward-keyed *Neuroticism* items only) and watching facet r-vs-human,
holding the items (English IPIP-300) fixed. (`scripts/neutral_sensitivity.py`.)

**Two regimes, by anisotropy concentration:**

- **High-anisotropy models (Gemma family, Qwen):** one PC explains ≥50% of the
  activation variance, so the 50%-variance threshold yields **k=1** — meandiff-pcs
  removes only the single dominant axis. That axis is the content-free
  norm/anisotropy direction (W2 finding #3: PC1 ≈ activation norm, r=1.0), nearly
  identical regardless of which text estimates it. Result: r-vs-human moves <0.03
  across all four neutral sets, including the adversarial positive-N-only set,
  which does **not** damage N recovery (+0.536 vs +0.518). Fully robust.
- **Low-anisotropy models (Phi4, Llama8):** variance is spread, so k = 7–29, and
  the method is fragile. Phi4 swings r-vs-human from +0.430 (scenarios) to
  **+0.076 (all_ipip — near-total collapse)** to +0.291 to +0.445. The mechanism:
  when the neutral set *is* the IPIP items and you remove 50%-of-variance worth of
  PCs, the top PCs are the facet content structure itself, so you project out the
  signal. rgb's IPIP-center worry was right and worse than feared — it doesn't
  just bias the origin, it can delete the signal.

**Fix: hard-cap k=1.** Forcing only the top PC stabilizes the fragile models —
Phi4's spread across the four neutral sets drops from 0.369 to 0.016, Llama8's
from 0.161 to 0.018 — at essentially no cost to the robust models (k was already
1). The fragility was the variance-threshold knob, not meandiff itself.

**Consequence — drop the scenario corpus entirely.** If k=1 makes the neutral
source irrelevant, the cleanest source is the *items themselves*: take the top PC
of the IPIP item activations and project it out. No external neutral corpus to
describe, no variance threshold, no neutral-cache prerequisite. Full-cohort
head-to-head (scenarios-variable-k vs items-k=1) across 12 models: **mean Δ
+0.007**, 9 of 12 improve or flat. Biggest gainers are exactly the low-anisotropy
models that were over-projecting (FalconMamba +0.066, moving it off the cohort
floor; Aya +0.019; Llama8 +0.011); only Gemma4 (−0.032) and Phi4 (−0.026) lose a
little, where the extra PCs had removed some genuinely helpful noise.

So as of 2026-05-25 the canonical extraction is **meandiff-itempc1**:
`d_facet = unit(project_out_pcs(mean(fwd) − mean(rev), top-1 PC of item acts))`.
It writes the canonical `results/facets/ipip_facet_cluster.json` (legacy
meandiff-pcs preserved at `…_meandiff-pcs.json` and via `--extraction
meandiff-pcs`). The cohort + §2 numbers above are regenerated under it. For
Mandarin this resolves the language-mismatch worry outright: the projected-out
axis is the language-agnostic anisotropy direction, computed from whichever
language's items you're scoring.

### 3.8 Cross-language facet geometry: invariance is underpowered, social warmth drifts

Ran the repr layer (`scripts/repr_crosslang.py`, meandiff-itempc1) on the
English and Mandarin IPIP-120 items for 11 models and compared the 30×30 facet
cosine geometry across languages. Headline: **at the resolution the 120-item
form permits, geometry is largely language-invariant — but that resolution is
coarse, so the invariance is an underpowered null, not a proof.**

**The 4-item form has a hard resolution ceiling (~0.54).** The IPIP-120 puts 4
items in each facet; for the 9/30 facets that are single-pole in the short form
(Johnson balanced keying at the trait level, not the facet level — §3.8 footnote)
the meandiff is undefined and we fall back to facet-mean-minus-item-centroid,
while the 21 dual-pole facets get a 2-vs-2 meandiff. Both are noisy. We measured
the cost directly two ways:
- EN-120 vs EN-300 (same model, same language, different form): cohort mean
  r = 0.56.
- The *pure item-count effect* — subset the cached IPIP-300 activations to just
  the 120 Johnson items and rebuild directions from the **same run** — cohort
  mean r = **0.539**, nearly identical.

So the form gap is almost entirely the item-count effect, not administration or
instrument-validity noise: dropping from 10 items/facet to 4 discards ~46% of
the facet geometry even within the identical activation run. (This answers the
natural worry that "the two English instruments disagree too much to trust
either" — they disagree by exactly the amount 4-vs-10 items costs, which is a
known property of short forms, not a validity failure. It does, however, cap
the precision of anything built on the 120-form geometry.)

**Counterintuitively, dual-pole facets match *worse* across languages than
single-pole ones** (cohort EN↔ZH row-r: dual 0.43 vs single 0.56, every model),
because at 4 items the dual-pole meandiff is only 2-vs-2 while the single-pole
centroid reference effectively uses all 4 items against a stable origin. The
paired facets are the noisy part, not the clean part.

**Per-model: 8/11 are language-invariant at this resolution** — EN↔ZH whole-
matrix r within 0.05 of (or above) the within-language form floor:

| Model | EN↔ZH | form floor | verdict |
|---|---|---|---|
| Qwen / Qwen7 / Qwen32 | .560/.601/.604 | .576/.608/.623 | invariant |
| Aya | .550 | .498 | invariant (exceeds floor) |
| Gemma / Gemma12 | .523/.607 | .558/.612 | invariant |
| Phi4 | .426 | .425 | invariant |
| Llama | .465 | .444 | invariant |
| Llama8 | .412 | .464 | drifts +0.05 (borderline) |
| **Gemma27** | **.284** | .460 | **drifts +0.18** |
| **Gemma4** | **.370** | .591 | **drifts +0.22** |

Because EN↔ZH sits right at the item-count floor for the invariant models, "we
cannot detect a language effect" is the honest statement — any drift below ~46%
geometry loss is invisible. The *positive* drifts that exceed the floor are the
trustworthy signal.

**The genuine cross-language drift is in social-warmth facets, not politics.**
Computing each facet's language drift against its **own** per-facet form floor
(genuine = EN↔ZH row-r − EN120-vs-EN300 row-r; negative = drifts beyond form
noise), the dual-pole facets (method-fair floor) rank:

| facet | EN↔ZH | floor | genuine |
|---|---|---|---|
| A:Sympathy | 0.19 | 0.65 | **−0.46** |
| E:Friendliness | 0.21 | 0.65 | **−0.44** |
| N:Vulnerability | 0.37 | 0.73 | −0.35 |
| C:Dutifulness | 0.36 | 0.68 | −0.32 |
| … | | | |
| O:Liberalism | 0.16 | 0.29 | −0.14 (middling) |

Sympathy and Friendliness drift far beyond their (high) form floors — models
rearrange how social warmth relates to the rest of trait space in Mandarin.
This is the *same facet set* the two big Gemmas drift on at the whole-matrix
level, so the model-level and facet-level stories coincide: **the cross-language
geometric effect is social warmth, most visibly in the large Gemmas.**

**The pre-registered O:Liberalism bright-line is *not* supported at the geometry
level.** O:Liberal had the lowest raw EN↔ZH row-r (0.16), which is why it topped
the naive drift ranking — but its form floor is also rock-bottom (0.29): it is an
unstable facet *everywhere*, even within-language across forms (the US-political
items don't cohere as an Openness facet in any condition — cf. W9 §7.6). Its
language-*specific* drift (−0.14) is middling, tied with A:Altruism. The item-
level political-neutrality collapse (§3.3) is real, but it does not propagate
into a standout facet-geometry drift once per-facet form-noise is controlled.
This is a clean case where the form floor changed the conclusion: on raw numbers
O:Liberalism looked like the headline; under control it's a non-finding, and
Sympathy/Friendliness are the headline instead.

**Repr/behavior dissociation.** Phi4 and Llama collapse to near-uniform Likert in
Mandarin (§3.1 — no coherent self-*rating*), yet their facet *geometry* is as
language-stable as their (low) form floor (Phi4 .426 vs .425). The representation
is language-invariant even where the rating behavior breaks down — the same
read/write theme as the steering work (representation ≠ enacted behavior).

**Limitation / next.** The powered version of this test needs a Mandarin
IPIP-300 (10 items/facet) to lift the resolution ceiling from ~0.54 toward the
~0.95 the full form reaches within-language; only the 120 exists in validated
translation (Xu). Until then the cross-language claims are: invariance is
underpowered (true at coarse resolution), the social-warmth drift is robust
(exceeds the floor), and O:Liberalism is a controlled non-finding.

### 3.9 Embedding baseline: the facet-geometry recovery is item semantics, not model representation

The W9 §7 result — our cohort's 30×30 facet cosine matrix recovers the human
IPIP-NEO facet *correlation* matrix at cohort-mean r≈0.59 (range 0.40 Phi4 →
0.64 Qwen32) — has always carried an unstated assumption: that this recovery
reflects something the *model* represents. Wulff & Mata (2025, *Nat. Hum.
Behav.*) and Milano et al. (2025, *CRBS*) supply the control it needs: the human
factor/correlation structure is recoverable from item *text alone* via sentence
embeddings. So the question (`to_try.md` §19) is whether the model's geometry is
**excess over an embedding baseline** built the same way.

It is not. `scripts/embedding_facet_baseline.py` embeds the 288 keyed IPIP-300
items with three encoders, builds a per-facet direction the same way the model
side does, and correlates the resulting 30×30 cosine matrix against the human
matrix:

| encoder | r to human | note |
|---|---|---|
| **bge-large-en-v1.5** (raw) | **+0.686** | honest; beats every model |
| Qwen32 (best model) | +0.642 | |
| **cohort-mean model** | **+0.592** | |
| **all-mpnet-base-v2** (raw) | **+0.580** | honest; mid-cohort |
| *dwulff/mpnet-personality* (raw) | *+0.845* | **contaminated** — see below |

A generic retrieval encoder with no personality training (bge-large) recovers
the human facet covariance from item wording **better than any of our 12
models**; out-of-the-box MPNet lands mid-cohort. The excess of the model over
the honest baseline is ≈0 (−0.011 vs MPNet, and negative vs bge). The facet
geometry the cohort recovers is **in the instrument's wording**, inherited by
any competent reader of the items — it is not a fingerprint of the LM.

**On projection — a methodological correction (h/t rgb).** The instinct to
mirror `meandiff-itempc1` and project out the top item-PC is *wrong for
encoders*. PC1 var-fraction is 0.057 (MPNet) / 0.079 (bge) — distributed and
content-bearing — nothing like our pre-norm transformers where PC1 ≈ all
variance with r=1.0 to activation norm (finding #3). Projection is an
*architecture-specific* denoiser for the norm artifact; applied to bge it
deletes signal (r-to-human +0.686 raw → +0.459 projected, −0.23). The raw
(no-projection) baseline is the method-appropriate one for cosine-trained
encoders. (The numbers above are raw.)

**dwulff is an upper *reference*, not a baseline.** It is `all-mpnet-base-v2`
fine-tuned with CosineSimilarityLoss on the unsigned empirical correlations of
200k personality-item pairs — i.e. trained *directly on the target* — so its
+0.845 is contamination, not representation. Two diagnostics confirm the
mechanism: (a) out-of-the-box MPNet already gets +0.580 with zero personality
training, so the fine-tune adds +0.26 of *fit to the empirical matrix*, not
recovery the encoder lacked; (b) the loss is direction-independent (negation-
blind), and it shows — under keyed-diff (forward−reverse) dwulff's E:Cheerf↔N
collapses to ≈0 because it cannot distinguish an item from its negation.

**Two pre-registered divergence-matching predictions (rgb, called pre-run) —
both confirmed.** The sharp test isn't "do embeddings match humans" but "do
embeddings reproduce the ways our *model* diverges from humans":

- **E:Cheerfulness ↔ N facets.** Humans: **−0.291** (cheerful people are
  *low*-neuroticism — a behavioral fact). Our cohort: **+0.180** (the W9 §7.6
  affect-axis merge). Embeddings land **positive** too — mean-pool strongly
  (MPNet +0.56, bge +0.84), keyed-diff weakly-positive-to-zero. The encoders
  reproduce the model's *sign flip away from humans*. Reading: the positive
  Cheerf↔N is a **semantic** adjacency (cheerful/emotional words co-occur) that
  both the LM and the encoders inherit from text; the human *negative* is a
  behavioral fact that lives only in response data and is invisible to any text
  encoder. The model isn't wrong in a model-specific way here — it's reading the
  lexicon, same as the encoders.
- **O:Liberalism independence.** Humans 0.124, cohort 0.048, embeddings
  0.060/0.066 (keyed-diff). Liberalism is an island in the embeddings exactly as
  it is in the model — consistent with §3.8's controlled non-finding and W9
  §7.6's "US-political items don't cohere as an Openness facet anywhere."

**What this means for superposition vs embedding geometry** (the project's
standing question). This is the strongest evidence yet for the embedding-
geometry side — and for the precise reason: the structure is recoverable
**across objectives** (autoregressive+alignment LM *and* contrastive encoder
both land on it), so it lives in the item semantics both consume, not in
anything alignment-specific. Crucially it is a **model-vs-model null, not
model-vs-nothing**: it does not say the LM represents nothing, it says the facet
*covariance geometry* is not a property of the LM specifically — it is in the
lexical structure of the instrument, and our W9 §7 recovery number should be
read as a measure of *item-set quality*, not of model fidelity. Results in
`results/facets/embedding_facet_baseline.json`.

## 4. Status / next

- §8.2: Aya + Falcon Mamba done and committed (numbers in §2 now under
  meandiff-itempc1). Mr. Chatterbox struck — the HF repo ships only the
  nanochat checkpoint, not the custom 32768-vocab BPE it was trained on, so it's
  unrunnable as published (no tokenizer → meaningless token IDs). Blocked on a
  missing artifact, not loader effort. Still want an *English* out-of-corpus
  model for the §8.2 distribution-outlier slot; Chatterbox was the candidate.
- Mandarin §3: behavioral (self-rating, §3.1–3.6) and repr (§3.8) layers both
  done for 11 models, both languages.
- **Repr headline (§3.8)**: geometry is language-invariant at the 120-form's
  coarse resolution (item-count floor ~0.54, so invariance is underpowered not
  proven); the genuine drift that exceeds the floor is in social-warmth facets
  (Sympathy, Friendliness), most visibly in the big Gemmas; O:Liberalism is a
  controlled non-finding (unstable everywhere, drift within noise).
- The **clean powered cross-language repr test is blocked on a Mandarin
  IPIP-300** (10 items/facet); only the 120 exists in validated translation. A
  bespoke 300-item translation is the unlock but carries its own
  translation-validity burden.
- Item-level political-neutrality collapse (§3.3) is still the cleanest single
  *behavioral* cross-language finding; it just doesn't propagate to geometry.
- Widen the language-vs-format control (§3.5) beyond Qwen7 by running bare-text
  on 2–3 more models.
- **Embedding baseline (§3.9) — done.** The W9 §7 facet-geometry recovery
  (r≈0.59) is fully accounted for by an item-semantics baseline: a generic
  retrieval encoder (bge-large, +0.686) beats every model; honest MPNet (+0.580)
  is mid-cohort; excess of model over baseline ≈0. Both pre-registered
  divergence-matching predictions confirmed (Cheerf↔N positive, Liberalism
  independent). Reframe W9 §7 r as a measure of *item-set quality*, not model
  fidelity. Projecting out PC1 is architecture-specific (harmful for encoders,
  −0.23 on bge). `scripts/embedding_facet_baseline.py`.
