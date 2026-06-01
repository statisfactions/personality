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

**The progenitor and the break point of the lexical hypothesis.** This line
starts with Cutler & Condon (2022, "Deep Lexical Hypothesis," arXiv:2203.02092)
— three years before Wulff & Mata and Milano — who recover the survey factor
structure from transformer representations over psycholexical *adjectives*
(first three factors congruent at 0.89/0.79/0.79) with one crucial caveat in
their abstract: **"Neuroticism and Openness are only weakly and inconsistently
recovered."** That caveat is the same wedge our §3.9 finds from the other side.
C&C name the strong claim they lean on — "it is the essence of the lexical
hypothesis that these [semantic vs behavioral] structures are the same" — and
report N/O as where it fails. Our Cheerf↔N result is a direct counterexample to
that identity claim at N: the human (behavioral) correlation is **−0.29**, the
semantic one is positive, and the LLMs side with the semantics. Two independent
methods land on **N as the break point of the strong lexical hypothesis** —
which is exactly why the project's stance (the rep/behavior gap is signal, not
the ~37% "unexplained" variance a pooled recovery-r averages away) is the
load-bearing one. A per-trait *within-block* check on the §3.9 matrices did not
cleanly reproduce the N/O deficit (N recovers fine here; C is the soft spot),
but that measures within-trait facet shape, not C&C's between-trait factor
congruence — the proper adjective-vs-item adjudication (per-trait cluster
separability) is unrun.

**Direct LLM↔encoder distance — the encoder is nearly just another model.**
The §3.9 framing above is "does the LLM beat the encoder at recovering humans"
(no). The sharper question is how close the LLM's own facet matrix is to the
*encoder's* facet matrix, compared to humans and to another LLM. Cohort means
(upper-tri Pearson r, encoders in their honest raw mode, model side canonical
meandiff-itempc1; `scripts/llm_vs_embedding_geometry.py`):

| neighbor | cohort-mean r |
|---|---|
| LLM ↔ LLM (cross-model) | **+0.906** |
| LLM ↔ encoder (mpnet/bge avg) | **+0.840** |
| encoder ↔ human (bge / mpnet) | +0.686 / +0.580 |
| LLM ↔ human | **+0.561** |

Every model is closer to the encoder than to humans, by **+0.279 cohort-mean**
(smallest Gemma4 +0.20, largest Llama8 +0.32). And the encoder is almost as
close to the LLM as another LLM is: LLM↔LLM 0.906 vs LLM↔encoder 0.840, gap
**0.066** — that 0.066 is the entire LLM-distinctive-and-shared-across-LLMs
residual; the other ~0.84 of cross-LLM agreement is item semantics that any
text model recovers. Bidirectional contrastive encoding (mpnet mean-pool,
bge CLS) lands on the same facet covariance as autoregressive+alignment LMs at
r≈0.84 — and recovers humans *better* than the LMs do — without ever having
attended causally or seen RLHF. (Caveat on the absolutes: the model/encoder
matrices share the gross within>across block structure that inflates all these
r's; the human matrix is the one with genuinely different fine structure, which
is most of why the human row is lower. The *ordering* is what matters and it is
unambiguous.) Mechanism-wise: for the embedding/word-geometry view the result
nails down the "what's the LLM-specific part?" residual at ≤7 r-points off a
contrastive encoder, ≤16 off a generic one — small enough that the §3.9
relabeling of the W9 §7 number as item-set quality stands, and small enough
that the project's positive claim has to live in the deviations (RepE/BC/free-
text dissociation, persona internalization, the read/write gap), not in this
geometry.

**Projection-robustness control (notes only; `scripts/facet_projection_sweep.py`).**
Two worries about the per-facet row-correlation finding (A:Sympath *inverts* vs
human, E:Cheerf/O:Liberal fail): is it an artifact of (a) the fwd−rev contrast
still carrying anisotropy, or (b) the *specific* top-1 item-PC projection
(`meandiff-itempc1`)? Sweeping the number of projected item PCs k∈{0,1,2,3}
(k=0 = raw meandiff, no projection; k=1 = canonical), cohort-mean row-r vs human:

| facet | k=0 (raw) | k=1 | k=2 | k=3 |
|---|---|---|---|---|
| A:Sympath | −0.111 | **−0.180** | −0.150 | −0.119 |
| E:Cheerf | +0.022 | +0.208 | +0.156 | +0.096 |
| O:Liberal | +0.030 | −0.232 | −0.219 | −0.195 |
| C:Order | +0.122 | −0.054 | −0.069 | −0.034 |
| O:Emotion | +0.115 | +0.375 | +0.411 | +0.443 |
| N:Anger | +0.313 | +0.765 | +0.763 | +0.739 |
| A:Altru | +0.253 | +0.395 | +0.405 | +0.368 |
| **[full matrix]** | **+0.257** | **+0.569** | +0.576 | +0.566 |

Two readings. **(a) Raw (k=0) is anisotropy-degraded — rgb's prior confirmed.**
Full-matrix r drops to +0.257, but the cohort mean hides the two-regime split of
§3.7: the massive-activation models collapse hardest (Gemma27 +0.060, Qwen32
+0.130 — near-nonsense), while Gemma-4 (no massive activation, §3.10) holds at
+0.561. So "raw" is not an architecture-fair comparison point — it is content for
the clean models and anisotropy for the spiky ones, which is exactly why there is
no honest "raw model vs raw encoder" control (the encoder's raw PC1 is
content-bearing; the transformer's is the norm artifact). The projection
asymmetry is forced, not a confound. **(b) The divergences are robust to k=1–3.**
Sympath stays inverted (−0.18→−0.12), Cheerf stays weak-positive, Liberal stays
inverted, full-matrix r is flat (.569/.576/.566) — none is a knife-edge of the
top-1 choice. The Sympath inversion is the sturdiest: it is negative even at k=0
(−0.111), whereas Liberal/Order only invert once the anisotropy axis is removed.
**One exception worth flagging:** Gemma-4 reverses Sympath to *positive* by k=2
(+0.166) / k=3 (+0.232) — its massive-activation-free residual stream behaves
differently under projection; the cohort sign holds because the other nine
dominate. Full results in `results/facets/facet_projection_sweep.json`.

### 3.10 Is the geometry in the token embeddings? (logit-lens) — weak lexical seed, amplified by depth

If §3.9 is right that the facet geometry is item semantics "so close to the
textual level," a logit-lens intuition (rgb) predicts it should already be in
the LLM's *token embeddings* — readable before the transformer stack does any
work. Two tests, both reading existing artifacts.

**Depth sweep on items (`scripts/facet_geometry_layer_sweep.py`).** Rebuilding
the meandiff-itempc1 facet matrix at *every* layer from the cached IPIP
activations (layer 0 = HF `hidden_states[0]` = embedding-layer output), then
correlating each layer's 30×30 matrix to the human matrix:

| | layer 0 | peak | peak layer | 2/3-depth (canonical) |
|---|---|---|---|---|
| cohort mean r-to-human | **0.110** | **0.652** | (mid–late) | 0.561 |

Layer 0 is near-floor (0.11); the geometry is *built up with depth*, peaking
mid-to-late, and the peak layer scales with model size (Phi4/Llama ~12–14,
Qwen32 ~45, Gemma27 ~62 — bigger models build it deeper). Not a projection
artifact (raw, no-PC1, is equally low at layer 0). **But this is a sentence-
composition result, not a test of the prediction:** IPIP items are multi-token,
mean-pooled, and ~75% diluted by identical chat-template tokens, so layer 0 here
is a bag-of-token-embeddings of a *sentence* — exactly the representation that
needs depth to compose. The prediction is about *single tokens*.

**Single-token marker test (`scripts/marker_embedding_geometry.py`).** Read the
input embedding matrix W_E directly (no forward pass; embedding tensor pulled
from cached safetensors, fine for the 32B) and look up single-token Big Five
adjective markers — Saucier (1994) Mini-Markers (Table 2, 40 items; keyed by
loading sign) and Goldberg (1992) transparent bipolar markers (Appendix B, 35
pairs). Cohort:

| set | within−across cos | NN-purity | 5-cluster purity | trait-dir \|off-diag\| |
|---|---|---|---|---|
| Saucier | +0.038 | 0.66 | 0.53 | 0.054 |
| Goldberg | +0.032 | 0.67 | 0.42 | 0.090 |

Same-trait markers cluster in W_E in **all 24 model×set cells** (within > across
everywhere), and a 500-draw label-permutation null confirms it is real and
significant (w−a several SDs above null; NN-purity 0.46–0.78 vs chance ~0.19;
p<0.01 even for Qwen7, the weakest). The meandiff trait directions come out
near-orthogonal, like human Big Five axes. **So the prediction is directionally
confirmed:** the trait geometry *is* present in the static token embeddings —
the lexical seed is there, before any composition. But it is **coarse** —
NN-purity ~0.66, 5-cluster purity ~0.45, far from the clean ~1.0 you'd want, and
far below the depth-composed geometry (r≈0.65 to the human facet matrix). (Phi4
is the outlier: its W_E is so anisotropic that everything sits at cosine ~0.81,
leaving the trait signal a thin +0.02 sliver — consistent with Phi4's other
peculiarities. Model size does not monotonically help: Qwen32 has the *weakest*
clustering.)

**Reconciliation.** All three results fit together: the facet geometry is item
semantics (§3.9); a *seed* of the trait structure is already in the per-token
embeddings (§3.10 markers, above chance); but turning a multi-token *item* into
the sharp, human-matching facet geometry requires the transformer stack
(§3.10 sweep, peaks mid–late). rgb's logit-lens intuition was right in origin —
the structure starts lexically — and the earlier negative layer-0-on-items
result was indeed a composition artifact (single tokens show the structure that
pooled sentences hid). The honest one-liner: **weak lexical seed in W_E,
substantially amplified by depth.**

**Matched W_E-vs-mid test, and a massive-activation trap
(`scripts/marker_layer_compare.py`).** To pin "depth amplifies" on matched
stimuli, we ran the *same* single-token markers through each model and computed
the clustering metrics at W_E vs the 2/3-depth layer (marker run bare,
BOS+token, reading the marker position — no chat template). This surfaced a trap
worth recording. **Gemma-class models carry massive activations**: Gemma-3-12B's
mid-layer hidden states have one dimension (2339) with mean magnitude ~116,679
against a median of ~45 — and 4 dims over 1,000. This single near-constant
offset saturates raw cosine to ~1.0 between *every* pair — items and single
tokens alike (raw item within 1.000 / across 0.999). The cure is **subtraction,
not projection**: meandiff (fwd−rev) cancels the shared offset exactly, which is
why item `meandiff-itempc1` was always immune; but PC-projection alone does
*not* remove a near-constant dimension (low variance → not in the top PCs, yet it
dominates the norm). Mean-centering the vector set before the PC step reproduces
the canonical Gemma geometry (centered + top-1 PC: within 0.132 / across −0.008,
matching `meandiff-itempc1`'s 0.178 / 0.024). A first pass of the matched test
that projected without centering produced a spurious "bare tokens collapse at
depth" result for the massive-activation models — corrected here.

With centering applied, the matched comparison is clean and confirms the
depth-amplification claim on identical stimuli (cohort, Saucier):

| location (centered) | within | across | w−a | NN-purity | 5cl-purity |
|---|---|---|---|---|---|
| W_E, self-PC1 | −0.015 | −0.042 | +0.027 | 0.49 | 0.54 |
| mid, self-PC1 | 0.073 | −0.016 | +0.089 | 0.58 | 0.54 |
| mid, **neutral-PC1** | 0.406 | 0.095 | **+0.310** | **0.64** | 0.54 |

Same method top-to-bottom (center + self-PC1), the trait geometry sharpens with
depth (W_E w−a 0.027 → mid 0.089). And **neutral-PC1 beats self-PC1 by a wide
margin** (mid w−a 0.310 vs 0.089): after centering, the markers' *own* top PC is
partly trait variance, so removing it over-corrects, whereas the independent
neutral anisotropy axis removes only the shared junk. This is the W9 `single-pcs`
lesson reproduced at the single-token scale, and one neutral PC does nearly all
the work (nPC1 ≈ nPCs50). Gemma-3 recovers fully under centering (mid w−a 0.134 /
NN 0.70).

**Gemma-4-31B is the diagnostic exception — and it vindicates the neutral axis.**
Under self-PC1 it looks like a weak outlier (mid w−a 0.007), but that is a method
artifact, not a model defect. Gemma-4 has **no massive activation** (top mid dim
|val| 139 vs Gemma-3's ~116,000; zero dims >300) and its anisotropy is
*distributed* — the neutral mid PC1 explains only **10%** of variance vs Gemma-3's
**80%**. With no dominant axis, the markers' *own* top PC isn't the anisotropy
direction (it's trait or noise), so self-PC1 both misses the junk and eats
signal. The 80%-on-one-axis Gemma-3 models get away with self-PC1 because it
accidentally grabs the right axis; Gemma-4 doesn't. Switch to the *neutral* axis
and Gemma-4 falls back in range (Saucier mid_nPC1 w−a 0.092 / NN 0.53). And on
items — our actual instrument — Gemma-4 is among the *better* models (meandiff
r=0.616 to human, NN 0.57, purity 0.60, layer-sweep peak 0.706), so the newest
Gemma is healthy and, by shedding the massive-activation pathology, arguably
*cleaner* for representation work than Gemma-3. (Residual: Gemma-4's noisier
Goldberg markers stay weak even under neutral-PC1, 0.012 — distributed anisotropy
wanting more than one PC for that set.)

**Practical takeaway (for any future neutral/unipolar extraction):
center + neutral-PC1 is a reasonable first try** — it de-anisotropizes
massive-activation and rogue-dimension models alike (subtraction handles the
constant offsets, the neutral axis handles the shared direction without eating
trait signal), and it reproduces single-token trait geometry from the middle
layers where naive single-pole extraction (W9 single-zero/-neutral) was
degenerate. Caveat: where anisotropy is *distributed* rather than rank-1
(Gemma-4: PC1 only 10% of variance), one PC under-cleans and meandiff/more PCs
may be needed — the single-token bare-stimulus probe stresses this, but item
meandiff is unaffected.

### 3.11 Single-adjective track: the affect-merge is a separable lexical axis

§3.9 established the facet geometry is item semantics shared across text models;
§3.10 found a weak lexical seed in W_E. This section runs the **adjective-vs-item
adjudication** C&C named and §3.9 flagged unrun, on a *theory-neutral* human
substrate, and lands a mechanism: the affect-merge is a separable "affect-presence"
axis present in encoders and LLMs alike.

**Substrate (`scripts/fetch_external_data.py`, `adjective_corr_cluster.py`).** The
Johnson IPIP-NEO human matrix is Big-Five-scaffolded (30 facets pre-assigned to 5
factors), so it cannot show an affect axis cutting *across* the Big Five. Saucier's
**525-PDA** (Harvard Dataverse, `doi:10.7910/DVN/GHYMEV`) is raw 1–7 self-ratings
of 525 single adjectives (N=700), no a-priori factor structure — re-clusterable
from scratch, and in the same single-adjective units the model/encoder geometry
uses. **Not ipsatized** (unlike C&C's S&G-1996), so we control standardization.
Two columns are corrupted (data contradicts label) and dropped → **523 adjectives**:
`Inspirational` behaves like "Inconsiderate" (r +0.53), `Insensitive` like a
surgency word (+0.25 with Sensitive). Found via `adjective_audit.py` (encoder as
semantic oracle: a column whose empirical neighbors are semantic opposites of its
label is suspect, + antonym-pair check). The audit over-flags rare negations /
bare affect, so flags need per-column adjudication; ~0.4% corruption, not provably
exhaustive. (cf. the Johnson `.por` pre-reversed-key lesson — published ≠ clean.)

**Human structure is valence-organized.** Factor-then-nearest-factor (PCA+varimax,
`adjective_factor_heatmap.py`) recovers a clean Big Five even on RAW data (the
general evaluative axis is the big first PC; varimax distributes it). Hierarchical
clustering (`adjective_hclust_heatmap.py`) instead bifurcates the space into two
giant valence blobs (288 positive / 206 negative, anti-correlated) — valence is the
*primary* cut. Either way Cheerful/Sympathy sit in warmth (F1), Angry/Sad in
neg-affect (F3), and F1⊥F3 oppose: human Cheerful-Angry −0.36, Cheerful-Sad −0.43.

**LLM extraction (`extract_adjectives.py`).** 12 models × 4 framings × 523
adjectives, all layers. Carrier ending in the adjective, read the **adjective span
only** (`split_prefix`) so the rep carries personality context without carrier
dilution; framings hold read-span fixed and vary context: `self` "I am {adj}",
`pers` "My personality is {adj}", `desc` "Someone who is {adj}", `bare` "{adj}"
(floor; whole-prompt read). Gotcha logged: **fp16 overflows Gemma-3's massive
activations (~1e5 > 65504) to inf** — store fp32. Single adjectives have no fwd/rev
contrast, so de-anisotropization is **center on the 523-adjective mean + top-1 PC**
(centering does the work; raw cosines sit at a +0.99 anisotropy floor).

**Framing matters, and "this is personality" helps (rgb's prior, confirmed).**
Cohort-mean matrix-r to human: **pers +0.40 > self/desc +0.33 > bare +0.28**.
Explicit personality framing recovers best; bare word worst. Framing-r ≈ 0.75
(moderate — the three context frames broadly agree). `pers` is canonical going
forward.

**The affect-merge holds at adjective resolution, cohort-wide.** *No* model
reproduces the human valence opposition: human Cheerful-Sad −0.43 / Happy-Angry
−0.42 / Cheerful-Angry −0.36; all 12 models land 0 to +0.46 (Qwen/Llama most merged
+0.2–0.46, Gemmas least, near 0). So the merge is **not** an item-format artifact —
it survives the items→adjectives change, comparable in magnitude to the facet-level
+0.18. (Correction to a mid-analysis read: an early "adjectives un-merge" claim came
from Gemma12, which turns out to be the *least*-merged model in the cohort.)
Sympathy is under-tied to warmth too (human Sympathy-Kind +0.57, model +0.20),
echoing the facet-level detachment of Sympathy from the prosocial cluster.

**The merge is a separable affect-presence axis — valence is masked, not absent
(`affect_analysis.py`).** Projecting out the **affect centroid** (unit-mean of
pos+neg emotion adjectives — the common "is this emotional" component) recovers the
human opposition: cohort Cheerful-Angry +0.08 → **−0.22**, Cheerful-Sad +0.17 →
−0.14, affect-block r +0.589 → +0.676 (full-matrix r barely moves, −0.02). So the
LLM *has* the behavioral valence structure; a dominant affect-presence axis masks it
in the raw cosines. This is the "associative feature dominating the residual stream"
prediction of the symbolic-vs-associative frame — "emotional-or-not" is an
associative magnet outweighing the behavioral valence sign.

**Whose axis is it? Lexical — encoders merge too.** Same analysis on bge+mpnet
adjective geometry (centered cosine, raw per §3.9): the **encoder also merges**
(Cheerful-Angry −0.01, Cheerful-Sad +0.10 — not the human −0.36/−0.43) and **also
recovers under the same affect-center projection, even better** (Cheerful-Angry →
−0.38 ≈ human −0.36; affect-block r +0.720 → +0.793). A contrastive encoder with no
RLHF and no causal attention carries the same affect-presence axis and the same
recoverable valence underneath. So the affect axis is a **lexical** feature both
text-model types consume — the §3.9 thesis pinned to a specific separable direction,
and the mechanism behind C&C's "Neuroticism only weakly recovered" N break-point:
affect words cluster by emotional presence in text, humans by behavioral valence.
The LLM is *slightly* more merged than the encoder (Cheerful-Angry +0.08 vs −0.01) —
a thin LLM-specific amplification on the shared lexical axis, echoing the facet-level
LLM>encoder affect residual. Encoders also out-recover LLMs overall (full r +0.59 vs
+0.48), consistent with §3.9. 4-panel figure: `results/adjectives/affect_human_vs_model.png`.

**Caveat on the projection.** The affect centroid is defined from a hand-picked
emotion-word set, *and* projection is the wrong tool: it is cardinality-biased
(an imbalanced 6-pos/12-neg centroid leans negative and destroys the negative
block's coherence, wNeg 0.26→0.01) and it can only *lower* cosines, so it cannot
manufacture the human's high within-block coherence (best balanced projection
reaches cross −0.18 / within 0.11 vs human cross −0.28 / within 0.47). Removing
the affect-subspace PC1 makes the merge *worse* (PC1 is ≈valence, sign-consistency
0.11, not presence) — so presence ≈ the balanced centroid, valence ≈ the contrast.

**Denoise is regime-dependent — the Gemma-3 catch (`scripts/adjective_geom.py`).**
Single adjectives have no fwd/rev contrast, so the model matrix needs de-
anisotropization — but *how much* is model-dependent, and getting it wrong
manufactures artifacts. Centering (subtract the 523-adjective mean) handles the
anisotropy/norm offset for most models, and the top PC of the *centered* data is
then a **real** axis of variation (e.g. a settled↔assertive contrast) that must
be **kept** — not the facet pipeline's raw-anisotropy PC. Removing it (an earlier
mistake) deletes signal. **Exception: the Gemma-3 family.** Its massive activations
(~1e5 in 1–2 dims, §3.10) *survive centering* and dominate the variance; centered-
PC1 *is* that rogue dim (inverse participation ratio = 1–2, vs 28–1175 for every
other model — a clean split). There the top PC must be **removed**. So the
principled rule is **adaptive**: center, then drop top-1 PC iff PC1's IPR < 10
(rogue) — auto-routing Gemma-3 → remove, the other 9 → keep. (Confusing the two
inflates Gemma-3's affect cosines and *deflates* the others' real structure.)

**Principled version — RSA decomposition (`scripts/affect_rsa.py`).** Drop the
projection; *decompose* the affect-block similarities (12 pos + 12 neg, balanced)
onto two template RDMs: presence (`+1` every pair) and valence (`+1` same-pole,
`−1` cross). Fit `sim ≈ a·presence + b·valence`: **`a` = block elevation (the
uniform "all emotional" merge), `b` = pos/neg opposition, R² = valence's share of
the block variance.** Composition-robust, no knob (numbers under adaptive denoise):

| | presence a | valence b | R² |
|---|---|---|---|
| **Human 525-PDA** | +0.079 | **+0.368** | **0.93** |
| **LLM cohort mean** | +0.124 | +0.133 | 0.44 |
| bge-large / mpnet | +0.151 / +0.198 | +0.176 / +0.133 | 0.58 / 0.39 |
| gemma-3-12b | +0.135 | +0.184 | 0.65 |
| Qwen32 | +0.137 | +0.161 | 0.56 |

Every text model sits the block *higher* (a 0.12 > human 0.08 — more uniformly
merged) *and* carries ~2.8× *weaker* valence (b 0.13 vs 0.37), with much of its
block variance neither presence nor valence (R² 0.44 vs human 0.93). So "valence
masked, not absent" → valence is **present** (b>0 for all 12) but **weak and
underdeveloped**, presence ≈ valence rather than human's valence ≫ presence.
**Encoders sit in the same model range** — the presence-dominance is lexical,
shared across text models, none reaching human valence-dominance. Figure:
`affect_presence_vs_valence.png` (humans alone up-left). *Corrections from the
first pass (uniform center+top1):* the "Qwen32 worst / strong family gradient" and
"model space is flat" were **denoise artifacts** — under adaptive denoise Qwen32 is
b=0.16/R²=0.56 (mid-pack) and the cohort is fairly uniform; retracted.

**Are the human Big Five even the LLM's axes? (`adjective_factor_congruence.py`,
`factor_rotation_compare.py`).** Tucker congruence of each model's 5-factor varimax
loadings to the human Big Five (the C&C metric — they ran it on an *encoder*, never
decoder LLMs). Cohort-mean |congruence|: **A 0.59, E 0.44, N 0.58, C 0.35,
O 0.08** (mean 0.41). A/E/N recover moderately; C weakly; **Openness is essentially
absent, and that O-collapse is the most denoise-stable result (0.11→0.08).** The
model's *own* varimax factors are interpersonal/affective — warmth, evaluation
(Wonderful/Amazing/Great), distress, antagonism (Rude/Obnoxious/Arrogant) — with
**no Conscientiousness or Openness factor**:

| LLM factor | top adjectives | reading | closest Big Five |
|---|---|---|---|
| M1 | Compassionate, Supportive, Considerate, Caring, Respectful | warmth | A (.64) |
| M2 | Wonderful, Amazing, Impressive, Awesome, Great | evaluation / admiration | E (.44) |
| M3 | Self-assured, Good-natured, Well-Liked, Self-sufficient | confidence / likability | — (≤.16) |
| M4 | Disappointed, Ashamed, Upset, Afraid, Worried | distress / anxiety | (neg-affect) |
| M5 | Annoying, Irritating, Rude, Obnoxious, Arrogant | antagonism | N (.70) |

(cohort-mean varimax; var-frac 0.073/0.061/0.037/0.031/0.027 vs human
0.159/0.074/0.040/0.030/0.024.) So the LLM organizes trait-adjectives
by affect/evaluation/interpersonal content, not the full Big Five; A/E/N are there,
C/O are not. (Caveat: absolute congruences aren't comparable to C&C — different
human reference/carrier/matching — and the encoders also score ~0.45, so part is
that the 525-PDA Big Five is self-rating structure carrying a response-style
evaluative axis no text model has.) Figure `factor_rotation_compare.png`: human
varimax = clean Big-Five diagonal blocks; LLM varimax recovers A/E/N blocks but the
C/O columns wash out.

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
- **Logit-lens / token-embedding test (§3.10) — done.** Facet geometry is
  near-floor at layer 0 for multi-token items (r=0.11), built up with depth
  (peak 0.65, deeper for bigger models) — but that's sentence composition. The
  pure single-token-marker W_E test (Saucier + Goldberg) shows the trait seed
  IS in the static embeddings (within>across in all 24 cells, p<0.01 vs a
  permutation null) but coarse (NN-purity ~0.66). Matched W_E-vs-mid test
  confirms depth sharpens it (cohort w−a 0.027→0.089 same method). **Massive-
  activation trap found (h/t rgb):** Gemma dim 2339 ≈116k vs median 45 saturates
  raw cosine to ~1.0 for items and tokens alike — cure is subtraction/centering
  (meandiff is immune), not PC-projection. After centering, **center +
  neutral-PC1** is the best single-token extractor (cohort w−a 0.310, NN 0.64)
  and the recommended first-try neutral/unipolar de-anisotropizer. Verdict: weak
  lexical seed, amplified by depth; subtract before you project.
  `scripts/marker_embedding_geometry.py`, `scripts/facet_geometry_layer_sweep.py`,
  `scripts/marker_layer_compare.py`.
