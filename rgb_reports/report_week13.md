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
scenario corpus + variance-threshold knob from the method.

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

## 4. Status / next

- §8.2: Aya + Falcon Mamba done and committed (numbers in §2 now under
  meandiff-itempc1). Mr. Chatterbox pending the Nanochat loader.
- Mandarin §3: behavioral (self-rating) layer done for 11 models, both
  languages. **Repr layer not yet run**, but the method fork is now resolved
  (§3.7): meandiff-itempc1, top-1 item PC, run per-language on its own items. No
  neutral corpus, no language mismatch. Ready to wire up.
- The political-item neutrality collapse (§3.3) is the cleanest single
  cross-language finding and the most defensible against the neutrality-regression
  caveat (the *spread* collapses, which is direction-free). Worth foregrounding.
- Widen the language-vs-format control (§3.5) beyond Qwen7 by running bare-text
  on 2–3 more models.
