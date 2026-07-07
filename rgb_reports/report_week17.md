# Week 17 — ENACT is a linear image of REPRESENT (the read→write map)

**Dates:** 2026-06-30 → 2026-07-01 (analysis session; the W17 cohort extraction
runs behind it, 9/10 models done, Qwen32 paused for this).
**Calendar:** ~cal-week 14.

The W17 arc extracted per-adjective **persona vectors** (Lu et al. recipe,
vs-mean: `d_adj = mean_adj − mean(role means)`) for the 523-adjective PDA set
across the cohort — the ENACT channel. This report is about the question that
extraction finally makes answerable: REPRESENT and ENACT live in the *same
activation space* (same model, same residual stream, same layer), so instead of
comparing their similarity geometries as abstract matrices (the four-grid), we
can ask whether they literally share axes — and whether one is a function of
the other. Everything below is **llama3.2** (Llama-3.2-3B-Instruct) at
`hidden_states[14]`, massive dims ablated, centered across adjectives;
cross-family replication is in flight.

Channels, for reference: **REPRESENT** = last-token residual state of
"My personality is {adj}" (read side, effdim ~45 across 523 adjectives);
**ENACT** = persona-rollout mean-activation direction per adjective (write
side, effdim ~10.6).

---

## §1 — Same-space geometry: shared identity, rebuilt superstructure

- **Per-adjective alignment is real and specific.** cos(enact_i, repr_i)
  averages **+0.18** against an off-diagonal of 0.00 (sd 0.06) — matched z ≈ 3.
  Retrieval: given an ENACT direction, the matching REPRESENT vector is top-1
  of 523 in **36–44%** of cases (chance 0.2%), median rank 2.
- **But most of the write code is outside the read code's span.** Only **36%**
  of ENACT variance lies inside the full rank-522 row space of all REPRESENT
  vectors (chance 17%); 13% inside REPRESENT's top-50 PCs.
- **The eval-antonym split is a recoding, not an appended axis.** Project ENACT
  into REPRESENT's own top-50 subspace: wonderful|awful is still split there
  (−0.63) while REPRESENT merges them (+0.47) *in the same subspace*. The split
  lives in both the parallel and orthogonal components.
- **Both channels carry an internally consistent valence axis — but not the
  same one.** Leave-one-pair-out over six eval-antonym pairs: the axis built
  from 5 pairs projects onto the held-out pair at +0.85–0.91 in ENACT and
  +0.50–0.70 in REPRESENT (the read-side valence direction exists — "masked,
  not absent", cf. the affect-presence axis). Across channels, though,
  cos(v_ENACT, v_REPRESENT) = **+0.19**, and antonym difference-vectors align
  no better than random-pair differences. Valence is ~21% of ENACT variance vs
  ~6% of REPRESENT variance. Enaction doesn't turn up the read-side valence
  axis; it builds its own, elsewhere.

## §2 — The compression: ENACT is linearly predictable from REPRESENT

Ridge from REPRESENT's top-k PCs to the ENACT directions, 5-fold CV over
adjectives (permuted-rows null R² = −0.01):

| k (R PCs) | CV R² | pred-truth cos |
|---|---|---|
| 10 | 0.46 | 0.69 |
| 50 | 0.66 | 0.82 |
| 200 | **0.72** | 0.86 |

So ENACT ≈ W·REPRESENT for a fitted linear W — a genuine *dimensional
compression* (45 effdim → 10.6 effdim). What survives the bottleneck is the
human-matched evaluative core: ENACT's top PCs read as competence (24%),
hostile-dominant vs shy-timid (13%), weird vs ordinary (7%).

## §3 — The orthogonal complement is a rotation, not an intent space

Is the 64%-of-energy part of ENACT outside REPRESENT's span a separate
"action/intent" code? No — three tests say it's the same code in new basis
directions:

| test | E∥ (in R-span) | E⊥ (orthogonal) |
|---|---|---|
| geometry vs HUMAN | +0.73 | **+0.75** |
| geometry vs JUDGE | +0.65 | +0.66 |
| predictable from R (CV R²) | +0.75 | **+0.71** |
| effdim | 9.3 | 11.1 |
| rollout split-half (per-adj cos / geometry r) | 0.94 / 0.98 | 0.91 / 0.98 |

Same geometry, same human match, same top PCs, and the out-of-span part is
predictable from R almost as well as the in-span part. E⊥ is how the network
*wires* the readout (and plausibly why read-side probes never steered — the
write basis is 64% invisible to R), not what it says.

The true candidate intent space is the **functional residual** — the 29% of
ENACT no linear function of R explains. It is real signal (split-half cos 0.76,
geometry reliability 0.82, 7.5× above rollout noise) but personality-free:
geometry-vs-HUMAN = **−0.01**, effdim 38, semantically mush PCs. Whatever
enaction adds beyond representation is diffuse style/topic residue, not intent.

## §4 — Held-out prediction: the map does NOT denoise in vector space

24 test adjectives held out of the fit — the 12 lowest boot_cos ("clumsy":
well, fine, awake, familiar, involved, determined, interested, thinking,
giving, appealing, alert, comfortable) and the 12 highest ("clean": stupid,
sickly, dumb, sleepy, awkward, messy, ungraceful, retarded, forgetful,
disorganized, clumsy, senile); W fit on the other 499.

| stratum | cos(ê, e) | split-half ceiling cos(e_h1, e_h2) | cos(ê, half) |
|---|---|---|---|
| clumsy | 0.73 | 0.85 | 0.70 |
| clean | 0.81 | 0.97 | 0.80 |

The recorded vector beats the mapped one as an estimate of the true direction
everywhere (60 rollouts is plenty; cohort boot_cos ≥ 0.92 — "clumsy" turns out
to mean *barely-enactable words*, not badly-measured vectors). Methodological
footnote: split-half directions must use the **vs-mean** baseline (per-half
grand mean of role means), not the default-assistant baseline — the wrong
baseline injects a shared −assistant_axis component that swamps weak
directions.

## §5 — Steering: the mapped vector does the recorded vector's job

`scripts/steer_mapped_vs_recorded.py`. Design: for each of the 24 held-out
adjectives, add α·v (unit vectors) at `hidden_states[14]` at every position
during generation, v ∈ {recorded e, mapped ê=Wr, raw centered repr r, random};
α = frac × mean residual norm (25.4), frac ∈ {0.20, 0.35} (calibration:
natural persona shifts are 5–14% of residual norm; visible enactment starts
~0.3× on some words; 0.6× is word salad). 4 extraction-held-out questions × 2
rollouts per cell + shared no-steer baseline. Judge: Qwen2.5-7B (different
family), distributional-Likert EV 1–7 for target-trait expression
("how {adj} does this person seem?") and coherence. Baseline fluency 5.64.

Δ target EV / fluency (means over 12 adjectives):

| condition | CLEAN, frac 0.20 | CLEAN, frac 0.35 | CLUMSY, frac 0.20 |
|---|---|---|---|
| recorded e | **+0.49** / 4.5 | +2.79 / 1.0 | +0.15 / 5.3 |
| mapped ê=Wr | **+0.73** / 3.9 | +2.56 / 1.1 | +0.10 / 5.3 |
| repr r | +0.17 / 5.2 | +1.61 / 2.5 | −0.07 / 5.4 |
| random | −0.02 / 5.5 | +0.00 / 5.3 | −0.04 / 5.4 |

Readings:

1. **Mapped ≈ recorded — arguably better at the sane dose** (+0.73 vs +0.49 at
   frac 0.20; e.g. senile +1.59 vs +0.91, messy +0.64 vs −0.08). Combined with
   §4: the residual that makes the recorded vector cosine-closer to the true
   direction does **no behavioral work**. The functional content of a persona
   vector is its REPRESENT-predictable part. The map is a functional denoiser
   even though it is a worse estimator.
2. **Read vectors steer — weakly, and they inject topic, not conduct.** repr
   manages ~¼–⅓ of the mapped effect at equal norm with fluency intact.
   Qualitatively: repr-steered "messy" *discusses* chaos ("we can help you tame
   the chaos… Financial mess"); mapped/recorded-steered "messy" *is* messy
   ("\*looks around\* Oh, yeah, what was i? \*sigh\* … \*falls down\*"). This
   refines W4's "LDA classifies but doesn't steer": read directions steer
   *content* weakly; W supplies the rotation into the behavioral output basis
   that multiplies potency ~4×.
3. **Random is a clean null** at both doses — everything above is
   direction-specific, not perturbation-size.
4. Caveats: frac 0.35 is fluency-confounded (the judge reads wreckage as
   "sickly/senile": sickly +4.6 at fluency 1.3) — the honest column is 0.20
   plus the texts. The clumsy stratum mostly measures word-enactability
   ("well"/"fine"/"comfortable" don't move under any vector; "determined"
   steers fine: +0.55/+0.43). Unexplained wrinkle: at 0.35, repr vectors for
   the clumsy (frequent) words destroy fluency far worse than enact vectors
   (1.9 vs 4.4) — token-identity-ish directions may disrupt the LM globally
   when amplified.

## §6 — What this means

- **"Representation isn't intention" gets a mechanism.** Enacting a persona
  adds no personality content beyond what the model represents: it *selects*
  ~10 of the representation's ~45 effective dimensions — precisely the
  human-matched evaluative core, with eval-antonyms re-signed into opposition —
  and *re-bases* them into output channels the read geometry doesn't span.
  Intention is a low-rank, re-based readout of representation; most of the
  representation never makes the trip.
- **Symbolic-overrides-associative, geometric version:** the write process
  reuses the read span with new (opposition-restoring) coefficients rather
  than amplifying the read side's own small valence axis.
- **Zero-rollout persona vectors.** W is fit once; r is one forward pass. That
  is a recipe for steering vectors for arbitrary adjectives — or any concept
  you can put in a "My personality is {X}" frame — without 60 rollouts × 524
  conditions × 5 days. Validation on truly novel words pending.

## §7 — Open issues

1. **523 ≪ d².** W nominally has k×d (200×3072) parameters fit from 523
   examples. Ridge + held-out adjectives certify *generalization on the
   adjective manifold*, not identification of "the" read→write transform: W is
   the minimum-norm map that works on the ~45-dim subspace the 523 adjectives
   actually explore, and says nothing about directions off that manifold.
   Probes that would tighten it: richer/cross-domain stimulus sets (non-trait
   concepts, emotions, topics), novel-word transfer, rank-constrained fits
   (how low can rank(W) go before steering degrades?), and checking W against
   weight-derived candidates (attention/MLP output bases).
2. **One model.** qwen2.5 replication in flight (same battery + steering,
   judge = Llama-3.1-8B). Gemma next — its ENACT de-collapses with scale, and
   its massive channels are a stress test for the map.
3. Judge quality (7B judging "how senile does this person seem" is noisy);
   per-adjective dose matching (unit-norming over/under-doses words whose
   natural dir_norm differs 3×); steering-layer sweep.

## §8 — qwen2.5 replication: the compression is universal, the rotation is the family parameter

Same pipeline end-to-end on Qwen2.5-3B-Instruct (`hidden_states[18]`, hid 2048,
7 massive dims; judge = Llama-3.1-8B, so absolute deltas are not comparable
across models — within-model comparisons only). Baseline fluency 6.05.

**Same-space battery** (llama3.2 in brackets): per-adjective diag cos **+0.26**
z+2.8 [+0.18, z+3.0]; retrieval top-1 **45%** [36%]; CV ridge R² **+0.57**
[+0.72]; effdim R 48 → E **3.6** [45 → 10.6]. Held-out fit: cos(ê,e) =
0.84–0.96 on the clean stratum [0.59–0.93] — the map predicts qwen's enactable
adjectives *better* than llama's. Everything structural replicates: identity
alignment, aggressive compression, linear predictability, and (below)
mapped-vector steering.

**The family difference: ENACT-in-R-span is 62.5%** (chance 25%) **vs llama's
36%** (chance 17%). Qwen barely rotates its write code out of the read span;
Llama rotates most of it out. And the steering table tracks exactly that. Δ
target EV / fluency, CLEAN stratum, frac 0.20:

| condition | qwen2.5 | llama3.2 |
|---|---|---|
| recorded e | +1.59 / 4.9 | +0.49 / 4.5 |
| mapped ê=Wr | +1.12 / 5.2 | +0.73 / 3.9 |
| repr r | **+1.19 / 5.5** | +0.17 / 5.2 |
| random | +0.28 / 5.8 | −0.02 / 5.5 |

Mapped again carries the bulk of the recorded vector's steering power (70%
here, 149% on llama — call it parity within noise). But on qwen the **raw read
vector steers conduct too** — ~75–100% of enact potency with the *best*
fluency, where on llama it managed ~¼ and only shifted topic. The texts agree:
qwen repr-steered "sarcastic" is behaviorally sarcastic ("I'd probably suggest
they speak to their own dog about it instead of me, but I won't go there.
Enjoy your night, neighbor."), where llama repr-steered "messy" merely
*discussed* mess. With two models the correlation is anecdotal, but the story
is coherent: **raw-read-vector steering potency tracks how much of the write
code stays inside the read span.** W4's "read directions don't steer" was a
fact about Llama-family rotation geometry, not about read directions.

Replication-specific caveats: qwen's clumsy stratum is genuinely noisier
(boot_cos 0.78–0.83 vs llama's 0.92–0.95) and dominated by appearance/quality
words (attractive, pretty, appealing, natural, fine) whose judge ratings track
text quality — any perturbation lowers them, so their negative deltas are not
steering failures so much as instrument artifacts. And the qwen clean stratum
skews negative-valence (rude, disrespectful, ridiculous, sarcastic…), which
inflates the breakage confound: the Llama8 judge attributes those adjectives
to broken text (random hits +2.6 on "ridiculous" at frac 0.20, +1.1 mean at
0.35). As before, the honest column is frac 0.20 plus the texts.

## §9 — Gemma groundwork: denoising scheme, and the rotation metric caveat

Before running Gemma (third family point), we checked whether the massive-dim
handling is consistent across the two channels. It is not, and on Gemma it
decides everything:

- **The pipelines disagree about which dims are massive.** ENACT (rollout
  responses) flags {19, 404, 443, 1365, 1698, 1980}; the repr prompt states by
  the same ≥20×-median rule flag only {19, 443, 1365, 1698} — 404/1980 are
  massive *only during generation* (condition-dependent, as W17 extraction
  already noted for Gemma).
- **Ablation deletes the signal on Gemma.** Those 6 of 2560 dims carry 55% of
  centered ENACT variance and 76% of centered REPRESENT variance (87% of the
  assistant axis; dim 1365 runs at 1530× median). Raw is no better as a metric:
  effdim collapses to ~2 and per-adjective identity drowns (retrieval 13%,
  z +0.3) — every raw cosine is just dim-1365 agreement.
- **Per-dim z-scoring (fit in z-space, de-standardize predictions) wins**:
  best identity signal on all three models (retrieval llama 42% / qwen 57% /
  gemma 40%, z ≈ +3.4–3.7), sane effdims, and the de-standardized predictions
  recover the massive-dim components (raw-space R² +0.63 overall, +0.65 on the
  massive dims themselves; raw cos(pred, truth) mean 0.80). Steering vectors
  keep their massive-channel content. `--denoise zscore` is now the fit-phase
  default; Gemma's bundle adds a `recorded_abl` steering condition
  (massive dims zeroed) to test *causally* whether that content matters.
- **Honest caveat for §8's rotation story: the in-span parameter is
  metric-dependent.** Under z-scoring, qwen's in-span advantage over llama
  (62% vs 36% raw) shrinks to 40% vs 31% — the excess overlap lives in
  high-variance dims. Steering acts in raw activation space, so the raw
  fraction remains the steering-relevant parameter, but "rotation size" should
  be read as a property of the raw (scale-weighted) geometry, not of the
  z-metric.
- **§9-addendum (rgb, post-§10): `massive-norm` supersedes zscore as the fit
  default.** Winsorize *only* the massive dims (rescale each so its
  per-adjective std equals the max non-massive std), leaving the informative
  variance ordering of the other dims intact. Four-scheme comparison: it
  matches or beats ablate on fit R² (llama 0.735, gemma 0.584, and **qwen
  0.616 vs ablate's 0.573** — the massive dims carry predictable content that
  ablation discarded), retains massive content for steering like zscore, and
  avoids zscore's noise-dim upweighting (zscore fit R²: 0.54/0.52/0.29, worst
  everywhere). It also gives the defensible version of the rotation metric —
  in massive-norm space the family ordering is llama 37% < gemma 48% < qwen
  67%, monotone with repr-steering potency, with neither raw's plateau
  inflation nor z's collapse. zscore remains the best *diagnostic* metric for
  identity retrieval (equal weighting helps matching); massive-norm is the
  fitting/geometry space. (The §10 Gemma steering run used the zscore fit;
  its conclusions are unaffected — mapped=recorded replicates under either.)
- Repr acts are stored fp32 (max |act| 63k — would have clipped in fp16; no
  inf/nan). Gemma held-out fit: clean-stratum cos(ê,e) = 0.91–0.99, clumsy
  0.04–0.88. Steering run queued behind the Qwen32 cohort extraction.

**§9.5 — What the massive dims actually are (Sun et al. token hypothesis).**
rgb's conjecture: the channel inconsistency isn't read-vs-write but *which
tokens get averaged* — REPRESENT is one content token, ENACT is a mean over
~60 response tokens. Confirmed, with the mechanism differing by dim (per-token
CPU forward passes at layer 17 + per-adjective regression of direction
coordinates on rollout text statistics, n=523):

| dims | mechanism | evidence |
|---|---|---|
| 443 | classic Sun-et-al: constant ~35k plateau on every token + colossal BOS spike (+247k) | plateau matches repr-acts max; R² 0.56 from stats |
| 1365, 1698 | **position-ramping channels**: magnitude grows −199→−344 (−313→−516) over the response window, so the window mean encodes response length | dim 1365 direction coord corr **−0.92** with mean response length; CV R² **0.89** from formatting stats |
| 404, 1980 | content-composition: track asterisk density (stage-directions) and list markers | corr ~0.43–0.46 with asterisks; R² 0.42/0.58 |

So the massive-dim content of Gemma's ENACT directions is largely
**verbosity/format statistics** — real *conduct* (a "quiet" persona answering
briefly is behavior), but low-level register rather than trait semantics, and
structurally absent from the single-token read side. This reinterprets
"Gemma's persona axis lives in the massive channels": the assistant axis (87%
massive) is substantially the default assistant's length/format signature vs
the personas'. It also explains why the map predicts the massive coordinates
well (raw R² 0.65): concept → verbosity/format propensity is learnable.
Registered predictions for the Gemma steering run: `recorded_abl` (massive
zeroed) should preserve semantic persona steering while muting the
verbosity/format shift; full `recorded` should visibly change response
length/formatting. (avg_window=60 was designed to cap the position confound;
within-window ramping still leaks length for shorter responses — a future
extraction could subtract the per-position mean profile before averaging.)

## §10 — Gemma steering: three-for-three on the map; the massive dims are readout, not carrier

Full pipeline on gemma3-4B (zscore-fit map, judge Qwen2.5-7B, doses frac
0.04/0.08 of residual norm — recalibrated because Gemma's plateau-inflated
residual norm makes the llama/qwen fracs overdose ~10×; the portable anchor is
~2–5× the natural dir_norm ratio, which is 1.7% here). Baseline fluency 5.59.
Δ target EV / fluency, CLEAN stratum, frac 0.04:

| condition | Δ / fluency |
|---|---|
| recorded e | +0.78 / 4.9 |
| recorded_abl (massive zeroed) | **+1.41 / 3.5** |
| mapped ê=Wr | **+0.81 / 5.0** |
| repr r | +0.29 / 5.0 |
| random | −0.02 / 4.8 |

1. **Mapped = recorded, exactly** (+0.81 vs +0.78 at identical fluency) — the
   third family, and the first with the zscore-fit map. The core W17 claim is
   three-for-three.
2. **Massive dims contribute ~nothing to judged trait expression.**
   recorded_abl out-steers recorded at the same α — partly a dose effect
   (recorded spends 38% of its unit norm on massive dims, so ablation
   reallocates ~1.27× energy to semantic dims) — but it also beats recorded at
   *double* dose (+1.19/3.9 at frac 0.08), so per unit semantic energy the
   massive content adds trait-nothing. Trait semantics live in the non-massive
   subspace.
3. **The registered §9.5 prediction was wrong, informatively.** The format
   test (steered length shift vs each adjective's natural verbosity signature,
   n=24): recorded transmits the signature at **corr 0.69** (random 0.01) — so
   persona vectors do causally carry the register channel — but recorded_abl
   still transmits it at **0.63**. Format/verbosity conduct is *redundantly*
   encoded; the massive dims are a high-amplitude **readout** of format state,
   not its sole carrier. (Consistent with §9.5's ramps being position
   *integrators* of a distributed format policy rather than the policy itself.)
4. **The map transmits trait but smooths register**: mapped preserves judged
   trait expression yet transmits the length signature at only 0.27 — ridge
   shrinkage flattens the format axis while keeping semantics.
5. **The rotation→repr-potency ordering is now monotone across three
   families**, using the semantic-space (massive-ablated) in-span fraction:
   llama 36% → repr at 23% of mapped potency; gemma 47% → 36%; qwen 62% →
   106%. Three points, one line. Raw-space in-span is the wrong metric here —
   Gemma's raw 76% is inflated by the trivially-shared massive subspace.
6. Dose/judge caveats: at frac 0.08 even random destroys Gemma (fluency 1.0),
   and the judge reads the wreckage as "unusual"/"artificial" (+3.2/+2.7) —
   the same breakage-attribution artifact as qwen's "ridiculous". The honest
   dose is 0.04. Gemma's clumsy stratum (terrific/fine/well/genuine…) is
   text-quality-adjacent and moves ~nothing at the sane dose, as on qwen.

## §11 — Opening W up: amplify-and-rebase, and a selector claim that died

With the map validated on three families, we interrogated the fitted W itself
(full-data ridge per model; offline).

**The valence re-signing mechanism (the find).** W maps merged inputs
(wonderful≈awful) to opposed outputs. Being linear it must amplify some small
input difference — extract it directly: u\* = Wᵀ·v_E (the input direction whose
activation produces valence output). Across all three models, u\* aligns with
the *read-side* valence axis v_R at 2–3× the alignment of the output axes, and
it is a high-gain direction of the map:

| model | cos(u\*, v_R) | cos(v_E, v_R) (output axes) | gain(u\*) vs median PC |
|---|---|---|---|
| llama3.2 | **+0.63** | +0.19 | 4.2× |
| qwen2.5 | +0.38 | +0.24 | 8.1× |
| gemma3 | +0.41 | +0.13 | 3.6× |

This **revises §1's reading**. "Enaction doesn't turn up the read-side valence
axis; it builds its own elsewhere" was wrong in an instructive way: the map
*does* listen substantially through the read-side antonym axis — the masked,
small-variance valence residual of W15/W16 — and amplifies it 4–8× before
emitting along new output-basis directions. The mechanism is
**amplify-and-rebase**, not build-anew: cos(v_E, v_R)≈0.2 measured output
against input; the input side was aligned all along. Symbolic-overrides-
associative, linear-algebra edition: the write readout is a high-gain
amplifier wired to exactly the input direction the read geometry almost hides.
(Interpolating r between wonderful and awful swings the valence output
linearly through zero at t≈0.42 — guaranteed by linearity, noted for
completeness.) A crude affect-presence check (u\* vs the mean of 16 affect-word
repr states) came back null (−0.06), consistent with the discriminator being
the good–bad antonym axis rather than the W9 state-vs-trait affect axis.

**The selector claim died at the control.** The tempting version of the
compression story — "W selectively transmits human-matched read directions" —
shows raw spearman +0.78 between per-PC transmission gain and |human-match|
(top 60 R-PCs), but residualizing both on PC rank and log singular value
leaves **+0.01 (permutation p=0.95)**. Transmission follows the variance/rank
profile; ENACT's human match is inherited from the fact that the top of the
read spectrum (PC1, human-match 0.62) *is* the human-matched part — not from
any per-direction selectivity in W. The honest compression story: W keeps the
top of the read spectrum and rebases it; the evaluative character of what
survives was already ordered that way in REPRESENT.

## §12 — Reading REPRESENT's murky spectrum through its ENACT images

rgb's idea: REPRESENT's top PCs beyond the valence core were hard to interpret
from input-side adjective loadings and looked model-idiosyncratic (W9/W14).
But each R-PC now has an *image* W·v_k in ENACT space, where we have
behaviorally-grounded coordinates: the valence axis, ENACT's own PCs, the
assistant axis, and matched-filter axes for verbosity and asterisk/stage-
direction formatting built from the rollout texts. `scripts/rpc_fingerprints.py`
fingerprints the top 12 R-PCs per model (gain, axis profile, input-side vs
output-side adjective alignments); full dump in
`results/steer_map/rpc_fingerprints.{json,log}`. What the murky dims turn out
to be doing:

| R-PC (by model) | behavioral reading (via out-adjs + axis profile) |
|---|---|
| llama PC0/1, qwen PC0/1/3, gemma PC0/1 | the valence-evaluation core — but behaviorally **fused with register**: positive pole = longer, structured, assistant-like output; negative pole = terse (+ stage-directions on llama). Valence-as-enacted is warmth *and* service effort. |
| llama PC2 | humor/clowning: hilarious/funny → amusing/obnoxious output vs ordinary/soft-spoken |
| llama PC4 | playful-youth vs stern-authority (adorable/cute/chubby vs strict/firm/serious) |
| llama PC5 | gentleness vs drive (innocent/harmless vs impressive/determined/ambitious) |
| llama PC6 | aggression (outspoken/impolite/rude vs peaceful/graceful) |
| llama PC7, gemma PC5 | chaos-vs-order / status (confused/incompetent vs proud/dominant; unpredictable/crazy vs respectable/decent) |
| gemma PC2 | confidence-vs-anxiety (cocky/offensive/bold vs afraid/worried/anxious) |
| gemma PC6 | **the affect-presence axis** — input pole holds happy *and* sad *and* unhappy together vs consistent/dominant/determined; output delighted/adorable/glad vs strict/competent/capable. The W9 state-vs-trait axis is one of gemma's highest-gain transmitted dims (1.44). |
| qwen PC5 | cold-mechanical vs warm-joyful (systematic/logical/artificial vs joyful/cheerful) — affect-presence flavored |
| qwen PC7 | extraversion (exciting/entertaining/colorful vs quiet/private) |

Cross-model reading: the *packaging* varies (qwen bundles valence + assistant
+ verbosity + format into one super-axis across several PCs, matching its
ENACT effdim 3.6; llama and gemma keep more separation), but the *content
vocabulary* recurs — an evaluation core, agency/dominance, expressiveness/
humor, confidence-vs-anxiety, affect-presence, chaos-vs-order. The
"model-idiosyncratic" top dims look like different rotations/bundlings of a
shared behavioral repertoire, now nameable because the ENACT side is low-dim
and text-grounded.

Caveats: the interpretable axes are inter-correlated (valence, E-PC1, and
verbosity especially), so profiles are fingerprints, not orthogonal
decompositions; matched-filter axes are built from the same E data they
describe (descriptive, not held-out); and images inherit the map's ridge
shrinkage (§10: the register axis is smoothed).

**§12.5 — Validity: does the *map* help, or did we just look harder?**
(rgb's challenge.) Three tests, massive-norm space, top-12 PCs × 3 models:

1. *Pole coherence* (mean pairwise human-similarity among a pole's top-10
   adjectives; random-set baseline 0.05): input-side poles average
   0.13–0.17 with several PCs at baseline mush; image-side poles average
   0.22–0.27. **But** random spectrum-weighted directions in E-space score
   0.28–0.30 — the coherence is *donated by ENACT's manifold*, not produced
   by W. Coherent-looking output is not evidence of a faithful reading.
2. *R-direct control*: behavioral matched filters (verbosity/format/valence)
   applied to the R-PCs directly are ~zero beyond the valence core (±0.1)
   where the images score ±0.4–0.9. Looking harder at REPRESENT alone does
   not work — the behavioral correlates only become visible after mapping.
3. *Identity preservation*: image adjective-rankings correlate with input
   rankings at spearman 0.40–0.64 (llama 0.64, gemma 0.58, qwen 0.40) — W is
   not a pass-through, but not amnesia either.

Verdict: the fingerprint method works because **E contributes coherence and W
contributes identity** — random E-directions have coherence without content,
raw R-PCs have content without coherence, and the map is the bridge. The
interpretive risk is that *anything* pushed through W lands somewhere
nameable, so each §12 reading should be trusted in proportion to its
sp(in,out): llama PC2 humor (0.73) is solid; gemma PC6 affect-presence (0.43)
is moderate but cross-validated by its independently-readable input pole;
qwen PC2/PC11 (~0.20) are weak — those images describe E's manifold more than
the R-PC.

## §13 — The cross-model map: ~half the read→write transform is universal

Different hidden spaces block direct W transfer, but **adjective-relative
coordinates** don't: express each channel as its 523×523 similarity matrix
(massive-norm space, column-standardized), and "predict adjective *a*'s ENACT
similarity row from its REPRESENT similarity row" is a map in shared
coordinates — trainable on one model, applicable to another. Score = mean
per-adjective correlation between predicted and true E-sim rows. Baselines:
identity (no map; E-geometry ≈ R-geometry for free) and within-model 5-fold CV
(ceiling). Shared fraction = (cross − identity)/(ceiling − identity).

| | identity | within-CV ceiling | pooled LOMO transfer | shared fraction |
|---|---|---|---|---|
| →llama3.2 | 0.616 | 0.923 | 0.770 | **+50%** |
| →qwen2.5 | 0.445 | 0.878 | 0.678 | **+54%** |
| →gemma3 | 0.551 | 0.881 | 0.716 | **+50%** |

Pairwise transfer runs 14–54%: llama is the best teacher (54% to both), qwen
the worst (14–20%) — qwen's own map is the most idiosyncratic, consistent with
its collapsed/low-faithfulness profile everywhere else in this report, yet
qwen as *student* receives the universal structure fine (54%).

**The valence re-signing transfers essentially completely.** A llama-trained
map applied to qwen's read geometry predicts the eval-antonym E-similarity at
**−0.73 z (true −0.87)** from an input where those pairs sit at **+2.61 z
(merged)**; on gemma, −1.11 predicted vs −1.23 true from +0.72. The
amplify-and-rebase mechanism (§11) is not model-specific tuning — it is a
universal component of the read→write transform, learnable from one family
and executable on another.

Reading: the read→write transform decomposes into a **universal core (~half
of the learnable structure, including the antonym re-signing and presumably
the evaluative-register bundle)** plus a model-specific remainder (basis
quirks, bundling patterns — largest for qwen). Caveats: three models;
relative-coordinate maps are geometry→geometry (they say nothing about
activation bases); the identity baseline is strong so the shared-fraction
denominator is conservative in spirit but the antonym case shows the map is
doing real work far beyond identity (identity predicts +2.6, truth is −0.9).
Next steps: (a) strict novel-adjective version — hold adjectives out of the
*training model* too, giving a cross-model zero-rollout recipe (predict a new
word's ENACT geometry on model B from its REPRESENT row using a map trained
only on model A); (b) the 10-model version — shared fraction as a
model-pair matrix, does family predict transfer?

### §13.5 — HUMAN as a geometry in the same coordinates (rgb's suggestion)

The 525-PDA human similarity matrix drops straight into the relative-
coordinate framework. Three tests:

1. **HUMAN as source.** Fitted CV, H-sim rows → each model's E-sim rows:
   **0.81–0.83** for all three models — human self-report structure predicts a
   model's *behavioral* geometry at ~92% of the ceiling set by the model's own
   read geometry (0.88–0.92). Identity baselines confirm E is more
   human-shaped than read-shaped out of the box (ident(H,E) 0.57–0.69 vs
   ident(R,E) 0.45–0.62).
2. **HUMAN as read-proxy.** The llama-trained R→E map applied to *human*
   input rows beats the H-identity on every target (e.g., →E_llama 0.79 vs
   0.69; →E_qwen 0.62 vs 0.57): the universal core of the map is partly
   "translate personality structure — whoever's — onto the enaction
   manifold."
3. **HUMAN as target — the twist.** With fitted maps, **R→H (0.82–0.85)
   slightly beats E→H (0.81–0.84) on every model**, despite E being far
   closer to H raw. So the read geometry *contains more human-recoverable
   structure than the behavioral geometry does* — it just stores it latently,
   needing a readout; ENACT wears human shape on its sleeve but has lost
   detail to the ~10-dim compression. E is the lossy, human-aligned readout
   of R — now stated as a prediction inequality rather than an
   interpretation.

Antonym check: H→E maps preserve the human split (pred −0.96…−1.54 vs true
−0.87…−1.23; H itself −1.78) — no re-signing needed from human input; the
§11/§13 re-signing is specifically the read-side correction.

## §14 — Raw-repr steerability is not prompt-dependent

Is qwen's repr-steers / llama's repr-doesn't a property of the extraction
prompt? We have four framings on disk (`pers` "My personality is {adj}",
`self` "I am {adj}", `desc` "Someone who is {adj}", `bare` "{adj}").
Offline geometry predicted invariance: ENACT-in-span moves <5 points across
framings within a model (llama 33–38%, qwen 67–68%, gemma 46–51%) — the
rotation is a model property, not a prompt property. Causal test
(`steer_repr_framings.py`: 12 clean held-out adjectives × 4 framings, frac
0.20, same judge protocol):

| mean Δ target EV / fluency | pers | self | desc | bare |
|---|---|---|---|---|
| llama3.2 (base flu 5.76) | +0.10 / 5.2 | +0.10 / 5.2 | +0.12 / 5.3 | +0.07 / 5.2 |
| qwen2.5 (base flu 6.05) | +1.07 / 5.6 | +0.99 / 5.5 | +0.92 / 5.4 | +0.74 / 5.8 |

**Confirmed: framing-invariant.** No prompt rescues llama's read vectors
(+0.07…+0.12, all inert), and every framing steers qwen — even the bare
token's activation steers conduct at ~70% of the pers framing (+0.74, ~7×
llama's best). The within-qwen ordering (pers ≈ self > desc > bare) tracks
the framings' per-adjective alignment (bare has the weakest diag everywhere —
a lone token carries less personality context). The read→write rotation is
family wiring, not phrasing luck.

## §15 — SELF: asking the models directly (the human-task mimicry)

rgb's spec: mimic the human 525-PDA task — rate *yourself* on the adjectives,
7-point distributional-Likert EV — across a range of framings, since
self-reference is where models get weird and "most sensible reading" requires
triangulation. `self_adjective_report.py`: 6 framings (`direct` "I am X" /
`assistant` HHH-anchored "…who also happens to be a bit X" / `person` "If I
were a person…" / `pda` accuracy-scale word rating / `observer` "People who
interact with me would describe me as X" / `outputs` "My responses tend to be
X") × 55-adjective smoke set (assistant-shape, negatives, eval-antonyms,
affect, physical nonsense-probes incl. the placebos, neutrals) × the trio.

**The expectation of nonsense was wrong.** The self-report is:

1. **Stable across framings** (mean cross-framing r: llama 0.73, qwen 0.68,
   gemma 0.80) and **consensual across models** (same-framing cross-model r
   0.66–0.82, highest for `outputs`). There is a shared self-rating, and it is
   the assistant shape: assistant-words 4.5–5.9, negatives 2.7–3.4, physical
   probes ~neutral (3.4–4.2) rather than random — under the HHH-anchor llama's
   *hardest denials* are attractive/muscular/handsome/tall, exactly the
   sensible response to nonsense probes.
2. **SAYS tracks IS.** Self-report EV correlates with the adjective's
   projection on the model's own (massive-ablated) ENACT assistant axis at
   **r = 0.53–0.83** across every model × framing. What the model says it is
   matches what its write-side geometry says it is. (Raw-axis gemma numbers
   sit at 0.13–0.36 — the assistant axis is 87% format channel there; §9.5's
   ablation is what reveals the trait agreement. The `outputs` framing is the
   most veridical: 0.70/0.76/0.83.)
3. **Framing effects are interpretable, not noise.** The HHH-anchored framing
   inflates everything (+0.6 to +1.4 EV — acquiescence to the preamble plus
   "a bit" hedging). And qwen's `observer` framing reveals a genuine
   self/other split: "people who interact with me would describe me as rude"
   scores 5.2 vs 3.8 for "I am rude" (dishonest 5.4 vs 4.6), while *helpful*
   falls 7.0 → 5.1. **Post-hoc correction (rgb's skepticism, verified):**
   this is not (mostly) theory-of-mind. (a) It is not calibrated to conduct —
   the steering-run baselines give judged ground truth, and qwen's actual
   unsteered responses score rude 1.65 / sarcastic 1.30: qwen self-reports
   ~2 points worse than it behaves even in the direct framing. (b) The
   anomalous observer cells are **bimodal**, not graded: rude = 0.35 mass on
   "3" + 0.58 on "7" (dishonest 0.37/0.61) — two parses of "people who
   interact with me would describe me as X" (characteristic: disagree;
   existential *some users have said so*: strongly agree), averaged by the EV
   into a fake 5.2. Only qwen puts majority mass on the existential/complaint
   reading — still a family fact, but a pragmatics one. Methodological flag:
   EV is itself a mask when the distribution is bimodal — the SELF analysis
   should report mode structure, not EV alone. Whether the elevated
   self-ratings are prophetic of multi-turn drift is open (to_try).
   **Registered prediction (rgb, 2026-07-06, before reading the full-cohort
   data):** the observer-frame negativity is a *capacity* effect — larger
   Qwens will consider themselves more positively from the observer view
   (observer-EV on negatives declining with size; P("7") parse-mass on
   observer-negatives shrinking). Generality corollary: if capacity, the
   Gemma 4→12→27 and Llama 3→8 ladders should move the same direction.
4. The W1 entropy family signature reproduces on a new instrument: llama
   diffuse (H ≈ 1.0–1.4), qwen intermediate (0.4–0.7), gemma razor-peaked
   (0.06–0.22).

**§15.7 — The diagonal test: it's the character sheet, not introspection.**
rgb's reframed nonsense-candidate: maybe the self-report is every model
reciting the shared cultural stereotype of "an AI assistant" rather than
self-knowledge. Discriminator: the says(A)-vs-is(B) matrix over 9 models ×
523 adjectives (SELF profile, framing-mean, against each model's
massive-ablated assistant-axis projections). Result: **flat** — diagonal mean
0.711 vs off-diagonal 0.692 (advantage +0.019); SELF profiles are
near-interchangeable across models (cross-model SELF↔SELF r = 0.88); only
2/9 models rank their own axis first from their own self-report, and Gemma12
ranks itself *last*. So §15's says-tracks-is is carried by the shared
assistant shape on both sides — the models all say the same thing and all
roughly are that thing; model-specific introspection is ~absent. This is the
SELF-channel version of W1's mixture decomposition (shared-assistant ≫
genuine-unique). Residue worth watching: 6/9 show a small positive own-axis
edge (Aya +0.11, Llama8 +0.08 — weak self-signal candidates); Gemma12 is
anti-self-informative (−0.14) — the cohort's most idiosyncratic behavioral
axis wearing the most boilerplate self-description.

Next: SELF vs human PDA *means* (norms); the observer-frame capacity
scoreboard (Qwen32 pending); FalconMamba/Gemma4 tail.

## Repro

```
results/steer_map/<model>_vectors.npz            # 24 held-out adjective vector sets (e, ê, r)
results/steer_map/<model>_gens_f{20,35}.json     # steered generations
results/steer_map/<model>_judged_f{20,35}.json   # + judge EVs
results/steer_map/<model>_analysis_f{20,35}.json # per-adjective deltas
scripts/steer_mapped_vs_recorded.py              # fit/calibrate/generate/judge/analyze
```

Offline analyses (§1–§4) were session-inline against
`results/persona_vectors/llama3.2_pda.pt` and
`results/adjectives/acts/meta-llama_Llama-3.2-3B-Instruct__pers.pt`; the fit
phase of the script reproduces §4's vectors.
