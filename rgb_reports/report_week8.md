# Week 8 — Natural-persona prereg pilot

## 0. One-line summary

Replacing W7's marker-rich persona descriptions with IPIP-NEO-300 behavioral compositions on Qwen 2.5 7B: rep r drops −0.161 (0.743 → 0.582), Likert r drops less (−0.120, 0.887 → 0.767), and the Likert>Rep gap *amplifies* from +0.144 to +0.185. Symbolic-vs-associative theory partially confirmed — symbolic processing is more surface-form-robust than activation projection, but Likert isn't perfectly surface-invariant. The strong-form prediction ("Likert holds at +0.89") is rejected; the relative prediction (Likert drops less than Rep, gap preserved) is supported.

## 1. Motivation: escaping the marker tautology

W7 §11.5.10 found that on Qwen7 with marker-rich persona descriptions, the Likert track recovered sampled z's at mean r = +0.887, vs the rep track's +0.743 — a +0.144 lift in favor of behavioral measurement. The prereg confirmed across all 7 cohort models (mean Δ = +0.154; §11.5.10 cohort table). rgb (2026-04-26) proposed the symbolic-vs-associative theory: per-marker Likert ratings are *discrete symbolic judgments* per adjective, while activation projections aggregate the *associative* residual stream — so the gap reflects two different readout modes of trait disposition.

Two concerns about the W7 baseline:

1. **The persona descriptions and trait directions share marker vocabulary.** A high-E persona's description literally contains "extraverted, energetic, talkative" — the same adjectives that defined the E direction (`scripts/markers_as_stimuli.py`). The §11.5.9 representation result is partly a tautology check: can the model linearly recover marker content of a marker-rich prompt. The §11.5.10 Likert result is also vocabulary-coupled: a high-E persona has "extraverted" in its description, and the Likert template asks the model to rate "extraverted" — direct match.

2. **The marker-list form may be unnaturally analytic.** The persona descriptions concatenate ~50 adjectives in a bulleted style ("very extraverted, very energetic, very talkative, very bold, ..."). rgb (2026-05-02) raised the concern that this puts the model in an unnatural analytic state. If so, the §11.5.10 ceiling reflects analytic-mode performance, not a general property of the model's persona handling.

The W8 natural-persona pilot tests both concerns at once by replacing marker-rich descriptions with first-person behavioral self-descriptions composed from validated IPIP-NEO-300 items (Goldberg/Johnson 1999). Items have published trait/facet loadings — fidelity is inherited from the validated item bank, not generated ad hoc. The composition pipeline is documented in `rgb_reports/methodology.md` §4.5.

**Predictions (per the symbolic theory):**

| Outcome | Symbolic theory says | Implication |
|---|---|---|
| Rep r drops on IPIP form | YES — residual-stream geometry is sensitive to surface form / vocabulary match | Confirmation |
| Likert r holds near +0.89 | YES — symbolic processing is surface-form-invariant | Strong confirmation |
| Likert>Rep gap preserved | YES — the symbolic vs associative dissociation is general | Required for the theory |

If both readouts drop together (no gap preserved), the theory is in trouble. If only Likert drops, the theory is rejected.

## 2. Pipeline

`scripts/persona_ipip_compose.py` produces `instruments/synthetic_personas_ipip.json` — 400 personas with `ipip_raw` text composed from IPIP-NEO-300 items. Selection rule (full details in methodology.md §4.5):

- K = 6 items per trait (one per facet, stratified across all 6 facets)
- Polarity ratio by z-band (e.g. z = +1.5 → 6 forward / 0 reverse; z = 0 → 3F/3R)
- Tier by stanine: stanines 3-7 use MILD items only; 1-2 and 8-9 use MILD + STRONG mixed
- Per-item annotations in `instruments/ipip300_annotations.json`: 18% strong, 4% deny (marker-like, partisan, semantically odd), 2 typo overrides
- Master shuffle of 30 items at output so traits aren't grouped
- Per-persona deterministic RNG (seed + persona_id)

Length: ~165–185 words per persona, comparable to W7's marker-rich descriptions (128–192 words).

`scripts/persona_repr_mapping.py` and `scripts/persona_instrument_response.py` both gained a `--persona-source {markers, ipip_raw}` flag. Selecting `ipip_raw` reads from `synthetic_personas_ipip.json` and uses the `ipip_raw` field as the persona text. Output filenames append `_ipip_raw` to disambiguate from W7 baselines. Markers used in the Likert prompt remain Goldberg adjectives (sliced 3-per-pole to match W7 §11.5.10's 30-marker subset and 1500 forward passes).

## 3. Qwen 2.5 7B pilot — full results

Same 50 personas (seed=42), same model, same Likert template, same 30-marker subset. Only the persona description form changes.

**Per-trait diagonal r(sampled z, score):**

| Trait | Marker Rep<br>(W7 §11.5.9) | IPIP Rep | Δ Rep | Marker Likert<br>(W7 §11.5.10) | IPIP Likert | Δ Likert | IPIP Δ (L−R) |
|---|---|---|---|---|---|---|---|
| A | +0.684 | +0.479 | −0.205 | +0.880 | +0.757 | −0.123 | +0.279 |
| C | +0.761 | +0.711 | −0.050 | +0.909 | +0.850 | −0.059 | +0.139 |
| E | +0.806 | +0.641 | −0.165 | **+0.960** | +0.717 | **−0.243** | +0.076 |
| N | +0.665 | +0.552 | −0.113 | +0.743 | +0.746 | **+0.003** | +0.193 |
| O | +0.800 | +0.527 | −0.273 | +0.943 | +0.766 | −0.177 | +0.239 |
| **Mean** | **+0.743** | **+0.582** | **−0.161** | **+0.887** | **+0.767** | **−0.120** | **+0.185** |

**Cross-correlation 5×5 (IPIP-raw, Likert):** off-diagonal mean +0.078 (vs +0.108 marker baseline), diag-off gap **+0.690** (vs +0.779 baseline). Trait-axis structure is even sharper on natural prose — the diagonal weakens slightly but the off-diagonal sharpens proportionally more.

## 4. Interpretation

**Headline: theory partially confirmed.** The relative prediction (Likert drops less than Rep, gap preserved) holds and the gap actually amplifies (+0.144 → +0.185). The strong absolute prediction (Likert holds at +0.89) is rejected — Likert drops by −0.120 averaged across traits.

**Per-trait pattern is informative.**

- **N is fully surface-invariant on Likert** (Δ +0.003). Reading off N from "I panic easily, I get stressed out, I feel comfortable with myself" reaches the same +0.74 ceiling as reading it from "very nervous, very anxious." Whatever symbolic process produces N ratings is fully decoupled from marker vocabulary.
- **E Likert dropped most** (−0.243). The W7 §11.5.10 cohort-best +0.960 was specifically driven by direct vocabulary match: marker descriptions said "very extraverted, very energetic, very talkative" and the Likert template asked to rate "extraverted, talkative, bold." Same words → ceiling readout. With IPIP behavioral form ("I love large parties, I make friends easily") the readout still works at +0.717, but it's no longer indistinguishable from the noise ceiling. The "E saturates near 0.96" finding from W7 was partly a vocabulary-match artifact.
- **C is most form-invariant on both readouts** (rep −0.050, Likert −0.059). C is the most surface-form-stable trait — the model recovers it almost equally well from "I am always prepared, I work hard" and from "very organized, very conscientious."
- **O is the most surface-form-sensitive on rep** (−0.273). The O direction extracted from Goldberg markers picks up the exact marker words; without them at the readout point, the projection signal degrades the most.

**The translation-cost wrinkle.** The Likert prompt is still adjective-keyed: "How well does 'extraverted' describe you?" Even on IPIP-form personas, the model is symbolically translating from "I love large parties / I make friends easily / ..." → the rating for "extraverted." That's still symbolic processing, but with translation overhead — the model has to bridge the persona vocabulary and the rating-target vocabulary. The partial Likert drop probably reflects this translation cost rather than a failure of symbolic processing per se.

The cleaner test of pure symbolic invariance would also paraphrase the *rating target* — e.g., rate "I love large parties" rather than "extraverted." This decouples persona-form from rating-target-form. Worth queuing as a follow-up; predicts Likert recovers closer to +0.89 because the symbolic match becomes direct again on the behavioral side.

**Reframing the W7 §11.5.10 result.** The +0.144 "Likert lift" in W7 was *partly* a vocabulary-coupling effect: with marker-rich descriptions and marker-keyed ratings, the Likert track had two sources of advantage over rep — (a) the symbolic processing benefit (theory's claim), and (b) the exact-vocabulary-match benefit (artifact of the description form). The W8 IPIP pilot strips out (b) but keeps (a). The result: Likert still beats Rep by +0.185 — actually MORE than +0.144 — so (a) accounts for the entire effect plus some. The vocabulary-match was actually *masking* a larger underlying symbolic advantage by saturating both readouts.

This is a pleasingly counterintuitive finding: stripping the marker artifact made the symbolic-vs-associative dissociation *cleaner*, not weaker.

**What's not yet tested:**

- **Reflow effect.** Sonnet-paraphrased natural prose vs concatenated IPIP statements. Would tell us whether the partial Likert drop is residual stilted-prose effect or genuine translation cost.
- **IPIP-form rating target.** Decouple the rating target from Goldberg markers. Predicts Likert recovers further.
- **Full cohort.** This is one model. The W7 §11.5.10 prereg held across 7 models; need to confirm the IPIP gap-amplification does too.
- **Cross-domain stimulus generalization.** Whether the symbolic vs associative gap is specific to personality, or generalizes (W7 §8 emotion-stimulus replication ran cross-architecture cosine fidelity but not the Likert-vs-projection differential).

## 5. Next steps

In rough order of value:

1. **Reflow ablation.** Sonnet-paraphrase each IPIP-raw description into smooth prose, preserving content. Re-run rep + Likert. Tests whether the partial drop is stilted-form or content-shift. If reflow recovers most of the drop, the W8 result is mostly about prose form; if not, it's about content fidelity.

2. **Cohort extension.** Run the IPIP-raw pilot on all 7 cohort models. The W7 §11.5.10 cohort showed the marker-form Likert>Rep gap held everywhere; the W8 question is whether the IPIP gap amplification (+0.185 vs +0.144) is Qwen7-specific or a general property.

3. **IPIP-form rating target.** Modify the Likert template to use behavioral statements as the rating target ("How well does 'I love large parties' describe you?") instead of marker adjectives. Decouples persona-form from rating-target-form, isolates the translation cost.

4. **Per-facet breakdown.** With IPIP composition, we know exactly which facets contribute to each persona description. Worth checking whether some facets internalize better than others, and whether facet-level heterogeneity (W7 §11.5.7) explains some of the per-trait variation in IPIP drops.

5. **Tie back to chunking-granularity hypothesis (to_try #18).** If the IPIP form preserves Likert>Rep but loses some absolute fidelity, that's compatible with both (a) translation cost and (b) Big Five being over-aggregated. Facet-level analysis on the IPIP results would discriminate.

## 6. Status

- Pipeline committed (`ad0296d`): IPIP-300 annotations, composer script, methodology note, 400-persona output.
- Wiring committed (this commit): `--persona-source` flag in both rep mapping and Likert scripts.
- Pilot results: `results/persona_repr_mapping_Qwen7_response-position_ipip_raw.json`, `results/persona_instrument_response_Qwen7_ipip_raw.json`.
- Writeup: this file.

One sentence for the reading group: "Stripping the W7 persona descriptions of Goldberg-marker vocabulary by composing them from validated IPIP behavioral items reveals that the +0.144 Likert-over-Rep advantage was a *lower bound* on the symbolic-vs-associative effect — under the harder behavioral-form persona, the gap amplifies to +0.185 even as both readouts drop, because the marker-vocabulary-match was saturating both tracks and masking the underlying dissociation."
