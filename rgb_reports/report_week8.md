# Week 8 — Natural-persona prereg pilot

## 0. One-line summary

The W7 §11.5.10 +0.144 Likert-over-rep gap was substantially a *vocabulary-coupling artifact* across three channels — Goldberg adjectives in persona descriptions, Goldberg adjectives in Likert rating targets, and Goldberg-derived rep directions. As we strip each coupling (§3 → §4 → §5), the gap shrinks: +0.185 → +0.107 → **+0.047**. Under fully-matched IPIP-form conditions on Qwen 2.5 7B, rep and Likert nearly converge. The strong symbolic-vs-associative claim ("instruments measure what activations don't") is rejected; the residual +0.047 may still reflect a real dissociation, or may be further measurement-channel artifacts we haven't yet isolated. We don't know we're at the destination.

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

## 3. Pilot 1 — IPIP persona, marker rating target, marker rep direction

The first pilot changes only the persona description form. Rating template, rating-target words, and trait-direction extraction stay on Goldberg markers.

Same 50 personas (seed=42), same model (Qwen 2.5 7B), same Likert template, same 30-marker rating subset, same direction-extraction recipe.

| Trait | W7 Marker Rep | IPIP Rep | Δ Rep | W7 Marker Likert | IPIP Likert (marker target) | Δ Likert | Δ (L−R) on IPIP |
|---|---|---|---|---|---|---|---|
| A | +0.684 | +0.479 | −0.205 | +0.880 | +0.757 | −0.123 | +0.279 |
| C | +0.761 | +0.711 | −0.050 | +0.909 | +0.850 | −0.059 | +0.139 |
| E | +0.806 | +0.641 | −0.165 | **+0.960** | +0.717 | **−0.243** | +0.076 |
| N | +0.665 | +0.552 | −0.113 | +0.743 | +0.746 | **+0.003** | +0.193 |
| O | +0.800 | +0.527 | −0.273 | +0.943 | +0.766 | −0.177 | +0.239 |
| **Mean** | **+0.743** | **+0.582** | **−0.161** | **+0.887** | **+0.767** | **−0.120** | **+0.185** |

**Apparent finding:** rep drops with surface form, Likert drops less, the Likert>Rep gap *amplifies* from +0.144 (W7) to +0.185.

**Apparent interpretation (the one we held briefly):** the W7 +0.144 was a *lower bound* on the symbolic-vs-associative effect. Marker vocabulary in both persona descriptions and rating targets was saturating both readouts; stripping the description-side marker vocabulary reveals a cleaner +0.185 dissociation.

This interpretation lasted until we asked: what's the rating target still doing?

**Cross-correlation 5×5 (IPIP persona, marker target, Likert):** off-diagonal mean +0.078 (vs +0.108 marker baseline), diag-off gap +0.690 (vs +0.779 baseline). Trait-axis structure was sharper on the IPIP form than the marker baseline — at the time, this looked like additional support for the symbolic theory.

Heatmaps: `results/persona_repr_heatmap_ipip_raw_*.html` (cohort rep, 7 models), `results/persona_likert_heatmap_ipip_raw_*.html` (Qwen7 only).

## 4. Pilot 2 — IPIP persona, **IPIP rating target**, marker rep direction

The Likert prompt is still adjective-keyed: "How well does 'extraverted' describe you?" Even on IPIP-form personas, the model has to symbolically translate "I love large parties / I make friends easily" → the rating for "extraverted." That's still symbolic, but with vocabulary-translation overhead. The translation-cost hypothesis predicted: paraphrase the rating target into IPIP form (rate "I love large parties" instead of "extraverted"), Likert should recover toward the W7 ceiling because the symbolic match becomes direct again on the behavioral side.

Implemented in `scripts/persona_instrument_response.py` via `--rating-target ipip`. Per-persona rating set: 6 IPIP items per trait (3 forward + 3 reverse), facet-stratified, mild only, no deny, **excluding any items used in that persona's composition** (`picks` provenance). Each persona gets a held-out rating set drawn from items it wasn't itself constructed from.

Same 50 Qwen7 personas, same persona descriptions as §3, same rep direction extraction.

| Trait | IPIP Rep (marker dir) | IPIP Likert (marker target, §3) | IPIP Likert (IPIP target, §4) | Δ §4 vs §3 | Δ (L−R) §4 |
|---|---|---|---|---|---|
| A | +0.479 | +0.757 | +0.621 | −0.136 | +0.143 |
| C | +0.711 | +0.850 | +0.779 | −0.071 | +0.068 |
| E | +0.641 | +0.717 | +0.627 | −0.090 | −0.014 |
| N | +0.552 | +0.746 | +0.644 | −0.102 | +0.092 |
| O | +0.527 | +0.766 | **+0.774** | **+0.008** | +0.247 |
| **Mean** | **+0.582** | **+0.767** | **+0.689** | **−0.078** | **+0.107** |

**Hypothesis rejected.** Likert went *down* under IPIP rating target, not up. The translation-cost story was wrong: Goldberg adjectives weren't a translation barrier — they were doing real work as clean trait-label readouts. Goldberg adjectives are extremely well-trained as trait names in the model; rating "extraverted" against any persona description is a clean operation regardless of vocabulary match. Behavioral-statement rating items are noisier per-trait readouts because they probe specific facets and pull in facet-level variance.

**Off-diagonal collapses to +0.016** (vs +0.078 in §3, +0.108 in W7 baseline). Diag-off gap +0.673. With behavioral rating items, cross-trait leakage drops near zero — when rating "I love large parties" the model isn't pulling in C, A, or O. Cleaner trait separation than under any prior condition, at the cost of lower diagonal absolute r. The trade-off suggests Goldberg-marker ratings benefit from a kind of trait-label aggregation effect that conflates trait variance into the rating, while item-specific behavioral ratings stay isolated.

**O went UP** (0.766 → 0.774) — the only trait that improved. With IPIP rating items probing specific Openness facets (Imagination, Artistic Interests, Intellect), O resolves more cleanly than the broad marker rating ("imaginative") allows.

**Apparent interpretation at this stage:** the symbolic-vs-associative gap survives at +0.107 under matched persona/rating vocabulary, smaller than +0.185 in §3 but still positive. We thought: this is the cleanest measure of the gap, with both vocabulary couplings stripped.

This interpretation lasted until rgb asked: what about the rep direction?

Heatmaps: `results/persona_likert_heatmap_ipip_raw_target_ipip_*.html` (Qwen7).

## 5. Pilot 3 — IPIP persona, IPIP rating target, **IPIP rep direction**

The rep readout was still using Goldberg-derived trait directions (`MARKERS` dict, `mean(high adjectives) − mean(low adjectives)`, neutral-PC projected). Under §4 conditions, the Likert track was fully behavioral (IPIP persona + IPIP rating items) but rep was extracting trait directions from Goldberg adjectives and projecting onto IPIP-form persona activations — vocabulary mismatch at the readout layer.

Pilot 3 fixes this: trait directions extracted from IPIP behavioral items (forward-keyed mean − reverse-keyed mean per trait, same neutral-PC recipe, ~280 IPIP items minus 12 deny-listed). Implemented via `--direction-source ipip` flag.

| Trait | §3 Rep (marker dir) | §5 Rep (IPIP dir) | Δ Rep | §4 Likert (IPIP target) | §5 Δ (L−R) |
|---|---|---|---|---|---|
| A | +0.479 | +0.517 | +0.038 | +0.621 | +0.104 |
| C | +0.711 | +0.749 | +0.038 | +0.779 | +0.030 |
| E | +0.641 | **+0.752** | **+0.111** | +0.627 | **−0.125** |
| N | +0.552 | +0.447 | **−0.105** | +0.644 | +0.197 |
| O | +0.527 | **+0.748** | **+0.221** | +0.774 | +0.026 |
| **Mean** | **+0.582** | **+0.642** | **+0.060** | **+0.689** | **+0.047** |

**The symbolic-vs-associative gap collapses to +0.047** under fully-matched IPIP-form conditions on Qwen7. Rep recovers most of what looked like a "symbolic readout advantage" once direction extraction is matched to the rating vocabulary.

**E flips signs.** Rep r (+0.752) *beats* Likert r (+0.627) by −0.125 on E. Clean E behavioral items ("I love large parties", "I make friends easily") project onto IPIP-derived axes more reliably than they yield Likert ratings. The §3-§4 framing of "Likert always beats rep" doesn't hold trait-by-trait under matched conditions.

**N moves the wrong way.** N is the only trait where IPIP-derived directions are *worse* than Goldberg-derived directions (0.552 → 0.447). Hypothesis: IPIP forward-N items include clinical-tone content (panic, depression, suffer, overwhelmed) that's semantically heterogeneous, while Goldberg N markers ("nervous, anxious, tense, irritable") give a cleaner, narrower trait label. The N facet structure is internally heterogeneous in IPIP space (W7 §11.5.7 found N's anxiety-vs-depression facets barely correlate, r=+0.07). This connects to the chunking-granularity hypothesis (to_try #18): N may be a poorly-chunked Big Five category whose model-natural structure crosses facet boundaries.

**O and E are the big winners** of switching to IPIP directions (+0.221 and +0.111). For these traits, behavioral exemplars apparently carve up the trait axis better than Goldberg adjective markers do. Possibly because Goldberg O markers ("imaginative, creative, sophisticated") are abstract trait labels with semantic spread, while IPIP O items name specific behaviors ("I have a vivid imagination, I see beauty in things others miss") that load more cleanly.

**Final interpretation at this stage.** The W7 +0.144 Likert-over-rep gap was substantially a vocabulary-coupling artifact. Three couplings, identified one at a time:

| Pilot | What changed | Apparent gap |
|---|---|---|
| W7 §11.5.10 | (baseline) | +0.144 |
| §3 | Persona vocabulary (markers → IPIP behavioral) | +0.185 |
| §4 | + Rating target vocabulary (markers → IPIP behavioral) | +0.107 |
| §5 | + Rep direction extraction (markers → IPIP behavioral) | **+0.047** |

The story isn't "the gap amplifies under behavioral form" (§3 framing) or "the gap shrinks but persists" (§4 framing) — it's "the gap mostly evaporates as we strip vocabulary couplings." The §5 +0.047 may still reflect a real symbolic-vs-associative effect, or may further reduce as we identify the next coupling.

Heatmaps: `results/persona_repr_heatmap_ipip_full_*.html`, `results/persona_likert_heatmap_ipip_full_*.html` (Qwen7).

## 6. Cohort: rep r on IPIP personas (marker dir, response-position)

The §3 pilot is one model. We ran the rep half of §3 on the full 7-model cohort (Likert cohort still pending — ~3hr × 2 conditions). Rep extraction is fast (~2min/model).

| Model | A | C | E | N | O | Mean | Off-diag |
|---|---|---|---|---|---|---|---|
| Gemma 4B  | +0.628 | +0.677 | +0.733 | +0.785 | +0.731 | **+0.711** | +0.071 |
| Llama 3B  | +0.624 | +0.632 | +0.749 | +0.786 | +0.672 | +0.693 | +0.080 |
| Llama 8B  | +0.690 | +0.596 | +0.805 | +0.674 | +0.704 | +0.694 | +0.085 |
| Gemma 12B | +0.556 | +0.612 | +0.697 | +0.696 | +0.587 | +0.630 | +0.042 |
| Qwen 7B   | +0.479 | +0.711 | +0.641 | +0.552 | +0.527 | +0.582 | +0.117 |
| Qwen 3B   | +0.616 | +0.620 | +0.590 | +0.364 | +0.517 | +0.541 | +0.138 |
| Phi4-mini | −0.025 | +0.698 | +0.247 | +0.512 | +0.209 | **+0.328** | −0.037 |
| **Cohort mean** | **+0.510** | **+0.649** | **+0.637** | **+0.624** | **+0.564** | **+0.597** | **+0.071** |

**Cohort drop from W7 marker baseline** (cohort mean was +0.730 in §11.5.9): mean Δ = **−0.133**. Every model drops; magnitude varies from −0.058 (Llama 3B) to −0.210 (Phi4). The IPIP behavioral form is harder for all 7 models, consistently.

**Phi4-mini becomes much worse on IPIP** (0.538 → 0.328, A goes negative). Phi4 was already the cohort weak point (W7 §11.5.9 found it had different A/N coding axes than the others). IPIP form amplifies this — Phi4's residual-stream geometry is less robust to surface form change than the other models'. This connects to the §11.5.10 cohort finding that Phi4 had the *largest* Likert lift (+0.330) — Phi4's behavioral readout (Likert) was making up for a much weaker representational readout, in a way that cleanly fits the now-revised interpretation: Phi4 is just bad at residual-stream trait recovery, and Likert was masking that with marker-target vocabulary.

**Gemma 4B is again the cohort best** (+0.711) — same finding as W7 §11.5.9 (best at +0.838 there). Scale doesn't monotonically help; architecture and post-training recipe matter more.

Cohort §5 (rep with IPIP directions) and cohort §4 (Likert with IPIP target) are not yet run. These are the natural extensions to confirm whether the §5 collapse-to-+0.047 generalizes.

## 7. Interpretation

**The W8 finding deflates the W7 §11.5.10 framing.** "Instruments measure what activations don't" was the strong symbolic-vs-associative claim, and at +0.144 it looked supported. At +0.047 under matched-vocabulary conditions, the claim is in trouble — under our cleanest measurement, rep and Likert are nearly equivalent on Qwen7.

**This is methodologically clarifying, not invalidating.** The W7 finding is real: under marker-form persona descriptions and marker-keyed rating targets, the Likert track recovers sampled z's at substantially higher r than the rep track. That's a robust empirical fact across 7 models. What we now understand is *why*: vocabulary coupling. Three Goldberg-marker channels (description, rating target, direction extraction) reinforced each other into a Likert-track advantage that mostly dissolves once any one channel is decoupled.

**The remaining +0.047 may or may not be meaningful.** Possible explanations:

- **Real residual symbolic effect.** Per-item Likert ratings could still benefit slightly from being a behavioral judgment rather than a residual-stream projection, even under matched vocabulary. Would need cohort + further controls to confirm.
- **Per-persona rating-set noise.** §4/§5 use per-persona rating sets (different items per persona to enforce non-overlap with composition). Adds variance; could account for some of the residual gap.
- **Direction-extraction sample size.** IPIP direction extraction uses ~280 items vs ~104 for markers. More noise per direction estimate. Could go either way — more items = more signal averaging, but also more facet-level variance.
- **Layer-locality.** Both readouts read at common_layer (~2/3 depth). Different surface forms might be most cleanly separated at different layers. Not tested.

**Per-trait pattern is the most informative slice.** Aggregated by mean across 5 traits, the cleaner picture obscures big per-trait differences:

- **E and O do well with behavioral readout** (§5: rep r 0.752, 0.748). Behavioral exemplars carve these axes cleanly. Marker rating targets don't add much.
- **N does badly with behavioral readout on the rep side** (§5: rep r 0.447). N is the most internally heterogeneous trait; IPIP direction extraction reflects that. Goldberg N markers are abstract enough to give a coherent direction; IPIP N items aren't.
- **C is robust on both** (§5: rep r 0.749, Likert r 0.779). Most surface-form-stable trait we've seen.
- **A is weak across all conditions** (§5: rep r 0.517, Likert r 0.621). A may genuinely be the hardest Big Five trait to recover from this kind of measurement, regardless of channel choice.

**Connection to the chunking-granularity hypothesis (to_try #18).** N's bad behavior under IPIP directions is exactly what the chunking hypothesis predicts: if Big Five categories are over-aggregating model-natural primitives, the trait whose internal structure is most heterogeneous (W7 §11.5.7 already named N) should suffer most when direction extraction averages over heterogeneous behavioral exemplars. Goldberg's adjective form coarsens away the internal heterogeneity into a clean label; IPIP's behavioral form preserves it, which is informative for some traits (E, O) and hurts for others (N).

**Where this leaves the symbolic-vs-associative theory.** Weakened but not killed. The strong form ("instruments and activations measure different things") is rejected at the +0.047 residual. The weaker form ("there's still some advantage to per-item symbolic judgment over residual-stream projection") might survive — but the burden of proof has shifted. We need cleaner controls, not bigger numbers.

## 8. Heatmaps and figures

All figures in `results/`. HTML — open in a browser.

**W7 baselines (markers throughout):**
- `persona_repr_heatmap_per_model.html` — 8-panel rep grid (Σ + 7 cohort models, marker dir, marker persona)
- `persona_repr_heatmap_cross_model.html` — cross-model agreement per trait, 7×7
- `persona_repr_heatmap_scatter.html` — per-trait scatter
- `persona_likert_heatmap_*.html` — same three layouts for Likert

**§3 (IPIP persona, marker dir, marker target):**
- `persona_repr_heatmap_ipip_raw_*.html` — 7-model cohort
- `persona_likert_heatmap_ipip_raw_*.html` — Qwen7 only

**§4 (IPIP persona, marker dir, IPIP target):**
- `persona_likert_heatmap_ipip_raw_target_ipip_*.html` — Qwen7 only

**§5 (IPIP persona, IPIP dir, IPIP target):**
- `persona_repr_heatmap_ipip_full_*.html` — Qwen7 only
- `persona_likert_heatmap_ipip_full_*.html` — Qwen7 only

To regenerate: `scripts/persona_repr_heatmap.py --variant {markers, ipip_raw, ipip_raw_target_ipip, ipip_full}`.

## 9. Next steps

In rough order of value:

1. **Cohort §5 — rep with IPIP directions across 7 models.** Confirms whether the +0.060 rep recovery is Qwen7-specific. ~15min of compute. If cohort holds, the §5 collapse generalizes; if Qwen7 is an outlier, the residual +0.047 is suspect.

2. **Cohort §4 — Likert with IPIP target across 7 models.** ~85min of compute. Tells us if the off-diagonal collapse to +0.016 generalizes. Together with #1, gives us the four-cell × 7-model matrix.

3. **Reflow ablation.** Sonnet-paraphrase IPIP-raw descriptions into smooth prose, preserving content. Re-run §3 / §5. Tests whether stilted "I... I... I..." form is contributing to the per-trait variability, or whether the §5 picture is form-stable.

4. **Per-facet rep direction extraction.** Direct test of the chunking hypothesis (to_try #18). Build per-facet directions instead of per-trait; see if N's facet directions cohere or fragment, and whether the W7 §8.4 cross-stimulus failure improves at facet granularity.

5. **Layer sweep.** Both readouts at common_layer (~2/3 depth). The right layer for behavioral-form readouts may differ. Worth a small ablation over 5-7 layers around the chosen one.

## 10. Status

Commits:
- `ad0296d` — IPIP composer pipeline (annotations, composer script, methodology note, 400-persona output)
- `461fd81` — §3 wiring + Qwen7 §3 result
- (this commit) — §4 IPIP-target Likert + §5 IPIP-direction rep + cohort §3 rep + heatmap-script variants + report §3-§7 expansion

Result files (gitignored, regenerable):
- `persona_repr_mapping_<MODEL>_response-position_ipip_raw.json` — §3 cohort rep (7 models)
- `persona_repr_mapping_Qwen7_response-position_ipip_raw_dir-ipip.json` — §5 rep (Qwen7)
- `persona_instrument_response_Qwen7_ipip_raw.json` — §3 Likert (Qwen7)
- `persona_instrument_response_Qwen7_ipip_raw_target-ipip.json` — §4/§5 Likert (Qwen7)

One sentence for the reading group: "The W7 §11.5.10 +0.144 'Likert beats Rep' gap was substantially a vocabulary-coupling artifact across three Goldberg-marker channels (persona description, rating target, direction extraction); stripping all three on Qwen 2.5 7B collapses the gap to +0.047, but per-trait the picture is much more interesting — E and O move strongly toward rep ≈ Likert ≈ 0.75, while N exhibits the opposite (rep drops to 0.45 with IPIP directions because N is internally heterogeneous), suggesting that Big Five trait categories may not be the right units for cross-channel personality measurement in models."
