# Week 8 — Natural-persona prereg pilot

## 0. One-line summary

**Rep is vocabulary-bound, reasoning is vocabulary-free.** The W7 §11.5.10 +0.144 Likert-over-rep gap was substantially a vocabulary-coupling artifact: when persona, rep direction, and rating target all use Goldberg adjectives, both readouts are at peak. As we de-Goldberg the persona, the rep readout takes systematic hits because Goldberg-derived directions are geometrically far from the residual-stream encoding of behavioral scenarios; Likert ratings are largely unaffected by the vocabulary mismatch — the model's symbolic forward reasoning bridges the gap that the activation projection can't. Cohort matched-condition gap (full-IPIP readout): +0.052 (raw) or +0.075 (reflowed), range −0.028 (Llama 8B raw, rep beats Likert) to +0.122 (Gemma 12B reflowed). The strong symbolic-vs-associative claim ("instruments and activations measure different things") is rejected; the residual gap is small. **The interesting finding is not the size of the gap but the asymmetric vocabulary-binding** — activation projections are tied to extraction-vocabulary in a way symbolic reasoning isn't. We don't know we're at the destination, but we now have a sharper question: why does the residual-stream geometry decode behavioral scenarios poorly via Goldberg-derived axes, when the same model can rate behavioral statements against a Goldberg-marker persona just fine?

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

Heatmaps: `results/persona_repr_heatmap_ipip_full_*.html`, `results/persona_likert_heatmap_ipip_full_*.html` (cohort, post-§6).

## 6. Cohort sweep — all 7 models × all four conditions

The W8 §3-§5 pilots were single-model (Qwen7). To confirm the picture generalizes, we ran the §3 rep, §5 rep, and §4 Likert tracks on the full 7-model cohort.

### 6.1. §3 rep (IPIP persona, marker dir, response-position)

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

Cohort drop from W7 marker baseline (cohort mean +0.730): −0.133. Every model drops, magnitude −0.058 (Llama 3B) to −0.210 (Phi4). The behavioral persona form is universally harder than the marker form for rep.

**Phi4 collapses dramatically** (0.538 → 0.328, A goes negative). Already the cohort weak point in W7 with "different A/N coding axes" — IPIP form amplifies this. **Gemma 4B remains cohort best** (+0.711). Scale doesn't monotonically help.

### 6.2. §5 rep (IPIP persona, **IPIP dir**, response-position)

| Model | A | C | E | N | O | Mean | Off-diag | Δ vs §3 |
|---|---|---|---|---|---|---|---|---|
| Gemma 4B  | +0.314 | +0.649 | +0.727 | +0.803 | +0.718 | +0.642 | +0.007 | **−0.069** |
| Llama 3B  | +0.549 | +0.700 | +0.714 | +0.710 | +0.647 | +0.664 | +0.088 | −0.029 |
| Llama 8B  | +0.557 | +0.729 | +0.845 | +0.652 | +0.767 | +0.710 | +0.093 | +0.016 |
| Gemma 12B | +0.749 | +0.704 | +0.788 | +0.807 | +0.782 | **+0.766** | +0.028 | **+0.136** |
| Qwen 7B   | +0.517 | +0.749 | +0.752 | +0.447 | +0.748 | +0.642 | +0.104 | +0.060 |
| Qwen 3B   | +0.477 | +0.652 | +0.636 | +0.335 | +0.544 | +0.529 | +0.127 | −0.012 |
| Phi4-mini | +0.504 | +0.784 | +0.631 | +0.674 | +0.516 | +0.622 | +0.078 | **+0.294** |
| **Cohort mean** | **+0.524** | **+0.710** | **+0.728** | **+0.633** | **+0.674** | **+0.654** | **+0.075** | **+0.057** |

**Cohort gain from §3 to §5: +0.057.** But Phi4 dominates: +0.294 alone. Without Phi4, cohort mean improvement is +0.012 — essentially flat. **For 6 of 7 models, marker-vs-IPIP direction extraction is roughly a wash on average rep r**, with mixed per-trait shifts (Gemma 4B drops on A from 0.628 → 0.314 but gains on C/N/O; the model-level mean change masks substantial per-trait reorganization).

Phi4's recovery is the cleanest per-model finding: under Goldberg directions Phi4 had A = −0.025, N = +0.512, mean +0.328; under IPIP directions it's A = +0.504, N = +0.674, mean +0.622. **Phi4's W7 §11.5.9 "different coding axes" was specifically a Goldberg-marker artifact** — when directions are extracted from behavioral content, Phi4's residual-stream trait recovery is in line with the rest of the cohort.

**Gemma 12B is the new cohort top** at +0.766 (was 0.630 under §3). Gemma 12B's representations are well-aligned with IPIP-derived axes specifically; IPIP item activations carve cleaner trait directions than Goldberg adjectives do for this model.

**N is the trait that goes the wrong way for several models**: Gemma 4B is OK on N (0.803 stable from §3 0.785), but Qwen 7B's N drops 0.552 → 0.447, and Qwen 3B's drops 0.364 → 0.335. The N-with-IPIP-dirs problem is real but heterogeneous. Goldberg N markers are narrower-clinical ("anxious, nervous, irritable"); IPIP N items pull in clinical Depression-facet content ("I dislike myself", "I feel desperate") that may extract a less coherent direction.

Off-diagonal mean across cohort: +0.075. Comparable to §3's +0.071. IPIP directions don't sharpen the off-diagonal at the rep level.

### 6.3. §4 Likert (IPIP persona, **IPIP target**)

| Model | A | C | E | N | O | Mean | Off-diag | Δ vs W7 baseline |
|---|---|---|---|---|---|---|---|---|
| Gemma 4B  | +0.717 | +0.750 | +0.707 | +0.793 | +0.667 | +0.727 | +0.044 | −0.196 |
| Llama 3B  | +0.751 | +0.736 | +0.727 | +0.697 | +0.837 | +0.750 | +0.063 | −0.153 |
| Llama 8B  | +0.737 | +0.790 | +0.648 | +0.558 | +0.680 | +0.682 | +0.021 | −0.217 |
| Gemma 12B | +0.865 | +0.783 | +0.790 | +0.822 | +0.838 | **+0.819** | +0.035 | −0.101 |
| Qwen 7B   | +0.621 | +0.779 | +0.627 | +0.644 | +0.774 | +0.689 | +0.016 | −0.198 |
| Qwen 3B   | +0.303 | +0.559 | +0.616 | +0.584 | +0.590 | +0.530 | +0.031 | −0.263 |
| Phi4-mini | +0.722 | +0.731 | +0.755 | +0.743 | +0.742 | +0.739 | +0.035 | −0.129 |
| **Cohort mean** | **+0.674** | **+0.733** | **+0.696** | **+0.692** | **+0.733** | **+0.705** | **+0.035** | **−0.180** |

**Cohort drop from W7 Likert baseline: −0.180** (cohort mean 0.885 → 0.705). Every model drops. Largest drop on Qwen 3B (−0.263) — Qwen 3B's marker-vocabulary advantage was the largest, and stripping it costs the most.

**Cohort off-diagonal collapses to +0.035** — sharpest trait-axis structure at any condition, including the W7 baseline (+0.108). The off-diagonal collapse generalizes: across all 7 models, behavioral rating items minimize cross-trait leakage.

**Per-trait pattern is more uniform than under §5 rep.** Cohort means: A 0.674, C 0.733, E 0.696, N 0.692, O 0.733. Range across traits ≈ 0.06 (vs §5 rep's 0.20 spread). Likert under behavioral targets gives more even per-trait recovery than rep does.

### 6.4. The cohort matched gap (§4 Likert − §5 rep)

| Model | §5 Rep | §4 Likert | **Δ matched** |
|---|---|---|---|
| Llama 8B | 0.710 | 0.682 | **−0.028** |
| Qwen 3B | 0.529 | 0.530 | +0.001 |
| Qwen 7B | 0.642 | 0.689 | +0.047 |
| Gemma 12B | 0.766 | 0.819 | +0.053 |
| Gemma 4B | 0.642 | 0.727 | +0.085 |
| Llama 3B | 0.664 | 0.750 | +0.086 |
| Phi4-mini | 0.622 | 0.739 | +0.117 |
| **Cohort mean** | **0.654** | **0.705** | **+0.052** |

**The Qwen7 +0.047 generalizes.** Cohort matched gap is +0.052, range −0.028 (Llama 8B!) to +0.117 (Phi4). The symbolic-vs-associative gap survives at cohort level under matched IPIP conditions, but is small (~+0.05) and **has counterexamples** — Llama 8B genuinely shows rep beating Likert.

The W7 +0.144 cohort Likert-over-Rep was inflated by 3× compared to the matched-condition measurement. Three vocabulary couplings were each contributing roughly +0.03 to the apparent gap, plus Phi4's representational weakness was inflating the average specifically on the rep side under markers (which Likert was implicitly compensating for).

## 7. Interpretation

### 7.1. Headline framing — vocabulary-bound rep, vocabulary-free reasoning

The cleanest single contrast in the W8 data is between two reflowed-persona conditions that differ *only* in which vocabulary the rep readout uses:

| Persona | Readout vocab | Rep r (cohort) | Likert r (cohort) |
|---|---|---|---|
| IPIP-reflow | Goldberg dir + Goldberg target | **0.584** | **0.752** |
| IPIP-reflow | IPIP dir + IPIP target | **0.669** | **0.744** |
| **vocabulary-match cost** | | **+0.085 rep gains** | **−0.008 Likert (flat)** |

Same persona prose. Switching the readout vocabulary from Goldberg to IPIP gives rep a +0.085 boost and Likert essentially nothing. The activation projection is *bound* to the vocabulary used to extract its directions: Goldberg-derived axes are geometrically far from where the residual-stream encoding of behavioral scenarios lives. The Likert response, by contrast, can rate Goldberg adjectives ("extraverted") against an IPIP-behavioral persona just as accurately as it rates IPIP behavioral statements ("I love large parties") against the same persona. Symbolic forward reasoning bridges the vocabulary gap that the activation projection can't.

This reframes the whole W8 result. The story is not "the symbolic-vs-associative gap shrinks under matched vocabulary" (a technical correction to W7's framing). The story is **the model's residual-stream geometry is vocabulary-tied, but its forward reasoning is vocabulary-free.** Cf. Mahowald et al. (2024) on formal vs functional linguistic competence — a similar dissociation, in a different domain.

This connects to the broader recognition-vs-execution literature on LLM representations (CARE 2024; Wu et al. 2026 "Knowing without Acting," arXiv:2603.05773, which demonstrates a causal double-dissociation between recognition and execution subspaces in safety mechanisms). Where they find recognition and execution use different subspaces for one vocabulary, we find a finer-grained version: even within "recognition" (rep readout), the geometry is contingent on the extraction vocabulary. The model can SYMBOLICALLY recognize across vocabularies; the LINEAR PROJECTION can't.

### 7.2. The numerical story

**The W8 finding deflates the W7 §11.5.10 framing.** "Instruments measure what activations don't" was the strong symbolic-vs-associative claim, and at +0.144 it looked supported. At +0.052 cohort matched-vocabulary (raw) or +0.075 (reflow), the strong claim is in trouble. Under our cleanest measurement, rep and Likert are roughly equivalent on average, with substantial per-model variation.

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

**Where this leaves the symbolic-vs-associative theory.** Weakened but not killed. The strong form ("instruments and activations measure different things") is rejected at the +0.052 cohort residual. The weaker form ("there's still some advantage to per-item symbolic judgment over residual-stream projection on average") survives at cohort level — but it has model-level counterexamples (Llama 8B), per-trait counterexamples (E in Qwen7), and a substantial portion of the W7-apparent gap turned out to be Phi4's idiosyncratic representational weakness under markers. The burden of proof has shifted. We need cleaner controls, not bigger numbers.

**The cohort matched gap (+0.052) has interesting per-model structure.** Three regimes:

- **Pro-Likert** (gap +0.05 to +0.12): Phi4, Llama 3B, Gemma 4B, Gemma 12B. These models show a real but small symbolic advantage.
- **Indifferent** (gap ±0.01): Qwen 3B, Qwen 7B (the Qwen family). Likert and rep nearly equivalent under matched conditions.
- **Pro-rep** (gap −0.03): Llama 8B. Rep beats Likert.

The Qwen family hugging the rep readout is the most theoretically interesting subset — it's evidence that the symbolic-vs-associative dissociation isn't a universal property of how LLMs handle personality, but is contingent on the post-training recipe. Qwen 2.5 in particular may have post-training that aligns the residual-stream trait representation more directly with behavioral judgment than other model families do.

## 8. Reflow ablation (Qwen7 + cohort)

The §3 pilot's "gap amplifies +0.144 → +0.185" finding had an alternative explanation we hadn't tested: the IPIP-raw form is choppy first-person prose ("I X. I Y. I Z.") that may activate analytic mode in the model. If so, the §3 amplification was a stilted-prose penalty on rep, not a symbolic-vs-associative effect. The reflow ablation tests this directly.

`scripts/sonnet_reflow_personas.py` paraphrases each IPIP-raw persona via Anthropic API (claude-sonnet-4-6) into smooth first-person prose that preserves all behavioral statements and their magnitude/qualifier without introducing trait-name adjectives. Result stored in `synthetic_personas_ipip.json` as a parallel `ipip_reflowed` field. Reflowed at the 50 pilot personas (seed=42); cost ~$0.30, ~3.5min API time.

Smoke-checked outputs: every input statement appears in some form in the reflowed prose; magnitude qualifiers preserved ("love" stays "love", "rarely" stays "rarely"); no trait-name adjective leakage. Connectives ("though", "yet", "but") sometimes added — minor risk of altering parallelism into contrast, but acceptable.

### 8.1. Qwen7 four-cell summary (initial pilot)

| Condition | Persona | Rep dir | Target | Rep r | Likert r | Δ (L−R) |
|---|---|---|---|---|---|---|
| W7 baseline | Marker | Marker | Marker | 0.743 | 0.887 | +0.144 |
| §3 raw | IPIP raw | Marker | Marker | 0.582 | 0.767 | +0.185 |
| **§3 reflowed** | IPIP reflowed | Marker | Marker | **0.673** | **0.770** | **+0.097** |
| §4 raw | IPIP raw | Marker | IPIP | 0.582 | 0.689 | +0.107 |
| §4 reflowed | IPIP reflowed | Marker | IPIP | 0.673 | 0.756 | +0.083 |
| §5 raw | IPIP raw | IPIP | IPIP | 0.642 | 0.689 | +0.047 |
| **§5 reflowed** | IPIP reflowed | IPIP | IPIP | **0.689** | 0.756 | +0.067 |

On Qwen7 specifically, reflow recovers most of the rep penalty from going to behavioral form — and the §3 gap drops from +0.185 to +0.097. We initially read this as "the §3 amplification was a stilted-prose artifact." But this turned out to be a Qwen-specific phenomenon. The cohort sweep tells a different story.

### 8.2. Cohort reflow (full 7-model sweep)

| Model | §5 raw R | §5 ref R | Δ R | §4 raw L | §4 ref L | Δ L | raw Gap | ref Gap | Δ Gap |
|---|---|---|---|---|---|---|---|---|---|
| Gemma 4B  | 0.642 | 0.672 | +0.030 | 0.727 | 0.745 | +0.018 | +0.085 | +0.073 | −0.012 |
| Llama 3B  | 0.664 | 0.708 | +0.044 | 0.750 | 0.777 | +0.027 | +0.086 | +0.069 | −0.017 |
| Phi4-mini | 0.622 | 0.638 | +0.016 | 0.739 | 0.757 | +0.018 | +0.117 | +0.119 | +0.002 |
| Qwen 3B   | 0.529 | 0.605 | **+0.076** | 0.530 | 0.665 | **+0.135** | +0.001 | +0.061 | **+0.060** |
| Gemma 12B | **0.766** | **0.662** | **−0.104** | 0.819 | 0.784 | −0.035 | +0.053 | +0.122 | **+0.069** |
| Llama 8B  | 0.710 | 0.706 | −0.004 | 0.682 | 0.720 | +0.038 | −0.028 | +0.014 | +0.042 |
| Qwen 7B   | 0.642 | 0.689 | +0.047 | 0.689 | 0.756 | +0.067 | +0.047 | +0.067 | +0.020 |
| **Cohort** | **0.654** | **0.669** | **+0.015** | **0.705** | **0.744** | **+0.039** | **+0.052** | **+0.075** | **+0.023** |

### 8.3. Cohort findings, in tension with the Qwen7 single-model story

- **Reflow effects on rep are heterogeneous, not uniformly positive.** Cohort §5 rep mean +0.015 (small). But per-model: Qwen 3B +0.076 and Qwen 7B +0.047 (both gain), Llama 3B +0.044 and Gemma 4B +0.030 (small gain), Phi4 and Llama 8B flat, **Gemma 12B −0.104** (big drop). The "stilted prose hurt rep" hypothesis from the Qwen7 pilot doesn't generalize — Gemma 12B's residual stream is *more* aligned with raw IPIP prose than with reflowed prose.
- **Reflow consistently helps Likert** at cohort level (+0.039). Larger and more uniform than the rep effect. Cohort range +0.018 to +0.135 (Qwen 3B is the big winner; Gemma 12B is the only model where Likert drops, and only by −0.035).
- **Reflow widens the matched gap on average** (cohort +0.052 → +0.075). Driven by the asymmetric Likert > rep recovery — Likert benefits more from reflow than rep does, so the gap grows. This is **the opposite of what the Qwen7-only story predicted** (where reflow shrinks the §3 gap dramatically). The Qwen7 §3 finding was specifically about Qwen7's stilted-prose vulnerability on rep, which doesn't generalize.
- **All 7 models have positive matched gap under reflow.** Llama 8B (the cohort raw counterexample at −0.028) flips to +0.014 under reflow. Qwen 3B (the near-zero raw counterexample at +0.001) jumps to +0.061. The pro-rep counterexamples disappear under reflow.
- **Per-model variation stays large.** Range +0.014 (Llama 8B) to +0.122 (Gemma 12B). Reflow compresses the negative tail but not the positive tail.

### 8.4. Revised story

The cohort sweep substantially complicates the §8.1 Qwen7 framing:

- **The §3 raw "+0.185 amplification" is real but Qwen-specific.** Qwen7 has a particular vulnerability to stilted prose on rep that reflow rescues. Other models don't show this — Gemma 4B/Llama 3B/Llama 8B/Gemma 12B all have raw §3 rep > reflow §3 rep or roughly equal.
- **The §5 matched gap is roughly form-stable at cohort level** (+0.052 → +0.075). The W8 §5 finding survives reflow at cohort level, just with a slight upward shift. The neighborhood is +0.05–+0.08, not exactly +0.052.
- **Reflow asymmetrically helps Likert more than rep.** This is the consistent cohort finding. Behavioral rating items benefit more from coherent prose than rep-direction projections do.
- **The Llama 8B "rep beats Likert" counterexample doesn't survive reflow.** Under matched IPIP form with reflowed prose, every cohort model has Likert ≥ rep. The W8 §6 cohort claim "model-level counterexamples exist" weakens — under reflow conditions, the cohort is uniformly pro-Likert (small to moderate).

### 8.5. Per-model regimes, revised under reflow

Three regimes from W8 §6 (raw matched gap):
- Pro-Likert (+0.05 to +0.12): Phi4, Llama 3B, Gemma 4B, Gemma 12B
- Indifferent (±0.01): Qwen 3B, Qwen 7B
- Pro-rep (−0.03): Llama 8B

Under reflow:
- All pro-Likert (cohort range +0.014 to +0.122, mean +0.075).
- Phi4 stable as biggest-Likert.
- Gemma 12B becomes biggest-Likert tied with Phi4 (+0.122). Driven by rep dropping, not Likert rising.
- Qwen 3B moves from indifferent (+0.001) to pro-Likert (+0.061).
- Llama 8B moves from pro-rep (−0.028) to weakly pro-Likert (+0.014).

The "Qwen family is indifferent" subfinding from §6 doesn't survive reflow — Qwen 3B in particular jumps significantly. Whatever was making Qwen 3B's matched gap zero under raw conditions was prose-form-specific.

### 8.6. Per-trait cohort matched gap (raw vs reflow)

| Trait | §5 R raw | §4 L raw | gap raw | §5 R reflow | §4 L reflow | gap reflow | Δ gap |
|---|---|---|---|---|---|---|---|
| A | +0.524 | +0.674 | **+0.150** | +0.588 | +0.758 | **+0.170** | +0.020 |
| C | +0.709 | +0.733 | +0.023 | +0.703 | +0.745 | +0.042 | +0.019 |
| E | +0.728 | +0.696 | **−0.032** | +0.706 | +0.731 | +0.024 | +0.057 |
| N | +0.633 | +0.692 | +0.059 | +0.675 | +0.698 | +0.024 | −0.035 |
| O | +0.675 | +0.733 | +0.058 | +0.671 | +0.786 | **+0.114** | +0.056 |
| **Mean** | **+0.654** | **+0.705** | **+0.052** | **+0.669** | **+0.744** | **+0.075** | **+0.023** |

**Per-trait findings:**

- **A is consistently strongly pro-Likert** (raw +0.150, reflow +0.170). The biggest symbolic-vs-associative effect by trait — Likert recovers A about 0.16 better than rep does, regardless of prose form. Plausibly because A behavioral items are about social-relational behaviors that the residual stream encodes diffusely, while Likert ratings synthesize them into a per-item judgment.
- **E flips cohort-level under raw matched conditions** (gap −0.032). Rep beats Likert on E. Per-model: 4 of 7 models have rep ≥ Likert on E, including Llama 8B (−0.197) and Qwen 7B (−0.125). E behavioral items ("I love large parties," "I make friends easily") project onto IPIP-derived E directions more reliably than they yield Likert ratings on those same statements. The W8 §5 Qwen7-specific E-flip generalizes to a cohort-level pattern. Under reflow this disappears (gap +0.024) — but only because rep drops slightly more than Likert gains, not because the underlying alignment changed.
- **C is the most form-stable trait** (gap +0.023 raw, +0.042 reflow). Always small. C is the trait where rep and Likert measure roughly the same thing, and reflow doesn't change that.
- **N moves the wrong way under reflow** (+0.059 → +0.024). N's rep gains more from reflow than its Likert does — opposite to all other traits. Consistent with N's internal heterogeneity (W7 §11.5.7 anxiety-vs-depression r=+0.07): smoother prose may help the rep extraction average over N's heterogeneous items more cleanly. Connects to chunking-granularity hypothesis (to_try #18).
- **O grows substantially under reflow** (+0.058 → +0.114). O's Likert recovery is form-sensitive — IPIP behavioral items for O ("I have a vivid imagination," "I see beauty in things others miss") give cleaner ratings under coherent prose than choppy. O is mostly carried by the Likert side under reflow.

**The W8 §5 "gap collapses to +0.047" framing was particularly misleading on E.** The Qwen7 single-model gap of +0.047 averaged a strongly positive A (+0.143) with strongly negative E (−0.125), C/N/O small. The cohort confirms this: even at cohort level, the +0.052 raw matched gap conceals a per-trait spread from −0.032 (E) to +0.150 (A). The trait-aggregated "gap" obscures more than it reveals.

### 8.7. Methodological note

The reflow is *generated by Sonnet*, an LLM that may or may not perfectly preserve content. Spot-checks on s1 / s6 / s50 looked good, but we don't have an automated content-preservation check. The composer's per-pick provenance is the ground truth (which IPIP items were used), but we can't verify reflowed prose contains all of them without item-level NLI matching. For the W8 ablation as currently scoped this is acceptable; a future tighter ablation could use a Claude-as-judge content-preservation validator.

Reflow was run on the 50 pilot personas (seed=42); the remaining 350 in `synthetic_personas_ipip.json` have no `ipip_reflowed` field. Re-run with `--persona-ids` for additional sets if needed.

Cohort reflow heatmaps available at `results/persona_repr_heatmap_ipip_reflowed_full_*.html` and `persona_likert_heatmap_ipip_reflowed_full_*.html` (matched-condition: reflowed persona × IPIP dir × IPIP target).

### 8.8. Headline figures

`scripts/persona_w8_summary_plot.py` generates two summary HTML figures:

- `results/persona_w8_trajectory.html` — cohort mean rep + Likert across the 5 conditions (W7 → §3 raw → §4 raw → §5 raw → §5 reflow), with the matched gap shown as bars below. The rep recovers slowly across vocabulary-coupling strip-down; Likert drops; gap shrinks then partially recovers under reflow.
- `results/persona_w8_per_model_gap.html` — per-model matched gap raw → reflow slope plot. Shows the heterogeneity directly: most models grow under reflow, Gemma 4B and Llama 3B shrink slightly, all converge into the +0.01 to +0.12 band. Cohort-mean diamond marker shows aggregate trajectory.

## 9. Heatmaps and figures

All figures in `results/`. HTML — open in a browser.

**W7 baselines (markers throughout):**
- `persona_repr_heatmap_per_model.html` — 8-panel rep grid (Σ + 7 cohort models, marker dir, marker persona)
- `persona_repr_heatmap_cross_model.html` — cross-model agreement per trait, 7×7
- `persona_repr_heatmap_scatter.html` — per-trait scatter
- `persona_likert_heatmap_*.html` — same three layouts for Likert

**§3 (IPIP persona, marker dir, marker target):**
- `persona_repr_heatmap_ipip_raw_*.html` — full 7-model cohort
- `persona_likert_heatmap_ipip_raw_*.html` — Qwen7 only

**§4 (IPIP persona, marker dir, IPIP target):**
- `persona_likert_heatmap_ipip_raw_target_ipip_*.html` — full 7-model cohort

**§5 (IPIP persona, IPIP dir, IPIP target):**
- `persona_repr_heatmap_ipip_full_*.html` — full 7-model cohort
- `persona_likert_heatmap_ipip_full_*.html` — full 7-model cohort

To regenerate: `scripts/persona_repr_heatmap.py --variant {markers, ipip_raw, ipip_raw_target_ipip, ipip_full}`.

## 10. Next steps

In rough order of value:

1. **Per-facet rep direction extraction.** Direct test of the chunking hypothesis (to_try #18). Build per-facet directions instead of per-trait; check whether the cross-stimulus failure improves at facet granularity, and whether N's facet directions cohere or fragment. The cohort §5 result shows N is the trait most affected by direction-source choice (Gemma 4B 0.785 → 0.803, but Qwen 7B 0.552 → 0.447, Qwen 3B 0.364 → 0.335) — facet-level analysis should clarify whether N is genuinely heterogeneous or just unstably extracted.

2. **Cohort reflow extension.** Run the §8 reflow ablation on the other 6 cohort models. Tests whether the per-model regimes (pro-Likert / indifferent / pro-rep) are reflow-stable. Particularly interesting: does Llama 8B's "pro-rep" gap (−0.028) survive reflow, or does rep recover further and tip even more toward rep?

3. **Layer sweep.** Both readouts at common_layer (~2/3 depth). The right layer for behavioral-form readouts may differ. Worth a small ablation over 5-7 layers around the chosen one. Could explain Llama 8B's pro-rep result if rep is naturally cleaner at a different depth on that architecture.

4. **Investigate Qwen-family anomaly.** Qwen 3B and Qwen 7B both have near-zero matched gap (+0.001, +0.047) — substantially smaller than other model families. Worth checking whether this is an artifact of Qwen's post-training recipe, tokenizer, or something else.

5. **Headline visualization.** A multi-panel figure showing the four conditions × reflow × 5 traits × 7 models would be useful for the reading group. Currently the data lives across many HTML heatmaps; a single comparison plot (mean diagonal r per condition, faceted by model or trait) would be the headline figure for W8.

6. **Tighter reflow content-preservation check.** The current reflow trusts Sonnet to preserve content; we don't have an automated check. Could add a Claude-as-judge step that rates reflowed prose on whether each input statement is faithfully expressed. Mostly a methodology improvement; the W8 finding probably survives without it given how stable §5 reflowed → raw was (+0.067 vs +0.047).

## 11. Status

Commits:
- `ad0296d` — IPIP composer pipeline (annotations, composer script, methodology note, 400-persona output)
- `461fd81` — §3 wiring + Qwen7 §3 result
- `b7335eb` — §4/§5 wiring + Qwen7 §4/§5 + cohort §3 rep + heatmap-script variants + report §3-§7 expansion
- `347e0a6` — cohort §4 Likert + cohort §5 rep + report §6 cohort sweep + §0/§7/§8/§9 updates
- (this commit) — Sonnet reflow pipeline + Qwen7 reflow ablation + report §8 reflow section + ipip_reflowed wired into rep+Likert scripts

Result files (gitignored, regenerable):
- `persona_repr_mapping_<MODEL>_response-position_ipip_raw.json` — §3 cohort rep (7 models)
- `persona_repr_mapping_<MODEL>_response-position_ipip_raw_dir-ipip.json` — §5 cohort rep (7 models)
- `persona_instrument_response_<MODEL>_ipip_raw_target-ipip.json` — §4 cohort Likert (7 models)
- `persona_repr_mapping_Qwen7_response-position_ipip_reflowed*.json` — §8 reflow rep (Qwen7)
- `persona_instrument_response_Qwen7_ipip_reflowed*.json` — §8 reflow Likert (Qwen7)

One sentence for the reading group: "The W7 §11.5.10 cohort +0.144 'Likert beats Rep' gap was substantially a vocabulary-coupling artifact across three Goldberg channels (persona description, rating target, rep direction extraction) plus a stilted-prose penalty on rep specifically; stripping all of these on Qwen 2.5 7B reduces the gap to +0.067, and the cohort matched-condition gap (+0.052 mean) confirms the picture across 7 models with model-level variation including Llama 8B as a counterexample (rep beats Likert, −0.028) — suggesting the symbolic-vs-associative dissociation is real but small, contingent on post-training recipe, and not a universal property of LLM personality processing."
