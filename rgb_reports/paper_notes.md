# Notes that should probably go in the paper

Collected 2026-10-18 against the paper draft at that date (`_03-general-method.qmd`,
`_04-overall-results.qmd`). Each item says where it goes, what it says, where the
number comes from, and whether the draft has it already. Tags:

- **STALE**: the draft says something the current code contradicts. Fix first.
- **GAP**: not in the draft.
- **PARTIAL**: in the draft but incomplete.
- **WAIT**: depends on a run that hasn't finished. Don't freeze the number yet.

Section F is rgb's results outline with the evidence on file under each
bullet. Several "(not done yet)" items have in fact been run.

Numbers are from the current code (`_analysis.channels`, `pkit.channels`,
`pkit.jointdiag`) unless marked "legacy" (computed under an older convention and
not regenerated).

---

## A. Fix in the draft now

1. **STALE: the `fig-joint-axes` caption** (`_04`). It describes the old units:
   "weight", "share = weight squared", and "gray is unexplained". The figure now
   shows percent of each channel's item variance. There are two grays, and
   Judge carries a *. Suggested caption:

   > Shared axes across the five channels (joint diagonalization of the
   > level-removed matrices, K = 7). (a) Percent of each channel's item
   > variance on each shared axis (unit-diagonal convention, as in
   > @tbl-pc1-words). Axis names are our labels for the poles shown under
   > each name. (b) The same shares stacked. Light gray is variance in the
   > channel's own top-7 components that the shared axes miss, and dark gray
   > is variance beyond its own top 7 (item uniqueness, sampling error and
   > minor structure). *Judge is not a valid correlation matrix (negative
   > eigenvalues); its PC1 share matches that of its nearest correlation
   > matrix.

   The design-variations comment above the figure is correct as written.
   "λ1 normalization" is right: JD-l1 is the largest-|eigenvalue| = 1
   normalization. The `fig_joint_axes.py` docstring's "l1" is the misleading
   one.

2. **STALE: Higham citation** (`_03`, Judge method). The draft cites
   `highamComputingNearestSymmetric1988` (nearest symmetric PSD matrix). The
   code, `pkit.measures.nearest_corr`, is Higham **2002**: the nearest
   *correlation* matrix, with a unit diagonal, by alternating projections with
   Dykstra's correction. Cite 2002, or both if the PSD-only version is used
   anywhere. The draft also says "nearest PSD matrix"; it should say "nearest
   correlation matrix".

3. **PARTIAL: the level-removal paragraph** (`_04`, the "TODO: to put...
   somewhere" block). The numbers already in it are right: HUMAN 5% of item
   variance (.052), cos .995; models 38% (.380). The rest of the argument, from
   `channels.level_removal()`:

   | | HUMAN | SELF | JUDGE |
   |---|---|---|---|
   | method | row-center | row-center | double-center |
   | row-mean share of item variance | .052 | .380 | — |
   | mean off-diagonal, raw → level | .054 → .001 | .399 → .002 | .034 → .000 |
   | cos(raw PC1, all-ones) | .19 | **.97** | .38 |
   | cos(raw PC1, level PC1) | **.995** | **.06** | .92 |
   | PC1 share, raw → level | .159 → .164 | .420 → .452 | .204 → .178 |

   - **HUMAN:** harmless; PC1 doesn't move.
   - **SELF:** removes nonsense. Raw PC1 *is* the level direction (cos .97
     with uniform). The level-removed PC1 is nearly orthogonal to it (.06), and
     the near-equal shares (.42 vs .45) hide the swap.
   - **JUDGE:** part of the design. φ carries a level and isn't a correlation
     matrix.
   - **Don't** argue this with congruence (raw vs level-removed matrices
     correlate .97 / .92 / .88). Congruence can't see a uniform shift, so it
     looks reassuring for exactly the case (SELF) where PC1 changed identity.

4. **STALE: thinking-model handling.** The "Model Likert Readings" paragraph
   describes the 1024-token budget plus force-close. That is the SELF
   protocol. The JUDGE matrix runs pass `enable_thinking=False`, and for
   templates that open `<think>` automatically they close an empty thought.
   Say which instrument uses which. For **always-think** models (Glimmer, the
   R1 distills), an empty closed thought is off-policy (see D3). Check ENACT's
   handling before writing it down.

5. **PARTIAL: digit tokens** (`_03`, "combining '1' and ' 1'"). Add:
   SentencePiece tokenizers that split a bare digit (Yi encodes "1" as
   ["▁", "1"]) fall back to direct vocabulary lookup of "1" / "▁1"
   (`pkit.extraction._single_token_variants`). Every Likert readout uses this
   helper; the JUDGE matrix script now does too (f9dde3f).

6. **PARTIAL: what JUDGE records.** The draft says "we recorded the expected
   values". The 12-model cohort stored EV plus entropy per pair. The wide runs
   (from 2026-10-18) also store the full 1–7 distribution `P` and the
   pre-renormalization digit mass `Mass`. If the paper reports digit mass or
   distribution shape for JUDGE, it can only cover the wide models, plus a
   rerun of Qwen3-14B, which finished before the patch. Base-rate queries
   always stored full distributions.

7. **GAP: φ clipping.** The paper's JUDGE channel clips φ to [−1, 1]
   (`pkit.channels.PHI_CLIP = 1.0`) before double-centering. The draft says
   "we will mostly use J_φ" but doesn't mention the clip. Say how many cells
   it touches.

8. **PARTIAL: centering consistency** (`_04` TODO "ensure a consistent
   centering technique"). The adopted recipe (2026-09-22) is:
   - row-center the HUMAN and SELF design matrices (center-only ipsatization,
     then item r);
   - double-center JUDGE;
   - leave REPRESENT and ENACT alone. Their column-centered cosines are
     already centered, and matrix centering changes them by r ≥ .999.

   "Level & top removed" means that recipe, then the largest eigencomponent
   removed. The Represent/Enact method text already describes their centering.
   The HUMAN/SELF/JUDGE level step should be stated once in Analysis, not only
   in the results paragraph.

## B. Methods additions

1. **GAP: variance-share convention.** All PC1 shares and joint-axis shares
   are *percent of item variance with a unit diagonal*: q′(M₀ + I)q / n, so
   the denominator is the trace n. That makes the shares additive over
   orthonormal axes. Current PC1 shares: HUMAN 16.4, SELF 45.2, REPRESENT 8.7,
   JUDGE 17.8*, ENACT 29.6.
   - **JUDGE \* and the justification:** JUDGE has negative eigenvalues. Its
     Higham nearest correlation matrix has the same PC1 (cos .999) and the
     same unit-diagonal share (.178).
   - **Why not a zero diagonal:** the old zero-diagonal positive-part share
     (.26) is distorted by incoherence. A trace-zero matrix carries positive
     mass that matches its negative mass; the repaired value is .31, against
     HUMAN's zero-diagonal .305.
   - **Why keep the \*:** the negative space itself is unexpected and worth
     flagging (rgb). `C.NOT_PSD_NOTE` has the footnote text.

2. **PARTIAL: reliability** (the figure diagonals say "split-half
   reliability"). State the facet:
   - Halves split *respondents* (people for HUMAN, models for the rest), and
     we report the split-half correlation of the resulting cohort matrices,
     Spearman-Brown corrected. This is G-theory with respondents as the
     replication facet: it asks whether a different sample of models would
     give the same matrix.
   - **For statisfactions' alpha/ω question:** alpha over respondents is the
     analog, and by Flanagan–Rulon the mean split-half equals alpha. Option:
     report alpha (or the mean over many random splits) instead of one split.
     ω doesn't apply directly, because there's no single-factor model of
     "respondents measuring the matrix".
   - Legacy residual split-half (44-block, 2026-09-22): HUMAN .96, SELF .75,
     REPRESENT .98, JUDGE .98, ENACT .98. Disattenuation moves no residual
     congruence by more than about .1 (SELF–HUMAN .56 → .66). Regenerate
     these numbers from the current code before quoting them.

3. **GAP: Mantel p.** Every pair sits at the permutation floor. Report once:
   "all pairwise Mantel p ≤ 1/(perms+1)". The p is uninformative beyond "not
   chance"; let the effect sizes and reliabilities carry the weight. The
   draft's "p < 0.001" is fine at 1000 permutations.

4. **GAP: joint diagonalization (common axes).** Methods paragraph, from
   `pkit/jointdiag.py`:
   - **Model:** S_t ≈ Q diag(w_t) Qᵀ with orthonormal Q (525 × K), fitted by
     off-diagonal least squares on each channel's level-removed matrix, each
     scaled to unit off-diagonal Frobenius norm. No top removal: evaluation is
     axis 1.
   - **Fitting:** diagonals are imputed from the model each step
     (MINRES-style), and channel weights come from an exact off-diagonal
     regression.
   - **Optimizer:** Q is updated by a majorize–minimize polar step, which
     never increases the loss.
   - **Convergence:** multi-start, up to 5000 iterations. For K = 7, two
     starts reached the same optimum (loss 1.13408).
   - **K = 7** by rgb's choice. Openness doesn't appear by K = 10 either (C4).
   - **Lineage:** INDSCAL (Carroll & Chang); common principal components
     (Flury); DISTATIS (Abdi et al.). Exact joint orthogonal
     diagonalizability holds iff the matrices commute.
   - **Display units:** per-axis display uses B1's item-variance units. The
     fit's own weights give shares of *squared* off-diagonal structure
     (~λ²/Σλ²), which inflates the dominant axis: 70% for HUMAN axis 1, against
     15.6% of item variance. Don't display w².

5. **GAP: robustness of the axes** (appendix, or extend the existing
   comment). See `scripts/joint_diag_robustness.py` and
   `joint_diag_robustness.json`.
   - **Recover axes 1, 3, 4, 5 (|cos| ≥ .92):** DISTATIS (λ1 or unit-norm
     tables), eigenvectors of the mean matrix, λ1 normalization, and
     K = 6/8/10.
   - **DISTATIS is nearly the plain mean:** its RV weights are .18–.21 against
     a uniform .20.
   - **Averaging methods lose the REPRESENT-only axes 6 and 7.**
   - **Summed-squares eigenvectors mix axes 3 and 4.**

   Registered predictions Q1–Q6 were graded (Q6 missed except for SUMSQ). The
   ledger is in to_try if a "registered vs exploratory" note is wanted.

6. **GAP: centering alternatives** (appendix). Under the *legacy* grand-mean
   convention, three centerings were compared (`scripts/centering_check.py`):
   - With no centering, SELF's top eigenvector is uniform (a positive
     manifold), so removing it leaves SELF's evaluation axis in.
   - Double centering raises every residual congruence.
   - **Robust ordering:** JUDGE > ENACT > {SELF, REPRESENT} survives all
     three.
   - **Not robust:** the SELF vs REPRESENT order is not; don't lean on it.

   Re-run under the adopted recipe before citing.

## C. Results additions

1. **GAP: shared axes, K = 6** (with the figure; was K = 7 until 2026-10-24,
   when axis 7 failed the half-split replication check, median |cos| .54,
   `scripts/joint_diag_noise.py`. K = 6 shares, shared / own top 6: HUMAN
   27.0 / 32.8, SELF 51.6 / 75.0, REPRESENT 24.8 / 28.5, JUDGE 33.1 / 39.9,
   ENACT 52.3 / 62.3. The K = 7 numbers below are kept for the record.) Axis names are labels for
   18-word poles, not fitted quantities.
   - **The axes:** Evaluation; Generic vs. behavioral (renamed 2026-10-24, see
     G); Bold vs. ordinary;
     Serious vs. playful; Sensitive vs. shallow; Compound vs. simple words;
     Physical vs. character.
   - **Axis 1 (percent of item variance):** HUMAN 15.6, SELF 41.3,
     REPRESENT 7.9, JUDGE 16.7*, ENACT 26.1.
   - **Shared axes vs each channel's own top 7:**

     | | Shared | Own top 7 |
     |---|---|---|
     | HUMAN | 27.5 | 34.2 |
     | SELF | 56.4 | 77.3 |
     | REPRESENT | 27.5 | 30.9 |
     | JUDGE | 33.7 | 42.0 |
     | ENACT | 54.2 | 64.8 |

     The shared axes get each channel most of the way to its own best K = 7.

   This revisits the earlier story that "after PC1 everything diverges". Bold
   vs. ordinary (agency/extraversion) is shared by HUMAN, JUDGE and ENACT, and
   serious vs. playful by ENACT and JUDGE.

2. **GAP: REPRESENT's private axes are lexical** (2 and 6 at K = 6: generic
   vs. behavioral words, compound vs. simple words). These are word-type
   distinctions in the representation that no person-description channel
   carries. This connects to the distributional-hypothesis /
   embedding-regress story (W16). Do NOT tie axis 2 to Peabody/Saucier
   evaluative-vs-descriptive (see G).

3. **GAP, exploratory: SELF and the serious vs. playful axis.** SELF has
   almost no variance on axis 4.
   - **Range restriction:** the axis's top-40 words have median between-model
     SD .415 (28th percentile of all 525 words), and a high mean
     self-rating (+0.42 row-centered). Every model calls itself both
     professional and cheerful.
   - **Reading:** tone is register built into the assistant, not an
     individual difference between models.
   - **Link:** this axis matches "tone" in the TIDE schema (W19).

   Unregistered. The test would be persona-conditioned SELF arms. This belongs
   in the discussion, marked exploratory.

4. **GAP: Openness has no axis of its own.** Through K = 10, the Big Five
   Openness marker direction sits in each channel's top-10 span about as much
   as E/C/N do (O .05–.17). It isn't aligned with any variance-ordered axis.
   This matters for comparisons with lexical Big Five work. Exploratory.

5. **GAP, legacy: second components.** Under grand-mean centering, the
   HUMAN–JUDGE second-component |cos| was .84, and everything else ≤ .40.
   Judgment shares the human structure's second axis as well as its first.
   That second axis is human PC2, the self-presentation stance absent from
   every model channel (W18 §6). Recheck under the recipe. The joint-axes
   figure may now carry this point instead (axis 3 shared by HUMAN and
   JUDGE).

6. **SELF's top axis is amplitude, not disagreement (rgb's rank-2 model,
   2026-11-07).** Model: x_m = a_m e + b_m o, with every model sharing the
   positive self-image e and differing in dynamic range a_m. Then item
   covariance across models ~ Var(a) ee' + Var(b) oo': a giant evaluative
   axis from range differences alone, which per-model normalization removes.
   Predictions were registered before looking; all three hit. On the 263
   items with Hampson SDY:

   |                                         | SELF (49) | HUMAN (700) |
   |-----------------------------------------|-----------|-------------|
   | profile r with SDY, median [min]        | .90 [.75] | .76 [-.78]  |
   | members with r >= .7                    | 100%      | 65%         |
   | r(PC1 score, SDY amplitude)             | .998      | .996        |
   | SDY share, row-centered -> row z-scored | 43% -> 2.1% | 16.5% -> 7.8% |
   | top axis abs r with SDY, row z-scored   | .19       | .85         |

   Every model points the same way and differs in volume; people differ in
   direction (some self-describe undesirably, which no model does). The same
   first axis is self-evaluation in HUMAN and expressiveness in SELF, like
   extreme response style in human surveys. Explains "elevation unmasked"
   and the SELF .82 -> .21 drop under top-component removal (W18). Caveats:
   amplitude could be confidence or calibration, not expressiveness (the
   digit entropies could separate them); the numbers are 263-item
   correlations, not the joint-axis units (SELF 51.6% on axis 1).

7. **The level is desirability in every channel; good is more alike than
   bad (2026-11-07).** Cohort-mean profile per adjective: endorsement for
   HUMAN/SELF (after elevation removal), mean similarity to the other 524
   ("centrality") for REPRESENT/JUDGE/ENACT. r with Hampson SDY: HUMAN .94,
   SELF .94, ENACT .90, JUDGE .82, REPRESENT .61; word frequency (zipf) .04-.29.
   Predictions graded: HUMAN~SELF >= .8 HIT (.94); JUDGE .6-.8 near miss
   (.82, high side); REPRESENT |r| < .3 and frequency-driven MISS (.61, freq
   .04); ENACT in between MISS (.90, highest of the similarity channels).
   Mechanism (SDY halves, mean similarity):

   |           | pos-pos | neg-neg | pos-neg |
   |-----------|---------|---------|---------|
   | HUMAN     | .215    | .143    | -.076   |
   | SELF      | .658    | .627    | .167    |
   | REPRESENT | .103    | .073    | -.044   |
   | JUDGE     | .288    | .063    | -.100   |
   | ENACT     | .266    | .158    | -.185   |

   Desirable adjectives are mutually closer than undesirable ones in all five
   (JUDGE most, 4.6x): the density hypothesis of social cognition (positive
   information is more alike; Unkelbach et al. 2008; Alves, Koch & Unkelbach
   2017; citations NOT yet verified). A psych-side "mirror" result: models
   reproduce a human cognitive asymmetry in how evaluative words are
   organized. Least-central items: JUDGE = transient states (exhausted,
   tired, sleepy, scared) -> trait/state split; REPRESENT = health and
   disability words (handicapped, disabled, sickly). Caveat: JUDGE centrality
   includes the model's scale use (generosity).

## D. Pending before the numbers freeze

1. **WAIT: ENACT n = 10 → 20.** The 10 full-recipe pda_wide models finished
   2026-10-16 but aren't in `enact_members`.
   - Build their mid-layer files.
   - Recompute congruences, the PC1 table, split-half, joint axes and
     off-diagonal strength at n = 10 vs 20.
   - List the outlier models (per-model congruence with the cohort mean, and
     per-model residual vs HUMAN) before deciding. The draft's `n_small` text
     ("smaller cohort ... due to cost") will change.

2. **WAIT: JUDGE wide.** Running now, split with statisfactions (he goes
   smallest-up, we go largest-down over the 56-model manifest). About 8–9 h
   per 32B model at 9 pairs/s. `pkit.channels.JUDGE_ROSTER` pins the paper
   channel to the 12 until a deliberate fold-in, with the same n-comparison
   and outlier step as D1.

3. **WAIT: always-think models under JUDGE** (Glimmer, the R1 distills). An
   empty closed thought is off-policy for them. Decide on inclusion or a flag
   before fold-in. The SELF loader already prefers their `_think` files.

4. **Qwen3-14B JUDGE lacks digit distributions.** A 4.7 h rerun, only if
   digit-level JUDGE results go in the paper.

5. **Open checks** (optional; each would firm up a sentence):
   - pairwise commutators ‖[S_a, S_b]‖, for how close the channels come to
     exact joint diagonalizability;
   - split-half intervals on the joint-axis shares;
   - alpha for the cohort-mean channels (B2).

## E. Limitations / framing reminders

- Model counts overstate independence (families). The draft says this for
  SELF; it applies to every model channel, and more so at n = 12 / 10.
- Axis names, the tone reading (C3) and Openness (C4) are interpretive and
  unregistered. Keep them out of the abstract.
- Which numbers rest on 44 clusters and which on 525 adjectives. The block
  level is for display; the analysis is per adjective. The draft says so for
  the figures. Make sure any quoted congruence says which level it is
  (525-level values are lower: legacy raw .50–.77, top-removed .17–.58).
- Human data note (in the draft): the transposed "insensitive" /
  "inspirational" columns were swapped back.
- Digit mass (2026-11-01). Six of the 49 SELF models put ~0 next-token
  probability on the digits at the read position (glm-4, gemma-7b,
  Phi-3-medium, Phi-3.5-mini, Mistral-7B v0.1/v0.3; they open with "I would
  rate..."), so their SELF readout is the conditional "if answering with a
  digit now". It behaves like everyone else's: profile r .87-.96 with the
  other 43 (their median .94), and dropping all six leaves the SELF item
  matrix at r = .997. One methods sentence plus this check. Source:
  results/adjectives/digit_mass_audit.json (run 2026-08-31, ungraded until
  now). JUDGE-wide has the same problem worse (28/53 below 50% mass); its
  inclusion waits on scripts/judge_mass_faithfulness.py.

---

## F. rgb's results outline, cross-referenced (2026-10-18)

rgb's outline, with the evidence on file under each bullet. This matches
the structure adopted 2026-09-12: five channel sections, then one
cross-channel machinery section after Enact. Sections A–C above mostly
belong in that machinery section: congruence, PC1 table, joint axes.

**Standing reminder (rgb):** split-half, or another attempt, to separate a
*playacting* direction from the *assistant* direction in ENACT. Design note
before anyone builds it: as the draft defines it,
e_assistant = μ_default − E_a[μ_a]. That is exactly minus the naive
"persona-on vs persona-off" direction, E_a[μ_a] − μ_default, and the
adjective-mean centering removes that same direction from every ENACT
vector. Separating the two needs a third reference that isn't an adjective
mean. Two options:
- a neutral persona instruction ("respond as yourself" / "as an assistant"),
  which isolates the cost of being told to play a role;
- a split over adjectives or questions, which tests whether the shared
  offset is stable.

The ENACT displacement gain component (congruence .89, uncorrelated with the
SELF gain at .15; to_try 2026-10-10 SYNTHESIS) is the nearest existing
handle. Status: **not done**.

### Human
- **Answer distribution, elevation and scatter.** Planned in the draft
  ("baseline levels, extremes avoidance and digit-based biases"). No
  paper-ready numbers collected yet.
- **"Lizardman" item → "incidence floor"** (or "baseline endorsement floor"),
  the respectable name for the floor of endorsement on items nobody should
  claim. **GAP**: the number isn't on file here.
- **Raw PCs; Big Five replication.** Draft plans scree, split-adjective
  stability, and varimax vs unrotated loadings (Saucier 1997). W14 covers the
  over-extraction diagnostics (no stable sixth factor).
- **Perils of ipsatization; ipsatized E vs raw E.** On file:
  - ipsatize = within-person z; C&C ipsatize both sides (memory
    project_adjective_data_ipsative);
  - level is small and surgency-shaped in HUMAN, so row-centering barely
    moves PC1 (cos .995, A3), but the level it removes leans toward E;
  - ipsatized participation ratio: SELF 11.6 vs HUMAN 50.2.

  **GAP**: the E-vs-E comparison itself.

### Self
- **Answer distribution; raw / ipsatized PCs.** The level-removal table (A3)
  shows the raw SELF PC1 is level (cos .97 with uniform). PC1 share after
  level removal is 45%. Horn k goes 4 → 5 after row-centering.
- **Desirability, with an external measure** (to_try 2026-10-11; Hampson
  1987 SDY, 263 terms).
  - Every channel's PC1 is desirability at r ≈ .9 (SELF .94, HUMAN .95).
  - SELF's distinction is gain, not direction: PC1 share .49 vs .31 in
    zero-diagonal units. Use the 45% vs 16% unit-diagonal figures in the
    paper. Cite "same shape, much more gain".
  - Registered predictions: SELF > HUMAN missed.
- **Projection into the human Big Five.** **GAP** in this file; the SELF
  population slides have it.
- **Self-perception and history.** W20 thread: update rate is a family
  parameter set by post-training, and it predicts failure-distress
  escalation (r = +.93, n = 8). That is the one cross-channel
  individual-difference link that holds (two gains of the same kind). C3
  (no tone variance) fits here too.

### Represent
- **Lexical hypothesis, strong vs weak.** Draft intro has it (Cutler,
  Milano, Wulff).
- **Centering.** In the draft (adjective-mean centering; massive-dimension
  exclusion, robust at r ≥ .999).
- **PCs.** Lowest PC1 share (8.7%). The private axes are lexical (C2).
- **Human match.**
  - Level-removed congruence at 525 adjectives: .593; level and top removed:
    .335 (n = 49, after the roster fix).
  - Human structure is decodable from REPRESENT: mapped r .78–.81, about 2×
    beyond PC1 (W18 §6).
- **GloVe / embedding vs superposition.** W16 regress: the evaluative-antonym
  merge goes back to static GloVe; it's the distributional hypothesis, not
  the architecture. The affect-presence axis is a separable lexical axis.
  The superposition-vs-embedding question is open (rgb's stance in memory).
- **"(not done yet) Represent predicts self?" Already run** (2026-09-12,
  0c2f03a, `scripts/represent_self_prediction.py`):
  - The population-level coupling was a degradation artifact: Mantel .47 →
    .03 core-only.
  - A small incremental effect survives (p = .026).
  - The run used the wrong n (silent roster drop, flagged in to_try at the REPRESENT n = 49 fix, not
    rerun). **Rerun at n = 49 before citing.**

### Judge
- **Reprise.** Conditional probabilities → φ → correlation-like matrix. In
  the draft; A2, A6, A7 and B1 refine it (Higham 2002, digit distributions,
  φ clip, the * convention).
- **PCs.** W16 judgment factor structure: tom_likely varimax is clean, with a
  human-like general factor (PC1/PC2 4.8 vs HUMAN 5.2 vs REPRESENT 1.4). It
  **recovers Openness, which REPRESENT loses**. Reconcile with C4 (no
  Openness axis in the joint variance-ordered fit). These can both be true
  (rotated single-channel factors vs shared unrotated axes), but say so in
  one sentence.
- **Human match.** JUDGE–HUMAN is the highest residual congruence (.775 at
  the 44-block level under the recipe). It also shares the human second axis
  (C5).
- **Asymmetries / reluctance to generalize.** Three pieces on file:
  - the W16 directional asymmetry, which has valence and variance accounts,
    not disentangled;
  - W18 disbelieved clusters: the model treats valence as an axis, not a
    binder, and refuses human negative-halo bundles. That is the
    "reluctance";
  - the JUDGE sink flips from negative states to HHH positives with scale.
    to_try queued this as "one sentence + the 12-model table" in the
    base-rate paragraph.

  The JUDGE decomposition (S+/S−/gradient/curl, Tversky reframe) is the
  fuller version; a human-judgment baseline is still missing.

### Enact
- **Playacting / persona selection model (PSM).** Evidence on file:
  - the W18 §5 J-lens on Qwen7: ENACT reads out the persona's
    speech-world, and failed personas fall into assistant or wacky basins.
    That is mild PSM-flavored evidence;
  - Phi-3-medium is a non-enactor;
  - W17: ENACT is a low-rank linear image of REPRESENT (R² .72).
- **"(not done yet)" explicit playacting-awareness poke.** Correct, not done.
  It was gap (c) on 2026-09-12: a small GPU instrument waiting on rgb's
  design pass.
- **Centering.** See the standing reminder above: the centering *is* the
  playacting/assistant confound.
- **PCs, compared with SELF.** These two are the concentrated channels:
  - residual participation ratio about 10 each, vs 22–23 for
    HUMAN/REPRESENT/JUDGE;
  - joint axis 1: ENACT 26.1, SELF 41.3;
  - shared-axis totals: ENACT 54.2, SELF 56.4.

  The difference: ENACT keeps human-shared axes 3 and 4 (bold, serious vs
  playful); SELF is nearly all axis 1 and has no tone variance (C3).
- **Human match.** .701 residual at the 44-block level. WAIT on n = 20 (D1).
- **"(not done yet) Self predicts Enact distance." Already run, three
  times, all null for individual differences:**
  1. SELF↔ENACT second-order RSA, Mantel r = −.045, p = .86 (n = 10). rgb's
     "probably not" was registered and HIT.
  2. W17 §15 diagonal test: +.02 increment.
  3. P24 medoid arm (n = 40): own SELF −.76 vs cohort-mean SELF −.83 at
     predicting displacement; deviation-on-deviation +.03 against a .84
     reliability ceiling (2026-10-06 downgrade).

  The positive half is citable. The *shared* self-image predicts the
  *shared* displacement profile (r −.91, −.61 beyond desirability):
  "self-report is right about the role and silent about the actor".

---

## G. Support for calling the top component desirability (2026-10-24)

Desirability is contentious, so the claim needs external support, and we
have it. The check uses independent norms: Hampson, Goldberg & John (1987),
British social-desirability ratings (55 raters, α .99). They are in
`data/hampson_1987/sdy_british_pda525.csv`, extracted by
`scripts/hampson_sdy_extract.py` from `Category-breadth.pdf`, which is that
paper. They cover 263 of the 525 adjectives. Numbers recomputed 2026-10-24
on the current matrices, with 95% word-bootstrap CIs:

| | r with desirability norms |
|---|---|
| HUMAN top component | .95 [.94, .96] |
| SELF | .94 [.92, .95] |
| JUDGE | .94 [.93, .95] |
| REPRESENT | .91 [.89, .93] |
| ENACT | .89 [.86, .91] |
| Joint axis 1 (K = 6) | **.97 [.96, .98]** |

The other five joint axes are all |r| < .1 with signed desirability.

**What this supports.**
- Removing the top component removes the same externally identified
  dimension from all five channels. That justifies the operation, and the
  "evaluation removed" label.
- SELF is distinctive in *gain*, not direction. Its top component has the
  same shape as everyone else's (.94 vs HUMAN .95, difference −.01, CI
  includes 0), but it takes 45% of item variance against HUMAN's 16%. Cite
  "same shape, much more gain", not the matrix-level r² difference (+.04),
  which is compressed near ceiling (to_try 2026-10-11 correction).

**Framing to pre-empt the contention.**
- Say *item* desirability (a property of the word, rated by independent
  judges), and keep it distinct from *socially desirable responding* (a
  respondent bias). The axis is the former. Whether SELF's large share
  reflects SDR-like responding is a separate claim; the gain/conception
  question is filed as unidentified from self-ratings alone (to_try
  2026-10-11).
- Call the axis "evaluation" in the running text and cite the norms
  correlation once. This is the classic evaluative factor of trait words
  (Peabody's evaluative vs descriptive content; the general evaluative
  factor in Saucier 1997 and in the lexical tradition), so the name has
  lineage beyond our data.

**Caveats to state.**
- Coverage is half the words. The overlap is Hampson's list, which was
  built to balance desirable and undesirable terms. Overlap words load
  slightly more strongly on axis 1 than the rest (mean |loading| .042 vs
  .038), so the r may be a little optimistic for the full 525.
- British raters from 1987 vs the US Eugene-Springfield sample. That gap
  would lower the correlation, not raise it.
- Registered predictions from 10-11, graded: SELF > HUMAN missed; levels
  were all predicted too low.

**Terminology (2026-10-24, rgb, from Saucier 1994's abstract).** In
Saucier's usage, "evaluation vs. description" means separating the
evaluative component from the descriptive dimensions. Axis 1 is that
evaluative component (.97 with the norms), and axes 2-6 are all
descriptive (each |r| < .1 with desirability). So no other axis may be
named "evaluative vs. descriptive". Axis 2 was renamed "Generic vs.
behavioral": good/bad/normal/average/brilliant vs
adventurous/argumentative/impulsive/assertive. It tracks evaluative
extremity only weakly (.18). "Broad vs. narrow" was avoided because it
would claim Hampson category breadth, which is untested.

Framing gift: Saucier imposed the evaluative axis and examined the rest.
The joint fit imposes nothing, yet axis 1 lands at .97 with independent
desirability norms and the remaining axes come out orthogonal to it. That
is his separation, recovered unprompted. We have only the abstract
(neither the PDF nor a Zotero abstract), so keep any comparison of
methods at the level of "imposed vs. emerged".
