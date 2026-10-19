# Notes that should probably go in the paper

Collected 2026-10-18 against the paper draft at that date (`_03-general-method.qmd`,
`_04-overall-results.qmd`). Each item says where it goes, what it says, where the
number comes from, and whether the draft has it already. Tags:

- **STALE**: the draft says something the current code contradicts. Fix first.
- **GAP**: not in the draft.
- **PARTIAL**: in the draft but incomplete.
- **WAIT**: depends on a run that hasn't finished. Don't freeze the number yet.

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

1. **GAP: shared axes, K = 7** (with the figure). Axis names are labels for
   18-word poles, not fitted quantities.
   - **The axes:** Evaluation; Evaluative vs. descriptive; Bold vs. ordinary;
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

2. **GAP: REPRESENT's private axes are lexical** (2, 6, 7: evaluative vs.
   descriptive words, compound vs. simple words, physical vs. character).
   These are word-type distinctions in the representation that no
   person-description channel carries. This connects to the
   distributional-hypothesis / embedding-regress story (W16) and to Peabody
   on evaluative vs. descriptive content.

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
