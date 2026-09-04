<!-- Captions for the wide-cohort facet deck (facet_slides_wide.py).
     Edit this file, then rebuild:
         .venv/bin/python scripts/facet_slides_wide.py
     Format: "# KEY" per slide; bold line = title; rest = subtitle.
     {n_self} / {n_rep} are filled in from the data at build time. -->

# HUMAN
**Personality measurement is a theory of human variance**

Saucier 525-PDA: 700 human self-ratings on 523 adjectives, aggregated
to 44 trait blocks (pole-respecting Ward on the human correlations;
coherence-filtered, no size cap — the large blocks are the trait
majors: warmth, anxiety, anger, honesty). This block structure is what
the Big Five compress.

# SELF
**Ask models the same questions: the population is tiny — and not
human-shaped**

Same construction with models as respondents: n = {n_self} deployed
instruct models (wide-n cohort + standing; 6 framings averaged) — the
correlation estimate is now full-rank at block level. Raw congruence
is a desirability freebie; the top-component-removed number is the
honest one.

# REPRESENT
**Models represent something richer — but human-shaped mostly in the
top component**

Residual-stream cosine between adjective activations (pers framing,
mid layer), {n_rep}-model cohort mean (wide-n capture). More structure
than SELF, but beyond the shared evaluative axis it organizes traits
its own way.

# JUDGE
**Asked about trait relations, models reconstruct human structure**

“How likely is a {X} person to be {Y}?” — EV over the answer
distribution, symmetrized, 10-model cohort mean. Explicit judgment
recovers the human covariance the other channels lose — including
with the top component removed.  [cohort-12 channel — not wide-n
captured]

# ENACT
**But models play the traits out at lower fidelity**

Cosine between persona-vector directions extracted from each model's
own trait-enacting rollouts, 10-model cohort mean. Enactment keeps
some human structure — less than judgment. Part is difficulty of
enacting; part may be intrinsic.

# SUMMARY
**Remove each space's top component: judgment reconstructs human
structure; enactment trails; self-report and representation weakest**

All five 44-block grids with the top eigencomponent of the underlying
523² matrix removed; bars = congruence with HUMAN (off-diagonal r).

Aggregation-free check — full-resolution 523² item level, top
component removed: SELF 0.20, REPRESENT 0.31, JUDGE 0.54, ENACT 0.49.
Same ranking; blocks are visualization, not load-bearing.
