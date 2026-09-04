<!-- Captions for the wide-cohort facet deck (facet_slides_wide.py).
     Edit this file, then rebuild:
         .venv/bin/python scripts/facet_slides_wide.py
     Format: "# KEY" per slide; bold line = title; rest = subtitle.
     {n_self} / {n_rep} are filled in from the data at build time. -->

# HUMAN
**Measured Human Self-Report Correlations**

Correlations from Saucier's 525-PDA.  For visualization, these are clustered
by a signed version of Ward's minimum variance (so antonyms stay separate).
Cell value is the mean of the pairwise correlations, with cluster cohesion on
the diagonal.

# SELF
**The Same for Models**

n = {n_self} small instruct models, mean value across 6 question framings.
Correlations drop the scale here, but model's absolute variance is much
smaller.  It's also easy to see the desirability block washes out everything
else from this view.

# REPRESENT

**Models' Residual-stream Represents are More Complex**

Gathered as midlayer residual-stream states on adjective input tokens in
a personality frame, recentered on all adjectives.  Mean cosine similarity
across attributes in the clusters across models is shown below.
Note the strong diagonal (cluster cohesion), but the off-diagonal
quite different from human.

# JUDGE

**Models' Understandings of Trait Relations**

We directly ask the models "How likely is a very {X} person to be {Y}?"  And
the models step up: this is the best match of human behavior.  Here we
zscore the entries and symmetrize.

# ENACT

**But, Asked to Enact this Understanding, Models Fall Short**

This time we ask models to enact the given attributes when answering
user questions.  We again gather mid-layer activations, and recenter
across attributes, to come up with enacted-persona vectors.  Again,
cosine similarity, meaned across models, is shown below.  These models,
at least, seem to have trouble being funny without being annoying.

# SUMMARY

**Correspondence with Humans is Often Shallow**

If you remove each instrument's top principal component, the visual shapes 
become notably different, and the congruence with human falls notably,
except for JUDGE.
