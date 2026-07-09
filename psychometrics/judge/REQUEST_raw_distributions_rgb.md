# Request to rgb: raw per-pair rating distributions for the JUDGE channel

**From:** ecb (JUDGE psychometric track, `psychometrics/judge/`) · **Date:** 2026-07-09

## The ask

The `judge_likert` release ships, per model, only the **mean** `B` and **entropy** `Hent`
of each pair's 1–7 rating distribution. For the response-process side of the psychometric
analysis I'd like the **full per-pair category distribution** — the 7-vector of
probabilities `d` over the rating tokens {1,…,7} that your generator already computes.

In `scripts/adjective_introspection.py` it's right there and then discarded:

```python
d, _, h = likert_distribution(...)   # d = the full 7-category distribution
...                                   # only entropy (Hent) and EV (B) are persisted
```

## Why it matters (what only the raw `d` can answer)

With just `(B, Hent)` I can't recover the response *shape*: an EV of 4 from "all mass on
4" vs "half on 1, half on 7" are opposite response processes with the same mean. Right now
I proxy shape with a mean-detrended **spread index** (entropy vs the min/max attainable at
that mean), but it assumes unimodality and can't see:

- **True modal-response frequencies** and real floor/ceiling category usage (not
  EV-rounded).
- **Bimodality / polarization** (the case the spread proxy is blind to).
- **Invalid-mass** structure (how much probability lands off the 1–7 tokens).

These are core to the "how does each model use the scale" analysis (Report I) and to
validating the spread proxy.

## Format options (either is fine)

1. **Full tensor** — per model, `dist (523, 523, 7)` float16 alongside the existing
   `B`/`Hent` in the same npz. ≈ 3.8 MB/model raw (~46 MB for 12; less gzipped). Simplest
   for me; re-run of the existing sweep with one extra `np.savez` field.
2. **Compact summary** — if the full tensor is unwelcome, per pair: `argmax` category,
   top-2 mass, `P(1)`, `P(7)`, and total invalid (off-token) mass. ~5 small arrays/model.
   Covers modal usage + floor/ceiling + a bimodality flag (top-2 mass split).

Option 1 is preferable (lets me do anything downstream); option 2 is the low-bandwidth
fallback.

## One data-quality flag while you're in there

I found that **~0.5% of judgments have `Hent` fractionally *below* the theoretical minimum
entropy for their `B`** (min over integer-support distributions with that mean). That's
impossible if `B` and `Hent` come from the *same* normalized 7-category distribution — it
suggests they're computed over slightly different supports (e.g. `B` over valid-renormalized
tokens, `Hent` including some invalid mass, or vice versa). Could you confirm the
normalization each uses? It's minor for current conclusions but I'd like to state it
correctly, and shipping `d` would let me reconcile it directly.

Thanks — happy to take whichever format is least effort on your end.
