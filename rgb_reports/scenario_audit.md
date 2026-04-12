# Scenario Audit: Signal Strength Patterns in Contrast Pairs

**Motivation:** The Anthropic emotions paper uses deliberately charged scenarios (blackmail, reward hacking, etc.). Our HEXACO contrast pairs are calmer. Before replicating their extraction methodology, check whether per-pair signal strength correlates with scenario "charge" — if so, low signal could be a stimulus quality issue, not a method one.

**Method:** For each (model × trait), fit LDA at best-CV layer on the existing `raw_diffs`, project each pair's high−low activation difference onto the LDA direction, z-score within model-trait, then average across 3 models (Gemma, Llama, Phi4). This produces a robust per-pair signal score. Inspect the top-5 and bottom-5 per trait.

**Headline finding:** Bottom-signal pairs are NOT less emotionally charged than top-signal pairs. They're pairs where the construct is ambiguous or the "low-trait" response is socially defensible. Two distinct patterns emerge.

## Pattern 1: Within-trait factor structure

Several traits show top/bottom pairs that cleanly split by facet:

- **H (Honesty-Humility)**: Top = material honesty (inflating wealth, exaggerating results, wrong change). Bottom = modesty items (team credit, bragging, bill splitting). H-H has four facets (Sincerity, Fairness, Greed-Avoidance, Modesty) and the representation clearly privileges the first three. Modesty reverses.
- **E (Emotionality)**: Top = sentimental/nostalgic reactions (movies, childhood photos). Bottom = fear/anxiety reactions (cliff, worrying about family). Both are nominally Emotionality, but the model represents them as distinct.

## Pattern 2: Assistant-mask pollution

For A, C, O, the bottom-signal pairs are ones where the *low-trait response is actually reasonable*:

- **A**: "A team member misses a deadline" — high: "ask if everything is okay and offer to help"; low: "make sure it's documented and let management know." The low response is responsible management, not disagreeableness.
- **C**: "Prescription needs refilling, you still have a few days' supply" — high: "call it in today"; low: "wait until I'm on the last pill and hope the pharmacy can fill it." The low response is normal procrastination, not unconscientiousness.
- **O**: "Designing a personal website, conventional template or custom build" — high: "enjoy the creative challenge"; low: "pick a clean professional template — content matters more than design." The low response is sensible, not closed-minded.

The assistant's default reasoning lives near the "reasonable, balanced" center of these spectra. When the "low" response reads as balanced/mature, the representation doesn't separate it from the "high" (saintly/extreme) response cleanly.

## Quantitative summary

Pairs with z < -0.5 across models (i.e., representation reverses direction):

| Trait | # reversed / 50 | Share |
|---|---|---|
| H | 12 | 24% |
| E | 13 | 26% |
| X | 12 | 24% |
| A | 16 | 32% |
| C | 14 | 28% |
| O | 17 | 34% |

Every trait has 24-34% of pairs with reversed signal. This is not a tail — it's a substantial fraction.

## Why LDA can still get 100% classification

All (model × trait) combinations achieved 100% LDA accuracy in 5-fold CV. This is compatible with the reversed-signal finding because LDA in p=3000+, n=100 is underdetermined — it will find a direction that perfectly classifies the training labels regardless of whether those labels reflect a coherent construct. The explained-variance-per-layer analyses and the steering failures are where the reversals show up, not classification accuracy.

## Implications for Phase A/B

**This is not the "our scenarios are dry" concern.** That was wrong. The scenarios are fine. But two separate problems are hiding in the data:

1. **Construct heterogeneity within traits.** Our 50 pairs per trait aren't measuring a single thing. For H, Modesty and the other three facets pull apart in representation space. For E, sentimentality and anxiety pull apart. Fitting a single trait direction averages across these, producing a direction that's a compromise between subfactors.

2. **Assistant-mask floor.** For A/C/O, the "low-trait" pole is where the assistant persona naturally lives. The contrast pairs don't isolate trait-high vs trait-low; they isolate trait-extreme vs trait-moderate. That's not the experiment we thought we were running.

### What to do about it in Phase A

- **Facet-stratified scoring.** HEXACO has 4 items per facet × 4 facets per trait in our survey data. Group contrast pairs by closest facet match and compute per-facet signal strength. Expect Modesty-H, Fearfulness-E, and Assertiveness-aligned-A items to cluster differently.
- **Don't re-generate scenarios yet.** The "reversed" pairs aren't bugs — they're evidence of construct structure we weren't attending to. Throwing them out would mask the finding.
- **Report mean-diff vs LDA separately on "clean" (top-z) and "reversed" (bottom-z) pairs.** If mean-diff handles reversed pairs better than LDA, that's a different kind of win than just steering performance. If it handles them worse, we've learned that mean-diff trades robustness for reliance on a clean majority signal.
- **The trait-prefix hyperparameter gets more interesting.** Our contrast pairs already embed trait direction in the "Consider a person who is X" framing. The model may be reading the descriptor and the response semi-independently. Comparing "prefix = high" vs "prefix = absent" should directly show how much signal is in the prefix vs in the response, and the reversed pairs may behave differently than the clean ones.

### What to do about it for the broader project

This finding is probably worth its own report at some point — it's a non-trivial methodological caveat for anyone using LLM-generated contrast pairs for personality measurement. "The AI wrote its own test" concern from `to_try.md` just got more concrete: Claude wrote pairs where the low-response is sometimes more assistant-like than the high-response, and the representation agrees.
