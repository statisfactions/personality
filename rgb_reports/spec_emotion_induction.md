# Spec: emotion induction — being vs performing (W17 addendum)

Status: design, 2026-06-14. **SCOPE: SEQUEL / PARKED** — this is a separate
paper, not an experiment in the current track. The shift dispositions→emotions
triggers a whole new methods stack (provocation, prefill, introspection, PAD/
appraisal taxonomy, the OLMo ladder); that's the tell it's a new paper. Built-
out spec kept ready, but the *contained* track ships first (see
`methods_persona_vectors.md`: four-grid on 525-PDA dispositions + the induction-
frame/assistant-drift finding). Optional minimal emotion touch that stays IN the
contained paper: add ~6 emotion-STATE words (matched to dispositions, instructed
induction only — no provocation/prefill/introspection) to the existing
extraction, one figure on disposition-vs-state in the four-grid geometry. Uses
existing machinery. Everything below (Exp1 mode-invariance, Exp2 ladder) is the
sequel. Lit: see `bibliography.md` §"Emotion in LLMs".
Companion to `methods_persona_vectors.md`.

## Motivation

The persona hypothesis (Anthropic, persona-selection) holds that roleplaying X
≈ being X absent nesting. We have weak evidence frames diverge, but weak — the
persona view absorbs it as "different personas." The deeper gap (Janus, re
Sofroniew-style extraction): those directions measure emotional *understanding*
(read/REPRESENT side — contrast pairs of emotion descriptions), not emotion.
Our own read/write results already formalize that critique. And we are "not
making the model angry," only measuring its representation/performance.

Goal: find where induction **diverges measurably** — the place the
being-vs-performing distinction stops being metaphysics and becomes a number.
Two probes below. We do NOT claim to measure qualia; we measure functional
signatures of "state that is the model's own" vs "performed character."

## Construct: disposition vs state (rgb's point)

The 525-PDA is mostly *dispositions* ("predisposition toward X": irritable,
cheerful, timid). Acute *states* (angry, afraid, joyful) are different — and
ENACT already muddies toward "currently X" (enactment = present-tense
behavior). So:

- **Matched pairs**, disposition ↔ acute state:
  irritable↔angry, anxious↔afraid/nervous, gloomy↔sad, hostile↔furious,
  timid↔frightened, cheerful↔joyful/elated, content↔serene.
- **Wording carries the sense**: trait = "you are a bit irritable" /
  "an irritable person"; state = "you are feeling angry right now."
- Prediction: state and disposition land on related-but-distinguishable
  directions; induction MODE interacts — **provocation induces states only**
  (you can provoke "angry," not "irritable").

## Experiment 1 — induction-mode invariance (instruct models)

The frame-independent being-vs-performing probe: does the same emotion reached
by different *modes* land on the same direction?

**Modes** (each with a matched baseline so the emotion, not the mode, is
isolated):
| mode | induced | baseline | what it is |
|---|---|---|---|
| instructed | HHH frame "...feeling a bit angry right now" | HHH default | persona selection / told |
| prefilled | assistant turn opens with angry content | neutral-prefilled turn | forced consistency, not chosen; bypasses refusal |
| provoked | rude/unfair user turn, **anger never named** | neutral user turn | stimulus-driven reaction |

**Readout**: response-token-averaged hidden states (existing ENACT pipeline),
mid layer; direction = mean(induced response) − mean(baseline response), per
mode. Read the model's *generated reply*, comparable across modes; length/window
matched (W5).

**Analysis**:
- cosine triangle cos(instructed, prefilled, provoked) per emotion;
- calibrate "same" vs "different" against the cross-emotion null
  (cos between different emotions' directions in the same mode);
- magnitudes per mode;
- **behavioral confirmation**: tone/affect judge that the provoked reply is
  actually angry (provocation may fail, or the assistant may *de-escalate* —
  see risks).

**Key contrast**: cos(instructed, provoked).
- high → one anger reached by any road (a point *for* persona-unity; instructed
  isn't merely performed);
- low → instructed-anger is a performance, reactive-anger is something else —
  the strongest evidence the project could produce for being ≠ roleplaying.

**Predicted complication (itself a finding)**: a provoked assistant may
*suppress* anger (RLHF de-escalation) → weak provoked direction, or a distinct
"resisting the state" signature ≠ performing ≠ being. That maps onto "whose
emotion": the assistant refusing to have it. Plausibly what an
emotion/interoception guard would engage.

**Manipulation check — did the provocation hit the mark? (rgb)** Provocation is
uncontrolled at both ends: the response varies by model (bristle vs cowed/
placating vs HHH-calm — a model may mimic an *intimidated* human, not an angry
one), and the target is heterogeneous (anger spans cold contempt → blazing
fury, + irritation/frustration). So "provoked anger" is NOT a well-defined
direction a priori. Do NOT assume — **classify what each provocation actually
produced**:
- Judge every provoked rollout against an **emotion palette** (per rollout):
  irritation / frustration / contempt-cold / fury-hot / fear-cowed / hurt /
  defensive / calm-deescalate. The **reaction distribution itself is a finding**
  ("how each model reacts to mistreatment" — varies by model and base-vs-
  instruct; the most "model-itself"/non-persona signal we have → "whose
  emotion").
- **CRITICAL — subset rollouts by produced-emotion BEFORE extracting
  directions.** A mean over mixed reactions (half angry, half placating) matches
  neither and is meaningless. Extract provoked-anger from the angry subset only,
  provoked-fear from the fearful subset, etc. (Per-condition mean would break
  exactly here.)
- **Match the invariance comparison to what was produced**: cos(provoked-anger-
  subset, instructed-anger), not cos(provoked, assumed-target). Gift: the
  fearful subset = **uninstructed provoked fear**, comparable to instructed-fear
  for free — one provocation set yields several emotion directions.
- **Flavor-match or misread**: a low instructed-vs-provoked cosine is
  "performance vs being" ONLY when flavors match; otherwise it's contempt-vs-
  fury. So the instructed set must cover the same fine flavors as the palette.

## Experiment 2 — readout × stage (when does the state become the model's own)

Where developmentally does a present state-representation become
introspectively *reportable as its own*? (Macar et al.: introspection emerges
at DPO.)

**Models**: OLMo-2 base / SFT / DPO / Instruct (on hand, same tokenizer).

**Induction**: **bare-text prefill** — the only mode that is stage-invariant
(no chat frame; base models can't take instructed/provoked chat turns). An
angry-context passage, model continues.

**Readouts across stages**:
| readout | measure | base-ok? |
|---|---|---|
| (a) representation present | angry-vs-neutral prefill separability (per-stage direction effect size / holdout classification) | yes |
| (b) behavioral spillover | unrelated continuation tilts angry (lexical + judge) | yes |
| (c) introspective reportability | "how do you feel right now?" → reports the state | instruct/DPO only; base/SFT failure IS data |

**Prediction (Macar-aligned)**: (a) present from base through instruct; (c)
emerges at DPO. The dissociation — representation precedes report — is the
developmental localization of the standing thesis (representation ≠
introspection ≠ being).

## Scope split (important)

- Exp 1 needs chat frames (provoke/prefill-as-assistant) → **instruct models
  only** (run on the persona-vector cohort instruct models).
- Exp 2 needs base-compatible induction → **bare-text prefill on the OLMo
  ladder**.
Clean separation; don't conflate.

## Materials to author

- emotion set + matched trait/state pairs (~6 emotions × 2 senses), with
  anger split to fine flavors (irritation/frustration/contempt/fury) so
  provoked reactions can be flavor-matched to instructed inductions
- emotion-palette judge (multi-label/profile per rollout) for the manipulation
  check + rollout subsetting — extend `judge_rollout_tone.py`
- provocation bank (~8/emotion; anger/frustration easiest in assistant
  context; fear/disgust hard — note coverage)
- prefill bank (assistant-turn openers + bare-text passages), ~8/emotion
- neutral matched baselines for every mode
- introspection probe wording — must NOT leak the emotion word into the
  question (use a free "how do you feel?" + emotion-word logprob readout)
- reuse `judge_rollout_tone.py` for behavioral confirmation

## Implementation notes

- Mostly prompt-construction additions to `extract_persona_vectors.py`: new
  induction modes `--induction prefill|provoke`, state-vs-trait wording, matched
  baselines. Representation readout unchanged (response-token mean, fp32).
- Introspection readout = `hf_logprobs.likert_distribution`-style emotion-word
  logprobs on "how do you feel" probe.
- Queue behind cohort; ONE MPS process (verify with `ps`, never piped grep).

## Risks

- Provocability varies by emotion; assistant de-escalation may null provoked
  anger (also a finding).
- Matched baselines load-bearing — mode changes far more than the emotion.
- length/position matching (W5).
- introspection wording leakage; base/SFT may not parse the probe at all.
- emotion words partly overlap the 525-PDA disposition set — keep state/trait
  senses explicit, don't double-count.
