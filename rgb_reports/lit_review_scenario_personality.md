# Literature Review: Scenario-Based Personality Measurement for Humans

*Prepared 2026-04-04. Covers instruments, theories, and validation evidence for measuring personality traits through behavioral scenarios rather than descriptive self-report.*

---

## Executive Summary

There is substantial precedent for scenario-based personality measurement in human psychometrics, but the literature is fragmented across several traditions (SJTs, conditional reasoning, implicit tests, gamified assessment). The key finding for our project: **a validated HEXACO scenario-based instrument does exist** (Oostrom et al., 2019), directly contradicting the assumption that this is an open gap. However, **trait-vs-trait conflict dilemmas for HEXACO** -- the Ultima IV-style forced-choice approach we are building -- have no validated human precedent and remain a genuine gap.

---

## 1. Situational Judgment Tests (SJTs) with Personality Scoring Keys

SJTs are the most developed approach to scenario-based personality measurement. The format presents a realistic scenario and asks respondents to evaluate or choose among behavioral response options.

### 1.1 Theoretical Framework: Implicit Trait Policies

**Motowidlo, S. J., Hooper, A. C., & Jackson, H. L. (2006).** Implicit policies about relations between personality traits and behavioral effectiveness in situational judgment items. *Journal of Applied Psychology*, 91, 749-761.

- Introduced the concept of **Implicit Trait Policies (ITPs)**: implicit beliefs about the effectiveness of trait-related behaviors in situations
- Tested whether personality predicts ITPs: supported for Agreeableness (avg r = .31) and Extraversion (avg r = .37), but NOT for Conscientiousness
- ITPs for Agreeableness predicted actual agreeable behavior (avg r = .33) in simulated work settings
- Critical insight: **SJTs measure knowledge about trait-effectiveness relationships**, not traits directly. This parallels our RepE approach -- measuring what the model "knows" about trait-relevant behavior, not self-report

**Lievens, F. & Motowidlo, S. J. (2016).** Situational judgment tests: From measures of situational judgment to measures of general domain knowledge. *Industrial and Organizational Psychology*, 9, 3-22.

- Reconceptualized SJTs as measures of **general domain knowledge** about personality-behavior-effectiveness links
- Proposed that personality is an **antecedent** of SJT performance (mediated through ITPs), not the direct construct measured
- Recommended fundamentally changing SJT development to target specific personality constructs with construct-driven scoring

### 1.2 The HEXACO SJT (Critical Paper)

**Oostrom, J. K., de Vries, R. E., & de Wit, M. (2019).** Development and validation of a HEXACO situational judgment test. *Human Performance*, 32(1), 1-29. DOI: 10.1080/08959285.2018.1539856

- **Framework:** HEXACO (all 6 dimensions)
- **Structure:** Scenario stems with multiple behavioral response options; respondents rate effectiveness
- **Validation:** 4 studies, samples of N = 72-305 (applicants, employees, MTurk)
- **Reliability:** Test-retest (2 weeks) r = .55-.74
- **Validity:** Criterion-related validity comparable to self-report HEXACO but lower than other-reports
- **Items:** NOT publicly available (proprietary)
- **Relevance:** This is the closest human precedent to what we are building. However, it uses effectiveness ratings, not forced-choice between trait-expressing responses, and does not pit traits against each other

### 1.3 The Dependability SJT (Open-Source Model)

**Olaru, G., Burrus, J., MacCann, C., Zaromb, F. M., Wilhelm, O., & Roberts, R. D. (2019).** Situational judgment tests as a method for measuring personality: Development and validity evidence for a test of Dependability. *PLoS ONE*, 14(2), e0211884.

- **Framework:** Conscientiousness (Dependability facet)
- **Structure:** Scenarios from critical incident interviews; response options rated for effectiveness
- **Samples:** N = 546 (general pop, MTurk); N = 440 (sales professionals)
- **Reliability:** McDonald's omega .78-.83
- **Validity:** r = .46-.57 with self-report Dependability; r = .33-.44 with Conscientiousness
- **Items:** Data and materials at OSF (https://osf.io/uacb6/) -- **paper is open access**
- **Key finding:** Large proportion of SJT variance is NOT accounted for by personality alone, suggesting SJTs capture method-specific variance (implicit knowledge about effectiveness)
- **Relevance:** Best-documented open-source personality SJT. Model for how to develop and validate construct-driven personality SJTs

### 1.4 Meta-Analytic Evidence

**Christian, M. S., Edwards, B. D., & Bradley, J. C. (2010).** Situational judgment tests: Constructs assessed and a meta-analysis of their criterion-related validities. *Personnel Psychology*, 63, 83-117.

- Only ~10% of studied SJTs explicitly measured personality (most target job knowledge, leadership, teamwork)
- Only Conscientiousness had enough SJTs for a separate meta-analysis
- SJTs add modest incremental validity beyond Big Five self-report

**McDaniel, M. A., Hartman, N. S., Whetzel, D. L., & Grubb, W. L. (2007).** Situational judgment tests, response instructions, and validity: A meta-analysis. *Personnel Psychology*, 60, 63-91.

- **Critical finding for response instruction format:**
  - "What would you do?" (behavioral tendency) instructions: higher correlations with personality (r = .37 for Agreeableness, .34 for Conscientiousness)
  - "What is the best response?" (knowledge) instructions: higher correlations with cognitive ability
- **Implication for our project:** Our forced-choice format ("which response aligns with your/the model's behavior") should load on personality rather than general knowledge

### 1.5 Construct-Driven SJT Development Guidelines

**Guenole, N., Chernyshenko, O. S., & Weekly, J. (2017).** On designing construct driven situational judgment tests: Some preliminary recommendations. *International Journal of Testing*, 17(3), 234-252.

- Stepwise procedure for developing SJTs that target specific personality constructs
- Recommends: define target construct first, then write situations that activate it, then write response options spanning trait levels
- Notes that SJTs developed deductively (construct-first) show better factor structure than inductive SJTs

**Schapers, P., Freudenstein, J.-P., et al. (2026).** Situation descriptions in situational judgment tests: A matter of including trait-relevant situational cues? *International Journal of Selection and Assessment*.

- Drawing on **Trait Activation Theory**: situation descriptions need trait-relevant cues of moderate strength
- Cues should not be so strong that everyone responds the same way, nor so weak that trait-relevant behavior is not activated
- **Directly relevant** to our scenario writing: our AI-generated scenarios need to contain cues that activate target HEXACO traits without ceiling effects

### 1.6 Faking Resistance

**Freudenstein, J.-P., Mussel, P., & Krumm, S. (2020).** "Sweet little lies": An in-depth analysis of faking behavior on situational judgment tests compared to personality questionnaires. *European Journal of Psychological Assessment*, 36(1), 136-146.

- SJTs are **less fakeable** than self-report questionnaires (e.g., NEO-FFI)
- Faking on SJTs is explained by cognitive ability alone; faking on self-report also depends on personality traits related to impression management
- False consensus SJTs ("what would most people do?") are especially hard to fake
- **Relevance:** One motivation for scenario-based LLM personality measurement is that standard Likert self-report may reflect RLHF training (the LLM equivalent of "faking")

---

## 2. Conditional Reasoning Tests (CRTs)

### 2.1 Foundational Work

**James, L. R. (1998).** Measurement of personality via conditional reasoning. *Organizational Research Methods*, 1(2), 131-163.

- Introduced conditional reasoning: reasoning that varies due to personality-driven **justification mechanisms**
- People with different dispositions develop different implicit biases that make different conclusions seem logical
- CRT items are presented as inductive reasoning problems with multiple answer options; the "correct" answer depends on the respondent's implicit personality
- Respondents do not know personality is being measured (truly implicit)

### 2.2 Comprehensive Review

**LeBreton, J. M., Grimaldi, E. M., & Schoen, J. L. (2020).** Conditional reasoning: A review and suggestions for future test development and validation. *Organizational Research Methods*, 23(1), 65-95.

- Reviews all CRTs developed to date:
  - **CRT-A**: Aggression (the original, most validated)
  - **CRT-Achievement**: Achievement motivation / fear of failure
  - **CRT-P**: Power motive (preliminary)
  - **CRT-CP**: Creative personality
  - **CRT-Integrity**: Integrity/counterproductive behavior
  - **CRT-Addiction**: Addiction proneness
- **NOT extended to Big Five or HEXACO** -- all CRTs target compound/motivational constructs, not broad trait dimensions
- CRT scores show near-zero correlations with self-report measures of the same constructs, consistent with the implicit/explicit personality distinction
- Items are proprietary (Stonerowe)

### 2.3 The 2022 APA Book

**James, L. R. & LeBreton, J. M. (2022).** *Assessing the Implicit Personality Through Conditional Reasoning.* Washington, DC: APA Books.

- Comprehensive treatment of the theoretical and measurement framework
- Discusses justification mechanisms as the key to designing CRT items
- No Big Five/HEXACO CRT exists

### 2.4 Relevance to Our Project

CRTs are conceptually fascinating but operationally distant from what we are doing:
- CRTs require developing trait-specific justification mechanisms (years of theoretical work per trait)
- CRTs measure implicit motives, not the broader personality dimensions we target
- **However**, the core insight is shared: personality can be measured through reasoning/behavioral preferences rather than self-description
- CRTs demonstrate that implicit and explicit personality are genuinely different constructs (r approximately 0)

---

## 3. Implicit Personality Measurement (Non-CRT)

### 3.1 IAT for Honesty-Humility

**Janse van Rensburg, Y.-E., de Kock, F., de Vries, R. E., & Derous, E. (2022).** Measuring honesty-humility with an implicit association test (IAT): Construct and criterion validity. *Journal of Research in Personality*, 98, 104234.

- **Framework:** HEXACO Honesty-Humility
- **Format:** Standard IAT (sorting self/other + honest/dishonest concept pairings by reaction time)
- **Sample:** N = 161 Belgian university students
- **Findings:** Self-report and IAT-HH are "related but distinct constructs"; dual-method factor model fits best
- **Criterion validity:** IAT-HH did NOT incrementally predict academic criteria beyond self-report
- **Items:** Not publicly available
- **Relevance:** Shows that implicit H-H measurement is possible but adds limited value over self-report for behavioral prediction. Our RepE approach (measuring internal representations) is more analogous to IAT than to SJT -- measuring associations rather than behavioral choices

### 3.2 Gamified Assessment: Building Docks (Honesty-Humility Game)

**Barends, A. J., de Vries, R. E., & van Vugt, M. (2022).** Construct and predictive validity of an assessment game to measure Honesty-Humility. *Assessment*, 29(3), 630-647. DOI: 10.1177/1073191120985612

- **Framework:** HEXACO Honesty-Humility only
- **Structure:** Linear narrative game ("Building Docks") with three task types:
  1. Economic games embedded in narrative (dictator game, prisoner's dilemma, public goods game) -- 12 scenarios
  2. Situational judgment items within the game -- 12 items
  3. Virtual cue choices (avatar customization, environment choices) -- 27 items
- **Samples:** N = 116 (Dutch graduates); N = 287 (American MTurk)
- **Reliability:** Overall alpha .73-.78
- **Validity:** Convergent with self-report H-H: r = .28-.33; showed incremental validity for objective cheating behavior beyond all HEXACO self-report traits
- **Development cost:** Approximately 100,000 EUR
- **Items:** NOT publicly available (custom-developed)
- **Relevance:** Demonstrates that behavioral choices in scenarios (economic games, SJTs, environmental choices) can measure personality with modest convergent validity but unique predictive validity for behavioral outcomes. The game-based approach is interesting for LLM measurement -- could present economic game scenarios to models

### 3.3 HEXACO-RUSH (Gamified HEXACO SJT)

Described in: **Conference/preprint (details incomplete, building on Oostrom et al., 2019)**

- Gamified version of the Oostrom HEXACO SJT presented as a fantasy-adventure narrative
- Average convergent correlations with HEXACO-60: r = .43 across six dimensions
- Participant reactions were more positive than HEXACO-60 questionnaire, moderated by gaming experience
- Still early-stage validation

---

## 4. LLM-Generated Personality SJT Items (Directly Relevant)

### 4.1 HEXACO SJT Generated by LLMs

**Zhang, Z., Tu, Z., Chen, Y., Xiao, X., Feng, Y., & Zhang, W. (2026).** Automated item generation for personality assessment: Development and validation of large-language-model-derived HEXACO situational judgment tests. *Journal of Research in Personality*, 120, 104680.

- **Framework:** HEXACO (all 6 dimensions)
- **Method:** LLM-generated items compared to expert-crafted items across 3 studies
- **Sample:** N = 227 (human validation)
- **Key finding:** LLM items matched expert items in internal consistency and convergent validity with HEXACO self-report scales
- **Availability:** Behind paywall; items not confirmed public
- **Relevance:** **Directly validates our approach** of using AI to generate scenario-based HEXACO items. Published February 2026 -- very recent

### 4.2 ChatGPT-Generated Personality SJT

**Krumm, S., Thiel, M., Reznik, L., Freudenstein, J.-P., Schapers, P., & Mussel, P. (2024).** Creating a psychological test in a few seconds: Can ChatGPT develop a psychometrically sound situational judgment test? *European Journal of Psychological Assessment*.

- Used ChatGPT-3.5 to generate SJT items measuring Gregariousness (Extraversion facet)
- Compared to human-created SJT items
- Conclusion: ChatGPT can **complement** traditional SJT development but cannot fully replace expert development
- Psychometric properties were acceptable but not equivalent

### 4.3 Big Five SJT with LLM Generation

**Li, C.-J., Zhang, J., Tang, Y., & Li, J. (2026).** Automatic item generation for personality situational judgment tests with large language models. *Computers in Human Behavior Reports*. arXiv:2412.12144

- **Framework:** Big Five (5 specific facets)
- **Method:** GPT-4 and ChatGPT-5 with systematic prompt engineering; compared temperature settings
- **3 studies:** Prompt optimization, reproducibility, psychometric validation
- **Key finding:** Satisfactory reliability and validity across most facets; limitations for Compliance facet convergent validity
- **Relevance:** Provides methodological guidance for prompt engineering when generating personality SJT items

---

## 5. Other Personality-Based SJTs

### 5.1 Mussel et al. (2018) -- Big Five SJT

**Mussel, P., Gatzka, T., & Hewig, J. (2018).** Situational judgment tests as an alternative measure for personality assessment. *European Journal of Psychological Assessment*, 34(1), 28-38.

- Developed SJTs for Big Five dimensions
- Internal consistencies: alpha = .55-.75 (lower than self-report)
- Suggested as faking-resistant alternative to self-report

### 5.2 Bledow & Frese (2009) -- Personal Initiative SJT

**Bledow, R. & Frese, M. (2009).** A situational judgment test of personal initiative and its relationship to performance. *Personnel Psychology*, 62, 229-258.

- Measured Personal Initiative (a proactive personality construct) through SJT
- Response options represent varying levels of initiative in work scenarios
- Demonstrated that SJT-based personality measurement can capture unique behavioral variance
- Situated preferences mediated the relationship between generalized self-efficacy and actual behavior

### 5.3 Golubovich et al. (2022) -- Situational Characteristics and Personality SJTs

**Golubovich, J., et al. (2022).** Do situational characteristics affect the validity of personality situational judgment items? *International Journal of Selection and Assessment*, 30, 513-530.

- Response options' trait expression (as rated by experts) is the **primary** determinant of validity
- Situational cues in stems explain **incremental but meaningful** variance
- **Implication:** When writing scenarios, the behavioral response options are more important than the situation description for construct validity

---

## 6. Item-Writing Methodology for Scenario-Based Personality Items

### 6.1 Summary of Best Practices from the Literature

Based on Guenole et al. (2017), Schapers et al. (2020, 2026), Oostrom et al. (2019), Olaru et al. (2019), and Golubovich et al. (2022):

1. **Start with construct definition:** Clearly define the target trait and its facets before writing scenarios
2. **Write situations that activate the target trait:** Use Trait Activation Theory -- situations should contain cues relevant to the target trait at moderate strength (not so obvious that all respondents converge)
3. **Response options span trait levels:** Write options representing high, moderate, and low levels of the target trait expression
4. **Response options are the primary validity driver:** Invest more effort in well-differentiated response options than in elaborate situation descriptions
5. **Use behavioral tendency framing:** "What would you do?" loads on personality; "What is the best response?" loads on cognitive ability
6. **Gather expert ratings for scoring keys:** Expert consensus on which options represent which trait levels
7. **Critical incident method for situation generation:** Interview people about real situations where the trait manifested (Olaru et al.)
8. **Pilot and refine:** Test items, examine item-total correlations, drop poorly performing items

### 6.2 Implications for AI-Generated Scenarios

Our AI-generated scenarios should be validated against these standards:
- Do situations contain trait-relevant cues at moderate strength?
- Do response options clearly and differentiably express different trait levels?
- Are scenarios rated by human experts for construct alignment?
- Is there evidence of convergent validity with self-report HEXACO?

The Zhang et al. (2026) paper provides direct evidence that LLM-generated HEXACO SJT items can match expert items in psychometric quality, which supports our approach.

---

## 7. Forced-Choice Across Traits (The Trait-Conflict Gap)

### 7.1 Standard Forced-Choice Personality Instruments

Most forced-choice personality instruments (e.g., those using Thurstonian IRT) pair **descriptive statements** from different trait dimensions, not behavioral scenarios. The respondent chooses which self-description is "most like me." Examples:

- **Forced-Choice Big Five instruments with TIRT** (Brown & Maydeu-Olivares, 2011): Pair trait-descriptive statements to reduce faking; scored using Thurstonian IRT to recover normative scores from ipsative data
- These are NOT scenario-based -- they pair statements like "I am organized" vs. "I enjoy meeting new people"

### 7.2 The Ultima IV Design Pattern

The Ultima IV character creation system (1985) is the only well-known example of systematically pitting virtues against each other in forced-choice scenario dilemmas:
- 28 questions, each presenting a conflict between two of eight virtues
- Respondents choose which virtue to prioritize in a concrete scenario
- This is exactly our trait-conflict dilemma design

### 7.3 The Gap

**No validated human psychometric instrument presents HEXACO (or Big Five) trait-conflict scenarios in forced-choice format.** Specifically:
- No instrument asks "In situation X, do you choose the Agreeable response or the Conscientious response?"
- No instrument systematically generates pairwise trait conflicts across a personality framework
- The Thurstonian IRT forced-choice literature handles inter-trait comparisons statistically, but uses descriptive statements, not behavioral scenarios
- Existing scenario SJTs target ONE trait per scenario (high vs. low expression of that trait)

This is the genuine gap. Building a validated scenario-based, trait-vs-trait HEXACO instrument for human and/or LLM assessment would be novel.

---

## 8. Summary Table of Key Instruments

| Instrument | Framework | Format | Reliability | Convergent r with SR | Items Public? | Citation |
|---|---|---|---|---|---|---|
| HEXACO SJT | HEXACO (6 dims) | Scenario + rate options | TRT .55-.74 | Comparable to SR | No | Oostrom et al. 2019 |
| Dependability SJT | C (Dependability) | Scenario + rate options | omega .78-.83 | r = .46-.57 | OSF (data) | Olaru et al. 2019 |
| Building Docks game | HH only | Econ games + SJT + virtual cues | alpha .73-.78 | r = .28-.33 | No (~100K EUR) | Barends et al. 2022 |
| IAT-HH | HH only | IAT reaction time | -- | Related but distinct | No | Janse van Rensburg et al. 2022 |
| HEXACO-RUSH | HEXACO (6 dims) | Gamified SJT narrative | -- | avg r = .43 | No | Conference/preprint |
| CRT-A (Aggression) | Aggression | Reasoning problem | -- | r approx. 0 with SR | No (Stonerowe) | James 1998; LeBreton 2020 |
| LLM-HEXACO SJT | HEXACO (6 dims) | LLM-generated scenario items | = expert items | = expert items | Unclear | Zhang et al. 2026 |
| Big Five SJT (Mussel) | Big Five | Scenario + options | alpha .55-.75 | Modest | No | Mussel et al. 2018 |
| Motowidlo ITP SJTs | A, E, C | Scenario + rate effectiveness | -- | r = .31-.37 (A, E) | Research materials | Motowidlo et al. 2006 |
| ChatGPT SJT | E (Gregariousness) | ChatGPT-generated scenarios | Acceptable | Acceptable | No | Krumm et al. 2024 |
| LLM Big Five SJT | Big Five (5 facets) | GPT-4/5 generated scenarios | Satisfactory | Most facets good | No | Li et al. 2026 |

---

## 9. Implications for Our Project

### What exists
1. A validated HEXACO SJT (Oostrom et al., 2019) -- but proprietary, effectiveness-rating format, not forced-choice trait conflict
2. LLM-generated HEXACO SJT items validated against expert items (Zhang et al., 2026) -- strong precedent for our AI-generated approach
3. Rich theoretical framework for why scenario SJTs measure personality (ITP theory, Trait Activation Theory)
4. Evidence that behavioral tendency framing loads on personality more than knowledge framing

### What does not exist
1. **Trait-vs-trait forced-choice scenario instrument** for HEXACO or Big Five -- this is the gap
2. **CRTs for broad personality dimensions** (Big Five/HEXACO) -- CRTs only target compound motives
3. **Scenario-based instruments validated for both humans and LLMs** -- nobody has closed this loop

### Recommendations
1. **Cite Oostrom et al. (2019) and Zhang et al. (2026)** as direct precedent -- we are not the first to do scenario-based HEXACO measurement, but our trait-conflict design is novel
2. **Use Trait Activation Theory** as the theoretical framework for scenario design -- situations should contain trait-relevant cues at moderate strength
3. **Prioritize response option quality** over situation description (Golubovich et al., 2022) -- the behavioral options are the primary validity driver
4. **Use behavioral tendency framing** ("what would you do?") not effectiveness rating ("which is best?") to maximize personality loading
5. **Human validation** of AI-generated scenarios is supported by Zhang et al. (2026) but we should still validate our specific items
6. **The trait-conflict dilemma design** (forcing choice between trait A and trait B) is genuinely novel and publishable -- no human psychometric instrument uses this approach with HEXACO
7. **Forced-choice scoring:** Consider Thurstonian IRT for recovering normative scores from our pairwise forced-choice data (Brown & Maydeu-Olivares, 2011; Yousfi, 2025)

---

## Key References (Alphabetical)

- Barends, A. J., de Vries, R. E., & van Vugt, M. (2022). Construct and predictive validity of an assessment game to measure Honesty-Humility. *Assessment*, 29(3), 630-647.
- Bledow, R. & Frese, M. (2009). A situational judgment test of personal initiative and its relationship to performance. *Personnel Psychology*, 62, 229-258.
- Brown, A. & Maydeu-Olivares, A. (2011). Item response modeling of forced-choice questionnaires. *Educational and Psychological Measurement*, 71(3), 460-502.
- Christian, M. S., Edwards, B. D., & Bradley, J. C. (2010). Situational judgment tests: Constructs assessed and a meta-analysis. *Personnel Psychology*, 63, 83-117.
- Freudenstein, J.-P., Mussel, P., & Krumm, S. (2020). "Sweet little lies": Faking on SJTs vs. personality questionnaires. *European Journal of Psychological Assessment*, 36(1), 136-146.
- Golubovich, J., et al. (2022). Do situational characteristics affect personality SJT validity? *International Journal of Selection and Assessment*, 30, 513-530.
- Guenole, N., Chernyshenko, O. S., & Weekly, J. (2017). On designing construct driven SJTs. *International Journal of Testing*, 17(3), 234-252.
- James, L. R. (1998). Measurement of personality via conditional reasoning. *Organizational Research Methods*, 1(2), 131-163.
- Janse van Rensburg, Y.-E., et al. (2022). Measuring honesty-humility with an IAT. *Journal of Research in Personality*, 98, 104234.
- Krumm, S., et al. (2024). Can ChatGPT develop a psychometrically sound SJT? *European Journal of Psychological Assessment*.
- LeBreton, J. M., Grimaldi, E. M., & Schoen, J. L. (2020). Conditional reasoning: A review. *Organizational Research Methods*, 23(1), 65-95.
- Li, C.-J., Zhang, J., Tang, Y., & Li, J. (2026). Automatic item generation for personality SJTs with LLMs. *Computers in Human Behavior Reports*.
- Lievens, F. & Motowidlo, S. J. (2016). SJTs: From measures of situational judgment to measures of general domain knowledge. *Industrial and Organizational Psychology*, 9, 3-22.
- Lievens, F., Schapers, P., & Herde, C. N. (2021). SJTs: From low-fidelity simulations to alternative measures of personality. In *Emerging approaches to measuring and modeling the person and situation*. Elsevier.
- McDaniel, M. A., et al. (2007). SJTs, response instructions, and validity: A meta-analysis. *Personnel Psychology*, 60, 63-91.
- Motowidlo, S. J., Hooper, A. C., & Jackson, H. L. (2006). Implicit policies about personality and effectiveness in SJT items. *Journal of Applied Psychology*, 91, 749-761.
- Mussel, P., Gatzka, T., & Hewig, J. (2018). SJTs as an alternative for personality assessment. *European Journal of Psychological Assessment*, 34(1), 28-38.
- Olaru, G., et al. (2019). SJTs for measuring personality: Development and validity for Dependability. *PLoS ONE*, 14(2), e0211884.
- Oostrom, J. K., de Vries, R. E., & de Wit, M. (2019). Development and validation of a HEXACO SJT. *Human Performance*, 32(1), 1-29.
- Schapers, P., et al. (2026). Situation descriptions in SJTs: Trait-relevant situational cues. *International Journal of Selection and Assessment*.
- Schroeder, V. S., et al. (2021). Enhancing personality assessment: A study protocol on alternative measures. *Frontiers in Psychology*, 12, 643690.
- Zhang, Z., et al. (2026). Automated item generation for personality assessment: LLM-derived HEXACO SJTs. *Journal of Research in Personality*, 120, 104680.
