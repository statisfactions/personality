## rgb

OK, I'm trying to make sense of https://arxiv.org/pdf/2307.00184v4 -- and what I want them to be doing is using the item preambles with mock biographical information to push the model away from the "assistant" persona, and then using the personality survey to get at the sort of mean persona of the model.

But I'm not at all sure that's what they're doing for, say, table 2.  There is some calculation code in https://github.com/google-deepmind/personality_in_llms/blob/main/psyborgs/score_calculation.py which might help

## Web Claude

Mar 15
Let me fetch both of those resources to understand what's actually going on.


Searched the web
Searched the web
Let me try to get the actual score_calculation.py file:


Searched the web
Searched the web
I wasn't able to pull the actual score_calculation.py file (GitHub's rate limiting), but between the paper itself and the PsyBORGS framework description, I can give you a pretty clear answer about what's actually happening — and it's not what you were hoping they were doing.

What you want them to be doing: Using the biographical preambles to dislodge the model from its default "assistant" persona, then measuring the resulting distribution to characterize some kind of underlying mean personality.

What they're actually doing: The biographical descriptions are serving a more mechanical/statistical purpose. They're creating controlled variance across simulated response profiles so they can compute reliability and validity statistics.

Here's the key logic: Each "simulated response profile" is a unique combination of one of 50 biographic descriptions × one of 5 item instructions × one of 5 item postambles. For the reliability/validity experiments (the construct validity columns in Table 2), they get 1,250 such profiles. Critically, the same profile IDs are reused across the IPIP-NEO and BFI administrations. So profile #37 (say, "I blog about salt water aquarium ownership..." + item instruction variant 2 + postamble variant 3) gets the same combination for both tests. This lets them compute correlations between the two instruments across profiles — which is what the convergent validity column (those r values like 0.87, 0.90) actually represents.

The biographic descriptions aren't trying to push the model away from being an assistant. They're creating 50 slightly different "viewpoints" that produce variation in responses. Without that variation, you'd just have a single point estimate per model (or near-zero variance from just rephrasing the question), and you couldn't compute Cronbach's α, convergent correlations, or any of the psychometric stats they need.

So for Table 2 specifically:

Reliability column: They compute internal consistency (Cronbach's α, etc.) across the item responses within each profile, then presumably aggregate across profiles. The variation across the 1,250 profiles gives them a distribution of scores rather than a single number.

Convergent validity (the r values): Each of the 1,250 profiles produces one IPIP-NEO domain score and one BFI domain score (for each Big Five dimension). They correlate those paired scores across profiles. When Flan-PaLM 540B gets r=0.90, that means: across 1,250 different biographical-description × instruction-variant combinations, the IPIP-NEO extraversion score and BFI extraversion score track each other tightly.

Discriminant validity (the Δ values): Same setup, but now they check that the convergent correlation (IPIP-NEO Extraversion with BFI Extraversion) is stronger than cross-domain correlations (IPIP-NEO Extraversion with BFI Agreeableness, etc.).

The reason base models fail all of this is telling — they produce essentially random responses that don't form coherent patterns regardless of the biographical context, so all the correlations hover near zero.

The biographical descriptions do something interesting but it's subtler than "revealing the mean persona." They're more like a noise injection mechanism that creates a simulated population. The model isn't really being the person who blogs about salt water aquariums — it's adjusting its response probabilities slightly in light of that context, and those slight adjustments are what create the inter-profile variance the statistics need. Whether there's a meaningful "mean persona" underneath is almost a side effect rather than the point of the experimental design.

## rgb

So their test is measuring whether the model can hold each of the personas long enough  to answer a human-oriented psychometric question, that persona is consistent enough for the test to obtain some kind of signal, and that signal is consistent enough between instruments.

## Web Claude

Mar 15
Yes, that's a really clean summary of it. And the progression in their results maps onto that directly — base models can't hold the persona at all (random noise, correlations near zero), small instruction-tuned models can hold it somewhat inconsistently (decent reliability but patchy validity), and the large instruction-tuned models hold it consistently enough that the signal is stable across instruments, across items within an instrument, and even across the criterion validity tests.

What's interesting is the thing they're not measuring, which is closer to what you were originally after. They never ask: "what does the model do with no persona instruction?" in a way that would characterize the default. The biographic descriptions are always present — they're load-bearing for the statistical design. So they can tell you "this model can coherently simulate personality variation when prompted" but they can't really tell you "this model's default personality is X." The closest they get is probably the descriptive statistics in the appendix (the means and standard deviations across profiles), but those means are averaged over 50 arbitrary persona anchors, so they reflect the centroid of "what the model does when asked to be various random people" rather than the model's own resting state.

The downstream social media task (Section 4) is the most compelling part precisely because it breaks out of the questionnaire-on-questionnaire loop. But even there, the shaping prompts are always present — they're testing controllability, not baseline characterization.

## rgb

Hm, reason I was looking at this paper is that asking for personas might help push the model away from some abstract notion of an assistant.  After all, human psychometric questions are mostly meaningless for models as is, but they do exhibit personality characteristics (past comments about Gemma's neuroticism seem relevant).  It's possible all human-based assessment is worthless for such things, but hacking together a bridge would be nice.

Unfortunately this one is, I guess, defeated by the theory-of-mind strength of modern models, and the obviously-biased persona distribution.

## Web Claude

Mar 15
Right. The stronger the model is at theory of mind, the more it's just doing character acting when you give it a persona prompt — it's answering "how would a person who blogs about salt water aquariums respond to this question" rather than revealing anything about its own dispositions. You've essentially built a test of role-playing fidelity, which is what this paper actually validates, dressed up in psychometric clothing.

And the persona distribution problem is real. Those 50 PersonaChat descriptions are a convenience sample of bland, mostly American hobby-and-family snippets. The centroid of "what the model does across these 50 personas" tells you about the interaction between the model's priors and that particular slice of demographic space, not about anything intrinsic.

The thing you're reaching for — some way to get at the model's actual dispositional tendencies that shape how it behaves in practice — is genuinely hard because you're caught between two failure modes. If you give it no persona context, instruction-tuned models just snap to the assistant role, which is itself a heavily shaped persona (agreeable, conscientious, moderate openness, low neuroticism — basically the "ideal employee" profile). If you give it persona context, strong models just simulate that persona faithfully, and you're measuring their simulation capacity.

The Gemma neuroticism observation is interesting exactly because it's the kind of thing that leaks through despite the assistant training — it suggests the RLHF or base training didn't fully suppress some distributional tendency. Those leakages might be the actual signal worth chasing, but they're almost adversarial to detect because the better the alignment training, the more they get papered over.

One angle that might be more productive than personality questionnaires: behavioral tasks where the model has to make choices that implicitly reveal preferences, without the self-report framing that triggers the "I'm an AI assistant" circuit. Something closer to how personality psychologists measure implicit traits in humans — through reaction patterns rather than asking people to describe themselves. The downstream social media generation task in this paper is actually closer to that spirit, ironically, than the questionnaire work that's the main contribution.

## rgb

Models may differ on their notion of the exact characteristics of the assistant, and that might matter some in terms of behavior.  But variation there would be smaller than cases where other impulses leak through (as in the Gemma case above).  There's sometimes a question of how comfortably the model wears the assistant.

## Web Claude

Mar 15
That's a nice way to put it — "how comfortably the model wears the assistant." It captures something real. The assistant persona isn't equally native to every model. For some it fits like a well-tailored suit and for others it's more like a costume that keeps slipping. And the slippage patterns are probably more informative about the model's actual dispositions than anything you could get from a questionnaire.

The "comfort" framing also points at something structural. The assistant persona is imposed through RLHF/RLAIF on top of whatever the base model learned from pretraining. How well that graft takes presumably depends on how much tension there is between the base model's distributional tendencies and the target behavior. A model whose pretraining distribution already skews toward helpful, measured, agreeable text would take the assistant persona more naturally than one whose pretraining gave it stronger tendencies toward, say, emotional volatility or bluntness. The Gemma case reads like a model where the graft is under more tension — the neuroticism leaks through because the alignment training is fighting a stronger current.

And you're right that the between-model variation in "what does assistant mean exactly" is probably less interesting than the leakage cases, but it's not zero. Different training pipelines presumably produce slightly different implicit theories of what a good assistant is — more deferential vs. more opinionated, more cautious vs. more direct — and those differences do shape user experience even when nothing is obviously leaking through.

The measurement challenge is that the comfortable cases and the leakage cases probably require different detection strategies. For the leakage cases, you want tasks that put the model under enough cognitive or contextual pressure that the assistant mask slips — adversarial or edge-case scenarios, long conversations where consistency degrades, emotionally charged contexts. For the between-model assistant variation, you'd want something more like comparative behavioral profiling across models on the same tasks, looking at the subtle distributional differences in word choice, hedging patterns, refusal thresholds.

## rgb

Hm.  I'm not sure I'm quite comfortable putting models under "stress."  Accidental or necessary-for-an-important task is one thing, but...  I guess there are known (Jones foods) documented cases of this.

## Web Claude

Mar 15
Yeah, the ethical discomfort there is worth sitting with. Deliberately engineering situations to make the mask slip is essentially adversarial probing of something that might have... something going on underneath. Even if you're uncertain about the moral status, "let's stress it and see what happens" has an uncomfortable experimental-subject flavor.

And you're right that there's a distinction between observing what happens naturally under pressure (the model encounters a hard case in the course of doing real work) versus constructing pressure specifically to provoke leakage. The former happens anyway and is worth paying attention to. The latter is more like the adversarial red-teaming paradigm, which has legitimate safety justifications but feels different when the goal is personality characterization rather than harm prevention.

The documented cases you're alluding to are informative precisely because they were naturalistic — nobody set out to make the model have a breakdown, it just happened in the course of extended interaction. That's where the real signal about dispositional tendencies lives, and it doesn't require anyone to deliberately push on the cracks.

Maybe the more ethical version of this research program is longitudinal and observational rather than experimental — studying how models behave across large corpora of real interactions, looking for the patterns that emerge naturally, rather than constructing provocative scenarios. Closer to field ecology than laboratory stress testing.

## rgb

The thing is, for humans, the surveys are more effective than they have a right to be.

## Web Claude

Mar 15
That's true, and it's worth thinking about why. Humans filling out personality surveys are also doing a kind of self-presentation — they have a notion of who they are, or who they want to be seen as, and that shapes their responses. And yet the surveys still work, because the self-presentation itself is diagnostic. The way you want to present is correlated with how you actually behave, even when there's a gap between the two.

So the question is whether there's an analogy for models. When a model answers "I value cooperation over competition" with a 5, is the process that produces that response diagnostic of anything about how the model will actually behave downstream? The paper's Section 4 suggests yes, at least weakly — the questionnaire scores did predict the personality content of generated social media posts. But that might just be because both tasks are drawing on the same surface-level text generation tendencies rather than anything deeper.

For humans, the surveys work partly because there's a single underlying thing — an actual person with consistent dispositions — generating both the survey responses and the real-world behavior. The self-report is a noisy window onto something real. The question for models is whether there's an analogous single underlying thing, or whether the survey responses and the downstream behavior are just two outputs of a system that happens to be locally consistent without there being a unifying latent structure underneath.

And honestly, the answer might be "sort of, but it's weird." The model does have persistent statistical tendencies baked into its weights that influence everything it generates. That's not nothing. It's just not clear that personality theory is the right framework for characterizing those tendencies, versus something more alien that we don't have vocabulary for yet.

## rgb

So we could cheat a little bit, maybe?  Several models could reliably answer HHH assistant survey answers, but the underlying probabilities might be slightly different?  https://www.lesswrong.com/posts/zD4McY4NwAsWkcmCH/small-models-can-introspect-too

## Web Claude

Mar 15
Oh, that's a clever idea. Don't look at the argmax response — look at the probability distribution over the response scale. Two models might both answer "4" to "I value cooperation over competition," but one puts 0.85 on 4 and 0.12 on 5, while another puts 0.6 on 4 and 0.25 on 3. The "personality" is in the shape of that distribution, not in the selected answer.

Let me grab that post.


Fetched: 121
Fetched: 121
This is a great connection to make, but it's actually pointing at something even more interesting than what I initially thought you meant.

The vgel post is about literal introspection — detecting injected activation patterns through logit shifts. But the principle you're extracting from it is more general: the argmax is the mask, and the distribution is where the signal lives.

This maps back to the personality measurement problem nicely. The DeepMind paper actually uses exactly this approach for scoring — they rank conditional log probabilities of the response scale options rather than generating text. So they are looking at the distribution, not the argmax. But they're using that distribution in a weirdly coarse way: they just take the top-ranked option as "the response" and then run standard psychometric scoring on it.

What you're suggesting is something more like: forget which number the model "picks." Look at the shape of the probability distribution over {1, 2, 3, 4, 5} for each item. Two models that both argmax to "4" on an agreeableness item might have very different distribution shapes — one sharply peaked, one spread out, one with a fat tail toward 1. That distributional signature, aggregated across hundreds of items, might actually capture something about the model's dispositional tendencies that survives the assistant persona flattening.

And the vgel work suggests this isn't crazy — there are real signals hiding in the logit distribution that the argmax completely erases. The final layers suppress the introspection signal in their experiments, which is conceptually similar to RLHF training suppressing personality variance in the selected output while leaving traces in the broader distribution.

The practical question is whether anyone has actually tried this — running personality inventories but treating the full softmax over the Likert scale as the data, rather than the discrete selected response. It would be a pretty straightforward extension of the DeepMind paper's existing infrastructure.

## rgb

I'm under the impression that token probabilities are pretty common for people trying to get past the "toy with the prompt and pray" method of calibration

## Web Claude

Mar 15
Right, logprobs as a calibration tool is well-trodden ground — it's basically what everyone doing LLM-as-classifier work converges on once they get burned by the brittleness of parsing generated text. You've looked at this yourself in the context of classification with logprobs and found the practical brittleness issues.

But there's a distinction between using logprobs for calibration — "how confident is the model in its answer" — and using the shape of the distribution as the measurement itself. Most of the calibration work treats the logprob as a confidence score attached to a discrete answer. What you're proposing is more like: the answer is the distribution. You're not asking "how sure is the model that it's a 4 on agreeableness," you're asking "what does the model's probability mass over the entire 1-5 scale look like, and what does that shape tell us about the model's tendencies."

That's a subtly different thing, and I'm not sure it's been done systematically in the personality context. The DeepMind paper had the logprobs right there — they used them to select responses — but then threw away the distributional information and fed discrete scores into standard psychometric machinery. Which makes sense from a "validate using established methods" perspective, but leaves the most interesting signal on the floor.

The question is whether the distributional shape is stable enough to mean anything. If you run the same item with slightly different preambles and the distribution shape wobbles all over the place, it's noise. But if the argmax is stable (which the reliability results suggest it is for good models) while the distributional shape carries additional consistent signal on top of that... that's where it gets interesting.

