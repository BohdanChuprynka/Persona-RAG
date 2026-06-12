= Introduction <sec-intro>

A general-purpose assistant is trained to sound like _no one in particular_ —
fluent, helpful, register-neutral. This report asks the opposite question: can a
model be made to sound like _one particular person_ texting, and — harder — how
would you _prove_ it had? The subject is the author's own messaging history:
roughly eleven thousand reply turns of dense Ukrainian/Russian/English
code-switching, terse multi-bubble bursts, a `)` smiley tic, almost no `!`, and
lowercase casing. The artifact is Persona-RAG, a Telegram bot that answers in that
voice; the contribution is less the bot than the _honest measurement_ of whether a
small, free, local fine-tune reproduces the voice better than the fluent
gpt-4o-mini product it was built to replace.

== Why this is not the usual problem

The framing matters because four adjacent, well-studied problems each look like
this one and are not:

- *Not a chatbot.* The objective is not task success or helpfulness but fidelity
  to one idiolect; a "better" answer that does not read like the person is a
  _failure_, not an improvement.
- *Not knowledge-RAG.* Retrieval here fetches the owner's _own past replies as
  style exemplars_, not facts to answer with. As the results show, the decisive
  win comes from the weights, not the retrieval — the "RAG" is the baseline the
  fine-tune beats, an irony the name preserves.
- *Not generic style transfer.* There is no parallel corpus and no discrete style
  label @shen2017styletransfer; the "style" is a single private individual's,
  learned only from their raw chat logs, and the evaluation must survive
  code-switching @dogruoz2021survey and burst shape that English-centric metrics
  ignore.
- *Not a memory system.* Memory carries _facts_ (who someone is, what was said);
  voice lives in the _weights_. Part of this report's point is that the two
  decompose cleanly, and that conflating them — answering a casual message with a
  recited fact sheet — is itself a voice failure.

== Research questions

The study is organised around five questions, the first of which it pre-registers
as _primary_ and, candidly, cannot resolve:

+ *RQ0 (primary, unresolved).* Does the fine-tune pass a blind human "which is more
  like something you'd send?" test? This is the only direct measure of voice, and
  for reasons of rater bias and privacy (@sec-ethics) it cannot be cleanly
  established here.
+ *RQ1.* On measurable surface register, does the local fine-tune sound more like
  the person than (a) bare gpt-4o-mini, (b) the same-family no-LoRA Qwen base, and
  (c) the fully-equipped product?
+ *RQ2.* What does the production scaffold — rich prompt, retrieval, decode levers —
  actually _contribute_ over the bare weights?
+ *RQ3.* Can the comparison be _trusted_: a leak-free split, a rule fixed in
  advance, uncertainty on every headline number?
+ *RQ4.* Can a thin grounding layer add identity facts the fine-tune never learned
  _without_ destroying the voice a heavy prompt would?

== Contributions

- *A trustworthy harness, and the leak it was built to expose.* We document a
  disqualifying \~90% train/test leak in the original evaluation, then replace it
  with a leak-free, pre-registered, two-arm (model-vs-product) comparison carrying
  paired bootstrap intervals and a per-item retrieval guard (@sec-build, @sec-trust).
- *A like-for-like verdict.* On a level field the fine-tune is decisively closer to
  the person's reply length and code-switch register, matches the no-`!` rule, and
  beats the same-family no-LoRA Qwen base; fully equipped, the product's machinery
  only reaches a _tie_ — a result we decode-robustness-check across three independent
  re-decodes (@sec-relative).
- *An anatomy of the scaffold.* We isolate what the prompt-plus-retrieval-plus-lever
  stack buys on the API side, and show it mostly recovers ground the bare fine-tune
  already held (@sec-relative).
- *A grounding layer that respects the voice.* A thin, intent-routed fact card
  raises the local model's correct-identity-answer rate from 0.05 to 0.33 under a
  question-clustered probe, with the voice intact (@sec-grounding).
- *An honest accounting of the limits.* A single subject, a single recall-biased
  rater, surface proxies standing in for an unmeasured target, and an unresolved
  primary endpoint — stated as findings, not footnotes (@sec-learned).

A note on scope, made once and meant: this is a _single-subject case study_. Every
quantitative claim is about this one persona and this one corpus; none generalises
to personas in the abstract, and the report is written to make that boundary
visible rather than to paper over it.
