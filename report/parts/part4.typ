= Absolute fidelity

Beating the baseline is the relative question. The harder, more interesting one is
absolute: not "is the fine-tune better than the API", but "is it good enough to pass
for the person themselves?" This is where automatic metrics run out — only a human
can answer it — and where the strongest possible judge is available: the person
whose voice is being cloned.

#let pending(body) = rect(
  width: 78%,
  inset: 12pt,
  radius: 3pt,
  fill: rgb("#f8fafc"),
  stroke: (paint: rgb("#94a3b8"), dash: "dashed"),
)[#align(center)[#text(9pt, style: "italic", fill: rgb("#475569"))[#body]]]

== The blind human panel

The relative verdict has a direct human read. A blind, randomized forced-choice
panel presents anonymized reply pairs and asks only "which is more like something
you'd send?", with the backend identities stored separately and attention checks
mixed in. Rated by eye, the result is unambiguous: the API is *trivially
discriminable* — the owner picks it out every time — so the fine-tune wins voice
decisively over gpt-4o-mini. That makes *three independent methods* — the controlled
arm, the production arm, and the human eye — agree on the same conclusion.

The formal statistic is a win-rate with a Wilson 95% interval @wilson1927, scored as
a real preference only if the interval excludes chance (0.5). At the panel size used
for a ship decision (\~100 decisive items) this can resolve a clear 60/40 preference;
a subtle 55/45 would need \~400. The kit and scorer exist; only the rating remains
(@fig-human).

#figure(
  pending[Figure pending: the blind-panel LoRA win-rate with its Wilson 95% CI
  renders here once the rater kit (`reports/main/human_eval/rater.html`) is scored to
  `choices.json`. Current verdict is qualitative — the API is trivially
  discriminable by eye.],
  caption: [Blind human preference panel: LoRA win-rate vs. chance (awaiting ratings).],
) <fig-human>

== The Turing test

With the relative question settled, the open frontier is the absolute one: can the
person distinguish the fine-tune's reply from their _own_ real reply? For each
held-out context the LoRA's generation is paired against the genuine reply, shown
blind, and judged with a single question — "which is the machine?" Both replies
already exist in the comparison data, so this costs nothing to assemble.

The pass condition _flips_ relative to the API panel. There, a high win-rate was the
goal; here, success is _failure to discriminate_: a detection rate whose Wilson
interval _includes_ 0.5 means the judge cannot beat chance, i.e. the fine-tune is
statistically indistinguishable from the person. It is the harshest test in the
report — the persona-target is the strongest conceivable discriminator of their own
voice (@fig-turing).

#figure(
  pending[Figure pending: the Turing detection rate with its Wilson 95% CI renders
  here once the LoRA-vs-real kit (`reports/main/turing/rater.html`) is rated. An
  interval spanning 0.5 means indistinguishable from the real person.],
  caption: [Turing slice: human detection rate vs. chance (awaiting ratings).],
) <fig-turing>

== What the tells will tell us

Every catch in the Turing panel also records a one-tap _tell_, and the tells bucket
into two kinds that point at different fixes. *Voice tells* — wording, length,
punctuation, too-generic — are addressable by decode parameters or more training.
*Knowledge tells* — the real reply carried a specific fact the model could not have
known — are addressable only by retrieval and grounding. The ratio between them
_sizes the next investment_: if catches are mostly missing-facts, the voice is
effectively solved and the remaining gap is a grounding problem (an Obsidian /
chat-history RAG layer); if they are mostly voice, more retrieval will not help. A
prior, to be replaced by the run, puts detection around 65–75% with roughly half the
catches being missing-facts.

One honest caveat frames the whole test: there is exactly one ground-truth reply per
context, yet many different replies could be equally "him". A catch may therefore
reflect a missing private fact rather than a voice defect — which is precisely why
the verdict is read together with the voice-vs-knowledge split, never from the
detection rate alone.
