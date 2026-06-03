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

== The human verdict, and why it is not the primary here

The pre-registered design names a blind human win-rate as the _primary_ verdict — a
forced choice over anonymized pairs ("which is more like something you'd send?"),
scored as a real preference only if its Wilson 95% interval @wilson1927 excludes
chance. In practice that verdict cannot be cleanly established for _this_ system, for
two reasons that are themselves findings.

First, the only rater with standing — the author, whose voice it is — is
*recall-biased*: he recognizes his own real messages, so when he can tell a generated
reply from a real one, it may be memory rather than a voice defect. Second, the
corpus is dense with private personal content, so recruiting outside raters is
precluded. An informal self-read did find the bare-model API replies obviously
off-voice, which is suggestive — but it is a single, recall-biased rater looking at the
_Arm B_ (thin-prompt) outputs, not a scored, unbiased panel, and the pre-registered
rule returns a _tie_ on the primary channel until an unbiased win-rate exists
(@fig-human). The verdict in this report therefore rests on the automatic arms; the
human read is a weak, confounded corroborator, not a third independent method. The
kit and scorer exist (the win-rate would resolve a clear \~60/40 preference at
\~100 items, a 55/45 at \~400), but a clean reading needs raters the privacy
constraint rules out.

#figure(
  pending[Figure pending: the blind-panel LoRA win-rate with its Wilson 95% CI would
  render here once the rater kit (`reports/main/human_eval/rater.html`) is scored to
  `choices.json`. No unbiased win-rate exists today — the lone available rater is
  recall-biased and privacy precludes others; the informal self-read is suggestive
  only.],
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
voice (@fig-turing). That strength is also the catch: because the author recognizes
his _own_ real replies, a raw detection rate measures recall as much as voice and
would _overstate_ discriminability. A meaningful run has to control for memory —
re-rating after a long delay, or restricting to contexts whose replies he does not
recall — or it cannot separate "I remember sending this" from "this sounds like me."
With outside raters barred on privacy grounds, that is why the Turing result is
reported here as genuinely prospective, not as a number this study can yet defend.

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
