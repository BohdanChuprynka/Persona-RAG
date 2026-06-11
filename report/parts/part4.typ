= Absolute fidelity <sec-absolute>

Beating the baseline is the relative question. The harder, more interesting one is
absolute: not "is the fine-tune better than the API", but "is it good enough to pass
for the person themselves?" This is where automatic metrics run out — only a human
can answer it — and where the strongest possible judge exists _in principle_: the
person whose voice is being cloned, though, as below, that judge turns out to be
confounded in practice.

== The human verdict, and why it can't be resolved here

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
rule returns a _tie_ on the primary channel until an unbiased win-rate exists. The
verdict in this report therefore rests on the automatic arms; the
human read is a weak, confounded corroborator, not a third independent method. The
kit and scorer exist (the win-rate would resolve a clear \~60/40 preference at
\~100 items, a 55/45 at \~400), but a clean reading needs raters the privacy
constraint rules out.

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
voice. That strength is also the catch: because the author recognizes
his _own_ real replies, a raw detection rate measures recall as much as voice and
would _overstate_ discriminability. A meaningful run has to control for memory —
re-rating after a long delay, or restricting to contexts whose replies he does not
recall — or it cannot separate "I remember sending this" from "this sounds like me."
With outside raters barred on privacy grounds, that is why the Turing result is
reported here as genuinely prospective, not as a number this study can yet defend.

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

== From knowledge tells to a grounding layer <sec-grounding>

The voice/knowledge split is not only a way to read the tells — it is a build order.
If the catches a reader would make are mostly _knowledge_ tells, then voice is the
solved problem and the remaining gap is retrieval, not more training. This section
reports the grounding layer built against that gap, and a probe that measures, on a
level field, what it actually buys.

The mechanism is deliberately thin, because the fine-tune's whole value is a voice a
heavy prompt destroys (the dominant finding of the two-arm comparison). Durable identity facts are
distilled offline from the owner's own notes into a small store. At serving time an
intent router reads the incoming message: a vague self-intro ("розкажи про себе") is
left to the trained voice with no facts attached — in a casual register a recited
fact sheet is the wrong answer, and it is also where fabrication costs least — while a
_specific_ factual question retrieves the matching identity facts and folds a short,
in-language card into the system turn. The persona anchor stays byte-identical to
training; the card is a brief addendum, and because the system turn is masked from the
training loss it is a conditioning nudge, not the train/serve skew the thin
train==serve invariant (@sec-build) exists to avoid.

To measure it we run a bare-vs-grounded probe under the identical-prompt discipline of
Arm B: the same local fine-tune answers 30 identity questions (five decodes each,
$n = 150$ generations per condition) twice — once on the thin prompt alone (_bare_),
once with the fact card (_grounded_) — everything else held fixed. Each generation is
labelled _correct_ / _hallucinated_ / _deflected_ by an LLM judge against the known
fact. Factual correctness is far more objective than the voice judgment that stalls
the human panel above — which is why this probe _resolves_ where that one cannot; a
stratified hand-check agreed with the judge on 11 of 12 sampled labels.

#figure(
  image("../fig/f12_grounding.png", width: 98%),
  caption: [Bare vs grounded local fine-tune, $n = 150$ generations per condition.
  Left: grounding lifts the correct-fact rate from 0.05 to 0.33 (question-clustered
  95% intervals disjoint) and lowers hallucination from 0.29 to 0.18 (intervals
  overlap — a directional drop). Right: replies stay short (24 → 38 characters) and the
  "!" rate stays at 0.00 — the card adds facts without moving the voice.],
) <fig-grounding>

The result is a clean win on correctness and an honest, partial one on hallucination
(@fig-grounding). Handed no facts, the bare fine-tune confabulates fluently and in
register — asked which city he lives in, it answers "London" or "Chicago" with the
same casual confidence it brings to a real reply — and is correct on just *0.05* of
generations (95% CI 0.01–0.12), deflecting two-thirds of the time. The card raises the
correct-fact rate to *0.33* (0.19–0.47): a real sixfold gain whose interval is disjoint
from the bare model's. These are _question-clustered_ intervals — a two-stage bootstrap
that resamples the 30 probes, not the 150 correlated generations, so the five decodes
per question cannot inflate the precision (a naive Wilson interval on $n = 150$ would
understate the uncertainty; see Appendix E). Hallucination falls from *0.29* (0.16–0.43)
to *0.18* (0.09–0.27), but here the clustered intervals _overlap_, so by this report's
own disjoint-interval rule the drop is directional, not certified. The voice is preserved throughout: the
exclamation rate stays at 0.00 and mean reply length rises only 24 → 38 characters —
longer because the reply now carries a fact, still squarely inside the person's short
register. (The Latin-script rate rises 0.48 → 0.59, but that is content, not drift:
the surfaced facts carry Latin proper nouns — place and company names — so a grounded
answer is _expected_ to read more Latin.)

The same numbers size the next step honestly. Even handed the right fact, the
3-billion-parameter model still _deflects_ on about half of grounded generations
(0.49) and contradicts it on 0.18; restricting to the 27 probes that actually received
a card barely moves this (correct 0.36, hallucinated 0.19). The bottleneck is no
longer retrieval — the router fires and the card carries the fact — but the small
model's willingness to _use_ a fact it has been given. That points the next investment
not at more retrieval but at a larger or grounding-tuned local model, and is exactly
the kind of claim the voice/knowledge split was built to make. Two limits scope it:
the probe targets facts the vault contains, so it measures retrieval, injection, and
register-preservation — the live deployment scenario — not generalization to facts the
vault never held (those still correctly deflect); and the judge, far more reliable on
fact-matching than on voice, is a spot-checked proxy, not an oracle. Within that
envelope the reading is clean: grounding makes the local fine-tune measurably _more
right_ and somewhat _less wrong_ on identity questions without touching the voice —
turning the voice/knowledge decomposition from an analytic frame into an actionable
one.
