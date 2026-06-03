= Can we trust the measurement?

Before any comparison can mean anything, the evaluation itself has to be trustworthy.
This part is the part most write-ups skip: the metrics, the leak that was found and
fixed, the harness, and the acceptance rule fixed _before_ the results.

== Measuring a voice

No single number captures "sounds like the person", so the evaluation uses a small
vector of surface metrics, each targeting one of the voice fingerprints from Part I.
The two headline distances compare _distributions_ against the real held-out replies:

- *Message shape* ($"shape"_"JS"$): the Jensen-Shannon divergence between the
  generated and real distributions of bubble-count per reply (how many separate
  messages a reply is split into). Bounded to $[0, 1]$; lower is closer. It catches
  a constant-3-bubble mode collapse that a mean would score as perfect.
- *Reply length* ($W_1$): the 1-Wasserstein (earth-mover) distance @rubner2000emd
  between the per-bubble character-length distributions, in characters. Lower is closer.

Alongside these run a set of register rates: the exclamation rate, the `)` smiley
rate (detected by unbalanced close-parens so real parentheticals do not
false-positive), the Latin-script rate (the code-switch signal), opener entropy
($H = -sum_i p_i log p_i$ over first words — higher is more varied), and the
distinct-reply rate (a mode-collapse guard). A copy / near-copy rate against the
training replies guards against memorization, but only ever read against a measured
_natural floor_ — the rate at which the person's own held-out replies coincide with
their past replies, because short casual texts ("ok", "+") recur naturally.

== The audit: a \~90% leak, found and fixed

The project's original evaluation runner was structurally unfair, on a
disqualifying count. It scored the model on the _temporal_ `eval_split`, but the
fine-tune had held out the _recipient-stratified_ `eval_split_for` (@fig-d2). The
two are nearly disjoint partitions, so roughly *90% of the "held-out" turns the
runner scored were in the fine-tune's training pool*, while the API had seen none —
a one-sided leak that flattered the fine-tune on every metric. The same runner also
confounded the backend with the prompt, the retrieval, and the decode levers (all
of which move together with "which backend"), and reported point estimates with no
uncertainty at all. No "fine-tune beats RAG" claim from that runner was defensible.
Catching this is the foundation everything else rests on.

== The fair harness

The replacement is a small, pure, unit-tested scoring core. Both backends are
scored on the _same_ recipient-stratified, LoRA-disjoint hold-out. Every headline
distance carries a *paired bootstrap 95% confidence interval* (2000 resamples), and
a difference counts only if its interval excludes zero. Degenerate generations
return `NaN`, never `0.0`, so an all-empty backend can never score an artificial
perfect distance. The screen runs at $n = 300$, which drops the shape-divergence
sampling-noise floor from \~0.06 (at $n = 80$) to \~0.02.

== Two arms: model versus product

A single comparison cannot answer both "which _model_ is better" and "which
_product_ ships better", because the deployed API bundles a rich prompt, retrieval,
and decode levers with its weights. So the evaluation runs two arms (@fig-d3). *Arm
B (controlled)* gives both backends the identical thin prompt, with no retrieval,
directives, or levers — it isolates the weights. *Arm A (production)* pits the fully
shipped API stack against the thin LoRA — it measures the real deployed gap.

#figure(
  include "../diagrams/d3_two_arm.typ",
  caption: [The two-arm design. Arm B isolates the model (identical thin prompt);
  Arm A isolates the product (shipped API stack vs. thin LoRA). Both feed one
  scorer with paired bootstrap CIs and the leak/copy guards.],
) <fig-d3>

== The retrieval leak guard

The production arm has its own contamination risk: because the held-out gold turns
sit in the very corpus the API retrieves from, retrieval can hand the model the
exact answer key for the item being scored. Measured directly, with exclusion
disabled, *28% of items (17 of 60)* retrieved their own gold reply into the
few-shot pool. A per-item guard excludes the scored turn's id from retrieval, driving
the *exact answer-key* — the gold turn by id, and its verbatim text under the same
context — to *zero*, while leaving the mean top-1 similarity essentially unchanged
(0.386 vs. 0.389): the guard removes contamination, not retrieval quality
(@fig-leak). It is an exact-match guard, so a near-paraphrase under a different id
could in principle slip through; here no retrieved neighbour exceeded 0.9 similarity,
but that residual near-duplicate risk is named among the limitations.

#figure(
  image("../fig/f1_leak_guard.png", width: 72%),
  caption: [The retrieval leak guard. Disabling exclusion lets the API retrieve the
  held-out gold reply for 28% of items; the per-item guard removes all of it without
  degrading top-1 similarity. $n = 60$ per condition.],
) <fig-leak>

== The rule, fixed in advance

To defend against metric-shopping, the definition of "better" and the ship rule were
fixed before the results. The *primary verdict is the blind human win-rate*: a
backend wins on voice only if its Wilson 95% interval over the other excludes 0.5;
otherwise it is a voice tie. Guardrails override a numeric win — a win by
memorization (copy rate above the natural floor) or mode collapse (distinct-reply
rate too low) does not count. And ties break toward the local fine-tune on
cost, latency, and offline capability, because those are its reason for existing.
