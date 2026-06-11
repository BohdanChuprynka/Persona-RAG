= Relative fidelity <sec-relative>

The first fidelity question is relative: does the fine-tune beat the _strong_
baseline — the full gpt-4o-mini product — at sounding like the person? This part
answers it across the two arms, with ablations that explain _why_, and an effect
size that keeps the answer honest.

== Arm B: on a level playing field

Under the identical thin prompt, the fine-tune is decisively closer on length and
matches the no-`!` rule exactly, while message shape is a statistical tie. The
result replicates across two seeds (@tab-armB, @fig-armB).

#figure(
  table(
    columns: 6,
    stroke: none,
    inset: (x: 5pt, y: 5pt),
    align: (left, right, right, right, left, center),
    table.hline(),
    table.header([Metric], [API], [LoRA], [Real], [Δ API−LoRA (95% CI)], [verdict]),
    table.hline(stroke: 0.4pt),
    [Message shape (JS, ↓0)], [0.052], [0.024], [0], [+0.028 (-0.014, 0.073)], [tie],
    [Reply length ($W_1$, ↓0)], [128.8], [*2.9*], [0], [+125.9 (107.6, 142.4)], [*LoRA*],
    [Exclamation rate (→)], [0.651], [*0.000*], [0.00], [—], [LoRA],
    [Code-switch (Latin, →)], [0.023], [*0.260*], [0.235], [—], [LoRA],
    [Opener entropy (→)], [5.02], [*5.77*], [6.96], [—], [LoRA],
    [Cost / 1k replies], [\$0.082], [*\$0*], [—], [—], [LoRA],
    table.hline(),
  ),
  caption: [Arm B (controlled), $n = 300$, replicated at $n = 150$ seed 1. Identical
  thin prompt to both backends. *Real* is the person's own held-out value — the
  target: distance metrics (↓0) aim at zero, rate metrics (→) aim at the Real value.
  The length win is large and its CI excludes zero in both seeds; on the rate rows the
  LoRA is closer to the person on every count — notably the code-switch register, which
  the API essentially drops (0.02 vs the person's 0.24). Two honest notes: both
  backends still undershoot the person's opener variety (6.96), and both barely produce
  the `)` smiley (real rate 0.051, API 0.004, LoRA 0.008 — a tic the bare LoRA does
  _not_ reproduce). Rate rows are descriptive (no CI / multiple-comparison correction).],
) <tab-armB>

#figure(
  image("../fig/f2_armB_headline.png", width: 88%),
  caption: [Arm B headline distances. Message shape is a tie; reply length favors the
  LoRA decisively (shorter is more like the person).],
) <fig-armB>

== Arm A: the product fully equipped

Giving the API its entire shipped advantage — rich prompt, retrieval, levers —
changes the picture but does not overturn it (@tab-armA, @fig-armA-h). Shape and
`!` are now ties; the LoRA stays ahead on reply-length distribution and opener
variety, at zero cost.

#figure(
  table(
    columns: 6,
    stroke: none,
    inset: (x: 5pt, y: 5pt),
    align: (left, right, right, right, left, center),
    table.hline(),
    table.header([Metric], [API], [LoRA], [Real], [Δ API−LoRA (95% CI)], [verdict]),
    table.hline(stroke: 0.4pt),
    [Message shape (JS, ↓0)], [0.0353], [0.0339], [0], [+0.001 (-0.040, 0.040)], [tie],
    [Reply length ($W_1$, ↓0)], [6.97], [3.41], [0], [+3.57 (1.53, 4.66)], [distrib.#super[†]],
    [Exclamation rate (→)], [0.000], [0.000], [0.00], [—], [tie],
    [Code-switch (Latin, →)], [0.003], [*0.118*], [0.224], [—], [LoRA],
    [Opener entropy (→)], [3.70], [*5.76*], [6.86], [—], [LoRA],
    [Cost / 1k replies], [\$0.37], [*\$0*], [—], [—], [LoRA],
    [Latency (p50)], [0.96s], [1.01s], [—], [—], [\~tie],
    table.hline(),
  ),
  caption: [Arm A (production), $n = 300$, leak guard active (`id_leaks` = 0). *Real* is
  the person's own held-out value (the target). The shipped machinery ties the LoRA on
  shape and `!`. #super[†]On reply length the LoRA leads in _distribution_ (the corpus
  Wasserstein CI excludes zero) — an edge that holds across three independent re-decodes
  (Δ = 3.57, 4.05, 4.42; every CI excludes zero, App. A) — but _per individual message_
  the effect is negligible (Cliff's δ = 0.04 — a tie; @sec-effect). The LoRA keeps the
  code-switch and opener edges, though here even the API's rich prompt barely
  code-switches (0.003 vs the person's 0.224). Rate rows are descriptive (no CI /
  multiple-comparison correction).],
) <tab-armA>

#figure(
  image("../fig/f3_armA_headline.png", width: 88%),
  caption: [Arm A headline distances. Even fully equipped, the API only reaches a
  tie on shape and trails on the length _distribution_ (per-message, length is a tie
  too; @sec-effect).],
) <fig-armA-h>

== What the machinery buys

The two arms together isolate exactly what the production scaffold contributes on
the API side (@fig-machinery). It is decisive engineering: reply-length distance
collapses from 128.8 to 7.0, and the exclamation rate from 0.65 to 0.00. But it only
brings the API up to where the bare fine-tune already sat — the LoRA's length
distance barely moves (2.9 → 3.4). The one counter-intuitive cost: the few-shot and
directives _homogenize_ the API's openers, dropping its entropy from 5.02 to 3.70,
below even the bare model.

#figure(
  image("../fig/f4_machinery.png", width: 92%),
  caption: [What the production machinery buys (API side, Arm B → Arm A). The API
  plummets onto the LoRA's flat reference line; the scaffold is what makes it
  competitive, and only reaches parity.],
) <fig-machinery>

== Steered or learned?

Is the API's perfect no-`!` an artifact of the hard-coded logit bias? Re-running Arm
A with the levers off shows it is mostly _earned_ by the prompt: the exclamation
rate falls from 0.651 (bare) to 0.033 (rich prompt, no bias), and the shipped bias
supplies only the final nudge to 0.000 (@fig-steered). Everything else is
lever-insensitive, and the length-distribution delta is unchanged with the levers off
(Δ 4.27, CI (2.51, 5.07)). The levers move the tic, not the verdict.

#figure(
  image("../fig/f5_steered_vs_learned.png", width: 88%),
  caption: [Steered vs. learned (Arm A). The logit bias only finishes a job the rich
  prompt mostly does; the length-distribution delta favors the LoRA either way.],
) <fig-steered>

== Per-language

The verdict is essentially the Cyrillic result, where 87% of the data lives: there
the LoRA clearly leads on the length _distribution_ (@fig-lang). On the small English slice ($n = 27$) it
is a tie with wide intervals — and, separately, _both_ backends are markedly worse
in English than in Cyrillic. That is a shared weakness on a minority register, not a
differentiator, and is reported as low-confidence given the sample.

#figure(
  image("../fig/f6_by_language.png", width: 92%),
  caption: [Per-language fidelity (Arm A). Cyrillic ($n = 261$) drives the verdict;
  English ($n = 27$) is a tie and a shared weakness for both backends. The remaining
  \~12 mixed-script items fall below the per-language reporting threshold.],
) <fig-lang>

== Effect size: how big, and how consistent <sec-effect>

A corpus-level distance with a CI does not say how _consistent_ an advantage is
across individual messages. A per-item effect size does (Cliff's δ @cliff1993), and it
splits the two arms sharply — which is itself informative. On Arm B the per-item
length-error effect is overwhelming: *δ = 0.949 (large)*, with the LoRA closer on
*292 of 299* decisive items (sign-test $p approx 8 times 10^(-77)$; Wilcoxon
signed-rank $z = 14.9$). On Arm A it is *negligible*: *δ = 0.043*, the LoRA closer on
just 147 of 293 — a per-item coin-flip (sign-test $p = 1.0$, Wilcoxon $p = 0.54$, both
confirming the wash).

This is not a contradiction. Arm A's LoRA length advantage is _distributional_ (the
corpus earth-mover CI still excludes zero), but per _individual message_ the
production machinery has closed the gap to a wash. In other words, the machinery's
length fix is so complete that the per-message edge the bare LoRA held in Arm B all
but disappears under the product — a sharper reading than "the LoRA wins length",
and a more honest one. The forest plot (@fig-forest) shows every run at once: length
excludes zero in both Arm B seeds and both Arm A passes; shape is a tie everywhere;
the underpowered $n = 60$ leak-ablation arms straddle zero, as expected. And the lone
surviving Arm-A edge is not single-decode luck: re-decoding the LoRA arm twice more
(App. A) leaves the length delta excluding zero every time (Δ = 3.57, 4.05, 4.42),
even as the per-message effect stays a wash.

#figure(
  image("../fig/f7_forest.png", width: 98%),
  caption: [Effect sizes across all runs: API−LoRA delta with bootstrap 95% CIs
  (color = excludes zero). The seed-1 replication agrees with the primary run, and
  removing the retrieval leak does not move either verdict.],
) <fig-forest>

== Not memorization

Could the fine-tune simply be parroting training lines? The evidence says no, with a
caveat. Its copy / near-copy rate (0.103) sits near the measured _natural floor_ — the
rate at which the person's own held-out replies coincide with their past replies
(0.070) — because short casual texts recur for anyone (@fig-copy); the API, writing
novel prose, sits near zero. Two honest qualifiers: the detector is a _capped_ proxy,
so 0.103 is a lower bound, and neither rate carries a CI, so the \~3-point gap above
the floor is not formally tested. Read conservatively, the fine-tune reuses text at
roughly the rate a real person repeats themselves — not the signature of memorization,
but not proven flush with the floor either.

#figure(
  image("../fig/f9_copy_floor.png", width: 70%),
  caption: [Copy / near-copy against the training replies, with the human reuse floor
  as a reference band. The LoRA sits just above the floor; the API rarely reuses text.],
) <fig-copy>

== The operational case

Finally, the deployment trade-off (@fig-ops). The two are near-parity on median
latency, and the API even has a tighter tail — but it reaches that parity by shipping
\~2,400 input tokens of prompt and retrieved context per reply against \~210 for the
thin LoRA, at \$0.37 per thousand replies versus \$0, with a per-message embedding
call and the retrieval leak surface the guard exists to close.

#figure(
  image("../fig/f8_operational.png", width: 96%),
  caption: [Operational profile (Arm A). The LoRA is free and lean; the API pays a
  \~11× input-token tax and a per-call fee to reach parity on shape and the `!`
  register.],
) <fig-ops>
