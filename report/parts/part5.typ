= What we learned

== The verdict

Under the rule fixed in advance, the decision favors the fine-tune — but the strength
of the claim differs sharply by arm, and it is worth being precise. On a level field
(Arm B) the fine-tune is decisively closer to the person's reply length — a large,
per-item-consistent effect — and matches the no-`!` register, with shape a tie: a
genuine voice win over the bare model. In the shipped configuration (Arm A), the
API's full machinery pulls it to a _voice tie_ on the automatic metrics — shape, the
`!` register, and, per individual message, reply length are all statistical ties (the
fine-tune keeps only a distributional length edge and more varied openers, the latter
uncorrected for multiple comparisons). The shipped-arm decision is therefore not a
demonstrated voice advantage; it is a tie that cost, privacy, and offline capability
break toward the local model. And the pre-registered _primary_ channel — an unbiased
human win-rate — is unresolved (recall bias; no outside raters). Stated cleanly:
_the fine-tune beats the bare model on voice outright, matches the shipped product on
voice, and wins decisively on cost, privacy, and offline operation._ These are surface
metrics, not a certified _feels-like-me_ — but on the dimensions we can actually
measure the conclusion is not in doubt: the local fine-tune is at least the equal of
the shipped product on the measured register, and — once \$0 cost, privacy, and offline
operation are counted — the better overall replica. The only thing left unresolved is
the subjective human seal, and for the reasons below it may stay that way — a
limitation on the validation, not a hole in the metric verdict. The headline reading of the study:
_an elaborate production stack serves mainly to reach a tie with where a small local
fine-tune already sits — and the residual differences are deployment, not voice._

== Threats to validity

The result is honest only with its limits stated plainly.

- *Construct validity.* Every automatic metric is a surface proxy; none measures
  "feels like the person" directly. The blind human panel is the only direct
  measure, and the metric↔human agreement that would validate the proxies cannot be
  obtained here — an unbiased panel is precluded by the corpus's privacy.
- *The human channel is doubly blocked, so the primary verdict is unresolved.* The
  only rater with standing is the owner, and he is _recall-biased_: he recognizes his
  own messages, so his ability to tell model from real conflates memory with voice.
  Recruiting unbiased raters is precluded by the private content of the corpus. So the
  pre-registered primary statistic (the human win-rate) is not merely unrated but
  hard to establish cleanly at all — a real limitation, not a to-do. The automatic
  arms carry the decision; the human read is a confounded corroborator.
- *Single decode per item* at temperature 0.8 — and this is load-bearing for Arm A,
  not a minor caveat. The bootstrap intervals resample item indices only, so they
  capture which-items sampling noise but not decode stochasticity. The Arm-A length
  gap is 3.6 characters: its CI excludes zero, but that gap is small enough to sit
  within plausible re-decode noise, and nothing here shows it survives a re-decode. A
  greedy or multi-seed pass would bound it; Arm B's 126-character gap is not at risk.
- *Leakage residual.* The legacy \~90% split-mismatch leak was found and fixed, both
  arms now score the recipient-stratified disjoint split, and the production arm runs
  a per-item retrieval guard. But that guard is _exact-match_ — it removes the gold
  turn by id and verbatim same-context text only; a near-paraphrase under a different
  id, or a thread-adjacent turn sharing the incoming context, is not caught and sits
  below the top-similarity flag. So "removes the exact answer-key" is exact; "leak-free"
  is not — the residual near-duplicate rate is unmeasured, and a paper-grade claim
  would add a similarity-based guard and re-train on the exact unified split.
- *External validity.* The hold-out is 87% Cyrillic and one person's chat history
  (\~300 held-out turns); the aggregate verdict is essentially the Cyrillic result,
  English is low-$n$, and both backends degrade there. Claims are about this persona,
  not personas in general.
- *Confounds, by design.* In the production arm the backend moves together with
  prompt, retrieval, and levers; the controlled arm exists precisely to isolate the
  weights. Equal `max_tokens` is also not equal character budget across the two
  tokenizers — a small caveat on any length metric.
- *Aggregate cross-arm.* Arm B and Arm A score the same hold-out _distribution_ but
  not byte-identical item sets, so cross-arm deltas are aggregate, not paired.
- *Replay fidelity.* The production-arm state (empty first-contact memory, session
  reconstructed from each item's own context, insights from time-of-run tables, the
  runtime `ctx[-1]` query) is a faithful reconstruction of live serving, not the live
  system itself.
- *Multiple comparisons & provenance.* About a dozen metrics are reported. The two
  headline distances carry bootstrap CIs, but the tic metrics (exclamation rate,
  opener entropy) are *descriptive only* — they were _not_ given a Holm-Bonferroni
  correction, so where they point toward the fine-tune they should be read as
  directional, not as tested wins. And the adapter, quantization, and `llama.cpp`
  build hash are not pinned in the run record (MLflow logging is wired but uncalled) —
  a reproducibility gap on the local-model numbers.

== Conclusion and future work

A 3-billion-parameter local fine-tune reproduces one person's texting voice at least
as faithfully as a fully-equipped frontier-API product — at zero marginal cost, no
phone-home, and no leak surface — and the product's prompt-plus-retrieval-plus-lever
machinery serves mainly to recover ground the fine-tune already holds.

The clearest next steps follow the open threads. Rate the two built panels, turning
the qualitative human verdict into a win-rate with a Wilson interval and running the
Turing test against the person's real replies. Add one or two raters for an
inter-rater agreement check. The voice-versus-knowledge split already pointed past
voice to grounding, and the layer built here (@sec-grounding) acts on it: a thin,
intent-routed fact card lifts the local model's correct-identity-answer rate from 0.05
to 0.33 with the voice intact, while only directionally trimming hallucination. The gap
it exposes is the next target — a 3B model under-uses even a fact it is handed, so the
remaining identity accuracy lives in a larger or grounding-tuned local model, not in
more retrieval. And for a paper-grade leak-free claim,
re-train the fine-tune on a single unified split and add a decode-variance
robustness pass. The frontier is no longer "fine-tune or API" — that is settled — but
"fine-tune versus the person".
