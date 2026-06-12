= What we learned <sec-learned>

== The verdict

Under the rule fixed in advance, the decision favors the fine-tune — but the strength
of the claim differs sharply by arm, and it is worth being precise. On a level field
(Arm B) the fine-tune is decisively closer to the person's reply length — a large,
per-item-consistent effect — and matches the no-`!` register, while also beating the
same-family no-LoRA Qwen base on the measured register: a genuine voice win over the
bare models. In the shipped configuration (Arm A), the
API's full machinery pulls it to a _voice tie_ on the automatic metrics — shape, the
`!` register, and, per individual message, reply length are all statistical ties (the
fine-tune keeps only a distributional length edge and more varied openers, the latter
uncorrected for multiple comparisons). The shipped-arm decision is therefore not a
demonstrated voice advantage; it is a tie that cost, privacy, and offline capability
break toward the local model. And the pre-registered _primary_ channel — an unbiased
human win-rate — is unresolved (recall bias; no outside raters). Stated cleanly:
_the fine-tune beats the bare models on voice outright, matches the shipped product on
the measurable register, and wins decisively on cost, privacy, and offline operation._
These are surface metrics, not a certified _feels-like-me_. On the dimensions we _can_
measure, the fine-tune is at least the shipped product's equal on register and, once
\$0 cost, privacy, and offline operation are counted, the better _deployment_. Whether
it is the better _voice_ replica is deliberately left unclaimed: that rests on the
pre-registered primary human channel, which is unresolved, and on surface proxies we
cannot yet validate against it. So the honest headline is narrower than a winner, and
more interesting: _an elaborate production stack serves mainly to reach a tie with
where a small local fine-tune already sits — and the residual differences are
deployment, not demonstrated voice._

== Threats to validity

The result is honest only with its limits stated plainly.

- *Construct validity.* Every automatic metric is a surface proxy; none measures
  "feels like the person" directly. The blind human panel is the only direct
  measure, and the metric↔human agreement that would validate the proxies cannot be
  obtained here — an unbiased panel is precluded by the corpus's privacy. A
  literature-standard authorship detector @stamatatos2009survey, built and validated
  here (@sec-relative), _corroborates_ rather than transcends the surface metrics:
  per-message lexical authorship in this casual register is thin (held-out ROC-AUC 0.57),
  and the strong reply-level detector (AUC 0.99) is mostly length and structure. It
  confirms the LoRA matches the owner's reply distribution where the bare API does not,
  but supplies no _independent_ lexical-voice validation — so the human panel remains the
  only route to certifying "feels like me".
- *The human channel is doubly blocked, so the primary verdict is unresolved.* The
  only rater with standing is the owner, and he is _recall-biased_: he recognizes his
  own messages, so his ability to tell model from real conflates memory with voice.
  Recruiting unbiased raters is precluded by the private content of the corpus. So the
  pre-registered primary statistic (the human win-rate) is not merely unrated but
  hard to establish cleanly at all — a real limitation, not a to-do. The automatic
  arms carry the decision; the human read is a confounded corroborator.
- *Single headline decode per item* at temperature 0.8. The bootstrap intervals
  resample item indices only, so they capture which-items sampling noise but not decode
  stochasticity. This was load-bearing for Arm A — its length gap is only \~3.6
  characters — so we re-decoded the LoRA arm twice more: the delta CI excludes zero in
  all three decodes (Δ = 3.57, 4.05, 4.42; App. A), so the one surviving Arm-A edge
  survives decode variance, even as the per-message effect stays a wash. The check
  bounds decode noise on the arm where it mattered without fully characterising it; Arm
  B's 126-character gap was never at risk.
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
  directional, not as tested wins. On provenance: the served GGUF is now pinned by
  SHA-256 (App. A) and the comparison harness logs each run to MLflow, but the
  `llama.cpp` build behind the originally-reported decodes is not separately stamped —
  a reproducibility gap narrowed, not fully closed, on the local-model numbers.

== Conclusion and future work

For one person, a 3-billion-parameter local fine-tune matches a fully-equipped
frontier-API product on every surface register we can measure — at zero marginal cost,
no phone-home, and no leak surface — while the product's prompt-plus-retrieval-plus-lever
machinery serves mainly to recover ground the fine-tune already holds. Whether it is the
better _voice_ replica, as opposed to the better _deployment_, is the question the
(unresolved) human panel exists to answer.

The clearest next steps follow the open threads. Rate the two built panels, turning
the qualitative human verdict into a win-rate with a Wilson interval and running the
Turing test against the person's real replies. Add one or two raters for an
inter-rater agreement check. The voice-versus-knowledge split already pointed past
voice to grounding, and the layer built here (@sec-grounding) acts on it: a thin,
intent-routed fact card lifts the local model's correct-identity-answer rate from 0.05
to 0.33 (question-clustered intervals disjoint) with the voice intact, while only
directionally trimming hallucination. The gap it exposes is the next target — a 3B model
under-uses even a fact it is handed, so the remaining identity accuracy lives in a larger
or grounding-tuned local model, not in more retrieval. And for a paper-grade leak-free
claim, re-train the fine-tune on a single unified split (the decode-variance robustness
pass is now done, App. A). The frontier is no longer "fine-tune or API" — that question
is answered on measurable register — but "fine-tune versus the person".
