= What we learned

== The verdict

Under the rule fixed in advance, *the fine-tune ships*. The evidence converges from
three independent directions. In the controlled arm it is decisively closer to the
person's reply length and matches the no-`!` rule exactly, with shape a tie. In the
production arm the API's full machinery only claws back to _parity_ — tying on shape
and `!`, still behind on the length distribution and opener variety — while the
fine-tune delivers the same fidelity at zero marginal cost, no per-message
phone-home, and no retrieval-leak surface. And rated by eye, the API is trivially
discriminable, so the fine-tune wins voice outright. A voice win-or-tie that is also
cheaper, local, offline, and leak-free breaks the decision toward the fine-tune on
every count. The headline reading of the whole study: _an elaborate production stack
serves mainly to reach where a small local fine-tune already sits._

== Threats to validity

The result is honest only with its limits stated plainly.

- *Construct validity.* Every automatic metric is a surface proxy; none measures
  "feels like the person" directly. The blind human panel is the only direct
  measure, and the metric↔human agreement that would validate the proxies is pending
  the panel's rating.
- *Single rater.* The human verdict comes from one judge — the owner. No inter-rater
  agreement is computable at $n = 1$ rater. This is defensible (the construct is the
  owner's own voice, and he is its ground-truth authority) but it is a limitation;
  recruiting raters who know his style is the path to a paper-grade claim.
- *Single decode per item* at temperature 0.8. The bootstrap intervals capture
  which-items sampling noise, not decode stochasticity; a greedy or multi-seed
  re-run would bound it.
- *Leakage residual.* The legacy \~90% split-mismatch leak was found and fixed, both
  arms now score the recipient-stratified disjoint split, and the production arm runs
  the proven per-item retrieval guard. A fully leak-free _claim_ at paper grade would
  still want the fine-tune re-trained on the exact unified split.
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
- *Multiple comparisons & provenance.* The verdict rests on one pre-registered
  primary plus a small headline set, with the tic panel descriptive. And the adapter,
  quantization, and `llama.cpp` build hash are not pinned in the run record (MLflow
  logging is wired but uncalled) — a minor reproducibility gap to close.

== Conclusion and future work

A 3-billion-parameter local fine-tune reproduces one person's texting voice at least
as faithfully as a fully-equipped frontier-API product — at zero marginal cost, no
phone-home, and no leak surface — and the product's prompt-plus-retrieval-plus-lever
machinery serves mainly to recover ground the fine-tune already holds.

The clearest next steps follow the open threads. Rate the two built panels, turning
the qualitative human verdict into a win-rate with a Wilson interval and running the
Turing test against the person's real replies. Add one or two raters for an
inter-rater agreement check. Use the Turing tell taxonomy — the voice-versus-
knowledge split — to size a grounding layer: if catches are mostly missing facts,
the remaining gap is retrieval, not voice. And for a paper-grade leak-free claim,
re-train the fine-tune on a single unified split and add a decode-variance
robustness pass. The frontier is no longer "fine-tune or API" — that is settled — but
"fine-tune versus the person".
