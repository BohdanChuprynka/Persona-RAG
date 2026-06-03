#pagebreak()
= Appendices

== A · Reproduce

All runs are deterministic given the seed. From the repo root:

```bash
# Arm B (controlled), two seeds
make compare                                              # n=300, seed 0 -> main
uv run python scripts/compare_persona.py --n 150 --seed 1 --name seed1
# Arm A (production) + its ablations
make compare-arma                                         # shipped levers -> armA
uv run python scripts/compare_persona_armA.py --n 300 --learned --name armA_learned
uv run python scripts/compare_persona_armA.py --n 60 --leak-on --name armA_leakon
# Per-item effect sizes + per-language + the report
uv run python scripts/compute_effect_sizes.py --name main   # also: armA, armA_learned
uv run python scripts/score_by_language.py --name armA
bash report/build.sh                                      # figures + compile the PDF
```

Pinned parameters: temperature 0.8, `max_tokens` 200, 2000 bootstrap resamples;
API `gpt-4o-mini`; LoRA Qwen2.5-3B served as GGUF `Q5_K_M` via `llama.cpp`; the
hold-out is the recipient-stratified `eval_split_for`. (Known provenance gap: the
adapter / quant / `llama.cpp` build hash is not stamped in the run record.)

== B · Metric formulas

- *Message shape* — Jensen-Shannon divergence of the bubble-count histograms,
  $ "JS"(P, Q) = 1/2 D_"KL"(P || M) + 1/2 D_"KL"(Q || M), quad M = (P + Q) / 2, $
  in base 2 so the value lies in $[0, 1]$.
- *Reply length* — the 1-Wasserstein distance between per-bubble character-length
  distributions, $W_1(P, Q) = integral_(-oo)^(oo) |F_P(t) - F_Q(t)| dif t$.
- *Opener entropy* — $H = -sum_i p_i log_2 p_i$ over reply first-words.
- *Cliff's delta* — $delta = (\#{x > y} - \#{x < y}) / (n_x n_y)$ over the per-item
  length errors of the two backends; magnitude thresholds 0.147 / 0.33 / 0.474.
- *Win / detection intervals* — Wilson score 95% interval on the binomial
  proportion @wilson1927.

== C · Full per-run results

#table(
  columns: 7,
  stroke: none,
  inset: (x: 7pt, y: 5pt),
  align: (left, center, right, right, right, right, right),
  table.hline(),
  table.header(
    [Run], [n], [Shape A/L], [Length A/L], [Excl. A/L], [Cliff δ], [\$/1k],
  ),
  table.hline(stroke: 0.4pt),
  [main (B)], [300], [0.052 / 0.024], [128.8 / 2.9], [0.65 / 0.00], [0.949], [0.082],
  [seed1 (B)], [150], [0.081 / 0.042], [134.7 / 1.5], [0.62 / 0.00], [—], [0.089],
  [armA (A)], [300], [0.035 / 0.034], [6.97 / 3.41], [0.00 / 0.00], [0.043], [0.370],
  [armA-learned (A)], [300], [0.034 / 0.018], [7.24 / 2.97], [0.03 / 0.00], [—], [0.371],
  table.hline(),
),
The $n = 60$ leak-ablation arms (`armA_leakon` / `armA_leakoff`) are underpowered and
omitted here; their only role is the leak-guard proof (@fig-leak) and the forest plot
(@fig-forest). "A/L" = API / LoRA; LoRA cost is \$0 (local) throughout.

== D · Supporting distributions

The headline figures summarise these underlying distributions, shown here for the
controlled arm (and the API's tic profile under the production stack).

#figure(image("../fig/voice_tics_armB.png", width: 82%), caption: [Arm B voice tics vs. the real reference (closer to gray is more like the person).]) <fig-tics-b>

#figure(image("../fig/length_dist_armB.png", width: 82%), caption: [Arm B per-message length distribution: the API writes far longer messages; the LoRA hugs the real curve.]) <fig-lendist>

#figure(image("../fig/shape_dist_armB.png", width: 82%), caption: [Arm B bubble-count distribution per reply.]) <fig-shapedist>

#figure(image("../fig/voice_tics_armA.png", width: 82%), caption: [Arm A tic profile: the production machinery suppresses the API's `!` and tightens its register.]) <fig-tics-a>
