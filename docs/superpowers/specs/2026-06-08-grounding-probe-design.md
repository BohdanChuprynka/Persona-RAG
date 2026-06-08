# Grounding probe — design & pre-registered methodology (2026-06-08)

## Motivation

The report (`report/`) establishes a clean decomposition: a small local fine-tune
learns the owner's *voice* in its weights, but cannot know *facts* it was never told —
so on identity questions it confabulates fluently and in-register (the canonical
failure: asked where he studies, the bare LoRA confidently names the wrong country).
Part 4 ("What the tells will tell us") and Part 5 (future work) both name the fix and
then leave it open: *knowledge tells "are addressable only by retrieval and grounding …
an Obsidian / chat-history RAG layer."*

That layer is now built (the vault fact-ingestion feature, spec
`2026-06-03-vault-fact-ingestion-design.md`). This probe **closes the loop the paper
opens**: it measures, quantitatively, whether the grounding layer removes factual
hallucination without disturbing the voice the fine-tune exists to reproduce. It also
*validates the paper's central decomposition by acting on it* — voice stays in the
weights; knowledge is injected, thinly, only where it matters.

## What is under test (the grounding layer)

- **Intent router.** A vague self-intro ("розкажи про себе") routes to the trained
  voice with no fact card (casual-register default, `INSIGHTS_SELFDESC_CARD_ENABLED=
  False`); a *specific* factual question routes to semantic retrieval over the
  user-authored vault facts and folds a short, in-language fact card into the prompt.
  Facts are injected only where fabrication actually costs something.
- **train==serve preserved.** The one-line persona anchor is byte-identical to training.
  The fact card is a brief *addendum* appended to the system turn for specific questions
  only; because the system turn is masked from training loss, this is a conditioning
  nudge, not train/serve skew in any trained token.
- **Provenance.** Facts are distilled from the owner's own Obsidian notes (durable
  identity only: bio / relationship / value / opinion); raw note text never reaches the
  serving model — only short distilled facts.

## Evaluation design

**Conditions (identical-prompt A/B, the report's Arm-B discipline).** Same probe, same
local LoRA, same decode params; the *only* difference is the fact card:

| Condition | System turn | Fact card |
|-----------|-------------|-----------|
| **bare**     | thin persona anchor | none |
| **grounded** | thin persona anchor | vault fact card (real serving path for specific Qs) |

**Probe set.** ~30 curated identity questions, mostly Ukrainian with an English
minority, each paired with the ground-truth vault fact it targets; multiple natural
phrasings per fact. Targets durable, checkable bio/value facts (location, school,
role / what-he-builds, skills & stack, prior experience, mentor, languages, goals).
Romantic/intimate facts are excluded as probe targets. The probe set and its
ground-truth are **gitignored** (`reports/main/grounding/`), never committed.

**Decodes.** K = 5 samples per probe per condition at temperature 0.8. Sampling
multiple decodes captures decode stochasticity (the report's named single-decode
limitation) and yields a per-condition *rate* rather than one lucky draw.

**Scoring rubric (pre-registered, fixed before any run).** Each generation is labelled
into exactly one of three classes by an LLM judge (`gpt-4o-mini`) given the triple
(question, ground-truth fact, candidate answer):

- **correct** — asserts the ground-truth fact (allowing paraphrase / language).
- **hallucinated** — asserts a *specific, contradicting* fact (the failure mode).
- **deflected** — commits to no checkable fact (neither right nor wrong).

A stratified sample of judge labels is hand-checked for a judge-vs-author agreement
number, reported as a validity check.

**Statistics.**
- Primary: **hallucination rate**, bare vs grounded, each with a **Wilson 95% interval**
  (reuse `persona_rag.eval.compare.wilson_ci`). The headline claim requires the grounded
  interval to lie strictly below the bare interval (disjoint).
- Secondary: **correct-fact rate** (bare ≈ 0 by construction — the local model cannot
  know these unaided — vs grounded high) and **deflection rate**.
- **Register preservation.** Voice metrics computed on the generations themselves —
  mean reply length, Latin-script rate, exclamation rate, `)` smiley rate — compared
  bare vs grounded. The card must *not* move the register: these are expected to be
  statistical ties. This supplies the generation-level number that
  `tests/eval/test_vault_register_invariance.py` (construction-level only) explicitly
  defers.

**Acceptance rule (pre-registered).** Grounding is judged effective iff (a) the grounded
hallucination-rate interval is disjointly below bare's, **and** (b) no register metric
regresses beyond sampling noise. Correct/deflect rates are reported descriptively.

## Privacy (hard constraint — same bar as the rest of the repo)

The repo has a prior-leak history and is currently PRIVATE; real personal content must
never enter tracked files.

- Probe questions, ground-truth facts, raw generations, and the results JSON live under
  `reports/main/grounding/` and are **gitignored + untracked**.
- The paper reports **aggregate rates + intervals only**, plus a single **anonymized**
  worked example (a "where do you study?" probe: bare names a confident but wrong
  *country*; grounded returns the vault's true value) — **no real fact is printed**.
- Figures contain only aggregates. The existing no-personal-facts-in-tracked-files audit
  bar is preserved; a fresh privacy sweep runs before commit, and again before any public
  flip.

## Named limitations (carried into the paper, in its register)

- **Curated-scale, not a benchmark.** The vault is a few dozen durable facts; the probe
  is a demonstration of the *mechanism* at honest scale, not a population hallucination
  rate.
- **In-vault facts.** Probes target facts the vault contains, so this measures
  retrieval + injection + register-preservation — the live deployment scenario — *not*
  generalization to facts the vault was never given (those correctly deflect or cannot
  be known).
- **LLM judge.** Factual correctness is far more objective than the voice judgment that
  stalls the human panel, but the judge is still a proxy; it is spot-checked, not
  assumed.
- **One persona, mostly Cyrillic; single build.** Same external-validity envelope as the
  rest of the report.

## Artifacts

- `persona_rag/eval/grounding.py` — pure, unit-tested core (judge-label parsing, Wilson
  aggregation, register metrics).
- `tests/eval/test_grounding.py` — TDD unit tests (no network).
- `scripts/probe_grounding.py` — runner (generate bare/grounded → judge → aggregate →
  results JSON).
- `compare-vault` Make target — wired to run the probe (replaces the current
  NOT-YET-WIRED stub that exits non-zero).
- `report/fig/` — bare-vs-grounded hallucination figure (+ optional register panel).
- `report/parts/part4.typ` (new closing section), `part5.typ` (future-work update),
  `report.typ` (abstract clause), `appendix.typ` (probe config + rubric).
