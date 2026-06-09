# Cloning a texting voice: RAG vs fine-tuning, compared fairly

> Aggregate numbers only; no private chat content. The full study — both arms, the grounding layer, and all figures — is in the research report (`report/persona-rag-report.pdf`).

## The problem

Persona-RAG is a Telegram bot that replies *in my voice*, grounded in my own exported chats. "My voice" is specific and measurable: uk/en/ru code-switching that *mirrors* whoever I'm talking to, terse multi-bubble bursts, a `)` smiley tic, basically no `!`, casual lowercase. The question that drove this experiment:

**Do you need to fine-tune a model to capture a texting voice, or does retrieval-augmented prompting of a strong API model get you there?**

Two approaches:
- **API (RAG):** `gpt-4o-mini` + a rich prompt — retrieved real past replies as few-shot, register/shape directives, decode-time nudges.
- **Fine-tune (local):** a Qwen2.5-3B QLoRA trained on my own context→reply pairs, served locally (llama.cpp), on a deliberately *thin* prompt (train == serve).

## The part nobody shows: making the comparison trustworthy

The first instinct — "run the existing eval on each backend and compare" — was **wrong, and confidently so.** A multi-agent audit of the eval stack found a disqualifying flaw: the runner scored both backends on a *temporal* hold-out, but the fine-tune had held out a *different* (recipient-stratified) split. ~90% of the "test" turns were in the model's *training* data. The fine-tune would have "won" by being graded on what it memorized.

Fixing that was most of the work:
- **One shared, model-disjoint hold-out** (recipient-stratified hash; the fine-tune never trained on it).
- **A controlled arm:** the *identical* thin prompt to both models, so the result reflects *weights*, not prompt scaffolding.
- **Paired bootstrap confidence intervals** (a difference only counts if its CI excludes 0), replicated across seeds.
- **Anti-gaming guards:** copy/leak rate vs training data, mode-collapse rate, empty-output accounting (return `NaN`, never a free 0.0).
- **A blind human-preference panel** as the actual verdict — the metrics are only proxies for "feels like me."

## Headline result (controlled arm, n=300, replicated at n=150)

Under *identical minimal prompting*:

- **Length:** the fine-tune matches my terse replies almost exactly (earth-mover distance ~2–3 chars); raw `gpt-4o-mini` is ~128–135 away — it writes paragraphs where I write fragments.
- **Punctuation:** the fine-tune learned "no `!`" (rate 0.00); the API adds them ~63% of the time.
- **Message shape** (bubbles per reply): a statistical tie.
- **Cost/latency:** local fine-tune runs at $0/reply, comparable median latency.

So: a 3B model fine-tuned on a few thousand of my replies reproduces my *register* far better than a much larger API model **when both are stripped to the same prompt** — which is exactly what fine-tuning is supposed to buy.

## The honest caveats (what a good comparison admits)

- This is the **controlled** arm — it isolates the weights. It is **not** the shipped product: the deployed API path has retrieval + directives + decode nudges built to fix these exact gaps. "Which product ships better" is a separate comparison — run in the report's production arm (Arm A), where the full API stack pulls up to a voice *tie* with the thin fine-tune.
- The fine-tune near-copies ~7–15% of training replies. For *short* casual texts this is partly unavoidable (everyone reuses `ок`, `та норм`) — but it needs a length-controlled baseline before claiming overfitting.
- Automatic metrics are surface proxies; the blind human panel is the real test.

## Takeaways

1. **The evaluation is the hard part.** A plausible-looking number can be confidently wrong; one leaked split flips the entire conclusion.
2. **Fine-tuning buys register cheaply and locally.** For voice/style (not knowledge), a small local LoRA is a strong, $0-marginal-cost option.
3. **train == serve.** The fine-tune only wins because it's served under the same thin prompt it trained on.

*Methods, metrics, and the full protocol are detailed in the research report (`report/persona-rag-report.pdf`).*
