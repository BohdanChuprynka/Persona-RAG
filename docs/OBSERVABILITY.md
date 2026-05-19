# Observability

Three signals, three stores.

| Concern | Tool | Where |
|---|---|---|
| LLM chain traces (per-message LangGraph runs) | LangSmith | https://smith.langchain.com |
| Eval runs (per-script run, persona-match metrics) | MLflow | `localhost:5000` via docker-compose |
| Service logs (bot lifecycle, errors, business events) | structlog | stdout (JSON), redirected to file in prod |

Each tool has one job. No overlap.

## LangSmith (LLM tracing)

Every LangGraph invocation is auto-traced when these env vars are set:

```
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=<your key>
LANGCHAIN_PROJECT=persona-rag
```

You get a trace per incoming message showing:
- Each LangGraph node's input/output/duration
- Retrieval query → matched documents (with scores)
- LLM call: full prompt, full response, token counts, cost
- Errors with full stack

UI usage:
- Filter by `user_id` (set via `traceable` metadata) to debug what one user gets
- Filter by latency to find slow paths
- Filter by model to compare gpt-4o-mini vs gpt-4o runs side-by-side

**Privacy:** every prompt + completion goes to LangSmith. If that's not acceptable, set `LANGCHAIN_TRACING_V2=false`. Run breakage: none — graph still runs, just no traces.

**Cost:** Free tier covers individual use. Persona-RAG runs locally on demand → well within.

## MLflow (eval tracking)

Each `make eval` run produces one parent run + child runs (one per metric — stylometry, perplexity-proxy, shadow-A/B). See [`EVAL.md`](EVAL.md) for the run anatomy.

Local backends:
- File backend (default for solo dev): `MLFLOW_TRACKING_URI=file:./mlruns`
- Server backend (via docker-compose): `MLFLOW_TRACKING_URI=http://localhost:5000`

Server mode is preferred when iterating on prompts — you can leave the UI open and refresh.

Conventions:
- One experiment: `persona-rag-eval`
- Run names: `YYYY-MM-DD-<short-tag>` (e.g. `2026-05-17-baseline`, `2026-05-17-alpha-0.5`)
- Tags: `dataset_version`, `persona_name`, `prompt_version`
- Params: every dial that affects retrieval/generation (top_k, alpha, model, temperature, recency_half_life)
- Metrics: see EVAL.md
- Artifacts: per-turn diff CSV, sample replies markdown

The MLflow UI replaces what a FastAPI admin dashboard would have done. Cleaner, batteries-included.

## structlog (service logs)

`persona_rag/_logging.py` configures structlog once at startup:

```python
import structlog, sys, logging

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
)
```

Every log line is a JSON object. Bind context once per request, every line inherits it:

```python
log = structlog.get_logger()

async def on_message(msg: Message):
    with structlog.contextvars.bound_contextvars(
        user_id=msg.from_user.id,
        chat_id=msg.chat.id,
        message_id=msg.message_id,
    ):
        log.info("message_received", text_len=len(msg.text))
        ...
        log.info("reply_sent", reply_len=len(reply))
```

Event naming: lowercase snake_case verbs in past tense (`message_received`, `auth_approved`, `memory_updated`, `guardrail_triggered`). Easy to grep.

**Severities:**
- `info` — normal lifecycle
- `warning` — degraded but functional (retry succeeded, fallback used)
- `error` — exception thrown, request failed
- `critical` — admin-level: rate limit breach, guardrail blocked a slur, OpenAI down

**Persistence:** stdout in dev. In docker-compose, the bot's stdout is the container log — `docker logs persona-rag-bot`. For long-running collection: pipe to a file via the host or attach a log driver. Out of scope for v1.

## What we deliberately don't use

- **OpenTelemetry / Jaeger / Tempo** — distributed tracing for distributed systems. Persona-RAG is a single process. LangSmith already covers the per-message trace; structlog covers the rest. Adding OTel here is resume-driven complexity.
- **Sentry / Bugsnag** — error aggregation for production fleets. For a solo-run local bot, structlog `error`/`critical` lines surfaced in `docker logs` are enough.
- **Prometheus / Grafana** — service metrics for production. No SLO to track here.

## Adding observability for a new feature

Checklist when wiring a new module:

- [ ] Inject a `structlog` logger with bound context (user/chat/message IDs as available)
- [ ] Log entry + exit at info level, exceptions at error/critical
- [ ] If it's a LangGraph node, name it descriptively — that name appears in LangSmith
- [ ] If it changes a tunable (new env var), log the effective value at startup
- [ ] If it's a new eval signal, add it as a metric in `scripts/eval_persona.py` so it lands in MLflow
