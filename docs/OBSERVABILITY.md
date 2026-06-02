# Observability

Three signals, three stores. Each tool has one job, no overlap.

| Concern | Tool | Where |
|---|---|---|
| LLM chain traces (per-message LangGraph runs) | LangSmith | https://smith.langchain.com |
| Eval runs (persona-match metrics) | MLflow | `http://localhost:5001` via docker-compose, or a local file store |
| Service logs (bot lifecycle, errors, business events) | structlog | stdout as JSON, the container log under docker-compose |

## structlog (service logs)

`persona_rag/_logging.py` owns logging config. `configure_logging(level=logging.INFO)` is called once at startup (from `amain()` in `persona_rag/bot/main.py`) and installs this processor chain:

```python
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(level),
    logger_factory=_DynamicStdoutLoggerFactory(),
    cache_logger_on_first_use=False,
)
```

Every log line is a JSON object on stdout, with an ISO timestamp and the log level merged in.

**Why the custom factory.** `_DynamicStdoutLoggerFactory` returns `structlog.PrintLogger(sys.stdout)`, re-resolving `sys.stdout` on every logger construction. structlog's built-in `PrintLoggerFactory` captures the stdout handle once. When pytest's `capsys`/`capfd` swaps `sys.stdout` mid-suite and that captured stream is later closed, log calls raise `ValueError: I/O operation on closed file`. Looking up `sys.stdout` per call avoids that. `cache_logger_on_first_use=False` is paired with it on purpose: caching the bound logger would defeat the per-call re-resolution and reintroduce the closed-file bug.

**Getting a logger.** Call `get_logger(name=None)` from `persona_rag._logging`. It is a thin typed wrapper over `structlog.get_logger`. Modules across the package use it, including `bot/main.py`, `bot/handlers/chat.py`, the graph nodes under `graph/nodes/`, the ingest pipeline, and the insights consolidator and verifier.

**Binding context.** `merge_contextvars` sits first in the processor chain, so anything bound with `structlog.contextvars.bound_contextvars(...)` is merged into every line inside the block. The chat handler (`bot/handlers/chat.py`) logs a single event per message rather than binding a block:

```python
log = get_logger()

log.info("message_processed", user_id=user.id, reply_len=len(final.get("reply", "")))
```

For a longer flow where you want the same context on multiple lines, bind it once:

```python
with structlog.contextvars.bound_contextvars(user_id=user.id, chat_id=message.chat.id):
    ...
```

**Event naming.** Lowercase snake_case verbs in past or progressive tense, easy to grep. Real events include `bot_starting` (with `admin_id`, `persona`) and `langsmith_enabled` (with `project`) at startup, and `message_processed` (with `user_id`, `reply_len`) per handled message.

**Levels.** The filtering bound logger is set to `INFO` by default, so `debug` is dropped unless `configure_logging` is called with a lower level. Use `info` for normal lifecycle, `warning` for degraded-but-functional, `error` for a thrown exception, `critical` for admin-level events (rate-limit breach, a blocked guardrail, OpenAI down).

**Persistence.** stdout in dev. Under docker-compose the bot's stdout is the container log: `make logs` runs `docker-compose logs -f --tail=100`. For long-running collection, attach a log driver or pipe to a file on the host.

## LangSmith (LLM tracing, optional)

Tracing is opt-in and gated on a key. In `amain()` (`persona_rag/bot/main.py`), the wiring only fires when `LANGCHAIN_API_KEY` is set:

```python
if s.LANGCHAIN_API_KEY:
    os.environ["LANGCHAIN_TRACING_V2"] = str(s.LANGCHAIN_TRACING_V2).lower()
    os.environ["LANGCHAIN_API_KEY"] = s.LANGCHAIN_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = s.LANGCHAIN_PROJECT
    log.info("langsmith_enabled", project=s.LANGCHAIN_PROJECT)
```

With no key set, the block is skipped, no env vars are exported, and the graph runs untraced. There is no breakage either way.

Settings keys (`persona_rag/config.py`, all under the LangSmith section):

| Key | Default | Effect |
|---|---|---|
| `LANGCHAIN_API_KEY` | `None` | The gate. Tracing stays off until this is set. |
| `LANGCHAIN_TRACING_V2` | `True` | Exported as the lowercased string when the key is present. |
| `LANGCHAIN_PROJECT` | `"persona-rag"` | Project name the traces land under. |

Because the LangChain SDK reads `LANGCHAIN_*` from the process environment, setting these vars before the graph runs is enough to auto-trace each LangGraph invocation. A trace shows each node's input/output, the retrieval query and matched documents, the LLM call (prompt, response, token counts, cost), and any errors with stack.

**Privacy.** Every prompt and completion goes to LangSmith when the key is set. To keep tracing off, leave `LANGCHAIN_API_KEY` unset.

## MLflow (eval tracking)

Eval runs are written through `persona_rag/eval/mlflow_wrap.py`. `log_eval_run(...)` calls `_ensure_experiment()` (which points the tracking URI at `MLFLOW_TRACKING_URI`, creates the experiment named by `MLFLOW_EXPERIMENT` if missing, then sets it as active), opens a single `mlflow.start_run(run_name=...)`, logs params (stringified), metrics, optional tags, and any artifact paths that exist on disk, and returns the run id.

Settings keys (`persona_rag/config.py`, MLflow section):

| Key | Default |
|---|---|
| `MLFLOW_TRACKING_URI` | `file:./mlruns` |
| `MLFLOW_EXPERIMENT` | `persona-rag-eval` |

**Two backends:**
- File backend (default for solo dev): `MLFLOW_TRACKING_URI=file:./mlruns`. No server needed.
- Server backend (via docker-compose): point `MLFLOW_TRACKING_URI` at the running server. From the host that is `http://localhost:5001`. The bot and streamlit services inside the compose network reach it at `http://mlflow:5000`.

**Port note (macOS).** `docker-compose.yml` maps the MLflow service as `"5001:5000"`. The server is launched with `--port 5000` and listens on 5000 inside the container. The host port is 5001 because macOS AirPlay Receiver holds 5000. So the UI is `http://localhost:5001` from your machine, while everything inside the compose network still talks to port 5000. `make mlflow-ui` opens `http://localhost:5001`.

**Run anatomy.** See [`EVAL.md`](EVAL.md) for the eval layout. The signal is distributional stylometry computed in `persona_rag/eval/distribution.py` (message shape, per-bubble length, punctuation, code-switch, opener choice, paren-smiley rate). The wrapper opens one flat run per call. Note that `scripts/eval_persona.py` currently writes `scorecard.json` and `pairs.csv` under `data/eval/<name>/` and does not call `log_eval_run`, so MLflow logging is wired but not yet invoked by `make eval`.

**Conventions:**
- One experiment, `persona-rag-eval`.
- Run names dated, for example `2026-05-17-baseline`.
- Tags such as `dataset_version`, `persona_name`, `prompt_version`.
- Params: every dial that affects retrieval or generation (`TOP_K`, `HYBRID_DENSE_ALPHA`, model, `TEMPERATURE`, `RECENCY_HALF_LIFE_DAYS`).
- Artifacts: `pairs.csv` (the incoming/real/generated A/B triple) and `scorecard.json` (distributional distances plus params), the same two files `eval_persona.py` writes under `data/eval/<name>/`.

## docker-compose ports

| Service | Container | Host | Note |
|---|---|---|---|
| qdrant | 6333 (REST), 6334 (gRPC) | 6333, 6334 | |
| mlflow | 5000 | 5001 | host 5001 maps to container 5000; macOS AirPlay holds 5000 |
| streamlit | 8501 | 8501 | `--profile demo` |
| bot | n/a | n/a | `--profile bot`; talks to mlflow at `http://mlflow:5000` |

`make up` starts qdrant and mlflow. `make down` stops the stack.

## What this project deliberately does not run

- **OpenTelemetry / Jaeger / Tempo.** Persona-RAG is a single process. LangSmith covers the per-message trace, structlog covers the rest.
- **Sentry / Bugsnag.** For a solo-run local bot, structlog `error` and `critical` lines surfaced in the container log are enough.
- **Prometheus / Grafana.** No SLO to track here.

## Adding observability for a new feature

- [ ] Get a logger from `persona_rag._logging.get_logger` and bind context (user, chat, message IDs as available).
- [ ] Log entry and exit at info, exceptions at error or critical.
- [ ] If it is a LangGraph node, name it descriptively. That name shows up in LangSmith.
- [ ] If it adds a tunable env var, log the effective value at startup.
- [ ] If it is a new eval signal, log it as a metric through `log_eval_run` so it lands in MLflow.
