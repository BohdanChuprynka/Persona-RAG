# Persona-RAG

> A LangGraph-orchestrated retrieval-augmented Telegram bot that replies like *you*, grounded in your real chat history.
> Index your Telegram and Instagram exports, run docker-compose, and approved friends can chat with your digital persona.

> **Status:** scaffold + design docs only. No implementation yet.

---

## What it does

You give it your past chat conversations. A LangGraph state machine handles every incoming message:

1. **Auth check** — is this user on your whitelist? If not, route to admin approval.
2. **Retrieve** — embed the message; pull top-K similar past replies from Qdrant via hybrid (dense + BM25) search; rerank with recency decay.
3. **Build prompt** — assemble cacheable persona system prompt + few-shot pairs + per-user memory summary + current session turns. OpenAI's automatic prompt caching cuts ~50% of input cost on repeat prefixes.
4. **Generate** — gpt-4o-mini drafts a reply.
5. **Guard** — strip PII leaks, length-cap, refuse-list check.
6. **Send + log** — Telegram reply out; every node traced in LangSmith; session turn appended.
7. **Memory update** — on session timeout, an async job distills the session into the user's persistent memory summary.

Because the model literally sees your past replies as few-shot examples, the persona transfer doesn't require fine-tuning. You can ship it in an evening with ~1k–10k messages.

A **shadow mode** captures `(incoming, your_real_reply, bot_reply)` triples without sending — feeds future DPO fine-tuning and persona-match A/B evaluation.

---

## Why this exists (and why not fine-tuning)

The predecessor project (`PersonaGPT`) tried to fine-tune small LLMs on chat history. Six ranked failure modes (augmentation destroyed style, R1-distill template leaked, loss mis-targeted over prompt, forced Q/A alternation, undersized LoRA, BLEU/ROUGE for persona) are documented in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md#why-rag-not-sft).

Retrieval sidesteps all of them. The trade-off: dependence on OpenAI as base model. Mitigation path documented in `docs/ARCHITECTURE.md#future-extensions`.

---

## Quickstart (build your own persona)

### 1. Prerequisites

- Python 3.12+, [`uv`](https://docs.astral.sh/uv/), Docker + docker-compose
- Telegram bot token (from [@BotFather](https://t.me/BotFather))
- Your numeric Telegram user ID (DM [@userinfobot](https://t.me/userinfobot))
- OpenAI API key
- (Optional) LangSmith API key for tracing — free tier at https://smith.langchain.com

### 2. Clone + configure

```bash
git clone https://github.com/<your-user>/Persona-RAG.git
cd Persona-RAG
uv sync
cp .env.example .env
# Fill in: TELEGRAM_BOT_TOKEN, ADMIN_TELEGRAM_ID, OPENAI_API_KEY,
# PERSONA_NAME, PERSONA_LANGUAGE, PERSONA_DESCRIPTION, LANGCHAIN_API_KEY.
```

### 3. Export your conversations

| Source | How |
|---|---|
| **Telegram** | Telegram Desktop → Settings → Advanced → Export Telegram Data → Personal chats only → Machine-readable JSON |
| **Instagram** | instagram.com → Settings → Your activity → Download your information → Messages only → JSON |

Drop the exports into `data/raw/` (gitignored):

```
data/raw/
├── telegram/result.json
└── instagram/your_instagram_activity/messages/inbox/
```

### 4. Start backing services

```bash
docker-compose up -d qdrant mlflow
```

Qdrant on `localhost:6333`. MLflow UI on `localhost:5000`.

### 5. Ingest (one-time)

```bash
uv run python scripts/ingest.py
```

Parses → PII redacts → groups conversations → embeds → writes to Qdrant.

### 6. Run the bot

```bash
uv run python -m persona_rag.bot.main
```

First non-admin DM triggers admin approval flow in your Telegram.

### 7. Try without Telegram (Streamlit demo)

```bash
uv run streamlit run streamlit_app/main.py
```

A web UI where you can paste messages and see persona replies. Useful for sharing the project without giving someone your bot.

---

## Configuration cheat-sheet

All in `.env`. See `.env.example` for the full list with comments.

| Group | Key vars |
|---|---|
| **Persona** | `PERSONA_NAME`, `PERSONA_LANGUAGE`, `PERSONA_DESCRIPTION` |
| **OpenAI** | `OPENAI_API_KEY`, `OPENAI_CHAT_MODEL`, `OPENAI_EMBEDDING_MODEL`, `ENABLE_PROMPT_CACHING` |
| **Qdrant** | `QDRANT_URL`, `QDRANT_API_KEY`, `QDRANT_COLLECTION` |
| **LangSmith** | `LANGCHAIN_TRACING_V2`, `LANGCHAIN_API_KEY`, `LANGCHAIN_PROJECT` |
| **MLflow** | `MLFLOW_TRACKING_URI`, `MLFLOW_EXPERIMENT` |
| **Retrieval** | `TOP_K`, `HYBRID_DENSE_ALPHA`, `RECENCY_HALF_LIFE_DAYS` |
| **Generation** | `TEMPERATURE`, `MAX_REPLY_TOKENS` |
| **Memory** | `CURRENT_SESSION_WINDOW`, `SESSION_TIMEOUT_MINUTES` |
| **Shadow** | `SHADOW_MODE`, `SHADOW_LOG_PATH` |

---

## Privacy

- Chat data lives **only in `data/`** (gitignored). Nothing personal enters the repo.
- Incoming/outgoing bot messages and retrieved snippets are sent to **OpenAI** for embedding and chat completion. Full-local alternative documented in `docs/ARCHITECTURE.md#future-extensions` (Ollama swap).
- PII redaction (addresses, phone numbers, friend names) runs at ingest time. Rules in `docs/DATA-PIPELINE.md`.
- LangSmith stores traces of every chain. If that's a privacy issue, set `LANGCHAIN_TRACING_V2=false`.

---

## Architecture docs

| Doc | What |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | System overview, why RAG over SFT, three subsystems, LangGraph design |
| [`docs/DATA-PIPELINE.md`](docs/DATA-PIPELINE.md) | Parser specs, PII redaction, conversation building, Qdrant schema |
| [`docs/AUTH-FLOW.md`](docs/AUTH-FLOW.md) | Owner-admin gated whitelist, LangGraph FSM, admin commands |
| [`docs/PROMPT-DESIGN.md`](docs/PROMPT-DESIGN.md) | System prompt, few-shot assembly, prompt-caching layout, per-user memory |
| [`docs/EVAL.md`](docs/EVAL.md) | MLflow-tracked stylometry + perplexity + shadow-mode A/B + persona-match |
| [`docs/OBSERVABILITY.md`](docs/OBSERVABILITY.md) | structlog conventions, LangSmith setup, MLflow runs |

Diagrams (mermaid): [`docs/diagrams/`](docs/diagrams/).

---

## Stack

Python 3.12 · `uv` · **aiogram 3** (bot) · **LangGraph** (orchestration) · **LangSmith** (tracing) · **OpenAI** (LLM + embeddings, with prompt caching) · **Qdrant** (vector DB, hybrid retrieval) · **MLflow** (eval tracking) · **Streamlit** (demo UI) · SQLite + SQLModel (user state) · pydantic-settings · structlog · pytest + pytest-asyncio · ruff + mypy strict + pre-commit

---

## License

MIT. See [`LICENSE`](LICENSE).

---

## Note on CI

The CI workflow template lives at [`docs/ci-template.yml`](docs/ci-template.yml). To activate it: `gh auth refresh -s workflow`, then move the file to `.github/workflows/ci.yml`.
