# Persona-RAG

> A retrieval-augmented Telegram bot that replies like *you*, grounded in your real chat history.
> Paste your own Telegram and Instagram exports, run a few commands, and approved friends can chat with your digital persona.

> **Status:** scaffold + design docs. No implementation yet.

---

## What it does

You give it your past chat conversations. It indexes them. When an approved friend messages your bot, it:

1. Embeds the incoming message
2. Retrieves the top-K times *you* said something in a similar context
3. Builds a prompt with your persona + those few-shot examples + the current convo + per-user memory
4. Asks GPT to reply *as you*
5. Sends the reply over Telegram

Because the model literally sees your past replies, the persona transfer doesn't require fine-tuning — and you can ship it in an evening with ~1k–10k messages.

---

## Why this exists (and why not fine-tuning)

The predecessor project (`PersonaGPT`) tried to fine-tune small LLMs on chat history. It failed because:

- Augmentation (back-translation, synonym swap) destroyed the writing style being learned
- Reasoning-model bases (DeepSeek-R1) leaked their `<think>` template through training
- Loss was computed over the prompt, not just the reply
- Style transfer with LoRA r=8 on 1.5B params is undersized

Retrieval sidesteps all of it. See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full postmortem.

---

## Quickstart (build your own persona)

### 1. Prerequisites

- Python 3.12+, [`uv`](https://docs.astral.sh/uv/)
- A Telegram bot token (from [@BotFather](https://t.me/BotFather))
- Your numeric Telegram user ID (DM [@userinfobot](https://t.me/userinfobot))
- An OpenAI API key

### 2. Clone + install

```bash
git clone https://github.com/<your-user>/Persona-RAG.git
cd Persona-RAG
uv sync
cp .env.example .env
# Open .env and fill in TELEGRAM_BOT_TOKEN, ADMIN_TELEGRAM_ID, OPENAI_API_KEY,
# PERSONA_NAME, PERSONA_LANGUAGE, PERSONA_DESCRIPTION.
```

### 3. Export your conversations

| Source | How |
|---|---|
| **Telegram** | Telegram Desktop → Settings → Advanced → Export Telegram Data → Personal chats only → Machine-readable JSON |
| **Instagram** | instagram.com → Settings → Your activity → Download your information → Messages only → JSON |

Drop the exports into `data/raw/` (folder is gitignored — your data never leaves your machine except for embedding/inference API calls):

```
data/raw/
├── telegram/result.json
└── instagram/your_instagram_activity/messages/
```

### 4. Ingest (one-time)

```bash
uv run python scripts/ingest.py
```

Parses → redacts PII → groups conversations → embeds → writes to `data/vectors.lance`.

### 5. Run the bot

```bash
uv run python -m persona_rag.bot.main
```

Bot polls Telegram for messages. First time anyone other than you DMs the bot, you'll get an authorization request — approve to add them to the whitelist.

The bot runs only while this process is alive. No 24/7 deploy.

---

## Configuration cheat-sheet

All in `.env`:

| Var | Default | Notes |
|---|---|---|
| `PERSONA_NAME` | — | Your name, e.g. `Bohdan` |
| `PERSONA_LANGUAGE` | `en` | ISO 639-1 — your primary chat language |
| `PERSONA_DESCRIPTION` | — | 1–3 sentence persona briefing for the LLM |
| `OPENAI_CHAT_MODEL` | `gpt-4o-mini` | Bump to `gpt-4o` for higher quality |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | |
| `TOP_K` | `8` | Retrieved examples per reply |
| `TEMPERATURE` | `0.8` | LLM sampling temperature |
| `SESSION_TIMEOUT_MINUTES` | `30` | Silence → end of session → write memory summary |

---

## Privacy

- All chat data lives **only in `data/`** which is gitignored. Nothing about your conversations enters the repo.
- Incoming/outgoing bot messages and retrieved snippets **are sent to OpenAI** for embedding and chat completion. If you need full local privacy, see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the Ollama swap path.
- PII redaction (addresses, phone numbers, friend names) runs on ingest. Tune the rules in [`docs/DATA-PIPELINE.md`](docs/DATA-PIPELINE.md).

---

## Architecture docs

| Doc | What |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | System overview, why RAG over SFT, top-level diagrams |
| [`docs/DATA-PIPELINE.md`](docs/DATA-PIPELINE.md) | Parser specs, PII redaction, conversation turn building, schema |
| [`docs/AUTH-FLOW.md`](docs/AUTH-FLOW.md) | Owner-admin gated whitelist, FSM, admin commands |
| [`docs/PROMPT-DESIGN.md`](docs/PROMPT-DESIGN.md) | System prompt template, few-shot assembly, per-user memory |
| [`docs/EVAL.md`](docs/EVAL.md) | Stylometry, A/B protocol, persona-match metrics |

Diagrams: [`docs/diagrams/`](docs/diagrams/) (mermaid `.mmd`).

---

## Stack

Python 3.12 · `uv` · `aiogram 3` · OpenAI API · LanceDB · SQLite/SQLModel · pydantic-settings · pytest

---

## License

MIT. See [`LICENSE`](LICENSE).

---

## Note on CI

The CI workflow template lives at [`docs/ci-template.yml`](docs/ci-template.yml). To activate it, the repo owner needs to grant the `workflow` scope to their GitHub token (`gh auth refresh -s workflow`), then move the file to `.github/workflows/ci.yml`.
