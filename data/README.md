# `data/` — runtime artifacts

This directory holds everything the bot reads or writes at runtime. **All contents except this README and the `.gitkeep` markers are gitignored.** Personal chats never enter git.

## Layout

```
data/
├── README.md                      # this file (tracked)
├── raw/                           # YOU put exports here before running ingest
│   ├── telegram/
│   │   └── result.json            # Telegram Desktop export
│   └── instagram/
│       └── your_instagram_activity/
│           └── messages/inbox/    # Instagram messages JSON tree
├── processed/                     # reserved for future export-clean snapshots
├── eval/                          # eval-run artifacts (per-turn CSV pairs)
│   └── YYYY-MM-DD-HHMM-baseline-pairs.csv
│
├── persona.db                     # SQLite — conversations, persona turns, users, memory, audit
├── style_anchors.json             # cached stylometric features for prompt prefix
├── bm25.pkl                       # in-memory BM25 index, pickled
└── shadow_log.jsonl               # SHADOW_MODE=true writes (incoming, generated, real) triples
```

## How files appear

| File | Created by |
|---|---|
| `data/raw/**` | **You** drop chat exports here before running `make ingest` |
| `data/persona.db` | `make ingest` → SQLite via `persona_rag.db.engine.make_engine` |
| `data/style_anchors.json` | `make ingest` → `persona_rag.ingest.stylometry.compute_anchors` |
| `data/bm25.pkl` | `make ingest` → `persona_rag.index.bm25_store.save` |
| `data/shadow_log.jsonl` | bot runtime when `SHADOW_MODE=true` |
| `data/eval/*.csv` | `make eval` → `scripts/eval_persona.py` |

## What's outside `data/`

- **Qdrant vectors** live inside the `qdrant_storage` Docker volume (managed by docker-compose, not on this filesystem).
- **MLflow runs** live in `./mlruns/` (or wherever `MLFLOW_TRACKING_URI` points). The MLflow Docker container writes to its own `mlflow_data` volume.

## Reset

To wipe everything except `data/raw/`:

```bash
rm -f data/persona.db data/style_anchors.json data/bm25.pkl data/shadow_log.jsonl
rm -rf data/eval/
docker volume rm persona-rag_qdrant_storage   # also nuke Qdrant
```

To wipe absolutely everything (including your exports):

```bash
git clean -fdx data/
```

## Bootstrap from clean clone

After `git clone`:

```bash
make install
make init-data          # ensures dir structure (idempotent)
cp .env.example .env    # then fill in tokens
```

Then drop your exports into `data/raw/telegram/` and `data/raw/instagram/`.
