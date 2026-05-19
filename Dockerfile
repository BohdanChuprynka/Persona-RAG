FROM python:3.12-slim

# uv binary
COPY --from=ghcr.io/astral-sh/uv:0.4 /uv /usr/local/bin/uv

WORKDIR /app

# Cache deps layer
COPY pyproject.toml uv.lock* ./
RUN uv sync --frozen --no-dev || uv sync --no-dev

# App code
COPY persona_rag ./persona_rag
COPY scripts ./scripts
COPY streamlit_app ./streamlit_app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Bot is the default entrypoint; override via compose for streamlit/ingest.
CMD ["uv", "run", "python", "-m", "persona_rag.bot.main"]
