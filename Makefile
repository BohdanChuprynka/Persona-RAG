.PHONY: install lint test ingest run eval clean

install:
	uv sync --all-extras

lint:
	uv run ruff check persona_rag tests scripts
	uv run ruff format --check persona_rag tests scripts

format:
	uv run ruff format persona_rag tests scripts

test:
	uv run pytest -v

ingest:
	uv run python scripts/ingest.py

run:
	uv run python -m persona_rag.bot.main

eval:
	uv run python scripts/eval_persona.py

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
