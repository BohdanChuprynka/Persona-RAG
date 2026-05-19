.PHONY: install lint format type test ingest run streamlit eval up down logs mlflow-ui clean

install:
	uv sync --all-extras

lint:
	uv run ruff check persona_rag tests scripts streamlit_app
	uv run ruff format --check persona_rag tests scripts streamlit_app

format:
	uv run ruff format persona_rag tests scripts streamlit_app

type:
	uv run mypy persona_rag

test:
	uv run pytest -v

ingest:
	uv run python scripts/ingest.py

run:
	uv run python -m persona_rag.bot.main

streamlit:
	uv run streamlit run streamlit_app/main.py

eval:
	uv run python scripts/eval_persona.py

up:
	docker-compose up -d qdrant mlflow

down:
	docker-compose down

logs:
	docker-compose logs -f --tail=100

mlflow-ui:
	@echo "MLflow UI: http://localhost:5000"
	@open http://localhost:5000 2>/dev/null || true

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
