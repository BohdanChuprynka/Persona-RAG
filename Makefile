.PHONY: install hooks init-data whitelist-admin lint format type test ingest run run-local streamlit eval up down logs mlflow-ui clean compare compare-plot compare-human compare-score compare-turing compare-turing-score compare-arma turing-build turing-score

install:
	uv sync --all-extras

hooks:
	uv run pre-commit install

init-data:
	uv run python scripts/init_data.py

whitelist-admin:
	uv run python scripts/whitelist_admin.py

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

# Same pipeline, served by the local fine-tuned LoRA via Ollama (needs `ollama
# serve` + `ollama create bohdan -f Modelfile`). Preflights the model on startup.
run-local:
	uv run python -m persona_rag.bot.main --local

streamlit:
	uv run streamlit run streamlit_app/main.py

eval:
	uv run python scripts/eval_persona.py

# Fair API-vs-LoRA comparison (start llama-server first; see docs/superpowers/specs).
compare:
	uv run python scripts/compare_persona.py --n 300 --seed 0 --name main

compare-plot:
	uv run python scripts/plot_comparison.py --name main

compare-human:
	uv run python scripts/build_human_eval.py --name main --n 40

compare-score:
	uv run python scripts/score_human_eval.py --name main

# Turing panel: your REAL reply vs the LoRA ("which is the bot?") + tell capture.
compare-turing:
	uv run python scripts/build_human_eval.py --name main --n 40 --mode turing

compare-turing-score:
	uv run python scripts/score_human_eval.py --name main --mode turing

# Arm A: production-realism (shipped API rich+retrieval+levers vs LoRA thin). Needs
# Qdrant up + index built + llama-server. See docs/superpowers/specs/2026-06-02-arm-a-*.
compare-arma:
	uv run python scripts/compare_persona_armA.py --n 300 --seed 0 --name armA

# LoRA-vs-real Turing kit (which reply is the machine?) + its scorer.
turing-build:
	uv run python scripts/build_turing_eval.py --name main --n 100

turing-score:
	uv run python scripts/score_turing_eval.py --name main

up:
	docker-compose up -d qdrant mlflow

down:
	docker-compose down

logs:
	docker-compose logs -f --tail=100

mlflow-ui:
	@echo "MLflow UI: http://localhost:5001"
	@open http://localhost:5001 2>/dev/null || true

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +

.PHONY: insights insights-full insights-dry insights-vault compare-vault

insights:
	uv run python scripts/distill_insights.py --mode incremental

insights-full:
	uv run python scripts/distill_insights.py --mode full

insights-dry:
	uv run python scripts/distill_insights.py --dry-run

# Vault fact ingestion (spec 2026-06-03): full-rebuild from data/raw/vault/.
insights-vault:
	uv run python scripts/ingest_vault.py

# Factual-grounding probe (spec 2026-06-08): bare vs grounded local LoRA, judged for
# hallucination. Routes through the REAL retrieve_insights + build_fact_card path
# (the generation-level A/B the construction-level test deferred), aggregates
# hallucination/correct rates with Wilson CIs + a register-preservation profile.
# Needs llama-server + Qdrant (self_insights) + OPENAI_API_KEY. Inputs/outputs are
# personal -> gitignored under reports/main/grounding/.
compare-vault:
	uv run python scripts/probe_grounding.py
