from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow

from persona_rag.config import get_settings


def _ensure_experiment() -> None:
    s = get_settings()
    mlflow.set_tracking_uri(s.MLFLOW_TRACKING_URI)
    if mlflow.get_experiment_by_name(s.MLFLOW_EXPERIMENT) is None:
        mlflow.create_experiment(s.MLFLOW_EXPERIMENT)
    mlflow.set_experiment(s.MLFLOW_EXPERIMENT)


def log_eval_run(
    *,
    run_name: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    tags: dict[str, str] | None = None,
    artifacts: list[Path] | None = None,
) -> str:
    _ensure_experiment()
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({k: str(v) for k, v in params.items()})
        mlflow.log_metrics(metrics)
        if tags:
            mlflow.set_tags(tags)
        for path in artifacts or []:
            if path.exists():
                mlflow.log_artifact(str(path))
        run_id: str = run.info.run_id
        return run_id
