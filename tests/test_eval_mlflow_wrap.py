from persona_rag.config import get_settings


def test_log_eval_run_creates_run(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file:{tmp_path}/mlruns")
    monkeypatch.setenv("MLFLOW_EXPERIMENT", "test-exp")
    get_settings.cache_clear()
    # Reload wrapper module so it picks up the new settings
    import importlib

    import persona_rag.eval.mlflow_wrap as wrap

    importlib.reload(wrap)

    wrap.log_eval_run(
        run_name="test-run",
        params={"top_k": 8, "model": "gpt-4o-mini"},
        metrics={"stylometry_composite": 1.23},
        tags={"persona_name": "Tester"},
    )

    import mlflow

    mlflow.set_tracking_uri(f"file:{tmp_path}/mlruns")
    exp = mlflow.get_experiment_by_name("test-exp")
    assert exp is not None
    runs = mlflow.search_runs([exp.experiment_id])
    assert len(runs) == 1
    assert runs.iloc[0]["params.model"] == "gpt-4o-mini"

    get_settings.cache_clear()
