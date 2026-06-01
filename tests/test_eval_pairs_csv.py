# ruff: noqa: RUF001
"""pairs.csv must round-trip replies that contain quotes / commas / newlines
(code-review #4). Loads write_pairs_csv from the eval script by path."""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

_PATH = Path(__file__).resolve().parent.parent / "scripts" / "eval_persona.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("eval_persona_script", _PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_pairs_csv_roundtrips_quotes_commas_newlines(tmp_path):
    mod = _load_script()
    p = tmp_path / "pairs.csv"
    rows = [
        ("hi", 'він сказав "ок", і пішов', "ну ок"),
        ("multi", "рядок1\nрядок2", "ok)"),
    ]
    mod.write_pairs_csv(p, rows)

    with p.open(newline="", encoding="utf-8") as f:
        got = list(csv.reader(f))

    assert got[0] == ["incoming", "real", "generated"]
    assert all(len(r) == 3 for r in got)  # never splits a quoted/comma field
    assert got[1] == ["hi", 'він сказав "ок", і пішов', "ну ок"]
    assert got[2] == ["multi", "рядок1\nрядок2", "ok)"]
