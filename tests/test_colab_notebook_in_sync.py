"""The committed Colab notebook is a GENERATED artifact. It must always equal
``scripts/build_colab_notebook.py`` output, or a user opens a stale kit (this
exact drift shipped the pre-audit notebook once). This guard fails loudly if the
two diverge — run ``python scripts/build_colab_notebook.py`` and commit.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_generator():
    spec = importlib.util.spec_from_file_location(
        "build_colab_notebook", ROOT / "scripts" / "build_colab_notebook.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_committed_notebook_matches_generator() -> None:
    mod = _load_generator()
    on_disk = json.loads((ROOT / "notebooks" / "finetune_persona_colab.ipynb").read_text())
    assert on_disk == mod.build_notebook(), (
        "notebooks/finetune_persona_colab.ipynb is stale — run "
        "`python scripts/build_colab_notebook.py` and commit the regenerated notebook."
    )
