# ruff: noqa: RUF001
# Reason: the persona anchor is intentionally written in Ukrainian.
"""Single source of truth for the fine-tune / thin-serving persona anchor.

This exact string is the ``system`` turn the LoRA is trained on
(``finetune/dataset.py``) AND the ``system`` turn served when
``GENERATION_BACKEND == "ollama"`` (``generate/prompt.py``). They MUST be byte
-identical: the adapter conditions its whole learned voice on this anchor, so a
mismatch at serving is the train/serve skew the audit flagged. Keep it short and
in-language — long English instructions drown the style signal.
"""

THIN_SYSTEM = "Ти Богдан. Пиши так, як ти зазвичай пишеш у телеграмі."
