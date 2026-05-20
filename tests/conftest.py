import os

import pytest


@pytest.fixture(autouse=True, scope="session")
def _test_env() -> None:
    os.environ.setdefault("PERSONA_NAME", "TestPersona")
    os.environ.setdefault("TELEGRAM_BOT_TOKEN", "test-token")
    os.environ.setdefault("ADMIN_TELEGRAM_ID", "42")
    os.environ.setdefault("OPENAI_API_KEY", "sk-test")
