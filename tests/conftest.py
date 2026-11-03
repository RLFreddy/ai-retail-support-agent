"""Shared fixtures that keep the test suite hermetic.

Every test runs without the developer's ``.env``, without provider API keys
and (via pytest-socket, see ``pyproject.toml``) without network access, so the
suite gives the same result on any machine and in CI.
"""

import os

import pytest

from tiendahogar import config

#: Settings fields that can leak in from the environment or ``.env``.
_SETTINGS_ENV = {name.upper() for name in config.Settings.model_fields}


@pytest.fixture(autouse=True)
def _hermetic_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # LiteLLM calls load_dotenv() at import time, so the real .env may already
    # be in os.environ: strip every Settings variable and any *_API_KEY.
    for name in list(os.environ):
        if name in _SETTINGS_ENV or name.endswith("_API_KEY"):
            monkeypatch.delenv(name, raising=False)
    # Never read the developer's .env file.
    monkeypatch.setitem(config.Settings.model_config, "env_file", None)
    config.get_settings.cache_clear()
