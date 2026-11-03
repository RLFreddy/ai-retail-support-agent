"""Shared fixtures that keep the test suite hermetic.

Every test runs without the developer's ``.env``, without provider API keys
and (via pytest-socket, see ``pyproject.toml``) without network access, so the
suite gives the same result on any machine and in CI.
"""

import os

import pytest

from tiendahogar import cli, config

#: Settings fields that can leak in from the environment or ``.env``.
_SETTINGS_ENV = {name.upper() for name in config.Settings.model_fields}


@pytest.fixture(autouse=True)
def _hermetic_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # Strip every Settings variable, any provider key or endpoint and the OTel exporter.
    for name in list(os.environ):
        if (
            name in _SETTINGS_ENV
            or name.startswith("OTEL_")
            or name.endswith(("_API_KEY", "_BASE_URL"))
        ):
            monkeypatch.delenv(name, raising=False)
    # Never read the developer's .env file, neither through Settings nor
    # through the CLI's load_dotenv() (which feeds the provider keys).
    monkeypatch.setitem(config.Settings.model_config, "env_file", None)
    monkeypatch.setattr(cli, "load_dotenv", lambda *args, **kwargs: False)
    config.get_settings.cache_clear()
