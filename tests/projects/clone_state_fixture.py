"""The ``isolated_state`` fixture: a private state dir and config-store reset per test.

Registered as a plugin in ``tests/conftest.py`` (pytest ``pytest_plugins``,
https://docs.pytest.org/en/stable/how-to/writing_plugins.html). Not autouse: the
clone-flow tests ask for it with ``pytest.mark.usefixtures``.
"""

from __future__ import annotations

import pytest

from src.services.config_store.store import reset_config_stores


@pytest.fixture
def isolated_state(tmp_path, monkeypatch):
    import src.config.curation_accessors as curation_mod

    monkeypatch.setenv('OP_STATE_DIR', str(tmp_path / 'state'))
    monkeypatch.setenv('OP_PROJECTS_DATA_ROOT', str(tmp_path / 'projects_data'))
    curation_mod._default_curation_config = None
    reset_config_stores()
    yield
    reset_config_stores()
    curation_mod._default_curation_config = None
