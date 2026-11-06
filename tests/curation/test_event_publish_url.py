"""Where the detection worker publishes its events: ``OP_EVENT_API_URL``, else the API base URL."""

from __future__ import annotations

import importlib

import pytest


@pytest.fixture
def _reload_bulk_writer_after_env_restored():
    """Reload ``bulk_writer`` once more after ``monkeypatch`` reverts the
    env it patched, so the module-level ``_EVENT_API_URL`` constant
    doesn't leak a test value into later tests importing this module.

    Fixture teardown order is LIFO: this fixture must be requested
    *before* ``monkeypatch`` in the test signature so its teardown runs
    *after* monkeypatch's (env already restored by then).
    """
    yield
    from scripts.curation.worker import bulk_writer

    importlib.reload(bulk_writer)


@pytest.mark.usefixtures('_reload_bulk_writer_after_env_restored')
def test_bulk_writer_event_api_url_falls_back_to_api_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OP_EVENT_API_URL is the explicit override; with only OP_API_BASE_URL
    set (the trainer/campaign convention for "where api lives"), the
    detection worker's publish path should still resolve a URL instead of
    silently going dark."""
    monkeypatch.delenv('OP_EVENT_API_URL', raising=False)
    monkeypatch.setenv('OP_API_BASE_URL', 'http://api:8000/')
    monkeypatch.delenv('OP_API', raising=False)

    from scripts.curation.worker import bulk_writer

    importlib.reload(bulk_writer)
    assert bulk_writer._EVENT_API_URL == 'http://api:8000'


@pytest.mark.usefixtures('_reload_bulk_writer_after_env_restored')
def test_bulk_writer_event_api_url_prefers_explicit_over_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv('OP_EVENT_API_URL', 'http://explicit:8000')
    monkeypatch.setenv('OP_API_BASE_URL', 'http://fallback:8000')

    from scripts.curation.worker import bulk_writer

    importlib.reload(bulk_writer)
    assert bulk_writer._EVENT_API_URL == 'http://explicit:8000'
