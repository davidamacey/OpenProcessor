"""A bulk-ingest run sends its batches to the product's API address, the front (``api:8000`` on
the compose network), never to a loopback address: the Python API behind the front listens on
4613, so ``127.0.0.1:8000`` in its container is nothing."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.services.curation.bulk_ingest import limits


if TYPE_CHECKING:
    import pytest


def test_the_default_is_the_front_on_the_compose_network(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('OP_API_LOOPBACK_URL', raising=False)
    assert limits.loopback_url() == 'http://api:8000'


def test_the_setting_overrides_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('OP_API_LOOPBACK_URL', 'http://other:9000')
    assert limits.loopback_url() == 'http://other:9000'
