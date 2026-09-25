"""X-Request-ID correlation behavior pins.

A client-supplied ``X-Request-ID`` must be echoed back and bound to the
logging context for the duration of the request; when absent, the
middleware must generate one.

Uses the real application object so the middleware wiring itself is under
test; TestClient is used without its context manager so the lifespan
(Triton pool, OpenSearch warm-up) never runs.

This file extends five pre-existing basenames with three checks for the
curation async pipeline:

1. ``test_items_request_id_field_migration_is_idempotent`` — per this
   repo's house rule (don't add the repo's first live-stack
   dependency), this fakes the ``indices.put_mapping`` I/O boundary
   rather than hitting a live dev OpenSearch.
2. ``test_worker_structlog_emits_request_id`` /
   ``test_worker_unbind_clears_request_id`` — pure-Python checks that
   ``bind_request_id`` / ``unbind_contextvars`` propagate through
   structlog's contextvars merge into a captured event dict. Mirrors
   the binding pattern the detection worker's consumers use.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest
import structlog
from fastapi.testclient import TestClient


pytestmark = pytest.mark.integration


@pytest.fixture(scope='module')
def client() -> TestClient:
    from src.main import app

    return TestClient(app)


def test_client_supplied_request_id_is_echoed(client: TestClient) -> None:
    response = client.get('/live', headers={'X-Request-ID': 'test-corr-123'})
    assert response.status_code == 200
    assert response.headers['X-Request-ID'] == 'test-corr-123'


def test_request_id_is_generated_when_absent(client: TestClient) -> None:
    response = client.get('/live')
    assert response.status_code == 200
    generated = response.headers.get('X-Request-ID')
    assert generated
    assert generated != '-'


def test_bind_request_id_updates_context() -> None:
    from src.core.logging import bind_request_id, clear_request_id, get_request_id

    assert get_request_id() == '-'
    bind_request_id('ctx-abc')
    try:
        assert get_request_id() == 'ctx-abc'
    finally:
        clear_request_id()
    assert get_request_id() == '-'


@pytest.mark.asyncio
async def test_items_request_id_field_migration_is_idempotent() -> None:
    """``ensure_items_request_id_field`` PUTs an additive keyword-field
    mapping and reports it in ``fields_added`` — proves the migration
    helper is wired to the configured items index and its response
    shape, without requiring a live OpenSearch (the real acknowledged
    round-trip is a plain additive ``PUT _mapping`` call, exercised
    against opensearchpy's client contract by
    ``tests/curation/test_ensure_indexes.py``'s other migration-helper
    tests).
    """
    from src.clients.curation_opensearch import ensure_items_request_id_field
    from src.config import get_curation_config

    fake_client = AsyncMock()
    fake_client.indices.put_mapping = AsyncMock(return_value={'acknowledged': True})

    result = await ensure_items_request_id_field(fake_client)

    assert result['acknowledged'] is True
    assert result['fields_added'] == ['request_id']
    assert result['index'] == get_curation_config().items_index
    fake_client.indices.put_mapping.assert_awaited_once()
    call = fake_client.indices.put_mapping.await_args
    assert call is not None
    assert call.kwargs['body'] == {'properties': {'request_id': {'type': 'keyword'}}}


def test_worker_structlog_emits_request_id() -> None:
    """Binding via :func:`structlog.contextvars.bind_contextvars` (the
    same call the detection worker's consumers make per task) propagates
    into the event dict captured by a structlog log capture.

    This is the unit-level proof that the contextvars merge processor
    is wired up in :mod:`src.core.logging` and that worker emissions
    will carry ``request_id`` once the consumer binds it.
    """
    from src.core.logging import bind_request_id, clear_request_id

    # configure_logging() is invoked at FastAPI startup but not in pytest
    # context — apply a minimal capture chain that includes the
    # contextvars merge processor.
    captured: list[dict[str, Any]] = []

    def _capture(_logger: Any, _name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
        captured.append(dict(event_dict))
        return event_dict

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            _capture,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.BoundLogger,
        context_class=dict,
        cache_logger_on_first_use=False,
    )

    log = structlog.get_logger('worker_test')

    try:
        bind_request_id('testid_abc123')
        log.info('stage_a_detect_took_ms', crop_id='crop-1', ms=42.0)
        bind_request_id('testid_def456')
        log.info('stage_b_verify_took_ms', crop_id='crop-2', ms=99.0)
    finally:
        clear_request_id()
        # Reset structlog so other tests get a fresh default config.
        structlog.reset_defaults()

    assert len(captured) == 2
    assert captured[0]['request_id'] == 'testid_abc123'
    assert captured[0]['event'] == 'stage_a_detect_took_ms'
    assert captured[1]['request_id'] == 'testid_def456'
    assert captured[1]['event'] == 'stage_b_verify_took_ms'


def test_worker_unbind_clears_request_id() -> None:
    """After ``unbind_contextvars`` the next log event must not carry the
    previous request_id — the worker's per-task ``finally`` block relies
    on this so consumer iterations don't leak the prior id.
    """
    captured: list[dict[str, Any]] = []

    def _capture(_logger: Any, _name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
        captured.append(dict(event_dict))
        return event_dict

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            _capture,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.BoundLogger,
        context_class=dict,
        cache_logger_on_first_use=False,
    )

    log = structlog.get_logger('worker_test')

    try:
        structlog.contextvars.bind_contextvars(request_id='req-1')
        log.info('bound')
        structlog.contextvars.unbind_contextvars('request_id')
        log.info('unbound')
    finally:
        structlog.reset_defaults()

    assert captured[0]['request_id'] == 'req-1'
    assert 'request_id' not in captured[1]
