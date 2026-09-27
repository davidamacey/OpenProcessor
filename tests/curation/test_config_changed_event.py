"""W2: activating through the config store publishes exactly one
``config.changed`` event, stamped with the bound project (never
unbound) -- any_domain_plan.md §9 W2."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from curation._fake_config_opensearch import FakeConfigOpenSearch


if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def _process_event_bus(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """In-process bus only -- no shared JSONL log needed for this test."""
    monkeypatch.setenv('OP_EVENT_BUS', 'process')
    from src.services.curation import event_hub

    event_hub._HUB = None
    yield
    event_hub._HUB = None


@pytest.fixture(autouse=True)
def _reset_store() -> Iterator[None]:
    from src.services.config_store.store import reset_config_stores

    reset_config_stores()
    yield
    reset_config_stores()


@pytest.mark.asyncio
async def test_activation_publishes_one_config_changed_event_with_bound_project() -> None:
    from src.services.config_store.store import activate_axis, get_config_store
    from src.services.curation.event_hub import GLOBAL_EVENT_TOPIC, get_event_hub

    client = FakeConfigOpenSearch()
    store = get_config_store()  # bound to 'default' by the autouse fixture
    hub = get_event_hub()
    sub = await hub.subscribe(project='default')

    await activate_axis(
        store, client, axis='detection_profile', name='wheel', revision=1, expected_active=None
    )

    event = sub.queue.get_nowait()
    assert event['type'] == 'config.changed'
    assert event['axis'] == 'detection_profile'
    assert event['name'] == 'wheel'
    assert event['project'] == 'default'
    assert event['topic'] != GLOBAL_EVENT_TOPIC  # project-scoped, not the global stream
    assert sub.queue.empty()  # exactly one event


@pytest.mark.asyncio
async def test_deactivation_publishes_name_null() -> None:
    from src.services.config_store.store import activate_axis, get_config_store
    from src.services.curation.event_hub import get_event_hub

    client = FakeConfigOpenSearch()
    store = get_config_store()
    hub = get_event_hub()
    sub = await hub.subscribe(project='default')

    await activate_axis(
        store, client, axis='prompt_pack', name='wheel_pack', revision=1, expected_active=None
    )
    sub.queue.get_nowait()
    await activate_axis(
        store,
        client,
        axis='prompt_pack',
        name=None,
        revision=None,
        expected_active={'name': 'wheel_pack', 'revision': 1},
    )
    event = sub.queue.get_nowait()
    assert event['name'] is None
    assert event['axis'] == 'prompt_pack'
