"""Activating through the config store publishes exactly one
``config.changed`` event, stamped with the bound project (never
unbound)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from curation._fake_config_opensearch import FakeConfigOpenSearch


if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def _reset_store() -> Iterator[None]:
    from src.services.config_store.store import reset_config_stores

    reset_config_stores()
    yield
    reset_config_stores()


@pytest.mark.asyncio
async def test_activation_publishes_one_config_changed_event_with_bound_project(
    captured_events: list[dict],
) -> None:
    from src.services.config_store.store import activate_axis, get_config_store
    from src.services.events.publish import GLOBAL_EVENT_TOPIC

    client = FakeConfigOpenSearch()
    store = get_config_store()  # bound to 'default' by the autouse fixture
    await activate_axis(
        store, client, axis='detection_profile', name='wheel', revision=1, expected_active=None
    )

    (event,) = captured_events  # exactly one event
    assert event['type'] == 'config.changed'
    assert event['axis'] == 'detection_profile'
    assert event['name'] == 'wheel'
    assert event['project'] == 'default'
    assert event['topic'] != GLOBAL_EVENT_TOPIC  # project-scoped, not the global stream


@pytest.mark.asyncio
async def test_deactivation_publishes_name_null(captured_events: list[dict]) -> None:
    from src.services.config_store.store import activate_axis, get_config_store

    client = FakeConfigOpenSearch()
    store = get_config_store()
    await activate_axis(
        store, client, axis='prompt_pack', name='wheel_pack', revision=1, expected_active=None
    )
    captured_events.clear()
    await activate_axis(
        store,
        client,
        axis='prompt_pack',
        name=None,
        revision=None,
        expected_active={'name': 'wheel_pack', 'revision': 1},
    )
    (event,) = captured_events
    assert event['name'] is None
    assert event['axis'] == 'prompt_pack'
