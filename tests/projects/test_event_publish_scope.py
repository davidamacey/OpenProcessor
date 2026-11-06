"""Which stream an event lands on: the bound project's, or the global one for the global families.

The per-stream delivery (a subscriber sees only its own project's rows) is in tests/events/.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from src.config.curation_accessors import base_curation_config
from src.config.project_context import ProjectNotBound, bind_project
from src.config.projects import ProjectRecord, resources_for_new
from src.services.events import publish as publish_module
from src.services.events.publish import GLOBAL, publish_crop_created, publish_global_event


# Binding is part of what is under test here.
pytestmark = pytest.mark.unbound


def _record(slug: str) -> ProjectRecord:
    now = datetime.now(UTC).isoformat()
    return ProjectRecord(
        slug=slug,
        display_name=slug,
        description='',
        status='active',
        revision=1,
        created_at=now,
        updated_at=now,
        origin=None,
        resources=resources_for_new(slug, base_curation_config()),
    )


def test_a_project_event_is_stamped_with_the_bound_project(
    captured_events: list[dict[str, Any]],
) -> None:
    with bind_project(_record('alpha')):
        publish_crop_created('alpha-item-0001')
    (event,) = captured_events
    assert (event['type'], event['project'], event['crop_id']) == (
        'crop.created',
        'alpha',
        'alpha-item-0001',
    )


def test_an_unbound_publish_is_refused(captured_events: list[dict[str, Any]]) -> None:
    """No implicit global fallback: a project event published from a context that lost its
    binding raises instead of broadcasting."""
    with pytest.raises(ProjectNotBound):
        publish_module.publish('crop.created', {'crop_id': 'x'})
    assert captured_events == []


def test_only_global_families_may_go_global(captured_events: list[dict[str, Any]]) -> None:
    with pytest.raises(ValueError, match='not a global event type'):
        publish_global_event('crop.created', crop_id='x')
    assert captured_events == []


def test_combine_progress_goes_global_with_target(captured_events: list[dict[str, Any]]) -> None:
    """The combine target is ``building`` (binds refused), so its progress rides the global
    stream and names the target, whatever project the publisher has bound."""
    with bind_project(_record('alpha')):
        publish_global_event('combine.progress', target='cars-all', job_id='cmb_1', done=3, total=9)
    (event,) = captured_events
    assert event['type'] == 'combine.progress'
    assert event['project'] is None
    assert event['target'] == 'cars-all'
    assert event['topic'] == 'project'
    assert (event['job_id'], event['done'], event['total']) == ('cmb_1', 3, 9)


def test_global_event_target_is_always_on_the_wire(captured_events: list[dict[str, Any]]) -> None:
    publish_global_event('project.deleted')
    (event,) = captured_events
    assert event['target'] is None
    assert event['project'] is None


def test_the_vlm_family_carries_its_own_topic(captured_events: list[dict[str, Any]]) -> None:
    publish_global_event('vlm.changed', axis='endpoints')
    assert captured_events[0]['topic'] == 'vlm'


@pytest.mark.parametrize('key', ['type', 'topic', 'project'])
def test_global_event_fields_cannot_override_the_envelope(
    captured_events: list[dict[str, Any]], key: str
) -> None:
    """``type=`` (or any envelope key) in ``fields`` must not smuggle a project event onto the
    global stream."""
    with pytest.raises(ValueError, match='envelope'):
        publish_global_event('project.updated', **{key: 'crop.classified'}, crop_id='beta-1')
    assert captured_events == []


def test_an_explicit_global_target_ignores_the_bound_project(
    captured_events: list[dict[str, Any]],
) -> None:
    with bind_project(_record('alpha')):
        publish_module.publish('project.created', {}, project=GLOBAL)
    assert captured_events[0]['project'] is None
