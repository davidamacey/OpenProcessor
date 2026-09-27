"""Event fan-out is project-scoped and fails closed (projects_plan.md
§2.5, review deltas 1 and 4, P1 review B2): a project's events reach only
that project's subscribers; ``project: null`` events reach only the
global stream; an unbound publish is refused; an event that names another
project is refused; an unstamped event reaches no one. ``combine.*``
progress rides the global stream, addressed by ``target``."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from src.config.curation import base_curation_config
from src.config.project_context import bind_project
from src.config.projects import ProjectRecord, resources_for_new
from src.services.curation import event_hub as event_hub_mod
from src.services.curation.event_hub import GLOBAL_STREAM, EventHub, publish_global_event


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


@pytest.fixture
def hub(monkeypatch: pytest.MonkeyPatch) -> EventHub:
    monkeypatch.setenv('OP_EVENT_BUS', 'process')
    fresh = EventHub()
    monkeypatch.setattr(event_hub_mod, '_HUB', fresh)
    return fresh


def _drain(sub: Any) -> list[dict[str, Any]]:
    out = []
    while not sub.queue.empty():
        out.append(sub.queue.get_nowait())
    return out


@pytest.mark.asyncio
async def test_project_event_reaches_only_its_own_project(hub: EventHub) -> None:
    alpha_sub = await hub.subscribe(project='alpha')
    beta_sub = await hub.subscribe(project='beta')
    global_sub = await hub.subscribe(project=GLOBAL_STREAM)

    with bind_project(_record('alpha')):
        hub.publish({'type': 'crop.created', 'topic': 'crop', 'crop_id': 'alpha-item-0001'})

    assert [e['crop_id'] for e in _drain(alpha_sub)] == ['alpha-item-0001']
    assert _drain(beta_sub) == []
    assert _drain(global_sub) == []


@pytest.mark.asyncio
async def test_publish_stamps_the_bound_project(hub: EventHub) -> None:
    sub = await hub.subscribe(project='alpha')
    with bind_project(_record('alpha')):
        hub.publish({'type': 'crop.created', 'topic': 'crop', 'crop_id': 'x'})
    (event,) = _drain(sub)
    assert event['project'] == 'alpha'


def test_unbound_publish_is_refused(hub: EventHub) -> None:
    """No implicit global fallback: a project event published from a
    context that lost its binding raises instead of broadcasting."""
    from src.config.project_context import ProjectNotBound

    with pytest.raises(ProjectNotBound):
        hub.publish({'type': 'crop.created', 'topic': 'crop', 'crop_id': 'x'})
    assert hub.stats(None)['events_published'] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize('injected', ['alpha', None])
async def test_publish_cannot_name_another_project(hub: EventHub, injected: str | None) -> None:
    alpha_sub = await hub.subscribe(project='alpha')
    global_sub = await hub.subscribe(project=GLOBAL_STREAM)
    with bind_project(_record('beta')), pytest.raises(event_hub_mod.EventProjectMismatchError):
        hub.publish({'type': 'crop.created', 'crop_id': 'beta-item-0001', 'project': injected})
    assert _drain(alpha_sub) == []
    assert _drain(global_sub) == []


@pytest.mark.asyncio
async def test_unstamped_event_reaches_no_one(hub: EventHub) -> None:
    """A log line with no ``project`` key (a stale or foreign writer) is
    dispatched by the tail loop without stamping; it must go nowhere."""
    subs = [await hub.subscribe(project=p) for p in ('alpha', 'beta', GLOBAL_STREAM)]
    hub._dispatch({'type': 'crop.created', 'crop_id': 'x'})
    assert all(_drain(sub) == [] for sub in subs)


def test_only_global_families_may_go_global(hub: EventHub) -> None:
    with pytest.raises(ValueError, match='not a global event type'):
        publish_global_event('crop.created', crop_id='x')


@pytest.mark.asyncio
async def test_stats_count_only_the_asked_stream(hub: EventHub) -> None:
    await hub.subscribe(project='alpha')
    await hub.subscribe(project='beta')
    await hub.subscribe(project='beta')
    with bind_project(_record('alpha')):
        hub.publish({'type': 'crop.created', 'crop_id': 'a'})
    assert hub.stats('beta')['subscribers'] == 2
    assert hub.stats('beta')['events_published'] == 0
    assert hub.stats('alpha')['events_published'] == 1


@pytest.mark.asyncio
async def test_combine_progress_goes_global_with_target(hub: EventHub) -> None:
    """Delta 4: the combine target is ``building`` (binds refused), so its
    progress rides the global stream and names the target."""
    global_sub = await hub.subscribe(project=GLOBAL_STREAM)
    beta_sub = await hub.subscribe(project='beta')

    with bind_project(_record('alpha')):  # a bound publisher still publishes globally
        publish_global_event('combine.progress', target='cars-all', job_id='cmb_1', done=3, total=9)

    (event,) = _drain(global_sub)
    assert event['type'] == 'combine.progress'
    assert event['project'] is None
    assert event['target'] == 'cars-all'
    assert event['topic'] == 'project'
    assert (event['job_id'], event['done'], event['total']) == ('cmb_1', 3, 9)
    assert _drain(beta_sub) == []  # global events never reach a project stream


@pytest.mark.asyncio
async def test_global_event_target_is_always_on_the_wire(hub: EventHub) -> None:
    global_sub = await hub.subscribe(project=GLOBAL_STREAM)
    publish_global_event('project.deleted')
    (event,) = _drain(global_sub)
    assert event['target'] is None
    assert event['project'] is None


@pytest.mark.asyncio
async def test_scoped_events_route_subscribes_as_the_bound_project(hub: EventHub) -> None:
    """The scoped ``/events`` stream subscribes under the request's bound
    project, and the SSE generator keeps it (``test_sse_keeps_project``)."""
    from src.routers.curation.events import curation_events

    with bind_project(_record('alpha')):
        response = await curation_events(topic=None, class_id=None)
        hub.publish({'type': 'crop.created', 'topic': 'crop', 'crop_id': 'alpha-item-0001'})
    with bind_project(_record('beta')):
        hub.publish({'type': 'crop.created', 'topic': 'crop', 'crop_id': 'beta-item-0001'})

    (sub,) = hub._subscribers
    assert sub.project == 'alpha'
    body = response.body_iterator
    assert await body.__anext__() == ': connected\n\n'
    frame = await body.__anext__()
    assert 'alpha-item-0001' in frame
    assert sub.queue.empty()  # beta's event was never queued for alpha
    await body.aclose()
    assert hub._subscribers == set()


@pytest.mark.asyncio
async def test_global_events_route_subscribes_to_the_global_stream(hub: EventHub) -> None:
    from src.routers.curation.global_status import global_events

    response = await global_events(topic=None)
    (sub,) = hub._subscribers
    assert sub.project is None
    await response.body_iterator.aclose()
