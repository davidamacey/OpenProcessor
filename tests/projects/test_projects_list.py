"""P1: GET /projects and GET /projects/{project} (§4 rows 1 and 3, rev 2
list-membership + capacity + prefix rules)."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any, Literal

import pytest
from fastapi import HTTPException
from opensearchpy.exceptions import ConnectionError as OpenSearchConnectionError

from src.config.curation_accessors import base_curation_config
from src.config.projects import ProjectRecord, resources_for_new
from src.routers.curation import projects as projects_router
from src.schemas.curation.config_common import ConfigErrorDetail
from src.services.projects.registry import ProjectRegistry


pytestmark = [pytest.mark.db, pytest.mark.usefixtures('pg_registry')]


def _record(
    slug: str,
    status: Literal['building', 'active', 'archived', 'deleting', 'deleted', 'failed'] = 'active',
) -> ProjectRecord:
    now = datetime.now(UTC).isoformat()
    resources = (
        resources_for_new('default', base_curation_config())
        if slug == 'default'
        else resources_for_new(slug, base_curation_config())
    )
    return ProjectRecord(
        slug=slug,
        display_name=slug.title(),
        description='',
        status=status,
        revision=1,
        created_at=now,
        updated_at=now,
        origin=None,
        resources=resources,
    )


def _unreachable() -> OpenSearchConnectionError:
    return OpenSearchConnectionError('N/A', 'no fake cluster wired for this test', OSError('none'))


class _FakeClient:
    """An unreachable cluster: ``_count`` and capacity calls degrade gracefully
    (unknown counts, capacity None) without needing a live cluster."""

    class _Transport:
        async def perform_request(self, method, url, params=None, **kwargs):
            raise _unreachable()

    def __init__(self) -> None:
        self.transport = self._Transport()

    async def count(self, **_: object) -> dict[str, int]:
        raise _unreachable()


def _seed(records: tuple[ProjectRecord, ...]) -> None:
    from src.db.repos import projects as repo
    from src.db.session import async_session_factory, deployment_tx

    async def go() -> None:
        for r in records:
            new = repo.NewProject(
                slug=r.slug,
                display_name=r.display_name,
                index_prefix='op_prj_',
                storage_prefix=f'projects/{r.slug}',
                model_prefix=f'{r.slug}__',
                mlflow_experiment=f'openprocessor-{r.slug}',
                indexes={'images': f'op_prj_{r.slug}__images', 'items': f'op_prj_{r.slug}__items'},
                created_by='api',
                status='active' if r.status in ('deleted', 'deleting') else r.status,
            )
            async with async_session_factory()() as session, deployment_tx(session, 'seed'):
                await repo.create_project(session, new)
            if r.status in ('deleted', 'deleting'):
                async with async_session_factory()() as session, deployment_tx(session, 'seed'):
                    await repo.mark_deleting(session, r.slug, from_states=('active',))
            if r.status == 'deleted':
                async with async_session_factory()() as session, deployment_tx(session, 'seed'):
                    await repo.finish_delete(session, r.slug)

    asyncio.run(go())


def _list(**kwargs: Any) -> Any:
    from src.db.session import async_session_factory

    async def go() -> Any:
        async with async_session_factory()() as session:
            return await projects_router.list_projects(session=session, **kwargs)

    return asyncio.run(go())


def _registry_with(*records: ProjectRecord) -> ProjectRegistry:
    _seed(records)
    reg = ProjectRegistry()
    reg._by_slug = {r.slug: r for r in records if r.status != 'deleted'}
    reg._retired = frozenset(r.slug for r in records if r.status == 'deleted')
    reg._revision = 1
    return reg


@pytest.fixture(autouse=True)
def _patch_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _fake_registry_ensure_fresh(self) -> None:
        return None

    monkeypatch.setattr(ProjectRegistry, 'ensure_fresh', _fake_registry_ensure_fresh)

    async def _fake_make_curation_opensearch():
        return _FakeClient()

    monkeypatch.setattr(projects_router, 'make_curation_opensearch', _fake_make_curation_opensearch)


def test_list_membership_active_always_listed(monkeypatch: pytest.MonkeyPatch) -> None:
    reg = _registry_with(_record('default'), _record('cars', status='archived'))
    monkeypatch.setattr(projects_router, 'get_project_registry', lambda: reg)

    result = _list(include_archived=False)
    assert {p.slug for p in result.projects} == {'default'}


def test_list_membership_archived_only_with_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    reg = _registry_with(_record('default'), _record('cars', status='archived'))
    monkeypatch.setattr(projects_router, 'get_project_registry', lambda: reg)

    result = _list(include_archived=True)
    assert {p.slug for p in result.projects} == {'default', 'cars'}


def test_list_membership_building_failed_deleting_always_listed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reg = _registry_with(
        _record('default'),
        _record('building-one', status='building'),
        _record('failed-one', status='failed'),
        _record('deleting-one', status='deleting'),
        _record('gone', status='deleted'),
    )
    monkeypatch.setattr(projects_router, 'get_project_registry', lambda: reg)

    result = _list(include_archived=False)
    assert {p.slug for p in result.projects} == {
        'default',
        'building-one',
        'failed-one',
        'deleting-one',
    }


def test_selectable_and_writable_rules(monkeypatch: pytest.MonkeyPatch) -> None:
    reg = _registry_with(_record('default'), _record('archived-one', status='archived'))
    monkeypatch.setattr(projects_router, 'get_project_registry', lambda: reg)

    result = _list(include_archived=True)
    by_slug = {p.slug: p for p in result.projects}
    assert by_slug['default'].selectable is True
    assert by_slug['default'].writable is True
    assert by_slug['archived-one'].selectable is True
    assert by_slug['archived-one'].writable is False


def test_deletable_only_false_for_default(monkeypatch: pytest.MonkeyPatch) -> None:
    reg = _registry_with(_record('default'), _record('cars'))
    monkeypatch.setattr(projects_router, 'get_project_registry', lambda: reg)

    result = _list(include_archived=False)
    by_slug = {p.slug: p for p in result.projects}
    assert by_slug['default'].deletable is False
    assert by_slug['cars'].deletable is True


def test_prefix_matches_project_api_base(monkeypatch: pytest.MonkeyPatch) -> None:
    reg = _registry_with(_record('cars'))
    monkeypatch.setattr(projects_router, 'get_project_registry', lambda: reg)

    result = _list(include_archived=False)
    assert result.projects[0].prefix == '/curation/projects/cars'


def test_capacity_null_when_cluster_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    reg = _registry_with(_record('default'))
    monkeypatch.setattr(projects_router, 'get_project_registry', lambda: reg)

    result = _list(include_archived=False)
    assert result.capacity is None


def test_limits_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    reg = _registry_with(_record('default'))
    monkeypatch.setattr(projects_router, 'get_project_registry', lambda: reg)

    result = _list(include_archived=False)
    assert result.limits.slug_min == 2
    assert 'combine' in result.limits.reserved_slugs
    assert result.default_slug == 'default'


def test_get_project_returns_resources_and_null_error(monkeypatch: pytest.MonkeyPatch) -> None:
    reg = _registry_with(_record('cars'))
    monkeypatch.setattr(projects_router, 'get_project_registry', lambda: reg)

    result = asyncio.run(projects_router.get_project(project='cars'))
    assert result.slug == 'cars'
    assert result.error is None
    assert 'items' in result.resources['indexes']


def test_get_project_missing_raises_404(monkeypatch: pytest.MonkeyPatch) -> None:
    reg = _registry_with(_record('cars'))
    monkeypatch.setattr(projects_router, 'get_project_registry', lambda: reg)

    with pytest.raises(Exception) as excinfo:  # noqa: PT011 - HTTPException from api_error
        asyncio.run(projects_router.get_project(project='missing'))
    detail = excinfo.value.detail  # type: ignore[attr-defined]
    parsed = ConfigErrorDetail.model_validate(detail)
    assert parsed.error == 'project_not_found'


# --- Review deltas 2 and 3 (exact wire shapes Cropwright consumes) ---

SUMMARY_KEYS = {
    'slug',
    'display_name',
    'description',
    'prefix',
    'status',
    'writable',
    'selectable',
    'is_default',
    'deletable',
    'archivable',
    'unarchivable',
    'revision',
    'created_at',
    'updated_at',
    'counts',
    'origin',
    'paused',
}


def test_summary_exact_key_set(monkeypatch: pytest.MonkeyPatch) -> None:
    reg = _registry_with(_record('default'))
    monkeypatch.setattr(projects_router, 'get_project_registry', lambda: reg)

    result = _list(include_archived=False)
    assert set(result.projects[0].model_dump()) == SUMMARY_KEYS
    assert result.projects[0].revision == 1


def test_list_serves_status_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    reg = _registry_with(_record('default'))
    monkeypatch.setattr(projects_router, 'get_project_registry', lambda: reg)

    result = _list(include_archived=False)
    assert set(result.labels.status) == {
        'building',
        'active',
        'archived',
        'deleting',
        'deleted',
        'failed',
    }
    assert all(result.labels.status.values())


def test_list_serves_retired_slugs(monkeypatch: pytest.MonkeyPatch) -> None:
    reg = _registry_with(_record('default'), _record('gone', status='deleted'))
    monkeypatch.setattr(projects_router, 'get_project_registry', lambda: reg)

    result = _list(include_archived=False)
    assert result.limits.retired_slugs == ['gone']
    assert 'gone' not in {p.slug for p in result.projects}


@pytest.mark.parametrize(
    ('status', 'selectable', 'writable'),
    [
        ('active', True, True),
        ('archived', True, False),
        ('building', False, False),
        ('failed', False, False),
        ('deleting', False, False),
    ],
)
def test_selectable_writable_per_status(
    monkeypatch: pytest.MonkeyPatch, status: str, selectable: bool, writable: bool
) -> None:
    reg = _registry_with(_record('default'), _record('cars', status=status))  # type: ignore[arg-type]
    monkeypatch.setattr(projects_router, 'get_project_registry', lambda: reg)

    result = _list(include_archived=True)
    cars = next(p for p in result.projects if p.slug == 'cars')
    assert (cars.selectable, cars.writable) == (selectable, writable)


def test_get_project_exact_key_set(monkeypatch: pytest.MonkeyPatch) -> None:
    reg = _registry_with(_record('cars'))
    monkeypatch.setattr(projects_router, 'get_project_registry', lambda: reg)

    result = asyncio.run(projects_router.get_project(project='cars'))
    assert set(result.model_dump()) == SUMMARY_KEYS | {'resources', 'error'}


def test_lifecycle_envelope_shape() -> None:
    """P3's create/patch/archive/unarchive/clone_settings all answer
    ``{project: ProjectSummary, warnings: [{code, message}],
    keymap_clone_conflicts: [...], job_id}`` -- ``keymap_clone_conflicts`` is only ever
    non-empty on a ``clone_settings`` response whose ``keymap`` axis
    dropped a conflicting action, and ``job_id`` is set only by a delete, which finishes in
    its ``project_delete`` job."""
    from src.schemas.curation.projects import ProjectLifecycleResponse, ProjectWarning

    assert set(ProjectLifecycleResponse.model_fields) == {
        'project',
        'warnings',
        'keymap_clone_conflicts',
        'job_id',
    }
    assert set(ProjectWarning.model_fields) == {'code', 'message', 'detail'}


def test_revision_conflict_is_an_error_code() -> None:
    detail = ConfigErrorDetail(error='revision_conflict', message='stale', current_revision=5)
    assert detail.current_revision == 5


def _many(monkeypatch: pytest.MonkeyPatch) -> None:
    reg = _registry_with(
        _record('default'),
        *(_record(f'cars-{i:02d}') for i in range(5)),
        _record('faces'),
    )
    monkeypatch.setattr(projects_router, 'get_project_registry', lambda: reg)


def test_q_filters_by_slug_or_name_on_the_server(monkeypatch: pytest.MonkeyPatch) -> None:
    _many(monkeypatch)
    result = _list(include_archived=False, q='CARS')
    assert [p.slug for p in result.projects] == [f'cars-{i:02d}' for i in range(5)]
    assert result.total == 5


def test_limit_caps_the_page_and_total_says_how_many_matched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _many(monkeypatch)
    result = _list(include_archived=False, limit=2)
    assert [p.slug for p in result.projects] == ['cars-00', 'cars-01']
    assert result.total == 7


def test_no_q_or_limit_lists_everything(monkeypatch: pytest.MonkeyPatch) -> None:
    _many(monkeypatch)
    result = _list(include_archived=False)
    assert len(result.projects) == result.total == 7


def test_selectable_filter_counts_only_openable_projects(monkeypatch: pytest.MonkeyPatch) -> None:
    reg = _registry_with(
        _record('default'),
        _record('building-one', status='building'),
        _record('deleting-one', status='deleting'),
        _record('cars'),
    )
    monkeypatch.setattr(projects_router, 'get_project_registry', lambda: reg)
    result = _list(include_archived=False, selectable=True)
    assert [p.slug for p in result.projects] == ['cars', 'default']
    assert result.total == 2


def _sixty(monkeypatch: pytest.MonkeyPatch) -> None:
    reg = _registry_with(
        _record('default'),
        *(_record(f'proj-{i:02d}') for i in range(58)),
        _record('building-one', status='building'),
    )
    monkeypatch.setattr(projects_router, 'get_project_registry', lambda: reg)


def test_selectable_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    _sixty(monkeypatch)
    result = _list(include_archived=False, selectable=True)
    assert 'building-one' not in {p.slug for p in result.projects}
    assert all(p.selectable for p in result.projects)
    assert result.total == 59


def test_status_filter_is_repeatable(monkeypatch: pytest.MonkeyPatch) -> None:
    _sixty(monkeypatch)
    result = _list(include_archived=False, status=['building', 'failed'])
    assert [p.slug for p in result.projects] == ['building-one']


def test_cursor_walks_all_projects(monkeypatch: pytest.MonkeyPatch) -> None:
    _sixty(monkeypatch)
    seen: list[str] = []
    cursor = None
    pages = 0
    while True:
        page = _list(include_archived=False, limit=25, cursor=cursor)
        pages += 1
        seen += [p.slug for p in page.projects]
        assert page.total == 60
        cursor = page.next_cursor
        if cursor is None:
            break
    assert pages == 3
    assert len(seen) == len(set(seen)) == 60
    assert seen == sorted(seen)


def test_cursor_walks_a_descending_sort(monkeypatch: pytest.MonkeyPatch) -> None:
    _sixty(monkeypatch)
    seen: list[str] = []
    cursor = None
    while True:
        page = _list(include_archived=False, limit=7, cursor=cursor, sort='updated_at:desc')
        seen += [p.slug for p in page.projects]
        cursor = page.next_cursor
        if cursor is None:
            break
    assert len(seen) == len(set(seen)) == 60


def test_cursor_from_another_sort_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    _sixty(monkeypatch)
    first = _list(include_archived=False, limit=5)
    with pytest.raises(HTTPException) as excinfo:
        _list(include_archived=False, limit=5, cursor=first.next_cursor, sort='items:desc')
    assert excinfo.value.status_code == 400


def test_sort_options_served(monkeypatch: pytest.MonkeyPatch) -> None:
    _sixty(monkeypatch)
    result = _list(include_archived=False, limit=1)
    assert [o['id'] for o in result.sort_options] == ['name:asc', 'updated_at:desc', 'items:desc']
    assert all(o['label'] for o in result.sort_options)
    assert result.sort == 'name:asc'


def test_unknown_sort_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    _sixty(monkeypatch)
    with pytest.raises(HTTPException) as excinfo:
        _list(include_archived=False, sort='bogus')
    assert excinfo.value.status_code == 400


def test_items_sort_states_its_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    _sixty(monkeypatch)
    ranked = _list(include_archived=False, limit=5, sort='items:desc')
    assert ranked.sort_cap == 200
    assert ranked.next_cursor is None
    assert len(ranked.projects) == 5
    assert _list(include_archived=False, limit=5).sort_cap is None


def test_status_options_are_served_with_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    _sixty(monkeypatch)
    result = _list(include_archived=False, limit=1)
    assert result.status_options == [
        {'id': 'active', 'label': 'Active'},
        {'id': 'archived', 'label': 'Archived'},
        {'id': 'building', 'label': 'Building'},
        {'id': 'failed', 'label': 'Failed'},
        {'id': 'deleting', 'label': 'Deleting'},
    ]
