"""P1: GET /projects and GET /projects/{project} (§4 rows 1 and 3, rev 2
list-membership + capacity + prefix rules)."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Literal

import pytest

from src.config.curation import base_curation_config
from src.config.projects import ProjectRecord, resources_for_new
from src.routers.curation import projects as projects_router
from src.routers.curation._config_common_models import ConfigErrorDetail
from src.services.projects.registry import ProjectRegistry


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


class _FakeClient:
    """No index data; ``_cat/indices`` and capacity calls degrade
    gracefully (empty counts, capacity None) without needing a live
    cluster."""

    class _Transport:
        async def perform_request(self, method, url, params=None, **kwargs):  # noqa: ARG002
            raise ConnectionError('no fake cluster wired for this test')

    def __init__(self) -> None:
        self.transport = self._Transport()


def _registry_with(*records: ProjectRecord) -> ProjectRegistry:
    reg = ProjectRegistry(lambda: None)
    reg._by_slug = {r.slug: r for r in records}
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

    result = asyncio.run(projects_router.list_projects(include_archived=False))
    assert {p.slug for p in result.projects} == {'default'}


def test_list_membership_archived_only_with_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    reg = _registry_with(_record('default'), _record('cars', status='archived'))
    monkeypatch.setattr(projects_router, 'get_project_registry', lambda: reg)

    result = asyncio.run(projects_router.list_projects(include_archived=True))
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

    result = asyncio.run(projects_router.list_projects(include_archived=False))
    assert {p.slug for p in result.projects} == {
        'default',
        'building-one',
        'failed-one',
        'deleting-one',
    }


def test_selectable_and_writable_rules(monkeypatch: pytest.MonkeyPatch) -> None:
    reg = _registry_with(_record('default'), _record('archived-one', status='archived'))
    monkeypatch.setattr(projects_router, 'get_project_registry', lambda: reg)

    result = asyncio.run(projects_router.list_projects(include_archived=True))
    by_slug = {p.slug: p for p in result.projects}
    assert by_slug['default'].selectable is True
    assert by_slug['default'].writable is True
    assert by_slug['archived-one'].selectable is True
    assert by_slug['archived-one'].writable is False


def test_deletable_only_false_for_default(monkeypatch: pytest.MonkeyPatch) -> None:
    reg = _registry_with(_record('default'), _record('cars'))
    monkeypatch.setattr(projects_router, 'get_project_registry', lambda: reg)

    result = asyncio.run(projects_router.list_projects(include_archived=False))
    by_slug = {p.slug: p for p in result.projects}
    assert by_slug['default'].deletable is False
    assert by_slug['cars'].deletable is True


def test_prefix_matches_project_api_base(monkeypatch: pytest.MonkeyPatch) -> None:
    reg = _registry_with(_record('cars'))
    monkeypatch.setattr(projects_router, 'get_project_registry', lambda: reg)

    result = asyncio.run(projects_router.list_projects(include_archived=False))
    assert result.projects[0].prefix == '/curation/projects/cars'


def test_capacity_null_when_cluster_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    reg = _registry_with(_record('default'))
    monkeypatch.setattr(projects_router, 'get_project_registry', lambda: reg)

    result = asyncio.run(projects_router.list_projects(include_archived=False))
    assert result.capacity is None


def test_limits_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    reg = _registry_with(_record('default'))
    monkeypatch.setattr(projects_router, 'get_project_registry', lambda: reg)

    result = asyncio.run(projects_router.list_projects(include_archived=False))
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
    detail = excinfo.value.detail
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

    result = asyncio.run(projects_router.list_projects(include_archived=False))
    assert set(result.projects[0].model_dump()) == SUMMARY_KEYS
    assert result.projects[0].revision == 1


def test_list_serves_status_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    reg = _registry_with(_record('default'))
    monkeypatch.setattr(projects_router, 'get_project_registry', lambda: reg)

    result = asyncio.run(projects_router.list_projects(include_archived=False))
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

    result = asyncio.run(projects_router.list_projects(include_archived=False))
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

    result = asyncio.run(projects_router.list_projects(include_archived=True))
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
    keymap_clone_conflicts: [...]}`` -- the last one (W2b) is only ever
    non-empty on a ``clone_settings`` response whose ``keymap`` axis
    dropped a conflicting action."""
    from src.routers.curation._project_models import ProjectLifecycleResponse, ProjectWarning

    assert set(ProjectLifecycleResponse.model_fields) == {
        'project',
        'warnings',
        'keymap_clone_conflicts',
    }
    assert set(ProjectWarning.model_fields) == {'code', 'message'}


def test_revision_conflict_is_an_error_code() -> None:
    detail = ConfigErrorDetail(error='revision_conflict', message='stale', current_revision=5)
    assert detail.current_revision == 5
