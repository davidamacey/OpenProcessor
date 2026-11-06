"""Script / worker entry points: ``--project`` binds the whole process, and
their OpenSearch clients carry the same project guard as the API's."""

from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime
from typing import Any

import pytest

from src.config.curation_accessors import base_curation_config
from src.config.project_context import ProjectNotBound, bind_process_project, current_project
from src.config.projects import ProjectRecord, resources_for_new
from src.services.projects import guard, guard_errors, registry as registry_mod, script_binding


# Binding is what is under test here.
pytestmark = pytest.mark.unbound


def _record(slug: str, status: str = 'active') -> ProjectRecord:
    now = datetime.now(UTC).isoformat()
    return ProjectRecord(
        slug=slug,
        display_name=slug,
        description='',
        status=status,  # type: ignore[arg-type]
        revision=1,
        created_at=now,
        updated_at=now,
        origin=None,
        resources=resources_for_new(slug, base_curation_config()),
    )


class _StaticSource:
    """A registry source over a fixed list of records, or one that is down."""

    def __init__(self, records: list[ProjectRecord], *, down: bool = False) -> None:
        self.records = records
        self.down = down

    def read_revision(self) -> int:
        if self.down:
            raise ConnectionError('database down')
        return 1

    def read_all(self) -> tuple[list[ProjectRecord], frozenset[str]]:
        return list(self.records), frozenset()

    def read_one(self, slug: str) -> ProjectRecord | None:
        return next((r for r in self.records if r.slug == slug), None)


class _FakeInnerTransport:
    """Records every call that got past the guard."""

    def __init__(self) -> None:
        self.passed: list[str] = []

    async def perform_request(self, method: str, url: str, **_kwargs: Any) -> Any:
        self.passed.append(f'{method} {url}')
        return {'hits': {'hits': []}}

    async def close(self) -> None:
        return None


_REAL_LOAD_REGISTRY = script_binding.load_registry
_REAL_REGISTRY = registry_mod.ProjectRegistry


def _serve(
    monkeypatch: pytest.MonkeyPatch, records: list[ProjectRecord], *, down: bool = False
) -> None:
    """Every registry a script builds reads ``records`` instead of the database."""

    def build(source: Any = None) -> Any:
        return _REAL_REGISTRY(source or _StaticSource(records, down=down))

    monkeypatch.setattr(registry_mod, 'ProjectRegistry', build)
    monkeypatch.setattr(script_binding, 'ProjectRegistry', build)


@pytest.fixture(autouse=True)
def _clear_process_binding(monkeypatch: pytest.MonkeyPatch) -> Any:
    # The real registry read, not tests/conftest.py's default-only stub.
    monkeypatch.setattr(script_binding, 'load_registry', _REAL_LOAD_REGISTRY)
    yield
    bind_process_project(None)


def test_project_argument_defaults_to_env(monkeypatch: pytest.MonkeyPatch) -> None:
    parser = argparse.ArgumentParser()
    monkeypatch.setenv('OP_CURATION_PROJECT', 'beta')
    script_binding.add_project_argument(parser)
    assert parser.parse_args([]).project == 'beta'
    assert parser.parse_args(['--project', 'alpha']).project == 'alpha'


def test_default_binds_the_whole_process_from_the_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    _serve(monkeypatch, [_record('default')])
    with pytest.raises(ProjectNotBound):
        current_project()
    record = script_binding.bind_script_project('default')
    assert current_project().record.slug == 'default'
    assert not current_project().read_only
    assert record.resources.indexes == current_project().record.resources.indexes


def test_archived_default_binds_read_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """The stored status applies to ``default`` too: an archived default
    is never writable from a script."""
    _serve(monkeypatch, [_record('default', status='archived')])
    script_binding.bind_script_project('default')
    assert current_project().read_only


def test_unreadable_registry_refuses_to_bind(monkeypatch: pytest.MonkeyPatch) -> None:
    _serve(monkeypatch, [], down=True)
    with pytest.raises(SystemExit, match='cannot read the project registry'):
        script_binding.bind_script_project('default')
    with pytest.raises(ProjectNotBound):
        current_project()


def test_script_client_is_guarded(monkeypatch: pytest.MonkeyPatch) -> None:
    alpha, beta = _record('alpha'), _record('beta')
    _serve(monkeypatch, [alpha, beta])
    inner = _FakeInnerTransport()
    client = guard.make_script_opensearch(['http://127.0.0.1:9'])
    client.transport._inner = inner
    bind_process_project(beta)

    async def _run() -> None:
        await client.search(index='op_prj_beta__items', body={})
        with pytest.raises(guard_errors.CrossProjectAccess):
            await client.search(index='op_prj_alpha__items', body={})

    asyncio.run(_run())
    assert inner.passed == ['POST /op_prj_beta__items/_search']


def test_unknown_or_building_project_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    _serve(monkeypatch, [_record('newbie', status='building')])
    with pytest.raises(SystemExit, match="no project named 'ghost'"):
        script_binding.bind_script_project('ghost')
    with pytest.raises(SystemExit, match='is building'):
        script_binding.bind_script_project('newbie')
