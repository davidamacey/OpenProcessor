"""Script / worker entry points: ``--project`` binds the whole process, and
their OpenSearch clients carry the same project guard as the API's."""

from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime
from typing import Any

import pytest

from src.config.curation import base_curation_config
from src.config.project_context import ProjectNotBound, bind_process_project, current_project
from src.config.projects import ProjectRecord, resources_for_new
from src.services.projects import guard, script_binding
from src.services.projects.registry import record_to_doc


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


class _FakeInnerTransport:
    """Answers the registry reads (``op_projects``) and records every
    other call that got past the guard."""

    def __init__(self, records: list[ProjectRecord]) -> None:
        self.docs = {f'project:{r.slug}': record_to_doc(r) for r in records}
        self.passed: list[str] = []

    async def perform_request(self, method: str, url: str, **_kwargs: Any) -> Any:
        if url.startswith('/op_projects/_doc/'):
            if url.endswith('meta%3Aprojects_revision'):
                return {'_source': {'revision': 1}}
            raise KeyError(url)
        if url == '/op_projects/_search':
            return {'hits': {'hits': [{'_source': d} for d in self.docs.values()]}}
        self.passed.append(f'{method} {url}')
        return {'hits': {'hits': []}}

    async def close(self) -> None:
        return None


_REAL_LOAD_REGISTRY = script_binding.load_registry


@pytest.fixture(autouse=True)
def _clear_process_binding(monkeypatch: pytest.MonkeyPatch) -> Any:
    # The real registry read, not tests/conftest.py's default-only stub.
    monkeypatch.setattr(script_binding, 'load_registry', _REAL_LOAD_REGISTRY)
    yield
    bind_process_project(None)


_REAL_FACTORY = guard.make_script_opensearch


def _guarded_client(records: list[ProjectRecord]) -> Any:
    inner = _FakeInnerTransport(records)
    client = _REAL_FACTORY(['http://127.0.0.1:9'])
    # Swap the real transport under the guard for the fake one.
    client.transport._inner = inner
    client.transport._registry._client_factory = lambda: guard._RegistryReader(inner)
    return client, inner


def test_project_argument_defaults_to_env(monkeypatch: pytest.MonkeyPatch) -> None:
    parser = argparse.ArgumentParser()
    monkeypatch.setenv('OP_PROJECT', 'beta')
    script_binding.add_project_argument(parser)
    assert parser.parse_args([]).project == 'beta'
    assert parser.parse_args(['--project', 'alpha']).project == 'alpha'


def test_default_binds_the_whole_process_from_the_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    def _factory(_hosts: list[str], **_kw: Any) -> Any:
        client, _inner = _guarded_client([])
        return client

    monkeypatch.setattr(guard, 'make_script_opensearch', _factory)
    with pytest.raises(ProjectNotBound):
        current_project()
    record = script_binding.bind_script_project('default')
    assert current_project().record.slug == 'default'
    assert not current_project().read_only
    assert record.resources.indexes == current_project().record.resources.indexes


def test_archived_default_binds_read_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """The stored status applies to ``default`` too: an archived default
    is never writable from a script."""
    import dataclasses

    from src.services.projects.registry import default_project_record

    archived = dataclasses.replace(default_project_record(), status='archived')

    def _factory(_hosts: list[str], **_kw: Any) -> Any:
        client, _inner = _guarded_client([archived])
        return client

    monkeypatch.setattr(guard, 'make_script_opensearch', _factory)
    script_binding.bind_script_project('default')
    assert current_project().read_only


def test_unreadable_registry_refuses_to_bind(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Down:
        async def perform_request(self, *_a: Any, **_k: Any) -> Any:
            raise ConnectionError('opensearch down')

        async def close(self) -> None:
            return None

    def _factory(_hosts: list[str], **_kw: Any) -> Any:
        client = _REAL_FACTORY(['http://127.0.0.1:9'])
        client.transport._inner = _Down()
        return client

    monkeypatch.setattr(guard, 'make_script_opensearch', _factory)
    with pytest.raises(SystemExit, match='cannot read the project registry'):
        script_binding.bind_script_project('default')
    with pytest.raises(ProjectNotBound):
        current_project()


def test_script_client_is_guarded() -> None:
    alpha, beta = _record('alpha'), _record('beta')
    client, inner = _guarded_client([alpha, beta])
    bind_process_project(beta)

    async def _run() -> None:
        await client.search(index='op_prj_beta__items', body={})
        with pytest.raises(guard.CrossProjectAccess):
            await client.search(index='op_prj_alpha__items', body={})

    asyncio.run(_run())
    assert inner.passed == ['POST /op_prj_beta__items/_search']


def test_unknown_or_building_project_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    building = _record('newbie', status='building')

    def _factory(_hosts: list[str], **_kw: Any) -> Any:
        client, _inner = _guarded_client([building])
        return client

    monkeypatch.setattr(guard, 'make_script_opensearch', _factory)
    with pytest.raises(SystemExit, match="no project named 'ghost'"):
        script_binding.bind_script_project('ghost')
    with pytest.raises(SystemExit, match='is building'):
        script_binding.bind_script_project('newbie')
