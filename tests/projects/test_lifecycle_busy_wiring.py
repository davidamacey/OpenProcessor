"""P3F item 1/3: delete/archive's busy check runs through
``src.services.projects.busy.running_jobs`` (the §5.4 job inventory),
not a bespoke file scan, and the 409 ``project_busy`` body carries typed
``JobRef`` objects (Cropwright rev-3 delta 11), not raw ids.
"""

from __future__ import annotations

import asyncio

import pytest
from fastapi import HTTPException

from src.services.projects import lifecycle
from src.services.projects.registry import ProjectRegistry, set_project_registry

from .conftest import FakeLifecycleOpenSearch, seed_default_project


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    import src.config.curation as curation_mod
    from src.services.projects import capacity as capacity_mod

    monkeypatch.setenv('OP_STATE_DIR', str(tmp_path / 'state'))
    monkeypatch.setenv('OP_PROJECTS_DATA_ROOT', str(tmp_path / 'projects_data'))
    curation_mod._default_curation_config = None
    capacity_mod._cache = None
    set_project_registry(None)
    yield
    curation_mod._default_curation_config = None
    capacity_mod._cache = None
    set_project_registry(None)


def test_running_jobs_delegates_to_busy_module() -> None:
    """lifecycle.running_jobs must call busy.running_jobs, not re-scan
    job dirs itself -- proven by faking busy.running_jobs and asserting
    its fake JobRef surfaces through."""
    from src.services.projects import busy

    async def _run() -> None:
        client = FakeLifecycleOpenSearch()
        # P3F item 3 (B2(a) residual): create_project's registry
        # refresh_strict() needs a real, reachable registry bound during
        # the create itself -- set it before create_project runs.
        set_project_registry(ProjectRegistry(lambda: client))
        await seed_default_project(client)
        record, _ = await lifecycle.create_project(client, slug='cars', display_name='Cars')

        calls = []

        def _fake_running_jobs(rec):
            calls.append(rec.slug)
            return [busy.JobRef(kind='train', job_id='fake-job-1')]

        real = busy.running_jobs
        busy.running_jobs = _fake_running_jobs
        try:
            jobs = await lifecycle.running_jobs(record)
        finally:
            busy.running_jobs = real

        assert calls == ['cars']
        assert len(jobs) == 1
        assert jobs[0].kind == 'train'
        assert jobs[0].id == 'fake-job-1'

    asyncio.run(_run())


def test_archive_busy_409_carries_typed_job_refs(monkeypatch) -> None:
    from src.services.projects import busy

    async def _run() -> None:
        client = FakeLifecycleOpenSearch()
        set_project_registry(ProjectRegistry(lambda: client))
        await seed_default_project(client)
        record, _ = await lifecycle.create_project(client, slug='cars', display_name='Cars')

        monkeypatch.setattr(
            busy,
            'running_jobs',
            lambda _rec: [busy.JobRef(kind='bakeoff', job_id='run-42')],
        )

        with pytest.raises(HTTPException) as exc_info:
            await lifecycle.archive_project(client, slug='cars', expected_revision=record.revision)

        detail = exc_info.value.detail
        assert detail['error'] == 'project_busy'
        assert detail['jobs'] == [
            {
                'kind': 'bakeoff',
                'kind_label': 'Bake-off',
                'id': 'run-42',
                'label': 'run-42',
                'started_at': None,
            }
        ]

    asyncio.run(_run())


def test_delete_busy_409_carries_typed_job_refs(monkeypatch) -> None:
    from src.services.projects import busy

    async def _run() -> None:
        client = FakeLifecycleOpenSearch()
        set_project_registry(ProjectRegistry(lambda: client))
        await seed_default_project(client)
        _record, _ = await lifecycle.create_project(client, slug='cars', display_name='Cars')
        await lifecycle.create_project(client, slug='dogs', display_name='Dogs')

        monkeypatch.setattr(
            busy,
            'running_jobs',
            lambda _rec: [busy.JobRef(kind='train', job_id='run-7')],
        )

        with pytest.raises(HTTPException) as exc_info:
            await lifecycle.delete_project(client, slug='cars', confirm='cars')

        detail = exc_info.value.detail
        assert detail['error'] == 'project_busy'
        assert detail['jobs'] == [
            {
                'kind': 'train',
                'kind_label': 'Training run',
                'id': 'run-7',
                'label': 'run-7',
                'started_at': None,
            }
        ]

    asyncio.run(_run())
