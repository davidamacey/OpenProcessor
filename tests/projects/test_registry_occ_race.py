"""P3F item 5: a concurrent lifecycle write loses the OCC race cleanly
(``revision_conflict``, never a silent clobber or a bare 500). Two
callers both read the same stored revision/seq_no, then race to write;
the fake client's ``if_seq_no`` guard (tests/projects/conftest.py) makes
the second writer lose exactly like a real OpenSearch node would.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest
from fastapi import HTTPException

from src.services.projects import lifecycle
from src.services.projects.registry import (
    ProjectRegistry,
    get_record_with_seq,
    set_project_registry,
)

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


def test_two_concurrent_writers_one_loses_with_revision_conflict() -> None:
    """Both writers read the same stored seq_no/primary_term (as if two
    requests arrived back to back and each read before either wrote),
    then race their ``write_record`` calls concurrently. The fake
    client's ``if_seq_no`` guard makes exactly one lose, exactly like a
    real OpenSearch node would -- ``lifecycle.write_record`` must
    translate that loss into 409 ``revision_conflict``, never a silent
    clobber or a bare exception."""

    async def _run() -> tuple[list[str], list[BaseException]]:
        client = FakeLifecycleOpenSearch()
        # P3F item 3 (B2(a) residual): create_project's own registry
        # refresh_strict() now needs a real, reachable registry bound
        # during the create itself -- unlike the old soft ensure_fresh(),
        # it raises rather than degrading, so the registry must be set
        # before create_project runs, not after.
        set_project_registry(ProjectRegistry(lambda: client))
        await seed_default_project(client)
        _, _ = await lifecycle.create_project(client, slug='cars', display_name='Cars')

        stored, seq, term = await get_record_with_seq(client, 'cars')
        assert stored is not None

        async def _write(name: str) -> str:
            updated = replace(stored, display_name=name, revision=stored.revision + 1)
            await lifecycle.write_record(client, updated, if_seq_no=seq, if_primary_term=term)
            return name

        results = await asyncio.gather(_write('A'), _write('B'), return_exceptions=True)
        oks = [r for r in results if not isinstance(r, BaseException)]
        errs = [r for r in results if isinstance(r, BaseException)]
        return oks, errs

    oks, errs = asyncio.run(_run())

    assert len(oks) == 1, f'expected exactly one writer to win, got {oks!r}'
    assert len(errs) == 1, f'expected exactly one writer to lose, got {errs!r}'
    (err,) = errs
    assert isinstance(err, HTTPException)
    assert err.detail['error'] == 'revision_conflict'
