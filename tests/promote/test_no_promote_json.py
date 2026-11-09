"""`promote.json`, its lease and the detached wait are gone; the registry is the source."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.services.training import promote_job


SRC = Path(__file__).resolve().parents[2] / 'src'
EXEMPT = ('src/services/upgrade/',)  # the 0.6.0 reader parses the old files by design


def test_no_source_file_outside_the_upgrade_reader_names_promote_json() -> None:
    hits = []
    for path in SRC.rglob('*.py'):
        rel = path.relative_to(SRC.parent).as_posix()
        if rel.startswith(EXEMPT):
            continue
        text = path.read_text(encoding='utf-8')
        if 'promote.json' in text or 'promote_json' in text:
            hits.append(rel)
    assert hits == []


def test_the_promote_json_module_and_the_detached_wait_are_deleted() -> None:
    assert not (SRC / 'services/training/promote_json.py').exists()
    for name in ('run_detached', '_detached', 'claim', 'reconcile_orphaned_jobs', 'jobs_root'):
        assert not hasattr(promote_job, name), name


@pytest.mark.asyncio
async def test_the_promoted_model_listing_keeps_its_shape(model_registry_fake) -> None:
    """Control: the entries the listing route builds still carry the same four keys."""
    from src.services.training.promoted_models import discover_promoted_models

    model_registry_fake.promote('op_vehicle_v1', owner='default', labels=['car'], run_job_id='r1')

    [entry] = await discover_promoted_models()
    assert set(entry) == {'name', 'job_id', 'version', 'promoted_at'}
