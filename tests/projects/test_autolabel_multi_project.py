"""auto_label_worker.py: multi-project trigger discovery runs only the
single OLDEST pending trigger per cycle, and never lets one project's
run write into another project's autolabel_dir."""

from __future__ import annotations

import asyncio
import os
import time
from datetime import UTC, datetime
from typing import Any

import pytest

from scripts.curation import auto_label_worker
from src.config.curation import base_curation_config, get_curation_config
from src.config.project_context import bind_project
from src.config.projects import ProjectRecord, resources_for_new


pytestmark = pytest.mark.unbound


async def _fake_pipeline(*, opensearch: Any, progress: Any, **kwargs: Any) -> dict[str, Any]:
    return {'ok': True, 'kwargs': kwargs}


_FAKE_PIPELINE_PATH = f'{__name__}:_fake_pipeline'


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


def _write_trigger(record: ProjectRecord, *, mtime: float) -> None:
    with bind_project(record):
        cfg = get_curation_config()
        cfg.autolabel_dir.mkdir(parents=True, exist_ok=True)
        trigger_path = cfg.autolabel_dir / 'trigger.json'
        trigger_path.write_text(
            f'{{"job_id": "{record.slug}-job", "pipeline": "{_FAKE_PIPELINE_PATH}", "args": {{}}}}'
        )
        os.utime(trigger_path, (mtime, mtime))


@pytest.fixture
def two_projects(tmp_path, monkeypatch) -> tuple[ProjectRecord, ProjectRecord]:
    monkeypatch.setenv('OP_AUTO_LABEL_STATE_DIR', str(tmp_path / 'auto_label'))
    monkeypatch.setenv('OP_STATE_DIR', str(tmp_path / 'state'))
    monkeypatch.setenv('OP_PROJECTS_DATA_ROOT', str(tmp_path / 'projects'))
    alpha, beta = _record('alpha'), _record('beta')

    now = time.time()
    _write_trigger(alpha, mtime=now - 100.0)  # older
    _write_trigger(beta, mtime=now - 10.0)  # newer
    return alpha, beta


def test_oldest_pending_trigger_picks_the_older_one(two_projects) -> None:
    alpha, beta = two_projects
    picked = auto_label_worker._oldest_pending_trigger([alpha, beta])
    assert picked is not None
    record, _mtime = picked
    assert record.slug == alpha.slug


def test_run_one_writes_only_into_its_own_project_dir(two_projects) -> None:
    alpha, beta = two_projects

    oldest = auto_label_worker._oldest_pending_trigger([alpha, beta])
    assert oldest is not None
    record, _ = oldest
    assert record.slug == 'alpha'

    trigger = auto_label_worker._claim_trigger_for(record)
    assert trigger is not None
    assert trigger['job_id'] == 'alpha-job'

    # alpha's trigger is claimed (removed); beta's is still pending.
    with bind_project(alpha):
        assert not get_curation_config().autolabel_dir.joinpath('trigger.json').exists()
    with bind_project(beta):
        assert get_curation_config().autolabel_dir.joinpath('trigger.json').exists()

    asyncio.run(auto_label_worker._run_one(alpha, trigger, opensearch=None))

    with bind_project(alpha):
        alpha_dir = get_curation_config().autolabel_dir
    with bind_project(beta):
        beta_dir = get_curation_config().autolabel_dir

    assert (alpha_dir / 'state.json').exists()
    # Beta's dir got nothing from alpha's run -- only its own untouched trigger.
    assert {p.name for p in beta_dir.iterdir()} == {'trigger.json'}


def test_only_one_job_dispatched_per_discovery_cycle(two_projects) -> None:
    """A single discovery pass claims and runs at most one project's
    trigger -- the other project's trigger is left for the next cycle."""
    alpha, beta = two_projects

    picked = auto_label_worker._oldest_pending_trigger([alpha, beta])
    assert picked is not None
    record, _ = picked
    trigger = auto_label_worker._claim_trigger_for(record)
    assert trigger is not None
    asyncio.run(auto_label_worker._run_one(record, trigger, opensearch=None))

    # Only one trigger was claimed+run this cycle; beta's is untouched.
    remaining = auto_label_worker._oldest_pending_trigger([alpha, beta])
    assert remaining is not None
    assert remaining[0].slug == 'beta'
