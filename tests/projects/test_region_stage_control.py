"""The region-stage pause route and the re-run of gate-skipped items."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

import pytest
from _curation_app import SCOPED, mount_curation_routers
from curation.reprocess_fixtures import F, RegionStatus, docs, item, make_fake
from fastapi import FastAPI
from fastapi.testclient import TestClient

from scripts.curation._project_worker_utils import REGION_STAGE_PAUSED_FLAG_NAME
from src.services.curation.region_stage_control import rerun_skipped_request
from src.services.curation.reprocess import apply_reprocess


if TYPE_CHECKING:
    from pathlib import Path

NO_BOX = RegionStatus.NO_REGION_BOX.value
URL = f'{SCOPED}/region_stage'


def _corpus() -> list[dict[str, Any]]:
    skipped = item('skip', NO_BOX)
    skipped[F.gate_skip] = 'tier3_hit_rate'
    return [
        item('p1', RegionStatus.PENDING_DETECTION.value),
        item('p2', RegionStatus.PENDING_DETECTION.value),
        item('v1', RegionStatus.PENDING_VERIFICATION.value),
        skipped,
        item('miss', NO_BOX),  # a real miss: the segmenter looked and found nothing
    ]


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    import src.config.curation as curation_config
    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    monkeypatch.setenv('OP_STATE_DIR', str(tmp_path))
    monkeypatch.setattr(curation_config, '_default_curation_config', None)
    monkeypatch.setattr('src.routers.curation._ensure_indexes', AsyncMock(return_value=None))
    fake = make_fake(_corpus())
    app = FastAPI()
    mount_curation_routers(app, curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake
    return {'client': TestClient(app), 'fake': fake, 'state': tmp_path / 'projects' / 'default'}


@pytest.mark.usefixtures('not_stale_registry', 'reference_region_profile')
def test_pause_and_resume_flip_the_flag_the_worker_reads_and_report_the_waiting_work(
    env: dict[str, Any],
) -> None:
    client, flag = env['client'], env['state'] / REGION_STAGE_PAUSED_FLAG_NAME

    state = client.get(URL).json()
    assert state['paused'] is False
    assert state['counts'] == {'pending_detection': 2, 'pending_verification': 1, 'gate_skipped': 1}
    assert state['rerun_skipped']['scopes'] == ['region']

    paused = client.post(f'{URL}/pause').json()
    assert (paused['paused'], paused['pipeline_paused']) == (True, False)
    assert paused['paused_since'] is not None
    assert flag.exists()
    assert client.post(f'{URL}/pause').json()['paused'] is True  # idempotent

    resumed = client.post(f'{URL}/resume').json()
    assert (resumed['paused'], resumed['paused_since']) == (False, None)
    assert not flag.exists()
    assert client.post(f'{URL}/resume').status_code == 200  # idempotent
    # Pausing never touched an item.
    assert docs(env['fake'])['p1'][F.status] == RegionStatus.PENDING_DETECTION.value


@pytest.mark.usefixtures('not_stale_registry')
def test_without_a_region_profile_the_stage_routes_409(env: dict[str, Any]) -> None:
    assert env['client'].post(f'{URL}/pause').status_code == 409
    assert not (env['state'] / REGION_STAGE_PAUSED_FLAG_NAME).exists()


@pytest.mark.asyncio
async def test_the_rerun_request_requeues_only_the_gate_skipped_items() -> None:
    fake = make_fake(_corpus())
    body = rerun_skipped_request().model_copy(update={'dry_run': False})

    resp = await apply_reprocess(fake, body)

    (region,) = resp.scopes
    assert (region.selected, region.queued) == (1, 1)
    after = docs(fake)
    assert after['skip'][F.status] == RegionStatus.PENDING_DETECTION
    assert after['skip'].get(F.gate_skip) is None
    assert after['miss'][F.status] == NO_BOX  # a real miss is not re-run by this request
