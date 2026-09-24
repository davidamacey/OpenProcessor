"""Per-run ``detection_profile`` / ``prompt_pack`` overrides on the auto-label
endpoints (``POST /pipeline/auto_label/start`` and ``POST /pipeline/auto_label``).

An explicit id overrides the settings-doc default for that one job (never
written to settings); an unknown id is a 422 listing the valid ids; omitted
keeps the settings default. The resolved ids are echoed in the job ``args``
and actually reach the VLM labeler the job uses.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from curation.test_pipeline import _FakeClassEntry, _FakeOpenSearch, _FakeRegistry
from src.config.curation import CurationConfig
from src.services.labeling.vlm_prompts import GENERIC_ITEM_PACK


if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


@pytest.fixture
def packs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> CurationConfig:
    """Default pack ``pallet_v1`` + extra selectable pack ``food_v2``."""
    paths = {}
    for name in ('pallet_v1', 'food_v2'):
        data = GENERIC_ITEM_PACK.to_dict()
        data['name'] = name
        paths[name] = tmp_path / f'{name}.json'
        paths[name].write_text(json.dumps(data))
    cfg = CurationConfig(prompt_pack_path=paths['pallet_v1'], prompt_pack_paths=(paths['food_v2'],))
    monkeypatch.setattr('src.config.curation.get_curation_config', lambda: cfg)
    return cfg


@pytest.fixture
def job_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from src.services.curation.autolabel import job

    state_dir = tmp_path / 'auto_label'
    monkeypatch.setattr(job, '_STATE_DIR', state_dir)
    for attr, name in (
        ('_STATE_FILE', 'state.json'),
        ('_CANCEL_FLAG', 'cancel.flag'),
        ('_RUNNING_LOCK', 'running.lock'),
        ('_EXIT_CODE_FILE', 'exit_code'),
        ('_TRIGGER_FILE', 'trigger.json'),
        ('_HEARTBEAT_FILE', 'heartbeat'),
    ):
        monkeypatch.setattr(job, attr, state_dir / name)
    return state_dir


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    fake_os = AsyncMock()
    # No settings doc -> every axis resolves to its process default.
    fake_os.get = AsyncMock(side_effect=RuntimeError('no settings index'))
    monkeypatch.setattr('src.routers.curation._ensure_indexes', AsyncMock(return_value=None))
    app = FastAPI()
    app.include_router(curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake_os
    return TestClient(app)


# =============================================================================
# /start: validation + args echo
# =============================================================================


@pytest.mark.usefixtures('packs', 'job_dir')
def test_start_unknown_prompt_pack_is_422_listing_valid_ids(client: TestClient) -> None:
    r = client.post('/curation/pipeline/auto_label/start', params={'prompt_pack': 'nope'})
    assert r.status_code == 422
    detail = r.json()['detail']
    assert detail['axis'] == 'prompt_pack'
    assert detail['requested'] == 'nope'
    assert detail['valid_ids'] == sorted({'pallet_v1', 'food_v2', GENERIC_ITEM_PACK.name})


@pytest.mark.usefixtures('packs', 'job_dir', 'reference_region_profile')
def test_start_unknown_detection_profile_is_422(client: TestClient) -> None:
    r = client.post('/curation/pipeline/auto_label/start', params={'detection_profile': 'nope'})
    assert r.status_code == 422
    detail = r.json()['detail']
    assert detail['axis'] == 'detection_profile'
    assert detail['valid_ids'] == ['license_plate']


@pytest.mark.usefixtures('packs', 'reference_region_profile')
def test_start_echoes_overrides_in_job_args(client: TestClient, job_dir: Path) -> None:
    r = client.post(
        '/curation/pipeline/auto_label/start',
        params={'prompt_pack': 'food_v2', 'detection_profile': 'license_plate'},
    )
    assert r.status_code == 200, r.text
    args = r.json()['args']
    assert args['prompt_pack'] == 'food_v2'
    assert args['detection_profile'] == 'license_plate'
    # The worker runs from the trigger file -- it must carry the same ids.
    trigger = json.loads((job_dir / 'trigger.json').read_text())
    assert trigger['args']['prompt_pack'] == 'food_v2'


@pytest.mark.usefixtures('packs', 'job_dir')
def test_start_omitted_resolves_to_settings_default(client: TestClient) -> None:
    r = client.post('/curation/pipeline/auto_label/start')
    assert r.status_code == 200, r.text
    args = r.json()['args']
    assert args['prompt_pack'] == 'pallet_v1'  # OP_PROMPT_PACK_PATH pack
    assert args['detection_profile'] is None  # neutral: no region profile


@pytest.mark.usefixtures('packs', 'job_dir')
@pytest.mark.asyncio
async def test_start_honors_settings_doc_default_when_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.routers.curation import pipeline
    from src.services.curation.autolabel import job

    captured: dict[str, Any] = {}

    def _fake_start(_fn: object, kwargs: dict[str, Any]) -> dict[str, Any]:
        captured.update(kwargs)
        return {'args': kwargs}

    monkeypatch.setattr(job, 'start_job', _fake_start)
    monkeypatch.setattr(
        'src.clients.curation_opensearch.get_curation_settings',
        AsyncMock(return_value={'defaults': {'prompt_pack': 'food_v2'}}),
    )
    await pipeline.pipeline_auto_label_start(opensearch=object())
    assert captured['prompt_pack'] == 'food_v2'


# =============================================================================
# The job itself: the selected pack reaches the labeler
# =============================================================================


def _run_kwargs(**overrides: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        'train_clusters': False,
        'promote_min_purity': 0.85,
        'promote_min_members': 4,
        'gemma_batch_size': 32,
        'gemma_concurrency': 8,
        'max_gemma_crops': 0,
        'v6_confidence_skip_gemma': 0.80,
        'clustering_method': None,
        'run_gemma': True,
        'recluster_unvalidated': False,
        'reassign_only': False,
        'run_auto_promote': False,
        'gate_max_rank': None,
        'gate_min_blur_ratio': None,
        'n_clusters': None,
        'class_id': None,
    }
    kwargs.update(overrides)
    return kwargs


@pytest.fixture
def labeler_spy(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[Any]]:
    """Record every labeler the pipeline obtains (real instances)."""
    from src.routers.curation import pipeline, vlm

    vlm._get_vlm_labeler.__dict__.pop('_insts', None)
    got: list[Any] = []

    def _spy(pack_name: str | None = None) -> Any:
        inst = vlm._get_vlm_labeler(pack_name)
        got.append(inst)
        return inst

    monkeypatch.setattr(pipeline, '_get_vlm_labeler', _spy)
    monkeypatch.setattr(
        'src.routers.curation.get_class_registry',
        lambda: _FakeRegistry([_FakeClassEntry(3, 'wooden_pallet')]),
    )
    yield got
    vlm._get_vlm_labeler.__dict__.pop('_insts', None)


@pytest.mark.usefixtures('packs')
@pytest.mark.asyncio
async def test_job_uses_the_overridden_pack(labeler_spy: list[Any]) -> None:
    from src.routers.curation import pipeline

    summary = await pipeline.pipeline_auto_label(
        opensearch=_FakeOpenSearch({3: ['pallet-1']}), **_run_kwargs(prompt_pack='food_v2')
    )
    assert summary['prompt_pack'] == 'food_v2'
    assert [inst._pack.name for inst in labeler_spy] == ['food_v2']


@pytest.mark.usefixtures('packs')
@pytest.mark.asyncio
async def test_job_omitted_pack_uses_default(labeler_spy: list[Any]) -> None:
    from src.routers.curation import pipeline

    summary = await pipeline.pipeline_auto_label(
        opensearch=_FakeOpenSearch({3: ['pallet-1']}), **_run_kwargs()
    )
    assert summary['prompt_pack'] == 'pallet_v1'
    assert summary['detection_profile'] is None
    assert [inst._pack.name for inst in labeler_spy] == ['pallet_v1']


@pytest.mark.usefixtures('packs')
@pytest.mark.asyncio
async def test_job_rejects_unknown_pack(labeler_spy: list[Any]) -> None:
    from fastapi import HTTPException

    from src.routers.curation import pipeline

    with pytest.raises(HTTPException) as info:
        await pipeline.pipeline_auto_label(
            opensearch=_FakeOpenSearch({}), **_run_kwargs(prompt_pack='nope')
        )
    assert info.value.status_code == 422
    assert labeler_spy == []
