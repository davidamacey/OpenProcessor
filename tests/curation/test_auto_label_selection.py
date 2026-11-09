"""Per-run ``prompt_pack`` override on the auto-label endpoints
(``POST /pipeline/auto_label/start`` and ``POST /pipeline/auto_label``).

An explicit id overrides the settings-doc default for that one job (never
written to settings); an unknown id is a 422 listing the valid ids; omitted
keeps the settings default. The resolved id is echoed in the job ``args``
and actually reaches the VLM labeler the job uses.

``detection_profile`` is deliberately NOT a per-run option (no auto-label
stage runs region detection); a client still sending it gets a 422, not a
silent no-op.
"""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from curation.test_pipeline import _FakeOpenSearch
from src.services.curation.item_filter import ItemFilter
from src.services.labeling.vlm_prompts import GENERIC_ITEM_PACK, GENERIC_REGION_PACK


pytestmark = pytest.mark.usefixtures('vlm_env_active')


# =============================================================================
# /start: validation + args echo
# =============================================================================


@pytest.mark.usefixtures('packs', 'job_dir')
def test_start_unknown_prompt_pack_is_422_listing_valid_ids(client: TestClient) -> None:
    r = client.post(
        '/curation/projects/default/pipeline/auto_label/start', params={'prompt_pack': 'nope'}
    )
    assert r.status_code == 422
    detail = r.json()['detail']
    assert detail['axis'] == 'prompt_pack'
    assert detail['requested'] == 'nope'
    assert detail['valid_ids'] == sorted(
        {'pallet_v1', 'food_v2', GENERIC_ITEM_PACK.name, GENERIC_REGION_PACK.name}
    )


@pytest.mark.usefixtures('packs', 'job_dir', 'reference_region_profile')
@pytest.mark.parametrize('value', ['license_plate', 'nope'])
def test_start_rejects_any_detection_profile(client: TestClient, value: str) -> None:
    """Even the active profile's id: accepting it would echo an override
    that changes nothing."""
    r = client.post(
        '/curation/projects/default/pipeline/auto_label/start', params={'detection_profile': value}
    )
    assert r.status_code == 422
    detail = r.json()['detail']
    assert detail['param'] == 'detection_profile'
    # The error points at the region-profile CRUD activation route
    # (region_profiles.py), the real config surface for detection, not the
    # OP_REGION_PROFILE env var.
    assert 'POST /region_profiles/{name}/activate' in detail['error']


@pytest.mark.usefixtures('packs')
def test_start_echoes_prompt_pack_override_in_job_args(
    client: TestClient, job_dir: list[dict[str, Any]]
) -> None:
    r = client.post(
        '/curation/projects/default/pipeline/auto_label/start', params={'prompt_pack': 'food_v2'}
    )
    assert r.status_code == 202, r.text
    args = r.json()['args']
    assert args['prompt_pack'] == 'food_v2'
    assert 'detection_profile' not in args
    # The worker runs from the job's params -- they must carry the same id.
    assert job_dir[0]['prompt_pack'] == 'food_v2'


@pytest.mark.usefixtures('packs', 'job_dir')
def test_start_omitted_resolves_to_settings_default(client: TestClient) -> None:
    r = client.post('/curation/projects/default/pipeline/auto_label/start')
    assert r.status_code == 202, r.text
    assert r.json()['args']['prompt_pack'] == 'pallet_v1'  # OP_PROMPT_PACK_PATH pack


@pytest.mark.usefixtures('packs', 'job_dir')
@pytest.mark.asyncio
async def test_start_ignores_a_stale_settings_doc_pack_when_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.routers.curation import pipeline_start
    from src.services.curation.autolabel import service

    captured: dict[str, Any] = {}

    async def _fake_start(_session: object, kwargs: dict[str, Any]) -> dict[str, Any]:
        captured.update(kwargs)
        return {'args': kwargs}

    monkeypatch.setattr(service, 'start', _fake_start)
    monkeypatch.setattr(
        'src.services.curation.settings_store.get_curation_settings',
        AsyncMock(return_value={'defaults': {'prompt_pack': 'food_v2'}}),
    )
    await pipeline_start.pipeline_auto_label_start(
        session=cast('Any', object()), opensearch=object(), item_filter=ItemFilter()
    )
    assert captured['prompt_pack'] == 'pallet_v1'


# =============================================================================
# The job itself: the selected pack reaches the labeler
# =============================================================================


def _run_kwargs(**overrides: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        'vlm_batch_size': 32,
        'vlm_concurrency': 8,
        'max_vlm_crops': 0,
        'classifier_confidence_skip_vlm': 0.80,
        'run_vlm': True,
        'class_id': None,
    }
    kwargs.update(overrides)
    return kwargs


@pytest.mark.usefixtures('packs')
@pytest.mark.asyncio
async def test_job_uses_the_overridden_pack(labeler_spy: list[Any]) -> None:
    from src.routers.curation.pipeline_public import pipeline_auto_label

    summary = await pipeline_auto_label(
        opensearch=_FakeOpenSearch({3: ['pallet-1']}),
        **_run_kwargs(prompt_pack='food_v2'),
    )
    assert summary['prompt_pack'] == 'food_v2'
    assert [inst._pack.name for inst in labeler_spy] == ['food_v2']


@pytest.mark.usefixtures('packs')
@pytest.mark.asyncio
async def test_job_omitted_pack_uses_default(labeler_spy: list[Any]) -> None:
    from src.routers.curation.pipeline_public import pipeline_auto_label

    summary = await pipeline_auto_label(
        opensearch=_FakeOpenSearch({3: ['pallet-1']}), **_run_kwargs()
    )
    assert summary['prompt_pack'] == 'pallet_v1'
    assert 'detection_profile' not in summary
    assert [inst._pack.name for inst in labeler_spy] == ['pallet_v1']


@pytest.mark.usefixtures('packs')
@pytest.mark.asyncio
async def test_job_rejects_unknown_pack(labeler_spy: list[Any]) -> None:
    from src.core.exceptions import RequestRejectedError
    from src.routers.curation.pipeline_public import pipeline_auto_label

    with pytest.raises(RequestRejectedError) as info:
        await pipeline_auto_label(opensearch=_FakeOpenSearch({}), **_run_kwargs(prompt_pack='nope'))
    assert info.value.status_code == 422
    assert labeler_spy == []


@pytest.mark.usefixtures('packs')
@pytest.mark.asyncio
async def test_job_rejects_detection_profile(labeler_spy: list[Any]) -> None:
    from src.core.exceptions import RequestRejectedError
    from src.routers.curation.pipeline_public import pipeline_auto_label

    with pytest.raises(RequestRejectedError) as info:
        await pipeline_auto_label(
            opensearch=_FakeOpenSearch({}),
            **_run_kwargs(detection_profile='license_plate'),
        )
    assert info.value.status_code == 422
    assert labeler_spy == []
