"""Two external-images acknowledgement gates that only an ``allow_external``
endpoint reaches (W9 review m1): the per-run ack and the clone refusal.
Every other walk of the gate uses a DNS-denied endpoint, which a reverted
ack rule would still refuse."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from curation.conftest import SCOPED
from src.config.curation import IndexRole, base_curation_config
from src.config.project_context import bind_project
from src.config.projects import ProjectRecord, resources_for_new


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


def test_a_per_run_use_of_an_external_endpoint_needs_the_ack_even_when_allow_external(
    vlm_api, reference_region_profile
) -> None:
    vlm_api.dns['api.vendor.com'] = ['93.184.216.34']
    vlm_api.ready('ext', base_url='https://api.vendor.com/v1', allow_external=True)
    url = f'{SCOPED}/vlm/label_batch'
    refused = vlm_api.lenient.post(url, params={'vlm': 'ext'}, json={'crop_ids': ['c1']})
    assert refused.status_code == 422
    assert 'vlm_external_not_acknowledged' in refused.text
    acked = vlm_api.lenient.post(
        url, params={'vlm': 'ext', 'acknowledge_external': 'true'}, json={'crop_ids': ['c1']}
    )
    assert 'vlm_external_not_acknowledged' not in acked.text


def test_a_clone_refuses_an_acknowledged_external_endpoint(
    vlm_api, monkeypatch: pytest.MonkeyPatch, reference_region_profile
) -> None:
    """The ack is the SOURCE project's own decision; the target never made it."""
    from src.services.projects import lifecycle as lifecycle_mod

    vlm_api.dns['api.vendor.com'] = ['93.184.216.34']
    vlm_api.ready('ext', base_url='https://api.vendor.com/v1', allow_external=True)
    source, target = _record('alpha'), _record('beta')
    now = '2026-01-01T00:00:00+00:00'
    vlm_api.fake_os._docs.setdefault(source.resources.indexes[IndexRole.CONFIGS], {})[
        'activation:vlm'
    ] = {
        '_source': {
            'doc_type': 'activation',
            'axis': 'vlm',
            'name': 'ext',
            'revision': 1,
            'activated_at': now,
            'previous': None,
            'acked_refs': {'ext@1': now},
            'external_ack_at': now,
        },
        '_seq_no': 1,
    }
    monkeypatch.setattr(lifecycle_mod, '_resolve_existing', AsyncMock(return_value=source))

    async def clone() -> None:
        with bind_project(target):
            await lifecycle_mod.clone_settings(
                vlm_api.fake_os, target_record=target, from_slug='alpha', axes=['vlm_activation']
            )

    with pytest.raises(HTTPException) as caught:
        asyncio.run(clone())
    assert caught.value.status_code == 422
    assert 'vlm_external_not_acknowledged' in str(caught.value.detail)
