"""Found live: every ``GET .../{prompt_packs,region_profiles}/active`` answered
500 once a detection worker had written its runtime doc -- the worker writes
flat ``profile`` / ``profile_revision`` fields, the view read ``profile`` as a
mapping (``ActiveRef(**'wheel_profile')``)."""

from __future__ import annotations

from typing import Any

import pytest

from src.routers.curation._config_common_models import AppliedRuntime
from src.services.config_store.index import upsert_runtime_doc


class _Client:
    def __init__(self) -> None:
        self.bodies: list[dict[str, Any]] = []

    async def index(self, **kwargs: Any) -> None:
        self.bodies.append(kwargs['body'])


@pytest.mark.asyncio
async def test_a_worker_written_runtime_doc_reads_back() -> None:
    client = _Client()
    await upsert_runtime_doc(
        client,
        'idx',
        process='detection_worker',
        hostname='host-1',
        fields={
            'project': 'alpha',
            'applied_config_revision': 3,
            'profile': 'wheel_profile',
            'profile_revision': 2,
            'pack': 'generic_item_v1',
            'pack_revision': None,
            'vlm': 'env',
            'vlm_revision': None,
        },
    )

    runtime = AppliedRuntime.from_runtime_doc(client.bodies[0], config_revision=5)

    assert runtime.host == 'host-1'
    assert runtime.applied_at
    assert (runtime.profile.name, runtime.profile.revision) == ('wheel_profile', 2)
    assert (runtime.pack.name, runtime.pack.revision) == ('generic_item_v1', None)
    assert runtime.vlm is not None
    assert runtime.vlm.name == 'env'
    assert runtime.lagging is True


def test_a_current_worker_is_not_lagging() -> None:
    runtime = AppliedRuntime.from_runtime_doc(
        {'process': 'detection_worker', 'applied_config_revision': 5}, config_revision=5
    )

    assert runtime.lagging is False
