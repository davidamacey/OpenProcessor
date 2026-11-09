"""The auto-label job scoped to one cluster; direct calls get real defaults.

- ``cluster_uid`` scopes the auto-label job to one cluster: the server selects
  every unvalidated, non-holdout, non-excluded member (no silent 200-item
  truncation) and reports how many it selected.
- ``pipeline_auto_label`` is also called straight from Python by the
  auto-label worker with only the args in its trigger file. Every omitted
  parameter must be its plain default, never a FastAPI ``Query`` object
  (a leaked ``FieldInfo`` for ``class_id`` became a bogus term filter).
"""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import AsyncMock

import pytest

from curation.query_fakes import QueryFakeOpenSearch
from curation.test_pipeline import install_registry
from src.config.curation_accessors import base_curation_config
from src.services.labeling.vlm_client import VlmIdentity
from src.services.labeling.vlm_prompts import GENERIC_ITEM_PACK


ITEMS = base_curation_config().items_index
CLUSTER = uuid.uuid5(uuid.NAMESPACE_URL, 'cluster-scope')
OTHER_CLUSTER = uuid.uuid5(uuid.NAMESPACE_URL, 'other-cluster')


pytestmark = pytest.mark.usefixtures('vlm_env_active', 'readable_crops')


def _docs() -> dict[str, dict[str, Any]]:
    base = {
        'updated_at': '2026-09-01T00:00:00+00:00',
        'bbox_norm': [0, 0, 1, 1],
        'image_path': '/data/img.jpg',
        'pe_embedding': [0.1],
    }
    return {
        'a': {**base, 'crop_id': 'a', 'cluster_uid': str(CLUSTER)},
        'b': {**base, 'crop_id': 'b', 'cluster_uid': str(CLUSTER), 'class_source': 'vlm_unmatched'},
        # Classifier-confident: skipped by the global sweep, but an explicit
        # cluster request labels it.
        'c': {
            **base,
            'crop_id': 'c',
            'cluster_uid': str(CLUSTER),
            'class_source': 'det_model',
            'confidence': 0.99,
        },
        'v': {**base, 'crop_id': 'v', 'cluster_uid': str(CLUSTER), 'class_validated': True},
        'h': {**base, 'crop_id': 'h', 'cluster_uid': str(CLUSTER), 'test_holdout': True},
        'x': {**base, 'crop_id': 'x', 'cluster_uid': str(CLUSTER), 'class_excluded': True},
        'other': {**base, 'crop_id': 'other', 'cluster_uid': str(OTHER_CLUSTER)},
    }


class _Labeler:
    identity = VlmIdentity('env@None', 'test-vlm')
    model = 'fake-vlm'
    _pack = GENERIC_ITEM_PACK

    def __init__(self) -> None:
        self.sent: list[str] = []

    async def label_or_propose_batch(self, crops: list[Any], *_a: Any, **_k: Any) -> list[Any]:
        self.sent.extend(c.img_id for c in crops)
        return []


@pytest.fixture
def pipeline_env(monkeypatch: pytest.MonkeyPatch) -> _Labeler:
    from src.services.curation import auto_label_run
    from src.services.curation.autolabel import selection, vlm_stage

    monkeypatch.setattr(selection, 'classifier_class_sources', lambda: frozenset({'det_model'}))

    labeler = _Labeler()
    monkeypatch.setattr(vlm_stage, 'get_vlm_labeler', lambda *_a, **_k: labeler)
    monkeypatch.setattr(
        auto_label_run, 'resolve_run_prompt_pack', AsyncMock(return_value=(None, None))
    )
    monkeypatch.setattr(auto_label_run, 'pipeline_health_snapshot', AsyncMock(return_value={}))
    install_registry((3, 'thing'))
    monkeypatch.setattr(
        'src.services.labeling.vlm_class_names.format_class_catalog', lambda *_a, **_k: ''
    )
    return labeler


@pytest.mark.asyncio
async def test_direct_call_with_omitted_args_uses_plain_defaults(pipeline_env: _Labeler) -> None:
    from src.routers.curation.pipeline_public import pipeline_auto_label

    fake = QueryFakeOpenSearch({ITEMS: _docs()})
    summary = await pipeline_auto_label(opensearch=fake, run_vlm=True)
    assert summary['class_id'] is None
    assert summary['cluster_uid'] is None
    assert summary['prompt_pack'] is None
    # Global sweep: every unvalidated item minus the cost skips (b is
    # vlm_unmatched, c classifier-confident) and minus the frozen holdout (h) and
    # excluded (x) items: a, other.
    assert summary['stages']['unvalidated_after_promote'] == 2


@pytest.mark.asyncio
async def test_cluster_scope_selects_every_unvalidated_member(pipeline_env: _Labeler) -> None:
    from src.routers.curation.pipeline_public import pipeline_auto_label

    fake = QueryFakeOpenSearch({ITEMS: _docs()})
    summary = await pipeline_auto_label(opensearch=fake, run_vlm=True, cluster_uid=CLUSTER)
    assert summary['cluster_uid'] == str(CLUSTER)
    # a, b (vlm_unmatched) and c (classifier-confident); never v/h/x/other.
    assert summary['stages']['unvalidated_after_promote'] == 3
