"""VLM-labeling one cluster is a server-side job; direct calls get real defaults.

- ``POST /vlm/label_cluster/{cluster_id}`` queues the auto-label job with a
  cluster scope: VLM only, no re-clustering, no cap. The server selects
  every unvalidated, non-holdout, non-excluded member (no silent 200-item
  truncation) and reports how many it selected.
- ``pipeline_auto_label`` is also called straight from Python by the
  auto-label worker with only the args in its trigger file. Every omitted
  parameter must be its plain default, never a FastAPI ``Query`` object
  (a leaked ``FieldInfo`` for ``class_id`` became a bogus term filter).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

import pytest

from curation.query_fakes import QueryFakeOpenSearch
from curation.test_auto_label_selection import client, job_dir, packs  # noqa: F401 - fixtures
from curation.test_pipeline import _FakeClassEntry, _FakeRegistry
from src.config import get_curation_config


if TYPE_CHECKING:
    from pathlib import Path

    from fastapi.testclient import TestClient


ITEMS = get_curation_config().items_index
CLUSTER = 10005


def _docs() -> dict[str, dict[str, Any]]:
    base = {'updated_at': '2026-09-01T00:00:00+00:00', 'bbox_norm': [0, 0, 1, 1]}
    return {
        'a': {**base, 'crop_id': 'a', 'cluster_id': CLUSTER},
        'b': {**base, 'crop_id': 'b', 'cluster_id': CLUSTER, 'class_source': 'vlm_unmatched'},
        # Classifier-confident: skipped by the global sweep, but an explicit
        # cluster request labels it.
        'c': {
            **base,
            'crop_id': 'c',
            'cluster_id': CLUSTER,
            'class_source': 'det_model',
            'confidence': 0.99,
        },
        'v': {**base, 'crop_id': 'v', 'cluster_id': CLUSTER, 'class_validated': True},
        'h': {**base, 'crop_id': 'h', 'cluster_id': CLUSTER, 'test_holdout': True},
        'x': {**base, 'crop_id': 'x', 'cluster_id': CLUSTER, 'class_excluded': True},
        'other': {**base, 'crop_id': 'other', 'cluster_id': 3},
    }


class _Labeler:
    model = 'fake-vlm'
    _pack = None

    def __init__(self) -> None:
        self.sent: list[str] = []

    async def label_or_propose_batch(self, crops: list[Any], *_a: Any, **_k: Any) -> list[Any]:
        self.sent.extend(c.img_id for c in crops)
        return []


@pytest.fixture
def pipeline_env(monkeypatch: pytest.MonkeyPatch) -> _Labeler:
    from src.routers.curation import pipeline, pipeline_health
    from src.services.curation.autolabel import selection

    monkeypatch.setattr(selection, 'classifier_class_sources', lambda: frozenset({'det_model'}))

    labeler = _Labeler()
    monkeypatch.setattr(pipeline, '_get_vlm_labeler', lambda _pack=None: labeler)
    monkeypatch.setattr(pipeline, 'resolve_run_prompt_pack', AsyncMock(return_value=None))
    monkeypatch.setattr(pipeline_health, 'pipeline_health_snapshot', AsyncMock(return_value={}))
    monkeypatch.setattr(
        'src.routers.curation.get_class_registry',
        lambda: _FakeRegistry([_FakeClassEntry(3, 'thing')]),
    )
    monkeypatch.setattr(
        'src.services.labeling.vlm_labeler.format_class_catalog', lambda *_a, **_k: ''
    )
    return labeler


@pytest.mark.asyncio
async def test_direct_call_with_omitted_args_uses_plain_defaults(pipeline_env: _Labeler) -> None:
    from src.routers.curation import pipeline

    fake = QueryFakeOpenSearch({ITEMS: _docs()})
    summary = await pipeline.pipeline_auto_label(
        opensearch=fake, train_clusters=False, run_vlm=True
    )
    assert summary['class_id'] is None
    assert summary['cluster_id'] is None
    assert summary['prompt_pack'] is None
    # Global sweep: every unvalidated item minus the cost skips (b is
    # vlm_unmatched, c classifier-confident): a, h, x, other.
    assert summary['stages']['unvalidated_after_promote'] == 4


@pytest.mark.asyncio
async def test_cluster_scope_selects_every_unvalidated_member(pipeline_env: _Labeler) -> None:
    from src.routers.curation import pipeline

    fake = QueryFakeOpenSearch({ITEMS: _docs()})
    summary = await pipeline.pipeline_auto_label(
        opensearch=fake, train_clusters=False, run_vlm=True, cluster_id=CLUSTER
    )
    assert summary['cluster_id'] == CLUSTER
    # a, b (vlm_unmatched) and c (classifier-confident); never v/h/x/other.
    assert summary['stages']['unvalidated_after_promote'] == 3


@pytest.mark.usefixtures('packs')
def test_label_cluster_route_queues_a_scoped_vlm_job(client: TestClient, job_dir: Path) -> None:  # noqa: F811
    r = client.post(f'/curation/vlm/label_cluster/{CLUSTER}')
    assert r.status_code == 200, r.text
    args = json.loads((job_dir / 'trigger.json').read_text())['args']
    assert args['cluster_id'] == CLUSTER
    assert args['run_vlm'] is True
    assert args['train_clusters'] is False
    assert args['run_auto_promote'] is False
    assert args['max_vlm_crops'] == 0
