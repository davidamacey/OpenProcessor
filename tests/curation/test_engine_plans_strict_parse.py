"""Every plan the API builds for the frame engine is accepted by the strict plan parser.

The wire module refuses a ``version`` that is not the integer 1, ``NaN`` and ``Infinity`` literals
and duplicate keys (``tests/test_frame_engine_wire.py``); the native parser refuses the same. The
API's two plan builders (the ingest plan and the reprocess embed plan) must never send one: the
plan text each puts on the wire is parsed here by the engine's own parser, and its raw text is
checked for the three quirks again with an independent reader.
"""

from __future__ import annotations

import dataclasses
import json
from typing import TYPE_CHECKING, Any

import pytest

from curation.engine_fakes import engine_service
from curation.ingest_fakes import _jpeg_bytes
from src.domain import frame_engine_wire as wire
from src.services.curation import ingest_gpu_decode
from src.services.curation.ingest_policy import DetectFilter, EmbeddingPolicy, IngestPolicy
from src.services.curation.reprocess_gpu_embed import GpuEmbedRoute, embed_on_gpu


if TYPE_CHECKING:
    from pathlib import Path


def _refuse(name: str) -> Any:
    raise AssertionError(f'{name} literal in a plan the API built')


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    keys = [k for k, _ in pairs]
    assert len(keys) == len(set(keys)), f'duplicate key in {keys}'
    return dict(pairs)


def _assert_strict(text: str, plan: wire.EnginePlan, *, has_boxes: bool) -> None:
    raw = json.loads(text, parse_constant=_refuse, object_pairs_hook=_no_duplicates)
    assert raw['version'] == 1
    assert type(raw['version']) is int
    assert wire.parse_plan(text, has_boxes=has_boxes) == plan


@pytest.fixture
def labels(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / 'labels.txt'
    path.write_text('gadget\nwidget\ngizmo\n')
    monkeypatch.setenv('OP_TRITON_MODEL_REPO', str(tmp_path))
    return path


@pytest.mark.parametrize(
    'policy',
    [
        IngestPolicy(),
        IngestPolicy(embedding=EmbeddingPolicy(mode='lazy')),
        IngestPolicy(embedding=EmbeddingPolicy(mode='none')),
        IngestPolicy(embedding=EmbeddingPolicy(mode='selected', classes=['Widget'])),
        IngestPolicy(detect=DetectFilter(exclude_classes=['gadget'])),
    ],
)
def test_the_ingest_plan_passes_the_strict_parser(
    tmp_path: Path, labels: Path, policy: IngestPolicy
) -> None:
    svc, *_ = engine_service(tmp_path)
    svc.profile = dataclasses.replace(svc.profile, labels_path=str(labels))
    svc.policy = policy
    plan = ingest_gpu_decode.ingest_engine_plan(svc)
    _assert_strict(plan.to_json(), plan, has_boxes=False)


@pytest.mark.asyncio
async def test_the_reprocess_embed_plan_passes_the_strict_parser(tmp_path: Path) -> None:
    *_, engine, _sizes = engine_service(tmp_path)
    engine.read_sizes = True
    data = _jpeg_bytes((64, 48), seed=3)
    out = await embed_on_gpu(GpuEmbedRoute(engine), data, [(0.1, 0.1, 0.5, 0.6)])
    assert out is not None
    [call] = engine.calls  # the fake engine parsed the text the client sent, with parse_plan
    plan = call['plan']
    assert plan.branches == ('embed_frame', 'embed_crops')
    _assert_strict(plan.to_json(), plan, has_boxes=True)
