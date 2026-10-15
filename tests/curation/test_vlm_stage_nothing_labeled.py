"""A VLM stage that could label nothing says so (found running the selective recipe).

An unreachable VLM, or crops whose image cannot be read, used to leave the
auto-label job ``completed`` with ``predicted: 0``: indistinguishable from a
run that had nothing to do. Now the stage reports ``failed`` and ``unreadable``
counts, and a run in which every selected crop failed fails the job.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from curation.query_fakes import QueryFakeOpenSearch
from curation.test_vlm_empty_class_answer import ITEMS, _item
from src.services.labeling.vlm_client import VlmIdentity
from src.services.labeling.vlm_models import VlmClassPrediction
from src.services.labeling.vlm_prompts import GENERIC_ITEM_PACK


pytestmark = pytest.mark.usefixtures('vlm_env_active')


async def _run(
    monkeypatch: pytest.MonkeyPatch,
    answers: dict[str, VlmClassPrediction],
    *,
    unreadable: frozenset[str] = frozenset(),
) -> Any:
    from curation.test_pipeline import _FakeClassEntry, _FakeRegistry
    from src.routers.curation.pipeline_public import pipeline_auto_label
    from src.services.curation import auto_label_run, image_serving
    from src.services.curation.autolabel import selection, vlm_stage

    fake = QueryFakeOpenSearch({ITEMS: {cid: _item(cid) for cid in answers}})

    class _Labeler:
        identity = VlmIdentity('env@None', 'test-vlm')
        model = 'fake-vlm'
        _pack = GENERIC_ITEM_PACK

        async def label_or_propose_batch(self, crops: list[Any], *_a: Any, **_k: Any) -> list[Any]:
            return [answers[c.img_id] for c in crops]

    def thumb(path: str, *_a: Any, **_k: Any) -> bytes:
        if any(cid in path for cid in unreadable):
            raise FileNotFoundError(path)
        return b'jpeg'

    monkeypatch.setattr(selection, 'classifier_class_sources', lambda: frozenset({'det_model'}))
    monkeypatch.setattr(vlm_stage, 'get_vlm_labeler', lambda *_a, **_k: _Labeler())
    monkeypatch.setattr(
        auto_label_run, 'resolve_run_prompt_pack', AsyncMock(return_value=(None, None))
    )
    monkeypatch.setattr(auto_label_run, 'pipeline_health_snapshot', AsyncMock(return_value={}))
    monkeypatch.setattr(
        'src.clients.curation_opensearch.registry.get_class_registry',
        lambda: _FakeRegistry([_FakeClassEntry(7, 'sportscar')]),
    )
    monkeypatch.setattr(
        'src.services.labeling.vlm_class_names.format_class_catalog', lambda *_a, **_k: ''
    )
    monkeypatch.setattr(image_serving, 'resolve_crop_root', lambda _p: '/data')
    monkeypatch.setattr(image_serving, 'resolve_safe_path', lambda p, _r: p)
    monkeypatch.setattr(image_serving, 'THUMBNAIL_CACHE', SimpleNamespace(get_or_compute=thumb))
    return await pipeline_auto_label(opensearch=fake, train_clusters=False, run_vlm=True)


def _failed(cid: str) -> VlmClassPrediction:
    return VlmClassPrediction(img_id=cid, class_name='', confidence='low', failure='request_failed')


@pytest.mark.asyncio
async def test_every_request_failing_fails_the_run(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(RuntimeError, match='labeled nothing'):
        await _run(monkeypatch, {'a': _failed('a'), 'b': _failed('b')})


@pytest.mark.asyncio
async def test_every_image_unreadable_fails_the_run(monkeypatch: pytest.MonkeyPatch) -> None:
    ok = VlmClassPrediction(img_id='a', class_name='sportscar', confidence='high')
    with pytest.raises(RuntimeError, match='labeled nothing'):
        await _run(monkeypatch, {'a': ok, 'b': ok}, unreadable=frozenset({'a', 'b'}))


@pytest.mark.asyncio
async def test_a_partial_failure_is_counted_not_fatal(monkeypatch: pytest.MonkeyPatch) -> None:
    ok = VlmClassPrediction(img_id='a', class_name='sportscar', confidence='high')
    result = await _run(
        monkeypatch, {'a': ok, 'b': _failed('b'), 'c': ok}, unreadable=frozenset({'c'})
    )
    stage = result['stages']['vlm']
    assert (stage['predicted'], stage['failed'], stage['unreadable']) == (2, 1, 1)
