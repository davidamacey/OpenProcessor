"""Tests for ``src.services.curation.probe_predictions.run_probe_inference``.

No real model or OpenSearch cluster — ``_build_predictor`` is monkeypatched
to a canned predict function (the heavy ultralytics/onnxruntime imports
never run in these tests) and OpenSearch is a small in-memory fake that
genuinely applies the ``must_not: test_holdout`` filter from the query body
it receives, so ``test_probe_skips_test_holdout_crops`` is a real guard on
the query construction, not just a trusted assertion.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from PIL import Image


if TYPE_CHECKING:
    from pathlib import Path


# =============================================================================
# Fake OpenSearch — scroll + update, with a real must_not:test_holdout filter
# =============================================================================


class _FakeOpenSearch:
    def __init__(self, docs: list[dict[str, Any]]) -> None:
        self._docs = docs
        self.updates: list[dict[str, Any]] = []
        self.search_bodies: list[dict[str, Any]] = []
        self._scroll_done = False

    def _matches(self, body: dict[str, Any]) -> list[dict[str, Any]]:
        must_not = (body.get('query') or {}).get('bool', {}).get('must_not', [])
        excludes_holdout = any(
            clause.get('term', {}).get('test_holdout') is True for clause in must_not
        )
        docs = self._docs
        if excludes_holdout:
            docs = [d for d in docs if not d.get('test_holdout')]
        return docs

    async def search(self, index: str, body: dict[str, Any], scroll: str | None = None) -> dict:  # noqa: ARG002
        self.search_bodies.append(body)
        docs = self._matches(body)
        hits = [{'_id': d['crop_id'], '_source': d} for d in docs]
        return {'_scroll_id': 'scroll-1', 'hits': {'hits': hits}}

    async def scroll(self, scroll_id: str, scroll: str) -> dict:  # noqa: ARG002
        # First page already returned everything in these small tests —
        # the second scroll call always signals "no more hits" to end the loop.
        return {'_scroll_id': scroll_id, 'hits': {'hits': []}}

    async def clear_scroll(self, scroll_id: str) -> None:  # noqa: ARG002
        return None

    async def update(self, index: str, id: str, body: dict[str, Any]) -> None:  # noqa: A002
        self.updates.append({'index': index, 'id': id, 'doc': body['doc']})

    async def bulk(self, *, body: list[dict[str, Any]], refresh: bool | str = False) -> dict:  # noqa: ARG002
        self.bulk_calls = getattr(self, 'bulk_calls', 0) + 1
        for action, doc in zip(body[0::2], body[1::2], strict=True):
            meta = action['update']
            self.updates.append({'index': meta['_index'], 'id': meta['_id'], 'doc': doc['doc']})
        return {'errors': False, 'items': [{'update': {'status': 200}}] * (len(body) // 2)}


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def tiny_image(tmp_path: Path) -> Path:
    path = tmp_path / 'frame.jpg'
    Image.new('RGB', (64, 64), (10, 20, 30)).save(path)
    return path


def _install_canned_predictor(
    monkeypatch: pytest.MonkeyPatch, prediction: tuple[str | None, float, float, float]
) -> None:
    """Monkeypatch ``_build_predictor`` so ``run_probe_inference`` never
    touches ultralytics/onnxruntime or a real checkpoint file."""
    from src.services.curation import probe_predictions as pp

    def fake_build_predictor(model_path: Path, architecture: str):
        return (lambda crop: prediction), 'canned-v1'  # noqa: ARG005

    monkeypatch.setattr(pp, '_build_predictor', fake_build_predictor)


# =============================================================================
# test_probe_writes_all_seven_fields
# =============================================================================


@pytest.mark.asyncio
async def test_probe_writes_all_seven_fields(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, tiny_image: Path
) -> None:
    from src.services.curation import probe_predictions as pp

    _install_canned_predictor(monkeypatch, ('sedan', 0.81, 1.23, 0.44))
    monkeypatch.setattr(
        pp,
        '_resolve_image',
        lambda image_path, *, config: tiny_image,  # noqa: ARG005
    )

    docs = [
        {
            'crop_id': 'crop-1',
            'image_path': 'whatever.jpg',
            'bbox_norm': [0.0, 0.0, 1.0, 1.0],
            'class_name': 'suv',  # stored label differs from the canned prediction -> disagreement
        }
    ]
    fake_os = _FakeOpenSearch(docs)

    processed = await pp.run_probe_inference(
        tmp_path / 'fake_checkpoint.onnx',
        fake_os,  # type: ignore[arg-type]
        model_version=None,
        architecture='v6',
    )

    assert processed == 1
    assert len(fake_os.updates) == 1
    doc = fake_os.updates[0]['doc']
    assert set(doc) == {
        'probe_pred_class',
        'probe_pred_class_id',
        'probe_pred_confidence',
        'probe_pred_entropy',
        'probe_pred_margin',
        'probe_disagreement',
        'probe_model_version',
        'probe_scored_at',
    }
    assert doc['probe_pred_class'] == 'sedan'
    assert doc['probe_pred_confidence'] == pytest.approx(0.81)
    assert doc['probe_pred_entropy'] == pytest.approx(1.23)
    assert doc['probe_pred_margin'] == pytest.approx(0.44)
    assert doc['probe_disagreement'] is True  # 'sedan' (predicted) != 'suv' (stored)
    assert doc['probe_model_version'] == 'canned-v1'
    assert isinstance(doc['probe_scored_at'], str)
    assert doc['probe_scored_at']


@pytest.mark.asyncio
async def test_probe_writes_one_bulk_call_per_page_not_per_item(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, tiny_image: Path
) -> None:
    """F-26: probe scoring must issue one bulk() per scroll page, not one
    update() per crop -- these fakes' single search() call returns every
    doc as one page, so N crops should still cost exactly 1 bulk call."""
    from src.services.curation import probe_predictions as pp

    _install_canned_predictor(monkeypatch, ('sedan', 0.81, 1.23, 0.44))
    monkeypatch.setattr(
        pp,
        '_resolve_image',
        lambda image_path, *, config: tiny_image,  # noqa: ARG005
    )

    docs = [
        {
            'crop_id': f'crop-{i}',
            'image_path': 'whatever.jpg',
            'bbox_norm': [0.0, 0.0, 1.0, 1.0],
            'class_name': 'suv',
        }
        for i in range(5)
    ]
    fake_os = _FakeOpenSearch(docs)

    processed = await pp.run_probe_inference(
        tmp_path / 'fake_checkpoint.onnx',
        fake_os,  # type: ignore[arg-type]
        model_version=None,
        architecture='v6',
    )

    assert processed == 5
    assert len(fake_os.updates) == 5
    assert fake_os.bulk_calls == 1
    # sort: ['_doc'] on the scroll body (hygiene, no relevance needed).
    assert fake_os.search_bodies[0]['sort'] == ['_doc']


@pytest.mark.asyncio
async def test_probe_agreement_is_false_when_prediction_matches_stored_label(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, tiny_image: Path
) -> None:
    from src.services.curation import probe_predictions as pp

    _install_canned_predictor(monkeypatch, ('suv', 0.9, 0.1, 0.7))
    monkeypatch.setattr(
        pp,
        '_resolve_image',
        lambda image_path, *, config: tiny_image,  # noqa: ARG005
    )
    docs = [
        {
            'crop_id': 'crop-1',
            'image_path': 'whatever.jpg',
            'bbox_norm': [0.0, 0.0, 1.0, 1.0],
            'class_name': 'suv',
        }
    ]
    fake_os = _FakeOpenSearch(docs)
    await pp.run_probe_inference(tmp_path / 'fake.onnx', fake_os, architecture='v6')  # type: ignore[arg-type]
    assert fake_os.updates[0]['doc']['probe_disagreement'] is False


# =============================================================================
# test_probe_skips_test_holdout_crops
# =============================================================================


@pytest.mark.asyncio
async def test_probe_skips_test_holdout_crops(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, tiny_image: Path
) -> None:
    """The scroll query must exclude ``test_holdout=true`` crops — a real
    regression guard: the fake OpenSearch only applies the exclusion if the
    query body actually carries the ``must_not: {term: {test_holdout:
    true}}`` clause, so this fails if a future refactor drops it."""
    from src.services.curation import probe_predictions as pp

    _install_canned_predictor(monkeypatch, ('sedan', 0.5, 0.5, 0.2))
    monkeypatch.setattr(
        pp,
        '_resolve_image',
        lambda image_path, *, config: tiny_image,  # noqa: ARG005
    )
    docs = [
        {
            'crop_id': 'holdout-crop',
            'image_path': 'a.jpg',
            'bbox_norm': [0.0, 0.0, 1.0, 1.0],
            'class_name': 'suv',
            'test_holdout': True,
        },
        {
            'crop_id': 'normal-crop',
            'image_path': 'b.jpg',
            'bbox_norm': [0.0, 0.0, 1.0, 1.0],
            'class_name': 'suv',
            'test_holdout': False,
        },
    ]
    fake_os = _FakeOpenSearch(docs)

    processed = await pp.run_probe_inference(tmp_path / 'fake.onnx', fake_os, architecture='v6')  # type: ignore[arg-type]

    assert processed == 1
    updated_ids = {u['id'] for u in fake_os.updates}
    assert updated_ids == {'normal-crop'}

    # Belt-and-suspenders: the query itself must have asked OpenSearch to
    # exclude test_holdout, not just have gotten lucky with the fake's data.
    body = fake_os.search_bodies[0]
    must_not = body['query']['bool']['must_not']
    assert {'term': {'test_holdout': True}} in must_not


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


@pytest.mark.asyncio
async def test_probe_writes_the_registry_id_of_its_prediction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, tiny_image: Path
) -> None:
    from src.services.curation import probe_predictions as pp

    _install_canned_predictor(monkeypatch, ('sedan', 0.81, 1.23, 0.44))
    monkeypatch.setattr(pp, '_resolve_image', lambda image_path, *, config: tiny_image)  # noqa: ARG005
    docs = [{'crop_id': 'c', 'image_path': 'x.jpg', 'bbox_norm': [0, 0, 1, 1], 'class_name': 'suv'}]
    fake_os = _FakeOpenSearch(docs)
    await pp.run_probe_inference(
        tmp_path / 'f.onnx',
        fake_os,  # type: ignore[arg-type]
        architecture='v6',
        class_ids={'sedan': 12},
    )
    assert fake_os.updates[0]['doc']['probe_pred_class_id'] == 12

    fake_os = _FakeOpenSearch(docs)
    await pp.run_probe_inference(
        tmp_path / 'f.onnx',
        fake_os,  # type: ignore[arg-type]
        architecture='v6',
        class_ids={},
    )
    assert fake_os.updates[0]['doc']['probe_pred_class_id'] is None
