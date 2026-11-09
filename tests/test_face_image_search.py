"""Query-by-image face search and identification service."""

from types import SimpleNamespace

import pytest

from src.core.exceptions import RequestRejectedError
from src.services.face_identity import image_search


def _face(x: float) -> dict:
    return {'box': [x, 0, 1, 1], 'landmarks': [0.0] * 10, 'score': 0.9, 'quality': None}


def _detected(n: int) -> dict:
    return {'num_faces': n, 'faces': [_face(i / 10) for i in range(n)], 'embeddings': [[1.0]] * n}


class _Search:
    async def search_faces(self, **_kw):
        return [{'_id': 'f1', 'score': 0.8, 'box': [0, 0, 1, 1]}]


@pytest.fixture
def service():
    return SimpleNamespace(opensearch=_Search())


def _patch(monkeypatch, n: int) -> None:
    fake = SimpleNamespace(detect_faces=lambda _b, **_kw: _detected(n))
    monkeypatch.setattr(image_search, 'InferenceService', lambda: fake)


@pytest.mark.asyncio
async def test_search_maps_matches_and_defaults(monkeypatch, service) -> None:
    _patch(monkeypatch, 2)
    face, matches = await image_search.search_by_image(
        service, b'x', face_index=1, top_k=5, min_score=0.5, confidence=0.5
    )
    assert face['box'][0] == 0.1
    assert matches[0]['face_id'] == 'f1'
    assert matches[0]['confidence'] == 0.0
    assert matches[0]['image_path'] is None


@pytest.mark.asyncio
@pytest.mark.parametrize(('n', 'index'), [(0, 0), (2, 2)])
async def test_search_rejects_missing_or_out_of_range_face(monkeypatch, service, n, index) -> None:
    _patch(monkeypatch, n)
    with pytest.raises(RequestRejectedError) as err:
        await image_search.search_by_image(
            service, b'x', face_index=index, top_k=5, min_score=0.5, confidence=0.5
        )
    assert err.value.status_code == 400


@pytest.mark.asyncio
async def test_identify_returns_one_entry_per_face(monkeypatch, service) -> None:
    _patch(monkeypatch, 3)
    out = await image_search.identify_by_image(
        service, b'x', top_k=5, threshold=0.6, confidence=0.5
    )
    assert len(out) == 3
    assert out[0][1][0]['face_id'] == 'f1'
