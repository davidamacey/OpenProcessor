"""Ingest under a selective embedding policy: stored always, embedded by choice."""

from __future__ import annotations

from typing import Any

import pytest

from curation.test_ingest_service import FakePEEncoder, _jpeg_bytes, _make_service
from src.services.curation.ingest_policy import DetectFilter, EmbeddingPolicy, IngestPolicy


# class 0 and 1 of a registry-aligned model: gadget / widget, plus an unlabeled
# low-confidence box (below the 0.5 floor, so it stays an unclassed proposal).
DETECTIONS = [
    (0.05, 0.05, 0.40, 0.40, 0.95, 0),
    (0.50, 0.05, 0.90, 0.40, 0.90, 1),
    (0.05, 0.50, 0.30, 0.90, 0.30, 0),
]


class _IvfStore:
    def assign_one_with_distance(self, _embedding: Any) -> tuple[int, float]:
        return 7, 0.1


def _service(policy: IngestPolicy) -> Any:
    from curation.test_ingest_service import FakeClassRegistry, _FakeClassEntry

    registry = FakeClassRegistry([_FakeClassEntry(0, 'gadget'), _FakeClassEntry(1, 'widget')])
    svc, os_fake, _ = _make_service(detections=DETECTIONS, registry=registry)
    svc.policy = policy
    return svc, os_fake


def _by_score(os_fake: Any) -> dict[float, dict[str, Any]]:
    return {round(d['confidence'], 2): d for d in os_fake.items.values()}


@pytest.mark.asyncio
async def test_selected_mode_embeds_only_chosen_classes_and_stores_the_rest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        'src.services.curation.ingest_index.get_ivf_ingest_store', lambda: _IvfStore()
    )
    svc, os_fake = _service(
        IngestPolicy(embedding=EmbeddingPolicy(mode='selected', classes=['widget']))
    )
    encoder: FakePEEncoder = svc.pe_encoder
    result = await svc.ingest_one(_jpeg_bytes(), '/tmp/a.jpg')

    assert result.n_crops == 3
    assert (result.n_embedded, result.n_not_embedded) == (1, 2)
    assert encoder.embed_crops_calls == [1]  # only the selected crop reached the encoder

    docs = _by_score(os_fake)
    widget = docs[0.9]
    gadget = docs[0.95]
    unclassed = docs[0.3]
    assert widget['embedding_state'] == 'embedded'
    assert 'pe_embedding' in widget
    assert gadget['embedding_state'] == 'not_selected'
    assert 'pe_embedding' not in gadget
    assert gadget['cluster_id'] == 0  # a class-labeled item keeps its class cluster
    assert unclassed['embedding_state'] == 'not_selected'
    assert 'pe_embedding' not in unclassed
    assert 'cluster_id' not in unclassed  # no IVF placement without a vector


@pytest.mark.asyncio
async def test_selected_mode_still_places_an_embedded_unclassed_item(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        'src.services.curation.ingest_index.get_ivf_ingest_store', lambda: _IvfStore()
    )
    svc, os_fake = _service(
        IngestPolicy(
            embedding=EmbeddingPolicy(mode='selected', min_confidence=0.2, max_per_image=3)
        )
    )
    await svc.ingest_one(_jpeg_bytes(), '/tmp/a.jpg')
    unclassed = _by_score(os_fake)[0.3]
    assert unclassed['embedding_state'] == 'embedded'
    assert unclassed['cluster_id'] == 7 + 10_000


@pytest.mark.asyncio
async def test_lazy_mode_stores_everything_deferred() -> None:
    svc, os_fake = _service(IngestPolicy(embedding=EmbeddingPolicy(mode='lazy')))
    result = await svc.ingest_one(_jpeg_bytes(), '/tmp/a.jpg')
    assert (result.n_embedded, result.n_not_embedded) == (0, 3)
    assert svc.pe_encoder.embed_crops_calls == []
    assert {d['embedding_state'] for d in os_fake.items.values()} == {'deferred'}
    assert all('pe_embedding' not in d for d in os_fake.items.values())
    assert all(img.get('pe_embedding') for img in os_fake.images.values())  # frame vector stays on


@pytest.mark.asyncio
async def test_default_policy_embeds_everything() -> None:
    svc, _ = _service(IngestPolicy())
    result = await svc.ingest_one(_jpeg_bytes(), '/tmp/a.jpg')
    assert (result.n_embedded, result.n_not_embedded, result.n_filtered) == (3, 0, 0)


@pytest.mark.asyncio
async def test_detect_filter_drops_before_storing_and_counts_in_batch() -> None:
    svc, os_fake = _service(IngestPolicy(detect=DetectFilter(exclude_classes=['gadget'])))
    batch = await svc.ingest_batch(
        [_jpeg_bytes(seed=1), _jpeg_bytes(seed=2)], ['/tmp/a.jpg', '/tmp/b.jpg']
    )
    assert batch.summary.crops_indexed == 2  # the widget on each image
    assert batch.summary.n_filtered == 4  # both gadget boxes on each image
    assert [r.n_filtered for r in batch.results] == [2, 2]
    assert 'gadget' not in {d.get('class_name') for d in os_fake.items.values()}
