"""A true detection-only mode (#279) and the project clustering switch (#279, #292).

``embedding.mode = 'none'`` stores boxes with no vector of any kind (the
whole-frame vector included); ``clustering.automatic = false`` keeps ingest
from placing items in clusters and the background retrains from running.
"""

from __future__ import annotations

import asyncio
import io
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image

from curation.query_fakes import SettingsFakeOpenSearch
from src.domain.embedding_state import DEFERRED
from src.services.curation.ingest_index import ImageContext, index_items
from src.services.curation.ingest_policy import (
    EmbeddingPolicy,
    IngestPolicy,
    IngestPolicyBody,
    embedding_states,
)
from src.services.curation.item_doc import DetectedItem


if TYPE_CHECKING:
    import pytest


def test_the_default_policy_clusters_automatically() -> None:
    assert IngestPolicyBody().clustering.automatic is True


def test_mode_none_defers_every_unlabeled_detection() -> None:
    from src.services.curation.ingest_policy import Candidate

    cands = [Candidate(0.9, 0.1, None, None), Candidate(0.2, 0.01, 'car', None, labeled=False)]
    assert embedding_states(cands, EmbeddingPolicy(mode='none')) == [DEFERRED, DEFERRED]


class _Encoder:
    def __init__(self) -> None:
        self.crops = 0
        self.frames = 0

    async def embed_crops(self, crops: Any, **_kw: Any) -> np.ndarray:
        self.crops += len(crops)
        return np.ones((len(crops), 1024), dtype=np.float32)

    async def embed_whole_frame_image(self, _img: Any) -> np.ndarray:
        self.frames += 1
        return np.ones(1024, dtype=np.float32)


def _index(policy: IngestPolicy, tmp_path: Any) -> tuple[_Encoder, list[dict[str, Any]], Any]:
    encoder = _Encoder()
    indexed: dict[str, Any] = {}

    async def _bulk_index(image_doc, crop_docs, created_ids, *, seed_region=True):
        indexed.update(image=image_doc, crops=crop_docs)
        return {'crops_created': len(crop_docs)}

    service = SimpleNamespace(
        policy=policy,
        profile=SimpleNamespace(batch_limit=16),
        pe_encoder=encoder,
        config=SimpleNamespace(crop_cache_dir=str(tmp_path / 'crops'), crop_cache_max_bytes=0),
        region_seed_status=None,
        _bulk_index=_bulk_index,
        _publish_created=lambda *_a, **_k: None,
    )
    buf = io.BytesIO()
    Image.fromarray(np.zeros((96, 128, 3), dtype=np.uint8)).save(buf, 'JPEG')
    ctx = ImageContext(
        image_id='img',
        image_path='/x.jpg',
        created=True,
        pil=Image.open(io.BytesIO(buf.getvalue())).convert('RGB'),
        width=128,
        height=96,
        imohash='h',
        source='t',
    )
    items = [DetectedItem(bbox_pixel=(10.0, 10.0, 60.0, 50.0), score=0.9)]
    outcome = asyncio.run(index_items(service, ctx, items))  # type: ignore[arg-type]
    assert outcome.result.status == 'success'
    return encoder, indexed['crops'], indexed['image']


def test_mode_none_stores_the_box_with_no_vector_at_all(tmp_path: Any) -> None:
    encoder, crops, image = _index(IngestPolicy(embedding=EmbeddingPolicy(mode='none')), tmp_path)
    assert (encoder.crops, encoder.frames) == (0, 0)
    assert 'pe_embedding' not in image
    assert 'pe_embedding' not in crops[0]
    assert crops[0]['embedding_state'] == DEFERRED


def test_mode_lazy_still_embeds_the_whole_frame(tmp_path: Any) -> None:
    encoder, _crops, image = _index(IngestPolicy(embedding=EmbeddingPolicy(mode='lazy')), tmp_path)
    assert encoder.frames == 1
    assert 'pe_embedding' in image


def test_no_automatic_clustering_places_no_item(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.services.curation.ingest_index as ix

    def _boom() -> Any:
        raise AssertionError('the IVF store must not be consulted')

    monkeypatch.setattr(ix, 'get_ivf_ingest_store', _boom)
    policy = IngestPolicy(clustering={'automatic': False})  # type: ignore[arg-type]
    _encoder, crops, _image = _index(policy, tmp_path)
    assert 'cluster_id' not in crops[0]


def test_the_policy_store_round_trips_the_clustering_switch() -> None:
    from src.services.curation.ingest_policy_store import get_ingest_policy, put_ingest_policy

    fake = SettingsFakeOpenSearch({})

    async def go() -> bool:
        await put_ingest_policy(
            fake,
            IngestPolicyBody(clustering={'automatic': False}),
            expected_revision=0,  # type: ignore[arg-type]
        )
        return (await get_ingest_policy(fake)).clustering.automatic

    assert asyncio.run(go()) is False
