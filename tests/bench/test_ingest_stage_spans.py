"""The ingest stage timers that name the work which used to run untimed (WP-1.2, #210).

Each test drives the real function with fakes at the Triton and OpenSearch boundary and asserts
the stage histogram moved, so a stage that stops being recorded fails here rather than silently
turning back into unattributed API CPU.
"""

from __future__ import annotations

import asyncio
import io
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from prometheus_client import REGISTRY


def _count(stage: str) -> float:
    return REGISTRY.get_sample_value('op_pipeline_stage_seconds_count', {'stage': stage}) or 0.0


def _bytes(stage: str) -> float:
    return REGISTRY.get_sample_value('op_pipeline_stage_bytes_total', {'stage': stage}) or 0.0


def _jpeg(width: int = 96, height: int = 64) -> bytes:
    rng = np.random.default_rng(3)
    buf = io.BytesIO()
    pixels = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(buf, 'JPEG', quality=90)
    return buf.getvalue()


def test_stage_list_names_the_formerly_untimed_work() -> None:
    from src.utils.stage_timing import STAGES

    assert {
        'pe_preprocess',
        'hash',
        'persist',
        'letterbox',
        'triton_request',
        'whole_frame_preprocess',
        'blur',
        'doc_build',
        'bulk_serialize',
    } <= set(STAGES)
    assert 'resize' not in STAGES


def test_letterbox_is_timed_and_unchanged() -> None:
    from src.services.detection.geometry import letterbox_to_square

    before = _count('letterbox')
    chw, scale, pad = letterbox_to_square(Image.new('RGB', (200, 100), (10, 20, 30)), 64)
    assert chw.shape == (1, 3, 64, 64)
    assert (scale, pad) == (0.32, (0.0, 16.0))
    assert _count('letterbox') - before == 1


def test_file_hash_is_timed() -> None:
    from src.services.curation.ingest import _imohash_bytes

    data = _jpeg()
    before, nbytes = _count('hash'), _bytes('hash')
    assert _imohash_bytes(data) == _imohash_bytes(data)
    assert _count('hash') - before == 2
    assert _bytes('hash') - nbytes == 2 * len(data)


def test_whole_frame_preprocess_and_request_build_are_timed() -> None:
    from src.clients.pe_encoder import PEEncoder

    class _Result:
        def as_numpy(self, _name: str) -> np.ndarray:
            return np.ones((1, 1024), dtype=np.float32)

    class _Pool:
        async def infer(self, _model, _inputs, **_kw):
            return _Result()

    encoder = PEEncoder(triton_pool=_Pool())  # type: ignore[arg-type]
    whole, built = _count('whole_frame_preprocess'), _count('triton_request')
    vector = asyncio.run(
        encoder.embed_whole_frame_image(Image.open(io.BytesIO(_jpeg())).convert('RGB'))
    )
    assert vector is not None
    assert vector.shape == (1024,)
    assert _count('whole_frame_preprocess') - whole == 1
    assert _count('triton_request') - built == 1


def test_detector_request_build_is_timed() -> None:
    from src.config import DetectionProfile
    from src.services.curation.ingest_detect import WholeImageDetector

    class _Result:
        def as_numpy(self, name: str) -> np.ndarray:
            return {
                'num_dets': np.zeros((1, 1), dtype=np.int32),
                'det_boxes': np.zeros((1, 1, 4), dtype=np.float32),
                'det_scores': np.zeros((1, 1), dtype=np.float32),
                'det_classes': np.zeros((1, 1), dtype=np.float32),
            }[name]

    class _Pool:
        async def infer(self, _model, _inputs, **_kw):
            return _Result()

    detector = WholeImageDetector(
        triton_pool=_Pool(),
        registry=SimpleNamespace(get=lambda _i: None),
        profile=DetectionProfile(name='item', detector_model='det', input_size=64),
    )
    built = _count('triton_request')
    assert asyncio.run(detector.run_primary(Image.new('RGB', (80, 50)))) == []
    assert asyncio.run(detector.run_primary_batch([Image.new('RGB', (80, 50))])) == [[]]
    assert _count('triton_request') - built == 2


def test_opensearch_serializer_is_timed_and_encodes_like_the_stock_one() -> None:
    from opensearchpy.serializer import JSONSerializer

    from src.clients.opensearch.serializer import OrjsonSerializer

    doc = {'a': [1.5, 2.0], 'b': 'text', 'c': None}
    before = _count('bulk_serialize')
    assert OrjsonSerializer().dumps(doc) == JSONSerializer().dumps(doc)
    assert _count('bulk_serialize') - before == 1


@pytest.fixture
def index_service(tmp_path):
    from src.services.curation.ingest_policy import IngestPolicy

    class _Encoder:
        async def embed_crops(self, crops, **_kw):
            return np.ones((len(crops), 1024), dtype=np.float32)

        async def embed_whole_frame_image(self, _img):
            return np.ones(1024, dtype=np.float32)

    async def _bulk_index(image_doc, crop_docs, created_ids, *, seed_region=True):
        return {'crops_created': len(crop_docs)}

    return SimpleNamespace(
        policy=IngestPolicy(),
        profile=SimpleNamespace(batch_limit=16),
        pe_encoder=_Encoder(),
        config=SimpleNamespace(crop_cache_dir=str(tmp_path / 'crops'), crop_cache_max_bytes=0),
        region_seed_status=None,
        _bulk_index=_bulk_index,
        _publish_created=lambda *_a, **_k: None,
    )


def test_index_items_times_blur_and_document_build(index_service) -> None:
    from src.services.curation.ingest_index import ImageContext, index_items
    from src.services.curation.item_doc import DetectedItem

    data = _jpeg(128, 96)
    ctx = ImageContext(
        image_id='img',
        image_path='/x.jpg',
        created=True,
        pil=Image.open(io.BytesIO(data)).convert('RGB'),
        width=128,
        height=96,
        imohash='h',
        source='t',
    )
    items = [DetectedItem(bbox_pixel=(10.0, 10.0, 60.0, 50.0), score=0.9)]
    blur, docs = _count('blur'), _count('doc_build')
    outcome = asyncio.run(index_items(index_service, ctx, items))
    assert outcome.result.status == 'success'
    assert _count('blur') - blur == 2  # the whole frame, then the one box
    assert _count('doc_build') - docs == 2  # the images document, then the one item
