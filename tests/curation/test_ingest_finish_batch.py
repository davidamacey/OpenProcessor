"""The finish phase of an ingest request opens a bounded number of database transactions.

Every photo used to open its own: a lookup, the content claim, two pending-row writes, the
placement, the ``crop.created`` events, and a class create per detector label (about 4 to 6 per
photo, each ending in a COMMIT that waits for the WAL). A batch now does each of those steps once.

The counting seam is the session factory (``async_session_factory``, patched in every module that
imported it) with the repository functions behind it stubbed: one session is one pool checkout and
one transaction, whatever the code runs inside it. The stubs make no network call, so the same
test runs against the per-photo code too.
"""

from __future__ import annotations

import asyncio
import dataclasses
import uuid
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from curation.ingest_fakes import FakePEEncoder, FakeTritonPool, _jpeg_bytes
from curation.ingest_session_fakes import Sessions, count_sessions
from curation.occ_fakes import FakeIngestOpenSearch
from src.config import CurationConfig, DetectionProfile
from src.services.clustering import assign
from src.services.curation import ingest_index
from src.services.curation.ingest import CurationIngestService
from src.services.curation.ingest_policy import IngestPolicy
from src.services.curation.ingest_writer import IngestWriteError
from src.services.near_duplicates import content_key, ingest_claim
from src.services.near_duplicates.content_key import content_image_id, content_imohash


if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def sessions(monkeypatch: pytest.MonkeyPatch) -> Sessions:
    return count_sessions(monkeypatch)


class _Ensurer:
    """Stands in for the registry write; every call is one transaction (it yields, as a real one
    does, so racing photos would each reach it)."""

    def __init__(self) -> None:
        self.calls: list[list[tuple[str, str]]] = []

    async def __call__(self, *args: Any) -> Any:
        wanted = args[0] if isinstance(args[0], list) else [(args[0], args[1])]
        await asyncio.sleep(0.01)
        self.calls.append(list(wanted))
        found = {name: (100 + i, name) for i, (name, _label) in enumerate(wanted)}
        return found if isinstance(args[0], list) else found[args[0]]


def _service(tmp_path: Path, ensurer: Any = None) -> tuple[CurationIngestService, Any]:
    labels = tmp_path / 'labels.txt'
    labels.write_text('gadget\nwidget\n')
    os_fake = FakeIngestOpenSearch()
    profile = DetectionProfile(
        name='item',
        detector_model='proposer',
        assigns_class=False,
        labels_path=str(labels),
        input_size=320,
        batch_limit=8,
    )

    class Registry:
        classes: list[Any] = []

        def get(self, _class_id: int) -> None:
            return None

    svc = CurationIngestService(
        opensearch=os_fake,
        triton_pool=FakeTritonPool([(0.05, 0.05, 0.6, 0.6, 0.9, 1), (0.1, 0.6, 0.4, 0.9, 0.8, 0)]),
        registry=Registry(),  # type: ignore[arg-type]
        profile=profile,
        pe_encoder=FakePEEncoder(),
        config=dataclasses.replace(
            CurationConfig(), crop_cache_dir=tmp_path / 'crops', crop_cache_max_bytes=0
        ),
        policy=IngestPolicy(),
        class_ensurer=ensurer,
    )
    return svc, os_fake


@pytest.fixture(autouse=True)
def _labels_inside_the_model_repository(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('OP_TRITON_MODEL_REPO', str(tmp_path))


async def _ingest(tmp_path: Path, n: int, ensurer: Any = None) -> tuple[Any, Any]:
    svc, os_fake = _service(tmp_path, ensurer)
    images = [_jpeg_bytes(seed=i + 1) for i in range(n)]
    result = await svc.ingest_batch(images, [f'/p/{i}.jpg' for i in range(n)])
    assert [r.status for r in result.results] == ['success'] * n
    return result, os_fake


@pytest.mark.asyncio
async def test_sessions_per_batch_do_not_grow_with_the_photos(
    sessions: Sessions, tmp_path: Path
) -> None:
    await _ingest(tmp_path, 8)
    small = sessions.opened
    sessions.opened = 0
    result, os_fake = await _ingest(tmp_path, 32)
    assert len(os_fake.images) == 32
    assert sum(r.n_crops for r in result.results) == 64
    # the lookup, one transaction per commit group (at most 3), the events: at most 5, however
    # many photos (the formula of ingest_finish; tests/curation/test_ingest_flow.py pins it with
    # an engine that answers one photo at a time)
    assert small <= 5
    assert sessions.opened <= 5
    assert sessions.peak == 1  # one connection at a time, not one per photo


@pytest.mark.asyncio
async def test_a_batch_creates_each_missing_class_once_in_one_call(
    sessions: Sessions, tmp_path: Path
) -> None:
    ensurer = _Ensurer()
    await _ingest(tmp_path, 32, ensurer)
    assert [sorted(call) for call in ensurer.calls] == [
        [('gadget', 'gadget'), ('widget', 'widget')]
    ]


@pytest.mark.asyncio
async def test_a_second_batch_does_not_create_the_class_again(
    sessions: Sessions, tmp_path: Path
) -> None:
    ensurer = _Ensurer()
    svc, _ = _service(tmp_path, ensurer)
    for round_ in range(2):
        images = [_jpeg_bytes(seed=100 * round_ + i + 1) for i in range(4)]
        await svc.ingest_batch(images, [f'/p/{round_}-{i}.jpg' for i in range(4)])
    assert len(ensurer.calls) == 1


@pytest.mark.asyncio
async def test_a_failed_batch_claim_falls_back_to_one_transaction_per_photo(
    sessions: Sessions, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sqlalchemy.exc import OperationalError

    from src.services.curation import ops_metrics

    async def boom(*_args: Any) -> Any:
        raise OperationalError('claim', {}, Exception('server closed the connection'))

    monkeypatch.setattr(ingest_claim, 'claim_and_mark_in_session', boom)
    counted: list[tuple[str, str]] = []
    monkeypatch.setattr(
        ops_metrics,
        'count_batch_fallback',
        lambda stage, exc: counted.append((stage, type(exc).__name__)),
    )
    monkeypatch.setattr(ingest_index, 'count_batch_fallback', ops_metrics.count_batch_fallback)

    refused = content_key.ContentKeyChangedError('refused')
    real_claim = content_key.claim_bound
    seen: list[str] = []

    async def claim(imohash: str, image_id: str, path: str, data: bytes) -> str | None:
        seen.append(path)
        if path == '/p/2.jpg':
            raise refused
        return await real_claim(imohash, image_id, path, data)

    monkeypatch.setattr(content_key, 'claim_bound', claim)
    svc, os_fake = _service(tmp_path)
    images = [_jpeg_bytes(seed=i + 1) for i in range(5)]
    result = await svc.ingest_batch(images, [f'/p/{i}.jpg' for i in range(5)])

    assert [r.status for r in result.results] == [
        'success',
        'success',
        'failed',
        'success',
        'success',
    ]
    assert result.results[2].error_kind == 'bulk_index'
    assert result.results[2].error == 'refused'
    assert sorted(seen) == [f'/p/{i}.jpg' for i in range(5)]  # each photo had its own claim
    assert len(os_fake.images) == 4  # the refused photo wrote nothing
    assert counted == [('claim', 'OperationalError')]
    assert result.summary.failed == 1
    assert result.summary.successful == 4


@pytest.mark.asyncio
async def test_a_pool_timeout_of_the_batch_claim_is_not_retried_per_photo(
    sessions: Sessions, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sqlalchemy.exc import TimeoutError as PoolTimeoutError

    async def busy(*_args: Any) -> Any:
        raise PoolTimeoutError('QueuePool limit reached')

    each: list[int] = []

    async def claim_each(photos: Any) -> Any:
        each.append(len(photos))
        return []

    monkeypatch.setattr(ingest_claim, 'claim_and_mark_in_session', busy)
    monkeypatch.setattr(ingest_claim, 'claim_and_mark_each', claim_each)
    svc, _ = _service(tmp_path)
    with pytest.raises(PoolTimeoutError):
        await svc.ingest_batch([_jpeg_bytes(seed=1)], ['/p/0.jpg'])
    assert each == []


@pytest.mark.asyncio
async def test_a_failed_write_releases_its_claim_and_keeps_its_pending_rows(
    sessions: Sessions, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    released: list[list[str]] = []

    async def release(image_ids: Any) -> int:
        released.append(sorted(image_ids))
        return 1

    monkeypatch.setattr(content_key, 'release_bound', release)
    svc, os_fake = _service(tmp_path)
    real_bulk = svc._bulk_index

    async def bulk(image_doc: Any, crop_docs: Any, *a: Any, **k: Any) -> Any:
        if image_doc is not None and image_doc['image_path'] == '/p/1.jpg':
            raise IngestWriteError('write refused', release_claim=True)
        return await real_bulk(image_doc, crop_docs, *a, **k)

    monkeypatch.setattr(svc, '_bulk_index', bulk)
    images = [_jpeg_bytes(seed=i + 1) for i in range(3)]
    result = await svc.ingest_batch(images, [f'/p/{i}.jpg' for i in range(3)])
    assert [r.status for r in result.results] == ['success', 'failed', 'success']
    assert result.results[1].error_kind == 'bulk_index'
    # one release for the whole batch, of the failed photo's claim only
    assert released == [[content_image_id(content_imohash(images[1]))]]
    assert len(os_fake.images) == 2


def test_the_placement_of_a_batch_equals_the_placement_of_each_photo() -> None:
    rng = np.random.default_rng(3)
    uids = tuple(uuid.uuid4() for _ in range(4))
    prototypes = assign.PrototypeSet(
        partition_id=uuid.uuid4(),
        uids=uids,
        matrix=rng.normal(size=(4, 8)).astype(np.float32),
        bands=assign.AssignBands(assign_min=0.1, review_min=-0.2, min_margin=0.05),
    )
    prototypes.matrix[:] /= np.linalg.norm(prototypes.matrix, axis=1, keepdims=True)
    vectors = rng.normal(size=(12, 8)).astype(np.float32)
    whole = assign.place_vectors(prototypes, vectors)
    each = [assign.place_vectors(prototypes, vectors[i : i + 1])[0] for i in range(12)]
    # The same prototype and the same band for every vector; the scores agree to float32 noise (a
    # matrix product rounds by block size).
    assert [w[0] for w in whole] == [e[0] for e in each]
    assert [x for w in whole for x in w[1:]] == pytest.approx(
        [x for e in each for x in e[1:]], abs=1e-6
    )


@pytest.mark.asyncio
async def test_the_batch_places_all_its_items_in_one_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.services.clustering.assign import Placement
    from src.services.curation.item_doc import DetectedItem

    calls: list[list[str]] = []

    async def place(_session: Any, *, item_ids: Any, **_k: Any) -> list[Placement]:
        calls.append(list(item_ids))
        return [
            Placement(i, uuid.UUID(int=1), uuid.UUID(int=n), 'assigned', 0.9, 0.5)
            for n, i in enumerate(item_ids)
        ]

    monkeypatch.setattr(ingest_index, 'place_new', place)
    service: Any = SimpleNamespace(policy=IngestPolicy())
    session: Any = object()

    def prepared(ids: list[str]) -> Any:
        items = [DetectedItem(bbox_pixel=(0, 0, 5, 5), score=0.9) for _ in ids]
        for item in items:
            item.pe_embedding = np.ones(3, dtype=np.float32)
        return SimpleNamespace(items=items, cids=ids)

    first, second = prepared(['a', 'b']), prepared(['c'])
    await ingest_index.place_prepared(service, [first, second], session, uuid.uuid4())
    assert calls == [['a', 'b', 'c']]
    assert [i.placement.item_id for i in [*first.items, *second.items]] == ['a', 'b', 'c']


def test_photos_of_one_imohash_never_share_a_round() -> None:
    from src.services.curation.ingest_batch import _rounds

    hashes = ['a', 'b', 'a', 'c', 'a', 'b']
    assert _rounds([0, 1, 2, 3, 4, 5], hashes) == [[0, 1, 3], [2, 5], [4]]
    assert _rounds([], hashes) == []
