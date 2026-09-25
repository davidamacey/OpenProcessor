"""Shared fakes for the curation ingest integration tests.

Fakes the OpenSearch / Triton / PE-encoder I/O boundary (plan §6.0 house
rule) so ingest can run through the real app and the real
``CurationIngestService``. Used by ``test_ingest_roundtrip.py`` (path
ingest), ``test_ingest_upload.py`` (byte upload) and
``test_import_labeled_dataset.py`` (labeled-dataset driver).
"""

from __future__ import annotations

import io
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image


if TYPE_CHECKING:
    from collections.abc import Iterator

    import pytest


def jpeg_bytes(seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(200, 300, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, mode='RGB').save(buf, format='JPEG', quality=95)
    return buf.getvalue()


class FakeInferResult:
    def __init__(self, outputs: dict[str, np.ndarray]) -> None:
        self._outputs = outputs

    def as_numpy(self, name: str) -> np.ndarray:
        return self._outputs[name]


class FakeTritonPool:
    """One detection per image, normalized box [0.1,0.1,0.5,0.5], class 0.

    Replies with one row per *requested* image so the batched ingest path
    (``ingest_batch`` stacks N images into one call) is exercised end to
    end rather than silently reading row 0 N times.
    """

    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    async def infer(self, model_name: str, inputs: list, outputs: list) -> FakeInferResult:  # noqa: ARG002
        batch = int(inputs[0].shape()[0])
        self.batch_sizes.append(batch)
        return FakeInferResult(
            {
                'num_dets': np.full((batch, 1), 1, dtype=np.int32),
                'det_boxes': np.tile(
                    np.array([[[0.1, 0.1, 0.5, 0.5]]], dtype=np.float32), (batch, 1, 1)
                ),
                'det_scores': np.full((batch, 1), 0.95, dtype=np.float32),
                'det_classes': np.zeros((batch, 1), dtype=np.float32),
            }
        )


class FakePEEncoder:
    """Records which whole-frame path was taken: from a server-side path, or
    from the in-memory bytes (byte-upload ingest)."""

    text_ready = False

    def __init__(self) -> None:
        self.whole_frame_paths: list[str] = []
        self.whole_frame_bytes: list[bytes] = []

    async def embed_crops(self, crops: list[np.ndarray], max_batch: int = 32) -> np.ndarray:  # noqa: ARG002
        return np.tile(np.array([1.0, 0.0], dtype=np.float32), (len(crops), 1))

    async def embed_whole_frame(self, path: str) -> np.ndarray | None:
        self.whole_frame_paths.append(path)
        return np.array([0.0, 1.0], dtype=np.float32)

    async def embed_whole_frame_bytes(self, data: bytes) -> np.ndarray | None:
        self.whole_frame_bytes.append(data)
        return np.array([0.0, 1.0], dtype=np.float32)


class FakeClassEntry:
    def __init__(self, class_id: int, class_name: str) -> None:
        self.class_id = class_id
        self.class_name = class_name
        self.deprecated = False


class FakeRegistryFile:
    def __init__(self) -> None:
        self.classes = [FakeClassEntry(0, 'widget')]


class FakeRegistry:
    def load(self) -> FakeRegistryFile:
        return FakeRegistryFile()

    def get(self, class_id: int) -> FakeClassEntry | None:
        for c in self.load().classes:
            if c.class_id == class_id:
                return c
        return None


class FakeOpenSearch:
    """Just enough of AsyncOpenSearch for ingest + /crops + /ingest/status.

    ``near_real_time=True`` models OpenSearch's refresh semantics: a
    document written by ``bulk`` is invisible to ``search``/``msearch``
    until ``indices.refresh`` runs (``get``/``mget`` stay real-time, as in
    OpenSearch). The default keeps every write immediately searchable.
    """

    def __init__(self, *, near_real_time: bool = False) -> None:
        self.images: dict[str, dict[str, Any]] = {}
        self.items: dict[str, dict[str, Any]] = {}
        self.labels: dict[str, dict[str, Any]] = {}
        self.near_real_time = near_real_time
        self._searchable: dict[str, set[str]] = {'images': set(), 'items': set()}
        self.refresh_calls: list[str] = []
        self.indices = self._Indices(self)

    class _Indices:
        def __init__(self, outer: FakeOpenSearch) -> None:
            self._outer = outer

        async def exists(self, index: str) -> bool:  # noqa: ARG002
            return True

        async def create(self, index: str, body: dict) -> dict:  # noqa: ARG002
            return {'acknowledged': True}

        async def refresh(self, index: str) -> dict:
            self._outer.refresh_calls.append(index)
            self._outer._searchable['images'] = set(self._outer.images)
            self._outer._searchable['items'] = set(self._outer.items)
            return {'_shards': {}}

    def _view(self, store: str) -> dict[str, dict[str, Any]]:
        docs = self.images if store == 'images' else self.items
        if not self.near_real_time:
            return docs
        return {k: d for k, d in docs.items() if k in self._searchable[store]}

    def _items_index(self) -> str:
        from src.config import get_curation_config

        return get_curation_config().items_index

    async def search(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:
        query = body.get('query') or {}
        term = (query.get('term') or {}).get('imohash')
        if term is not None:
            store = self._view('images')
            hits = [
                {'_id': d.get('image_id', k), '_source': d}
                for k, d in store.items()
                if d.get('imohash') == term
            ]
            return {'hits': {'hits': hits[:1]}}
        # label_import's images-index lookup by exact source path.
        # /ingest/path_lookup's bulk existence check.
        path_terms = (query.get('terms') or {}).get('image_path')
        if path_terms is not None:
            wanted = set(path_terms)
            hits = [
                {'_id': d.get('image_id', k), '_source': d}
                for k, d in self._view('images').items()
                if d.get('image_path') in wanted
            ]
            return {'hits': {'hits': hits}}
        # BA-1: /ingest/path_lookup matches on image_path OR
        # source_identifier -- {'bool': {'should': [{'terms':
        # {'image_path': [...]}}, {'terms': {'source_identifier': [...]}}]}}.
        should = (query.get('bool') or {}).get('should')
        if should is not None and any('image_path' in (c.get('terms') or {}) for c in should):
            wanted_by_field: dict[str, set[str]] = {}
            for clause in should:
                for field, values in (clause.get('terms') or {}).items():
                    wanted_by_field.setdefault(field, set()).update(values)
            hits = [
                {'_id': d.get('image_id', k), '_source': d}
                for k, d in self._view('images').items()
                if any(d.get(field) in values for field, values in wanted_by_field.items())
            ]
            return {'hits': {'hits': hits}}
        path_term = (query.get('term') or {}).get('image_path')
        if path_term is not None:
            hits = [
                {'_id': d.get('image_id', k), '_source': d}
                for k, d in self._view('images').items()
                if d.get('image_path') == path_term
            ]
            return {'hits': {'hits': hits[:1]}}
        # label_import's items-by-image_id lookup (bool/filter term).
        musts = (query.get('bool') or {}).get('filter') or []
        image_id = next(
            (m['term']['image_id'] for m in musts if (m.get('term') or {}).get('image_id')), None
        )
        if image_id is not None:
            hits = [
                {'_id': k, '_source': d}
                for k, d in self._view('items').items()
                if d.get('image_id') == image_id and not d.get('test_holdout')
            ]
            return {'hits': {'hits': hits, 'total': {'value': len(hits)}}}
        # /curation/crops style query -- return every non-holdout item.
        if index == self._items_index():
            hits = [{'_id': k, '_source': d} for k, d in self._view('items').items()]
            size = body.get('size', len(hits))
            return {
                'hits': {'hits': hits[:size], 'total': {'value': len(hits)}},
            }
        # /curation/ingest/status style aggregation query over images.
        return {'hits': {'hits': [], 'total': {'value': len(self.images)}}, 'aggregations': {}}

    async def msearch(self, *, body: list[dict[str, Any]]) -> dict[str, Any]:
        responses = []
        for line in body[1::2]:
            term = ((line.get('query') or {}).get('term') or {}).get('imohash')
            hits = [
                {'_id': d.get('image_id', k), '_source': d}
                for k, d in self._view('images').items()
                if d.get('imohash') == term
            ]
            responses.append({'hits': {'hits': hits[:1]}})
        return {'responses': responses}

    async def mget(
        self,
        *,
        body: dict[str, Any],
        index: str | None = None,  # noqa: ARG002
        _source_excludes: list[str] | None = None,
    ) -> dict[str, Any]:
        # Two shapes land here: the {'ids': [...]} shape used directly by
        # this module's own callers, and mget_crops' {'docs': [{'_id':...,
        # '_index':...}, ...]} shape (occ_update_bulk, F-17/F-26).
        ids = body['ids'] if 'ids' in body else [d['_id'] for d in body['docs']]
        docs = []
        for doc_id in ids:
            if doc_id in self.items:
                docs.append(
                    {
                        '_id': doc_id,
                        'found': True,
                        '_source': self.items[doc_id],
                        '_seq_no': 1,
                        '_primary_term': 1,
                    }
                )
            else:
                docs.append({'_id': doc_id, 'found': False})
        return {'docs': docs}

    async def bulk(
        self,
        *,
        body: list[dict[str, Any]],
        refresh: bool | str = False,  # noqa: ARG002
    ) -> dict[str, Any]:
        from src.config import get_curation_config

        items_index = self._items_index()
        labels_index = get_curation_config().labels_confirmed_index
        result_items = []
        for action, doc in zip(body[0::2], body[1::2], strict=True):
            if 'index' in action:
                meta = action['index']
                if meta['_index'] == items_index:
                    store = self.items
                elif meta['_index'] == labels_index:
                    store = self.labels
                else:
                    store = self.images
                store[meta['_id']] = doc
                result_items.append({'index': {'_id': meta['_id'], 'status': 201}})
            elif 'update' in action:
                meta = action['update']
                store = self.items if meta['_index'] == items_index else self.images
                store.setdefault(meta['_id'], {}).update(doc.get('doc', {}))
                result_items.append({'update': {'_id': meta['_id'], 'status': 200}})
            elif 'create' in action:
                meta = action['create']
                if meta['_id'] in self.items:
                    result_items.append({'create': {'_id': meta['_id'], 'status': 409}})
                else:
                    self.items[meta['_id']] = doc
                    result_items.append({'create': {'_id': meta['_id'], 'status': 201}})
        return {'errors': False, 'items': result_items}

    async def update(
        self,
        *,
        index: str,  # noqa: ARG002
        id: str,  # noqa: A002
        body: dict[str, Any],
        **kwargs: Any,  # noqa: ARG002
    ) -> dict[str, Any]:
        self.items.setdefault(id, {}).update(body['doc'])
        return {'_id': id, 'result': 'updated', '_seq_no': 2, '_primary_term': 1}

    async def get(
        self,
        *,
        index: str,  # noqa: ARG002
        id: str,  # noqa: A002
        _source_excludes: list[str] | None = None,
    ) -> dict[str, Any]:
        return {'_id': id, '_source': self.items.get(id, {}), '_seq_no': 1, '_primary_term': 1}

    async def count(self, index: str, body: dict | None = None) -> dict[str, Any]:  # noqa: ARG002
        return {'count': len(self.items)}


@contextmanager
def curation_app(
    fake_opensearch: FakeOpenSearch,
    fake_triton: FakeTritonPool,
    monkeypatch: pytest.MonkeyPatch,
    *,
    registry: Any = None,
    pe_encoder: Any = None,
) -> Iterator[TestClient]:
    """The real app with the OpenSearch/Triton/PE/registry boundary faked."""
    import src.main as main_module
    from src.core.dependencies import get_async_triton, get_opensearch
    from src.routers.curation._common import _raw_opensearch_dep, _registry_dep

    reg = registry if registry is not None else FakeRegistry()
    main_module.app.dependency_overrides[get_opensearch] = lambda: fake_opensearch
    main_module.app.dependency_overrides[_raw_opensearch_dep] = lambda: fake_opensearch
    main_module.app.dependency_overrides[get_async_triton] = lambda: fake_triton
    main_module.app.dependency_overrides[_registry_dep] = lambda: reg

    monkeypatch.setenv('OP_INGEST_PRIMARY_DETECTOR_MODEL', 'fake_item_detector')
    # ingest.py's service factory reaches AppResources.async_triton_pool /
    # app.state.pe_encoder directly (not via FastAPI Depends()), and the
    # lifespan startup below unconditionally (re)builds both -- so these
    # must be patched AFTER entering the TestClient context, not before.
    monkeypatch.setattr(main_module, 'get_async_triton_pool', lambda: fake_triton)
    # These tests ingest images written under the system temp dir; declare it
    # the deployment's source root, as a real deployment declares its image
    # store (OP_SOURCE_ROOT), so path-based ingest accepts them.
    from src.services.curation import image_serving

    temp_root = Path(tempfile.gettempdir()).resolve()
    monkeypatch.setattr(image_serving, '_configured_roots', lambda config=None: (temp_root,))  # noqa: ARG005

    # BA-1: POST /ingest/upload persists bytes under CurationConfig.upload_root
    # -- give it a real, writable directory under the same temp root the
    # source-path tests already declare servable, and force the process-wide
    # config singleton to rebuild so it picks this env var up.
    import src.config.curation as curation_config_mod

    monkeypatch.setenv('OP_UPLOAD_ROOT', str(temp_root / 'op_test_uploads'))
    monkeypatch.setattr(curation_config_mod, '_default_curation_config', None)

    try:
        with TestClient(main_module.app) as c:
            main_module.app.state.pe_encoder = pe_encoder or FakePEEncoder()
            yield c
    finally:
        main_module.app.dependency_overrides.clear()
