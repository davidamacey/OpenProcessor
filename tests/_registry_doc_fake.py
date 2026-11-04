"""An OpenSearch stand-in for the class registry document store.

Enforces ``op_type=create`` and ``if_seq_no`` like the server. Pass ``Manager`` proxies
(``docs``, ``seqs``, ``lock``) to share it between spawned processes.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
from typing import Any

from opensearchpy.exceptions import ConflictError


class _Indices:
    async def exists(self, **_kw: Any) -> bool:
        return True

    async def create(self, **_kw: Any) -> dict[str, Any]:
        return {'acknowledged': True}

    async def refresh(self, **_kw: Any) -> dict[str, Any]:
        return {}


class FakeDocOpenSearch:
    def __init__(
        self,
        docs: Any = None,
        seqs: Any = None,
        lock: Any = None,
        mirror: Any = None,
        force_conflicts: int = 0,
    ) -> None:
        self.docs = docs if docs is not None else {}
        self.seqs = seqs if seqs is not None else {}
        self.mirror = mirror if mirror is not None else {}
        self.lock = lock if lock is not None else contextlib.nullcontext()
        self.force_conflicts = force_conflicts
        self.gets: list[bool] = []
        self.bulk_response: dict[str, Any] = {'errors': False, 'items': []}
        self.indices = _Indices()

    async def get(self, *, index: str, id: str, _source: bool = True, **_kw: Any) -> dict[str, Any]:  # noqa: A002
        await asyncio.sleep(0)
        self.gets.append(bool(_source))
        key = f'{index}/{id}'
        with self.lock:
            if key not in self.docs:
                return {'found': False, '_id': id}
            out: dict[str, Any] = {
                'found': True,
                '_id': id,
                '_seq_no': self.seqs[key],
                '_primary_term': 1,
            }
            if _source:
                out['_source'] = copy.deepcopy(self.docs[key])
            return out

    async def index(
        self,
        *,
        index: str,
        id: str,  # noqa: A002
        body: dict[str, Any],
        op_type: str | None = None,
        if_seq_no: int | None = None,
        if_primary_term: int | None = None,  # noqa: ARG002
        **_kw: Any,
    ) -> dict[str, Any]:
        await asyncio.sleep(0)
        key = f'{index}/{id}'
        with self.lock:
            if self.force_conflicts > 0:
                self.force_conflicts -= 1
                raise ConflictError(409, 'version_conflict_engine_exception', {})
            if op_type == 'create' and key in self.docs:
                raise ConflictError(409, 'version_conflict_engine_exception', {})
            if if_seq_no is not None and self.seqs.get(key) != if_seq_no:
                raise ConflictError(409, 'version_conflict_engine_exception', {})
            self.docs[key] = copy.deepcopy(body)
            self.seqs[key] = self.seqs.get(key, 0) + 1
        return {'_id': id, 'result': 'created'}

    async def bulk(self, *, body: list[dict[str, Any]], **_kw: Any) -> dict[str, Any]:
        for action, source in zip(body[::2], body[1::2], strict=True):
            meta = action['index']
            self.mirror[f'{meta["_index"]}/{meta["_id"]}'] = source
        return self.bulk_response


def _run_sync(coro: Any) -> Any:
    """Run ``coro`` to completion from sync code, also when an event loop is already running
    (the in-memory store has no loop affinity)."""
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


class InMemoryRegistry:
    """A real :class:`~src.clients.curation_opensearch.registry.ClassRegistry` over its own
    in-memory document store, for tests that hand a registry to code under test.

    The code under test passes its own (fake) OpenSearch client to every registry call; this
    class ignores that client and talks to its private :class:`FakeDocOpenSearch`, so the
    registry logic itself (OCC, id assignment, name rules) is the production one.
    """

    def __init__(
        self, base_dir: Any = None, names: list[str] | None = None, **entry_kw: Any
    ) -> None:
        from pathlib import Path

        from src.clients.curation_opensearch.registry import ClassRegistry

        self.fake = FakeDocOpenSearch()
        self.path = Path(base_dir or '/nonexistent-registry-dir') / 'class_registry.json'
        self._real = ClassRegistry(
            index='op_test__settings',
            classes_index='op_test__classes',
            legacy_path=self.path,
            project='test',
        )
        if names:
            self.seed(names, **entry_kw)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)

    # ---- sync setup / inspection (no event loop needed)

    def seed(self, names: list[str], **entry_kw: Any) -> None:
        """Append classes straight into the document, ids from the id counter."""
        from src.clients.curation_opensearch import RegistryClassEntry
        from src.clients.curation_opensearch.registry import DOC_KEY, REGISTRY_DOC_ID, RegistryDoc

        key = f'{self._real.index}/{REGISTRY_DOC_ID}'
        raw = self.fake.docs.get(key, {}).get(DOC_KEY)
        doc = RegistryDoc.model_validate(raw) if raw else RegistryDoc()
        for name in names:
            new_id = max(doc.next_class_id, max((c.class_id for c in doc.classes), default=-1) + 1)
            doc.classes.append(RegistryClassEntry(class_id=new_id, class_name=name, **entry_kw))
            doc.next_class_id = new_id + 1
        self.fake.docs[key] = {DOC_KEY: doc.model_dump()}
        self.fake.seqs[key] = self.fake.seqs.get(key, 0) + 1
        self._real._cache = None

    def seed_rows(self, rows: list[dict[str, Any]]) -> None:
        """Replace the classes with ``rows`` (dicts with ``class_id`` and ``class_name``)."""
        from src.clients.curation_opensearch import RegistryClassEntry
        from src.clients.curation_opensearch.registry import DOC_KEY, REGISTRY_DOC_ID, RegistryDoc

        doc = RegistryDoc(classes=[RegistryClassEntry(**row) for row in rows])
        doc.next_class_id = max((c.class_id for c in doc.classes), default=-1) + 1
        key = f'{self._real.index}/{REGISTRY_DOC_ID}'
        self.fake.docs[key] = {DOC_KEY: doc.model_dump()}
        self.fake.seqs[key] = self.fake.seqs.get(key, 0) + 1
        self._real._cache = None

    # ---- sync facades over the real store logic (test setup and inspection)

    def add(self, name: str, **kw: Any) -> int:
        """``add_class`` for synchronous test code; raises ``ClassRegistryError`` like the store."""
        return _run_sync(self._real.add_class(self.fake, name, **kw))

    def deprecate(self, class_id: int, deprecated: bool = True) -> Any:
        return _run_sync(self._real.set_deprecated(self.fake, class_id, deprecated))

    def merge(self, source_id: int, target_id: int) -> Any:
        return _run_sync(self._real.merge_class(self.fake, source_id, target_id))

    def rename_class(self, class_id: int, new_name: str) -> Any:
        return _run_sync(self._real.update_class(self.fake, class_id, name=new_name))

    def load(self) -> Any:
        from src.clients.curation_opensearch.registry import DOC_KEY, REGISTRY_DOC_ID, RegistryDoc

        raw = self.fake.docs.get(f'{self._real.index}/{REGISTRY_DOC_ID}', {}).get(DOC_KEY)
        return RegistryDoc.model_validate(raw) if raw else RegistryDoc()

    @property
    def classes(self) -> Any:
        """Live view, so a service handed this object as its registry snapshot sees later edits."""
        return self.load().classes

    def get(self, class_id: int) -> Any:
        return self.load().get(class_id)

    def validate_id(self, class_id: int) -> bool:
        return self.load().validate_id(class_id)

    # ---- the store API, with the caller's client replaced by the private fake

    async def snapshot(self, _client: Any = None, **kw: Any) -> Any:
        return await self._real.snapshot(self.fake, **kw)

    async def add_class(self, _client: Any, *a: Any, **kw: Any) -> Any:
        return await self._real.add_class(self.fake, *a, **kw)

    async def ensure_class(self, _client: Any, *a: Any, **kw: Any) -> Any:
        return await self._real.ensure_class(self.fake, *a, **kw)

    async def ensure_entry(self, _client: Any, *a: Any, **kw: Any) -> Any:
        return await self._real.ensure_entry(self.fake, *a, **kw)

    async def update_class(self, _client: Any, *a: Any, **kw: Any) -> Any:
        return await self._real.update_class(self.fake, *a, **kw)

    async def merge_class(self, _client: Any, *a: Any, **kw: Any) -> Any:
        return await self._real.merge_class(self.fake, *a, **kw)

    async def set_deprecated(self, _client: Any, *a: Any, **kw: Any) -> Any:
        return await self._real.set_deprecated(self.fake, *a, **kw)

    async def set_hotkeys(self, _client: Any, *a: Any, **kw: Any) -> Any:
        return await self._real.set_hotkeys(self.fake, *a, **kw)

    async def read_versioned(self, _client: Any) -> Any:
        return await self._real.read_versioned(self.fake)

    async def create_document(self, _client: Any, *a: Any, **kw: Any) -> Any:
        return await self._real.create_document(self.fake, *a, **kw)

    async def mutate(self, _client: Any, *a: Any, **kw: Any) -> Any:
        return await self._real.mutate(self.fake, *a, **kw)

    async def sync_to_opensearch(self, _client: Any) -> Any:
        return await self._real.sync_to_opensearch(self.fake)


def make_registry(base_dir: Any, names: list[str], **entry_kw: Any) -> InMemoryRegistry:
    return InMemoryRegistry(base_dir, names, **entry_kw)


def registry_doc_source(
    names: list[str], *, ids: list[int] | None = None, **entry_kw: Any
) -> dict[str, Any]:
    """The ``_source`` of a registry document holding ``names`` as classes ``ids`` (default 0..N-1)."""
    from src.clients.curation_opensearch import RegistryClassEntry
    from src.clients.curation_opensearch.registry import DOC_KEY, RegistryDoc

    class_ids = ids if ids is not None else list(range(len(names)))
    doc = RegistryDoc(
        classes=[
            RegistryClassEntry(class_id=i, class_name=n, **entry_kw)
            for i, n in zip(class_ids, names, strict=True)
        ],
        next_class_id=max(class_ids, default=-1) + 1,
    )
    return {DOC_KEY: doc.model_dump()}


def put_registry_doc(
    store: Any, index: str, names: list[str], *, ids: list[int] | None = None, **entry_kw: Any
) -> None:
    """Seed a registry document into a dict-of-docs fake (``QueryFakeOpenSearch.docs``)."""
    from src.clients.curation_opensearch.registry import REGISTRY_DOC_ID

    store.docs(index)[REGISTRY_DOC_ID] = registry_doc_source(names, ids=ids, **entry_kw)


class InMemoryRegistries(dict):  # type: ignore[type-arg]
    """Stands in for ``registry._registries``: every project gets its own empty in-memory
    registry the first time the code under test asks for it."""

    override: Any = None

    def get(self, key: Any, default: Any = None) -> Any:  # noqa: ARG002
        if self.override is not None:
            return self.override
        if key not in self:
            self[key] = InMemoryRegistry()
        return self[key]


def use_registry(registry: InMemoryRegistry) -> InMemoryRegistry:
    """Make ``registry`` the one every project's ``get_class_registry()`` answers in this test."""
    from src.clients.curation_opensearch import registry as registry_module

    registry_module._registries.override = registry
    return registry


def registry_of(record: Any) -> InMemoryRegistry:
    """The in-memory registry the autouse fixture gave ``record``'s project."""
    from src.clients.curation_opensearch import registry as registry_module

    key = (record.slug, str(record.resources.class_registry_path))
    reg = registry_module._registries.get(key)
    assert isinstance(reg, InMemoryRegistry)
    return reg
