"""Class registry: one OpenSearch document per project, written with optimistic
concurrency (``if_seq_no`` / ``if_primary_term``, ``op_type=create``).

OpenSearch index API: https://docs.opensearch.org/latest/api-reference/document-apis/index-document/
Get API (realtime, ``_source=false``): https://docs.opensearch.org/latest/api-reference/document-apis/get-documents/

``class_registry.json`` is no longer an authority: a project that still has the
file and no document answers 409 ``registry_not_migrated`` until
``openprocessor curation migrate-registry`` has run.
"""

from __future__ import annotations

import asyncio
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any

from opensearchpy.exceptions import ConflictError, NotFoundError

from src.clients.curation_opensearch import ClassRegistryFile, RegistryClassEntry
from src.clients.curation_opensearch.base import config, logger
from src.clients.curation_opensearch.registry_mirror import mirror_entries, sync_all
from src.clients.occ import _RETRY_BACKOFF_SEC
from src.config import IndexRole, index_name
from src.core.exceptions import RequestRejectedError
from src.schemas.curation.config_common import ErrorCode, error_body
from src.utils.class_names import resolve_class_by_name
from src.utils.timeutil import now_iso


if TYPE_CHECKING:
    from collections.abc import Callable

    from opensearchpy import AsyncOpenSearch


REGISTRY_DOC_ID = 'class_registry'
DOC_KEY = 'class_registry'
# About 1-2 MB of ``_source``, far below the default ``http.max_content_length`` (100 MB).
MAX_REGISTRY_CLASSES = 10_000
MAX_ATTEMPTS = 5


class ClassRegistryError(RuntimeError):
    """Raised on registry validation / mutation failures."""


class RegistryDoc(ClassRegistryFile):
    """The registry document: the export-file shape plus the bookkeeping fields."""

    next_class_id: int = 0
    registry_revision: int = 0
    migrated_from_file_at: str | None = None
    reversed_to_file_at: str | None = None


def registry_rejection(code: ErrorCode, message: str, **fields: Any) -> RequestRejectedError:
    return RequestRejectedError(409, error_body(code, message, **fields))


def _next_free_id(doc: RegistryDoc) -> int:
    return max(doc.next_class_id, max((c.class_id for c in doc.classes), default=-1) + 1)


class ClassRegistry:
    """The bound project's class registry document.

    Every method takes the OpenSearch client. Reads return an immutable-by-convention
    :class:`RegistryDoc` (a deep copy); the per-process cache is keyed by the document's
    ``(_seq_no, _primary_term)`` and costs one ``_source=false`` get per check.
    """

    def __init__(
        self,
        *,
        index: str,
        classes_index: str,
        legacy_path: Path,
        project: str = '',
    ) -> None:
        self.index = index
        self.classes_index = classes_index
        self.legacy_path = Path(legacy_path)
        self.project = project
        self._cache: tuple[tuple[int, int], RegistryDoc] | None = None

    # ------------------------------------------------------------------ reads

    async def _version(self, client: AsyncOpenSearch) -> tuple[int, int] | None:
        try:
            resp = await client.get(index=self.index, id=REGISTRY_DOC_ID, _source=False, ignore=404)
        except NotFoundError:
            return None
        if not isinstance(resp, dict) or not resp.get('found'):
            return None
        return int(resp['_seq_no']), int(resp['_primary_term'])

    async def _fetch(self, client: AsyncOpenSearch) -> tuple[RegistryDoc, int, int] | None:
        try:
            resp = await client.get(index=self.index, id=REGISTRY_DOC_ID, ignore=404)
        except NotFoundError:
            return None
        if not isinstance(resp, dict) or not resp.get('found'):
            return None
        raw = (resp.get('_source') or {}).get(DOC_KEY) or {}
        return RegistryDoc.model_validate(raw), int(resp['_seq_no']), int(resp['_primary_term'])

    def _not_migrated(self) -> RequestRejectedError:
        slug = self.project or config.project_slug
        return registry_rejection(
            'registry_not_migrated',
            f"this project's class registry is still the file {self.legacy_path.name} and has "
            f'not been migrated; run: openprocessor curation migrate-registry --project {slug}',
        )

    async def _create_empty(self, client: AsyncOpenSearch) -> None:
        doc = RegistryDoc(updated_at=now_iso())
        try:
            await client.index(
                index=self.index,
                id=REGISTRY_DOC_ID,
                body={DOC_KEY: doc.model_dump()},
                op_type='create',
                refresh=False,
            )
        except ConflictError:
            logger.info('curation_registry_create_raced', project=self.project)

    async def _read_or_create(self, client: AsyncOpenSearch) -> tuple[RegistryDoc, int, int]:
        found = await self._fetch(client)
        if found is not None:
            return found
        if self.legacy_path.exists():
            raise self._not_migrated()
        await self._create_empty(client)
        found = await self._fetch(client)
        if found is None:
            raise ClassRegistryError('class registry document missing after create')
        return found

    async def snapshot(self, client: AsyncOpenSearch, *, create: bool = True) -> RegistryDoc:
        """The current registry. A project with a leftover file and no document is refused
        (fail closed); a new project gets its empty document created, unless ``create`` is
        false (a read-only look at another project): that answers an empty registry."""
        version = await self._version(client)
        if version is not None and self._cache is not None and self._cache[0] == version:
            return self._cache[1].model_copy(deep=True)
        if version is None and not create:
            if self.legacy_path.exists():
                raise self._not_migrated()
            return RegistryDoc()
        doc, seq, term = await self._read_or_create(client)
        self._cache = ((seq, term), doc)
        return doc.model_copy(deep=True)

    # -------------------------------------------------------------- mutations

    async def _mutate(
        self, client: AsyncOpenSearch, fn: Callable[[RegistryDoc], Any]
    ) -> tuple[Any, RegistryDoc]:
        """Read, apply ``fn`` to a copy, write with the version; re-read and re-apply on 409
        up to :data:`MAX_ATTEMPTS` times. ``fn`` must be pure on the document it is handed."""
        for attempt in range(MAX_ATTEMPTS):
            doc, seq, term = await self._read_or_create(client)
            work = doc.model_copy(deep=True)
            result = fn(work)
            work.registry_revision = doc.registry_revision + 1
            work.updated_at = now_iso()
            try:
                await client.index(
                    index=self.index,
                    id=REGISTRY_DOC_ID,
                    body={DOC_KEY: work.model_dump()},
                    if_seq_no=seq,
                    if_primary_term=term,
                    refresh=False,
                )
            except ConflictError:
                logger.info('curation_registry_conflict', attempt=attempt + 1)
                # Jitter keeps contending writers from retrying in lockstep.
                base = _RETRY_BACKOFF_SEC[min(attempt, len(_RETRY_BACKOFF_SEC) - 1)]
                await asyncio.sleep(random.uniform(0.5 * base, 1.5 * base))
                continue
            self._cache = None
            return result, work
        raise registry_rejection(
            'registry_conflict',
            f'the class registry changed under {MAX_ATTEMPTS} attempts in a row; retry',
        )

    async def _mutate_and_mirror(
        self,
        client: AsyncOpenSearch,
        fn: Callable[[RegistryDoc], Any],
        changed: Callable[[Any], set[int]],
    ) -> Any:
        result, doc = await self._mutate(client, fn)
        ids = changed(result)
        await self._mirror(client, [c for c in doc.classes if c.class_id in ids])
        return result

    async def add_class(
        self,
        client: AsyncOpenSearch,
        name: str,
        group: str = 'unknown',
        notes: str = '',
        hotkey_letter: str | None = None,
    ) -> int:
        """Append a class; refuses a duplicate active name. Returns the new id."""
        entry, _created = await self._add(client, name, group, notes, hotkey_letter, ensure=False)
        return entry.class_id

    async def ensure_class(
        self, client: AsyncOpenSearch, name: str, group: str = 'unknown', notes: str = ''
    ) -> tuple[int, bool]:
        """Idempotent by name: the id of the active class called ``name``, creating it when
        absent. Returns ``(class_id, created)``; used by every automated writer."""
        entry, created = await self.ensure_entry(client, name, group, notes)
        return entry.class_id, created

    async def ensure_entry(
        self, client: AsyncOpenSearch, name: str, group: str = 'unknown', notes: str = ''
    ) -> tuple[RegistryClassEntry, bool]:
        """:meth:`ensure_class` returning the whole entry (callers need its spelling)."""
        return await self._add(client, name, group, notes, None, ensure=True)

    async def _add(
        self,
        client: AsyncOpenSearch,
        name: str,
        group: str,
        notes: str,
        hotkey_letter: str | None,
        *,
        ensure: bool,
    ) -> tuple[RegistryClassEntry, bool]:
        name_norm = name.strip()
        if not name_norm:
            raise ClassRegistryError('class_name must be non-empty')

        def fn(doc: RegistryDoc) -> tuple[RegistryClassEntry, bool]:
            taken = resolve_class_by_name(doc.classes, name_norm)
            if taken.active is not None:
                if ensure:
                    return taken.active.model_copy(), False
                raise ClassRegistryError(
                    f'duplicate class_name {name_norm!r} (id={taken.active.class_id})'
                )
            if len(doc.classes) >= MAX_REGISTRY_CLASSES:
                raise registry_rejection(
                    'registry_full', f'the class registry holds {MAX_REGISTRY_CLASSES} classes'
                )
            new_id = _next_free_id(doc)
            doc.next_class_id = new_id + 1
            entry = RegistryClassEntry(
                class_id=new_id,
                class_name=name_norm,
                group=group,
                notes=notes,
                hotkey_letter=hotkey_letter,
            )
            doc.classes.append(entry)
            return entry.model_copy(), True

        entry, created = await self._mutate_and_mirror(
            client, fn, lambda r: {r[0].class_id} if r[1] else set()
        )
        if created:
            logger.info(
                'curation_registry_add_class',
                class_id=entry.class_id,
                class_name=name_norm,
                group=group,
            )
        return entry, created

    async def update_class(
        self,
        client: AsyncOpenSearch,
        class_id: int,
        *,
        name: str | None = None,
        group: str | None = None,
        hotkey_letter: str | None = None,
        set_hotkey: bool = False,
    ) -> RegistryClassEntry:
        """One mutation for a whole ``PUT``: name, group and hotkey change together.

        ``hotkey_letter`` is applied only when ``set_hotkey`` is true (``None`` then clears it).
        """
        name_norm = name.strip() if name is not None else None
        if name is not None and not name_norm:
            raise ClassRegistryError('new_name must be non-empty')

        def fn(doc: RegistryDoc) -> RegistryClassEntry:
            target = next((c for c in doc.classes if c.class_id == class_id), None)
            if target is None:
                raise ClassRegistryError(f'class_id {class_id} not found')
            if name_norm is not None and name_norm != target.class_name:
                taken = resolve_class_by_name(
                    (c for c in doc.classes if c.class_id != class_id), name_norm
                )
                if taken.active is not None:
                    raise ClassRegistryError(
                        f'rename target {name_norm!r} already in use (id={taken.active.class_id})'
                    )
                target.class_name = name_norm
            if group is not None:
                target.group = group
            if set_hotkey:
                target.hotkey_letter = hotkey_letter
            return target.model_copy()

        entry: RegistryClassEntry = await self._mutate_and_mirror(
            client, fn, lambda e: {e.class_id}
        )
        logger.info('curation_registry_update_class', class_id=class_id)
        return entry

    async def merge_class(
        self, client: AsyncOpenSearch, source_id: int, target_id: int
    ) -> dict[str, Any]:
        """Mark ``source_id`` deprecated and record ``merged_into=target_id``. Does not relabel
        items; the id stays burned."""
        if source_id == target_id:
            raise ClassRegistryError('cannot merge a class into itself')

        def fn(doc: RegistryDoc) -> dict[str, Any]:
            source = next((c for c in doc.classes if c.class_id == source_id), None)
            target = next((c for c in doc.classes if c.class_id == target_id), None)
            if source is None:
                raise ClassRegistryError(f'source class_id {source_id} not found')
            if target is None:
                raise ClassRegistryError(f'target class_id {target_id} not found')
            if target.deprecated:
                raise ClassRegistryError(
                    f'target class_id {target_id} is deprecated; cannot merge into it'
                )
            source.deprecated = True
            source.merged_into = target_id
            return {
                'source_id': source_id,
                'target_id': target_id,
                'deprecated': True,
                'source_name': source.class_name,
                'target_name': target.class_name,
            }

        out: dict[str, Any] = await self._mutate_and_mirror(client, fn, lambda r: {r['source_id']})
        logger.info('curation_registry_merge_class', source_id=source_id, target_id=target_id)
        return out

    async def set_deprecated(
        self, client: AsyncOpenSearch, class_id: int, deprecated: bool
    ) -> RegistryClassEntry:
        """Toggle ``deprecated``. Deprecating clears the hotkey; restoring re-enters the name
        pool and is refused when another active class holds the name."""

        def fn(doc: RegistryDoc) -> RegistryClassEntry:
            target = next((c for c in doc.classes if c.class_id == class_id), None)
            if target is None:
                raise ClassRegistryError(f'class_id {class_id} not found')
            if not deprecated and target.deprecated:
                taken = resolve_class_by_name(
                    (c for c in doc.classes if c.class_id != class_id), target.class_name
                )
                if taken.active is not None:
                    raise ClassRegistryError(
                        f'cannot restore class_id {class_id}: name {target.class_name!r} '
                        f'already in use by class_id {taken.active.class_id}'
                    )
            target.deprecated = deprecated
            if deprecated:
                target.hotkey_letter = None
            return target.model_copy()

        entry: RegistryClassEntry = await self._mutate_and_mirror(
            client, fn, lambda e: {e.class_id}
        )
        logger.info('curation_registry_set_deprecated', class_id=class_id, deprecated=deprecated)
        return entry

    async def set_hotkeys(
        self,
        client: AsyncOpenSearch,
        letters: dict[int, str | None],
        *,
        only_if_unset: bool = False,
    ) -> dict[int, str | None]:
        """Set several hotkeys in one mutation. With ``only_if_unset`` a letter is written only
        where the class has none now (the compensating restore of a failed keymap save, which
        must not overwrite a concurrent change). Returns the letters actually applied."""

        def fn(doc: RegistryDoc) -> dict[int, str | None]:
            applied: dict[int, str | None] = {}
            for c in doc.classes:
                if c.class_id not in letters:
                    continue
                if only_if_unset and c.hotkey_letter is not None:
                    continue
                if c.hotkey_letter != letters[c.class_id]:
                    c.hotkey_letter = letters[c.class_id]
                    applied[c.class_id] = c.hotkey_letter
            return applied

        applied: dict[int, str | None] = await self._mutate_and_mirror(client, fn, set)
        return applied

    # --------------------------------------------- seams for registry_transfer

    async def read_versioned(self, client: AsyncOpenSearch) -> tuple[RegistryDoc, int, int] | None:
        """The document with its ``(_seq_no, _primary_term)``, or ``None`` when absent. Never
        creates it and never applies the not-migrated rule (the migration resolves that)."""
        return await self._fetch(client)

    async def create_document(self, client: AsyncOpenSearch, doc: RegistryDoc) -> bool:
        """Create the document with ``op_type=create``; ``False`` when it already exists."""
        try:
            await client.index(
                index=self.index,
                id=REGISTRY_DOC_ID,
                body={DOC_KEY: doc.model_dump()},
                op_type='create',
                refresh=False,
            )
        except ConflictError:
            return False
        self._cache = None
        return True

    async def mutate(self, client: AsyncOpenSearch, fn: Callable[[RegistryDoc], Any]) -> None:
        """One versioned read-modify-write of the whole document (see :meth:`_mutate`)."""
        await self._mutate(client, fn)

    async def sync_to_opensearch(self, client: AsyncOpenSearch) -> dict[str, int]:
        """Mirror every class into the ``classes`` index (deprecated ones stay, flagged)."""
        return await sync_all(client, self.classes_index, await self.snapshot(client))

    async def _mirror(self, client: AsyncOpenSearch, entries: list[RegistryClassEntry]) -> None:
        await mirror_entries(client, self.classes_index, entries)


# One store per (project, registry path): the cache is per project.
_registries: dict[tuple[str, str], ClassRegistry] = {}


def get_class_registry() -> ClassRegistry:
    """The bound project's store (one cached instance per project per process)."""
    key = (config.project_slug, str(config.class_registry_path))
    store = _registries.get(key)
    if store is None:
        store = ClassRegistry(
            index=index_name(config, IndexRole.SETTINGS),
            classes_index=index_name(config, IndexRole.CLASSES),
            legacy_path=config.class_registry_path,
            project=config.project_slug,
        )
        _registries[key] = store
    return store
