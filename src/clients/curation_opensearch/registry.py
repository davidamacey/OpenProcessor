"""Class registry: append-only ``class_registry.json`` with snapshots and
atomic writes, mirrored into the ``classes`` index."""

from __future__ import annotations

import asyncio
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.clients.curation_opensearch import ClassRegistryFile, RegistryClassEntry
from src.clients.curation_opensearch.base import config, logger
from src.clients.curation_opensearch.lifecycle import INDEX_BODIES
from src.config import IndexRole, index_name
from src.utils.class_names import resolve_class_by_name


if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch


class ClassRegistryError(RuntimeError):
    """Raised on registry validation / mutation failures."""


class ClassRegistry:
    """Append-only class registry with on-disk snapshots.

    Backed by ``class_registry.json`` (path from
    ``CurationConfig.class_registry_path``). All mutating ops write a
    snapshot to ``class_registry.<ISO-timestamp>.json`` in the same
    directory **before** rewriting the canonical file (atomic rename via
    tmp file).

    The registry is the source of truth for class IDs — every YOLO
    ``.txt`` label references these IDs.
    """

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path is not None else config.class_registry_path
        self._cache: ClassRegistryFile | None = None
        self._cache_mtime: float | None = None

    # ------------------------------------------------------------------ I/O

    def _read_disk(self) -> ClassRegistryFile:
        if not self.path.exists():
            logger.warning('curation_registry_missing', path=str(self.path))
            return ClassRegistryFile()
        with self.path.open('r', encoding='utf-8') as f:
            raw = json.load(f)
        return ClassRegistryFile.model_validate(raw)

    def load(self) -> ClassRegistryFile:
        """Load registry, with mtime-based cache invalidation."""
        if not self.path.exists():
            self._cache = ClassRegistryFile()
            self._cache_mtime = None
            return self._cache
        mtime = self.path.stat().st_mtime
        if self._cache is None or self._cache_mtime != mtime:
            self._cache = self._read_disk()
            self._cache_mtime = mtime
        return self._cache

    def _atomic_write(self, registry: ClassRegistryFile) -> None:
        """Snapshot existing file, then atomic-replace canonical file."""
        registry.updated_at = datetime.now(UTC).isoformat()
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # 1. Snapshot existing file (if any) BEFORE we mutate.
        if self.path.exists():
            ts = datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')
            snapshot_path = self.path.with_name(f'{self.path.stem}.{ts}.json')
            snapshot_path.write_bytes(self.path.read_bytes())
            logger.info('curation_registry_snapshot', snapshot=str(snapshot_path))

        # 2. Atomic write: tmp → fsync → replace.
        tmp_path = self.path.with_suffix(self.path.suffix + '.tmp')
        payload = registry.model_dump_json(indent=2)
        with tmp_path.open('w', encoding='utf-8') as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        tmp_path.replace(self.path)

        # 3. Refresh cache.
        self._cache = registry
        self._cache_mtime = self.path.stat().st_mtime
        logger.info(
            'curation_registry_written', path=str(self.path), n_classes=len(registry.classes)
        )

    # ------------------------------------------------------------------ ops

    def next_id(self) -> int:
        """Return ``max(existing_id) + 1``. ``0`` when registry empty."""
        reg = self.load()
        if not reg.classes:
            return 0
        return max(c.class_id for c in reg.classes) + 1

    def validate_id(self, class_id: int) -> bool:
        """True iff a non-deprecated class with this ID exists."""
        reg = self.load()
        return any(c.class_id == class_id and not c.deprecated for c in reg.classes)

    def get(self, class_id: int) -> RegistryClassEntry | None:
        for c in self.load().classes:
            if c.class_id == class_id:
                return c
        return None

    def add_class(self, name: str, group: str = 'unknown', notes: str = '') -> int:
        """Append a new class. Refuses duplicate (non-deprecated) names.

        Returns the assigned class_id.
        """
        reg = self.load()
        name_norm = name.strip()
        if not name_norm:
            raise ClassRegistryError('class_name must be non-empty')
        taken = resolve_class_by_name(reg.classes, name_norm)
        if taken.active is not None:
            raise ClassRegistryError(
                f'duplicate class_name {name_norm!r} (id={taken.active.class_id})'
            )

        new_id = (max((c.class_id for c in reg.classes), default=-1)) + 1
        entry = RegistryClassEntry(
            class_id=new_id,
            class_name=name_norm,
            group=group,
            notes=notes,
        )
        reg.classes.append(entry)
        self._atomic_write(reg)
        logger.info(
            'curation_registry_add_class', class_id=new_id, class_name=name_norm, group=group
        )
        return new_id

    def rename_class(self, class_id: int, new_name: str) -> RegistryClassEntry:
        """Rename a class in place. Class IDs are immutable; only the name changes."""
        reg = self.load()
        new_name_norm = new_name.strip()
        if not new_name_norm:
            raise ClassRegistryError('new_name must be non-empty')
        taken = resolve_class_by_name(
            (c for c in reg.classes if c.class_id != class_id), new_name_norm
        )
        if taken.active is not None:
            raise ClassRegistryError(
                f'rename target {new_name_norm!r} already in use (id={taken.active.class_id})'
            )
        target: RegistryClassEntry | None = None
        for c in reg.classes:
            if c.class_id == class_id:
                target = c
                old = c.class_name
                c.class_name = new_name_norm
                break
        if target is None:
            raise ClassRegistryError(f'class_id {class_id} not found')
        self._atomic_write(reg)
        logger.info(
            'curation_registry_rename_class',
            class_id=class_id,
            old_name=old,
            new_name=new_name_norm,
        )
        return target

    def merge_class(self, source_id: int, target_id: int) -> dict[str, Any]:
        """Mark ``source_id`` deprecated and record ``merged_into=target_id``.

        Does NOT rewrite labels — bulk relabeling of confirmed labels /
        item docs happens elsewhere.

        Returns:
            ``{source_id, target_id, deprecated, source_name, target_name}``.
        """
        if source_id == target_id:
            raise ClassRegistryError('cannot merge a class into itself')
        reg = self.load()
        source = next((c for c in reg.classes if c.class_id == source_id), None)
        target = next((c for c in reg.classes if c.class_id == target_id), None)
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
        # Note: source.class_id stays burned forever (never re-assigned).
        self._atomic_write(reg)
        logger.info(
            'curation_registry_merge_class',
            source_id=source_id,
            target_id=target_id,
            source_name=source.class_name,
            target_name=target.class_name,
        )
        return {
            'source_id': source_id,
            'target_id': target_id,
            'deprecated': True,
            'source_name': source.class_name,
            'target_name': target.class_name,
        }

    def set_deprecated(self, class_id: int, deprecated: bool) -> RegistryClassEntry:
        """Toggle ``deprecated`` on a class in place.

        Used by ``POST /classes/{id}/deprecate`` (``deprecated=True``) and
        ``POST /classes/{id}/restore`` (``deprecated=False``) -- the direct
        retirement path for a class with no data, as opposed to
        :meth:`merge_class` which deprecates the source while relabeling
        its items into a target.

        Deprecating clears ``hotkey_letter`` so the letter can't collide
        with a future class binding it (mirrors the "not bound to another
        *active* class" hotkey rule -- a deprecated class no longer counts
        as active, so its old letter must not linger as if it still did).

        Restoring (``deprecated=False``) re-enters the non-deprecated
        name-uniqueness pool: raises ``ClassRegistryError`` if another
        non-deprecated class already holds this name (same rule
        :meth:`rename_class` enforces on rename).

        Raises ``ClassRegistryError`` if ``class_id`` is unknown.
        """
        reg = self.load()
        target: RegistryClassEntry | None = None
        for c in reg.classes:
            if c.class_id == class_id:
                target = c
                break
        if target is None:
            raise ClassRegistryError(f'class_id {class_id} not found')

        if not deprecated and target.deprecated:
            taken = resolve_class_by_name(
                (c for c in reg.classes if c.class_id != class_id), target.class_name
            )
            if taken.active is not None:
                raise ClassRegistryError(
                    f'cannot restore class_id {class_id}: name {target.class_name!r} '
                    f'already in use by class_id {taken.active.class_id}'
                )

        target.deprecated = deprecated
        if deprecated:
            target.hotkey_letter = None
        self._atomic_write(reg)
        logger.info(
            'curation_registry_set_deprecated',
            class_id=class_id,
            class_name=target.class_name,
            deprecated=deprecated,
        )
        return target

    # ------------------------------------------------------------- OS sync

    async def sync_to_opensearch(self, client: AsyncOpenSearch) -> dict[str, int]:
        """Mirror the registry into the classes OpenSearch index.

        Existing docs are upserted by ``class_id``. Deprecated classes remain
        present (with ``deprecated=true``) so dashboards can show history.
        """
        reg = self.load()
        index = index_name(config, IndexRole.CLASSES)

        # Ensure the index exists.
        if not await client.indices.exists(index=index):
            await client.indices.create(index=index, body=INDEX_BODIES[IndexRole.CLASSES])
            logger.info('curation_index_created_on_sync', index=index)

        # One bulk() instead of one index() per class.
        upserted = 0
        if reg.classes:
            bulk_body: list[dict[str, Any]] = []
            for entry in reg.classes:
                bulk_body.append({'index': {'_index': index, '_id': str(entry.class_id)}})
                bulk_body.append(entry.model_dump())
            resp = await client.bulk(body=bulk_body, refresh=False)
            if isinstance(resp, dict) and resp.get('errors'):
                logger.warning(
                    'curation_registry_sync_partial_errors', sample=resp.get('items', [])[:3]
                )
            upserted = len(reg.classes)
        await client.indices.refresh(index=index)
        logger.info('curation_registry_sync', upserted=upserted, n_classes=len(reg.classes))
        return {'upserted': upserted, 'n_classes': len(reg.classes)}


# =============================================================================
# Module-level convenience
# =============================================================================


# One registry per (project, registry path): each project owns its own
# ``class_registry.json``, so a class added in one project is invisible
# in every other.
_registries: dict[tuple[str, str], ClassRegistry] = {}
_registry_lock = asyncio.Lock()


def get_class_registry() -> ClassRegistry:
    """Return the bound project's :py:class:`ClassRegistry` (one cached
    instance per project per process)."""
    key = (config.project_slug, str(config.class_registry_path))
    registry = _registries.get(key)
    if registry is None:
        registry = ClassRegistry(config.class_registry_path)
        _registries[key] = registry
    return registry
