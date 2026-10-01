"""A project as a dataset source (W10.19): another project's validated items,
region boxes, negatives and splits, read page by page under a read-only
binding and projected onto W10's :class:`ScanEntry` stream so the mapping,
completeness and provenance rules stay W10's.

Every read binds the source ``read_only`` around that one call only (never
across a ``yield``), so the project guard refuses any write to it and the
binding is restored before the caller's own writes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.clients.curation_opensearch import ClassRegistry
from src.config import IndexRole
from src.config.curation import BACKBONE_EMBEDDING_FIELD
from src.config.project_context import bind_project
from src.services.curation.dataset_import.issues import IssueCollector
from src.services.curation.dataset_import.scan import (
    DatasetScan,
    LabelBox,
    LabelState,
    ScanEntry,
    class_box_counts,
)


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from src.config.projects import ProjectRecord

PAGE_SIZE = 200
_ITEM_PAGE = 1000
_VECTOR_FIELDS = ('pe_embedding', BACKBONE_EMBEDDING_FIELD)


@dataclass
class SourceImage:
    """One source image doc with the items an include rule keeps."""

    image_id: str
    doc: dict[str, Any]
    items: list[dict[str, Any]] = field(default_factory=list)

    @property
    def imohash(self) -> str:
        return str(self.doc.get('imohash') or '')

    @property
    def path(self) -> str:
        return str(self.doc.get('image_path') or '')

    @property
    def is_negative(self) -> bool:
        return self.doc.get('import_label_state') == 'negative'

    @property
    def split(self) -> str | None:
        split = self.doc.get('dataset_split') or next(
            (i['dataset_split'] for i in self.items if i.get('dataset_split')), None
        )
        return str(split) if split else None

    @property
    def is_holdout(self) -> bool:
        return self.split == 'test' or any(i.get('test_holdout') for i in self.items)


def class_names_by_id(record: ProjectRecord) -> dict[int, str]:
    """The source registry's ``class_id -> name`` (read from its file)."""
    registry = ClassRegistry(path=record.resources.class_registry_path)
    return {c.class_id: c.class_name for c in registry.load().classes}


def item_class_name(item: dict[str, Any], names: dict[int, str]) -> str | None:
    """A class is its NAME: the item's own, else the registry name of its id."""
    name = item.get('class_name')
    if name:
        return str(name)
    class_id = item.get('class_id')
    return names.get(class_id) if isinstance(class_id, int) else None


def _excludes(with_vectors: bool) -> list[str]:
    return [] if with_vectors else list(_VECTOR_FIELDS)


async def _search(
    client: Any, record: ProjectRecord, role: IndexRole, body: dict[str, Any]
) -> list[dict[str, Any]]:
    index = record.resources.indexes[role]
    with bind_project(record, read_only=True):
        resp = await client.search(index=index, body=body)
    return (resp.get('hits') or {}).get('hits') or []


async def fetch_items(
    client: Any,
    record: ProjectRecord,
    image_ids: list[str],
    validated_only: bool = False,
    with_vectors: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    """``{image_id: items}`` for ``image_ids`` (``crop_id`` order); under
    ``validated_only`` only validated items."""
    by_image: dict[str, list[dict[str, Any]]] = {}
    filters: list[dict[str, Any]] = [{'terms': {'image_id': image_ids}}]
    if validated_only:
        filters.append({'term': {'class_validated': True}})
    cursor: list[Any] | None = None
    while True:
        body: dict[str, Any] = {
            'size': _ITEM_PAGE,
            'query': {'bool': {'filter': filters}},
            'sort': [{'crop_id': 'asc'}],
            '_source': {'excludes': _excludes(with_vectors)},
        }
        if cursor is not None:
            body['search_after'] = cursor
        hits = await _search(client, record, IndexRole.ITEMS, body)
        for hit in hits:
            src = hit.get('_source') or {}
            by_image.setdefault(str(src.get('image_id')), []).append(src)
        if len(hits) < _ITEM_PAGE:
            return by_image
        cursor = hits[-1].get('sort')


async def iter_source_pages(
    client: Any,
    record: ProjectRecord,
    *,
    validated_only: bool = False,
    with_vectors: bool = False,
    page_size: int = PAGE_SIZE,
    start_after: str | None = None,
) -> AsyncIterator[list[SourceImage]]:
    """Pages of :class:`SourceImage` in ``image_id`` order. Under
    ``validated_only`` an image is kept when it has a validated item or is a
    reviewed negative; its items are the validated ones."""
    cursor: list[Any] | None = [start_after] if start_after else None
    while True:
        body: dict[str, Any] = {
            'size': page_size,
            'query': {'match_all': {}},
            'sort': [{'image_id': 'asc'}],
            '_source': {'excludes': _excludes(with_vectors)},
        }
        if cursor is not None:
            body['search_after'] = cursor
        hits = await _search(client, record, IndexRole.IMAGES, body)
        if not hits:
            return
        docs = [h.get('_source') or {} for h in hits]
        ids = [str(d['image_id']) for d in docs]
        items = await fetch_items(client, record, ids, validated_only, with_vectors)
        page = [
            SourceImage(image_id=i, doc=d, items=items.get(i, []))
            for i, d in zip(ids, docs, strict=True)
        ]
        kept = [p for p in page if not validated_only or p.items or p.is_negative]
        if kept:
            yield kept
        cursor = hits[-1].get('sort') or [ids[-1]]
        if len(hits) < page_size:
            return


def to_scan_entry(image: SourceImage, names: dict[int, str]) -> ScanEntry:
    boxes = [
        LabelBox(name, tuple(item['bbox_norm']))  # type: ignore[arg-type]
        for item in image.items
        if (name := item_class_name(item, names)) and len(item.get('bbox_norm') or ()) == 4
    ]
    state: LabelState
    if boxes:
        state = 'labeled'
    elif image.is_negative:
        state = 'negative'
    else:
        state = 'unlabeled'
    return ScanEntry(
        rel_path=image.image_id,
        source_stem=image.image_id,
        abs_image_path=Path(image.path),
        split=image.split,
        label_state=state,
        boxes=boxes,
    )


async def scan_project(
    client: Any, record: ProjectRecord, *, validated_only: bool = False
) -> DatasetScan:
    """The whole source as a :class:`DatasetScan` (``format='project'``,
    ``root`` = the source slug): W10's mapping completeness reads its class
    box counts."""
    names = class_names_by_id(record)
    entries: list[ScanEntry] = []
    async for page in iter_source_pages(client, record, validated_only=validated_only):
        entries.extend(to_scan_entry(image, names) for image in page)
    return DatasetScan(
        format='project',
        root=Path(record.slug),
        entries=entries,
        issues=IssueCollector(),
        class_box_counts=class_box_counts(entries),
    )


__all__ = [
    'PAGE_SIZE',
    'SourceImage',
    'class_names_by_id',
    'fetch_items',
    'item_class_name',
    'iter_source_pages',
    'scan_project',
    'to_scan_entry',
]
