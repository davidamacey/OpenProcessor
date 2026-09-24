"""Registry-growth reclassification of unmatched VLM labels.

When the VLM labeler returns a class the registry cannot resolve, the item is
written ``class_source='<prefix>_unmatched'`` with the raw answer preserved
under ``<prefix>_raw_label`` (``GET /curation/review/unmatched_terms``
aggregates those). Once an operator grows the registry — new classes, or new
synonyms in the active :class:`~src.services.labeling.vlm_prompts.PromptPack`
— this pass re-resolves every unmatched raw label with the same
:func:`~src.services.labeling.vlm_labeler.resolve_class_name` the live
labeler uses and promotes the hits to::

    class_id / class_name = <resolved class>
    cluster_id            = class_id   (class buckets mirror class ids)
    cluster_subid         = cleared    (sub-clusters are cluster-local)
    class_source          = '<prefix>_reclassified'
    class_id_history      = appended

``class_validated`` is never set — these remain machine suggestions awaiting
human confirmation, just now attached to a real class. Human-owned,
already-validated and frozen ``test_holdout`` items are never touched (the
guard is re-checked against the freshest doc state at write time through
:func:`~src.clients.occ.occ_skip_on_conflict_bulk`).

Idempotent: a converted item no longer carries the unmatched source, so a
second run finds nothing to do. Resumable: pages are walked with
``search_after`` on ``crop_id``; every page reports its cursor, which can be
passed back as ``search_after`` to continue an interrupted run.

The ``<prefix>`` is not hardcoded: different labeling paths stamp different
prefixes, so callers pass one :class:`UnmatchedLabelSource` per path.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from src.clients.occ import occ_skip_on_conflict_bulk
from src.config import CurationConfig, get_curation_config
from src.core.logging import get_logger
from src.services.curation.class_write_guard import CLASS_GUARD_SOURCE_FIELDS, ClassWriteGuard
from src.services.curation.history import record_class_history


if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch

    from src.clients.curation_opensearch import ClassRegistry
    from src.services.labeling.vlm_prompts import PromptPack


logger = get_logger(__name__)

_CONFIDENCE_LEVELS = frozenset({'high', 'medium', 'low'})


@dataclass(frozen=True)
class UnmatchedLabelSource:
    """Field/value naming for one labeling path, derived from its prefix."""

    prefix: str

    def __post_init__(self) -> None:
        if not self.prefix or not self.prefix.replace('_', '').isalnum():
            raise ValueError(f'invalid label prefix {self.prefix!r}')

    @property
    def unmatched_source(self) -> str:
        return f'{self.prefix}_unmatched'

    @property
    def reclassified_source(self) -> str:
        return f'{self.prefix}_reclassified'

    @property
    def raw_label_field(self) -> str:
        return f'{self.prefix}_raw_label'

    @property
    def confidence_field(self) -> str:
        return f'{self.prefix}_confidence'


@dataclass
class ReclassifyResult:
    source: str
    dry_run: bool
    pages: int = 0
    scanned: int = 0
    matched: int = 0
    converted: int = 0
    skipped: int = 0
    errors: int = 0
    by_class: Counter[str] = field(default_factory=Counter)
    last_cursor: list[Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            'source': self.source,
            'dry_run': self.dry_run,
            'pages': self.pages,
            'scanned': self.scanned,
            'matched': self.matched,
            'converted': self.converted,
            'skipped': self.skipped,
            'errors': self.errors,
            'by_class': dict(self.by_class.most_common()),
            'last_cursor': self.last_cursor,
        }


def active_name_to_id(registry: ClassRegistry) -> dict[str, int]:
    """``{class_name: class_id}`` for non-deprecated registry classes only —
    a deprecated id must never be assigned."""
    return {c.class_name: c.class_id for c in registry.load().classes if not c.deprecated}


def unmatched_query(source: UnmatchedLabelSource) -> dict[str, Any]:
    return {
        'bool': {
            'filter': [
                {'term': {'class_source': source.unmatched_source}},
                {'exists': {'field': source.raw_label_field}},
            ],
            'must_not': [
                {'term': {'class_validated': True}},
                {'term': {'test_holdout': True}},
                {'term': {'class_excluded': True}},
            ],
        }
    }


async def reclassify_unmatched(
    opensearch: AsyncOpenSearch,
    *,
    source: UnmatchedLabelSource,
    registry: ClassRegistry,
    pack: PromptPack | None = None,
    config: CurationConfig | None = None,
    page_size: int = 1000,
    max_pages: int = 0,
    search_after: list[Any] | None = None,
    dry_run: bool = True,
) -> ReclassifyResult:
    """Promote ``<prefix>_unmatched`` items whose raw label now resolves.

    Args:
        opensearch: AsyncOpenSearch client.
        source: Which labeling path's unmatched items to revisit.
        registry: Class registry (only non-deprecated classes are targets).
        pack: Prompt pack supplying synonyms; defaults to the active pack.
        config: Supplies ``items_index``.
        page_size: Items per ``search_after`` page.
        max_pages: Stop after this many pages (0 = no limit).
        search_after: Resume cursor from a previous run's ``last_cursor``.
        dry_run: Count matches without writing.

    Returns:
        A :class:`ReclassifyResult` summary (``last_cursor`` is the resume
        point).
    """
    from src.services.labeling.vlm_labeler import resolve_class_name
    from src.services.labeling.vlm_prompts import resolve_prompt_pack

    cfg = config or get_curation_config()
    active_pack = pack or resolve_prompt_pack(cfg)
    name_to_id = active_name_to_id(registry)
    result = ReclassifyResult(source=source.prefix, dry_run=dry_run, last_cursor=search_after)
    if not name_to_id:
        logger.warning('registry_reclassify_no_active_classes')
        return result

    def _resolve(src: dict[str, Any]) -> str | None:
        raw = src.get(source.raw_label_field)
        if not isinstance(raw, str) or not raw.strip():
            return None
        conf = src.get(source.confidence_field)
        return resolve_class_name(
            raw.strip(),
            name_to_id,
            confidence=conf if conf in _CONFIDENCE_LEVELS else None,
            pack=active_pack,
        )

    cursor = search_after
    while True:
        body: dict[str, Any] = {
            'size': page_size,
            '_source': [
                'crop_id',
                source.raw_label_field,
                source.confidence_field,
                *CLASS_GUARD_SOURCE_FIELDS,
            ],
            'query': unmatched_query(source),
            'sort': [{'crop_id': 'asc'}],
        }
        if cursor is not None:
            body['search_after'] = cursor
        resp = await opensearch.search(index=cfg.items_index, body=body)
        hits = (resp.get('hits') or {}).get('hits') or []
        if not hits:
            break
        result.pages += 1
        result.scanned += len(hits)
        cursor = hits[-1].get('sort')

        targets: dict[str, str] = {}
        guard = ClassWriteGuard('registry_reclassify')
        for hit in hits:
            resolved = _resolve(hit.get('_source') or {})
            if resolved is not None:
                targets[hit['_id']] = resolved
                guard.remember(hit['_id'], hit.get('_source') or {})
                result.by_class[resolved] += 1
        result.matched += len(targets)

        if targets and not dry_run:
            now = datetime.now(UTC).isoformat()

            def _merge(
                doc_id: str,
                current: dict[str, Any],
                _now: str = now,
                _guard: ClassWriteGuard = guard,
            ) -> dict[str, Any]:
                if (
                    current.get('class_source') != source.unmatched_source
                    or current.get('test_holdout')
                    or current.get('class_excluded')
                    or not _guard.allows(doc_id, current)
                ):
                    return {}
                resolved = _resolve(current)
                if resolved is None:
                    return {}
                cid = name_to_id[resolved]
                return {
                    'class_id': cid,
                    'class_name': resolved,
                    'class_source': source.reclassified_source,
                    'cluster_id': cid,
                    'cluster_subid': None,
                    'class_id_history': record_class_history(
                        current, writer='registry_reclassify', now=_now
                    ),
                    'updated_at': _now,
                }

            written = await occ_skip_on_conflict_bulk(
                opensearch,
                doc_ids=list(targets),
                merger=_merge,
                index=cfg.items_index,
                refresh=False,
                writer_id='registry_reclassify',
            )
            result.converted += int(written.get('updated', 0))
            result.skipped += int(written.get('skipped_due_to_conflict', 0))
            result.errors += len(written.get('errors') or [])

        result.last_cursor = cursor
        logger.info(
            'registry_reclassify_page',
            source=source.prefix,
            page=result.pages,
            scanned=result.scanned,
            matched=result.matched,
            converted=result.converted,
            cursor=cursor,
        )
        if cursor is None or len(hits) < page_size:
            break
        if max_pages and result.pages >= max_pages:
            break

    if result.converted:
        await opensearch.indices.refresh(index=cfg.items_index)
    return result


__all__ = [
    'ReclassifyResult',
    'UnmatchedLabelSource',
    'active_name_to_id',
    'reclassify_unmatched',
    'unmatched_query',
]
