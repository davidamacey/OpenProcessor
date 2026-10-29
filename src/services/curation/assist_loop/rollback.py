"""Undo what one auto-label loop propagated, leaving every later change alone.

The items are the ones carrying the loop's write (``class_source`` ``assist_propagated``,
``class_detector_version`` = the loop id), optionally only those of chosen nodes. An item
is restored only while it is still exactly as the loop left it: the loop's class from
the node, its history ending with the loop's own entry (writer ``assist_loop:<loop_id>``;
every later class write appends one), and nothing has locked, validated, excluded or
frozen it since. It then gets back the class state that history entry recorded before
the loop wrote. Anything else is kept and counted ``kept_changed``. The check runs again
inside the OCC write, so a person's write racing the rollback wins. Sampled items keep
their direct VLM answer: rollback is for propagated labels.

Restoring is idempotent: a restored item no longer carries the loop's write, so a second
rollback finds nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from src.clients.curation_opensearch.crops import mget_crops
from src.config.curation_accessors import items_index
from src.domain.history import record_class_snapshot, restore_class_state
from src.domain.ingest_class_sources import PROPAGATED_CLASS_SOURCE
from src.services.curation.assist_loop.node_step import LOOP_WRITER
from src.services.curation.class_write_guard import CLASS_GUARD_SOURCE_FIELDS, class_write_locked
from src.services.curation.crop_bulk import occ_bulk_human_relabel
from src.services.curation.reprocess_targets import scan_items
from src.utils.class_names import normalize_class_name
from src.utils.timeutil import now_iso


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from opensearchpy import AsyncOpenSearch

    from src.services.curation.assist_loop.store import LoopStore

_READ_FIELDS = [*CLASS_GUARD_SOURCE_FIELDS, 'class_detector_version', 'test_holdout']


@dataclass
class RollbackResult:
    dry_run: bool
    matched: int
    """Items the loop wrote (in the chosen nodes) that no rollback restored yet."""
    restored: int
    """Restored, or that a dry run would restore."""
    kept_changed: int
    conflicts: int = 0
    restored_ids: list[str] = field(default_factory=list)


def rollback_writer(loop_id: str) -> str:
    return f'{LOOP_WRITER}_rollback:{loop_id}'


def as_left(source: dict[str, Any], loop_id: str, class_name: str | None) -> bool:
    """Whether ``source`` is still exactly what the loop wrote."""
    history = source.get('class_id_history') or []
    last = history[-1] if history and isinstance(history[-1], dict) else {}
    return (
        source.get('class_source') == PROPAGATED_CLASS_SOURCE
        and source.get('class_detector_version') == loop_id
        and last.get('writer') == f'{LOOP_WRITER}:{loop_id}'
        and (
            class_name is None
            or normalize_class_name(str(source.get('class_name') or ''))
            == normalize_class_name(class_name)
        )
        and not class_write_locked(source)
        and not source.get('class_excluded')
        and not source.get('test_holdout')
    )


def _restore(current: dict[str, Any], loop_id: str) -> dict[str, Any]:
    snapshot = (current.get('class_id_history') or [])[-1]
    update = restore_class_state(snapshot)
    update['class_id_history'] = record_class_snapshot(
        current, writer=rollback_writer(loop_id), restorable=False
    )
    update['updated_at'] = now_iso()
    return update


def _propagated_classes(store: LoopStore, node_paths: Sequence[str] | None) -> dict[str, str]:
    """``{crop_id: class}`` the loop's logged propagations wrote (in ``node_paths``)."""
    out: dict[str, str] = {}
    wanted = None if node_paths is None else set(node_paths)
    for event in store.events():
        if event.get('event') == 'propagated' and (wanted is None or event['path'] in wanted):
            for crop_id in event.get('written_ids') or []:
                out[crop_id] = event['class_name']
    return out


def _rolled_back(source: dict[str, Any], loop_id: str) -> bool:
    writer = rollback_writer(loop_id)
    return any(
        isinstance(h, dict) and h.get('writer') == writer
        for h in source.get('class_id_history') or []
    )


async def _written(
    opensearch: AsyncOpenSearch, loop_id: str, logged: Sequence[str], *, scan: bool
) -> dict[str, dict[str, Any]]:
    """Every item the loop wrote and no rollback restored yet: those its log names, and
    (``scan``) any carrying its write whose log line a crash lost."""
    docs = await mget_crops(opensearch, list(logged), source_includes=_READ_FIELDS)
    found = {cid: d.get('_source') or {} for cid, d in docs.items()}
    if scan:
        query = {
            'bool': {
                'filter': [
                    {'term': {'class_source': PROPAGATED_CLASS_SOURCE}},
                    {'term': {'class_detector_version': loop_id}},
                ]
            }
        }
        hits = await scan_items(opensearch, query, index=items_index(), includes=_READ_FIELDS)
        found.update(hits)
    return {cid: src for cid, src in found.items() if not _rolled_back(src, loop_id)}


async def rollback_loop(
    opensearch: AsyncOpenSearch,
    store: LoopStore,
    *,
    node_paths: Sequence[str] | None = None,
    dry_run: bool = True,
) -> RollbackResult:
    """Restore the items the loop in ``store`` propagated (only ``node_paths``' when
    given). A real rollback is logged as a ``rollback`` event."""
    loop_id = store.loop_id
    classes: Mapping[str, str] = _propagated_classes(store, node_paths)
    current = await _written(opensearch, loop_id, sorted(classes), scan=node_paths is None)
    restorable = [cid for cid, src in current.items() if as_left(src, loop_id, classes.get(cid))]
    if dry_run:
        return RollbackResult(
            dry_run=True,
            matched=len(current),
            restored=len(restorable),
            kept_changed=len(current) - len(restorable),
        )
    changed: set[str] = set()

    def merge(source: dict[str, Any]) -> dict[str, Any]:
        crop_id = str(source.get('crop_id', ''))
        if not as_left(source, loop_id, classes.get(crop_id)):
            changed.add(crop_id)
            return {}
        return _restore(source, loop_id)

    written = await occ_bulk_human_relabel(
        opensearch, sorted(restorable), merge, writer_id=rollback_writer(loop_id)
    )
    restored = [i for i in written['updated_ids'] if i not in changed]
    result = RollbackResult(
        dry_run=False,
        matched=len(current),
        restored=len(restored),
        kept_changed=len(current) - len(restorable) + len(changed),
        conflicts=len(written['conflicts']),
        restored_ids=restored,
    )
    if restorable or changed:
        store.append(
            'rollback',
            node_paths=list(node_paths) if node_paths is not None else None,
            restored=result.restored,
            kept_changed=result.kept_changed,
            conflicts=result.conflicts,
        )
    return result


__all__ = ['RollbackResult', 'as_left', 'rollback_loop', 'rollback_writer']
