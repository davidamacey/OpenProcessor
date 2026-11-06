"""Holdout freezes and label operations: the state files of a project, as :mod:`model` records."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

from src.services.upgrade.from_060 import sources as legacy
from src.services.upgrade.from_060.ids import stable_id
from src.services.upgrade.from_060.model import HoldoutSrc, LabelOpSrc
from src.services.upgrade.from_060.parse_core import parse_ts


if TYPE_CHECKING:
    from src.services.upgrade.from_060.model import ProjectSrc
    from src.services.upgrade.from_060.snapshot import Snapshot

UNKNOWN_FREEZER = 'unknown'
_OP_STATE = re.compile(r'^label_ops/(?P<op>[^/]+)/state\.json$')
_CURRENT = f'{legacy.HOLDOUT_DIR}/current.json'
# What an operation left half-done becomes: its worker is gone, so ``writing`` cannot be a live
# state in 0.7.0 and an interrupted undo is a written operation that can be undone again.
_STATUS_AT_UPGRADE = {'writing': 'failed', 'undoing': 'written'}


def parse_holdouts(snap: Snapshot, project: ProjectSrc) -> tuple[HoldoutSrc, ...]:
    """Every freeze record; the curated one that ``current.json`` equals is the current freeze."""
    files = snap.area_files(f'state:{project.slug}')
    prefix = f'{legacy.HOLDOUT_DIR}/'
    current_raw = files[_CURRENT].json() if _CURRENT in files else None
    found: list[HoldoutSrc] = []
    current_matched = False
    for rel, file in sorted(files.items()):
        if not rel.startswith(prefix) or rel == _CURRENT or not rel.endswith('.json'):
            continue
        record = file.json()
        curated = rel.count('/') == 1
        is_current = (
            curated
            and not current_matched
            and current_raw is not None
            and _same_freeze(record, current_raw)
        )
        current_matched = current_matched or is_current
        found.append(_holdout(project, rel, record, is_current=is_current))
    if current_raw is not None and not current_matched:
        found.append(_holdout(project, _CURRENT, current_raw, is_current=True))
    return tuple(found)


def _same_freeze(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return a.get('test_holdout_sha') == b.get('test_holdout_sha') and a.get('frozen_at') == b.get(
        'frozen_at'
    )


def _holdout(
    project: ProjectSrc, rel: str, record: dict[str, Any], *, is_current: bool
) -> HoldoutSrc:
    frozen_at = parse_ts(record.get('frozen_at'), project.updated_at)
    name = rel.removeprefix(f'{legacy.HOLDOUT_DIR}/').replace('/', '_')
    return HoldoutSrc(
        id=stable_id('holdout', project.slug, rel),
        is_current=is_current,
        source=str(record.get('kind') or 'curated'),
        item_count=int(record.get('n_frozen') or len(record.get('crop_ids') or [])),
        frozen_at=frozen_at,
        frozen_by=UNKNOWN_FREEZER,
        snapshot_key=f'projects/{project.slug}/holdout/{name}',
        record=record,
    )


def parse_label_ops(snap: Snapshot, project: ProjectSrc) -> tuple[LabelOpSrc, ...]:
    """The label-operation records (members are streamed by the load, not parsed here)."""
    out: list[LabelOpSrc] = []
    for rel, file in sorted(snap.area_files(f'state:{project.slug}').items()):
        match = _OP_STATE.match(rel)
        if not match:
            continue
        record: dict[str, Any] = file.json()
        legacy_id = str(record.get('op_id') or match['op'])
        stored = str(record.get('status') or 'written')
        status = _STATUS_AT_UPGRADE.get(stored, stored)
        passes = record.get('undo_passes') or []
        undone_at = None
        if status == 'undone':
            last = passes[-1] if passes else {}
            undone_at = parse_ts(last.get('at') or record.get('finished_at'), project.updated_at)
        params = {k: v for k, v in record.items() if k not in {'status', 'op_id'}}
        params['legacy_op_id'] = legacy_id
        if stored != status:
            params['status_at_upgrade'] = stored
        out.append(
            LabelOpSrc(
                id=stable_id('label_op', project.slug, legacy_id),
                legacy_id=legacy_id,
                kind=str(record.get('kind') or 'label'),
                status=status,
                params=params,
                created_at=parse_ts(record.get('at'), project.updated_at),
                created_by=str(record.get('writer') or UNKNOWN_FREEZER),
                undone_at=undone_at,
                members=(),
            )
        )
    return tuple(out)


def member_ids(data: bytes) -> list[str]:
    """The item ids of a ``members.json`` (``{"ids": [...]}``); empty when unreadable."""
    try:
        ids = json.loads(data)['ids']
    except (ValueError, KeyError, TypeError):
        return []
    return [str(i) for i in ids] if isinstance(ids, list) else []
