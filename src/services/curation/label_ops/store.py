"""On-disk record of label operations, one directory per operation under the
bound project's state dir (``<project_state_dir>/label_ops/<op_id>/``).

``state.json`` (atomic temp+rename through :class:`FileJob`) holds the record;
``members.json`` the ids the operation wrote; ``write.lock`` is held while the
operation writes and ``label_ops/undo.lock`` while any undo of the project runs.
The project's dir is the only place read or written: the id is matched against
:data:`LABEL_OP_ID_RE` before any path is built.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import shutil
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ValidationError

from src.config import get_curation_config
from src.config.project_context import current_project, ensure_marked_dir
from src.core.logging import get_logger
from src.domain.label_ops import (
    LABEL_OP_ID_RE,
    LabelOpKind,
    OpFingerprint,
    label_fingerprint,
    move_fingerprint,
    new_label_op_id,
)
from src.services.curation.file_job import FileJob
from src.services.curation.job_lock import exclusive_start_lock
from src.services.rejections import rejection
from src.utils.timeutil import now_iso


if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

logger = get_logger(__name__)

LABEL_OPS_KEEP = 200
"""Operations kept per project; the oldest are pruned when a new one is created."""

UNDO_PASSES_KEPT = 10

StoredStatus = Literal['writing', 'written', 'undoing', 'undone']
ServedStatus = Literal['writing', 'interrupted', 'written', 'undoing', 'undo_interrupted', 'undone']

_MEMBERS = 'members.json'
_WRITE_LOCK = 'write.lock'
_UNDO_LOCK = 'undo.lock'


class LabelOpRecord(BaseModel):
    """One operation's stored state (schema 1)."""

    schema_version: int = 1
    op_id: str
    kind: LabelOpKind
    writer: str
    at: str
    finished_at: str | None = None
    class_name: str | None = None
    class_id: int | None = None
    target_cluster_id: int | None = None
    target_kind: Literal['class', 'candidate'] | None = None
    label_source: str | None = None
    target: Literal['crop_ids', 'selection']
    selected: int
    count: int | None = None
    conflicts: int | None = None
    fingerprint: OpFingerprint
    status: StoredStatus = 'writing'
    undo_passes: list[dict[str, Any]] = []


class NewLabelOp(BaseModel):
    """What the write route knows before it writes."""

    kind: LabelOpKind
    writer: str
    class_name: str | None
    class_id: int | None
    target_cluster_id: int | None = None
    target_kind: Literal['class', 'candidate'] | None = None
    label_source: str | None = None
    target: Literal['crop_ids', 'selection']
    selected: int


def label_ops_root() -> Path:
    return get_curation_config().project_state_dir / 'label_ops'


def _not_found(op_id: str) -> Exception:
    return rejection(404, 'label_op_not_found', f'no label operation {op_id!r} in this project')


def op_dir(op_id: str) -> Path:
    """The operation's dir; ``404`` for an id of the wrong shape (no path is built)."""
    if not LABEL_OP_ID_RE.fullmatch(op_id):
        raise _not_found(op_id)
    return label_ops_root() / op_id


def create_op(new: NewLabelOp) -> LabelOpRecord:
    """Write the record (status ``writing``) before any item is written, and prune
    the oldest operations beyond :data:`LABEL_OPS_KEEP`."""
    if current_project().read_only:
        raise rejection(409, 'project_read_only', 'the project is read-only; nothing was written')
    now = datetime.now(UTC)
    op_id = new_label_op_id(now)
    at = now_iso()
    if new.kind == 'label':
        assert new.class_name is not None
        fingerprint = label_fingerprint(
            op_id=op_id, writer=new.writer, at=at, class_name=new.class_name
        )
    else:
        fingerprint = move_fingerprint(
            op_id=op_id,
            writer=new.writer,
            at=at,
            class_name=new.class_name if new.target_kind == 'class' else None,
            cluster_id=new.target_cluster_id,
        )
    record = LabelOpRecord(op_id=op_id, at=at, fingerprint=fingerprint, **new.model_dump())
    try:
        ensure_marked_dir(label_ops_root())
        save_op(record)
    except OSError as exc:
        logger.error('label_op_record_failed', error=str(exc))
        raise rejection(503, 'config_store_unavailable', 'could not record the operation') from exc
    _prune(keep=LABEL_OPS_KEEP)
    logger.info('label_op_created', op_id=op_id, kind=record.kind, selected=record.selected)
    return record


def save_op(record: LabelOpRecord) -> None:
    FileJob(op_dir(record.op_id)).write(record.model_dump(mode='json'))


def _read_file(directory: Path) -> LabelOpRecord | None:
    raw = FileJob(directory).read()
    if not raw:
        return None
    try:
        return LabelOpRecord.model_validate(raw)
    except ValidationError as exc:
        logger.warning('label_op_record_unreadable', op_id=directory.name, error=str(exc))
        return None


def read_op(op_id: str) -> LabelOpRecord:
    """The stored record; ``404`` when it is unknown, unreadable or pruned."""
    record = _read_file(op_dir(op_id))
    if record is None:
        raise _not_found(op_id)
    return record


def iter_ops() -> Iterator[LabelOpRecord]:
    """Readable records, newest first (ids sort in creation order). An unreadable one is skipped and logged."""
    root = label_ops_root()
    try:
        names = sorted(
            (p.name for p in root.iterdir() if LABEL_OP_ID_RE.fullmatch(p.name)), reverse=True
        )
    except FileNotFoundError:
        return
    for name in names:
        record = _read_file(root / name)
        if record is not None:
            yield record


def write_members(op_id: str, ids: list[str]) -> None:
    path = op_dir(op_id) / _MEMBERS
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps({'ids': ids}), encoding='utf-8')
    tmp.replace(path)


def read_members(op_id: str) -> list[str] | None:
    """The ids the operation wrote, or ``None`` when the file is missing or unreadable."""
    try:
        raw = json.loads((op_dir(op_id) / _MEMBERS).read_text(encoding='utf-8'))
        ids = raw['ids']
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return [str(i) for i in ids] if isinstance(ids, list) else None


@contextlib.contextmanager
def write_lock(op_id: str) -> Iterator[bool]:
    """Held while the operation writes; yields ``False`` when someone else holds it."""
    with exclusive_start_lock(op_dir(op_id) / _WRITE_LOCK) as got:
        yield got


@contextlib.contextmanager
def undo_lock() -> Iterator[bool]:
    """Held while any undo of the project runs (one at a time across workers)."""
    with exclusive_start_lock(label_ops_root() / _UNDO_LOCK) as got:
        yield got


def _lock_is_held(path: Path) -> bool:
    try:
        fd = os.open(str(path), os.O_RDWR)
    except FileNotFoundError:
        return False
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return True
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)


def served_status(record: LabelOpRecord) -> ServedStatus:
    """The stored status, with a dead writer or undoer shown as ``interrupted`` /
    ``undo_interrupted`` (its lock is free while the record still says it is working)."""
    if record.status == 'writing' and not _lock_is_held(op_dir(record.op_id) / _WRITE_LOCK):
        return 'interrupted'
    if record.status == 'undoing' and not _lock_is_held(label_ops_root() / _UNDO_LOCK):
        return 'undo_interrupted'
    return record.status


def _prune(*, keep: int) -> None:
    """Delete the oldest operations beyond ``keep``; one that is writing or undoing is never deleted."""
    root = label_ops_root()
    names = sorted(
        (p.name for p in root.iterdir() if LABEL_OP_ID_RE.fullmatch(p.name)), reverse=True
    )
    for name in names[keep:]:
        record = _read_file(root / name)
        if record is not None and served_status(record) in ('writing', 'undoing'):
            continue
        shutil.rmtree(root / name, ignore_errors=True)


__all__ = [
    'LABEL_OPS_KEEP',
    'UNDO_PASSES_KEPT',
    'LabelOpRecord',
    'NewLabelOp',
    'ServedStatus',
    'StoredStatus',
    'create_op',
    'iter_ops',
    'label_ops_root',
    'op_dir',
    'read_members',
    'read_op',
    'save_op',
    'served_status',
    'undo_lock',
    'write_lock',
    'write_members',
]
