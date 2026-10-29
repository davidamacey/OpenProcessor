"""The job directory of one auto-label loop, and the replay that rebuilds its tree.

The directory is the loop's source of truth: ``events.jsonl`` is append-only and every
line is flushed to disk before :meth:`LoopStore.append` returns, so whatever the loop
did before a crash is on record. JSON files (``budget.json``, ``request.json``, node
member files) are replaced atomically, so each is either the old or the new version.

A crash can tear only the line being written. :meth:`LoopStore.events` ignores a final
line without its newline (its ``append`` never returned, so nothing acted on it) and
the next append cuts it off; any other unreadable line means the log is damaged and
the loop refuses to go on rather than guess. :func:`replay` turns the events into the
tree the driver resumes from; the items themselves are the other half of the truth
(answers recovered from their ``assist`` slot, writes recognised by their provenance).
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from src.domain.loop_tree import parse_path
from src.services.curation.assist.proposal import ClassAnswer
from src.services.curation.file_job import FileJob
from src.utils.timeutil import now_iso


if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

VLM = 'vlm'
"""Budget pool of the primary VLM's requests."""
AUDIT = 'audit'
"""Budget pool of the second-opinion audit's requests."""

EVENTS_FILE = 'events.jsonl'
ROOTS_FILE = 'roots.jsonl'
NODES_DIR = 'nodes'


def _fsync_dir(directory: Path) -> None:
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_atomic(target: Path, text: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f'.{target.name}.tmp')
    with tmp.open('w', encoding='utf-8') as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    tmp.replace(target)
    _fsync_dir(target.parent)


class LoopStore:
    """Files of one loop under ``directory`` (named by the loop id)."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.job = FileJob(directory)
        self._lock = threading.Lock()
        self._seq: int | None = None

    @property
    def loop_id(self) -> str:
        return self.directory.name

    @property
    def events_file(self) -> Path:
        return self.directory / EVENTS_FILE

    @property
    def pause_file(self) -> Path:
        return self.directory / 'pause.flag'

    def cancel_requested(self) -> bool:
        return self.job.cancel_requested()

    def pause_requested(self) -> bool:
        return self.pause_file.exists()

    # ------------------------------------------------------------------ events

    def _parse(self) -> tuple[list[dict[str, Any]], int, int]:
        """``(events, bytes of whole lines, file size)``."""
        try:
            raw = self.events_file.read_bytes()
        except FileNotFoundError:
            return [], 0, 0
        events: list[dict[str, Any]] = []
        whole = 0
        *lines, _tail = raw.split(b'\n')
        for number, line in enumerate(lines, 1):
            try:
                event = json.loads(line)
            except ValueError as exc:
                msg = f'events log {self.events_file} is damaged at line {number}'
                raise ValueError(msg) from exc
            if not isinstance(event, dict):
                msg = f'events log {self.events_file} is damaged at line {number}'
                raise ValueError(msg)
            events.append(event)
            whole += len(line) + 1
        return events, whole, len(raw)

    def events(self) -> list[dict[str, Any]]:
        with self._lock:
            return self._parse()[0]

    def _open_log(self) -> int:
        """The last sequence number, after cutting off a line a crash tore."""
        events, whole, size = self._parse()
        if size > whole:
            os.truncate(self.events_file, whole)
        return int(events[-1].get('seq', len(events))) if events else 0

    def append(self, kind: str, /, **fields: Any) -> dict[str, Any]:
        """Append one event and return it once it is on disk."""
        with self._lock:
            if self._seq is None:
                self.directory.mkdir(parents=True, exist_ok=True)
                self._seq = self._open_log()
            event = {'seq': self._seq + 1, 'at': now_iso(), 'event': kind, **fields}
            line = (json.dumps(event, separators=(',', ':')) + '\n').encode('utf-8')
            fd = os.open(self.events_file, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
            try:
                os.write(fd, line)
                os.fsync(fd)
            finally:
                os.close(fd)
            self._seq += 1
            return event

    # ------------------------------------------------------------------ files

    def write_json(self, name: str, value: Any) -> None:
        _write_atomic(self.directory / name, json.dumps(value, default=str))

    def read_json(self, name: str) -> Any:
        """The file's value; ``None`` when it does not exist. An unreadable file is an
        error: a damaged record is not an empty one."""
        target = self.directory / name
        try:
            return json.loads(target.read_text(encoding='utf-8'))
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as exc:
            msg = f'{target} is unreadable: {exc}'
            raise ValueError(msg) from exc

    def write_node_members(self, path: str, ids: Iterable[str]) -> str:
        """Persist a node's member ids (sorted); the file's path relative to the job."""
        parse_path(path)
        rel = f'{NODES_DIR}/{path}.json'
        _write_atomic(self.directory / rel, json.dumps(sorted(ids)))
        return rel

    def read_node_members(self, path: str) -> list[str]:
        parse_path(path)
        target = self.directory / NODES_DIR / f'{path}.json'
        try:
            raw = json.loads(target.read_text(encoding='utf-8'))
        except (OSError, ValueError) as exc:
            msg = f'members of node {path} are missing or unreadable: {exc}'
            raise ValueError(msg) from exc
        if not isinstance(raw, list):
            msg = f'members of node {path} are not a list'
            raise ValueError(msg)
        return sorted(str(i) for i in raw)

    def write_roots(self, lines: list[dict[str, Any]]) -> None:
        _write_atomic(self.directory / ROOTS_FILE, ''.join(json.dumps(x) + '\n' for x in lines))

    def read_roots(self) -> list[dict[str, Any]] | None:
        try:
            text = (self.directory / ROOTS_FILE).read_text(encoding='utf-8')
        except FileNotFoundError:
            return None
        return [json.loads(line) for line in text.splitlines() if line.strip()]


# ---------------------------------------------------------------------- replay


@dataclass
class NodeRecord:
    """What the log says about one node."""

    path: str
    parent: str | None
    level: int
    n: int
    """Eligible members at snapshot."""
    digest: str
    answers: dict[str, ClassAnswer] = field(default_factory=dict)
    """Usable VLM answers by crop id (asked here, or reused from the parent)."""
    failed: set[str] = field(default_factory=set)
    """Asked, no usable answer."""
    pending: dict[int, list[str]] = field(default_factory=dict)
    """Reservations (by id) whose answers are not on record: in flight, or lost in a crash."""
    decision: dict[str, Any] | None = None
    children: list[str] | None = None
    propagate_begun: bool = False
    propagated: dict[str, Any] | None = None


@dataclass
class LoopTree:
    nodes: dict[str, NodeRecord] = field(default_factory=dict)
    audits: list[dict[str, Any]] = field(default_factory=list)
    rollbacks: list[dict[str, Any]] = field(default_factory=list)

    def node(self, path: str) -> NodeRecord:
        try:
            return self.nodes[path]
        except KeyError:
            msg = f'event for node {path} before it was opened'
            raise ValueError(msg) from None


def parse_answers(raw: dict[str, Any]) -> dict[str, ClassAnswer]:
    return {
        str(crop_id): ClassAnswer(
            class_name=value.get('class_name') or None, proposed=value.get('proposed') or ''
        )
        for crop_id, value in raw.items()
    }


def _apply(tree: LoopTree, event: dict[str, Any]) -> None:
    kind = event.get('event')
    if kind == 'node_open':
        path = event['path']
        tree.nodes.setdefault(
            path,
            NodeRecord(
                path=path,
                parent=event.get('parent'),
                level=int(event['level']),
                n=int(event['n']),
                digest=str(event.get('digest', '')),
            ),
        )
    elif kind == 'reserve':
        if event.get('pool') == VLM:
            tree.node(event['path']).pending[int(event['rid'])] = list(event['ids'])
    elif kind == 'answers':
        node = tree.node(event['path'])
        answers = parse_answers(event.get('answers') or {})
        node.answers.update(answers)
        node.failed |= set(event.get('failed') or []) - set(answers)
        if event.get('rid') is not None:
            node.pending.pop(int(event['rid']), None)
    elif kind == 'decision':
        tree.node(event['path']).decision = event
    elif kind == 'children':
        tree.node(event['path']).children = [c['path'] for c in event['children']]
    elif kind == 'propagate_begin':
        tree.node(event['path']).propagate_begun = True
    elif kind == 'propagated':
        tree.node(event['path']).propagated = event
    elif kind == 'audit':
        tree.audits.append(event)
    elif kind == 'rollback':
        tree.rollbacks.append(event)


def replay(events: Iterable[dict[str, Any]]) -> LoopTree:
    """The tree the events describe. Unknown event kinds are ignored; an event for a
    node that was never opened is an error."""
    tree = LoopTree()
    for event in events:
        _apply(tree, event)
    return tree


__all__ = [
    'AUDIT',
    'VLM',
    'LoopStore',
    'LoopTree',
    'NodeRecord',
    'parse_answers',
    'replay',
]
