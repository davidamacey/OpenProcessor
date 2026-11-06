"""An in-memory stand-in for the config repository, for tests of the config-store services.

It keeps the behaviours those tests lean on: documents with immutable revisions, optimistic
concurrency on a save and on an activation, the activation event log (an axis turned off is an
event with no target), the axis/kind rule, the "in use" refusal on delete, the per-scope epochs,
probes, runtime reports and the acknowledgement state. A test whose outcome depends on the
database itself (a lost race, a constraint, row-level security, the SQL of a join) uses the real
repository in the ``db`` tier instead.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import uuid
from dataclasses import replace
from typing import Any

import pytest

from src.db.models.config import ACTIVATION_AXES, DEPLOYMENT_CONFIG_KINDS
from src.db.repos import config as repo
from src.db.repos.config_base import (
    ActivationEvent,
    ActiveConflictError,
    ActiveRef,
    AxisKindError,
    ConfigDoc,
    ConfigInUseError,
    Probe,
    RevisionConflictError,
    RevisionInfo,
    RuntimeReport,
    check_scope,
)


_NAMESPACE = uuid.UUID('00000000-0000-4000-8000-00000000c0f1')
_AXIS_KIND = dict(ACTIVATION_AXES)


def _now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


class _Doc:
    def __init__(self, project_id: uuid.UUID | None, kind: str, name: str) -> None:
        self.id = uuid.uuid4()
        self.project_id = project_id
        self.kind = kind
        self.name = name
        self.current = 0
        self.created_at = _now()
        self.cloned_from_id: uuid.UUID | None = None
        self.cloned_from_revision: int | None = None
        self.revisions: dict[int, dict[str, Any]] = {}


class MemoryConfig:
    """Documents, activations, probes and reports of every scope, held in dictionaries."""

    def __init__(self) -> None:
        self.docs: dict[uuid.UUID, _Doc] = {}
        self.slugs: dict[uuid.UUID, str] = {}
        self.statuses: dict[uuid.UUID, str] = {}
        self.active: dict[tuple[uuid.UUID, str], ActiveRef] = {}
        self.events: list[tuple[uuid.UUID, ActivationEvent]] = []
        self.epochs: dict[uuid.UUID | None, int] = {}
        self.probes: dict[tuple[uuid.UUID, int], Probe] = {}
        self.reports: dict[tuple[uuid.UUID, str, str], RuntimeReport] = {}
        self.deployment_settings: dict[str, repo.DeploymentSettingRow] = {}
        self.state: dict[tuple[uuid.UUID, str], Any] = {}
        self._event_id = 0
        # the database is unreachable: every epoch read fails
        self.down = False

    # --- projects ---------------------------------------------------------------------
    def project_id(self, slug: str) -> uuid.UUID:
        pid = uuid.uuid5(_NAMESPACE, slug)
        self.slugs.setdefault(pid, slug)
        self.statuses.setdefault(pid, 'active')
        return pid

    def set_status(self, slug: str, status: str) -> None:
        self.statuses[self.project_id(slug)] = status

    async def project_id_for(self, slug: str) -> uuid.UUID:
        return self.project_id(slug)

    async def bound_project_id(self) -> uuid.UUID:
        from src.config.project_context import current_project

        return self.project_id(current_project().record.slug)

    # --- documents ----------------------------------------------------------------------
    def _bump(self, scope: uuid.UUID | None) -> None:
        self.epochs[scope] = self.epochs.get(scope, 0) + 1

    def _find(self, project_id: uuid.UUID | None, kind: str, name: str) -> _Doc | None:
        for d in self.docs.values():
            if d.project_id == project_id and d.kind == kind and d.name == name:
                return d
        return None

    def _tag(self, d: _Doc) -> str | None:
        src = self.docs.get(d.cloned_from_id) if d.cloned_from_id else None
        if src is None or d.cloned_from_revision is None:
            return None
        if src.project_id is None:
            return f'{src.name}@{d.cloned_from_revision}'
        return f'{self.slugs.get(src.project_id, "?")}:{src.name}@{d.cloned_from_revision}'

    def _doc(self, d: _Doc, revision: int) -> ConfigDoc | None:
        rev = d.revisions.get(revision)
        if rev is None:
            return None
        return ConfigDoc(
            d.id, d.project_id, d.kind, d.name, revision, rev['body'], rev['description'],
            d.created_at, rev['created_at'], rev['created_by'], d.cloned_from_id,
            d.cloned_from_revision, d.current, self._tag(d),
        )  # fmt: skip

    async def get_config(self, _session, *, project_id, kind, name, revision=None):
        d = self._find(project_id, kind, name)
        return None if d is None else self._doc(d, revision or d.current)

    async def get_config_by_id(self, _session, config_id, revision=None):
        d = self.docs.get(config_id)
        return None if d is None else self._doc(d, revision or d.current)

    async def list_configs(self, _session, *, project_id, kind=None):
        found = [
            d
            for d in self.docs.values()
            if d.project_id == project_id and (kind is None or d.kind == kind)
        ]
        found.sort(key=lambda d: (d.kind, d.name))
        return [self._doc(d, d.current) for d in found]

    async def list_revisions(self, _session, config_id):
        d = self.docs[config_id]
        return [
            RevisionInfo(n, r['created_at'], r['created_by'], r['description'], self._tag(d))
            for n, r in sorted(d.revisions.items())
        ]

    async def save_config(
        self, _session, *, project_id, kind, name, body, expected_revision, description='',
        cloned_from_id=None, cloned_from_revision=None, by='system',
    ):  # fmt: skip
        check_scope(kind, project_id)
        d = self._find(project_id, kind, name)
        if expected_revision is None:
            if d is not None:
                raise RevisionConflictError(d.current)
            d = _Doc(project_id, kind, name)
            d.cloned_from_id, d.cloned_from_revision = cloned_from_id, cloned_from_revision
            self.docs[d.id] = d
        elif d is None or d.current != expected_revision:
            raise RevisionConflictError(d.current if d else 0)
        d.current += 1
        d.revisions[d.current] = {
            'body': body, 'description': description, 'created_at': _now(), 'created_by': by,
        }  # fmt: skip
        self._bump(project_id)
        return self._doc(d, d.current)

    async def delete_config(self, _session, *, project_id, kind, name, expected_revision):
        d = self._find(project_id, kind, name)
        if d is None:
            raise KeyError(name)
        if d.current != expected_revision:
            raise RevisionConflictError(d.current)
        if any(a.config_id == d.id for a in self.active.values()):
            raise ConfigInUseError(name)
        del self.docs[d.id]
        for d2 in self.docs.values():
            if d2.cloned_from_id == d.id:
                d2.cloned_from_id = None
        self._bump(project_id)

    async def ensure_config(self, _session, *, project_id, kind, name, body, description):
        current = await self.get_config(_session, project_id=project_id, kind=kind, name=name)
        if current is not None and repo.body_sha256(current.body) == repo.body_sha256(body):
            return current
        return await self.save_config(
            _session, project_id=project_id, kind=kind, name=name, body=body,
            expected_revision=None if current is None else current.current_revision,
            description=description,
        )  # fmt: skip

    # --- activations ----------------------------------------------------------------------
    def _off(self, project_id: uuid.UUID, axis: str) -> ActivationEvent | None:
        mine = [e for p, e in self.events if p == project_id and e.axis == axis]
        return mine[-1] if mine and mine[-1].to_config_id is None else None

    async def get_activation(self, _session, project_id, axis):
        held = self.active.get((project_id, axis))
        if held is not None:
            return held
        off = self._off(project_id, axis)
        if off is None:
            return None
        return ActiveRef(axis, 'off', activated_at=off.at, activated_by=off.by)

    async def get_activations(self, _session, project_id):
        out = {}
        for axis, _ in ACTIVATION_AXES:
            ref = await self.get_activation(_session, project_id, axis)
            if ref is not None:
                out[axis] = ref
        return out

    async def activate(
        self, _session, *, project_id, axis, config_id, revision, expected_active, by='system',
        reason='activate',
    ):  # fmt: skip
        if axis not in _AXIS_KIND:
            raise AxisKindError(axis)
        current = await self.get_activation(_session, project_id, axis)
        held = current if current is not None and current.state == 'active' else None
        exp_name = (expected_active or {}).get('name')
        exp_rev = (expected_active or {}).get('revision')
        cur_name = held.name if held else None
        if not (exp_name is None and cur_name is None) and (
            exp_name != cur_name or exp_rev != (held.revision if held else None)
        ):
            raise ActiveConflictError(held.as_expected() if held else None)
        self._event_id += 1
        now = _now()
        target: ActiveRef | None = None
        if config_id is not None:
            doc = self.docs.get(config_id)
            if doc is None or revision not in doc.revisions:
                raise KeyError(config_id)
            if doc.kind != _AXIS_KIND[axis]:
                raise AxisKindError(axis)
            target = ActiveRef(axis, 'active', config_id, doc.kind, doc.name, revision, now, by)
            self.active[(project_id, axis)] = target
        else:
            self.active.pop((project_id, axis), None)
        from_doc = self.docs.get(held.config_id) if held and held.config_id else None
        self.events.append(
            (
                project_id,
                ActivationEvent(
                    self._event_id, axis, held.config_id if held else None,
                    held.revision if held else None, config_id, revision, reason, now, by,
                    from_doc.name if from_doc else None,
                ),
            )
        )  # fmt: skip
        self._bump(project_id)
        return await self.get_activation(_session, project_id, axis)  # type: ignore[return-value]

    async def list_activation_events(self, _session, project_id, axis, *, limit=50):
        mine = [e for p, e in self.events if p == project_id and e.axis == axis]
        out = []
        for e in reversed(mine[-limit:]):
            doc = self.docs.get(e.from_config_id) if e.from_config_id else None
            out.append(replace(e, from_name=doc.name if doc else None))
        return out

    async def rollback(self, _session, *, project_id, axis, expected_active):
        events = await self.list_activation_events(_session, project_id, axis, limit=2)
        last = events[0] if events else None
        if last is None or (last.from_config_id is None and len(events) < 2):
            raise LookupError('no_previous')
        if last.from_config_id is None:
            return await self.activate(
                _session, project_id=project_id, axis=axis, config_id=None, revision=None,
                expected_active=expected_active, reason='rollback',
            )  # fmt: skip
        if await self.get_config_by_id(_session, last.from_config_id, last.from_revision) is None:
            raise LookupError('previous_deleted')
        return await self.activate(
            _session, project_id=project_id, axis=axis, config_id=last.from_config_id,
            revision=last.from_revision, expected_active=expected_active, reason='rollback',
        )  # fmt: skip

    def _visible(self) -> dict[uuid.UUID, str]:
        return {p: s for p, s in self.slugs.items() if self.statuses.get(p) != 'deleting'}

    async def list_active_bodies(self, _session, axis):
        out = []
        for (pid, ax), ref in self.active.items():
            if ax != axis or pid not in self._visible() or ref.config_id is None:
                continue
            doc = self.docs[ref.config_id]
            out.append(
                repo.ActiveBody(
                    self.slugs[pid],
                    doc.name,
                    ref.revision or 0,
                    doc.revisions[ref.revision]['body'],
                )
            )
        return sorted(out, key=lambda a: a.slug)

    async def list_axis_state(self, _session, axis):
        out: dict[str, ActiveRef | None] = {}
        for pid, slug in self._visible().items():
            out[slug] = await self.get_activation(_session, pid, axis)
        return out

    # --- epochs, probes, reports, settings, state -----------------------------------------
    async def get_epoch(self, _session, project_id):
        if self.down:
            raise ConnectionError('database down')
        return self.epochs.get(project_id, 0)

    async def bump_epoch(self, _session, project_id):
        self._bump(project_id)
        return self.epochs[project_id]

    async def record_probe(self, _session, *, config_id, revision, ok, result):
        self.probes[(config_id, revision)] = Probe(config_id, revision, _now(), ok, result)
        self._bump(None)

    async def list_probes(self, _session):
        return list(self.probes.values())

    async def upsert_runtime_report(self, _session, *, project_id, process, host, config_epoch):
        self.reports[(project_id, process, host)] = RuntimeReport(
            project_id, process, host, config_epoch, _now()
        )

    async def list_runtime_reports(self, _session, *, project_id, process, max_age_s):
        cutoff = _now() - dt.timedelta(seconds=max_age_s)
        return [
            r
            for (p, proc, _), r in self.reports.items()
            if p == project_id and proc == process and r.applied_at >= cutoff
        ]

    async def get_deployment_setting(self, _session, key):
        return self.deployment_settings.get(key)

    async def put_deployment_setting(self, _session, *, key, value, expected_revision, by='system'):
        current = self.deployment_settings.get(key)
        if (expected_revision is None) != (current is None) or (
            current is not None and current.revision != expected_revision
        ):
            raise RevisionConflictError(current.revision if current else None)
        row = repo.DeploymentSettingRow(
            key, value, 1 if current is None else current.revision + 1, _now(), by
        )
        self.deployment_settings[key] = row
        self._bump(None)
        return row

    async def get_state(self, _session, project_id, key):
        from src.db.repos.settings import StateRow

        entry = self.state.get((project_id, key))
        return None if entry is None else StateRow(key, entry[0], entry[1], _now())

    async def put_state(self, _session, project_id, key, value, *, expected_revision):
        from src.db.repos.settings import StateRow

        entry = self.state.get((project_id, key))
        if (entry[1] if entry else 0) != expected_revision:
            raise RevisionConflictError(entry[1] if entry else 0)
        revision = expected_revision + 1
        self.state[(project_id, key)] = (value, revision)
        return StateRow(key, value, revision, _now())

    # --- reading helpers for assertions ---------------------------------------------------------
    def stored(self, kind: str, name: str, project: str | None = None) -> _Doc | None:
        scope = None if kind in DEPLOYMENT_CONFIG_KINDS else self.project_id(project or 'default')
        return self._find(scope, kind, name)


_installed: list[MemoryConfig] = []


def current() -> MemoryConfig:
    """The in-memory config the running test was given."""
    return _installed[-1]


@contextlib.asynccontextmanager
async def _tx(reason: str = 'config store'):
    del reason
    yield object()


_REPO_FUNCTIONS = (
    'get_config', 'get_config_by_id', 'list_configs', 'list_revisions', 'save_config',
    'delete_config', 'ensure_config', 'get_activation', 'get_activations', 'activate',
    'list_activation_events', 'rollback', 'list_active_bodies', 'list_axis_state', 'get_epoch',
    'record_probe', 'list_probes', 'upsert_runtime_report', 'list_runtime_reports',
    'get_deployment_setting', 'put_deployment_setting',
)  # fmt: skip


def install(monkeypatch: pytest.MonkeyPatch) -> MemoryConfig:
    """Route the config-store services to a fresh in-memory repository."""
    memory = MemoryConfig()
    _installed.append(memory)
    for name in _REPO_FUNCTIONS:
        monkeypatch.setattr(f'src.db.repos.config.{name}', getattr(memory, name))
    monkeypatch.setattr('src.db.repos.settings.get_state', memory.get_state)
    monkeypatch.setattr('src.db.repos.settings.put_state', memory.put_state)
    monkeypatch.setattr('src.services.config_store.store.get_epoch', memory.get_epoch)
    for target, replacement in (
        ('bound_project.project_id_for', memory.project_id_for),
        ('bound_project.bound_project_id', memory.bound_project_id),
        ('documents.project_id_for', memory.project_id_for),
        ('documents.bound_project_id', memory.bound_project_id),
        ('store.project_id_for', memory.project_id_for),
        ('documents.tx', _tx),
        ('snapshot_load.tx', _tx),
        ('store.tx', _tx),
        ('project_usage.tx', _tx),
        ('vlm_usage.tx', _tx),
    ):
        monkeypatch.setattr(f'src.services.config_store.{target}', replacement)
    return memory


@pytest.fixture
def memory_config(monkeypatch: pytest.MonkeyPatch) -> MemoryConfig:
    return install(monkeypatch)


@pytest.fixture(autouse=True)
def _offline_config_is_in_memory(request: pytest.FixtureRequest) -> None:
    """Every test outside the ``db`` tier reads and writes the in-memory config repository; a ``db``
    test uses the database."""
    if request.node.get_closest_marker('db') is None:
        request.getfixturevalue('memory_config')
