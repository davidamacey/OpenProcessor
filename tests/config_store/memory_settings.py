"""An in-memory stand-in for the project settings row, for tests of code that only reads and writes it.

It keeps the one behaviour those tests lean on (a single revision that every write must present)
and nothing else. A test whose outcome depends on the database itself (a lost race, a constraint,
row-level security) uses the real repository in the ``db`` tier instead.
"""

from __future__ import annotations

import datetime as dt
import sys
from typing import Any

import pytest

from src.db.repos.config import RevisionConflictError


class _Row:
    def __init__(self) -> None:
        self.defaults: dict[str, Any] = {}
        self.policies: dict[str, Any] = {}
        self.revision = 0
        self.updated_at: str | None = None
        self.refresh_state: dict[str, Any] | None = None
        self.refresh_revision = 0
        self.projection: dict[str, Any] | None = None
        self.keymap: dict[str, list[str]] = {}
        self.keymap_revision = 0


class MemorySettings:
    """One settings row per project slug; the attributes read and write the bound project's row."""

    def __init__(self) -> None:
        self._rows: dict[str, _Row] = {}

    @property
    def _row(self) -> _Row:
        from src.config.project_context import current_project

        return self._rows.setdefault(current_project().record.slug, _Row())

    @property
    def defaults(self) -> dict[str, Any]:
        return self._row.defaults

    @property
    def policies(self) -> dict[str, Any]:
        return self._row.policies

    @property
    def revision(self) -> int:
        return self._row.revision

    @property
    def projection(self) -> dict[str, Any] | None:
        return self._row.projection

    @projection.setter
    def projection(self, value: dict[str, Any] | None) -> None:
        self._row.projection = value

    @property
    def updated_at(self) -> str | None:
        return self._row.updated_at

    def _envelope(self) -> dict[str, Any]:
        return {
            'defaults': {k: v for k, v in self.defaults.items() if v is not None},
            'updated_at': self.updated_at,
            'updated_by': None if self.updated_at is None else 'system',
            'revision': self.revision,
        }

    def _touch(self, expected: int) -> None:
        row = self._row
        if expected != row.revision:
            raise RevisionConflictError(row.revision)
        row.revision += 1
        row.updated_at = dt.datetime.now(dt.UTC).isoformat()

    async def get_curation_settings(self, *, fresh: bool = False) -> dict[str, Any]:
        del fresh
        return self._envelope()

    async def update_curation_settings(
        self, defaults: dict[str, str | None], *, expected_revision: int
    ) -> dict[str, Any]:
        self._touch(expected_revision)
        for axis, value in defaults.items():
            if value is None:
                self.defaults.pop(axis, None)
            else:
                self.defaults[axis] = value
        return self._envelope()

    async def load_refresh_state(self):
        from src.services.curation.clustering.refresh_state_store import RefreshState

        row = self._row
        if row.refresh_state is None:
            return RefreshState()
        return RefreshState(**row.refresh_state, revision=row.refresh_revision)

    async def save_refresh_state(self, state):
        from src.services.curation.clustering.refresh_state_store import RefreshState
        from src.services.curation.policy_doc_store import PolicyConflictError

        row = self._row
        if state.revision != row.refresh_revision:
            raise PolicyConflictError(f'stored revision is {row.refresh_revision}')
        row.refresh_revision += 1
        row.refresh_state = state.model_dump(exclude={'revision'})
        return RefreshState(**row.refresh_state, revision=row.refresh_revision)

    async def get_keymap_doc(self):
        from src.services.curation.keymap import KeymapDoc

        row = self._row
        return KeymapDoc(
            overrides=dict(row.keymap),
            revision=row.keymap_revision,
            updated_at=None if row.keymap_revision == 0 else '2026-01-01T00:00:00+00:00',
            is_default=not row.keymap,
        )

    async def save_keymap_doc(self, *, overrides, expected_revision):
        from src.services.curation.keymap import KeymapDoc, load_registry

        row = self._row
        if expected_revision != row.keymap_revision:
            raise RevisionConflictError(row.keymap_revision)
        registry = load_registry()
        row.keymap = {
            aid: combos
            for aid, combos in overrides.items()
            if aid in registry.actions and list(combos) != list(registry.actions[aid].default)
        }
        row.keymap_revision += 1
        return KeymapDoc(
            overrides=dict(row.keymap),
            revision=row.keymap_revision,
            updated_at='2026-01-01T00:00:00+00:00',
            is_default=not row.keymap,
        )

    async def publish_keymap_changed(self, doc):
        from src.services.curation.event_hub import get_event_hub

        get_event_hub().publish(
            {
                'type': 'config.changed',
                'topic': 'config',
                'axis': 'keymap',
                'name': None,
                'config_revision': 0,
                'keymap_revision': doc.revision,
            }
        )

    async def load_run_metadata(self):
        from src.services.curation.embedding_viz_read import ProjectionNotBuiltError

        if self._row.projection is None:
            raise ProjectionNotBuiltError
        return dict(self._row.projection)

    async def save_run_metadata(self, *, scope, cluster_id, n_points, fitted_at):
        from src.services.curation import embedding_viz

        self._row.projection = {
            'projection_version': embedding_viz.VIZ_PROJECTION_VERSION,
            'scope': scope,
            'cluster_id': cluster_id,
            'n_points': n_points,
            'fitted_at': fitted_at,
        }

    async def read_policy(self, key, model, *, retired_fields=None):
        from src.services.curation.policy_doc_store import drop_retired_fields

        stored = drop_retired_fields(key, self.policies.get(key), retired_fields)
        return (model.model_validate(stored) if stored else model()), self._row.revision

    async def write_policy(self, key: str, new_policy: Any, revision: int) -> None:
        from src.services.curation.policy_doc_store import PolicyConflictError

        try:
            self._touch(revision)
        except RevisionConflictError as exc:
            raise PolicyConflictError(str(exc)) from exc
        self.policies[key] = new_policy.model_dump()


_installed: list[MemorySettings] = []


def current() -> MemorySettings:
    """The in-memory settings the running test was given."""
    return _installed[-1]


def install(monkeypatch: pytest.MonkeyPatch) -> MemorySettings:
    """Route the settings store and the policy stores to a fresh in-memory row."""
    memory = MemorySettings()
    _installed.append(memory)
    for module, name, replacement in (
        (
            'src.services.curation.settings_store',
            'get_curation_settings',
            memory.get_curation_settings,
        ),
        (
            'src.services.curation.settings_store',
            'update_curation_settings',
            memory.update_curation_settings,
        ),
        ('src.services.curation.policy_doc_store', 'read_policy', memory.read_policy),
        ('src.services.curation.policy_doc_store', 'write_policy', memory.write_policy),
        ('src.services.curation.ingest_policy_store', 'read_policy', memory.read_policy),
        ('src.services.curation.ingest_policy_store', 'write_policy', memory.write_policy),
        ('src.services.curation.vlm_policy_store', 'read_policy', memory.read_policy),
        ('src.services.curation.vlm_policy_store', 'write_policy', memory.write_policy),
        ('src.services.curation.vlm_rep_claims', 'read_policy', memory.read_policy),
        ('src.services.curation.vlm_rep_claims', 'write_policy', memory.write_policy),
    ):
        monkeypatch.setattr(f'{module}.{name}', replacement)
    for module in ('src.services.curation.keymap', 'src.routers.curation.keymap'):
        monkeypatch.setattr(f'{module}.get_keymap_doc', memory.get_keymap_doc)
        monkeypatch.setattr(f'{module}.save_keymap_doc', memory.save_keymap_doc)
        monkeypatch.setattr(f'{module}.publish_keymap_changed', memory.publish_keymap_changed)
    monkeypatch.setattr(
        'src.services.curation.embedding_viz_read._load_run_metadata', memory.load_run_metadata
    )
    monkeypatch.setattr(
        'src.services.curation.embedding_viz._save_run_metadata', memory.save_run_metadata
    )
    for module in (
        'src.services.curation.clustering.refresh_state_store',
        'scripts.curation.cluster_refresh_daemon',
    ):
        if module in sys.modules or module.startswith('src.'):
            monkeypatch.setattr(f'{module}.load_refresh_state', memory.load_refresh_state)
            monkeypatch.setattr(f'{module}.save_refresh_state', memory.save_refresh_state)
    return memory


@pytest.fixture
def memory_settings(monkeypatch: pytest.MonkeyPatch) -> MemorySettings:
    return install(monkeypatch)


@pytest.fixture(autouse=True)
def _offline_settings_are_in_memory(request: pytest.FixtureRequest) -> None:
    """Every test outside the ``db`` tier reads and writes the in-memory settings row; a ``db`` test
    uses the database."""
    if request.node.get_closest_marker('db') is None:
        request.getfixturevalue('memory_settings')


def revision_aware(client: Any) -> Any:
    """Make ``client.put('.../settings', json={'defaults': ...})`` carry the revision it just read.

    The route requires ``expected_revision``; a test about validation or about what a stored default
    changes should not have to thread it through every call.
    """
    plain_put = client.put

    def put(url: str, *args: Any, **kwargs: Any) -> Any:
        body = kwargs.get('json')
        if url.endswith('/settings') and isinstance(body, dict) and 'expected_revision' not in body:
            revision = client.get(url).json()['revision']
            kwargs['json'] = {**body, 'expected_revision': revision}
        return plain_put(url, *args, **kwargs)

    client.put = put
    return client
