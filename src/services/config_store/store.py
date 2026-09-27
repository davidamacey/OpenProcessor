"""The cross-process config-store cache: a global revision counter plus a
per-process, per-project in-memory snapshot (any_domain_plan.md §3.6,
projects_plan.md §11 W2).

A ``ConfigStore`` instance is scoped to one project's ``configs`` index.
:func:`get_config_store` keys instances by the *currently bound*
project's slug, so every route/task that already binds a project (P1)
gets an isolated snapshot with no extra plumbing.

Two modes:
- ``live`` (the API): :meth:`ConfigStore.refresh` publishes the new
  snapshot immediately -- read-after-refresh is exact.
- ``pinned`` (the detection worker): :meth:`refresh` only stages the
  fetched state (``pending_snapshot``); only :meth:`pin_active` swaps
  ``current``, so the worker controls exactly when the active
  profile/pack changes underneath running tasks (see the worker's
  quiesce-and-swap, §4.5).
"""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal

from src.core.logging import get_logger
from src.services.config_store.index import ConfigAxis, get_activation, get_config_revision


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

logger = get_logger(__name__)

#: ``(name, revision)`` = activated; ``None`` = never activated through
#: the store (the axis's env/file/registered default applies);
#: ``'off'`` = explicitly deactivated (``PUT /settings {"defaults": {axis:
#: null}}`` / ``/deactivate``, W4) -- distinct from "never activated" so a
#: deployment can turn an axis off on purpose without it silently
#: reverting to the hardcoded default.
AxisRef = tuple[str, int | None] | Literal['off'] | None


@dataclass(frozen=True)
class StoredConfig:
    """One ``pack:<name>``/``profile:<name>`` doc's current row."""

    kind: str
    name: str
    revision: int
    body: dict[str, Any]
    description: str = ''
    created_at: str | None = None
    updated_at: str | None = None
    cloned_from: str | None = None


@dataclass(frozen=True)
class ConfigSnapshot:
    """An immutable, atomically-swapped view of one project's config
    store. ``stale=True`` means a refresh failed and this is the last
    good snapshot, not "no config" -- see the module docstring's
    failure-mode note."""

    config_revision: int
    packs: dict[str, StoredConfig] = field(default_factory=dict)
    profiles: dict[str, StoredConfig] = field(default_factory=dict)
    active_pack: AxisRef = None
    active_profile: AxisRef = None
    loaded_at: float = 0.0
    stale: bool = False

    def active_ref(self, axis: ConfigAxis) -> AxisRef:
        return self.active_pack if axis == 'prompt_pack' else self.active_profile


_EMPTY_SNAPSHOT = ConfigSnapshot(config_revision=0)


def _axis_ref(activation_doc: dict[str, Any] | None) -> AxisRef:
    if not activation_doc:
        return None
    if not activation_doc.get('name'):
        return 'off'
    revision = activation_doc.get('revision')
    return (activation_doc['name'], int(revision) if revision is not None else 0)


class ConfigStore:
    """Per-project, per-process config-store cache. See module docstring."""

    def __init__(
        self, index: str, *, mode: Literal['live', 'pinned'] = 'live', label: str = 'default'
    ) -> None:
        self.index = index
        self.mode = mode
        self.label = label
        self.current: ConfigSnapshot = _EMPTY_SNAPSHOT
        self.pending_snapshot: ConfigSnapshot | None = None
        self._lock = asyncio.Lock()

    async def refresh(self, client: Any) -> ConfigSnapshot:
        """Fetch the global revision counter; if it changed, reload every
        config doc plus both activations and swap (``live``) or stage
        (``pinned``) the new snapshot. A no-op (returns ``current``) when
        the revision hasn't moved -- one cheap ``GET`` per call in the
        steady state."""
        async with self._lock:
            try:
                revision = await get_config_revision(client, self.index)
            except Exception as exc:
                logger.warning('config_store_refresh_failed', index=self.index, error=str(exc))
                self.current = replace(self.current, stale=True)
                return self.current

            if revision == self.current.config_revision and self.current.loaded_at:
                return self.current

            try:
                snapshot = await self._load_snapshot(client, revision)
            except Exception as exc:
                logger.warning('config_store_refresh_failed', index=self.index, error=str(exc))
                self.current = replace(self.current, stale=True)
                return self.current

            if self.mode == 'live':
                self.current = snapshot
            else:
                self.pending_snapshot = snapshot
            return snapshot

    async def _load_snapshot(self, client: Any, revision: int) -> ConfigSnapshot:
        resp = await client.search(
            index=self.index,
            body={'size': 1000, 'query': {'bool': {'filter': [{'term': {'doc_type': 'config'}}]}}},
        )
        packs: dict[str, StoredConfig] = {}
        profiles: dict[str, StoredConfig] = {}
        for hit in resp['hits']['hits']:
            src = hit['_source']
            item = StoredConfig(
                kind=src['kind'],
                name=src['name'],
                revision=int(src['revision']),
                body=src.get('body') or {},
                description=src.get('description') or '',
                created_at=src.get('created_at'),
                updated_at=src.get('updated_at'),
                cloned_from=src.get('cloned_from'),
            )
            if src['kind'] == 'prompt_pack':
                packs[src['name']] = item
            elif src['kind'] == 'region_profile':
                profiles[src['name']] = item

        pack_activation = await get_activation(client, self.index, 'prompt_pack')
        profile_activation = await get_activation(client, self.index, 'detection_profile')
        return ConfigSnapshot(
            config_revision=revision,
            packs=packs,
            profiles=profiles,
            active_pack=_axis_ref(pack_activation),
            active_profile=_axis_ref(profile_activation),
            loaded_at=time.monotonic(),
            stale=False,
        )

    async def ensure_fresh(self, client: Any, max_age_s: float = 1.0) -> ConfigSnapshot:
        """Refresh only if the snapshot is older than ``max_age_s``. Every
        async route that resolves a pack/profile by name calls this
        first (§3.6)."""
        if self.current.loaded_at and (time.monotonic() - self.current.loaded_at) < max_age_s:
            return self.current
        return await self.refresh(client)

    def pin_active(self) -> ConfigSnapshot:
        """``pinned`` mode only: publish the last-fetched ``pending_snapshot``
        as ``current``. Called by the worker at a quiesce point, never by
        ``refresh`` itself."""
        if self.pending_snapshot is not None:
            self.current = self.pending_snapshot
            self.pending_snapshot = None
        return self.current

    def apply_local(self, **patch: Any) -> None:
        """Apply this process's own just-completed write to ``current``
        immediately, so read-after-write in the same process is exact
        (the next poll cycle would otherwise be up to ``OP_CONFIG_POLL_S``
        stale). Recognized keys: ``pack``, ``profile`` (a
        :class:`StoredConfig` or ``None`` to delete; pair with ``name``
        when deleting), ``active_pack``, ``active_profile``
        (:data:`AxisRef`), ``config_revision``.

        In ``pinned`` mode this updates ``pending_snapshot`` too (the
        worker still only *acts* on it at the next :meth:`pin_active`),
        so a local write is visible to the next refresh diff without
        an extra round trip.
        """
        current = self.current
        packs = dict(current.packs)
        profiles = dict(current.profiles)
        if 'pack' in patch:
            item: StoredConfig | None = patch['pack']
            name = item.name if item is not None else patch['name']
            if item is None:
                packs.pop(name, None)
            else:
                packs[name] = item
        if 'profile' in patch:
            item = patch['profile']
            name = item.name if item is not None else patch['name']
            if item is None:
                profiles.pop(name, None)
            else:
                profiles[name] = item
        new_current = ConfigSnapshot(
            config_revision=patch.get('config_revision', current.config_revision),
            packs=packs,
            profiles=profiles,
            active_pack=patch.get('active_pack', current.active_pack),
            active_profile=patch.get('active_profile', current.active_profile),
            loaded_at=time.monotonic(),
            stale=False,
        )
        self.current = new_current
        if self.mode == 'pinned':
            self.pending_snapshot = new_current

    async def poll_loop(
        self, get_client: Callable[[], Awaitable[Any] | Any], interval: float
    ) -> None:
        """Background refresh loop (API: started from the lifespan next to
        ``arbiter_task``; cancelled on shutdown). ``get_client`` may be a
        sync callable returning a client, or an async one -- both are
        awaited/called fresh each cycle so a client rotated mid-run is
        picked up."""
        while True:
            try:
                client = get_client()
                if hasattr(client, '__await__'):
                    client = await client
                await self.refresh(client)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning('config_store_poll_failed', index=self.index, error=str(exc))
            await asyncio.sleep(interval)


async def activate_axis(
    store: ConfigStore,
    client: Any,
    *,
    axis: ConfigAxis,
    name: str | None,
    revision: int | None,
    expected_active: dict[str, Any] | None = None,
    publish_event: bool = True,
) -> dict[str, Any]:
    """Activate (or deactivate, ``name=None``) ``axis`` on ``store``'s
    project: writes through :func:`~src.services.config_store.index.activate`,
    applies the result to ``store`` immediately (read-after-write in this
    process), and publishes the ``config.changed`` SSE event (§9 W2) --
    the one place both effects happen together, so every caller (the
    ``PUT /settings`` bridge now; W3/W4's activate routes later) gets
    both for free.
    """
    from src.services.config_store.index import activate as _activate

    result = await _activate(
        client,
        store.index,
        axis=axis,
        name=name,
        revision=revision,
        expected_active=expected_active,
    )
    ref: AxisRef = (name, revision) if name else 'off'
    patch: dict[str, Any] = {'config_revision': result['config_revision']}
    if axis == 'prompt_pack':
        patch['active_pack'] = ref
    else:
        patch['active_profile'] = ref
    store.apply_local(**patch)
    if publish_event:
        from src.services.curation.event_hub import get_event_hub

        get_event_hub().publish(
            {
                'type': 'config.changed',
                'topic': 'config',
                'axis': axis,
                'name': name,
                'config_revision': result['config_revision'],
            }
        )
    return result


#: Default interval (seconds) for ``poll_loop`` -- fast enough that a
#: pack/profile activated in one uvicorn worker is visible in every
#: other within one tick, cheap enough (one ``GET`` per project per
#: tick when nothing changed) to run continuously.
OP_CONFIG_POLL_S_DEFAULT = 5.0


async def startup_bootstrap_config_store_safe() -> Any | None:
    """``src.main``'s lifespan hook: refresh the bound project's store
    once, then return a background poll task the caller owns cancelling
    at shutdown. Never raises -- a startup-time OpenSearch hiccup here
    must not block the rest of the app from starting; the next
    request-time ``ensure_fresh()`` call still runs.

    Only polls the project bound *at lifespan startup* (``default``,
    per ``bind_default_for_lifespan``) -- polling every active project
    (projects_plan.md §11 W2's ``_mget`` fan-out) is follow-on work once
    P2/P3's per-project background-task registry exists to drive it.
    """
    import asyncio
    import os

    try:
        from src.services.projects.guard import make_curation_opensearch

        client = await make_curation_opensearch()
        store = get_config_store(mode='live')
        await store.refresh(client)
        interval = float(os.environ.get('OP_CONFIG_POLL_S', str(OP_CONFIG_POLL_S_DEFAULT)))
        return asyncio.create_task(store.poll_loop(lambda: client, interval))
    except Exception as exc:
        logger.warning('config_store_bootstrap_skipped', error=str(exc))
        return None


async def shutdown_config_store_poll(task: Any | None) -> None:
    """Cancel the poll task :func:`startup_bootstrap_config_store_safe`
    started, and swallow the resulting ``CancelledError``."""
    import asyncio
    import contextlib

    if task is None:
        return
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await task


_STORES: dict[str, ConfigStore] = {}
_STORES_LOCK = threading.Lock()


def get_config_store(*, mode: Literal['live', 'pinned'] = 'live') -> ConfigStore:
    """The process's :class:`ConfigStore` for the *currently bound*
    project (P1's ``get_curation_config()``, which resolves
    ``project_slug``/``configs_index`` from whatever project is bound in
    this context -- there is always one bound, per P1's contract).

    ``mode`` only takes effect the first time a given slug's store is
    created in this process -- a process is consistently one mode (the
    API is always ``live``, the detection worker is always ``pinned``).
    """
    from src.config import get_curation_config

    cfg = get_curation_config()
    slug = cfg.project_slug
    index = cfg.configs_index
    with _STORES_LOCK:
        store = _STORES.get(slug)
        if store is None:
            store = ConfigStore(index=index, mode=mode, label=slug)
            _STORES[slug] = store
        return store


def reset_config_stores() -> None:
    """Test-only: drop every cached store so a test's fake OpenSearch
    starts from a clean snapshot."""
    with _STORES_LOCK:
        _STORES.clear()


__all__ = [
    'AxisRef',
    'ConfigSnapshot',
    'ConfigStore',
    'StoredConfig',
    'activate_axis',
    'get_config_store',
    'reset_config_stores',
]
