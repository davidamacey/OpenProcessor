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
import os
import threading
import time
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal

from src.core.logging import get_logger
from src.services.config_store.index import (
    ConfigAxis,
    ConfigKind,
    config_doc_id,
    get_activation,
    get_config_revision,
)


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
    # B1 fix: the *activated revision's* body, pinned at activation time --
    # independent of `packs[name]`/`profiles[name]`'s "current" doc, which
    # a later PUT advances without changing what's live. `None` means no
    # resolvable pinned body (never activated / 'off' / copy gone) --
    # callers fall back to the env/file default.
    active_pack_body: StoredConfig | None = None
    active_profile_body: StoredConfig | None = None
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
    # M6: preserve `None` (an env/file id, genuinely never revisioned)
    # rather than coercing it to 0 -- every process must agree on the
    # activation's revision, and a real stored revision is never 0
    # (`_next_revision` starts at 1), so 0 was never a legitimate value
    # here in the first place.
    return (activation_doc['name'], int(revision) if revision is not None else None)


async def _resolve_active_body(
    client: Any,
    index: str,
    *,
    kind: ConfigKind,
    ref: AxisRef,
    current: dict[str, StoredConfig],
) -> StoredConfig | None:
    """B1 fix: the activated ref's *exact* body -- reuse the already-loaded
    "current" doc when its revision matches, else fetch the immutable
    ``<kind>:<name>@<rev>`` copy, so a later PUT (which only advances the
    "current" doc) can never change what this resolves to. ``None`` when
    there's nothing to pin (never activated, 'off', or the pinned copy
    is gone)."""
    if not isinstance(ref, tuple):
        return None
    name, pinned_revision = ref
    local = current.get(name)
    if local is not None and (pinned_revision is None or local.revision == pinned_revision):
        return local
    if pinned_revision is None:
        return None
    try:
        doc = await client.get(index=index, id=config_doc_id(kind, name, pinned_revision))
    except Exception as exc:  # NotFoundError (deleted) or a transient I/O error
        logger.warning(
            'config_store_active_revision_fetch_failed', kind=kind, name=name, error=str(exc)
        )
        return None
    src = doc['_source']
    return StoredConfig(
        kind=kind,
        name=name,
        revision=int(src['revision']),
        body=src.get('body') or {},
        description=src.get('description') or '',
        created_at=src.get('created_at'),
        updated_at=src.get('updated_at'),
        cloned_from=src.get('cloned_from'),
    )


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
        # B5: `revision` above was read with a realtime GET; `_search` is
        # near-real-time and can otherwise return a stale doc set paired
        # with that fresh revision number (a poll landing in the ~1s
        # window after a save+activate). An explicit index refresh
        # forces the search segments current before we read them, so
        # the snapshot this call builds is never cached as "current for
        # revision N" while missing docs that made revision N happen.
        try:
            await client.indices.refresh(index=self.index)
        except Exception as exc:  # pragma: no cover - defensive; search below still runs
            logger.warning('config_store_index_refresh_failed', index=self.index, error=str(exc))
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
        active_pack = _axis_ref(pack_activation)
        active_profile = _axis_ref(profile_activation)
        active_pack_body = await _resolve_active_body(
            client, self.index, kind='prompt_pack', ref=active_pack, current=packs
        )
        active_profile_body = await _resolve_active_body(
            client, self.index, kind='region_profile', ref=active_profile, current=profiles
        )
        return ConfigSnapshot(
            config_revision=revision,
            packs=packs,
            profiles=profiles,
            active_pack=active_pack,
            active_profile=active_profile,
            active_pack_body=active_pack_body,
            active_profile_body=active_profile_body,
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
            active_pack_body=patch.get('active_pack_body', current.active_pack_body),
            active_profile_body=patch.get('active_profile_body', current.active_profile_body),
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
    kind: ConfigKind = 'prompt_pack' if axis == 'prompt_pack' else 'region_profile'
    current_map = store.current.packs if axis == 'prompt_pack' else store.current.profiles
    # B1 fix: pin the exact body being activated now, not a later
    # re-derivation by name that a subsequent PUT could change.
    body_ref = await _resolve_active_body(
        client, store.index, kind=kind, ref=ref, current=current_map
    )
    patch: dict[str, Any] = {'config_revision': result['config_revision']}
    if axis == 'prompt_pack':
        patch['active_pack'] = ref
        patch['active_pack_body'] = body_ref
    else:
        patch['active_profile'] = ref
        patch['active_profile_body'] = body_ref
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


async def _poll_all_active_projects(client: Any, interval: float) -> None:
    """M4: refresh every ACTIVE project's own store on each tick, not
    just the one bound at lifespan startup -- otherwise an activation
    made through uvicorn worker A stays invisible in worker B until
    some other route happens to call that project's ``ensure_fresh()``
    (only the two settings routes did). Cheap: one ``GET`` per active
    project per tick when nothing changed (:meth:`ConfigStore.refresh`
    short-circuits on an unmoved revision counter). A single project's
    refresh failure is logged and never stops the loop or the other
    projects' refreshes -- :meth:`ConfigStore.refresh` already swallows
    its own errors (marks ``stale``), so this only guards the registry
    read + project iteration itself.
    """
    import asyncio

    from src.config.project_context import bind_project
    from src.services.projects.registry import ProjectRegistry

    registry = ProjectRegistry(lambda: client)
    while True:
        try:
            await registry.ensure_fresh()
            for record in registry.active_projects():
                try:
                    with bind_project(record, read_only=True):
                        store = get_config_store(mode='live')
                        await store.refresh(client)
                except Exception as exc:
                    logger.warning(
                        'config_store_poll_project_failed', project=record.slug, error=str(exc)
                    )
        except Exception as exc:
            logger.warning('config_store_poll_registry_failed', error=str(exc))
        await asyncio.sleep(interval)


_CONFIG_STORE_BOOTSTRAP_RETRY_INITIAL_S = 1.0
_CONFIG_STORE_BOOTSTRAP_RETRY_MAX_S = 30.0


async def _bootstrap_config_store_once() -> tuple[Any, float]:
    """Ensure ``op_global_configs`` exists (M3) and resolve the poll
    interval. Raises while OpenSearch is unreachable, or while a lost
    index-create race (``resource_already_exists_exception`` from
    ``indices.create``, another worker having just won it) hasn't yet
    resolved into ``indices.exists`` seeing the winner's index -- the
    caller retries until it doesn't."""
    from src.services.projects.guard import make_curation_opensearch

    client = await make_curation_opensearch()
    # M3: needs no project bound (this index belongs to none) -- runs
    # first, so a later failure below never skips it.
    await ensure_global_configs_index(client)
    interval = float(os.environ.get('OP_CONFIG_POLL_S', str(OP_CONFIG_POLL_S_DEFAULT)))
    return client, interval


async def startup_bootstrap_config_store_safe() -> Any:
    """``src.main``'s lifespan hook: create ``op_global_configs`` (M3) if
    it does not exist yet, then return a background poll task -- fanned
    out over every ACTIVE project (M4), not just the one bound at
    lifespan startup -- that the caller owns cancelling at shutdown.

    Never raises, and always returns a real, running task. If OpenSearch
    is unreachable, or a lost index-create race surfaces as
    ``resource_already_exists_exception`` before ``indices.exists`` sees
    the winner's index, the task keeps retrying
    :func:`_bootstrap_config_store_once` with backoff instead of giving
    up, then enters :func:`_poll_all_active_projects` once it succeeds --
    so that first tick genuinely does eventually run, on every worker,
    every time.

    MJ1 (W2-finish review, 2026-09-27): this used to also call
    ``get_config_store(mode='live')`` + ``store.refresh(client)`` to warm
    "the bound project's store" -- but the lifespan runs unbound by
    design (``src.main``'s startup binds each project in turn only for
    the steps that need it), so that call always raised
    ``ProjectNotBound``. The broad ``except`` below then swallowed it and
    returned ``None`` on every real deployment, so the poll task never
    started at all. There is no bound project to warm here;
    ``_poll_all_active_projects`` binds and refreshes every active
    project on its own first tick.

    MJ3 (W2-finish review, 2026-09-27, fix-on-fix): MJ1's fix still ran
    ``ensure_global_configs_index`` inline before ``create_task``, inside
    a broad ``except`` that returned ``None`` on any failure there --
    including a plain unreachable-OpenSearch startup or a lost
    index-create race, both realistic at cold-stack-start (production
    ``yolo-api`` has no ``opensearch: service_healthy`` gate and runs 32
    workers). Nothing then ever retried, so that worker had no poll task
    for its entire lifetime (M4 inert), contrary to what this docstring
    used to claim. Fixed by mirroring
    ``src.services.projects.bootstrap.startup_bootstrap_project_registry_safe``'s
    retry-then-poll shape: try once inline, and if that fails, hand off
    to a task that keeps retrying with backoff until it succeeds, then
    runs the poll loop forever.
    """
    try:
        client, interval = await _bootstrap_config_store_once()
        return asyncio.create_task(_poll_all_active_projects(client, interval))
    except Exception as exc:
        logger.warning('config_store_bootstrap_deferred', error=str(exc))

    async def _retry_then_poll() -> None:
        delay = _CONFIG_STORE_BOOTSTRAP_RETRY_INITIAL_S
        while True:
            await asyncio.sleep(delay)
            try:
                client, interval = await _bootstrap_config_store_once()
            except Exception as exc:
                logger.warning('config_store_bootstrap_retry_failed', error=str(exc))
                delay = min(delay * 2, _CONFIG_STORE_BOOTSTRAP_RETRY_MAX_S)
                continue
            logger.info('config_store_bootstrap_recovered')
            await _poll_all_active_projects(client, interval)

    return asyncio.create_task(_retry_then_poll())


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


# =============================================================================
# The global (non-project-scoped) config store -- M3
# =============================================================================
#
# any_domain_plan.md / projects_plan.md §11 W2: "ConfigStores: one per
# project plus op_global_configs". Everything above this section is
# per-project (keyed by the bound project's slug, unusable unbound). This
# is its sibling: ONE store for the whole deployment, for a config axis
# that is not scoped to any project at all -- the concrete near-term
# consumer is W9's VLM endpoint registry (a `local_vlm:desired`-style
# global axis), but the mechanism itself is general, the same way
# ``op_projects`` (src.services.projects.registry) is the one other index
# that lives outside every project's own index set.
#
# No CRUD routes read/write this yet (that is W9's job); this pass only
# builds the storage primitive: the index name/bootstrap, and a
# process-singleton :class:`ConfigStore` that never touches
# ``current_project()`` -- calling it while a project happens to be bound
# has no effect on which store it returns, and calling it fully unbound
# (a script, a global route, the lifespan) works with no binding at all.


def global_configs_index() -> str:
    """The one config-store index that belongs to no project -- mirrors
    ``src.services.projects.registry.projects_index()`` for ``op_projects``."""
    return os.environ.get('OP_GLOBAL_CONFIGS_INDEX', 'op_global_configs')


# Same config-store row shapes as a project's folded ``configs`` index
# (``src.clients.curation_opensearch._configs_body``), minus the folded
# SETTINGS/UMAP_VIZ_STATE properties -- the global store never folds any
# other role onto it, so it carries only the config-store fields
# ``src.services.config_store.index``'s primitives read/write (``get``,
# ``save_config``, ``activate``, ``upsert_runtime_doc``, ...). Kept as its
# own literal body (not imported from ``curation_opensearch``) so this
# module stays dependency-light, per its own module docstring.
GLOBAL_CONFIGS_INDEX_BODY: dict[str, Any] = {
    'settings': {'index': {'number_of_shards': 1, 'number_of_replicas': 0}},
    'mappings': {
        'dynamic': False,
        'properties': {
            'doc_type': {'type': 'keyword'},
            'kind': {'type': 'keyword'},
            'name': {'type': 'keyword'},
            'revision': {'type': 'integer'},
            'body': {'type': 'object', 'enabled': False},
            'description': {'type': 'keyword', 'ignore_above': 512, 'index': False},
            'created_at': {'type': 'date'},
            'updated_at': {'type': 'date'},
            'updated_by': {'type': 'keyword'},
            'cloned_from': {'type': 'keyword'},
            'axis': {'type': 'keyword'},
            'previous': {'type': 'object', 'enabled': False},
            'config_revision': {'type': 'long'},
            'process': {'type': 'keyword'},
            'applied_at': {'type': 'date'},
        },
    },
}


async def ensure_global_configs_index(client: Any) -> None:
    """Create ``op_global_configs`` with its explicit mapping if it does
    not exist yet -- mirrors
    ``src.services.projects.bootstrap.ensure_projects_index`` for
    ``op_projects``. Idempotent; call at startup before the first
    global-store read/write. Needs no project bound (this index belongs
    to none)."""
    index = global_configs_index()
    if await client.indices.exists(index=index):
        return
    await client.indices.create(index=index, body=GLOBAL_CONFIGS_INDEX_BODY)


# Cached separately from `_STORES` (never by the same key/dict) so a
# global store instance can never collide with, or be mistaken for, any
# project's own store.
_GLOBAL_STORE: dict[str, ConfigStore] = {}
_GLOBAL_STORE_LOCK = threading.Lock()
_GLOBAL_STORE_KEY = 'op_global_configs'


def get_global_config_store(*, mode: Literal['live', 'pinned'] = 'live') -> ConfigStore:
    """The process's singleton :class:`ConfigStore` for
    ``op_global_configs``.

    Unlike :func:`get_config_store` (keyed by, and unusable without, the
    *bound* project), this never consults
    :func:`~src.config.project_context.current_project` at all -- it
    requires no project binding, and calling it while a project happens
    to be bound has no effect on which store it returns.

    That binding-independence is about which *store object* comes back,
    not its I/O: the object returned here still needs an unbound client
    to actually read/write (m2, W2-finish review) -- calling
    :meth:`ConfigStore.refresh` on it while a project is bound gets
    refused by the project guard (this index is unowned, so a bound
    request has no business touching it) and silently degrades to a
    stale, empty snapshot, the same as any other refresh failure. W9
    (the first real consumer with project-bound call sites) must decide
    the read-while-bound rule -- an unbound-read helper, or a guard
    exception allowing read-only access the way ``op_projects`` gets it.
    """
    with _GLOBAL_STORE_LOCK:
        store = _GLOBAL_STORE.get(_GLOBAL_STORE_KEY)
        if store is None:
            store = ConfigStore(index=global_configs_index(), mode=mode, label='__global__')
            _GLOBAL_STORE[_GLOBAL_STORE_KEY] = store
        return store


def reset_global_config_store() -> None:
    """Test-only: drop the cached global store so a test's fake
    OpenSearch starts from a clean snapshot."""
    with _GLOBAL_STORE_LOCK:
        _GLOBAL_STORE.clear()


__all__ = [
    'GLOBAL_CONFIGS_INDEX_BODY',
    'AxisRef',
    'ConfigSnapshot',
    'ConfigStore',
    'StoredConfig',
    'activate_axis',
    'ensure_global_configs_index',
    'get_config_store',
    'get_global_config_store',
    'global_configs_index',
    'reset_config_stores',
    'reset_global_config_store',
]
