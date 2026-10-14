"""The cross-process config-store cache: a global revision counter plus a
per-process, per-project in-memory snapshot.

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
import contextlib
import threading
import time
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Literal

from opensearchpy.exceptions import OpenSearchException

from src.core.logging import get_logger
from src.services.config_store.index import (
    ConfigAxis,
    forget_absent_revision,
    get_activation,
    get_config_revision,
)
from src.services.config_store.snapshot import (
    _EMPTY_SNAPSHOT,
    ConfigSnapshot,
    StoredConfig,
    _axis_ref,
    _resolve_active_body,
)


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

logger = get_logger(__name__)


class ConfigStore:
    """Per-project, per-process config-store cache. See module docstring."""

    def __init__(
        self,
        index: str,
        *,
        mode: Literal['live', 'pinned'] = 'live',
        label: str = 'default',
        is_global: bool = False,
    ) -> None:
        self.index = index
        self.mode = mode
        self.label = label
        # The one deployment-wide store (op_global_configs): loads the
        # endpoint registry instead of a project's packs/profiles/activations.
        self.is_global = is_global
        self.current: ConfigSnapshot = _EMPTY_SNAPSHOT
        self.pending_snapshot: ConfigSnapshot | None = None
        self._lock = asyncio.Lock()

    async def refresh(self, client: Any, *, force: bool = False) -> ConfigSnapshot:
        """``force`` skips the unchanged-revision shortcut and the absent-
        revision-doc cache: a by-name lookup that missed uses it, because a
        write made through another API worker bumps the counter in a doc this
        process may have cached as absent for a whole poll interval.

        Fetch the global revision counter; if it changed, reload every
        config doc plus both activations and swap (``live``) or stage
        (``pinned``) the new snapshot. A no-op (returns ``current``) when
        the revision hasn't moved -- one cheap ``GET`` per call in the
        steady state."""
        async with self._lock:
            with self._read_scope():
                return await self._refresh_locked(client, force=force)

    def _read_scope(self) -> contextlib.AbstractContextManager[None]:
        """The global store's reads are allowed while a project is bound
        (the project guard otherwise refuses an index that project does not
        own); a project's own store needs no allowance."""
        if not self.is_global:
            return contextlib.nullcontext()
        from src.services.projects.guard import global_configs_read

        return global_configs_read()

    async def _refresh_locked(self, client: Any, *, force: bool = False) -> ConfigSnapshot:
        if force:
            forget_absent_revision(client, self.index)
        try:
            revision = await get_config_revision(client, self.index)
        except Exception as exc:
            logger.warning('config_store_refresh_failed', index=self.index, error=str(exc))
            self.current = replace(self.current, stale=True)
            return self.current

        if not force and revision == self.current.config_revision and self.current.loaded_at:
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
        except OpenSearchException as exc:  # pragma: no cover - defensive; search below still runs
            logger.warning('config_store_index_refresh_failed', index=self.index, error=str(exc))
        resp = await client.search(
            index=self.index,
            body={'size': 1000, 'query': {'bool': {'filter': [{'term': {'doc_type': 'config'}}]}}},
        )
        packs: dict[str, StoredConfig] = {}
        profiles: dict[str, StoredConfig] = {}
        vlm_endpoints: dict[str, StoredConfig] = {}
        open_vocab_sets: dict[str, StoredConfig] = {}
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
            elif src['kind'] == 'vlm_endpoint':
                vlm_endpoints[src['name']] = item
            elif src['kind'] == 'open_vocab_set':
                open_vocab_sets[src['name']] = item

        from src.services.config_store import vlm_snapshot

        if self.is_global:
            vlm_fields = await vlm_snapshot.load_global_fields(client, self.index)
            vlm_fields['vlm_endpoints'] = vlm_endpoints
        else:
            vlm_fields = await vlm_snapshot.load_project_fields(
                client, await get_activation(client, self.index, 'vlm')
            )

        pack_activation = await get_activation(client, self.index, 'prompt_pack')
        profile_activation = await get_activation(client, self.index, 'detection_profile')
        active_pack = _axis_ref(pack_activation)
        active_profile = _axis_ref(profile_activation)
        active_open_vocab = _axis_ref(await get_activation(client, self.index, 'open_vocab'))
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
            open_vocab_sets=open_vocab_sets,
            active_open_vocab=active_open_vocab,
            active_open_vocab_body=await _resolve_active_body(
                client,
                self.index,
                kind='open_vocab_set',
                ref=active_open_vocab,
                current=open_vocab_sets,
            ),
            loaded_at=time.monotonic(),
            stale=False,
            **vlm_fields,
        )

    async def ensure_fresh(self, client: Any, max_age_s: float = 1.0) -> ConfigSnapshot:
        """Refresh only if the snapshot is older than ``max_age_s``: for
        read paths where a snapshot under a second old is fine (§3.6). A
        route that looks a config up by *name* so a caller can act on what it
        just wrote -- through another API worker -- calls :meth:`refresh`
        instead, which always checks the revision counter."""
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
        (:data:`AxisRef`), ``config_revision``, and the VLM fields
        (``vlm_endpoints``, ``vlm_probes``, ``local_vlm_desired``,
        ``active_vlm``, ``active_vlm_body``, ``active_vlm_ack_at``,
        ``acked_refs``), each replacing the whole field.

        In ``pinned`` mode this updates ``pending_snapshot`` too (the
        worker still only *acts* on it at the next :meth:`pin_active`),
        so a local write is visible to the next refresh diff without
        an extra round trip.
        """
        current = self.current
        packs = dict(current.packs)
        profiles = dict(current.profiles)
        open_vocab_sets = dict(current.open_vocab_sets)
        if 'open_vocab_set' in patch:
            ov_item: StoredConfig | None = patch['open_vocab_set']
            ov_name = ov_item.name if ov_item is not None else patch['name']
            if ov_item is None:
                open_vocab_sets.pop(ov_name, None)
            else:
                open_vocab_sets[ov_name] = ov_item
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
            vlm_endpoints=patch.get('vlm_endpoints', current.vlm_endpoints),
            vlm_probes=patch.get('vlm_probes', current.vlm_probes),
            local_vlm_desired=patch.get('local_vlm_desired', current.local_vlm_desired),
            active_vlm=patch.get('active_vlm', current.active_vlm),
            active_vlm_body=patch.get('active_vlm_body', current.active_vlm_body),
            active_vlm_ack_at=patch.get('active_vlm_ack_at', current.active_vlm_ack_at),
            acked_refs=patch.get('acked_refs', current.acked_refs),
            open_vocab_sets=open_vocab_sets,
            active_open_vocab=patch.get('active_open_vocab', current.active_open_vocab),
            active_open_vocab_body=patch.get(
                'active_open_vocab_body', current.active_open_vocab_body
            ),
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
    process), and publishes the ``config.changed`` SSE event --
    the one place both effects happen together, so every caller (the
    ``PUT /settings`` bridge and the activate routes) gets
    both for free.
    """
    from src.services.config_store.activation_apply import activate_and_apply

    result = await activate_and_apply(
        store, client, axis=axis, name=name, revision=revision, expected_active=expected_active
    )
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
    from src.services.config_store.vlm_snapshot import reset_vlm_revision_cache

    with _STORES_LOCK:
        _STORES.clear()
    reset_vlm_revision_cache()


__all__ = [
    'ConfigStore',
    'activate_axis',
    'get_config_store',
    'reset_config_stores',
]
