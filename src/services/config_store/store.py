"""The cross-process config-store cache: the config epoch plus a
per-process, per-project in-memory snapshot.

A ``ConfigStore`` instance is scoped to one project (or the deployment).
:func:`get_config_store` keys instances by the *currently bound*
project's slug, so every route/task that already binds a project
gets an isolated snapshot with no extra plumbing. The epoch is
``projects.config_epoch`` (``global_config_epoch`` for the deployment): every write bumps it,
and a process reloads its snapshot when the epoch it holds differs.

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
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Literal

from src.core.logging import get_logger
from src.db.repos.config import get_epoch
from src.services.config_store.bound_project import project_id_for
from src.services.config_store.documents import tx
from src.services.config_store.snapshot import _EMPTY_SNAPSHOT, ConfigSnapshot, StoredConfig
from src.services.config_store.snapshot_load import load_snapshot


if TYPE_CHECKING:
    import uuid

    from src.services.config_store.documents import Activated, ConfigAxis, Registered

logger = get_logger(__name__)

# Forced (by-name miss) reads are rate-limited to one per this many seconds per store.
MISS_REFRESH_MIN_INTERVAL_S = 1.0


class ConfigStoreUnavailableError(RuntimeError):
    """The store could not be read; the snapshot is unchanged (not marked stale)."""


class ConfigStore:
    """Per-project, per-process config-store cache. See module docstring."""

    def __init__(
        self,
        *,
        mode: Literal['live', 'pinned'] = 'live',
        label: str = 'default',
        is_global: bool = False,
    ) -> None:
        self.mode = mode
        # The project slug, or ``'__global__'`` for the deployment store.
        self.label = label
        # The one deployment-wide store: loads the endpoint registry instead of a project's
        # packs, profiles and activations.
        self.is_global = is_global
        self.current: ConfigSnapshot = _EMPTY_SNAPSHOT
        self.pending_snapshot: ConfigSnapshot | None = None
        self._lock = asyncio.Lock()
        self._miss_read_at = float('-inf')
        self._miss_read_failed = False

    async def _scope(self) -> uuid.UUID | None:
        return None if self.is_global else await project_id_for(self.label)

    async def _epoch(self) -> int:
        scope = await self._scope()
        async with tx('read config epoch') as session:
            return await get_epoch(session, scope)

    async def refresh(self) -> ConfigSnapshot:
        """Read the epoch; if it changed, reload every config plus the activations and swap
        (``live``) or stage (``pinned``) the new snapshot. A no-op (returns ``current``) when the
        epoch hasn't moved: one indexed read per call in the steady state."""
        async with self._lock:
            return await self._refresh_locked()

    async def refresh_on_miss(self) -> ConfigSnapshot:
        """Re-read after a by-name lookup missed. A write made through another API worker bumps
        the epoch, so the (cheap) epoch is read again; the documents reload only if it moved.
        Concurrent misses queue on the lock and the later ones reuse the first one's outcome
        (reads are at least ``MISS_REFRESH_MIN_INTERVAL_S`` apart), so an unknown-name storm costs
        one epoch read, not one reload each. A failed read raises
        :class:`ConfigStoreUnavailableError` and leaves the snapshot as it was: marking it stale
        would degrade every other reader for one caller's failed lookup. In ``pinned`` mode the
        result is staged, never published, like any refresh."""
        async with self._lock:
            if time.monotonic() - self._miss_read_at < MISS_REFRESH_MIN_INTERVAL_S:
                if self._miss_read_failed:
                    msg = f'config store {self.label} could not be read'
                    raise ConfigStoreUnavailableError(msg)
                return self.current
            try:
                await self._reload_if_moved()
            except Exception as exc:
                logger.warning('config_store_miss_refresh_failed', store=self.label, error=str(exc))
                self._miss_read_failed = True
                msg = f'config store {self.label} could not be read'
                raise ConfigStoreUnavailableError(msg) from exc
            finally:
                self._miss_read_at = time.monotonic()
            self._miss_read_failed = False
            return self.current

    async def _reload_if_moved(self) -> None:
        epoch = await self._epoch()
        known = self.pending_snapshot or self.current
        if epoch == known.config_revision and known.loaded_at and not known.stale:
            return
        self._publish(await self._load())

    def _publish(self, snapshot: ConfigSnapshot) -> None:
        if self.mode == 'live':
            self.current = snapshot
        else:
            self.pending_snapshot = snapshot

    async def _load(self) -> ConfigSnapshot:
        return await load_snapshot(await self._scope())

    async def _refresh_locked(self) -> ConfigSnapshot:
        try:
            epoch = await self._epoch()
        except Exception as exc:
            logger.warning('config_store_refresh_failed', store=self.label, error=str(exc))
            self.current = replace(self.current, stale=True)
            return self.current

        # A stale snapshot is reloaded even at the same epoch: nothing else clears
        # `stale`, so skipping here would leave a recovered store refusing forever.
        unchanged = epoch == self.current.config_revision and self.current.loaded_at
        if unchanged and not self.current.stale:
            return self.current

        try:
            snapshot = await self._load()
        except Exception as exc:
            logger.warning('config_store_refresh_failed', store=self.label, error=str(exc))
            self.current = replace(self.current, stale=True)
            return self.current

        self._publish(snapshot)
        return snapshot

    async def ensure_fresh(self, max_age_s: float = 1.0) -> ConfigSnapshot:
        """Refresh only if the snapshot is older than ``max_age_s``: for
        read paths where a snapshot under a second old is fine (§3.6). A
        route that looks a config up by *name* so a caller can act on what it
        just wrote -- through another API worker -- calls :meth:`refresh`
        instead, which always checks the epoch."""
        if self.current.loaded_at and (time.monotonic() - self.current.loaded_at) < max_age_s:
            return self.current
        return await self.refresh()

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
        ingest_profiles = dict(current.ingest_profiles)
        if 'ingest_profile' in patch:
            ip_item: StoredConfig | None = patch['ingest_profile']
            ip_name = ip_item.name if ip_item is not None else patch['name']
            if ip_item is None:
                ingest_profiles.pop(ip_name, None)
            else:
                ingest_profiles[ip_name] = ip_item
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
            ingest_profiles=ingest_profiles,
            active_ingest_profile=patch.get('active_ingest_profile', current.active_ingest_profile),
            active_ingest_profile_body=patch.get(
                'active_ingest_profile_body', current.active_ingest_profile_body
            ),
        )
        self.current = new_current
        if self.mode == 'pinned':
            self.pending_snapshot = new_current


async def activate_axis(
    store: ConfigStore,
    *,
    axis: ConfigAxis,
    name: str | None,
    revision: int | None,
    expected_active: dict[str, Any] | None = None,
    registered: Registered | None = None,
    publish_event: bool = True,
) -> Activated:
    """Activate (or deactivate, ``name=None``) ``axis`` on ``store``'s
    project: writes through :func:`~src.services.config_store.documents.activate`,
    applies the result to ``store`` immediately (read-after-write in this
    process), and publishes the ``config.changed`` SSE event --
    the one place both effects happen together, so every caller (the
    ``PUT /settings`` bridge and the activate routes) gets
    both for free.
    """
    from src.services.config_store.activation_apply import activate_and_apply

    result = await activate_and_apply(
        store,
        axis=axis,
        name=name,
        revision=revision,
        expected_active=expected_active,
        registered=registered,
    )
    if publish_event:
        publish_config_changed(axis, name, result.epoch)
    return result


def publish_config_changed(axis: str, name: str | None, epoch: int) -> None:
    from src.services.curation.event_hub import get_event_hub

    get_event_hub().publish(
        {
            'type': 'config.changed',
            'topic': 'config',
            'axis': axis,
            'name': name,
            'config_revision': epoch,
        }
    )


_STORES: dict[str, ConfigStore] = {}
_STORES_LOCK = threading.Lock()


def config_store_for(slug: str, *, mode: Literal['live', 'pinned'] = 'live') -> ConfigStore:
    """The process's :class:`ConfigStore` of project ``slug`` (created on first use).

    ``mode`` only takes effect the first time a given slug's store is created in this process:
    a process is consistently one mode (the API is always ``live``, the detection worker always
    ``pinned``).
    """
    with _STORES_LOCK:
        store = _STORES.get(slug)
        if store is None:
            store = ConfigStore(mode=mode, label=slug)
            _STORES[slug] = store
        return store


def get_config_store(*, mode: Literal['live', 'pinned'] = 'live') -> ConfigStore:
    """The process's :class:`ConfigStore` for the *currently bound* project (there is always
    one bound, per the project context's contract)."""
    from src.config import get_curation_config

    return config_store_for(get_curation_config().project_slug, mode=mode)


def reset_config_stores() -> None:
    """Test-only: drop every cached store so a test starts from a clean snapshot."""
    with _STORES_LOCK:
        _STORES.clear()


__all__ = [
    'ConfigStore',
    'ConfigStoreUnavailableError',
    'activate_axis',
    'config_store_for',
    'get_config_store',
    'publish_config_changed',
    'reset_config_stores',
]
