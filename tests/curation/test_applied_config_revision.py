"""Found live: after any config edit that does not touch a project's
runtime (a keymap save), ``GET .../active`` showed its detection worker as
``lagging`` for good: the worker reported the revision of its last *pinned*
snapshot, and a pinned store only pins at a swap."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from scripts.curation.worker.runtime import RuntimeHolder, applied_config_revision, current_want
from src.services.config_store.store import ConfigSnapshot


def _store(current_revision: int, pending_revision: int | None = None) -> Any:
    return SimpleNamespace(
        current=ConfigSnapshot(config_revision=current_revision),
        pending_snapshot=(
            None if pending_revision is None else ConfigSnapshot(config_revision=pending_revision)
        ),
    )


def test_an_unrelated_edit_does_not_leave_the_runtime_behind() -> None:
    store, registry = _store(5, pending_revision=6), _store(0)
    holder = RuntimeHolder()
    holder.set_synced_refs('alpha', current_want(store, registry))

    assert applied_config_revision(store, registry, holder, 'alpha') == 6


def test_a_change_the_runtime_has_not_applied_is_still_reported_behind() -> None:
    store, registry = _store(5), _store(0)
    holder = RuntimeHolder()
    holder.set_synced_refs('alpha', current_want(store, registry))
    store.pending_snapshot = ConfigSnapshot(config_revision=6, active_profile=('wheel', 1))

    assert applied_config_revision(store, registry, holder, 'alpha') == 5
