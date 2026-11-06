"""The config store's immutable snapshot: a stored config row and one project's
atomically-swapped view of every axis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal


if TYPE_CHECKING:
    from src.services.config_store.documents import ConfigAxis

#: ``(name, revision)`` = activated; ``None`` = never activated through
#: the store (the axis's env/file/registered default applies);
#: ``'off'`` = explicitly deactivated (``PUT /settings {"defaults": {axis:
#: null}}`` / ``/deactivate``) -- distinct from "never activated" so a
#: deployment can turn an axis off on purpose without it silently
#: reverting to the hardcoded default.
AxisRef = tuple[str, int | None] | Literal['off'] | None


@dataclass(frozen=True)
class StoredConfig:
    """One stored config at one revision (the current one unless a past revision was asked for)."""

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
    # resolvable pinned body -- callers fall back to the env/file default.
    active_pack_body: StoredConfig | None = None
    active_profile_body: StoredConfig | None = None
    loaded_at: float = 0.0
    stale: bool = False
    # Deployment-wide snapshot: the endpoint registry
    # (current docs), each endpoint's last probe record and the desired
    # local model. Empty on a project's store.
    vlm_endpoints: dict[str, StoredConfig] = field(default_factory=dict)
    vlm_probes: dict[str, dict[str, Any]] = field(default_factory=dict)
    local_vlm_desired: dict[str, Any] | None = None
    # Project snapshot: this project's VLM activation, the activated
    # revision's body pinned from the registry, and the ``name@rev`` refs of
    # external endpoints this project has acknowledged. Unset/empty on the
    # global store.
    active_vlm: AxisRef = None
    active_vlm_body: StoredConfig | None = None
    active_vlm_ack_at: str | None = None
    acked_refs: frozenset[str] = frozenset()
    # Open-vocabulary prompt sets: this project's stored sets and its
    # activation (the activated revision's body pinned, like the other axes).
    open_vocab_sets: dict[str, StoredConfig] = field(default_factory=dict)
    active_open_vocab: AxisRef = None
    active_open_vocab_body: StoredConfig | None = None
    # Ingest detector profiles: this project's stored profiles and its
    # activation (the activated revision's body pinned).
    ingest_profiles: dict[str, StoredConfig] = field(default_factory=dict)
    active_ingest_profile: AxisRef = None
    active_ingest_profile_body: StoredConfig | None = None

    def active_ref(self, axis: ConfigAxis) -> AxisRef:
        if axis == 'vlm':
            return self.active_vlm
        if axis == 'open_vocab':
            return self.active_open_vocab
        if axis == 'ingest_profile':
            return self.active_ingest_profile
        return self.active_pack if axis == 'prompt_pack' else self.active_profile


_EMPTY_SNAPSHOT = ConfigSnapshot(config_revision=0)
