"""The config store's immutable snapshot: a stored config row, one project's
atomically-swapped view of every axis, and the activated-revision body resolution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from opensearchpy.exceptions import NotFoundError

from src.core.logging import get_logger
from src.services.config_store.index import ConfigAxis, ConfigKind, config_doc_id


logger = get_logger(__name__)

#: ``(name, revision)`` = activated; ``None`` = never activated through
#: the store (the axis's env/file/registered default applies);
#: ``'off'`` = explicitly deactivated (``PUT /settings {"defaults": {axis:
#: null}}`` / ``/deactivate``) -- distinct from "never activated" so a
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
    # resolvable pinned body -- callers fall back to the env/file default.
    active_pack_body: StoredConfig | None = None
    active_profile_body: StoredConfig | None = None
    loaded_at: float = 0.0
    stale: bool = False
    # Global (op_global_configs) snapshot: the endpoint registry
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

    def active_ref(self, axis: ConfigAxis) -> AxisRef:
        if axis == 'vlm':
            return self.active_vlm
        if axis == 'open_vocab':
            return self.active_open_vocab
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
    """B1 fix: the activated ref's *exact* body -- reuse the loaded
    "current" doc when its revision matches, else fetch the immutable
    ``<kind>:<name>@<rev>`` copy, so a later PUT never changes what this
    resolves to. ``None`` when there's nothing to pin."""
    if not isinstance(ref, tuple):
        return None
    name, pinned_revision = ref
    local = current.get(name)
    if local is not None and (pinned_revision is None or local.revision == pinned_revision):
        return local
    if pinned_revision is None:
        return None
    # Only a 404 means "nothing to pin"; any other exception
    # must propagate (never fail-open, §3.6) -- `refresh()` then stales.
    try:
        doc = await client.get(index=index, id=config_doc_id(kind, name, pinned_revision))
    except NotFoundError as exc:
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
