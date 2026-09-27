"""Shared structured-error and validation models for the config-store /
projects surface (P1 creates this module ahead of W2, because P1 lands
first in the merge order -- see
docs/design/openprocessor_internal/projects_plan.md §7 and
any_domain_plan.md §7.1/§9 W2). W2 extends the Literals and adds the
validation-report shapes and ``ActiveRef``/``ActiveConfigResponse``;
W3/W4/W8/W9 extend the Literals further as their routes land.

Every project or config route raises through :func:`api_error`, never a
bare string ``detail`` -- gives Cropwright one stable shape
(``{"detail": ConfigErrorDetail}``) to parse everywhere.
"""

from __future__ import annotations

from typing import Any, Literal

from fastapi import HTTPException
from pydantic import BaseModel


# P1 seeded the project-related codes it raises. W2 adds the codes its
# own low-level primitives (``src.services.config_store``) and the
# ``PUT /settings`` bridge raise; W3/W4/W8/W9 add the rest as their
# routes land (see any_domain_plan.md §7.1's full table).
ErrorCode = Literal[
    'project_not_found',
    'project_archived',
    'project_read_only',
    'project_building',
    'project_deleting',
    'project_failed',
    'project_busy',
    'slug_taken',
    'slug_retired',
    'last_active_project',
    'shard_budget_exceeded',
    'project_protected',
    'target_not_empty',
    'preview_stale',
    'in_use',
    'slug_invalid',
    'confirm_mismatch',
    'combine_invalid',
    'clone_source_not_ready',
    'model_name_reserved',
    'internal_isolation_error',
    'revision_conflict',
    # W2
    'not_found',
    'unknown_revision',
    'active_conflict',
    'no_previous',
    'no_active_profile',
    'unknown_pack',
    'unknown_profile',
    'validation_failed',
    'config_store_unavailable',
    'read_only',
    'name_conflict',
    'invalid_transition',
    'export_outside_project',
    'model_not_found',
    # W2b
    'class_hotkey_conflict',
    'hotkey_reserved',
    'hotkey_taken',
]

# Seeded with the codes W2 raises (none yet -- W2 has no validated
# writes of its own, only the low-level OCC primitives). W3/W4 add the
# pack/profile validation codes; W8/W9 add theirs. W2b adds the
# keymap_* codes (CW-K §3.1).
ValidationCode = Literal[
    'name_conflict',
    'keymap_unknown_action',
    'keymap_action_locked',
    'keymap_combo_invalid',
    'keymap_too_many_combos',
    'keymap_key_locked',
    'keymap_browser_reserved',
    'keymap_context_collision',
    'keymap_overlay_unbound',
    'keymap_class_hotkey_conflict',
    'keymap_class_hotkey_shadowed',
    'keymap_focus_key',
    'keymap_context_no_confirm',
]


class ProjectCapacityWire(BaseModel):
    """The OpenSearch shard/heap capacity block (§2.3,
    ``src.services.projects.capacity.ProjectCapacity.to_wire``): served on
    ``GET /projects`` and on a 409 ``shard_budget_exceeded``."""

    status: Literal['ok', 'warn', 'blocked']
    active_shards: int
    per_project_shards: int
    soft_limit: int
    hard_limit: int
    heap_max_bytes: int
    max_shards_per_node: int
    data_nodes: int
    projects_until_soft_limit: int
    message: str
    labels: dict[str, str]


class JobRefWire(BaseModel):
    """One running job blocking a lifecycle action (Cropwright rev-3
    delta 11): typed, not a raw id string, so a caller can render
    "cars has 2 running jobs: Training run run-7, Bake-off run-42"
    without a second lookup."""

    kind: str
    kind_label: str
    id: str
    label: str
    started_at: str


class ConfigErrorDetail(BaseModel):
    """The ``detail`` body of every project/config-store route's 4xx/5xx.

    Optional fields are code-specific and ``None``/absent otherwise; kept
    typed so OpenAPI documents them rather than leaving ``detail`` as an
    opaque blob.
    """

    error: ErrorCode
    message: str
    project: str | None = None
    # delta 11: 409 project_busy carries typed JobRef objects, not raw ids.
    jobs: list[JobRefWire] | None = None
    projects: list[str] | None = None
    active_shards: int | None = None
    needed: int | None = None
    soft_limit: int | None = None
    hard_limit: int | None = None
    heap_max_bytes: int | None = None
    current_revision: int | None = None
    # W2 (activation OCC / validation reports)
    report: ValidationReport | None = None
    current: ActiveRef | None = None
    axis: str | None = None
    valid_ids: list[str] | None = None
    # Delta 12: the full capacity object on shard_budget_exceeded (and in
    # the shard_budget_high warning), so a create form re-renders from
    # one response instead of a second GET /projects.
    capacity: ProjectCapacityWire | None = None
    # invalid_transition: the status the project is in, and the action refused.
    project_status: str | None = None
    action: str | None = None
    # W2b: 409 class_hotkey_conflict / the unbind-and-save response, and
    # 422 hotkey_reserved / 409 hotkey_taken on PUT /classes/{id}.
    class_conflicts: list[dict[str, Any]] | None = None
    unbound_class_hotkeys: list[dict[str, Any]] | None = None
    actions: list[dict[str, Any]] | None = None
    class_id: int | None = None
    class_name: str | None = None


class ApiErrorResponse(BaseModel):
    """The body of every error :func:`api_error` raises."""

    detail: ConfigErrorDetail


def api_error(status: int, code: ErrorCode, message: str, **fields: Any) -> HTTPException:
    """Build ``HTTPException(status, {"detail": ConfigErrorDetail})``."""
    detail = ConfigErrorDetail(error=code, message=message, **fields)
    return HTTPException(status_code=status, detail=detail.model_dump(exclude_none=False))


class ValidationIssue(BaseModel):
    """One error/warning/info from a config validator (§3.3/§4.3)."""

    code: ValidationCode
    severity: Literal['error', 'warning', 'info']
    field: str | None = None
    message: str
    detail: dict[str, Any] = {}
    bypassable: bool = False


class ValidationReport(BaseModel):
    """The result of running a config validator (``/validate``, create,
    clone, PUT, activate). ``force_allowed`` is true only when every
    error present is individually bypassable -- the GUI's "activate
    anyway" affordance."""

    ok: bool
    errors: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []
    force_allowed: bool = False


class ActiveRef(BaseModel):
    """``{name, revision}`` for the currently-active config on one axis.
    ``name=None`` means the axis is off/unconfigured."""

    name: str | None = None
    revision: int | None = None


class ActiveConfigResponse(BaseModel):
    """``GET /prompt_packs/active`` / ``GET /region_profiles/active`` (W3/W4);
    also the activation-mutation response shape used by W2's settings
    bridge and by ``store.activate``'s callers."""

    axis: Literal['prompt_pack', 'detection_profile']
    active: ActiveRef
    previous: ActiveRef | None = None
    config_revision: int
    stale: bool = False


ConfigErrorDetail.model_rebuild()
