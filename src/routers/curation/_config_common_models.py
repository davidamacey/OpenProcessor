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
]

# Seeded with the codes W2 raises (none yet -- W2 has no validated
# writes of its own, only the low-level OCC primitives). W3/W4 add the
# pack/profile validation codes; W8/W9 add theirs.
ValidationCode = Literal['name_conflict']


class ConfigErrorDetail(BaseModel):
    """The ``detail`` body of every project/config-store route's 4xx/5xx.

    Optional fields are code-specific and ``None``/absent otherwise; kept
    typed so OpenAPI documents them rather than leaving ``detail`` as an
    opaque blob.
    """

    error: ErrorCode
    message: str
    project: str | None = None
    jobs: list[str] | None = None
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
