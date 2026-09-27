"""Shared structured-error model for the projects surface (P1 creates
this module now, ahead of W2, because P1 lands first in the merge order
-- see docs/design/openprocessor_internal/projects_plan.md §7 and its
note that W2 was originally going to create it).

Every project route raises through :func:`api_error`, never a bare
string ``detail`` -- gives Cropwright one stable shape
(``{"detail": ConfigErrorDetail}``) to parse everywhere.
"""

from __future__ import annotations

from typing import Any, Literal

from fastapi import HTTPException
from pydantic import BaseModel


# P1 seeds only the project-related codes it actually raises, plus
# revision_conflict (P1 doesn't implement PATCH, but the plan's review
# calls out seeding it now since P3's PATCH row already names it and it
# costs nothing to add early).
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
]


class ConfigErrorDetail(BaseModel):
    """The ``detail`` body of every project-route 4xx/5xx.

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
    # Delta 12: the full capacity object on shard_budget_exceeded (and in
    # the shard_budget_high warning), so a create form re-renders from
    # one response instead of a second GET /projects.
    capacity: dict[str, Any] | None = None


def api_error(status: int, code: ErrorCode, message: str, **fields: Any) -> HTTPException:
    """Build ``HTTPException(status, {"detail": ConfigErrorDetail})``."""
    detail = ConfigErrorDetail(error=code, message=message, **fields)
    return HTTPException(status_code=status, detail=detail.model_dump(exclude_none=False))
