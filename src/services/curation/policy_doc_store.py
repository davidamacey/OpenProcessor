"""Optimistic-revision storage of a per-project policy in the project's settings row.

Each policy is its own column of ``project_settings`` (``ingest_policy``, ``vlm_policy``,
``vlm_scope_reps``). A read returns the policy and the row revision it was read at; a write
presents that revision and a concurrent writer of any settings column makes it a
:class:`PolicyConflictError`, so a policy is never built from a stale read.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from src.config import get_curation_config
from src.core.logging import get_logger
from src.db.repos import settings as settings_repo
from src.db.repos.config import RevisionConflictError
from src.db.session import async_session_factory, project_tx
from src.services.config_store.bound_project import bound_project_id


logger = get_logger(__name__)


class PolicyConflictError(Exception):
    """The stored policy changed since the caller read it."""


def drop_retired_fields(
    key: str, stored: dict[str, Any] | None, retired_fields: dict[str, str] | None
) -> dict[str, Any] | None:
    """``stored`` without the fields the model no longer has, each logged with what replaced it."""
    if not stored or not retired_fields:
        return stored
    for name, replacement in retired_fields.items():
        if name in stored:
            stored = {k: v for k, v in stored.items() if k != name}
            logger.warning(
                'retired_policy_field_reset',
                policy=key,
                field=name,
                project=get_curation_config().project_slug,
                use=replacement,
            )
    return stored


async def read_policy[P: BaseModel](
    key: str, model: type[P], *, retired_fields: dict[str, str] | None = None
) -> tuple[P, int]:
    """The stored policy under ``key`` (the model's defaults when never written) and the settings
    revision it was read at (``0`` for a project with no settings row).

    ``retired_fields`` maps a field the model no longer has to what replaced it: a stored value is
    reset (dropped, with a logged warning naming the replacement) instead of failing the read,
    since the strict model would otherwise break every read of the project. The next write removes
    it."""
    project_id = await bound_project_id()
    async with async_session_factory()() as session, project_tx(session, project_id):
        row = await settings_repo.get_settings(session, project_id)
    if row is None:
        return model(), 0
    stored = drop_retired_fields(key, getattr(row, key), retired_fields)
    return (model.model_validate(stored) if stored else model()), row.revision


async def write_policy(key: str, new_policy: Any, revision: int) -> None:
    """Store ``new_policy`` under ``key``, guarded by the settings ``revision`` it was built from.
    A concurrent writer raises :class:`PolicyConflictError`."""
    project_id = await bound_project_id()
    try:
        async with async_session_factory()() as session, project_tx(session, project_id):
            await settings_repo.put_settings(
                session,
                project_id,
                expected_revision=revision,
                changes={key: new_policy.model_dump()},
            )
    except RevisionConflictError as exc:
        raise PolicyConflictError(str(exc)) from exc
