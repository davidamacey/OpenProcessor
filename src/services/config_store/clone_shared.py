"""Shared cross-project clone-source resolution for the prompt-pack and
region-profile CRUD routers (W3/W4 review 2026-09-28, Minor 1).

Both routers' ``POST /{name}/clone`` routes accept an optional
``from_project`` and need the exact same isolation-sensitive dance: resolve
the source project, refuse anything not ``active``/``archived``, and read
the source under a read-only :func:`~src.config.project_context.bind_project`
bind so the guard rejects any accidental write to it. This was previously
copy-pasted verbatim in both routers; factored here so there is exactly one
place that owns "how a cross-project clone read is scoped."
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.routers.curation._config_common_models import api_error


if TYPE_CHECKING:
    from src.config.projects import ProjectRecord
    from src.routers.curation._config_common_models import ValidationReport

_CLONE_SOURCE_READY_STATUSES = frozenset({'active', 'archived'})


def reject_invalid_clone_name(
    report: ValidationReport,
    *,
    new_name: str,
    existing: frozenset[str],
    name_codes: tuple[str, str],
    what: str,
) -> None:
    """M-2 fix (W3/W4 review 2026-09-28): both clone routes must reject a
    ``new_name`` that fails the same slug/reserved-word rules create
    already enforces (the ``pack_name_*`` / ``profile_name_*`` codes),
    then the plain uniqueness check -- shared so both nearly-identical
    checks stay in one place."""
    if any(e.code in name_codes for e in report.errors):
        raise api_error(
            422, 'validation_failed', f'the new {what} name is not usable', report=report
        )
    if new_name in existing:
        raise api_error(409, 'name_conflict', f'{new_name!r} is already taken')


async def resolve_clone_source_project(from_project: str) -> ProjectRecord:
    """The named project, refused (404/409) unless it is ``active`` or
    ``archived``. Raises the same structured errors both clone routers
    used to raise inline."""
    from src.services.projects.lifecycle import _require_found, _resolve_existing

    source_record = _require_found(await _resolve_existing(from_project), from_project)
    if source_record.status not in _CLONE_SOURCE_READY_STATUSES:
        raise api_error(
            409,
            'clone_source_not_ready',
            f"'{from_project}' is {source_record.status}; only an active or "
            'archived project can be cloned from',
            project=from_project,
            project_status=source_record.status,
        )
    return source_record


async def read_source_record(
    *,
    from_project: str | None,
    target_slug: str,
    opensearch: Any,
    resolve: Any,
) -> Any:
    """Resolve the clone source record for either the local-project path
    (``from_project`` unset or equal to the target) or the cross-project
    path (bound read-only for the duration of ``resolve``).

    ``resolve(opensearch)`` is an async callable that reads whichever
    axis-specific record (pack/profile) is being cloned, using
    ``get_config_store()`` for *whichever project is currently bound* --
    the caller doesn't need to know which project that is.

    After a cross-project read, the *target*'s own store is refreshed
    before returning (M-5 fix): `all_known_names()`/OCC checks the caller
    runs next must see the target's current state, not a snapshot from
    before this request bound the source project.
    """
    from src.config.project_context import bind_project
    from src.services.config_store import get_config_store

    if not from_project or from_project == target_slug:
        store = get_config_store()
        await store.ensure_fresh(opensearch)
        return await resolve(opensearch)

    source_record = await resolve_clone_source_project(from_project)
    with bind_project(source_record, read_only=True):
        source_store = get_config_store()
        await source_store.ensure_fresh(opensearch)
        source = await resolve(opensearch)

    # M-5 fix: the target's own store must be current after the
    # cross-project bind exits, not whatever this process last cached --
    # a stored name written by another worker/process is otherwise
    # invisible to the uniqueness check that runs right after this.
    target_store = get_config_store()
    await target_store.ensure_fresh(opensearch)
    return source


def cloned_from_tag(*, source_project: str, name: str, revision: int | None) -> str:
    """``'<project>:<name>@<revision|->'`` provenance tag both clone
    routes stamp on the new record's ``cloned_from``."""
    return f'{source_project}:{name}@{revision if revision is not None else "-"}'


__all__ = ['cloned_from_tag', 'read_source_record', 'resolve_clone_source_project']
