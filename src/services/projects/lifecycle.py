"""Project lifecycle service: create / patch / archive / unarchive /
clone_settings / delete (dry-run and guarded).

See ``docs/design/openprocessor_internal/projects_plan.md`` §4, §7, the
owner decisions in §13 (D4 capacity, D5 ``default`` is undeletable), and
the Cropwright rev-3 deltas 2 (envelope + revision), 3 (list membership,
implemented in P1's ``_project_models.py``), 10 (delete fits a proxy
timeout), 11 (structured refusals) and 12 (full ``capacity`` on the
409). There is no unscoped alias and no back-compat shim anywhere in
this module (owner decision, 2026-09-26): every route this backs lives
only under the global router or a project's scoped prefix.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any

from fastapi import HTTPException

from src.config.project_context import bind_project
from src.config.projects import ProjectRecord, is_valid_slug, resources_for_new
from src.core.logging import get_logger
from src.routers.curation._config_common_models import api_error
from src.routers.curation._project_models import ARCHIVABLE_STATUSES, UNARCHIVABLE_STATUSES
from src.services.projects.capacity import capacity_status
from src.services.projects.registry import (
    RevisionConflictError,
    get_project_registry,
    get_record_with_seq,
    write_record as _raw_write_record,
)


logger = get_logger(__name__)


def _now() -> str:
    return datetime.now(UTC).isoformat()


async def write_record(
    client: Any,
    record: ProjectRecord,
    *,
    if_seq_no: int | None = None,
    if_primary_term: int | None = None,
    op_type: str | None = None,
) -> None:
    """``registry.write_record``, with the storage-level OCC race
    (:class:`RevisionConflictError` -- another writer's bump landed between
    this caller's read and its write) translated into the API's 409
    ``revision_conflict``, exactly like a stale ``expected_revision``
    would be (:func:`_require_revision`). Every lifecycle mutation
    writes through here, never the raw registry function, so a losing
    concurrent writer never silently clobbers or 500s.

    ``op_type='create'`` (M1) is the create path's storage-level guard:
    two concurrent ``POST /projects`` for the same slug race the raw
    ``index`` call itself, not just this process's in-memory snapshot
    check, and the loser gets :class:`RevisionConflictError` here too --
    translated below into 409 ``slug_taken`` rather than
    ``revision_conflict``, since there is no prior revision to conflict
    with."""
    try:
        await _raw_write_record(
            client, record, if_seq_no=if_seq_no, if_primary_term=if_primary_term, op_type=op_type
        )
    except RevisionConflictError as exc:
        if op_type == 'create':
            raise api_error(
                409,
                'slug_taken',
                f"a project named '{record.slug}' already exists",
                project=record.slug,
            ) from exc
        raise api_error(
            409,
            'revision_conflict',
            f"'{record.slug}' was modified by another request; refresh and retry",
            project=record.slug,
        ) from exc


@dataclass(frozen=True)
class JobRef:
    """A running job blocking a lifecycle action (delta 11)."""

    kind: str
    kind_label: str
    id: str
    label: str
    started_at: str | None

    def to_wire(self) -> dict[str, Any]:
        return {
            'kind': self.kind,
            'kind_label': self.kind_label,
            'id': self.id,
            'label': self.label,
            'started_at': self.started_at,
        }


async def _capacity_error_or_warning(
    client: Any,
) -> list[dict[str, Any]]:
    """Runs the §2.3 capacity check for one more project's shards.
    Raises 409 ``shard_budget_exceeded`` when blocked; otherwise returns
    ``[{"code": "shard_budget_high", ...}]`` on warn, else ``[]``."""
    capacity = await capacity_status(client)
    if capacity is None:
        return []
    if capacity.status == 'blocked':
        raise api_error(
            409,
            'shard_budget_exceeded',
            capacity.message,
            active_shards=capacity.active_shards,
            needed=capacity.per_project_shards,
            soft_limit=capacity.soft_limit,
            hard_limit=capacity.hard_limit,
            heap_max_bytes=capacity.heap_max_bytes,
            capacity=capacity.to_wire(),
        )
    if capacity.status == 'warn':
        return [
            {
                'code': 'shard_budget_high',
                'message': capacity.message,
                'detail': capacity.to_wire(),
            }
        ]
    return []


async def _resolve_existing(slug: str) -> ProjectRecord | None:
    registry = get_project_registry()
    await registry.ensure_fresh()
    return registry.get(slug)


def _require_found(record: ProjectRecord | None, slug: str) -> ProjectRecord:
    if record is None or record.status == 'deleted':
        raise api_error(404, 'project_not_found', f"no project named '{slug}'", project=slug)
    return record


async def _get_mutable_record(
    client: Any, slug: str
) -> tuple[ProjectRecord, int | None, int | None]:
    """The stored record plus its OCC seq/term. ``default`` is now an
    ordinary project record (``bootstrap_default_project`` writes it at
    startup like any other project), so there is no synthesis fallback
    here -- a missing doc is a genuine 404."""
    stored, seq, term = await get_record_with_seq(client, slug)
    if stored is not None:
        return stored, seq, term
    raise api_error(404, 'project_not_found', f"no project named '{slug}'", project=slug)


async def _refetch_for_write(
    client: Any,
    slug: str,
    *,
    expect_status: str | frozenset[str] | None = None,
    **fields: Any,
) -> tuple[ProjectRecord, int | None, int | None]:
    """P3F m2: re-read the record fresh right before a status-transition
    write, and build the new doc FROM that fresh read -- never from a
    closure-captured snapshot taken earlier in the same call. The OCC
    ``if_seq_no``/``if_primary_term`` guard alone does not fix the bug
    this closes: it protects the *write* from a stale token (so a real
    race still 409s), but a snapshot record built from before an
    intervening ``await`` (e.g. ``delete_project_finish``'s up-to-60s
    drain wait) would still silently discard a concurrent PATCH's
    ``display_name``/``description`` even though the write itself
    succeeds under a since-refreshed seq/term.

    P3F pass-3 MA1: re-reading fresh right before the write, by itself,
    throws away the one thing OCC actually protects -- a token that is
    ALWAYS current (because it was just read) never conflicts, no matter
    what happened to the record between the caller's own precondition
    checks and this write. That is exactly how a slow ``create`` could
    resurrect a tombstoned slug through the N1 stale-``building`` escape
    hatch: create re-reads fresh right before its ``active`` write, gets
    a seq/term that trivially matches (nothing else is writing at that
    exact instant), and overwrites ``deleted`` with ``active`` because
    nothing ever checked what status the fresh read actually returned.
    ``expect_status`` is the missing check: a caller states the ONE
    status it still owns (e.g. ``'building'`` for create, ``'deleting'``
    for a delete-finish rollback/tombstone), and this raises 409
    ``invalid_transition`` instead of silently building the write from a
    status nobody validated -- never a resurrection, never a duplicate
    finish clobbering someone else's write."""
    stored, seq, term = await get_record_with_seq(client, slug)
    if stored is None:
        raise api_error(404, 'project_not_found', f"no project named '{slug}'", project=slug)
    if expect_status is not None:
        allowed = {expect_status} if isinstance(expect_status, str) else expect_status
        if stored.status not in allowed:
            raise api_error(
                409,
                'invalid_transition',
                f"'{slug}' is no longer '{sorted(allowed)}' (now '{stored.status}'); "
                'refusing to overwrite a status this caller never validated',
                project=slug,
                project_status=stored.status,
            )
    return replace(stored, **fields), seq, term


async def _cleanup_orphaned_by_concurrent_delete(
    client: Any, record: ProjectRecord, exc: HTTPException
) -> None:
    """F2 (known gap, documented rather than fully fixed -- see the P3
    review, "Re-review 2026-09-27, pass 3", item F2): create's own
    ``building``-status write (to ``active`` on success, or ``failed`` on
    a caught exception) can lose the MA1 ``expect_status='building'``
    race in :func:`_refetch_for_write` because a concurrent ``DELETE``
    won it first and tombstoned this slug (fresh status ``'deleted'``).
    That is the correct refusal -- resurrecting a retired slug would be
    worse -- but by the time it fires, this same create may already have
    made up to 7 ``op_prj_<slug>__*`` indexes (:func:`create_project`'s
    ``_ensure_indexes`` step) that are now permanently unreachable: the
    slug is retired forever, so no project-bound path can ever address
    them again. Reproducing this needs a create running past
    ``delete._BUILDING_STALE_SECONDS`` (120s, the N1 stale-``building``
    escape hatch's own threshold) with a user's ``DELETE`` landing inside
    that exact window -- rare, and it costs only shards held forever, not
    a correctness bug (a retired slug can never come back regardless).

    Best-effort cleanup: since the retired slug can never own these
    indexes again, delete them ourselves here rather than leaving them
    for an operator to find manually. Logged at error level either way,
    with the exact index names, so an operator can search for them if
    this cleanup itself fails (e.g. the same transient fault that made
    ``_ensure_indexes`` unreliable in the first place)."""
    detail = exc.detail if isinstance(exc.detail, dict) else {}
    if detail.get('project_status') != 'deleted':
        return
    orphaned = sorted(set(record.resources.indexes.values()))
    logger.error('project_create_orphaned_after_delete', slug=record.slug, indexes=orphaned)
    for name in orphaned:
        try:
            with bind_project(record):
                await client.indices.delete(index=name, ignore=[404])
        except Exception as cleanup_exc:
            logger.warning(
                'project_create_orphan_cleanup_failed',
                slug=record.slug,
                index=name,
                error=str(cleanup_exc),
            )


async def create_project(
    client: Any,
    *,
    slug: str,
    display_name: str,
    description: str = '',
    clone_settings_from: str | None = None,
    clone_axes: list[str] | None = None,
    origin: dict | None = None,
    activate: bool = True,
) -> tuple[ProjectRecord, list[dict[str, Any]]]:
    """§4 ``POST /projects``. Steps: validate → capacity → record
    ``building`` → create indexes/dirs → optional clone → ``active``. A
    failure midway leaves the record ``failed`` with an error, never
    partially ``active``.

    ``origin`` is stored on the record (a combine's target names its job and
    sources). ``activate=False`` stops with the verified record still
    ``building`` for a caller that fills the project first and then calls
    :func:`finish_building`."""
    from src.config.curation import base_curation_config

    if not is_valid_slug(slug):
        raise api_error(422, 'slug_invalid', f"'{slug}' is not a valid project slug", project=slug)

    existing = await _resolve_existing(slug)
    if existing is not None:
        if existing.status == 'deleted':
            raise api_error(
                409,
                'slug_retired',
                f"'{slug}' was used by a deleted project and its slug is retired",
                project=slug,
            )
        raise api_error(409, 'slug_taken', f"a project named '{slug}' already exists", project=slug)

    if clone_settings_from:
        # M7: every refusal a clone can raise -- unknown axis, clone into
        # itself, a source that does not exist or is not ready -- runs
        # BEFORE the first write. A refused clone must burn no slug, hold
        # no shards and leave no dirs (it used to leave the record
        # 'failed' with gamma's indexes already created -- see the P3
        # review's M7/m9).
        from src.services.projects.clone import _validate_clone_source

        await _validate_clone_source(
            client, target_slug=slug, from_slug=clone_settings_from, axes=clone_axes
        )

    warnings = await _capacity_error_or_warning(client)

    now = _now()
    resources = resources_for_new(slug, base_curation_config())
    record = ProjectRecord(
        slug=slug,
        display_name=display_name,
        description=description,
        status='building',
        revision=1,
        created_at=now,
        updated_at=now,
        origin=origin,
        resources=resources,
    )
    try:
        await write_record(client, record, op_type='create')
    except HTTPException:
        # lifecycle.write_record already translated a genuine
        # storage-level create conflict (RevisionConflictError, i.e.
        # this exact slug really was taken by someone else) into this
        # 409 -- propagate untouched, never touch a record that isn't
        # ours.
        raise
    except Exception:
        # N1: anything else here can only mean the storage-level
        # index() call for OUR OWN doc either never landed (nothing to
        # clean up, the slug is simply free again) or landed but the
        # registry's revision bump afterward failed: our own doc now
        # exists, wedged in 'building' with no failure path having run.
        # If it's there, flip it to 'failed' ourselves so a re-create
        # (slug_retired only fires on 'deleted') or an explicit DELETE
        # (the delete.py stale-building escape hatch) can recover it.
        existing, seq, term = await get_record_with_seq(client, slug)
        if existing is not None and existing.status == 'building':
            failed_doc = replace(existing, status='failed', updated_at=_now())
            await write_record(client, failed_doc, if_seq_no=seq, if_primary_term=term)
        raise

    registry = get_project_registry()

    try:
        # B2(a) residual: refresh_strict() raises on any failure (unlike
        # ensure_fresh's swallow-and-log), so a registry that can't see
        # this project yet aborts the create cleanly through the except
        # handler below, instead of proceeding to index creation while
        # the guard still doesn't recognize the new slug.
        await registry.refresh_strict()

        with bind_project(record):
            from src.routers.curation._common import _ensure_indexes

            resources.upload_root.mkdir(parents=True, exist_ok=True)
            resources.project_state_dir.mkdir(parents=True, exist_ok=True)
            await _ensure_indexes(client)
        if clone_settings_from:
            clone_conflicts = await clone_settings(
                client,
                target_record=record,
                from_slug=clone_settings_from,
                axes=clone_axes,
            )
            if clone_conflicts:
                # Review focus-item #5: a dropped keymap conflict used to
                # be logged only -- `keymap_clone_conflicts` on the 201
                # response is always `[]` for a create (that field is the
                # standalone POST clone_settings route's, not create's),
                # so the client had no way to know the clone-time drop
                # happened. Surface one ProjectWarning per conflict
                # (code `keymap_clone_conflict`) on the plain (record,
                # warnings) pair every caller already unpacks.
                logger.warning(
                    'project_create_keymap_clone_conflicts',
                    slug=slug,
                    from_slug=clone_settings_from,
                    conflicts=clone_conflicts,
                )
                for conflict in clone_conflicts:
                    warnings.append(
                        {
                            'code': 'keymap_clone_conflict',
                            'message': (
                                f"keymap action '{conflict['action_id']}' combo "
                                f"'{conflict['combo']}' collides with class "
                                f"'{conflict['class_name']}' (id {conflict['class_id']}) "
                                'so the keymap was not cloned'
                            ),
                        }
                    )
        # Idempotent: a clone that copied 'classes' (or _apply_clone's own
        # no-classes-copied branch) may already have seeded it, but every
        # new project gets one guaranteed call under its own binding.
        from src.services.curation.region_class import ensure_region_class

        with bind_project(record):
            ensure_region_class()

        # B2(a) residual: _ensure_indexes is itself fail-open (every
        # create/migration failure inside it is logged and swallowed, so
        # it never raises) -- verify every index this record claims to
        # own actually exists before ever calling the project 'active'.
        # Without this, a registry that caught up mid-create without
        # this project (e.g. a transient refresh right after the
        # 'building' write) could let create return 'active' with zero
        # real indexes underneath it -- the exact live B2 symptom.
        with bind_project(record, read_only=True):
            missing_indexes = [
                name
                for name in sorted(set(record.resources.indexes.values()))
                if not await client.indices.exists(index=name)
            ]
        if missing_indexes:
            raise RuntimeError(
                f"'{slug}' create verification found missing indexes: {missing_indexes}"
            )
    except Exception as exc:
        logger.error('project_create_failed', slug=slug, error=str(exc))
        # MA1: build the 'failed' write from a FRESH read of OUR OWN
        # 'building' status, never from the closure-captured `record`
        # (stale since before every await above -- index creation,
        # clone, verification). `expect_status='building'` additionally
        # refuses to write 'failed' over a status this handler never
        # validated (e.g. a delete that raced in and already tombstoned
        # this slug via the N1 stale-building escape hatch) -- see
        # MA1's exact resurrection probe in the P3 review.
        try:
            failed, seq, term = await _refetch_for_write(
                client, slug, expect_status='building', status='failed', updated_at=_now()
            )
        except HTTPException as refetch_exc:
            await _cleanup_orphaned_by_concurrent_delete(client, record, refetch_exc)
            raise
        await write_record(client, failed, if_seq_no=seq, if_primary_term=term)
        await registry.ensure_fresh()
        raise

    if not activate:
        return record, warnings

    return await finish_building(client, record), warnings


async def finish_building(client: Any, record: ProjectRecord, *, ok: bool = True) -> ProjectRecord:
    """The ``building`` -> ``active`` (or ``failed``) transition that ends a
    create, and a combine's fill of its target.

    MA1: the ``expect_status='building'`` guard is what stops a slow finish
    from resurrecting a slug a concurrent DELETE already tombstoned -- a
    re-read taken right before this write always has a current seq/term, so
    OCC alone would not catch it. F2 (known gap): see
    ``_cleanup_orphaned_by_concurrent_delete``.
    """
    try:
        finished, seq, term = await _refetch_for_write(
            client,
            record.slug,
            expect_status='building',
            status='active' if ok else 'failed',
            updated_at=_now(),
        )
    except HTTPException as refetch_exc:
        await _cleanup_orphaned_by_concurrent_delete(client, record, refetch_exc)
        raise
    await write_record(client, finished, if_seq_no=seq, if_primary_term=term)
    await get_project_registry().ensure_fresh()

    from src.services.projects.capacity import invalidate_capacity_cache

    invalidate_capacity_cache()  # m7: a create just changed this cluster's shard count
    return finished


async def patch_project(
    client: Any,
    *,
    slug: str,
    display_name: str | None,
    description: str | None,
    expected_revision: int,
) -> ProjectRecord:
    """§4 ``PATCH /projects/{project}``. The slug is immutable; OCC via
    ``expected_revision`` against the record's own ``revision`` field
    (not the registry-wide counter)."""
    record, seq, term = await _get_mutable_record(client, slug)
    _require_revision(record, expected_revision)
    updated = replace(
        record,
        display_name=display_name if display_name is not None else record.display_name,
        description=description if description is not None else record.description,
        revision=record.revision + 1,
        updated_at=_now(),
    )
    await write_record(client, updated, if_seq_no=seq, if_primary_term=term)
    await get_project_registry().ensure_fresh()
    return updated


_KIND_LABELS = {
    'train': 'Training run',
    'bakeoff': 'Bake-off',
    'autolabel': 'Auto-label',
    'export': 'Export',
    'detection_worker': 'Detection worker',
}


async def running_jobs(record: ProjectRecord) -> list[JobRef]:
    """The project's real running-job inventory, via
    ``src.services.projects.busy.running_jobs`` -- the single source of
    truth for "is this project busy" across every job-producing
    subsystem (§5.4). Delete/archive never re-implement their own file
    scan; they only adapt ``busy.JobRef`` (``kind``/``job_id``) to this
    module's wire-shaped ``JobRef`` (delta 11).

    P3F pass-3 m-b: ``label`` is the job source's own genuine human
    label (``busy.JobRef.label``, e.g. a train run's submitted
    ``mlflow_run_name``) when the source recorded one, falling back to
    the internal ``job_id`` ONLY when it didn't -- documented here
    rather than silently treating ``job_id`` as if it were always a
    human-meaningful name."""
    from src.services.projects import busy

    return [
        JobRef(
            kind=j.kind,
            kind_label=_KIND_LABELS.get(j.kind, j.kind),
            id=j.job_id,
            label=j.label or j.job_id,
            started_at=j.started_at,
        )
        for j in busy.running_jobs(record)
    ]


def _other_active_slugs(snapshot: dict[str, ProjectRecord], slug: str) -> list[str]:
    """Every *other* project that is ``active``. Archived, building,
    failed, deleting and deleted projects do not count: none of them can
    take writes. The one rule both the delete dry run and the real
    archive/delete guards use, so the dry run never disagrees."""
    return [s for s, r in snapshot.items() if s != slug and r.status == 'active']


_LAST_ACTIVE_MESSAGE = 'this is the only active project; leave at least one'


async def _last_active_check(record: ProjectRecord) -> None:
    """Refuse an archive or delete that would leave no active project.

    P3F m4 (documented, not fixed): this reads the registry snapshot and
    the caller's own OCC write happens later, so two concurrent
    archives/deletes of the last two active projects can both pass this
    check and both succeed -- the same class of race as two concurrent
    creates racing the capacity check. A real fix needs a single
    serialization point (a lock doc with its own OCC, or a distributed
    lock) this codebase has no infra for yet; every other OCC guard here
    protects one document's own read-modify-write, not an invariant
    spanning every document in the registry. Left as documented
    best-effort until that infra exists."""
    registry = get_project_registry()
    await registry.ensure_fresh()
    if not _other_active_slugs(dict(registry.snapshot()), record.slug):
        raise api_error(409, 'last_active_project', _LAST_ACTIVE_MESSAGE, project=record.slug)


def _require_transition(record: ProjectRecord, action: str, allowed: frozenset[str]) -> None:
    if record.status not in allowed:
        raise api_error(
            409,
            'invalid_transition',
            f"cannot {action} '{record.slug}' while it is {record.status}",
            project=record.slug,
            project_status=record.status,
            action=action,
        )


def _require_revision(record: ProjectRecord, expected_revision: int) -> None:
    if record.revision != expected_revision:
        raise api_error(
            409,
            'revision_conflict',
            f'expected revision {expected_revision}, current is {record.revision}',
            project=record.slug,
            current_revision=record.revision,
        )


async def archive_project(client: Any, *, slug: str, expected_revision: int) -> ProjectRecord:
    """``active`` -> ``archived`` only (409 ``invalid_transition``
    otherwise), never the last active project."""
    record, seq, term = await _get_mutable_record(client, slug)
    _require_transition(record, 'archive', ARCHIVABLE_STATUSES)
    _require_revision(record, expected_revision)
    jobs = await running_jobs(record)
    if jobs:
        raise api_error(
            409,
            'project_busy',
            f"'{slug}' has {len(jobs)} running job(s)",
            project=slug,
            jobs=[j.to_wire() for j in jobs],
        )
    await _last_active_check(record)
    updated = replace(record, status='archived', revision=record.revision + 1, updated_at=_now())
    await write_record(client, updated, if_seq_no=seq, if_primary_term=term)
    await get_project_registry().ensure_fresh()
    return updated


async def unarchive_project(client: Any, *, slug: str, expected_revision: int) -> ProjectRecord:
    """``archived`` -> ``active`` only (409 ``invalid_transition`` otherwise)."""
    record, seq, term = await _get_mutable_record(client, slug)
    _require_transition(record, 'unarchive', UNARCHIVABLE_STATUSES)
    _require_revision(record, expected_revision)
    updated = replace(record, status='active', revision=record.revision + 1, updated_at=_now())
    await write_record(client, updated, if_seq_no=seq, if_primary_term=term)
    await get_project_registry().ensure_fresh()
    return updated


# clone_settings / clone_settings_into live in clone.py (700-LOC
# ratchet); re-exported here so `lifecycle.clone_settings(...)` keeps
# working for every existing caller.
from src.services.projects.clone import clone_settings, clone_settings_into  # noqa: E402,F401

# dry_run_delete / delete_project / delete_project_finish live in
# delete.py (700-LOC ratchet); re-exported here so
# `lifecycle.delete_project(...)` keeps working for every existing caller.
from src.services.projects.delete import (  # noqa: E402,F401
    delete_project,
    delete_project_finish,
    dry_run_delete,
)
