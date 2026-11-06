"""The reaper: stuck-job recovery and retention, one beat tick (02b section 4).

``op.periodic.reaper`` runs every 30 s on queue ``periodic``. Each rule is its own short
transaction in deployment scope and begins with ``pg_try_advisory_xact_lock(REAPER, 0)`` (Postgres
explicit locking docs: transaction-level, released at commit): a tick that cannot take it stops, the
next one runs. Row selects use ``FOR UPDATE SKIP LOCKED`` so two reapers never act on one row, and
every move names the status it expects. Dispatches follow their commit (a worker must find the
row ``queued``), except rule c, whose rows are already committed and stay locked while their
message is sent. Valkey down: rules a, b, d, f, g still run; the marker of rule c reads
``UNKNOWN`` and nothing acts on it; an enqueue that raises leaves the row ``queued`` for rule c.

====  =========================================================================================
a     lease expired: ``running -> interrupted`` (``cancelling -> cancelled``), attempt ``lost``
b     interrupted and ``auto_resume``: requeue (a savepoint per row) or ``failed`` exhausted
c     ``queued`` with no recent dispatch and no marker: send the message again
d     ``queued`` past its max queue wait: ``failed`` (``queue_wait_exceeded``)
e, i  projection tasks without a job: start the sync job of their kind
f     retention
g     expired stream-worker claims
h     a system pause whose cause is gone: requeue
====  =========================================================================================
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from src.clients import task_queue
from src.config import retention
from src.config.advisory_locks import REAPER
from src.config.job_kinds import KINDS
from src.core.exceptions import RequestRejectedError
from src.db.engine import get_sync_engine
from src.db.repos import events as events_repo, job_events, jobs as repo, jobs_reaper as q
from src.db.session import sync_deployment_tx
from src.services.jobs import blockers, dispatch as dispatching
from src.services.jobs.service import insert_job
from src.tasks.runtime.event_wakeup import wake


if TYPE_CHECKING:
    import uuid
    from collections.abc import Callable

logger = logging.getLogger(__name__)

REQUESTER = 'reaper'
EVENT_BATCHES_PER_TICK = 10
# Projection task kind -> (rule, the job kind that syncs it).
PROJECTION_JOBS: dict[str, tuple[str, str]] = {
    'class_rename': ('e', 'class_projection_sync'),
    'class_merge': ('e', 'class_projection_sync'),
    'cluster_partition': ('i', 'cluster_projection_sync'),
    'near_dup_run': ('i', 'near_dup_projection_sync'),
}


def _mark_all(session: Session, job_ids: list[uuid.UUID]) -> None:
    for job_id in job_ids:
        repo.mark_dispatched(session, job_id)


class TickBusyError(Exception):
    """Another reaper tick holds the advisory lock; this one stops."""


class ReaperError(Exception):
    """A rule failed against the database; the tick ran the others and reports the first."""


@dataclass
class ReapReport:
    lost: int = 0
    resumed: int = 0
    exhausted: int = 0
    redispatched: int = 0
    queue_wait_failed: int = 0
    projection_started: int = 0
    system_resumed: int = 0
    claims_deleted: int = 0
    deleted: dict[str, int] = field(default_factory=dict)
    skipped_busy: bool = False

    def acted(self) -> bool:
        counts = (
            self.lost,
            self.resumed,
            self.exhausted,
            self.redispatched,
            self.queue_wait_failed,
            self.projection_started,
            self.system_resumed,
            self.claims_deleted,
            *self.deleted.values(),
        )
        return any(counts)


class Reaper:
    """One instance per tick; the collaborators are arguments so a test can fake the broker."""

    def __init__(
        self,
        *,
        queued_state: Callable[[str], task_queue.QueuedState] = task_queue.queued_state,
        dispatcher: Callable[[str, uuid.UUID], None] = dispatching.dispatch,
        vlm_available: Callable[[Session], bool] = blockers.vlm_available,
        session_factory: Callable[[], Session] | None = None,
    ) -> None:
        self._queued_state = queued_state
        self._dispatch = dispatcher
        self._vlm_available = vlm_available
        self._session_factory = session_factory or (
            lambda: Session(get_sync_engine(), expire_on_commit=False)
        )

    # --- plumbing --------------------------------------------------------------------------

    def _in_tx(self, reason: str, work: Callable[[Session], Any]) -> Any:
        """One rule's transaction: take the tick lock, run, commit."""
        with self._session_factory() as session, sync_deployment_tx(session, f'reaper {reason}'):
            if not q.try_tick_lock(session, REAPER):
                raise TickBusyError
            return work(session)

    def _announce(self, rows: list[repo.JobRow]) -> None:
        for project in {r.project_id for r in rows}:
            wake(project)

    def _dispatch_rows(self, rows: list[repo.JobRow]) -> int:
        """Send the message of each requeued row (after the commit that queued it)."""
        sent: list[uuid.UUID] = []
        for row in rows:
            try:
                self._dispatch(row.kind, row.id)
            except task_queue.QueueUnavailable:
                logger.warning('reaper_dispatch_deferred job_id=%s (rule c will retry)', row.id)
                break
            except KeyError:
                logger.error('reaper_unknown_job_kind job_id=%s kind=%s', row.id, row.kind)
                continue
            sent.append(row.id)
        if sent:
            self._in_tx('record dispatches', lambda s: _mark_all(s, sent))
        return len(sent)

    # --- the tick ---------------------------------------------------------------------------

    def tick(self) -> ReapReport:
        report = ReapReport()
        errors: list[SQLAlchemyError] = []
        rules: tuple[Callable[[ReapReport], None], ...] = (
            self.rule_a,
            self.rule_b,
            self.rule_c,
            self.rule_d,
            self.rule_e,
            self.rule_i,
            self.rule_h,
            self.rule_g,
            self.rule_f,
        )
        for rule in rules:
            try:
                rule(report)
            except TickBusyError:
                report.skipped_busy = True
                return report
            except SQLAlchemyError as exc:
                logger.exception('reaper_rule_failed rule=%s', rule.__name__)
                errors.append(exc)
        if errors:
            raise ReaperError(str(errors[0])) from errors[0]
        return report

    # --- the rules --------------------------------------------------------------------------

    def rule_a(self, report: ReapReport) -> None:
        def work(s: Session) -> list[repo.JobRow]:
            rows = q.lose_expired(s)
            for row in rows:
                job_events.emit_state(s, row)
            return rows

        rows = self._in_tx('a lost leases', work)
        report.lost = len(rows)
        self._announce(rows)

    def rule_b(self, report: ReapReport) -> None:
        def work(s: Session) -> tuple[list[repo.JobRow], list[repo.JobRow]]:
            resumed: list[repo.JobRow] = []
            exhausted: list[repo.JobRow] = []
            for candidate in q.select_resumable(s):
                if candidate.exhausted:
                    row = q.fail_exhausted(s, candidate.id)
                    target = exhausted
                else:
                    row = q.requeue(s, candidate.id, 'interrupted')
                    target = resumed
                    if row is None:
                        logger.info('resume_skipped_group_busy job_id=%s', candidate.id)
                if row is not None:
                    job_events.emit_state(s, row)
                    target.append(row)
            return resumed, exhausted

        resumed, exhausted = self._in_tx('b interrupted jobs', work)
        report.exhausted = len(exhausted)
        self._announce([*resumed, *exhausted])
        report.resumed = self._dispatch_rows(resumed)

    def rule_c(self, report: ReapReport) -> None:
        def work(s: Session) -> int:
            sent = 0
            for row in q.select_undispatched(s):
                if self._queued_state(str(row.id)) is not task_queue.QueuedState.ABSENT:
                    continue  # present: still waiting; unknown: Valkey down, nothing acts on it
                try:
                    self._dispatch(row.kind, row.id)
                except task_queue.QueueUnavailable:
                    logger.warning('reaper_redispatch_failed job_id=%s', row.id)
                    break
                except KeyError:
                    logger.error('reaper_unknown_job_kind job_id=%s kind=%s', row.id, row.kind)
                    continue
                repo.mark_dispatched(s, row.id)
                sent += 1
            return sent

        report.redispatched = self._in_tx('c undispatched jobs', work)

    def rule_d(self, report: ReapReport) -> None:
        def work(s: Session) -> list[repo.JobRow]:
            rows = q.fail_queue_wait_exceeded(s)
            for row in rows:
                job_events.emit_state(s, row)
            return rows

        rows = self._in_tx('d queue wait', work)
        report.queue_wait_failed = len(rows)
        self._announce(rows)

    def rule_e(self, report: ReapReport) -> None:
        report.projection_started += self._start_projection_syncs('e')

    def rule_i(self, report: ReapReport) -> None:
        report.projection_started += self._start_projection_syncs('i')

    def _start_projection_syncs(self, rule: str) -> int:
        kinds_of_rule = {k: job for k, (r, job) in PROJECTION_JOBS.items() if r == rule}
        ready = [k for k, job in kinds_of_rule.items() if job in KINDS]
        if not ready:
            return 0

        def work(s: Session) -> list[repo.JobRow]:
            started: list[repo.JobRow] = []
            for task in q.select_projection_tasks(s, ready):
                job_kind = kinds_of_rule[task.kind]
                params = {**dict(task.payload or {}), 'projection_task_id': str(task.id)}
                try:
                    result = insert_job(
                        s,
                        job_kind,
                        task.project_id,
                        params=params,
                        idempotency_key=f'{rule}:{task.id}',
                        requested_by=REQUESTER,
                    )
                except RequestRejectedError:
                    continue  # the group is busy: the next tick tries again
                except ValueError:
                    logger.exception('projection_task_unstartable task_id=%s', task.id)
                    continue
                q.link_projection_task(s, task.id, result.job.id)
                if result.created:
                    started.append(result.job)
            return started

        started = self._in_tx(f'{rule} projection tasks', work)
        self._announce(started)
        return self._dispatch_rows(started)

    def rule_h(self, report: ReapReport) -> None:
        def work(s: Session) -> list[repo.JobRow]:
            vlm_ok: bool | None = None
            resumed: list[repo.JobRow] = []
            for job in q.select_system_paused(s):
                if job.pause_reason == blockers.PROJECT_PAUSED:
                    ready = job.project_id is None or not blockers.project_is_paused(
                        s, job.project_id
                    )
                else:
                    if vlm_ok is None:
                        vlm_ok = self._vlm_available(s)
                    ready = vlm_ok
                if not ready:
                    continue
                row = q.requeue(s, job.id, 'paused')
                if row is not None:
                    job_events.emit_state(s, row)
                    resumed.append(row)
            return resumed

        resumed = self._in_tx('h system pauses', work)
        self._announce(resumed)
        report.system_resumed = self._dispatch_rows(resumed)

    def rule_g(self, report: ReapReport) -> None:
        report.claims_deleted = self._in_tx('g expired claims', q.delete_expired_claims)

    def rule_f(self, report: ReapReport) -> None:
        deleted = report.deleted
        deleted['events'] = self._delete_events()
        deleted['jobs'] = self._in_tx(
            'f old jobs', lambda s: q.delete_old_terminal_jobs(s, retention.TERMINAL_JOBS_MAX_AGE)
        )
        deleted['worker_capabilities'] = self._in_tx(
            'f stale workers',
            lambda s: q.delete_stale_workers(s, retention.WORKER_CAPABILITIES_MAX_AGE),
        )
        deleted['gpu_devices'] = self._in_tx(
            'f stale gpu devices',
            lambda s: q.delete_stale_gpu_devices(s, retention.GPU_DEVICES_MAX_AGE),
        )

    def _delete_events(self) -> int:
        """Batches of 10,000 rows, each its own transaction, while a batch comes back full."""
        total = 0
        for _ in range(EVENT_BATCHES_PER_TICK):
            count = self._in_tx(
                'f events',
                lambda s: events_repo.delete_older_than(
                    s, retention.EVENTS_MAX_AGE, batch=retention.DELETE_BATCH
                ),
            )
            total += count
            if count < retention.DELETE_BATCH:
                break
        return total


def reap() -> ReapReport:
    """One tick with the production collaborators (what ``op.periodic.reaper`` runs)."""
    report = Reaper().tick()
    if report.skipped_busy:
        logger.info('reaper_tick_skipped another tick holds the lock')
    elif report.acted():
        logger.info('reaper_acted report=%s', report)
    return report
