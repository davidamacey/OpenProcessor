"""A queued job is served with how many jobs are queued before it on its queue, so a waiting
dialog can say "3 ahead of you" instead of a bare "Working...". Only the single-job route fills
it; the list and every other state serve ``null``."""

from __future__ import annotations

import datetime as dt
import uuid

from src.db.repos.jobs import JobRow
from src.db.repos.jobs_listing import ListedJob
from src.schemas.jobs import JobView
from src.services.jobs.status import to_view


def _listed(status: str) -> ListedJob:
    now = dt.datetime(2026, 11, 12, tzinfo=dt.UTC)
    fields = dict.fromkeys(JobRow.__dataclass_fields__)
    fields.update(
        id=uuid.uuid4(),
        kind='label_op',
        status=status,
        scope='',
        queue='interactive',
        attempt=0,
        retry_count=0,
        max_attempts=3,
        progress={},
        requested_by='t',
        created_at=now,
        queued_at=now,
        updated_at=now,
    )
    return ListedJob(JobRow(**fields), 'default')  # type: ignore[arg-type]  # unused columns stay None


def test_the_view_carries_the_queue_ahead_count() -> None:
    assert 'queue_ahead' in JobView.model_fields
    assert to_view(_listed('queued'), queue_ahead=3).queue_ahead == 3


def test_the_count_defaults_to_null() -> None:
    assert to_view(_listed('queued')).queue_ahead is None
