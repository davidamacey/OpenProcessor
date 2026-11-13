"""An empty near-duplicate page says how many members the run could compare, and names the
members it could not (no vector in the level's space), the usual reason a run covers fewer
photos than the project holds."""

from __future__ import annotations

import datetime as dt
import uuid

from src.db.repos.near_dup import RunRow
from src.services.near_duplicates.status import empty_reason_text


def _run(scored: int) -> RunRow:
    now = dt.datetime.now(dt.UTC)
    return RunRow(
        id=uuid.uuid4(),
        project_id=uuid.uuid4(),
        level='image',
        mode='full',
        status='current',
        space='pe@images:frame',
        profile_config_id=None,
        profile_revision=1,
        params={'dedup_cosine': 0.98, 'leak_cosine': 0.9},
        members_scored=scored,
        dedup_groups=0,
        leak_groups=0,
        grouped_members=0,
        largest_group=0,
        oversized_split=0,
        holdout_absorbed=0,
        algorithm_version='nd-1',
        started_at=now,
        finished_at=now,
    )


def test_a_complete_run_names_only_the_cut_and_the_count() -> None:
    assert empty_reason_text(_run(142), 'image', 'dedup', 0) == (
        '0 dedup groups at cosine 0.98 among 142 images'
    )


def test_images_without_a_vector_are_named_with_the_remedy() -> None:
    text = empty_reason_text(_run(20), 'image', 'dedup', 122)
    assert text.startswith(
        '0 dedup groups at cosine 0.98 among 20 images; 122 images have no vector'
    )
    assert 'scope embed' in text
