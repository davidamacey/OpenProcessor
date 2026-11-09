"""Tasks of the source backfill: a thin registration over ``services/source_backfill``."""

from __future__ import annotations

import functools

from src.services.curation.job_binding import opensearch_client
from src.services.source_backfill import job
from src.tasks.runtime.register import register_job


register_job(
    'source_backfill',
    job.run,
    client_factories={'opensearch': functools.partial(opensearch_client, 60)},
)
