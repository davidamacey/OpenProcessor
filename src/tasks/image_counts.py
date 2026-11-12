"""Tasks of the per-image count backfill: a thin registration over ``services/curation``."""

from __future__ import annotations

import functools

from src.services.curation import image_counts_backfill
from src.services.curation.job_binding import opensearch_client
from src.tasks.runtime.register import register_job


register_job(
    'image_counts_backfill',
    image_counts_backfill.run,
    client_factories={'opensearch': functools.partial(opensearch_client, 60)},
)
