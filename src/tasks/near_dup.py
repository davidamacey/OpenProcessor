"""Tasks of the near-duplicate service: thin registrations over ``services/near_duplicates``."""

from __future__ import annotations

import functools

from src.services.curation.job_binding import opensearch_client
from src.services.near_duplicates import jobs
from src.tasks.runtime.register import register_job


_CLIENTS = {'opensearch': functools.partial(opensearch_client, 60)}

register_job('near_dup_score', jobs.score, client_factories=_CLIENTS)
register_job('near_dup_group', jobs.group, client_factories=_CLIENTS)
register_job('near_dup_rescore', jobs.run_rescore, client_factories=_CLIENTS)
register_job('near_dup_projection_sync', jobs.projection_sync, client_factories=_CLIENTS)
