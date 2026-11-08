"""Tasks of the near-duplicate service: thin registrations over ``services/near_duplicates``."""

from __future__ import annotations

from typing import Any

from src.config.settings import get_settings
from src.services.near_duplicates import jobs
from src.tasks.runtime.register import register_job


async def _opensearch() -> Any:
    from src.services.projects.guard import make_script_opensearch

    return make_script_opensearch([get_settings().opensearch_url], timeout=60)


_CLIENTS = {'opensearch': _opensearch}

register_job('near_dup_score', jobs.score, client_factories=_CLIENTS)
register_job('near_dup_group', jobs.group, client_factories=_CLIENTS)
register_job('near_dup_rescore', jobs.run_rescore, client_factories=_CLIENTS)
register_job('near_dup_projection_sync', jobs.projection_sync, client_factories=_CLIENTS)
