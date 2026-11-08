"""Tasks of the source backfill: a thin registration over ``services/source_backfill``."""

from __future__ import annotations

from typing import Any

from src.config.settings import get_settings
from src.services.source_backfill import job
from src.tasks.runtime.register import register_job


async def _opensearch() -> Any:
    from src.services.projects.guard import make_script_opensearch

    return make_script_opensearch([get_settings().opensearch_url], timeout=60)


register_job('source_backfill', job.run, client_factories={'opensearch': _opensearch})
