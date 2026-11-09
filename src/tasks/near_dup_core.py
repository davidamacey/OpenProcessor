"""Task of the core near-duplicate grouping: a thin registration over ``services/near_duplicates``."""

from __future__ import annotations

from typing import Any

from src.config.settings import get_settings
from src.services.near_duplicates import core_group
from src.tasks.runtime.register import register_job


async def _opensearch() -> Any:
    from src.services.projects.guard import make_script_opensearch

    return make_script_opensearch([get_settings().opensearch_url], timeout=60)


register_job('near_dup_core_group', core_group.run, client_factories={'opensearch': _opensearch})
