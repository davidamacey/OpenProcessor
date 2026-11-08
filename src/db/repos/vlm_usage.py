"""``vlm_usage_hourly``: one upsert per answered VLM call, for every caller.

The function takes the caller's sync session inside its project transaction and never commits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import func
from sqlalchemy.dialects.postgresql import insert

from src.db.models.assist_loop import VLM_USAGE_SOURCES, VlmUsageHourly


if TYPE_CHECKING:
    import uuid

    from sqlalchemy.orm import Session

    from src.domain.vlm_usage import VlmUsage


def record_vlm_usage(
    session: Session,
    project_id: uuid.UUID,
    endpoint_config_id: uuid.UUID,
    source: str,
    crops: int,
    usage: VlmUsage,
) -> None:
    """Add one answered call to the hour it was answered in (database time)."""
    if source not in VLM_USAGE_SOURCES:
        raise ValueError(f'unknown vlm usage source {source!r}')
    stmt = insert(VlmUsageHourly).values(
        project_id=project_id,
        endpoint_config_id=endpoint_config_id,
        source=source,
        hour=func.date_trunc('hour', func.now()),
        requests=1,
        crops=crops,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        usage_missing=0 if usage.present else 1,
    )
    session.execute(
        stmt.on_conflict_do_update(
            index_elements=['project_id', 'endpoint_config_id', 'source', 'hour'],
            set_={
                'requests': VlmUsageHourly.requests + stmt.excluded.requests,
                'crops': VlmUsageHourly.crops + stmt.excluded.crops,
                'prompt_tokens': VlmUsageHourly.prompt_tokens + stmt.excluded.prompt_tokens,
                'completion_tokens': (
                    VlmUsageHourly.completion_tokens + stmt.excluded.completion_tokens
                ),
                'usage_missing': VlmUsageHourly.usage_missing + stmt.excluded.usage_missing,
            },
        )
    )
