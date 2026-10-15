"""Readers match the RegionStatus vocabulary and nothing else.

The short names ``pending`` and ``pending_verify`` were retired with the long
forms; writers never emit them, so no reader may widen a query or a rollup to
include them.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from scripts.curation.worker.cascade import _build_pending_query
from src.config import get_region_fields
from src.config.region_state import RegionStatus
from src.services.curation import ops_metrics_refresh, region_eval_data
from src.services.training.preflight_checks import count_pending_ingest


RETIRED = {'pending', 'pending_verify'}


def _status_terms(node: Any) -> list[set[str]]:
    status = get_region_fields().status
    found: list[set[str]] = []
    if isinstance(node, dict):
        terms = node.get('terms')
        if isinstance(terms, dict) and status in terms:
            found.append({str(getattr(v, 'value', v)) for v in terms[status]})
        for value in node.values():
            found.extend(_status_terms(value))
    elif isinstance(node, list):
        for value in node:
            found.extend(_status_terms(value))
    return found


def test_the_worker_pending_query_names_only_current_statuses() -> None:
    (terms,) = _status_terms(_build_pending_query())
    assert terms == {RegionStatus.PENDING_DETECTION.value, RegionStatus.PENDING_VERIFICATION.value}


def test_the_queue_depth_aggregations_name_only_current_statuses() -> None:
    segment, label = _status_terms(ops_metrics_refresh._agg_body())
    assert segment == {RegionStatus.PENDING_DETECTION.value}
    assert label == {RegionStatus.PENDING_VERIFICATION.value}


@pytest.mark.asyncio
async def test_the_preflight_backlog_count_names_only_current_statuses() -> None:
    opensearch = AsyncMock()
    opensearch.count.return_value = {'count': 3}
    assert await count_pending_ingest(opensearch) == 3
    (terms,) = _status_terms(opensearch.count.call_args.kwargs['body'])
    assert not terms & RETIRED


@pytest.mark.parametrize('value', sorted(RETIRED))
def test_a_retired_status_is_not_rewritten_to_a_current_one(value: str) -> None:
    assert region_eval_data.normalize_status(value) == value
