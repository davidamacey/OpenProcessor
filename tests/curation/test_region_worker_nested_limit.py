"""The region worker never truncates a box list at the nested-document limit: an item whose
detected boxes would reach it is recorded as ``detection_failed`` with the reason, visibly, and
no box is written (OpenSearch k-NN issue 3623)."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from curation.occ_fakes import make_bulk_response, make_bulk_update_item, make_mget_response
from curation.region_worker_helpers import _make_task
from scripts.curation.worker.bulk_writer import _bulk_update
from src.config import get_region_fields
from src.config.region_state import RegionStatus
from src.domain.nested_limit import ERROR_CODE
from src.domain.region_boxes import RegionBox, new_box_placeholder


pytestmark = pytest.mark.usefixtures('reference_region_profile')

F = get_region_fields()


def _detected(n: int) -> list[RegionBox]:
    return [
        RegionBox(
            box_id=new_box_placeholder(i),
            bbox_norm=(0.01 * i, 0.1, 0.01 * i + 0.1, 0.4),
            state='accepted',
        )
        for i in range(n)
    ]


async def _run(n_boxes: int) -> tuple[int, dict[str, dict[str, Any]]]:
    task = _make_task(crop_id='a')
    task.pending_boxes = _detected(n_boxes)
    task.update_doc = {F.status: 'detected'}
    task.pending_empty_status = RegionStatus.NO_REGION_BOX
    captured: dict[str, dict[str, Any]] = {}

    async def _mget(*, body: dict[str, Any]) -> dict[str, Any]:
        return make_mget_response(
            {d['_id']: {F.status: 'pending', F.boxes: []} for d in body['docs']}
        )

    async def _bulk(*, body: list[dict[str, Any]], **kw: Any) -> dict[str, Any]:
        for action, doc in zip(body[0::2], body[1::2], strict=True):
            captured[action['update']['_id']] = doc['doc']
        return make_bulk_response([make_bulk_update_item(a['update']['_id']) for a in body[0::2]])

    opensearch = MagicMock()
    opensearch.mget = AsyncMock(side_effect=_mget)
    opensearch.bulk = AsyncMock(side_effect=_bulk)
    written, _skipped = await _bulk_update(opensearch, [task])
    return written, captured


@pytest.mark.asyncio
async def test_a_box_list_just_below_the_limit_is_written() -> None:
    written, captured = await _run(74)
    assert written == 1
    assert len(captured['a'][F.boxes]) == 74


@pytest.mark.asyncio
async def test_a_box_list_at_the_limit_is_recorded_as_failed_and_no_box_is_written() -> None:
    written, captured = await _run(75)
    assert written == 1
    assert F.boxes not in captured['a']
    assert captured['a'][F.status] == 'detection_failed'
    assert captured['a'][F.reason] == ERROR_CODE
