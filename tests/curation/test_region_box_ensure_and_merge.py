"""Guards the earlier suite let through: the ensure step's field set and the
verify merge's lock check."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.clients.curation_opensearch.ensure_fields import ensure_items_region_boxes_fields
from src.config import get_region_fields
from src.services.curation.region_boxes import RegionBox, boxes_write_fields
from src.services.curation.region_verify import BoxVerdict, verify_regions_update


F = get_region_fields()


@pytest.mark.asyncio
async def test_ensure_step_puts_every_region_field_with_the_embedding_geometry() -> None:
    client = MagicMock()
    client.indices.put_mapping = AsyncMock(return_value={'acknowledged': True})

    result = await ensure_items_region_boxes_fields(client)

    put: dict = {}
    for call in client.indices.put_mapping.await_args_list:
        put.update(call.kwargs['body']['properties'])
    assert set(put) == {
        F.boxes,
        F.box_embeddings,
        F.count,
        F.rejected_count,
        F.max_score,
        F.set_complete,
        F.revision,
        F.box_seq,
    }
    assert set(result['fields_added']) == set(put)
    # An existing index lacks this property; only the ensure step adds it.
    assert put[F.box_embeddings]['type'] == 'nested'
    assert put[F.box_embeddings]['properties']['bbox_norm'] == {
        'type': 'float',
        'index': False,
    }


def _item(boxes: list[RegionBox]) -> dict:
    return {'crop_id': 'c', F.status: 'detected', **boxes_write_fields(boxes, current_src={})}


def test_verify_merge_never_rewrites_a_box_locked_after_the_verdict_was_read() -> None:
    human = RegionBox(box_id='b1', bbox_norm=(0.1, 0.1, 0.3, 0.3), state='accepted', source='human')
    machine = RegionBox(box_id='b2', bbox_norm=(0.5, 0.5, 0.7, 0.7), state='proposed')
    verdicts = [
        BoxVerdict(
            box_id=b.box_id,
            bbox_norm=b.bbox_norm,
            is_region=False,
            confidence='high',
            reason='not a region',
        )
        for b in (human, machine)
    ]

    update = verify_regions_update(
        _item([human, machine]), verdicts, now='2026-01-01T00:00:00+00:00', pack_stamp=None
    )

    states = {b['box_id']: b['state'] for b in update[F.boxes]}
    assert states == {'b1': 'accepted', 'b2': 'rejected'}
