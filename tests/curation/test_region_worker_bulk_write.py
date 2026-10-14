"""Bulk write: one bulk call per iteration with one update and doc pair per crop that produced an update."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from curation.occ_fakes import make_bulk_response, make_bulk_update_item, make_mget_response
from curation.region_worker_helpers import _make_task
from scripts.curation.worker.bulk_writer import _bulk_update
from src.config import get_region_fields
from src.services.curation.class_write_guard import class_state_token


# The cascade needs an active region profile; the default is none.
pytestmark = pytest.mark.usefixtures('reference_region_profile')


# =============================================================================
# Bulk write
# =============================================================================


class TestBulkWrite:
    @pytest.mark.asyncio
    async def test_bulk_write_batches_only_updated_crops(self) -> None:
        """``_bulk_update`` issues ONE batched mget + ONE bulk call (P2-13)
        covering every crop with a non-empty ``update_doc``; tasks with no
        update doc never even reach the mget/bulk round trip."""
        F = get_region_fields()
        a = _make_task(crop_id='a')
        a.update_doc = {F.status: 'detected', F.max_score: 0.9}
        b = _make_task(crop_id='b')
        # No update — should be skipped.
        c = _make_task(crop_id='c')
        c.update_doc = {F.status: 'no_region_box'}

        async def _fake_mget(*, body: dict[str, Any]) -> dict[str, Any]:
            # Docs are still in the pending state the tasks were fetched in.
            found: dict[str, dict[str, Any]] = {
                d['_id']: {F.status: 'pending'} for d in body['docs']
            }
            return make_mget_response(found)

        async def _fake_bulk(*, body: list[dict[str, Any]], **kw: Any) -> dict[str, Any]:
            items = [
                make_bulk_update_item(action['update']['_id'], status=200) for action in body[0::2]
            ]
            return make_bulk_response(items)

        opensearch = MagicMock()
        opensearch.mget = AsyncMock(side_effect=_fake_mget)
        opensearch.bulk = AsyncMock(side_effect=_fake_bulk)

        n_written, n_skipped = await _bulk_update(opensearch, [a, b, c])
        assert n_written == 2
        assert n_skipped == 1
        # One batched mget + one batched bulk for the 2 crops with a
        # non-empty update_doc — not 2 round trips each.
        assert opensearch.mget.await_count == 1
        assert opensearch.bulk.await_count == 1
        mget_call = opensearch.mget.await_args
        assert mget_call is not None
        mget_ids = sorted(d['_id'] for d in mget_call.kwargs['body']['docs'])
        assert mget_ids == ['a', 'c']
        bulk_call = opensearch.bulk.await_args
        assert bulk_call is not None
        bulk_body = bulk_call.kwargs['body']
        bulk_ids = sorted(action['update']['_id'] for action in bulk_body[0::2])
        assert bulk_ids == ['a', 'c']

    @pytest.mark.asyncio
    async def test_bulk_write_skipped_when_no_updates(self) -> None:
        """No tasks need writing → no OS round-trip."""
        opensearch = MagicMock()
        opensearch.mget = AsyncMock()
        opensearch.bulk = AsyncMock()
        n_written, n_skipped = await _bulk_update(opensearch, [_make_task()])
        assert n_written == 0
        assert n_skipped == 1
        opensearch.mget.assert_not_awaited()
        opensearch.bulk.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_vlm_unmatched_write_clears_the_prior_class(self) -> None:
        """IT-2: a vlm_unmatched write must not keep the class_id/class_name
        the VLM's answer just contradicted -- and must reset a class-range
        cluster_id so the residual pass re-clusters the item."""
        F = get_region_fields()
        current_doc = {
            F.status: 'pending',
            'class_id': 5,
            'class_name': 'foo',
            'class_source': 'coco_yolo11_model',
            'cluster_id': 5,
        }
        a = _make_task(crop_id='a', status='pending')
        a.class_token = class_state_token(current_doc)
        a.update_doc = {'class_source': 'vlm_unmatched', 'vlm_raw_class': 'zzz'}

        async def _fake_mget(*, body: dict[str, Any]) -> dict[str, Any]:
            return make_mget_response({d['_id']: current_doc for d in body['docs']})

        captured: dict[str, dict[str, Any]] = {}

        async def _fake_bulk(*, body: list[dict[str, Any]], **kw: Any) -> dict[str, Any]:
            for action, doc in zip(body[0::2], body[1::2], strict=True):
                captured[action['update']['_id']] = doc['doc']
            items = [
                make_bulk_update_item(action['update']['_id'], status=200) for action in body[0::2]
            ]
            return make_bulk_response(items)

        opensearch = MagicMock()
        opensearch.mget = AsyncMock(side_effect=_fake_mget)
        opensearch.bulk = AsyncMock(side_effect=_fake_bulk)

        n_written, n_skipped = await _bulk_update(opensearch, [a])
        assert n_written == 1
        assert n_skipped == 0
        doc = captured['a']
        assert doc['class_id'] is None
        assert doc['class_name'] is None
        assert doc['class_detector'] is None
        assert doc['cluster_id'] == -1
        assert doc['cluster_subid'] is None

    @pytest.mark.asyncio
    async def test_vlm_unmatched_write_leaves_a_candidate_cluster_alone(self) -> None:
        """A candidate (residual) cluster isn't a class-cluster fact --
        clearing the class must not touch it."""
        F = get_region_fields()
        current_doc = {
            F.status: 'pending',
            'class_id': 5,
            'class_name': 'foo',
            'class_source': 'coco_yolo11_model',
            'cluster_id': 10042,
        }
        a = _make_task(crop_id='a', status='pending')
        a.class_token = class_state_token(current_doc)
        a.update_doc = {'class_source': 'vlm_unmatched', 'vlm_raw_class': 'zzz'}

        async def _fake_mget(*, body: dict[str, Any]) -> dict[str, Any]:
            return make_mget_response({d['_id']: current_doc for d in body['docs']})

        captured: dict[str, dict[str, Any]] = {}

        async def _fake_bulk(*, body: list[dict[str, Any]], **kw: Any) -> dict[str, Any]:
            for action, doc in zip(body[0::2], body[1::2], strict=True):
                captured[action['update']['_id']] = doc['doc']
            items = [
                make_bulk_update_item(action['update']['_id'], status=200) for action in body[0::2]
            ]
            return make_bulk_response(items)

        opensearch = MagicMock()
        opensearch.mget = AsyncMock(side_effect=_fake_mget)
        opensearch.bulk = AsyncMock(side_effect=_fake_bulk)

        await _bulk_update(opensearch, [a])
        doc = captured['a']
        assert doc['class_id'] is None
        assert 'cluster_id' not in doc
        assert 'cluster_subid' not in doc

    @pytest.mark.asyncio
    async def test_vlm_unmatched_write_never_clears_a_validated_item(self) -> None:
        F = get_region_fields()
        current_doc = {
            F.status: 'pending',
            'class_id': 5,
            'class_name': 'foo',
            'class_source': 'human',
            'class_validated': True,
            'cluster_id': 5,
        }
        a = _make_task(crop_id='a', status='pending')
        a.class_token = class_state_token(current_doc)
        a.update_doc = {'class_source': 'vlm_unmatched', 'vlm_raw_class': 'zzz'}

        async def _fake_mget(*, body: dict[str, Any]) -> dict[str, Any]:
            return make_mget_response({d['_id']: current_doc for d in body['docs']})

        captured: dict[str, dict[str, Any]] = {}

        async def _fake_bulk(*, body: list[dict[str, Any]], **kw: Any) -> dict[str, Any]:
            for action, doc in zip(body[0::2], body[1::2], strict=True):
                captured[action['update']['_id']] = doc['doc']
            items = [
                make_bulk_update_item(action['update']['_id'], status=200) for action in body[0::2]
            ]
            return make_bulk_response(items)

        opensearch = MagicMock()
        opensearch.mget = AsyncMock(side_effect=_fake_mget)
        opensearch.bulk = AsyncMock(side_effect=_fake_bulk)

        n_written, n_skipped = await _bulk_update(opensearch, [a])
        # class_write_allowed() strips every CLASS_WRITE_FIELDS key before
        # the vlm_unmatched clear even runs, leaving an empty merge result
        # -- a documented noop (neither updated nor skipped), and no bulk
        # round trip at all -- for a validated item's class fields.
        assert n_written == 0
        assert n_skipped == 0
        opensearch.bulk.assert_not_awaited()
        assert 'a' not in captured
