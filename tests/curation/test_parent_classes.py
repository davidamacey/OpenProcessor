"""``parent_classes`` restricts the region stage to matching items: ingest
seeds only those (``region_seed_status``) and the detection worker only
fetches those (``_build_pending_query``). Matched on ``class_name`` or
``proposal_name``, case-insensitively; an empty set matches every item."""

from __future__ import annotations

from typing import Any

import pytest

from scripts.curation.worker.cascade import _build_pending_query
from src.config import get_region_fields
from src.config.region_state import RegionStatus
from src.services.curation.item_doc import DetectedItem, region_seed_status


@pytest.fixture
def profile_env(monkeypatch: pytest.MonkeyPatch) -> Any:
    from src.services.detection import profile_registry

    def _activate(parent_classes: str) -> None:
        monkeypatch.setenv('OP_REGION_DETECTION_NAME', 'wheel')
        monkeypatch.setenv('OP_REGION_DETECTION_PARENT_CLASSES', parent_classes)
        profile_registry._reset_registry_for_tests()

    yield _activate
    profile_registry._reset_registry_for_tests()


def _item(class_name: str | None = None, proposal_name: str | None = None) -> DetectedItem:
    return DetectedItem(
        bbox_pixel=(0, 0, 10, 10), score=0.9, class_name=class_name, proposal_name=proposal_name
    )


def _parent_clause(query: dict[str, Any]) -> dict[str, Any] | None:
    for clause in query['bool']['filter']:
        should = clause.get('bool', {}).get('should')
        if should and all('term' in c for c in should):
            return clause
    return None


class TestSeedStatus:
    def test_no_profile_seeds_nothing(self) -> None:
        assert region_seed_status() is None
        assert region_seed_status(_item('car')) is None

    @pytest.mark.parametrize(
        ('class_name', 'proposal_name'),
        [('car', None), ('Car', None), (None, 'CAR'), ('truck', 'car'), ('unknown', ' car ')],
    )
    def test_matching_items_are_seeded(
        self, profile_env: Any, class_name: str | None, proposal_name: str | None
    ) -> None:
        profile_env('Car,Bus')
        assert (
            region_seed_status(_item(class_name, proposal_name)) is RegionStatus.PENDING_DETECTION
        )

    def test_non_matching_items_are_not_seeded(self, profile_env: Any) -> None:
        profile_env('car')
        assert region_seed_status(_item('person', 'person')) is None
        assert region_seed_status(_item()) is None
        # Without an item the answer is "the region stage is on".
        assert region_seed_status() is RegionStatus.PENDING_DETECTION

    def test_empty_parent_classes_seed_everything(self, profile_env: Any) -> None:
        profile_env('')
        assert region_seed_status(_item('person')) is RegionStatus.PENDING_DETECTION
        assert region_seed_status(_item()) is RegionStatus.PENDING_DETECTION


class TestPendingQuery:
    def test_parent_classes_filter_both_fields_case_insensitively(self, profile_env: Any) -> None:
        profile_env('Car,bus')
        clause = _parent_clause(_build_pending_query())
        assert clause is not None
        assert clause['bool']['minimum_should_match'] == 1
        terms = {
            (field, spec['value'], spec['case_insensitive'])
            for c in clause['bool']['should']
            for field, spec in c['term'].items()
        }
        assert terms == {
            ('class_name', 'bus', True),
            ('class_name', 'car', True),
            ('proposal_name', 'bus', True),
            ('proposal_name', 'car', True),
        }

    def test_empty_parent_classes_add_no_filter(self, profile_env: Any) -> None:
        profile_env('')
        assert _parent_clause(_build_pending_query()) is None

    def test_no_profile_adds_no_filter(self) -> None:
        assert _parent_clause(_build_pending_query()) is None

    def test_items_seeded_under_an_earlier_scope_are_skipped(self, profile_env: Any) -> None:
        from curation.query_fakes import matches

        status = get_region_fields().status
        pending = {'image_path': '/x.jpg', 'bbox_norm': [0, 0, 1, 1], status: 'pending_detection'}
        profile_env('car')
        query = _build_pending_query()
        assert matches({**pending, 'class_name': 'Car'}, query)
        assert matches({**pending, 'class_name': 'suv', 'proposal_name': 'CAR'}, query)
        assert not matches({**pending, 'class_name': 'person', 'proposal_name': 'person'}, query)


class TestIngestSeeding:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(('parent_classes', 'seeded'), [('WIDGET', True), ('gadget', False)])
    async def test_ingest_seeds_only_in_scope_items(
        self, profile_env: Any, parent_classes: str, seeded: bool
    ) -> None:
        from .test_ingest_service import _jpeg_bytes, _make_service

        profile_env(parent_classes)
        svc, os_fake, _ = _make_service()
        result = await svc.ingest_one(_jpeg_bytes(), '/tmp/photo.jpg')

        status = get_region_fields().status
        [doc] = list(os_fake.items.values())
        assert doc['class_name'] == 'widget'
        assert (status in doc) is seeded
        assert result.n_region_queued == (1 if seeded else 0)
