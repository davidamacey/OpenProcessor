"""Re-deriving stored region text under the validity rules (DQ-B1 repair)."""

from __future__ import annotations

from typing import Any

import pytest

from curation.query_fakes import QueryFakeOpenSearch
from src.config import get_curation_config, get_region_fields
from src.services.curation.region_text_repair import (
    apply_region_text_repair,
    plan_region_text_repair,
    rederive,
)
from src.services.detection.reference_profiles import REFERENCE_LICENSE_PLATE_PROFILE
from src.services.detection.region_text_rules import RegionTextRules


F = get_region_fields()
INDEX = get_curation_config().items_index
PROFILE = REFERENCE_LICENSE_PLATE_PROFILE
RULES = RegionTextRules.from_profile(PROFILE, prompt_examples=('ABC1234',))


def _vlm_row(crop_id: str, vlm: str, ocr: str | None = None, **extra: Any) -> dict[str, Any]:
    doc = {
        'crop_id': crop_id,
        F.status: 'detected',
        F.text: vlm,
        F.text_vlm: vlm,
        F.text_source: 'vlm',
        F.text_confidence: 0.92,
        F.text_engine_version: 'vlm-model',
        F.text_raw: ocr or vlm,
    }
    if ocr:
        doc[F.text_ocr] = ocr
        doc[F.text_disagreement] = True
    doc.update(extra)
    return doc


class TestRederive:
    def test_placeholder_gives_way_to_the_stored_ocr_reading(self) -> None:
        target = rederive(_vlm_row('a', 'ABC123', 'VWY7977'), profile=PROFILE, rules=RULES)
        assert target is not None
        assert target[F.text] == 'VWY7977'
        assert target[F.text_source] == 'ocr'
        assert target[F.text_confidence] is None
        assert target[F.text_engine_version] == 'paddleocr_det_trt:1+paddleocr_rec_trt:1'
        assert target[F.text_choice] == 'vlm_invalid'
        assert target[F.text_vlm_invalid] == 'placeholder'
        assert target[F.text_disagreement] is None

    def test_stock_run_without_ocr_clears_the_chosen_text(self) -> None:
        target = rederive(_vlm_row('a', '999'), profile=PROFILE, rules=RULES)
        assert target is not None
        assert target[F.text] is None
        assert target[F.text_source] is None
        assert target[F.text_choice] == 'no_valid_reading'

    def test_valid_vlm_reading_keeps_its_confidence_and_engine(self) -> None:
        target = rederive(_vlm_row('a', 'HK2E1K', 'HK2EIK'), profile=PROFILE, rules=RULES)
        assert target is not None
        assert target[F.text] == 'HK2E1K'
        assert target[F.text_source] == 'vlm'
        assert target[F.text_confidence] == 0.92
        assert target[F.text_engine_version] == 'vlm-model'
        assert target[F.text_choice] == 'vlm_preferred'

    def test_legacy_model_id_source_counts_as_the_vlm_reading(self) -> None:
        doc = {F.text: 'ABC', F.text_source: 'vlm-model', F.text_confidence: 0.7}
        target = rederive(doc, profile=PROFILE, rules=RULES)
        assert target is not None
        assert target[F.text] is None
        assert target[F.text_vlm_invalid] == 'placeholder'

    def test_human_text_is_never_rederived(self) -> None:
        doc = _vlm_row('a', 'ABC123')
        doc[F.text_source] = 'human'
        assert rederive(doc, profile=PROFILE, rules=RULES) is None


@pytest.fixture
def fake_os() -> QueryFakeOpenSearch:
    return QueryFakeOpenSearch(
        {
            INDEX: {
                'ph': _vlm_row('ph', 'ABC123', 'VWY7977'),
                'run': _vlm_row('run', '999'),
                'ok': _vlm_row('ok', 'HK2E1K'),
                'human': {
                    'crop_id': 'human',
                    F.text: 'ABC123',
                    F.text_vlm: 'ABC123',
                    F.text_source: 'human',
                },
                'none': {'crop_id': 'none', F.status: 'no_region_visible'},
            }
        }
    )


@pytest.mark.asyncio
async def test_plan_is_read_only_and_counts_changes(fake_os: QueryFakeOpenSearch) -> None:
    plan = await plan_region_text_repair(fake_os, index=INDEX, profile=PROFILE, rules=RULES)
    assert plan.scanned == 4
    assert plan.human_skipped == 1
    assert set(plan.changes) == {'ph', 'run', 'ok'}
    assert plan.text_changed == {'vlm -> ocr': 1, 'vlm -> none': 1}
    assert plan.vlm_invalid == {'placeholder': 1, 'sequence': 1}
    assert fake_os.docs(INDEX)['ph'][F.text] == 'ABC123'


@pytest.mark.asyncio
async def test_apply_rewrites_only_the_chosen_text_fields(fake_os: QueryFakeOpenSearch) -> None:
    plan = await plan_region_text_repair(fake_os, index=INDEX, profile=PROFILE, rules=RULES)
    result = await apply_region_text_repair(
        fake_os, plan, index=INDEX, profile=PROFILE, rules=RULES
    )
    assert result['updated'] == 3
    docs = fake_os.docs(INDEX)
    assert docs['ph'][F.text] == 'VWY7977'
    assert docs['ph'][F.text_vlm] == 'ABC123'
    assert docs['ph'][F.text_raw] == 'VWY7977'
    assert docs['run'].get(F.text) is None
    assert docs['run'][F.text_vlm] == '999'
    assert docs['ok'][F.text] == 'HK2E1K'
    assert docs['ok'][F.text_choice] == 'vlm_only'
    assert docs['human'][F.text] == 'ABC123'
    again = await plan_region_text_repair(fake_os, index=INDEX, profile=PROFILE, rules=RULES)
    assert again.changes == {}
