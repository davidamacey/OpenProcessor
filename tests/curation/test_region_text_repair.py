"""Re-deriving stored box text under the region-text validity rules (DQ-B1 repair)."""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest
from _region_profile_fixture import NEUTRAL_REGION_PROFILE

from curation.query_fakes import QueryFakeOpenSearch
from src.config import get_region_fields
from src.config.curation import base_curation_config
from src.services.curation.region_boxes import RegionBox, boxes_write_fields
from src.services.curation.region_text_repair import (
    apply_region_text_repair,
    plan_region_text_repair,
    rederive,
)
from src.services.detection.region_text_rules import RegionTextRules


F = get_region_fields()
INDEX = base_curation_config().items_index
PROFILE = NEUTRAL_REGION_PROFILE
RULES = RegionTextRules.from_profile(PROFILE, prompt_examples=('ABC1234',))


def _vlm_box(vlm: str, ocr: str | None = None, box_id: str = 'b1', **over: Any) -> RegionBox:
    kwargs: dict[str, Any] = {
        'box_id': box_id,
        'bbox_norm': (0.1, 0.1, 0.2, 0.2),
        'state': 'accepted',
        'text': vlm,
        'text_vlm': vlm,
        'text_source': 'vlm',
        'text_confidence': 0.92,
        'text_engine_version': 'vlm-model',
        'text_raw': ocr or vlm,
    }
    if ocr:
        kwargs['text_ocr'] = ocr
        kwargs['text_disagreement'] = True
    kwargs.update(over)
    return RegionBox(**kwargs)


def _item(crop_id: str, *boxes: RegionBox) -> dict[str, Any]:
    return {
        'crop_id': crop_id,
        F.status: 'detected',
        **boxes_write_fields(list(boxes), current_src={}),
    }


class TestRederive:
    def test_placeholder_gives_way_to_the_stored_ocr_reading(self) -> None:
        target = rederive(_vlm_box('ABC123', 'VWY7977'), profile=PROFILE, rules=RULES)
        assert target is not None
        assert target['text'] == 'VWY7977'
        assert target['text_source'] == 'ocr'
        assert target['text_confidence'] is None
        assert target['text_engine_version'] == 'paddleocr_det_trt:1+paddleocr_rec_trt:1'
        assert target['text_choice'] == 'vlm_invalid'
        assert target['text_vlm_invalid'] == 'placeholder'
        assert target['text_disagreement'] is None

    def test_stock_run_without_ocr_clears_the_chosen_text(self) -> None:
        target = rederive(_vlm_box('999'), profile=PROFILE, rules=RULES)
        assert target is not None
        assert target['text'] is None
        assert target['text_source'] is None
        assert target['text_choice'] == 'no_valid_reading'

    def test_valid_vlm_reading_keeps_its_confidence_and_engine(self) -> None:
        target = rederive(_vlm_box('HK2E1K', 'HK2EIK'), profile=PROFILE, rules=RULES)
        assert target is not None
        assert target['text'] == 'HK2E1K'
        assert target['text_source'] == 'vlm'
        assert target['text_confidence'] == 0.92
        assert target['text_engine_version'] == 'vlm-model'
        assert target['text_choice'] == 'vlm_preferred'

    def test_legacy_model_id_source_counts_as_the_vlm_reading(self) -> None:
        box = RegionBox(
            box_id='b1',
            bbox_norm=(0.1, 0.1, 0.2, 0.2),
            state='accepted',
            text='ABC',
            text_source='vlm-model',
            text_confidence=0.7,
        )
        target = rederive(box, profile=PROFILE, rules=RULES)
        assert target is not None
        assert target['text'] is None
        assert target['text_vlm_invalid'] == 'placeholder'

    def test_human_text_is_never_rederived(self) -> None:
        box = _vlm_box('ABC123', text_source='human')
        assert rederive(box, profile=PROFILE, rules=RULES) is None


@pytest.fixture
def fake_os() -> QueryFakeOpenSearch:
    return QueryFakeOpenSearch(
        {
            INDEX: {
                'ph': _item('ph', _vlm_box('ABC123', 'VWY7977')),
                'run': _item('run', _vlm_box('999')),
                'ok': _item('ok', _vlm_box('HK2E1K')),
                'human': _item('human', _vlm_box('ABC123', text_source='human')),
                # Two boxes on one item, each judged on its own reading.
                'two': _item('two', _vlm_box('HK2E1K'), _vlm_box('999', box_id='b2')),
                'none': {'crop_id': 'none', F.status: 'no_region_visible'},
            }
        }
    )


@pytest.mark.asyncio
async def test_plan_is_read_only_and_counts_changes(fake_os: QueryFakeOpenSearch) -> None:
    plan = await plan_region_text_repair(fake_os, index=INDEX, profile=PROFILE, rules=RULES)
    assert plan.scanned == 5
    assert plan.human_skipped == 1
    assert set(plan.changes) == {'ph', 'run', 'ok', 'two'}
    assert set(plan.changes['two']) == {'b1', 'b2'}
    assert plan.text_changed == {'vlm -> ocr': 1, 'vlm -> none': 2}
    assert plan.vlm_invalid == {'placeholder': 1, 'sequence': 2}
    assert fake_os.docs(INDEX)['ph'][F.boxes][0]['text'] == 'ABC123'


@pytest.mark.asyncio
async def test_apply_rewrites_only_the_chosen_text_attributes_of_the_planned_box(
    fake_os: QueryFakeOpenSearch,
) -> None:
    plan = await plan_region_text_repair(fake_os, index=INDEX, profile=PROFILE, rules=RULES)
    result = await apply_region_text_repair(
        fake_os, plan, index=INDEX, profile=PROFILE, rules=RULES
    )
    assert result['updated'] == 4
    docs = fake_os.docs(INDEX)
    ph = docs['ph'][F.boxes][0]
    assert ph['text'] == 'VWY7977'
    assert ph['text_vlm'] == 'ABC123'
    assert ph['text_raw'] == 'VWY7977'
    run = docs['run'][F.boxes][0]
    assert run.get('text') is None
    assert run['text_vlm'] == '999'
    ok = docs['ok'][F.boxes][0]
    assert ok['text'] == 'HK2E1K'
    assert ok['text_choice'] == 'vlm_only'
    assert docs['human'][F.boxes][0]['text'] == 'ABC123'
    # Each box of a two-box item is repaired on its own reading, in one
    # write that bumps the item's revision like any box write.
    assert [b['text'] for b in docs['two'][F.boxes]] == ['HK2E1K', None]
    assert docs['two'][F.revision] == 2
    again = await plan_region_text_repair(fake_os, index=INDEX, profile=PROFILE, rules=RULES)
    assert again.changes == {}


TEXT_FREE = dataclasses.replace(PROFILE, text_reader='none')


def test_text_free_profile_rederives_nothing() -> None:
    assert rederive(_vlm_box('ABC123', 'VWY7977'), profile=TEXT_FREE, rules=RULES) is None


@pytest.mark.asyncio
async def test_text_free_profile_plans_nothing(fake_os: QueryFakeOpenSearch) -> None:
    plan = await plan_region_text_repair(fake_os, index=INDEX, profile=TEXT_FREE, rules=RULES)
    assert plan.scanned == 0
    assert plan.changes == {}
    result = await apply_region_text_repair(
        fake_os, plan, index=INDEX, profile=TEXT_FREE, rules=RULES
    )
    assert result.get('updated', 0) == 0
    assert fake_os.docs(INDEX)['ph'][F.boxes][0]['text'] == 'ABC123'


@pytest.mark.asyncio
async def test_rederive_script_exits_clean_on_a_text_free_profile(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import argparse

    from scripts.curation import rederive_region_text as script

    monkeypatch.setattr(script, 'get_active_region_profile', lambda: TEXT_FREE)
    client = object()
    rc = await script.run(argparse.Namespace(placeholder=[], prompt_pack=None), client)
    assert rc == 0
    assert 'does not read text' in capsys.readouterr().out
