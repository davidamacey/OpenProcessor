"""A region-text reading that is not text is no reading (DQ-B1).

A VLM echoes its prompt's example value ("ABC1234", or a truncation such
as "ABC123" / "ABC"), answers a "can't read it" word, or a stock run
("999", "123456"). The region-text rules reject those, the chooser then
falls back to a valid OCR reading, and ``region_text_choice`` /
``region_text_vlm_invalid`` record which reading won and why.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

import pytest

from scripts.curation.worker.verify import _region_write_doc
from src.config import DetectionProfile, get_region_fields
from src.services.detection.cascade_detect import RegionCandidate
from src.services.detection.reference_profiles import REFERENCE_LICENSE_PLATE_PROFILE
from src.services.detection.region_text import (
    DominantTextConfig,
    OcrLine,
    read_dominant_text,
    resolve_region_text,
)
from src.services.detection.region_text_rules import RegionTextRules, text_key
from src.services.labeling.vlm_labeler import VlmCombinedReply
from src.services.labeling.vlm_prompts import GENERIC_ITEM_PACK, PromptPack, prompt_text_examples

from .test_region_cascade_integrity import _FakeOpenSearch, _item
from .test_region_text_worker import _drive


if TYPE_CHECKING:
    from pathlib import Path


F = get_region_fields()
REF = REFERENCE_LICENSE_PLATE_PROFILE
REF_RULES = RegionTextRules.from_profile(REF, prompt_examples=('ABC1234', 'DNV20'))
REF_CFG = DominantTextConfig.from_profile(REF)


def _pack_with_example(example: str) -> PromptPack:
    return dataclasses.replace(
        GENERIC_ITEM_PACK,
        name='example_pack',
        region_batch_user=(
            'Respond as a JSON array:\n'
            f'[{{"img": 1, "is_region": true, "text": "{example}" or null}}, ...]'
        ),
    )


class TestPrompts:
    def test_generic_pack_shows_no_example_reading(self) -> None:
        for name, value in dataclasses.asdict(GENERIC_ITEM_PACK).items():
            if isinstance(value, str):
                assert 'ABC' not in value, name
                assert '1234' not in value, name

    def test_generic_pack_quotes_no_example_value(self) -> None:
        # Only enumeration words and a JSON key name the prose mentions.
        assert prompt_text_examples(GENERIC_ITEM_PACK) <= {'high', 'medium', 'low', 'results'}

    def test_examples_are_quoted_values_not_keys_or_placeholders(self) -> None:
        pack = _pack_with_example('XYZ 987')
        pack = dataclasses.replace(
            pack, region_user='e.g. "DNV20" or {"text": "<the text>", "img": 1}'
        )
        examples = prompt_text_examples(pack)
        assert {'XYZ 987', 'DNV20'} <= examples
        assert not {'text', 'img', '<the text>', 'is_region'} & examples


class TestRules:
    @pytest.mark.parametrize('reading', ['ABC1234', 'abc 1234', 'ABC123', 'ABC', 'dnv20'])
    def test_prompt_example_and_its_truncations_are_placeholders(self, reading: str) -> None:
        assert REF_RULES.invalid_reason(reading) == 'placeholder'

    @pytest.mark.parametrize(
        'reading', ['NOT_READABLE', 'NOTAPPLICABLE', 'unreadable', 'N/A', 'null', '---', '']
    )
    def test_no_reading_words(self, reading: str) -> None:
        assert REF_RULES.invalid_reason(reading) == 'no_reading'

    @pytest.mark.parametrize('reading', ['999', '123', '789', '123456', '321', 'XYZ', '111'])
    def test_stock_runs_when_the_profile_rejects_them(self, reading: str) -> None:
        assert REF_RULES.invalid_reason(reading) == 'sequence'

    @pytest.mark.parametrize('reading', ['1987', 'HK2E1K', 'VWY7977', '2BSTUNG', 'AB'])
    def test_real_readings_pass(self, reading: str) -> None:
        assert REF_RULES.invalid_reason(reading) is None

    def test_length_bounds_come_from_the_profile(self) -> None:
        assert REF_RULES.invalid_reason('T') == 'too_short'
        assert REF_RULES.invalid_reason('ABCDEFGHJK12') == 'too_long'

    def test_generic_profile_keeps_runs_and_short_text(self) -> None:
        rules = RegionTextRules.from_profile(DetectionProfile(name='generic'))
        assert rules.invalid_reason('123') is None
        assert rules.invalid_reason('T') is None

    def test_format_and_configured_placeholders(self) -> None:
        profile = dataclasses.replace(
            REF, text_format='[A-Z]{3}[0-9]{3,4}', text_placeholders=frozenset({'QQQ111'})
        )
        rules = RegionTextRules.from_profile(profile)
        assert rules.invalid_reason('KDK788') is None
        assert rules.invalid_reason('KDK78') == 'format'
        assert rules.invalid_reason('qqq-111') == 'placeholder'

    def test_text_key(self) -> None:
        assert text_key(' not_readable! ') == 'NOTREADABLE'


def _ocr(text: str) -> Any:
    return read_dominant_text([OcrLine(text, (0.1, 0.3, 0.9, 0.7), 0.91)], REF_CFG)


def _resolve(mode: str, vlm: str | None, ocr: Any) -> dict[str, Any]:
    return resolve_region_text(
        mode,
        vlm_text=vlm,
        vlm_confidence='high',
        vlm_engine='vlm-model',
        ocr=ocr,
        ocr_engine='det:1+rec:1',
        normalizer=REF_CFG.normalizer,
        rules=REF_RULES,
    )


class TestChooser:
    def test_placeholder_vlm_reading_falls_back_to_valid_ocr(self) -> None:
        out = _resolve('both', 'ABC123', _ocr('VWY-7977'))
        assert out['text'] == 'VWY7977'
        assert out['text_source'] == 'ocr'
        assert out['text_choice'] == 'vlm_invalid'
        assert out['text_vlm'] == 'ABC123'
        assert out['text_vlm_invalid'] == 'placeholder'
        assert 'text_disagreement' not in out

    def test_invalid_vlm_reading_without_ocr_leaves_no_text(self) -> None:
        out = _resolve('both', '999', None)
        assert 'text' not in out
        assert out['text_choice'] == 'no_valid_reading'
        assert out['text_vlm_invalid'] == 'sequence'

    def test_valid_readings_that_disagree_keep_the_vlm_preference(self) -> None:
        out = _resolve('both', 'VWY7971', _ocr('VWY7977'))
        assert (out['text'], out['text_source']) == ('VWY7971', 'vlm')
        assert out['text_choice'] == 'vlm_preferred'
        assert out['text_disagreement'] is True

    def test_agreeing_readings(self) -> None:
        out = _resolve('both', 'vwy 7977', _ocr('VWY7977'))
        assert out['text_choice'] == 'readers_agree'

    def test_single_readers_and_ocr_mode(self) -> None:
        assert _resolve('both', 'VWY7977', None)['text_choice'] == 'vlm_only'
        assert _resolve('both', None, _ocr('VWY7977'))['text_choice'] == 'ocr_only'
        out = _resolve('ocr', 'VWY7971', _ocr('VWY7977'))
        assert (out['text'], out['text_choice']) == ('VWY7977', 'ocr_mode')

    def test_invalid_ocr_reading_is_not_stored_or_chosen(self) -> None:
        out = _resolve('both', 'VWY7977', _ocr('8888'))
        assert out['text'] == 'VWY7977'
        assert 'text_ocr' not in out
        assert out['text_choice'] == 'vlm_only'

    def test_only_a_rejected_ocr_reading_records_no_valid_reading(self) -> None:
        out = _resolve('both', None, _ocr('8888'))
        assert 'text' not in out
        assert out['text_choice'] == 'no_valid_reading'


class TestWriters:
    @pytest.mark.usefixtures('reference_region_profile')
    def test_direct_vlm_text_write_drops_a_non_reading(self) -> None:
        doc = _region_write_doc(
            plate_in_source=(0.1, 0.1, 0.2, 0.2),
            score=0.9,
            detector='det',
            detector_version='1',
            chain=[],
            plate_text='999',
            plate_text_confidence='high',
        )
        assert F.text not in doc
        assert doc[F.text_vlm] == '999'
        assert doc[F.text_vlm_invalid] == 'sequence'
        assert doc[F.text_choice] == 'no_valid_reading'

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('reference_region_profile')
    async def test_worker_replaces_a_prompt_example_with_the_ocr_reading(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from scripts.curation.worker import runner as runner_mod

        monkeypatch.setattr(
            runner_mod, 'resolve_prompt_pack', lambda: _pack_with_example('XYZ9876')
        )
        fake_os = _FakeOpenSearch({'c1': _item()}, search_delay=0.0, lag_searches=0)
        reply = VlmCombinedReply(
            img_id='c1',
            plate_visible=True,
            plate_bbox_correct=True,
            plate_text='XYZ987',
            plate_confidence='high',
        )
        await _drive(
            tmp_path,
            monkeypatch,
            fake_os=fake_os,
            primary=RegionCandidate(bbox_norm=(0.3, 0.6, 0.6, 0.75), score=0.9, source='det'),
            segmenter=None,
            vlm_url='http://vlm.invalid:8000',
            reply=reply,
            text_reader='vlm_then_ocr',
        )
        doc = fake_os.live['c1']
        assert doc[F.text] == 'ABC1234'
        assert doc[F.text_source] == 'ocr'
        assert doc[F.text_vlm] == 'XYZ987'
        assert doc[F.text_vlm_invalid] == 'placeholder'
        assert doc[F.text_choice] == 'vlm_invalid'


def test_human_typed_text_records_the_human_choice() -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from curation.query_fakes import QueryFakeOpenSearch
    from src.config import get_curation_config
    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    index = get_curation_config().items_index
    fake = QueryFakeOpenSearch(
        {index: {'c1': {'crop_id': 'c1', F.text: 'ABC123', F.text_choice: 'vlm_only'}}}
    )
    app = FastAPI()
    app.include_router(curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake
    with TestClient(app) as client:
        resp = client.patch('/curation/crops/c1/region_meta', json={'region_text': 'VWY7977'})
    assert resp.status_code == 200, resp.text
    doc = fake.docs(index)['c1']
    assert (doc[F.text], doc[F.text_source], doc[F.text_choice]) == ('VWY7977', 'human', 'human')
