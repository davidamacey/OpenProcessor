"""The shipped example region profiles and prompt packs load, and the
text-free ones never ask for or carry region text."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from src.services.detection.profile_registry import region_profile_from_file
from src.services.detection.region_text import validate_text_reader
from src.services.labeling.vlm_prompts import (
    GENERIC_REGION_PACK,
    PromptPack,
    available_prompt_packs,
)


def _never_asks_for_text(body: str) -> None:
    assert 'region_text' not in body
    assert '"text"' not in body
    assert 'text_confidence' not in body


class TestGenericRegionPack:
    def test_is_selectable(self) -> None:
        assert GENERIC_REGION_PACK.name == 'generic_region_v1'
        assert available_prompt_packs()['generic_region_v1'] == GENERIC_REGION_PACK

    def test_never_asks_for_text(self) -> None:
        _never_asks_for_text(json.dumps(GENERIC_REGION_PACK.to_dict()))


class TestRegionProfileFromDict:
    def test_decodes_json_shapes(self) -> None:
        from src.services.detection.profile_registry import region_profile_from_dict

        profile = region_profile_from_dict(
            {
                '_comment': 'ignored',
                'name': 'p',
                'text_reader': 'none',
                'auto_confirm_area_frac': [0.01, 0.5],
                'text_stopwords': ['A', 'B'],
            }
        )
        assert profile.reads_text is False
        assert profile.auto_confirm_area_frac == (0.01, 0.5)
        assert profile.text_stopwords == frozenset({'A', 'B'})

    def test_rejects_unknown_fields_naming_the_source(self) -> None:
        from src.services.detection.profile_registry import region_profile_from_dict

        with pytest.raises(ValueError, match=r"draft.*unknown field 'nme'"):
            region_profile_from_dict({'name': 'p', 'nme': 'x'}, source='draft')
        with pytest.raises(ValueError, match='missing the required "name"'):
            region_profile_from_dict({})


_EXAMPLES = Path(__file__).resolve().parents[2] / 'examples'
_REGION_PROFILES = sorted((_EXAMPLES / 'region_profiles').glob('*.json'))
_PROMPT_PACKS = sorted((_EXAMPLES / 'prompt_packs').glob('*.json'))


class TestExampleRegionProfiles:
    def test_examples_exist(self) -> None:
        names = {p.stem for p in _REGION_PROFILES}
        assert {'license_plate', 'vehicle_wheel'} <= names

    @pytest.mark.parametrize('path', _REGION_PROFILES, ids=lambda p: p.stem)
    def test_every_example_loads(self, path: Path) -> None:
        profile = region_profile_from_file(str(path))
        assert profile.name == path.stem
        validate_text_reader(profile.text_reader)

    def test_vehicle_wheel_is_text_free_and_segmenter_only(self) -> None:
        profile = region_profile_from_file(
            str(_EXAMPLES / 'region_profiles' / 'vehicle_wheel.json')
        )
        assert profile.reads_text is False
        assert profile.detector_model == ''
        assert profile.text_hint_enabled is False
        assert profile.text_hint_active(segmenter_enabled=True) is False
        assert profile.parent_classes == frozenset({'car'})
        assert profile.segmenter_text_prompt == 'wheel'
        assert profile.region_class_name == 'wheel'

    def test_license_plate_keeps_its_text_rules_explicitly(self) -> None:
        profile = region_profile_from_file(
            str(_EXAMPLES / 'region_profiles' / 'license_plate.json')
        )
        assert profile.reads_text is True
        assert profile.detector_model == ''
        assert profile.text_hint_enabled is True
        assert profile.text_hint_require_letters_and_digits is True


class TestExamplePromptPacks:
    def test_examples_exist(self) -> None:
        assert 'vehicle_wheel' in {p.stem for p in _PROMPT_PACKS}

    @pytest.mark.parametrize('path', _PROMPT_PACKS, ids=lambda p: p.stem)
    def test_every_example_loads(self, path: Path) -> None:
        pack = PromptPack.from_json(path)
        assert pack.name
        # Every declared field is present in the file (from_dict ignores
        # extras but a missing one would silently fall back nowhere).
        body = json.loads(path.read_text())
        assert {f.name for f in dataclasses.fields(PromptPack)} <= set(body)

    def test_vehicle_wheel_pack_never_asks_for_text(self) -> None:
        path = _EXAMPLES / 'prompt_packs' / 'vehicle_wheel.json'
        _never_asks_for_text(path.read_text())
        pack = PromptPack.from_json(path)
        assert 'wheel (tire plus rim)' in pack.combined_user_template
        assert 'wheel (tire plus rim)' in pack.region_user
        assert pack.class_descriptions == {}
        assert pack.synonyms == {}
