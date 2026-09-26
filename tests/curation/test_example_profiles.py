"""The shipped example region profiles and prompt packs load, and the
text-free ones never ask for or carry region text."""

from __future__ import annotations

import json

from src.services.labeling.vlm_prompts import GENERIC_REGION_PACK, available_prompt_packs


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
        import pytest

        from src.services.detection.profile_registry import region_profile_from_dict

        with pytest.raises(ValueError, match=r"draft.*unknown field 'nme'"):
            region_profile_from_dict({'name': 'p', 'nme': 'x'}, source='draft')
        with pytest.raises(ValueError, match='missing the required "name"'):
            region_profile_from_dict({})
