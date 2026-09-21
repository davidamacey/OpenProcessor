"""Tests for :mod:`src.services.labeling.vlm_prompts` (labeling-assist plan
task (a)): ``PromptPack`` JSON round-trip and ``resolve_prompt_pack``'s
deployment-config resolution + fallback behavior.
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from src.services.labeling.vlm_prompts import GENERIC_ITEM_PACK, PromptPack, resolve_prompt_pack


if TYPE_CHECKING:
    from pathlib import Path


def test_to_dict_from_dict_round_trips() -> None:
    data = GENERIC_ITEM_PACK.to_dict()
    rebuilt = PromptPack.from_dict(data)
    assert rebuilt == GENERIC_ITEM_PACK


def test_from_dict_ignores_unknown_keys() -> None:
    """A pack file may carry a ``_comment`` field (this repo's convention
    for its other example config files) alongside the real pack data."""
    data = {'_comment': 'a worked example', **GENERIC_ITEM_PACK.to_dict()}
    rebuilt = PromptPack.from_dict(data)
    assert rebuilt == GENERIC_ITEM_PACK


def test_from_dict_missing_required_field_raises() -> None:
    data = GENERIC_ITEM_PACK.to_dict()
    del data['class_system']
    with pytest.raises(TypeError):
        PromptPack.from_dict(data)


def test_to_json_from_json_round_trips(tmp_path: Path) -> None:
    path = tmp_path / 'pack.json'
    GENERIC_ITEM_PACK.to_json(path)
    loaded = PromptPack.from_json(path)
    assert loaded == GENERIC_ITEM_PACK
    # Pretty-printed, real JSON -- not a repr dump.
    parsed = json.loads(path.read_text())
    assert parsed['name'] == 'generic_item_v1'


def test_from_json_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        PromptPack.from_json(tmp_path / 'does_not_exist.json')


def test_example_pack_file_loads_and_differs_from_generic() -> None:
    """``data/prompt_pack.example.json`` (a worked, non-vehicle domain) must
    actually load and must not just be a renamed copy of the built-in
    generic pack."""
    from pathlib import Path as _Path

    example_path = _Path(__file__).resolve().parents[2] / 'data' / 'prompt_pack.example.json'
    pack = PromptPack.from_json(example_path)
    assert pack.name != GENERIC_ITEM_PACK.name
    assert pack.class_descriptions != GENERIC_ITEM_PACK.class_descriptions


class _FakeCfg:
    def __init__(self, prompt_pack_path: Path | None) -> None:
        self.prompt_pack_path = prompt_pack_path


def test_resolve_prompt_pack_falls_back_when_path_unset() -> None:
    assert resolve_prompt_pack(_FakeCfg(None)) is GENERIC_ITEM_PACK


def test_resolve_prompt_pack_falls_back_when_file_missing(tmp_path: Path) -> None:
    missing = tmp_path / 'nope.json'
    assert resolve_prompt_pack(_FakeCfg(missing)) is GENERIC_ITEM_PACK


def test_resolve_prompt_pack_falls_back_on_malformed_json(tmp_path: Path) -> None:
    bad = tmp_path / 'bad.json'
    bad.write_text('{not valid json')
    assert resolve_prompt_pack(_FakeCfg(bad)) is GENERIC_ITEM_PACK


def test_resolve_prompt_pack_loads_a_configured_pack(tmp_path: Path) -> None:
    custom = replace(GENERIC_ITEM_PACK, name='pallet_v1')
    path = tmp_path / 'pallet.json'
    custom.to_json(path)

    resolved = resolve_prompt_pack(_FakeCfg(path))
    assert resolved.name == 'pallet_v1'
    assert resolved == custom


def test_resolve_prompt_pack_uses_process_default_config_when_none_passed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.config.curation import CurationConfig

    monkeypatch.setattr(
        'src.config.curation.get_curation_config', lambda: CurationConfig(prompt_pack_path=None)
    )
    assert resolve_prompt_pack() is GENERIC_ITEM_PACK
