"""Labeling-assist plan task (a): the VLM labeler singleton and the
pipeline's inline class-catalog formatting must both resolve their
``PromptPack`` via ``resolve_prompt_pack()`` (config-driven) rather than
importing ``GENERIC_ITEM_PACK`` as a hardcoded module constant.

``src/routers/curation/models.py`` -- the third call site named in the
task -- is intentionally NOT re-tested here: it never imported
``GENERIC_ITEM_PACK`` directly, it only calls ``_get_vlm_labeler()``, so it
already inherits whatever pack that singleton resolves. Fixing ``vlm.py``'s
singleton construction (below) fixes ``models.py`` for free.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from src.routers.curation import vlm as vlm_mod


if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


@pytest.fixture(autouse=True)
def _reset_vlm_labeler_singleton() -> Iterator[None]:
    """The singleton is cached as a function attribute on
    ``_get_vlm_labeler`` -- clear it around every test in this module so
    one test's patched pack can't leak into another's."""
    vlm_mod._get_vlm_labeler.__dict__.pop('_insts', None)
    yield
    vlm_mod._get_vlm_labeler.__dict__.pop('_insts', None)


def test_get_vlm_labeler_uses_generic_pack_by_default() -> None:
    from src.services.labeling.vlm_prompts import GENERIC_ITEM_PACK

    labeler = vlm_mod._get_vlm_labeler()
    assert labeler._pack is GENERIC_ITEM_PACK


def test_get_vlm_labeler_resolves_a_configured_pack(tmp_path: Path) -> None:
    from dataclasses import replace

    from src.services.labeling.vlm_prompts import GENERIC_ITEM_PACK

    custom = replace(GENERIC_ITEM_PACK, name='pallet_v1')
    path = tmp_path / 'pack.json'
    custom.to_json(path)

    with patch(
        'src.services.labeling.vlm_prompts.resolve_prompt_pack', return_value=custom
    ) as mock_resolve:
        labeler = vlm_mod._get_vlm_labeler()
    mock_resolve.assert_called_once()
    assert labeler._pack.name == 'pallet_v1'


def test_pipeline_class_catalog_uses_resolved_pack(monkeypatch: pytest.MonkeyPatch) -> None:
    """``pipeline.py`` used to import ``GENERIC_ITEM_PACK`` directly at its
    ``format_class_catalog`` call site -- pin that it now goes through
    ``resolve_prompt_pack`` instead (source-level check: the hardcoded
    import must be gone)."""
    from pathlib import Path as _Path

    from src.routers.curation import pipeline

    src = _Path(pipeline.__file__).read_text()
    assert 'from src.services.labeling.vlm_prompts import GENERIC_ITEM_PACK' not in src
    assert 'resolve_prompt_pack' in src


def test_get_vlm_labeler_selects_and_caches_per_pack_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Selection is by pack name, one cached labeler per pack; the default
    (no name) is the OP_PROMPT_PACK_PATH pack; unknown names raise."""
    from dataclasses import replace

    from src.config.curation import CurationConfig
    from src.services.labeling.vlm_prompts import GENERIC_ITEM_PACK

    default_path = tmp_path / 'default.json'
    extra_path = tmp_path / 'extra.json'
    replace(GENERIC_ITEM_PACK, name='pallet_v1').to_json(default_path)
    replace(GENERIC_ITEM_PACK, name='food_v2').to_json(extra_path)
    cfg = CurationConfig(prompt_pack_path=default_path, prompt_pack_paths=(extra_path,))
    monkeypatch.setattr('src.config.curation.get_curation_config', lambda: cfg)

    assert vlm_mod._get_vlm_labeler()._pack.name == 'pallet_v1'
    food = vlm_mod._get_vlm_labeler('food_v2')
    assert food._pack.name == 'food_v2'
    assert vlm_mod._get_vlm_labeler('food_v2') is food
    assert vlm_mod._get_vlm_labeler(GENERIC_ITEM_PACK.name)._pack is GENERIC_ITEM_PACK
    with pytest.raises(ValueError, match='unknown prompt pack'):
        vlm_mod._get_vlm_labeler('nope')
