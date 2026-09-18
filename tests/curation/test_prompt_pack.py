"""Tests for ``PromptPack`` (§5 Chunk 7 "New" test).

Exercises the generic ``PromptPack`` shape plus the shipped neutral
``GENERIC_ITEM_PACK`` example: the dataclass field set, that the shipped
instance's templates are formattable, and that its prompt text uses the
same wire-key vocabulary ``RegionFields`` defaults to (per the design
note in ``vlm_prompts.py``). Deliberately has no dependency on
``vlm_labeler.py`` — this file lands in the same commit as
``vlm_prompts.py`` (§5 Chunk 7 "Commits"), before ``vlm_labeler.py``
exists on this branch. The mechanism functions that *consume* a
``PromptPack`` (``format_class_catalog`` / ``resolve_class_name``, which
live in ``vlm_labeler.py``) are tested in ``test_class_synonyms.py``
instead, landing with the second commit.
"""

from __future__ import annotations

from src.config import get_region_fields
from src.services.labeling.vlm_prompts import GENERIC_ITEM_PACK, PromptPack


def test_prompt_pack_is_frozen_dataclass() -> None:
    import dataclasses

    assert dataclasses.is_dataclass(PromptPack)
    fields = {f.name for f in dataclasses.fields(PromptPack)}
    # Mirrors the reference gemma_labeler.py's inline constant set (§3.4).
    assert fields == {
        'name',
        'class_system',
        'class_user_template',
        'open_class_system',
        'open_class_user_template',
        'combined_system',
        'combined_user_template',
        'combined_batch_system',
        'combined_batch_rules',
        'region_system',
        'region_user',
        'region_batch_system',
        'region_batch_user',
        'region_visible_system',
        'region_visible_user',
        'class_descriptions',
        'synonyms',
    }


def test_generic_item_pack_is_a_prompt_pack() -> None:
    assert isinstance(GENERIC_ITEM_PACK, PromptPack)
    assert GENERIC_ITEM_PACK.name


def test_generic_item_pack_templates_are_formattable() -> None:
    assert '{class_names_csv}' not in GENERIC_ITEM_PACK.class_user_template.format(
        class_names_csv='box, envelope'
    )
    assert '{class_names_csv}' not in GENERIC_ITEM_PACK.open_class_user_template.format(
        class_names_csv='box, envelope'
    )
    rendered = GENERIC_ITEM_PACK.combined_user_template.format(
        class_block='Classify it. ', region_block='No candidate region. '
    )
    assert 'Classify it.' in rendered
    assert 'No candidate region.' in rendered


def test_generic_item_pack_wire_keys_match_region_fields_defaults() -> None:
    """The shipped pack's combined-call prompts ask the VLM for the same
    key names ``vlm_labeler.py``'s reply parser reads via ``RegionFields``
    — see the design note in ``vlm_prompts.py``. This pins that
    agreement so the two can't silently drift apart."""
    fields = get_region_fields()
    for prompt_text in (
        GENERIC_ITEM_PACK.combined_system,
        GENERIC_ITEM_PACK.combined_batch_system,
    ):
        assert fields.visible in prompt_text
        assert fields.bbox_correct in prompt_text
        assert fields.text in prompt_text
        assert fields.confidence in prompt_text
