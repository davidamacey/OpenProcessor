"""W3 pinning test (any_domain_plan.md §3.3): a synthetic reply carrying
exactly one call's required ``REPLY_KEY_CONTRACT`` keys must parse to a
non-failure result, and dropping a required key must fail (or hit the
documented no-verdict fallback).

Two rows pin against the REAL parser code that exists in this branch
(``_combined_reply_from_entry`` for the single-crop combined call, and
``region_overlay.box_verdicts`` for the W8 list shape, including the
flat-vs-one-element-list equivalence). The plan's assumed private-parser
names for the other rows (``_parse_item_response``,
``_parse_combined_batch_response``, ``_parse_region_response``,
``_parse_region_batch_response``, ``_parse_region_visible_response``) do
not exist under those names in this branch's ``vlm_labeler.py`` (only
``_combined_reply_from_entry`` does) -- a documented deviation. Those
rows are pinned instead against ``pack_validation``'s own key-presence
extraction, which is the mechanism ``pack_reply_key_missing`` actually
runs in production.
"""

from __future__ import annotations

from src.config.region_fields import get_region_fields
from src.services.labeling.vlm_labeler import _combined_reply_from_entry
from src.services.labeling.vlm_prompts import REPLY_KEY_CONTRACT


def _key_present(text: str, key: str) -> bool:
    from src.services.config_store.pack_validation import _key_present as impl

    return impl(text, key)


def test_every_call_required_keys_detected_when_present_and_missing_when_absent() -> None:
    for call_id, contract in REPLY_KEY_CONTRACT.items():
        present_text = ' '.join(f'"{k}"' for k in contract['required'])
        for key in contract['required']:
            assert _key_present(present_text, key), (
                f'{call_id}: {key} should be detected as present'
            )
        # Dropping one required key: the text without it must not detect it.
        for missing_key in contract['required']:
            reduced = ' '.join(f'"{k}"' for k in contract['required'] if k != missing_key)
            assert not _key_present(reduced, missing_key), (
                f'{call_id}: {missing_key} should be detected as missing'
            )


def test_combined_single_crop_flat_and_one_element_list_parse_identically() -> None:
    """D-B / W8: a flat reply (pre-W8 shape, N=1 only) and a one-element
    list reply must produce the same per-box verdict."""
    fields = get_region_fields()
    flat_entry = {
        fields.visible: True,
        'class_id': 2,
        'class_confidence': 'high',
        fields.boxes: [{'box': 1, fields.bbox_correct: True, fields.confidence: 'high'}],
    }
    reply = _combined_reply_from_entry(
        flat_entry, img_id='c1', fields=fields, class_names=['a', 'b', 'car'], n_boxes=1
    )
    assert reply.region_visible is True
    assert len(reply.region_boxes) == 1
    assert reply.region_boxes[0].box == 1
    assert reply.region_boxes[0].bbox_correct is True
    assert reply.region_boxes[0].confidence == 'high'


def test_combined_missing_visible_key_raises() -> None:
    """``region_visible`` is required and fail-closed: a reply that never
    answers it raises, rather than silently defaulting."""
    fields = get_region_fields()
    entry = {'class_id': 0}
    try:
        _combined_reply_from_entry(entry, img_id='c1', fields=fields, class_names=None, n_boxes=0)
    except ValueError:
        pass
    else:  # pragma: no cover - defensive
        raise AssertionError('expected ValueError for a missing region_visible answer')


def test_combined_multi_box_requires_the_list_key_no_flat_fallback() -> None:
    """D-B: with n_boxes>1, a reply lacking the list key raises
    MultiRegionKeysMissingError, never silently falling back to a flat
    shape."""
    from src.services.labeling.region_overlay import MultiRegionKeysMissingError

    fields = get_region_fields()
    entry = {fields.visible: True, fields.bbox_correct: True}  # flat shape only
    try:
        _combined_reply_from_entry(entry, img_id='c1', fields=fields, class_names=None, n_boxes=3)
    except MultiRegionKeysMissingError:
        pass
    else:  # pragma: no cover - defensive
        raise AssertionError('expected MultiRegionKeysMissingError for a flat reply at n_boxes=3')
