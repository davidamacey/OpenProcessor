"""Unit tests for ``resolve_class_name``.

Exercises the synonym-resolution *mechanism* (exact match,
case/separator normalization, synonym rescue, low-confidence force-fit
bypass, unknown-phrase fallthrough) against a small local synonym table
plus the neutral ``GENERIC_ITEM_PACK`` shipped in ``vlm_prompts.py``,
rather than any domain-specific vocabulary.
"""

from __future__ import annotations

import pytest

from src.services.labeling.vlm_labeler import format_class_catalog, resolve_class_name
from src.services.labeling.vlm_prompts import GENERIC_ITEM_PACK, PromptPack


# =============================================================================
# GENERIC_ITEM_PACK -- registry-agnostic default
# =============================================================================


@pytest.mark.parametrize('registry', [frozenset(), frozenset({'person', 'car', 'bus'})])
def test_shipped_default_pack_validates_with_zero_warnings(registry: frozenset[str]) -> None:
    """A fresh install (empty or COCO registry) must not see vocabulary
    warnings for the pack it ships as the default."""
    from src.services.config_store.pack_validation import validate_pack

    report = validate_pack(None, GENERIC_ITEM_PACK.to_dict(), class_names=registry)
    assert report.ok
    loud = [w.code for w in report.warnings if w.severity == 'warning']
    assert not loud, loud


# =============================================================================
# resolve_class_name
# =============================================================================


@pytest.fixture
def synonym_pack() -> PromptPack:
    """A small local pack with an easy-to-reason-about synonym table,
    independent of whatever ``GENERIC_ITEM_PACK`` ships (kept separate
    so this test file doesn't need updating if the shipped pack's
    vocabulary changes)."""
    base = GENERIC_ITEM_PACK
    return PromptPack(
        name='test_pack',
        class_system=base.class_system,
        class_user_template=base.class_user_template,
        open_class_system=base.open_class_system,
        open_class_user_template=base.open_class_user_template,
        combined_system=base.combined_system,
        combined_user_template=base.combined_user_template,
        combined_batch_system=base.combined_batch_system,
        combined_batch_rules=base.combined_batch_rules,
        region_system=base.region_system,
        region_user=base.region_user,
        region_batch_system=base.region_batch_system,
        region_batch_user=base.region_batch_user,
        region_visible_system=base.region_visible_system,
        region_visible_user=base.region_visible_user,
        synonyms={
            'carton': 'box',
            'package': 'box',
            'mailer': 'envelope',
            'poly bag': 'envelope',
            'poly_bag': 'envelope',
        },
    )


@pytest.fixture
def name_to_id() -> dict[str, int]:
    """A representative subset of an active registry."""
    return {
        'box': 0,
        'envelope': 1,
        'tube': 2,
        'crate': 3,
    }


def test_resolve_returns_none_for_empty_input(name_to_id: dict[str, int], synonym_pack) -> None:
    assert resolve_class_name('', name_to_id, pack=synonym_pack) is None
    assert resolve_class_name(None, name_to_id, pack=synonym_pack) is None


def test_resolve_exact_match_passes_through(name_to_id: dict[str, int], synonym_pack) -> None:
    assert resolve_class_name('box', name_to_id, pack=synonym_pack) == 'box'
    assert resolve_class_name('envelope', name_to_id, pack=synonym_pack) == 'envelope'


def test_resolve_normalises_case_and_separators(name_to_id: dict[str, int], synonym_pack) -> None:
    # The registry stores ``poly_bag`` as unknown, but "Poly-Bag" should
    # normalise to "poly bag", which is also a synonyms key → mapped to
    # the registry slug.
    assert resolve_class_name('Poly-Bag', name_to_id, pack=synonym_pack) == 'envelope'


def test_resolve_uses_synonyms_when_no_direct_match(
    name_to_id: dict[str, int], synonym_pack
) -> None:
    assert resolve_class_name('carton', name_to_id, pack=synonym_pack) == 'box'
    assert resolve_class_name('package', name_to_id, pack=synonym_pack) == 'box'
    assert resolve_class_name('mailer', name_to_id, pack=synonym_pack) == 'envelope'


def test_resolve_handles_underscored_synonyms(name_to_id: dict[str, int], synonym_pack) -> None:
    """``poly_bag`` is a synonyms key spelt with ``_`` — the resolver
    normalises to spaces *and* re-tries with ``_`` rejoined, so both
    forms map to ``envelope``.
    """
    assert resolve_class_name('poly_bag', name_to_id, pack=synonym_pack) == 'envelope'
    assert resolve_class_name('poly bag', name_to_id, pack=synonym_pack) == 'envelope'


def test_resolve_returns_none_when_synonym_target_not_in_registry(synonym_pack) -> None:
    """If the synonym target isn't in the active registry, fall through to None."""
    name_to_id = {'box': 0}  # deliberately omit 'envelope'
    assert 'envelope' not in name_to_id
    assert resolve_class_name('mailer', name_to_id, pack=synonym_pack) is None


def test_resolve_returns_none_for_unknown_phrase(name_to_id: dict[str, int], synonym_pack) -> None:
    assert resolve_class_name('unicycle', name_to_id, pack=synonym_pack) is None
    assert resolve_class_name('not a thing', name_to_id, pack=synonym_pack) is None


def test_resolve_strips_whitespace(name_to_id: dict[str, int], synonym_pack) -> None:
    assert resolve_class_name('  carton  ', name_to_id, pack=synonym_pack) == 'box'


def test_resolve_skips_synonym_lookup_on_low_confidence(
    name_to_id: dict[str, int], synonym_pack
) -> None:
    """Low-confidence replies bypass the fuzzy/synonym rescue entirely —
    a low-confidence 'carton' should NOT resolve to 'box'."""
    assert resolve_class_name('carton', name_to_id, confidence='low', pack=synonym_pack) is None
    # Exact match is still honored even at low confidence.
    assert resolve_class_name('box', name_to_id, confidence='low', pack=synonym_pack) == 'box'


def test_resolve_class_name_defaults_to_generic_item_pack() -> None:
    name_to_id = {'box': 0, 'envelope': 1}
    assert resolve_class_name('box', name_to_id) == 'box'
    assert resolve_class_name('carton', name_to_id) is None  # no shipped vocabulary


# =============================================================================
# format_class_catalog
# =============================================================================


def test_format_class_catalog_uses_pack_descriptions() -> None:
    classes: list[dict[str, object]] = [
        {'class_name': 'box', 'group': 'containers'},
        {'class_name': 'envelope', 'group': 'flat'},
        {'class_name': 'deprecated_thing', 'group': 'flat', 'deprecated': True},
    ]
    pack = PromptPack.from_dict(
        {
            **GENERIC_ITEM_PACK.to_dict(),
            'name': 'described',
            'class_descriptions': {
                'box': 'rectangular cardboard shipping box',
                'envelope': 'flat paper or poly mailer',
            },
        }
    )
    catalog = format_class_catalog(classes, pack)
    assert 'containers: box (rectangular cardboard shipping box)' in catalog
    assert 'flat: envelope (flat paper or poly mailer)' in catalog
    assert 'deprecated_thing' not in catalog


def test_format_class_catalog_falls_back_to_bare_slug_without_description() -> None:
    pack = PromptPack(
        name='no-descriptions',
        class_system=GENERIC_ITEM_PACK.class_system,
        class_user_template=GENERIC_ITEM_PACK.class_user_template,
        open_class_system=GENERIC_ITEM_PACK.open_class_system,
        open_class_user_template=GENERIC_ITEM_PACK.open_class_user_template,
        combined_system=GENERIC_ITEM_PACK.combined_system,
        combined_user_template=GENERIC_ITEM_PACK.combined_user_template,
        combined_batch_system=GENERIC_ITEM_PACK.combined_batch_system,
        combined_batch_rules=GENERIC_ITEM_PACK.combined_batch_rules,
        region_system=GENERIC_ITEM_PACK.region_system,
        region_user=GENERIC_ITEM_PACK.region_user,
        region_batch_system=GENERIC_ITEM_PACK.region_batch_system,
        region_batch_user=GENERIC_ITEM_PACK.region_batch_user,
        region_visible_system=GENERIC_ITEM_PACK.region_visible_system,
        region_visible_user=GENERIC_ITEM_PACK.region_visible_user,
    )
    catalog = format_class_catalog([{'class_name': 'widget', 'group': 'misc'}], pack)
    assert catalog == 'misc: widget'
