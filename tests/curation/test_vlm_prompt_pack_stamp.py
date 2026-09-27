"""``prompt_pack_stamp`` -- the ``"<name>@<revision|sha12>"`` provenance
string every VLM write site stamps onto ``vlm_prompt_pack`` (any_domain_plan.md
§3.7/§9 W2, execution_schedule.md §4.4 W2 gap)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from src.services.labeling.vlm_prompts import GENERIC_ITEM_PACK, prompt_pack_stamp


if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def _reset_config_store() -> Iterator[None]:
    from src.services.config_store.store import reset_config_stores

    reset_config_stores()
    yield
    reset_config_stores()


def test_unactivated_pack_stamps_with_a_content_hash() -> None:
    stamp = prompt_pack_stamp(GENERIC_ITEM_PACK)
    name, _, suffix = stamp.partition('@')
    assert name == GENERIC_ITEM_PACK.name
    assert suffix  # some non-empty content hash
    assert suffix.isalnum()
    # Deterministic: the same pack content always hashes the same.
    assert prompt_pack_stamp(GENERIC_ITEM_PACK) == stamp


def test_different_pack_content_stamps_a_different_hash() -> None:
    from dataclasses import replace

    other = replace(GENERIC_ITEM_PACK, name='other_pack')
    assert prompt_pack_stamp(GENERIC_ITEM_PACK) != prompt_pack_stamp(other)


def test_activated_pack_stamps_with_its_store_revision() -> None:
    from src.services.config_store.store import get_config_store

    store = get_config_store()
    store.current = store.current.__class__(
        config_revision=store.current.config_revision,
        active_pack=(GENERIC_ITEM_PACK.name, 7),
    )
    assert prompt_pack_stamp(GENERIC_ITEM_PACK) == f'{GENERIC_ITEM_PACK.name}@7'


def test_activated_pack_with_a_different_name_does_not_borrow_the_revision() -> None:
    from dataclasses import replace

    from src.services.config_store.store import get_config_store

    store = get_config_store()
    store.current = store.current.__class__(
        config_revision=store.current.config_revision,
        active_pack=('some_other_pack', 3),
    )
    other = replace(GENERIC_ITEM_PACK, name='not_the_active_one')
    stamp = prompt_pack_stamp(other)
    assert stamp.startswith('not_the_active_one@')
    assert stamp != 'not_the_active_one@3'
