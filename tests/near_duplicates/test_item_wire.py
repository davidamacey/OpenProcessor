"""The item wire serves the ``near_dup_*`` group fields and none of the retired ``dup_*`` keys."""

from __future__ import annotations

from _embedding_registry_fake import DEFAULT_STATE_FIELDS

from src.services.curation.wire import serialize_item


def _source(**extra: object) -> dict[str, object]:
    return {'crop_id': 'c1', 'image_path': 'p.jpg', **extra}


def test_no_dup_key_is_served() -> None:
    served = serialize_item(
        _source(
            near_dup_group_id='g1',
            near_dup_group_size=3,
            near_dup_is_representative=False,
            # a 0.6.0 document still holds the old values until the upgrade step clears them
            dup_group_id='dup_3',
            dup_group_size=4,
            dup_is_representative=True,
        ),
        state_fields=DEFAULT_STATE_FIELDS,
        api_prefix='',
    )
    assert [k for k in served if k.startswith('dup_')] == []


def test_the_group_fields_are_served() -> None:
    served = serialize_item(
        _source(near_dup_group_id='g1', near_dup_group_size=3, near_dup_is_representative=False),
        state_fields=DEFAULT_STATE_FIELDS,
        api_prefix='',
    )
    assert served['near_dup_group_id'] == 'g1'
    assert served['near_dup_group_size'] == 3
    assert served['near_dup_is_representative'] is False


def test_an_ungrouped_item_serves_none() -> None:
    served = serialize_item(_source(), state_fields=DEFAULT_STATE_FIELDS, api_prefix='')
    assert (
        served['near_dup_group_id'],
        served['near_dup_group_size'],
        served['near_dup_is_representative'],
    ) == (None, None, None)
