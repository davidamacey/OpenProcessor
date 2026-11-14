"""The nested-document limit (OpenSearch k-NN issue 3623): the count, the write rule, the
repair write and the Painless guard text. Pure: no OpenSearch."""

from __future__ import annotations

import pytest

from src.clients.curation_opensearch.assist_fields import ASSIST_ITEM_MAPPING
from src.clients.curation_opensearch.bodies_core import _items_body
from src.config.region_fields import get_region_fields
from src.domain.embedding_spaces import Target, physical_names
from src.domain.nested_limit import (
    ASSIST_FIELD,
    NESTED_DOCS_LIMIT,
    NestedLimitError,
    box_vector_paths,
    guard_write,
    is_over_limit_bulk_error,
    max_boxes_per_item,
    nested_doc_count,
    over_limit_ids,
    painless_guard,
)


F = get_region_fields()
EXCLUDES = ['pe_embedding', F.box_embeddings]


def _extra_path() -> str:
    path = physical_names('k', Target.BOX).nested_path
    assert path is not None
    return path


def _boxes(n: int) -> list[dict]:
    return [{'box_id': f'b{i}'} for i in range(n)]


def _entries(n: int) -> list[dict]:
    return [{'box_id': f'b{i}', 'embedding': [0.0]} for i in range(n)]


def test_limit_is_the_upstream_threshold() -> None:
    assert NESTED_DOCS_LIMIT == 150


@pytest.mark.parametrize(('lists', 'expected'), [(1, 74), (2, 49), (3, 37)])
def test_max_boxes_per_item_follows_the_live_box_spaces(lists: int, expected: int) -> None:
    assert max_boxes_per_item(lists) == expected
    assert (expected * (1 + lists)) < NESTED_DOCS_LIMIT <= (expected + 1) * (1 + lists)


def test_assist_field_name_matches_the_mapping() -> None:
    assert list(ASSIST_ITEM_MAPPING['assist']['properties']) >= ['attributes', 'boxes']
    assert ASSIST_FIELD in ASSIST_ITEM_MAPPING


def test_every_nested_field_of_the_items_body_is_counted() -> None:
    """A new nested field on the items index must be added to the count, or this fails."""
    props = _items_body()['mappings']['properties']

    def nested_paths(properties: dict, prefix: str = '') -> set[str]:
        out: set[str] = set()
        for name, spec in properties.items():
            path = f'{prefix}{name}'
            if spec.get('type') == 'nested':
                out.add(path)
            out |= nested_paths(spec.get('properties', {}), f'{path}.')
        return out

    assert nested_paths(props) == {
        F.boxes,
        F.box_embeddings,
        'assist.attributes',
        'assist.boxes',
        'assist.boxes.attributes',
    }


def test_count_covers_boxes_vectors_and_assist() -> None:
    src = {
        F.boxes: _boxes(3),
        F.box_embeddings: _entries(3),
        'boxvec_x': _entries(2),
        ASSIST_FIELD: {
            'attributes': [{'key': 'k'}] * 2,
            'boxes': [{'box_id': 'b0', 'attributes': [{'key': 'k'}] * 4}, {'box_id': 'b1'}],
        },
    }
    assert nested_doc_count(src) == 3 + 3 + 2 + 2 + (1 + 4) + 1


def test_unread_box_vector_lists_count_one_entry_per_box() -> None:
    assert (
        nested_doc_count({F.boxes: _boxes(74)}, unread_box_vector_paths=[F.box_embeddings]) == 148
    )
    assert (
        nested_doc_count({F.boxes: _boxes(75)}, unread_box_vector_paths=[F.box_embeddings]) == 150
    )


def test_box_vector_paths_include_the_default_and_extra_spaces() -> None:
    extra = _extra_path()
    assert box_vector_paths(['pe_embedding', extra]) == {F.box_embeddings, extra}


def test_write_just_below_the_limit_is_accepted() -> None:
    update = {F.boxes: _boxes(74)}
    assert guard_write('c1', {F.boxes: []}, update, excludes=EXCLUDES) == update


def test_write_at_the_limit_is_refused_and_nothing_is_returned() -> None:
    with pytest.raises(NestedLimitError) as err:
        guard_write('c1', {F.boxes: []}, {F.boxes: _boxes(75)}, excludes=EXCLUDES)
    assert err.value.nested_docs == 150
    assert err.value.detail()['limit'] == 150
    assert err.value.detail()['error'] == 'item_over_nested_limit'


def test_a_second_box_space_lowers_the_box_limit() -> None:
    extra = _extra_path()
    excludes = [*EXCLUDES, extra]
    guard_write('c1', {}, {F.boxes: _boxes(49)}, excludes=excludes)
    with pytest.raises(NestedLimitError):
        guard_write('c1', {}, {F.boxes: _boxes(50)}, excludes=excludes)


def test_assist_attributes_count_against_the_limit() -> None:
    current = {F.boxes: _boxes(70), ASSIST_FIELD: {'attributes': [{'key': 'k'}] * 10}}
    with pytest.raises(NestedLimitError):
        guard_write('c1', current, {'class_id': 3}, excludes=EXCLUDES)


def test_other_writes_on_an_over_limit_item_are_refused() -> None:
    current = {F.boxes: _boxes(80)}
    with pytest.raises(NestedLimitError) as err:
        guard_write('c1', current, {'cluster_uid': 'u'}, excludes=EXCLUDES)
    assert 'already holds' in str(err.value)


def test_repair_write_under_the_limit_drops_every_box_vector() -> None:
    extra = _extra_path()
    current = {F.boxes: _boxes(80)}
    update = {F.boxes: _boxes(40), F.revision: 3}
    out = guard_write('c1', current, update, excludes=[*EXCLUDES, extra])
    assert out[F.boxes] == update[F.boxes]
    assert out[F.box_embeddings] == []
    assert out[extra] == []


def test_a_box_write_that_stays_over_the_limit_is_not_a_repair() -> None:
    with pytest.raises(NestedLimitError):
        guard_write('c1', {F.boxes: _boxes(80)}, {F.boxes: _boxes(76)}, excludes=EXCLUDES)


def test_an_embedding_write_on_an_item_under_the_limit_counts_the_real_list() -> None:
    current = {F.boxes: _boxes(74)}
    update = {F.box_embeddings: _entries(74)}
    assert guard_write('c1', current, update, excludes=EXCLUDES) == update


def test_an_empty_update_is_a_noop() -> None:
    assert guard_write('c1', {F.boxes: _boxes(500)}, {}, excludes=EXCLUDES) == {}


def test_over_limit_ids_lists_id_and_box_count() -> None:
    docs = {'a': {F.boxes: _boxes(10)}, 'b': {F.boxes: _boxes(75)}}
    assert over_limit_ids(docs, unread_box_vector_paths=[F.box_embeddings]) == {'b': (150, 75)}


def test_painless_guard_wraps_the_body_with_the_limit() -> None:
    noop = painless_guard('x();', on_over='noop')
    assert 'op_nested >= 150' in noop
    assert "ctx.op = 'noop'" in noop
    assert 'x();' in noop
    assert F.boxes in noop
    assert 'boxvec_' in noop
    assert 'IllegalStateException' in painless_guard('x();', on_over='throw')


def test_bulk_error_recognition() -> None:
    err = {'type': 'script_exception', 'caused_by': {'reason': 'op_nested_limit'}}
    assert is_over_limit_bulk_error(err)
    assert not is_over_limit_bulk_error({'type': 'script_exception', 'caused_by': {'reason': 'x'}})
