"""Erasure's catalogue is complete: every table that holds a subject key is in it (15a section 8, B5).

15a: ``test_erasure_catalogue_complete`` fails when a table with a column named ``photo_key``,
``item_key``, ``box_key``, ``face_key``, ``entity_key``, ``subject_key``, ``subject_*_id``, ``image_id``,
``crop_id``, ``face_id`` or ``member_id`` is missing from the catalogue. The same check runs here on the
models; the database tier runs it on the migrated catalog and adds the vector tables, found through
``embedding_spaces``.
"""

from __future__ import annotations

import re

import pytest

import src.db.models
from src.db.base import Base
from src.services.erasure import catalogue


SUBJECT_KEY = re.compile(
    r'^(photo_key|item_key|box_key|face_key|entity_key|subject_key|subject_\w+_id|image_id|crop_id|face_id|member_id)$'
)


def _registered_tables() -> set[str]:
    assert src.db.models.__all__  # importing the package registers every table on Base.metadata
    return set(Base.metadata.tables)


def _tables_with_a_subject_key() -> dict[str, set[str]]:
    found: dict[str, set[str]] = {}
    for table in Base.metadata.tables.values():
        columns = {c.name for c in table.columns if SUBJECT_KEY.match(c.name)}
        if columns:
            found[table.name] = columns
    return found


def test_every_table_that_holds_a_subject_key_is_in_the_catalogue() -> None:
    missing = sorted(set(_tables_with_a_subject_key()) - catalogue.tables())
    assert missing == [], f'tables with a subject key that erasure cannot reach: {missing}'


def test_every_catalogue_entry_names_a_real_column() -> None:
    for entry in catalogue.CATALOGUE:
        table = Base.metadata.tables.get(entry.table)
        assert table is not None, f'{entry.table} is not a table'
        assert entry.column in table.c, f'{entry.table}.{entry.column} is not a column'


def _cascades_from(root: str) -> set[str]:
    reach = {root}
    grew = True
    while grew:
        grew = False
        for table in Base.metadata.tables.values():
            for fk in table.foreign_key_constraints:
                if (
                    fk.ondelete == 'CASCADE'
                    and fk.referred_table.name in reach
                    and table.name not in reach
                ):
                    reach.add(table.name)
                    grew = True
    return reach


def test_the_listed_cascades_are_the_cascades_of_the_models() -> None:
    reached = _cascades_from('photos')
    assert catalogue.cascade_tables() == reached, sorted(catalogue.cascade_tables() ^ reached)


def test_a_person_cascades_to_its_state_edges_and_claims() -> None:
    reached = _cascades_from('entities')
    listed = {e.table for e in catalogue.CATALOGUE if e.via == 'entities'}
    assert listed - {'entities'} <= reached
    assert {'entity_state', 'entity_edges', 'membership_state', 'claims'} <= listed


@pytest.mark.parametrize('entry', [e for e in catalogue.CATALOGUE if e.reach == 'set_null'])
def test_a_set_null_entry_is_a_set_null_key(entry: catalogue.Entry) -> None:
    table = Base.metadata.tables[entry.table]
    kinds = {fk.ondelete for fk in table.c[entry.column].foreign_keys}
    assert kinds == {'SET NULL'}


@pytest.mark.parametrize('entry', [e for e in catalogue.CATALOGUE if e.reach == 'explicit'])
def test_an_explicit_entry_has_no_foreign_key_that_would_cascade(entry: catalogue.Entry) -> None:
    table = Base.metadata.tables[entry.table]
    assert not [fk for fk in table.c[entry.column].foreign_keys if fk.ondelete == 'CASCADE']


def test_the_vector_delete_is_by_subject_key_in_the_spaces_table() -> None:
    assert (
        catalogue.vector_delete_statement(12, 'frame_pe')
        == 'DELETE FROM vec.v12_frame_pe WHERE subject_key = ANY(:keys)'
    )
