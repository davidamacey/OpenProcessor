"""The statements that make one vector table per space, read as SQL (15a section 6.1). No statement is run."""

from __future__ import annotations

import pytest

from src.db import vector_ddl
from src.db.vector_ddl import VectorDdlError


def _statements(dim: int = 1024) -> tuple[str, ...]:
    return vector_ddl.create_statements(12, 'frame_pe', dim, app_role='op_app')


def test_a_table_is_made_in_the_vec_schema_named_for_the_project_and_the_space() -> None:
    assert vector_ddl.table_name(12, 'frame_pe') == 'v12_frame_pe'
    assert vector_ddl.qualified(12, 'frame_pe') == 'vec.v12_frame_pe'
    assert _statements()[0].startswith('CREATE TABLE vec.v12_frame_pe (')


def test_the_row_holds_the_key_the_project_the_input_revision_and_one_half_precision_vector() -> (
    None
):
    create = _statements(768)[0]
    assert 'subject_key bigint PRIMARY KEY' in create
    assert 'project_id uuid NOT NULL REFERENCES projects(id) ON DELETE CASCADE' in create
    assert 'input_rev integer NOT NULL DEFAULT 0' in create
    assert 'embedding halfvec(768) NOT NULL' in create


def test_the_vector_stays_inline_and_the_table_is_closed_by_forced_row_level_security() -> None:
    sql = _statements()
    assert sql[1] == 'ALTER TABLE vec.v12_frame_pe ALTER COLUMN embedding SET STORAGE PLAIN'
    assert 'ALTER TABLE vec.v12_frame_pe ENABLE ROW LEVEL SECURITY' in sql
    assert 'ALTER TABLE vec.v12_frame_pe FORCE ROW LEVEL SECURITY' in sql
    policy = next(s for s in sql if s.startswith('CREATE POLICY'))
    assert 'FOR ALL' in policy
    assert (
        policy.count(
            '(SELECT op_scope_is_deployment()) OR project_id = (SELECT op_scope_project())'
        )
        == 2
    )
    assert 'WITH CHECK' in policy


def test_the_application_role_may_read_insert_and_delete_but_not_change_or_create() -> None:
    grants = [s for s in _statements() if s.startswith('GRANT')]
    assert grants == ['GRANT SELECT, INSERT, DELETE ON vec.v12_frame_pe TO op_app']
    assert not any('CREATE' in g or 'UPDATE' in g for g in grants)


def test_the_approximate_index_is_not_made_here() -> None:
    assert not any('INDEX' in s for s in _statements())


def test_the_creating_role_cannot_bypass_row_security_and_has_its_timeouts_on_the_role() -> None:
    assert 'NOBYPASSRLS' in vector_ddl.ROLE_ATTRIBUTES
    assert 'NOSUPERUSER' in vector_ddl.ROLE_ATTRIBUTES
    assert vector_ddl.ROLE_SETTINGS == {'statement_timeout': '0', 'lock_timeout': '0'}
    assert vector_ddl.role_grants() == (
        'GRANT USAGE, CREATE ON SCHEMA vec TO op_vector',
        'GRANT REFERENCES (id) ON projects TO op_vector',
    )


def test_the_job_checks_both_row_security_flags_before_a_space_is_ready() -> None:
    sql = vector_ddl.row_security_flags(12, 'frame_pe')
    assert 'relrowsecurity AND relforcerowsecurity' in sql
    assert "'vec.v12_frame_pe'::regclass" in sql


@pytest.mark.parametrize(
    ('lock_key', 'space_key', 'dim', 'role'),
    [
        (1, 'Bad', 1024, 'op_app'),
        (1, 'x', 1024, 'op_app'),
        (1, 'a' * 49, 1024, 'op_app'),
        (-1, 'frame_pe', 1024, 'op_app'),
        (1, 'frame_pe', 4, 'op_app'),
        (1, 'frame_pe', 5000, 'op_app'),
        (1, 'frame_pe', 1024, 'op_app; DROP TABLE x'),
        (1, 'frame_pe', 1024, 'Op App'),
    ],
)
def test_a_name_or_dimension_that_could_not_be_a_space_is_refused(
    lock_key: int, space_key: str, dim: int, role: str
) -> None:
    with pytest.raises(VectorDdlError):
        vector_ddl.create_statements(lock_key, space_key, dim, app_role=role)


def test_the_longest_name_a_space_can_have_fits_the_sixty_characters_of_the_plan() -> None:
    name = vector_ddl.table_name(2_147_483_647, 'a' * 48)
    assert len(name) == 60
