"""The statements that make one vector table per embedding space (15a section 6.1), and the role grants.

Nothing here runs a statement. The embedding-space job (a later lane) runs them as the login role
``op_vector``, on its own autocommit connection: ``CREATE INDEX CONCURRENTLY`` cannot run inside a
transaction block, and ``SET ROLE`` does not apply a role's ``ALTER ROLE`` settings
(https://www.postgresql.org/docs/18/sql-createindex.html, https://www.postgresql.org/docs/18/sql-set-role.html).
The migration ``0010_storage_record`` creates only the schema ``vec`` and, when the role exists, its
grants; the tables are made here, when a space is added.

A vector lives alone in its row (R6). ``SET STORAGE PLAIN`` keeps the 2 kB of a 1,024-dimension
``halfvec`` inline: pgvector declares ``halfvec`` with ``STORAGE = external``, so without it every row
moves to the TOAST table, and an exact scan or a bulk read pays two extra page reads per row
(15a section 6.2). The approximate index is gate B's and is not made here.
"""

from __future__ import annotations

import re


SCHEMA = 'vec'
OWNER_ROLE = 'op_vector'
# The role of 15a section 6.1: a login role that is granted to nobody and cannot bypass row security.
# Its timeouts are set on the role (applied at login), not by SET ROLE.
ROLE_ATTRIBUTES = 'LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOBYPASSRLS'
ROLE_SETTINGS = {'statement_timeout': '0', 'lock_timeout': '0'}
MAX_TABLE_NAME = 60
_SPACE_KEY = re.compile(r'^[a-z][a-z0-9_]{2,47}$')  # the check on embedding_spaces.space_key
_IDENTIFIER = re.compile(r'^[a-z_][a-z0-9_]{0,62}$')
_POLICY = '(SELECT op_scope_is_deployment()) OR project_id = (SELECT op_scope_project())'


class VectorDdlError(ValueError):
    """A space that cannot have a table: its key, its dimension or a role name is not valid."""


def table_name(lock_key: int, space_key: str) -> str:
    """``v<lock_key>_<space_key>``, at most 60 characters (a project's ``lock_key`` is an integer)."""
    if lock_key < 0 or not _SPACE_KEY.fullmatch(space_key):
        raise VectorDdlError(f'not a space: lock_key={lock_key!r} space_key={space_key!r}')
    name = f'v{lock_key}_{space_key}'
    if len(name) > MAX_TABLE_NAME:
        raise VectorDdlError(f'table name {name!r} is longer than {MAX_TABLE_NAME} characters')
    return name


def qualified(lock_key: int, space_key: str) -> str:
    return f'{SCHEMA}.{table_name(lock_key, space_key)}'


def create_statements(lock_key: int, space_key: str, dim: int, *, app_role: str) -> tuple[str, ...]:
    """The table, its storage setting, row-level security (enabled and forced), the project policy
    and the application role's rights, in the order the job runs them."""
    if not 8 <= dim <= 4096:
        raise VectorDdlError(f'dimension {dim} is outside 8..4096')
    if not _IDENTIFIER.fullmatch(app_role):
        raise VectorDdlError(f'not a role name: {app_role!r}')
    table = qualified(lock_key, space_key)
    return (
        (
            f'CREATE TABLE {table} ('  # nosec B608  names are validated above, dim is an int
            'subject_key bigint PRIMARY KEY, '
            'project_id uuid NOT NULL REFERENCES projects(id) ON DELETE CASCADE, '
            'input_rev integer NOT NULL DEFAULT 0, '
            f'embedding halfvec({int(dim)}) NOT NULL)'
        ),
        f'ALTER TABLE {table} ALTER COLUMN embedding SET STORAGE PLAIN',
        f'ALTER TABLE {table} ENABLE ROW LEVEL SECURITY',
        f'ALTER TABLE {table} FORCE ROW LEVEL SECURITY',
        # the policy name is per table; the 02a form ``p_<table>_project`` would pass 63 characters
        f'CREATE POLICY p_vec_project ON {table} FOR ALL USING ({_POLICY}) WITH CHECK ({_POLICY})',
        f'GRANT SELECT, INSERT, DELETE ON {table} TO {app_role}',
    )


def drop_statement(lock_key: int, space_key: str) -> str:
    """The retire step: a short transaction with ``SET LOCAL lock_timeout`` retries this (15a 6.3)."""
    return f'DROP TABLE {qualified(lock_key, space_key)}'  # nosec B608  names validated by table_name


def row_security_flags(lock_key: int, space_key: str) -> str:
    """The query the job runs before a space becomes ``ready``, and the schema test on every table of
    ``vec``: both flags must be true, because an owner could turn ``FORCE`` off."""
    return (
        'SELECT relrowsecurity AND relforcerowsecurity FROM pg_class '  # nosec B608  validated name
        f"WHERE oid = '{qualified(lock_key, space_key)}'::regclass"
    )


def role_grants() -> tuple[str, ...]:
    """What the role needs from the schema owner: tables in ``vec``, and REFERENCES on ``projects`` for
    the project foreign key of the template. Run by the bootstrap that creates the role."""
    return (
        f'GRANT USAGE, CREATE ON SCHEMA {SCHEMA} TO {OWNER_ROLE}',
        f'GRANT REFERENCES (id) ON projects TO {OWNER_ROLE}',
    )
