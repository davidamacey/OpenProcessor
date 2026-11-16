"""Row-level security, the append-only trigger and its erasure guard, and the bootstrap check.

Policies are the direct form of 02a section 1.4 on every table with a ``project_id``. The claim log
refuses UPDATE and DELETE except under the guards of 15a section 8: the project's delete job
(``op_project_delete_allows``) or a running erasure job of the project (``op_erasure_allows``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from alembic import op

from src.config.database import VECTOR_USER
from src.domain.claim_kinds import KINDS


if TYPE_CHECKING:
    from src.db.migrations.schema_0010.ddl import Table


PROJECT_EXPRESSION = '(SELECT op_scope_is_deployment()) OR project_id = (SELECT op_scope_project())'
FUNCTIONS = ('claims_append_only()', 'analysers_immutable()', 'op_erasure_allows(uuid)')

# True only inside a running erasure job of the project: the job names its row in the
# transaction-local setting ``op.erasure_job`` (same pattern as ``op.project_delete``, 02a 1.2).
# The status is what a forged setting cannot fake: every app process is one role.
_ERASURE_ALLOWS_SQL = """
    CREATE FUNCTION op_erasure_allows(pid uuid) RETURNS boolean
      LANGUAGE sql STABLE AS $$
      SELECT pid IS NOT NULL
        AND coalesce(current_setting('op.erasure_job', true), '') <> ''
        AND EXISTS (SELECT 1 FROM erasure_jobs e
                    WHERE e.id::text = current_setting('op.erasure_job', true)
                      AND e.project_id = pid AND e.state = 'running') $$
"""

_CLAIMS_APPEND_ONLY_SQL = """
    CREATE FUNCTION claims_append_only() RETURNS trigger
      LANGUAGE plpgsql AS $$
      BEGIN
        -- the three cases 02a 1.2 (RN-1) and 15a section 8 allow: the project's delete job, a running
        -- erasure job of the project, and the cascade of a project row that is already gone
        IF TG_OP = 'DELETE' AND (op_project_delete_allows(OLD.project_id)
                                 OR op_erasure_allows(OLD.project_id)
                                 OR NOT EXISTS (SELECT 1 FROM projects p WHERE p.id = OLD.project_id)) THEN
          RETURN OLD;
        END IF;
        RAISE EXCEPTION 'claims_append_only: a claim is never changed or deleted (%)', TG_OP
          USING ERRCODE = 'integrity_constraint_violation';
      END $$
"""

_ANALYSERS_IMMUTABLE_SQL = """
    CREATE FUNCTION analysers_immutable() RETURNS trigger
      LANGUAGE plpgsql AS $$
      BEGIN
        RAISE EXCEPTION 'analyser_immutable: a new model build or configuration is a new row (%)', TG_OP
          USING ERRCODE = 'integrity_constraint_violation';
      END $$
"""


def create_policies(tables: tuple[Table, ...]) -> None:
    for table in tables:
        if not any(c.startswith('project_id ') for c in table.columns):
            continue
        t = table.name
        op.execute(f'ALTER TABLE {t} ENABLE ROW LEVEL SECURITY')
        op.execute(f'ALTER TABLE {t} FORCE ROW LEVEL SECURITY')
        op.execute(
            f'CREATE POLICY p_{t}_project ON {t} FOR ALL '
            f'USING ({PROJECT_EXPRESSION}) WITH CHECK ({PROJECT_EXPRESSION})'
        )


def create_functions_and_triggers() -> None:
    # erasure_jobs exists by now: op_erasure_allows reads it, and a SQL function body is checked
    # against the catalog when it is created.
    op.execute(_ERASURE_ALLOWS_SQL)
    op.execute(_CLAIMS_APPEND_ONLY_SQL)
    op.execute(
        'CREATE TRIGGER claims_append_only BEFORE UPDATE OR DELETE ON claims '
        'FOR EACH ROW EXECUTE FUNCTION claims_append_only()'
    )
    op.execute(_ANALYSERS_IMMUTABLE_SQL)
    op.execute(
        'CREATE TRIGGER analysers_immutable BEFORE UPDATE OR DELETE ON analysers '
        'FOR EACH ROW EXECUTE FUNCTION analysers_immutable()'
    )


def drop_functions() -> None:
    """Run after the tables: dropping a table drops its triggers."""
    for function in FUNCTIONS:
        op.execute(f'DROP FUNCTION {function}')


def _quoted(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def seed_claim_kinds() -> None:
    """The rows of ``claim_kinds`` are :data:`src.domain.claim_kinds.KINDS`: one list, read here.

    ``ON CONFLICT DO NOTHING``: a later revision that adds a kind inserts it the same way, so a
    kind the catalogue gained since this revision was written is not inserted twice.
    """
    rows = ',\n  '.join(
        f'({_quoted(k.value)}, {_quoted(i.subject_kind.value)}, {_quoted(i.layer.value)}, '
        f'{str(i.pins_geometry).lower()}, {str(i.asserts_object).lower()})'
        for k, i in sorted(KINDS.items(), key=lambda kv: kv[0].value)
    )
    op.execute(
        'INSERT INTO claim_kinds (kind, subject_kind, layer, pins_geometry, asserts_object) VALUES\n'
        f'  {rows}\nON CONFLICT (kind) DO NOTHING'  # nosec B608  rows come from the closed domain catalogue, quoted by _quoted
    )


_BOOTSTRAP = """
DO $$
DECLARE
  hint constant text := 'run the db-migrate step (python -m scripts.db.migrate upgrade), whose bootstrap (src/db/roles.py prepare_database) runs as the superuser';
  owner_name text;
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'pg_trgm') THEN
    RAISE EXCEPTION 'extension pg_trgm is not installed in database %: needed for the trigram indexes of text search (15a section 2.6); %', current_database(), hint
      USING ERRCODE = 'undefined_object';
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector') THEN
    RAISE EXCEPTION 'extension vector is not installed in database %: needed for the per-space vector tables (15a section 6); the pinned pgvector/pgvector image ships it; %', current_database(), hint
      USING ERRCODE = 'undefined_object';
  END IF;
  SELECT nspowner::regrole::text INTO owner_name FROM pg_namespace WHERE nspname = 'vec';
  IF owner_name IS NULL THEN
    RAISE EXCEPTION 'schema vec does not exist in database %: the owner role has no CREATE on the database, so it cannot make it; %', current_database(), hint
      USING ERRCODE = 'undefined_schema';
  ELSIF owner_name <> current_user THEN
    RAISE EXCEPTION 'schema vec is owned by %, not by the migration role %: it could not grant on it; %', owner_name, current_user, hint
      USING ERRCODE = 'insufficient_privilege';
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '@VECTOR_ROLE@') THEN
    RAISE EXCEPTION 'role @VECTOR_ROLE@ does not exist: it makes the per-space vector tables (15a section 6.1) and only a superuser can create it; %', hint
      USING ERRCODE = 'undefined_object';
  END IF;
END $$
"""


def require_bootstrap() -> None:
    """Check what the superuser bootstrap supplies, and fail naming the missing piece.

    The migration role has no ``CREATE`` on the database, so it can neither create a schema nor
    install an extension (PostgreSQL 18: ``CREATE SCHEMA`` and ``CREATE EXTENSION`` require
    ``CREATE`` on the database, https://www.postgresql.org/docs/18/sql-createschema.html,
    https://www.postgresql.org/docs/18/sql-createextension.html), and only a superuser creates a
    role. 02a section 1.5: those steps are the bootstrap's (``src/db/roles.py``, shared by the
    installer, production and the test fixtures through ``scripts/db/migrate.py``); the grants on
    ``vec`` and ``projects`` to the vector role are ``roles.apply_grants``'s.

    Deviation 1 is closed: the compose files pin the pgvector project's image
    (``pgvector/pgvector:0.8.7-pg18-trixie``, 01_package_pins.md), so a missing ``vector`` extension
    fails here like the others. Nothing in this revision uses the extension.
    """
    op.execute(_BOOTSTRAP.replace('@VECTOR_ROLE@', VECTOR_USER))
