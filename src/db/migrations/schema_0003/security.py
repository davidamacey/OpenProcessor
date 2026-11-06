"""Row-level security and the project-copy trigger of revision 0003_near_duplicates.

Every table carries ``project_id`` and gets the direct policy (02a section 1.4): a NULL project
is the core global index, visible to the deployment and core scopes. Groups and members copy
``project_id`` from their run in a BEFORE INSERT trigger, because a composite key is not checked
when ``project_id`` is NULL (MATCH SIMPLE) and a forged project could otherwise hide a row.
"""

from __future__ import annotations

from alembic import op


NULLABLE_PROJECT_TABLES = (
    'image_contents',
    'near_dup_runs',
    'near_dup_groups',
    'near_dup_members',
    'near_dup_edges',
    'near_dup_pending',
)
PROJECT_TABLES = ('near_dup_decisions', 'holdout_near_dup_moves')
CHILD_TABLES = ('near_dup_groups', 'near_dup_members')
FUNCTION = 'near_dup_children_project_from_run()'

_FUNCTION_SQL = """
    CREATE FUNCTION near_dup_children_project_from_run() RETURNS trigger
      LANGUAGE plpgsql SECURITY DEFINER SET search_path = public, pg_temp AS $$
      BEGIN
        SELECT project_id INTO NEW.project_id FROM near_dup_runs WHERE id = NEW.run_id;
        RETURN NEW;
      END $$
"""


def create() -> None:
    direct = '(SELECT op_scope_is_deployment()) OR project_id = (SELECT op_scope_project())'
    for table in (*NULLABLE_PROJECT_TABLES, *PROJECT_TABLES):
        expression = direct
        if table in NULLABLE_PROJECT_TABLES:
            expression += ' OR (project_id IS NULL AND (SELECT op_scope_is_core()))'
        op.execute(f'ALTER TABLE {table} ENABLE ROW LEVEL SECURITY')
        op.execute(f'ALTER TABLE {table} FORCE ROW LEVEL SECURITY')
        op.execute(
            f'CREATE POLICY p_{table}_project ON {table} FOR ALL '
            f'USING ({expression}) WITH CHECK ({expression})'
        )
    op.execute(_FUNCTION_SQL)
    for table in CHILD_TABLES:
        op.execute(
            f'CREATE TRIGGER near_dup_children_project_from_run BEFORE INSERT ON {table} '
            'FOR EACH ROW EXECUTE FUNCTION near_dup_children_project_from_run()'
        )


def drop_function() -> None:
    """Run after the tables: dropping a table drops its policies and triggers."""
    op.execute(f'DROP FUNCTION {FUNCTION}')
