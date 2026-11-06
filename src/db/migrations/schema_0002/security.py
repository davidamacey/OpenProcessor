"""Row-level security and the immutability triggers of revision 0002_research_seams.

Same shape as ``schema_0001.security``: the scope functions it relies on already exist. Tables
that carry ``project_id`` get the direct policy (a NULL project is a core or deployment-default row,
visible to the deployment and core scopes). The assist loop's child tables reach the project through
``run_id``; they are bounded by the run's budget, so the per-row ``EXISTS`` is acceptable there,
unlike the unbounded membership tables.
"""

from __future__ import annotations

from alembic import op


# Tables with a nullable ``project_id`` (NULL: core faces or the deployment default).
NULLABLE_PROJECT_TABLES = (
    'cluster_calibrations',
    'cluster_partitions',
    'cluster_identities',
    'partition_clusters',
    'cluster_memberships',
    'cluster_activations',
    'cluster_activation_events',
)
PROJECT_TABLES = (
    'embedding_spaces',
    'embedding_reindex_runs',
    'embedding_evaluation_runs',
    'assist_loop_runs',
    'vlm_usage_hourly',
)
RUN_CHILD_TABLES = (
    'assist_loop_iterations',
    'assist_loop_nodes',
    'assist_loop_selections',
    'assist_loop_proposals',
    'assist_loop_reviews',
    'assist_loop_budgets',
    'assist_loop_budget_ledger',
)
# serving_reports, embedding_model_versions and cluster_backend_validations are deployment data
# read by every process: no row-level security.

FUNCTIONS = (
    'cluster_calibrations_immutable()',
    'cluster_partitions_ready_immutable()',
    'cluster_children_ready_immutable()',
    'cluster_children_project_from_parent()',
    'embedding_model_versions_immutable()',
)

# A DELETE passes when the project is gone (the cascade of a project delete) or while the
# project_delete job holds op.project_delete for a project already marked deleting. A NULL-project
# row is never deleted this way: core and default rows go through retention only.
_PROJECT_GONE = """OLD.project_id IS NOT NULL AND (
             NOT EXISTS (SELECT 1 FROM projects p WHERE p.id = OLD.project_id)
             OR op_project_delete_allows(OLD.project_id))"""

_DEFINER = 'LANGUAGE plpgsql SECURITY DEFINER SET search_path = public, pg_temp'

_FUNCTIONS_SQL = (
    """
    CREATE FUNCTION cluster_calibrations_immutable() RETURNS trigger
      @DEFINER@ AS $$
      BEGIN
        IF TG_OP = 'DELETE' AND @GONE@ THEN
          RETURN OLD;
        END IF;
        RAISE EXCEPTION 'cluster_calibration_immutable: a calibration is never changed or deleted'
          USING ERRCODE = 'integrity_constraint_violation';
      END $$
    """,
    # Once 'ready' only the status may change, and job_id / parent_partition_id may go to NULL
    # (the ON DELETE SET NULL writes of job pruning and partition retention are UPDATEs).
    """
    CREATE FUNCTION cluster_partitions_ready_immutable() RETURNS trigger
      @DEFINER@ AS $$
      DECLARE
        kept text[] := ARRAY['status', 'job_id', 'parent_partition_id'];
      BEGIN
        IF OLD.status <> 'ready' THEN
          RETURN CASE TG_OP WHEN 'DELETE' THEN OLD ELSE NEW END;
        END IF;
        IF TG_OP = 'UPDATE' THEN
          IF (to_jsonb(NEW) - kept) = (to_jsonb(OLD) - kept)
             AND (NEW.job_id IS NULL OR NEW.job_id = OLD.job_id)
             AND (NEW.parent_partition_id IS NULL
                  OR NEW.parent_partition_id = OLD.parent_partition_id) THEN
            RETURN NEW;
          END IF;
          RAISE EXCEPTION 'cluster_partition_immutable: a ready partition only changes its status'
            USING ERRCODE = 'integrity_constraint_violation';
        END IF;
        IF @GONE@ THEN
          RETURN OLD;
        END IF;
        RAISE EXCEPTION 'cluster_partition_immutable: a ready partition is not deleted; discard it first'
          USING ERRCODE = 'integrity_constraint_violation';
      END $$
    """,
    # Children of a ready partition do not change. Retention sets the parent 'discarded' first and
    # then deletes in batches; a cascade finds the parent gone; the project_delete job sets the flag.
    """
    CREATE FUNCTION cluster_children_ready_immutable() RETURNS trigger
      @DEFINER@ AS $$
      DECLARE
        parent_status text;
      BEGIN
        SELECT status INTO parent_status FROM cluster_partitions WHERE id = OLD.partition_id;
        IF parent_status IS NULL OR parent_status <> 'ready'
           OR (TG_OP = 'DELETE' AND op_project_delete_allows(OLD.project_id)) THEN
          RETURN CASE TG_OP WHEN 'DELETE' THEN OLD ELSE NEW END;
        END IF;
        RAISE EXCEPTION 'cluster_partition_immutable: % of a row of a ready partition', TG_OP
          USING ERRCODE = 'integrity_constraint_violation';
      END $$
    """,
    # The composite key to the partition is not checked when project_id is NULL (MATCH SIMPLE), so a
    # project partition could otherwise get NULL-project children that every project scope misses.
    """
    CREATE FUNCTION cluster_children_project_from_parent() RETURNS trigger
      @DEFINER@ AS $$
      BEGIN
        SELECT project_id INTO NEW.project_id FROM cluster_partitions WHERE id = NEW.partition_id;
        RETURN NEW;
      END $$
    """,
    """
    CREATE FUNCTION embedding_model_versions_immutable() RETURNS trigger
      LANGUAGE plpgsql AS $$
      DECLARE
        mutable text[] := ARRAY['status', 'engine_parity_min_cos'];
      BEGIN
        IF (to_jsonb(NEW) - mutable) IS DISTINCT FROM (to_jsonb(OLD) - mutable) THEN
          RAISE EXCEPTION 'embedding_model_version_immutable: only status and the parity change'
            USING ERRCODE = 'integrity_constraint_violation';
        END IF;
        IF OLD.engine_parity_min_cos IS NOT NULL
           AND NEW.engine_parity_min_cos IS DISTINCT FROM OLD.engine_parity_min_cos THEN
          RAISE EXCEPTION 'embedding_model_version_immutable: the parity is recorded once; a lower parity is a new version'
            USING ERRCODE = 'integrity_constraint_violation';
        END IF;
        RETURN NEW;
      END $$
    """,
)

# (trigger and function name, table, event)
_TRIGGERS = (
    ('cluster_calibrations_immutable', 'cluster_calibrations', 'UPDATE OR DELETE'),
    ('cluster_partitions_ready_immutable', 'cluster_partitions', 'UPDATE OR DELETE'),
    ('cluster_children_ready_immutable', 'partition_clusters', 'UPDATE OR DELETE'),
    ('cluster_children_ready_immutable', 'cluster_memberships', 'UPDATE OR DELETE'),
    ('cluster_children_project_from_parent', 'partition_clusters', 'INSERT'),
    ('cluster_children_project_from_parent', 'cluster_memberships', 'INSERT'),
    ('embedding_model_versions_immutable', 'embedding_model_versions', 'UPDATE'),
)


def _enable(table: str) -> None:
    op.execute(f'ALTER TABLE {table} ENABLE ROW LEVEL SECURITY')
    op.execute(f'ALTER TABLE {table} FORCE ROW LEVEL SECURITY')


def _policy(table: str, name: str, expression: str, *, command: str = 'ALL') -> None:
    check = f' WITH CHECK ({expression})' if command == 'ALL' else ''
    op.execute(f'CREATE POLICY {name} ON {table} FOR {command} USING ({expression}){check}')


def _create_policies() -> None:
    direct = '(SELECT op_scope_is_deployment()) OR project_id = (SELECT op_scope_project())'
    for table in (*NULLABLE_PROJECT_TABLES, *PROJECT_TABLES):
        _enable(table)
        expression = direct
        if table in NULLABLE_PROJECT_TABLES:
            expression += ' OR (project_id IS NULL AND (SELECT op_scope_is_core()))'
        _policy(table, f'p_{table}_project', expression)
    for table in RUN_CHILD_TABLES:
        _enable(table)
        _policy(
            table,
            f'p_{table}_project',
            f'EXISTS (SELECT 1 FROM assist_loop_runs x WHERE x.id = {table}.run_id '  # nosec B608  names from the module constants above
            'AND op_scope_allows(x.project_id))',
        )
    # A project-scoped run reads the deployment default calibration but never writes it: the
    # write policy above admits a NULL project only to the deployment and core scopes.
    _policy(
        'cluster_calibrations',
        'p_cluster_calibrations_default_read',
        'project_id IS NULL OR op_scope_allows(project_id)',
        command='SELECT',
    )


def create() -> None:
    _create_policies()
    for statement in _FUNCTIONS_SQL:
        op.execute(statement.replace('@DEFINER@', _DEFINER).replace('@GONE@', _PROJECT_GONE))
    for name, table, event in _TRIGGERS:
        op.execute(
            f'CREATE TRIGGER {name} BEFORE {event} ON {table} '
            f'FOR EACH ROW EXECUTE FUNCTION {name}()'
        )


def drop_functions() -> None:
    """Run after the tables: dropping a table drops its policies and triggers."""
    for function in FUNCTIONS:
        op.execute(f'DROP FUNCTION {function}')
