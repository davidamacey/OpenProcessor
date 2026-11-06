"""Row-level security, the scope functions and the triggers of revision 0001.

Autogenerate sees none of these, so they are written out. The jobs transition guard is built
from ``src.domain.job_states`` so the table in Python and the trigger cannot differ.
"""

from __future__ import annotations

from alembic import op

from src.domain import job_states


# table -> the column that holds its project (default ``project_id``)
DIRECT_POLICY_TABLES: dict[str, str] = {
    'project_indexes': 'project_id',
    'class_registries': 'project_id',
    'project_classes': 'project_id',
    'search_projection_tasks': 'project_id',
    'config_documents': 'project_id',
    'config_activations': 'project_id',
    'config_activation_events': 'project_id',
    'config_runtime_reports': 'project_id',
    'project_settings': 'project_id',
    'project_keymaps': 'project_id',
    'project_state': 'project_id',
    'projection_states': 'project_id',
    'jobs': 'project_id',
    'events': 'project_id',
    'stream_worker_claims': 'project_id',
    'dataset_uploads': 'project_id',
    'dataset_imports': 'project_id',
    'ingest_sources': 'project_id',
    'label_operations': 'project_id',
    'holdout_freezes': 'project_id',
    'dataset_exports': 'project_id',
    'training_runs': 'project_id',
    'models': 'project_id',
    'promotions': 'project_id',
    'bakeoff_runs': 'project_id',
    'combine_runs': 'target_project_id',
}

# child table -> (its column, parent table, parent key, parent project column)
CHILD_POLICY_TABLES: dict[str, tuple[str, str, str, str]] = {
    'job_attempts': ('job_id', 'jobs', 'id', 'project_id'),
    'job_items': ('job_id', 'jobs', 'id', 'project_id'),
    'gpu_reservations': ('holder_job_id', 'jobs', 'id', 'project_id'),
    'dataset_import_images': ('import_id', 'dataset_imports', 'id', 'project_id'),
    'combine_sources': ('job_id', 'combine_runs', 'job_id', 'target_project_id'),
    'ingest_source_files': ('source_id', 'ingest_sources', 'id', 'project_id'),
    'label_operation_members': ('op_id', 'label_operations', 'id', 'project_id'),
    'config_revisions': ('config_id', 'config_documents', 'id', 'project_id'),
    'vlm_endpoint_probes': ('config_id', 'config_documents', 'id', 'project_id'),
}

SCOPE_FUNCTIONS = (
    'op_scope_project()',
    'op_scope_is_deployment()',
    'op_scope_allows(uuid)',
    'op_model_project(uuid)',
)

_SCOPE_FUNCTIONS_SQL = (
    # An unset scope raises instead of returning no rows: a missing binding shows up as an error,
    # never as a silently empty answer.
    """
    CREATE FUNCTION op_scope_project() RETURNS uuid
      LANGUAGE plpgsql STABLE AS $$
      BEGIN
        CASE current_setting('op.scope', true)
          WHEN 'project'    THEN RETURN nullif(current_setting('op.project_id', true), '')::uuid;
          WHEN 'deployment' THEN RETURN NULL;
          ELSE RAISE EXCEPTION 'scope_not_set' USING ERRCODE = 'insufficient_privilege';
        END CASE;
      END $$
    """,
    """
    CREATE FUNCTION op_scope_is_deployment() RETURNS boolean
      LANGUAGE sql STABLE AS $$
      SELECT coalesce(current_setting('op.scope', true) = 'deployment', false) $$
    """,
    """
    CREATE FUNCTION op_scope_allows(pid uuid) RETURNS boolean
      LANGUAGE sql STABLE AS $$
      SELECT op_scope_is_deployment() OR pid = op_scope_project() $$
    """,
    # Reads models past row-level security, so the policies of model_shares and model_engines can
    # name a model's owner without a policy cycle (models' shared-read policy reads model_shares).
    """
    CREATE FUNCTION op_model_project(mid uuid) RETURNS uuid
      LANGUAGE sql STABLE SECURITY DEFINER SET search_path = public, pg_temp AS $$
      SELECT project_id FROM models WHERE id = mid $$
    """,
)


def _project_expr(column: str) -> str:
    return f'(SELECT op_scope_is_deployment()) OR {column} = (SELECT op_scope_project())'


def _enable(table: str) -> None:
    op.execute(f'ALTER TABLE {table} ENABLE ROW LEVEL SECURITY')
    op.execute(f'ALTER TABLE {table} FORCE ROW LEVEL SECURITY')


def _policy(table: str, name: str, expression: str, *, command: str = 'ALL') -> None:
    check = f' WITH CHECK ({expression})' if command == 'ALL' else ''
    op.execute(f'CREATE POLICY {name} ON {table} FOR {command} USING ({expression}){check}')


def _create_policies() -> None:
    for table, column in DIRECT_POLICY_TABLES.items():
        _enable(table)
        _policy(table, f'p_{table}_project', _project_expr(column))
    for table, (column, parent, key, parent_project) in CHILD_POLICY_TABLES.items():
        _enable(table)
        _policy(
            table,
            f'p_{table}_project',
            f'EXISTS (SELECT 1 FROM {parent} x WHERE x.{key} = {table}.{column} '  # nosec B608  names from the module constants above
            f'AND op_scope_allows(x.{parent_project}))',
        )
    # Shares: only the model's owner writes them; the consumer only reads its own.
    _enable('model_shares')
    _policy('model_shares', 'p_model_shares_owner', 'op_scope_allows(op_model_project(model_id))')
    _policy(
        'model_shares',
        'p_model_shares_consumer',
        'op_scope_allows(consumer_project_id)',
        command='SELECT',
    )
    _policy(
        'models',
        'p_models_shared',
        'EXISTS (SELECT 1 FROM model_shares s WHERE s.model_id = models.id '
        'AND op_scope_allows(s.consumer_project_id))',
        command='SELECT',
    )
    _enable('model_engines')
    _policy(
        'model_engines', 'p_model_engines_project', 'op_scope_allows(op_model_project(model_id))'
    )
    _policy(
        'model_engines',
        'p_model_engines_shared',
        'EXISTS (SELECT 1 FROM models m WHERE m.id = model_engines.model_id)',
        command='SELECT',
    )


def _transition_guard_sql() -> str:
    pairs = ', '.join(f"('{a}', '{b}')" for a, b in sorted(job_states.TRANSITIONS))
    return f"""
    CREATE FUNCTION jobs_status_transition_guard() RETURNS trigger
      LANGUAGE plpgsql AS $$
      BEGIN
        IF (OLD.status, NEW.status) NOT IN ({pairs}) THEN
          RAISE EXCEPTION 'invalid_job_transition: % -> %', OLD.status, NEW.status
            USING ERRCODE = 'check_violation';
        END IF;
        RETURN NEW;
      END $$
    """


_REVISIONS_IMMUTABLE_SQL = """
    CREATE FUNCTION config_revisions_immutable() RETURNS trigger
      LANGUAGE plpgsql SECURITY DEFINER SET search_path = public, pg_temp AS $$
      BEGIN
        IF TG_OP = 'DELETE' AND NOT EXISTS (SELECT 1 FROM config_documents d WHERE d.id = OLD.config_id) THEN
          RETURN OLD;
        END IF;
        RAISE EXCEPTION 'config_revision_immutable: revisions are never changed or deleted'
          USING ERRCODE = 'integrity_constraint_violation';
      END $$
"""


def _create_triggers() -> None:
    op.execute(_transition_guard_sql())
    op.execute(
        'CREATE TRIGGER jobs_status_transition_guard BEFORE UPDATE OF status ON jobs '
        'FOR EACH ROW WHEN (OLD.status IS DISTINCT FROM NEW.status) '
        'EXECUTE FUNCTION jobs_status_transition_guard()'
    )
    op.execute(_REVISIONS_IMMUTABLE_SQL)
    op.execute(
        'CREATE TRIGGER config_revisions_immutable BEFORE UPDATE OR DELETE ON config_revisions '
        'FOR EACH ROW EXECUTE FUNCTION config_revisions_immutable()'
    )


def create() -> None:
    for statement in _SCOPE_FUNCTIONS_SQL:
        op.execute(statement)
    _create_policies()
    _create_triggers()


def drop_cross_table_policies() -> None:
    """The one policy that reads a table created after its own (models read model_shares)."""
    op.execute('DROP POLICY p_models_shared ON models')


def drop_functions() -> None:
    """Run after the tables: dropping a table drops its policies and triggers."""
    for function in ('jobs_status_transition_guard()', 'config_revisions_immutable()'):
        op.execute(f'DROP FUNCTION {function}')
    for function in reversed(SCOPE_FUNCTIONS):
        op.execute(f'DROP FUNCTION {function}')
