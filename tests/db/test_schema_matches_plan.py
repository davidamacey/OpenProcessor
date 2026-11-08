"""The migrated database equals the plan of docs/design/v070_rebuild/02a_postgres_schema.md.

The expectations below are written out by hand from the plan, not read from the models, so a
model and a migration that drift together still fail here. ``alembic check`` cannot see
policies, triggers, functions, storage parameters or the partial-index predicates.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import _near_dup_plan as near_dup
import _research_plan as research
import psycopg
import pytest


if TYPE_CHECKING:
    from fixtures.postgres import PgDatabase

pytestmark = pytest.mark.db

TABLES = {
    'project_slugs', 'projects', 'project_indexes', 'deployment_counters', 'deployment_settings',
    'class_registries', 'project_classes', 'search_projection_tasks',
    'config_documents', 'config_revisions', 'config_activations', 'config_activation_events',
    'vlm_endpoint_probes', 'config_runtime_reports', 'project_settings', 'project_keymaps',
    'project_state', 'projection_states',
    'jobs', 'job_attempts', 'job_items', 'events', 'stream_worker_claims', 'worker_capabilities',
    'gpu_devices', 'gpu_reservations',
    'dataset_uploads', 'dataset_imports', 'dataset_import_images', 'combine_runs',
    'combine_sources', 'ingest_sources', 'ingest_source_files', 'label_operations',
    'label_operation_members', 'holdout_freezes',
    'dataset_exports', 'training_runs', 'models', 'model_engines', 'promotions', 'model_shares',
    'bakeoff_runs', 'upgrade_runs',
}  # fmt: skip

UNIQUE_CONSTRAINTS = {
    'uq_projects_lock_key', 'uq_projects_slug', 'uq_project_indexes_index_name',
    'uq_config_documents_project_id_kind_name', 'uq_config_documents_id_kind',
    'uq_job_items_job_id_item_id', 'uq_dataset_uploads_storage_key', 'uq_dataset_imports_job_id',
    'uq_ingest_sources_project_id_name', 'uq_training_runs_job_id',
    'uq_models_project_id_name_version', 'uq_model_engines_build_key',
}  # fmt: skip

PARTIAL_INDEXES = {
    'uq_project_classes_hotkey', 'uq_project_classes_active_name',
    'ix_search_projection_tasks_pending', 'uq_jobs_one_active', 'uq_jobs_idempotency',
    'ix_jobs_live', 'ix_jobs_queued', 'ix_jobs_interrupted', 'ix_events_cursor_global',
    'ix_dataset_import_images_pending', 'ix_ingest_source_files_pending', 'uq_holdout_current',
    'uq_promotions_live', 'uq_upgrade_step_done',
}  # fmt: skip

PLAIN_INDEXES = {
    'ix_project_classes_name_key', 'ix_config_activation_events_axis', 'ix_jobs_project_recent',
    'ix_events_cursor', 'ix_events_created',
}  # fmt: skip

NULLS_NOT_DISTINCT = {
    'uq_config_documents_project_id_kind_name', 'uq_models_project_id_name_version',
    'uq_jobs_one_active', 'uq_jobs_idempotency',
}  # fmt: skip

# CHECKs a later revision alters: explicit, stable names.
NAMED_CHECKS = {
    'ck_config_documents_kind', 'ck_config_documents_scope', 'ck_config_activations_axis',
    'ck_config_activations_axis_kind', 'ck_search_projection_tasks_kind', 'ck_project_state_key',
}  # fmt: skip

FUNCTIONS = {
    'op_scope_project', 'op_scope_is_deployment', 'op_scope_is_core', 'op_scope_allows',
    'op_project_delete_allows', 'op_model_project',
    'jobs_status_transition_guard', 'config_revisions_immutable',
}  # fmt: skip

TRIGGERS = {
    ('jobs', 'jobs_status_transition_guard'),
    ('config_revisions', 'config_revisions_immutable'),
}

POLICIES = {
    ('model_shares', 'p_model_shares_owner'),
    ('model_shares', 'p_model_shares_consumer'),
    ('models', 'p_models_project'),
    ('models', 'p_models_shared'),
    ('model_engines', 'p_model_engines_project'),
    ('model_engines', 'p_model_engines_shared'),
}
_SCOPED = {
    'project_indexes', 'class_registries', 'project_classes', 'search_projection_tasks',
    'config_documents', 'config_activations', 'config_activation_events',
    'config_runtime_reports', 'project_settings', 'project_keymaps', 'project_state',
    'projection_states', 'jobs', 'events', 'stream_worker_claims', 'dataset_uploads',
    'dataset_imports', 'ingest_sources', 'label_operations', 'holdout_freezes',
    'dataset_exports', 'training_runs', 'promotions', 'bakeoff_runs', 'combine_runs',
    'job_attempts', 'job_items', 'gpu_reservations', 'dataset_import_images',
    'combine_sources', 'ingest_source_files', 'label_operation_members', 'config_revisions',
    'vlm_endpoint_probes',
}  # fmt: skip
POLICIES |= {(t, f'p_{t}_project') for t in _SCOPED}

# Tables with no row-level security: deployment data read by every process.
UNPROTECTED = {
    'project_slugs', 'projects', 'deployment_counters', 'deployment_settings',
    'worker_capabilities', 'gpu_devices', 'upgrade_runs',
}  # fmt: skip

# (table, column, referenced table, ON DELETE) as PostgreSQL stores it: c cascade, n set null,
# a no action, r restrict.
FOREIGN_KEYS = {
    ('bakeoff_runs', 'job_id', 'jobs', 'c'), ('bakeoff_runs', 'project_id', 'projects', 'c'),
    ('class_registries', 'project_id', 'projects', 'c'), ('combine_runs', 'job_id', 'jobs', 'c'),
    ('combine_runs', 'target_project_id', 'projects', 'c'),
    ('combine_sources', 'job_id', 'combine_runs', 'c'),
    ('combine_sources', 'source_project_id', 'projects', 'n'),
    ('config_activation_events', 'project_id', 'projects', 'c'),
    ('config_activations', 'config_id', 'config_documents', 'a'),
    ('config_activations', 'config_id', 'config_revisions', 'a'),
    ('config_activations', 'project_id', 'projects', 'c'),
    ('config_documents', 'cloned_from_id', 'config_documents', 'n'),
    ('config_documents', 'project_id', 'projects', 'c'),
    ('config_revisions', 'config_id', 'config_documents', 'c'),
    ('config_runtime_reports', 'project_id', 'projects', 'c'),
    ('dataset_exports', 'job_id', 'jobs', 'n'), ('dataset_exports', 'project_id', 'projects', 'c'),
    ('dataset_import_images', 'import_id', 'dataset_imports', 'c'),
    ('dataset_imports', 'job_id', 'jobs', 'n'), ('dataset_imports', 'project_id', 'projects', 'c'),
    ('dataset_imports', 'undo_job_id', 'jobs', 'n'),
    ('dataset_imports', 'upload_id', 'dataset_uploads', 'n'),
    ('dataset_uploads', 'extract_job_id', 'jobs', 'n'),
    ('dataset_uploads', 'project_id', 'projects', 'c'),
    ('events', 'job_id', 'jobs', 'c'), ('events', 'project_id', 'projects', 'c'),
    ('gpu_reservations', 'gpu_uuid', 'gpu_devices', 'r'),
    ('gpu_reservations', 'holder_job_id', 'jobs', 'c'),
    ('holdout_freezes', 'project_id', 'projects', 'c'),
    ('ingest_source_files', 'job_id', 'jobs', 'n'),
    ('ingest_source_files', 'source_id', 'ingest_sources', 'c'),
    ('ingest_sources', 'project_id', 'projects', 'c'),
    ('job_attempts', 'job_id', 'jobs', 'c'), ('job_items', 'job_id', 'jobs', 'c'),
    ('jobs', 'parent_job_id', 'jobs', 'n'), ('jobs', 'project_id', 'projects', 'c'),
    ('label_operation_members', 'op_id', 'label_operations', 'c'),
    ('label_operations', 'job_id', 'jobs', 'n'),
    ('label_operations', 'project_id', 'projects', 'c'),
    ('model_engines', 'build_job_id', 'jobs', 'n'), ('model_engines', 'model_id', 'models', 'c'),
    ('models', 'project_id', 'projects', 'c'),
    ('models', 'training_run_id', 'training_runs', 'n'),
    ('model_shares', 'consumer_project_id', 'projects', 'c'),
    ('model_shares', 'model_id', 'models', 'c'),
    ('project_classes', 'project_id', 'class_registries', 'c'),
    ('project_classes', 'project_id', 'project_classes', 'a'),
    ('project_indexes', 'project_id', 'projects', 'c'),
    ('projection_states', 'job_id', 'jobs', 'n'), ('projection_states', 'project_id', 'projects', 'c'),
    ('project_keymaps', 'project_id', 'projects', 'c'), ('projects', 'slug', 'project_slugs', 'a'),
    ('project_settings', 'project_id', 'projects', 'c'),
    ('project_state', 'project_id', 'projects', 'c'),
    ('promotions', 'job_id', 'jobs', 'n'), ('promotions', 'model_id', 'models', 'a'),
    ('promotions', 'project_id', 'projects', 'c'),
    ('search_projection_tasks', 'job_id', 'jobs', 'n'),
    ('search_projection_tasks', 'project_id', 'projects', 'c'),
    ('stream_worker_claims', 'project_id', 'projects', 'c'),
    ('training_runs', 'dataset_export_id', 'dataset_exports', 'n'),
    ('training_runs', 'job_id', 'jobs', 'a'), ('training_runs', 'project_id', 'projects', 'c'),
    ('vlm_endpoint_probes', 'config_id', 'config_documents', 'c'),
    ('vlm_endpoint_probes', 'config_id', 'config_revisions', 'c'),
}  # fmt: skip


# The two plan files are one contract: 02d's objects (revision 0002) join 02a's here.
TABLES |= research.TABLES
UNIQUE_CONSTRAINTS |= research.UNIQUE_CONSTRAINTS
PARTIAL_INDEXES |= research.PARTIAL_INDEXES
PLAIN_INDEXES |= research.PLAIN_INDEXES
NULLS_NOT_DISTINCT |= research.NULLS_NOT_DISTINCT
NAMED_CHECKS |= research.NAMED_CHECKS
FUNCTIONS |= research.FUNCTIONS
TRIGGERS |= research.TRIGGERS
POLICIES |= research.POLICIES
UNPROTECTED |= research.UNPROTECTED
FOREIGN_KEYS |= research.FOREIGN_KEYS
TABLES |= near_dup.TABLES
UNIQUE_CONSTRAINTS |= near_dup.UNIQUE_CONSTRAINTS
PARTIAL_INDEXES |= near_dup.PARTIAL_INDEXES
PLAIN_INDEXES |= near_dup.PLAIN_INDEXES
NULLS_NOT_DISTINCT |= near_dup.NULLS_NOT_DISTINCT
NAMED_CHECKS |= near_dup.NAMED_CHECKS
FUNCTIONS |= near_dup.FUNCTIONS
TRIGGERS |= near_dup.TRIGGERS
POLICIES |= near_dup.POLICIES
UNPROTECTED |= near_dup.UNPROTECTED
FOREIGN_KEYS |= near_dup.FOREIGN_KEYS


@pytest.fixture(scope='module')
def catalog(pg: PgDatabase):
    with psycopg.connect(pg.conninfo('super'), autocommit=True) as conn:
        yield conn


def _set(conn, query: str, *params) -> set:
    return {tuple(r) if len(r) > 1 else r[0] for r in conn.execute(query, params).fetchall()}


def test_exactly_the_planned_tables_exist(catalog) -> None:
    found = _set(
        catalog,
        "SELECT tablename FROM pg_tables WHERE schemaname = 'public' AND tablename <> 'alembic_version'",
    )
    assert found == TABLES


def test_unique_constraints(catalog) -> None:
    assert (
        _set(
            catalog,
            "SELECT conname FROM pg_constraint WHERE contype = 'u' AND connamespace = 'public'::regnamespace",
        )
        == UNIQUE_CONSTRAINTS
    )


def test_partial_and_plain_indexes(catalog) -> None:
    rows = catalog.execute(
        "SELECT indexname, indexdef FROM pg_indexes WHERE schemaname = 'public' "
        "AND indexname ~ '^(ix|uq)_'"
    ).fetchall()
    partial = {name for name, definition in rows if ' WHERE ' in definition}
    plain = {name for name, definition in rows if ' WHERE ' not in definition}
    assert partial == PARTIAL_INDEXES
    assert plain >= PLAIN_INDEXES


def test_singleton_indexes_treat_nulls_as_equal(catalog) -> None:
    found = _set(
        catalog,
        'SELECT i.relname FROM pg_index x JOIN pg_class i ON i.oid = x.indexrelid '
        'WHERE x.indnullsnotdistinct',
    )
    assert found == NULLS_NOT_DISTINCT


def test_the_active_set_predicate_comes_from_the_state_table(catalog) -> None:
    definition = catalog.execute(
        "SELECT indexdef FROM pg_indexes WHERE indexname = 'uq_jobs_one_active'"
    ).fetchone()[0]
    for status in ('queued', 'running', 'paused', 'cancelling'):
        assert f"'{status}'" in definition
    for status in ('interrupted', 'completed', 'failed', 'cancelled'):
        assert f"'{status}'" not in definition


def test_the_checks_a_later_revision_alters_have_stable_names(catalog) -> None:
    assert _set(catalog, "SELECT conname FROM pg_constraint WHERE contype = 'c'") >= NAMED_CHECKS


def test_foreign_keys_and_their_delete_rules(catalog) -> None:
    found = _set(
        catalog,
        'SELECT c.conrelid::regclass::text, a.attname, c.confrelid::regclass::text, c.confdeltype::text '
        'FROM pg_constraint c JOIN pg_attribute a ON a.attrelid = c.conrelid AND a.attnum = c.conkey[1] '
        "WHERE c.contype = 'f'",
    )
    assert found == FOREIGN_KEYS


def test_functions_and_triggers(catalog) -> None:
    assert (
        _set(catalog, "SELECT proname FROM pg_proc WHERE pronamespace = 'public'::regnamespace")
        >= FUNCTIONS
    )
    assert (
        _set(
            catalog,
            'SELECT tgrelid::regclass::text, tgname FROM pg_trigger WHERE NOT tgisinternal',
        )
        == TRIGGERS
    )


def test_policies(catalog) -> None:
    assert _set(catalog, 'SELECT tablename, policyname FROM pg_policies') == POLICIES


def test_every_table_with_a_project_column_has_forced_row_level_security(catalog) -> None:
    protected = _set(
        catalog,
        "SELECT relname FROM pg_class WHERE relnamespace = 'public'::regnamespace "
        "AND relkind = 'r' AND relrowsecurity AND relforcerowsecurity",
    )
    with_project_column = _set(
        catalog,
        "SELECT table_name FROM information_schema.columns WHERE table_schema = 'public' "
        "AND column_name IN ('project_id', 'target_project_id')",
    )
    assert protected == TABLES - UNPROTECTED
    # projects and project_slugs carry no project column; every other project table is protected
    assert (with_project_column - {'combine_sources'}) <= protected | {'model_shares'}


def test_hot_row_tables_carry_their_storage_parameters(catalog) -> None:
    rows = dict(
        catalog.execute(
            "SELECT relname, reloptions FROM pg_class WHERE relname IN ('jobs', 'events')"
        ).fetchall()
    )
    assert 'fillfactor=70' in rows['jobs']
    assert 'autovacuum_vacuum_scale_factor=0.02' in rows['jobs']
    assert 'autovacuum_vacuum_insert_scale_factor=0.05' in rows['events']


def test_entity_keys_default_to_uuidv7_and_events_use_xid8(catalog) -> None:
    defaults = _set(
        catalog,
        "SELECT table_name FROM information_schema.columns WHERE table_schema = 'public' "
        "AND column_name = 'id' AND data_type = 'uuid' AND column_default = 'uuidv7()'",
    )
    assert {'projects', 'jobs', 'config_documents', 'models', 'training_runs'} <= defaults
    xid = catalog.execute(
        "SELECT udt_name FROM information_schema.columns WHERE table_name = 'events' AND column_name = 'xid'"
    ).fetchone()[0]
    assert xid == 'xid8'
