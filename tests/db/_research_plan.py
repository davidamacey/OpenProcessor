"""The objects of docs/design/v070_rebuild/02d_research_seam_schema.md section 2, written by hand.

Written out from the plan, not read from the models, so a model and a migration that drift together
still fail. ``test_schema_matches_plan.py`` merges these sets into its exact comparisons, so the two
plan files are one contract.
"""

from __future__ import annotations


TABLES = {
    'serving_reports', 'cluster_calibrations', 'cluster_partitions', 'cluster_identities',
    'partition_clusters', 'cluster_memberships', 'cluster_activations',
    'cluster_activation_events', 'cluster_backend_validations', 'embedding_model_versions',
    'embedding_spaces', 'embedding_reindex_runs', 'embedding_evaluation_runs',
    'assist_loop_runs', 'assist_loop_iterations', 'assist_loop_nodes', 'assist_loop_selections',
    'assist_loop_proposals', 'assist_loop_reviews', 'assist_loop_budgets',
    'assist_loop_budget_ledger', 'vlm_usage_hourly',
}  # fmt: skip

# Deployment data: no project column, no row-level security.
UNPROTECTED = {'serving_reports', 'embedding_model_versions', 'cluster_backend_validations'}

UNIQUE_CONSTRAINTS = {
    'uq_cluster_partitions_id_project_id', 'uq_partition_clusters_display_no',
    'uq_cluster_activations_served_key', 'uq_embedding_model_versions_key_version',
    'uq_embedding_spaces_target_space_key', 'uq_embedding_spaces_vector_field',
    'uq_embedding_reindex_runs_job_id', 'uq_assist_loop_runs_job_id',
    'uq_assist_loop_reviews_item_queue',
}  # fmt: skip

PARTIAL_INDEXES = {'uq_embedding_spaces_one_primary', 'ix_assist_loop_nodes_open'}

PLAIN_INDEXES = {
    'ix_cluster_partitions_recent', 'ix_cluster_memberships_cluster', 'ix_assist_loop_runs_project',
    'ix_assist_loop_nodes_decision', 'ix_assist_loop_proposals_request',
    'ix_assist_loop_ledger_request',
}  # fmt: skip

NULLS_NOT_DISTINCT = {'uq_cluster_activations_served_key'}

# CHECKs a later revision alters (the closed lists of clustering_vocab, the research method values).
NAMED_CHECKS = {
    'ck_cluster_partitions_target', 'ck_cluster_partitions_level', 'ck_cluster_partitions_method',
    'ck_cluster_partitions_status', 'ck_cluster_partitions_backend',
    'ck_cluster_calibrations_kind', 'ck_cluster_identities_level',
    'ck_cluster_activations_target', 'ck_cluster_activations_level',
    'ck_cluster_identities_target', 'ck_cluster_calibrations_target',
    'ck_cluster_calibrations_level', 'ck_partition_clusters_transition',
    'ck_cluster_memberships_state',
}  # fmt: skip

FUNCTIONS = {
    'cluster_calibrations_immutable', 'cluster_partitions_ready_immutable',
    'cluster_children_ready_immutable', 'cluster_children_project_from_parent',
    'embedding_model_versions_immutable',
}  # fmt: skip

TRIGGERS = {
    ('cluster_calibrations', 'cluster_calibrations_immutable'),
    ('cluster_partitions', 'cluster_partitions_ready_immutable'),
    ('partition_clusters', 'cluster_children_ready_immutable'),
    ('cluster_memberships', 'cluster_children_ready_immutable'),
    ('partition_clusters', 'cluster_children_project_from_parent'),
    ('cluster_memberships', 'cluster_children_project_from_parent'),
    ('embedding_model_versions', 'embedding_model_versions_immutable'),
}  # fmt: skip

POLICIES = {(t, f'p_{t}_project') for t in TABLES - UNPROTECTED} | {
    ('cluster_calibrations', 'p_cluster_calibrations_default_read')
}

# (table, first column, referenced table, ON DELETE) as PostgreSQL stores it: c cascade,
# n set null, a no action, r restrict. A composite key is listed by its first column.
FOREIGN_KEYS = {
    ('cluster_calibrations', 'project_id', 'projects', 'c'),
    ('cluster_partitions', 'project_id', 'projects', 'c'),
    ('cluster_partitions', 'profile_config_id', 'config_revisions', 'a'),
    ('cluster_partitions', 'calibration_id', 'cluster_calibrations', 'a'),
    ('cluster_partitions', 'parent_partition_id', 'cluster_partitions', 'n'),
    ('cluster_partitions', 'job_id', 'jobs', 'n'),
    ('cluster_identities', 'project_id', 'projects', 'c'),
    ('cluster_identities', 'project_id', 'project_classes', 'a'),
    ('cluster_identities', 'first_partition_id', 'cluster_partitions', 'n'),
    ('cluster_identities', 'last_partition_id', 'cluster_partitions', 'n'),
    ('partition_clusters', 'partition_id', 'cluster_partitions', 'c'),
    ('partition_clusters', 'cluster_uid', 'cluster_identities', 'c'),
    ('cluster_memberships', 'partition_id', 'cluster_partitions', 'c'),
    ('cluster_activations', 'project_id', 'projects', 'c'),
    ('cluster_activations', 'serving_partition_id', 'cluster_partitions', 'a'),
    ('cluster_activations', 'pending_partition_id', 'cluster_partitions', 'a'),
    ('cluster_activation_events', 'project_id', 'projects', 'c'),
    ('embedding_spaces', 'project_id', 'projects', 'c'),
    ('embedding_spaces', 'model_version_id', 'embedding_model_versions', 'r'),
    ('embedding_spaces', 'built_by_job_id', 'jobs', 'n'),
    ('embedding_reindex_runs', 'project_id', 'projects', 'c'),
    ('embedding_reindex_runs', 'space_id', 'embedding_spaces', 'c'),
    ('embedding_reindex_runs', 'job_id', 'jobs', 'a'),
    ('embedding_evaluation_runs', 'project_id', 'projects', 'c'),
    ('embedding_evaluation_runs', 'job_id', 'jobs', 'n'),
    ('embedding_evaluation_runs', 'baseline_space_id', 'embedding_spaces', 'c'),
    ('embedding_evaluation_runs', 'candidate_space_id', 'embedding_spaces', 'c'),
    ('assist_loop_runs', 'project_id', 'projects', 'c'),
    ('assist_loop_runs', 'job_id', 'jobs', 'a'),
    ('assist_loop_runs', 'preset_config_id', 'config_revisions', 'a'),
    ('assist_loop_iterations', 'run_id', 'assist_loop_runs', 'c'),
    ('assist_loop_nodes', 'run_id', 'assist_loop_runs', 'c'),
    ('assist_loop_nodes', 'run_id', 'assist_loop_iterations', 'c'),
    ('assist_loop_nodes', 'label_op_id', 'label_operations', 'n'),
    ('assist_loop_selections', 'run_id', 'assist_loop_runs', 'c'),
    ('assist_loop_proposals', 'run_id', 'assist_loop_runs', 'c'),
    ('assist_loop_reviews', 'run_id', 'assist_loop_runs', 'c'),
    ('assist_loop_reviews', 'label_op_id', 'label_operations', 'n'),
    ('assist_loop_budgets', 'run_id', 'assist_loop_runs', 'c'),
    ('assist_loop_budget_ledger', 'run_id', 'assist_loop_runs', 'c'),
    ('assist_loop_budget_ledger', 'run_id', 'assist_loop_budgets', 'c'),
    ('vlm_usage_hourly', 'project_id', 'projects', 'c'),
    ('vlm_usage_hourly', 'endpoint_config_id', 'config_documents', 'c'),
}  # fmt: skip
