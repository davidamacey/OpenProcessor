"""The objects of 02d section 3 (revision 0003_near_duplicates), written by hand from the plan.

Not read from the models, so a model and a migration that drift together still fail.
``test_schema_matches_plan.py`` merges these sets into its exact comparisons.
"""

from __future__ import annotations


TABLES = {
    'image_contents', 'near_dup_runs', 'near_dup_groups', 'near_dup_members', 'near_dup_edges',
    'near_dup_decisions', 'near_dup_pending', 'holdout_near_dup_moves',
}  # fmt: skip

UNPROTECTED: set[str] = set()

UNIQUE_CONSTRAINTS = {
    'uq_image_contents_content', 'uq_image_contents_image', 'uq_near_dup_runs_id_project_id',
    'uq_near_dup_groups_id_project_id', 'uq_near_dup_edges_pair', 'uq_near_dup_decisions_pair',
    'uq_near_dup_pending_member', 'uq_holdout_near_dup_moves_item',
}  # fmt: skip

PARTIAL_INDEXES = {
    'uq_near_dup_runs_current',
    'uq_near_dup_runs_building',
    'uq_near_dup_members_rep',
}

PLAIN_INDEXES = {
    'ix_near_dup_runs_recent', 'ix_near_dup_groups_run', 'ix_near_dup_groups_lineage',
    'ix_near_dup_members_group', 'ix_near_dup_edges_b', 'ix_near_dup_pending_age',
}  # fmt: skip

NULLS_NOT_DISTINCT = {
    'uq_image_contents_content', 'uq_image_contents_image', 'uq_near_dup_runs_current',
    'uq_near_dup_runs_building', 'uq_near_dup_edges_pair', 'uq_near_dup_pending_member',
}  # fmt: skip

NAMED_CHECKS = {
    'ck_near_dup_runs_level', 'ck_near_dup_runs_mode', 'ck_near_dup_runs_status',
    'ck_near_dup_groups_purpose', 'ck_near_dup_members_purpose', 'ck_near_dup_edges_level',
    'ck_near_dup_edges_kind', 'ck_near_dup_decisions_level', 'ck_near_dup_decisions_decision',
    'ck_near_dup_pending_level', 'ck_near_dup_pending_reason',
}  # fmt: skip

FUNCTIONS = {'near_dup_children_project_from_run'}

TRIGGERS = {
    ('near_dup_groups', 'near_dup_children_project_from_run'),
    ('near_dup_members', 'near_dup_children_project_from_run'),
}

POLICIES = {(t, f'p_{t}_project') for t in TABLES}

FOREIGN_KEYS = {
    ('image_contents', 'project_id', 'projects', 'c'),
    ('near_dup_runs', 'project_id', 'projects', 'c'),
    ('near_dup_runs', 'job_id', 'jobs', 'n'),
    ('near_dup_runs', 'parent_run_id', 'near_dup_runs', 'n'),
    ('near_dup_runs', 'profile_config_id', 'config_revisions', 'a'),
    ('near_dup_groups', 'run_id', 'near_dup_runs', 'c'),
    ('near_dup_members', 'run_id', 'near_dup_runs', 'c'),
    ('near_dup_members', 'group_id', 'near_dup_groups', 'c'),
    ('near_dup_edges', 'project_id', 'projects', 'c'),
    ('near_dup_decisions', 'project_id', 'projects', 'c'),
    ('near_dup_pending', 'project_id', 'projects', 'c'),
    ('near_dup_pending', 'claimed_by_job_id', 'jobs', 'n'),
    ('holdout_near_dup_moves', 'project_id', 'projects', 'c'),
    ('holdout_near_dup_moves', 'freeze_id', 'holdout_freezes', 'c'),
    ('holdout_near_dup_moves', 'run_id', 'near_dup_runs', 'n'),
    ('holdout_freezes', 'near_dup_image_run_id', 'near_dup_runs', 'n'),
    ('holdout_freezes', 'near_dup_item_run_id', 'near_dup_runs', 'n'),
}  # fmt: skip
