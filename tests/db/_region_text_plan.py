"""The objects of 02d section 4 (revision 0005_region_text), written by hand from the plan.

``test_schema_matches_plan.py`` merges these sets into its exact comparisons.
"""

from __future__ import annotations


TABLES = {'region_text_runs', 'region_text_consensus_groups', 'region_text_consensus_members'}

UNPROTECTED: set[str] = set()

UNIQUE_CONSTRAINTS = {
    'uq_region_text_runs_id_project_id', 'uq_region_text_consensus_groups_id_project_id',
    'uq_region_text_consensus_members_box',
}  # fmt: skip

PARTIAL_INDEXES = {'uq_region_text_runs_open_pass'}

PLAIN_INDEXES = {'ix_region_text_runs_project', 'ix_region_text_consensus_groups_run'}

NULLS_NOT_DISTINCT = {'uq_region_text_runs_open_pass'}

NAMED_CHECKS = {
    'ck_region_text_runs_trigger', 'ck_region_text_runs_profile_pair',
    'ck_region_text_consensus_groups_basis', 'ck_region_text_consensus_groups_method',
    'ck_region_text_consensus_groups_tier', 'ck_region_text_consensus_groups_n_frames',
    'ck_region_text_consensus_groups_n_agree',
    'ck_region_text_consensus_groups_jurisdiction_conf',
}  # fmt: skip

POLICIES = {(t, f'p_{t}_project') for t in TABLES}

FOREIGN_KEYS = {
    ('region_text_runs', 'project_id', 'projects', 'c'),
    ('region_text_runs', 'job_id', 'jobs', 'n'),
    ('region_text_runs', 'profile_config_id', 'config_revisions', 'a'),
    ('region_text_consensus_groups', 'project_id', 'projects', 'c'),
    ('region_text_consensus_groups', 'run_id', 'region_text_runs', 'c'),
    ('region_text_consensus_members', 'project_id', 'projects', 'c'),
    ('region_text_consensus_members', 'group_id', 'region_text_consensus_groups', 'c'),
}  # fmt: skip
