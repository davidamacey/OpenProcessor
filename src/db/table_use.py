"""Who writes and who reads each table that a research revision creates, and in which wave.

Pure data (docs/design/v070_rebuild/02d_research_seam_schema.md section 0.1). A table without a
0.7.0 writer or reader says so, and a schema test stands in for the reader.
"""

TABLE_USE: dict[str, tuple[str, str, str]] = {
    'assist_loop_budget_ledger': (
        'the assist_loop job (lane 4D)',
        'the loop routes (status, metrics), the budget gate',
        'W2.3a',
    ),
    'assist_loop_budgets': (
        'the assist_loop job (lane 4D)',
        'the loop routes (status, metrics), the budget gate',
        'W2.3a',
    ),
    'assist_loop_iterations': (
        'the assist_loop job (lane 4D)',
        'the loop routes (status, metrics), the budget gate',
        'W2.3a',
    ),
    'assist_loop_nodes': (
        'the assist_loop job (lane 4D)',
        'the loop routes (status, metrics), the budget gate',
        'W2.3a',
    ),
    'assist_loop_proposals': (
        'the assist_loop job (lane 4D)',
        'the loop routes (status, metrics), the budget gate',
        'W2.3a',
    ),
    'assist_loop_reviews': (
        'none in 0.7.0 (v0.8.0 routes)',
        'schema-test assertion only: the blind-audit CHECK test and the metrics function of lane 4D',
        'W2.1',
    ),
    'assist_loop_runs': (
        'the assist_loop job (lane 4D)',
        'the loop routes (status, metrics), the budget gate',
        'W2.3a',
    ),
    'assist_loop_selections': (
        'the assist_loop job (lane 4D)',
        'the loop routes (status, metrics), the budget gate',
        'W2.3a',
    ),
    'cluster_activation_events': (
        '4F activation transaction',
        'the cluster serving routes, cluster_projection_sync, the assist loop',
        'W2.3a',
    ),
    'cluster_activations': (
        '4F activation transaction',
        'the cluster serving routes, cluster_projection_sync, the assist loop',
        'W2.3a',
    ),
    'cluster_backend_validations': (
        'the GPU agreement validation tool (a row is the owner gate)',
        'the cluster_run start refuses backend gpu without a passing row',
        'W2.3a',
    ),
    'cluster_calibrations': (
        'cluster_run and persons_cluster through 4F (kind assign_bands)',
        'assign.place_new of 4F-5 (incremental bands)',
        'W2.3a',
    ),
    'cluster_identities': (
        '4F (a new identity per new cluster, a name on human or import anchoring)',
        'the cluster routes and promotion to a class',
        'W2.3a',
    ),
    'cluster_memberships': (
        'cluster_run and persons_cluster (record_partition, 4F)',
        'the cluster routes, the activation transaction, cluster_projection_sync, the assist loop (pinned.partition)',
        'W2.3a',
    ),
    'cluster_partitions': (
        'cluster_run and persons_cluster (record_partition, 4F)',
        'the cluster routes, the activation transaction, cluster_projection_sync, the assist loop (pinned.partition)',
        'W2.3a',
    ),
    'embedding_evaluation_runs': (
        'none in 0.7.0 (the evaluation job is a later lane)',
        'schema-test assertion only: tests/db/test_embedding_registry_schema.py',
        'W2.1',
    ),
    'embedding_model_versions': (
        'the catalog sync in db-migrate; the W3 gate G7 writes engine_parity_min_cos once',
        'embedding_space() and the search builder (ES-A, ES-C)',
        'W2.2',
    ),
    'embedding_reindex_runs': (
        'the embedding_reindex job (ES-B)',
        'the job status route',
        'W2.3a',
    ),
    'embedding_spaces': (
        'project create (three default spaces) and the 0.6.0 upgrade',
        'the search builder (ES-C), embedding_reindex',
        'W2.2',
    ),
    'partition_clusters': (
        'cluster_run and persons_cluster (record_partition, 4F)',
        'the cluster routes, the activation transaction, cluster_projection_sync, the assist loop (pinned.partition)',
        'W2.3a',
    ),
    'serving_reports': (
        'op.periodic.serving_inventory',
        'the Triton router of B4 4B (a server whose report lacks the model gets no call)',
        'W2.1',
    ),
    'vlm_usage_hourly': (
        'every VLM caller',
        "the assist loop's budget gate (4D) and the VLM usage figures",
        'W2.3a',
    ),
}
