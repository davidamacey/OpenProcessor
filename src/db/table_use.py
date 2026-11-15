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
    'analysers': (
        'the analyser repository: ingest, region stage, VLM, auto-label, probe, scores (steps 2 to 4)',
        'every result reader (joined for kind, name, version); the analysis ledger work list',
        'storage steps 2-4',
    ),
    'analysis_ledger': (
        'ingest and every analyser writer, one row per photo and kind (steps 2 and 4)',
        're-run work lists (equality with the current analyser), the status routes',
        'storage steps 2-4',
    ),
    'assist_answers': (
        'the assist-loop answer writer (step 4)',
        'the item wire (assist answer attributes)',
        'storage steps 2-4',
    ),
    'box_state': (
        'the fold: apply() under the item lock (steps 3 and 4)',
        'box rows of the item wire, the region editor, text search (step 5)',
        'storage steps 2-4',
    ),
    'box_texts': (
        'the region stage and the text readers (step 4)',
        'the fold (text consensus per box); the region editor',
        'storage steps 2-4',
    ),
    'claim_kinds': (
        'revision 0010 (seeded from src/domain/claim_kinds.py)',
        'the claims foreign key; a test asserts it equals the domain catalogue',
        'storage steps 2-4',
    ),
    'claims': (
        'every human write path through the claim repository (step 3); imports and combine',
        'the fold (apply), the undo walk, the history routes (steps 3 and 5)',
        'storage steps 2-4',
    ),
    'class_opinions': (
        'the VLM, auto-label, classifier, propagation, adopt-detector and import writers (steps 3 and 4)',
        'the fold (machine class precedence), raw-label aggregations',
        'storage steps 2-4',
    ),
    'detections': (
        'ingest and re-detect (steps 2 and 4)',
        'the fold (machine geometry and detector class)',
        'storage steps 2-4',
    ),
    'entities': (
        'none before phase 5 (gallery claims and grouping jobs)',
        'schema-test assertion only: tests/db/test_storage_schema_matches_plan.py',
        'phase 5',
    ),
    'entity_edges': (
        'none before phase 5 (grouping and event jobs)',
        'schema-test assertion only: tests/db/test_storage_schema_matches_plan.py',
        'phase 5',
    ),
    'entity_state': (
        'none before phase 5 (the entity fold)',
        'schema-test assertion only: tests/db/test_storage_schema_matches_plan.py',
        'phase 5',
    ),
    'erasure_jobs': (
        'the erasure job (storage step 3)',
        'the append-only triggers (op_erasure_allows); the erasure routes',
        'storage steps 2-4',
    ),
    'face_state': (
        'none before phase 5 (the face fold)',
        'schema-test assertion only: tests/db/test_storage_schema_matches_plan.py',
        'phase 5',
    ),
    'faces': (
        'the face detector writer (step 4)',
        'the face routes and the people grouping',
        'storage steps 2-4',
    ),
    'item_notes': (
        're-detect and import reconcile (steps 3 and 4)',
        'the item wire (proposal_chain)',
        'storage steps 2-4',
    ),
    'item_projections': (
        'the projection job (step 4)',
        'the visualization routes',
        'storage steps 2-4',
    ),
    'item_region_results': (
        'the region stage (step 4)',
        'the fold (region status), the region worker poll',
        'storage steps 2-4',
    ),
    'item_scores': (
        'the uniqueness and mistakenness jobs (step 4)',
        'review sorts and the item wire',
        'storage steps 2-4',
    ),
    'item_state': (
        'the fold: apply() under the item lock (steps 3 and 4)',
        'every list, count and facet read (step 5)',
        'storage steps 2-4',
    ),
    'items': (
        'ingest and attach (steps 2 and 4)',
        'every item join; the item lock',
        'storage steps 2-4',
    ),
    'membership_state': (
        'none before phase 5 (the membership fold)',
        'schema-test assertion only: tests/db/test_storage_schema_matches_plan.py',
        'phase 5',
    ),
    'photo_capture': (
        'the capture parser at ingest (step 2)',
        'the timeline and map reads (phase 5)',
        'storage steps 2-4',
    ),
    'photo_derivatives': (
        'none before phase 5 (the derivative branch)',
        'schema-test assertion only: tests/db/test_storage_schema_matches_plan.py',
        'phase 5',
    ),
    'photo_files': (
        'none before phase 5 (the photo-library scanner)',
        'schema-test assertion only: tests/db/test_storage_schema_matches_plan.py',
        'phase 5',
    ),
    'photo_places': (
        'none before phase 5 (the reverse geocoder)',
        'schema-test assertion only: tests/db/test_storage_schema_matches_plan.py',
        'phase 5',
    ),
    'photo_provenance': (
        'dataset import and combine (step 3)',
        'the photo wire (import and origin fields)',
        'storage steps 2-4',
    ),
    'photo_signals': (
        'none before phase 5 (the photo-quality analysers)',
        'schema-test assertion only: tests/db/test_storage_schema_matches_plan.py',
        'phase 5',
    ),
    'photo_state': (
        'none before phase 5 (the photo fold)',
        'schema-test assertion only: tests/db/test_storage_schema_matches_plan.py',
        'phase 5',
    ),
    'photos': (
        'ingest, one transaction per commit group (step 2)',
        'every photo read; attach and erasure lock it',
        'storage steps 2-4',
    ),
    'probe_predictions': (
        'the probe scoring job (step 4)',
        'review sorts and the wire probe_actionable',
        'storage steps 2-4',
    ),
    'raw_label_clusters': (
        'the raw-label clustering job (step 4)',
        'raw-label aggregations',
        'storage steps 2-4',
    ),
    'region_boxes': (
        'the region stage and imports (step 4)',
        'the fold (machine boxes)',
        'storage steps 2-4',
    ),
    'text_lines': (
        'the OCR writers (step 4)',
        'text search and the photo wire',
        'storage steps 2-4',
    ),
    'vector_gaps': (
        'the embedding reindex job and the vector writers (step 6)',
        'the embedding-state reader',
        'storage steps 2-4',
    ),
}
