# Design documents

Index of the design docs and plans in `docs/design/`. The research notes have their own index in
[`docs/research/README.md`](../research/README.md). Status values: **implemented** (reference doc, code
matches), **open** (not started), **partially done** (evidence in the doc), **in progress**. Work is
announced and tracked in GitHub issues; the repository file is the canonical, versioned copy of each plan.

Plan lifecycle: a plan file is deleted once its work is done, in the same PR that closes its batch,
after the lasting knowledge moves to the reference docs ([`program/00_index.md`](program/00_index.md),
"Plan lifecycle"). Only `curation_api_contract.md`, `curation_design_rationale.md` and
`engineering_standards.md` stay as living reference.

## Program plan (current)

One release, v0.7.0 (B0 to B5, then R-final last), and v0.8.0 for platform breadth. The `program/` files
are removed as their batches close, and the whole folder after R-final and the v0.7.0 tag. Start at
the master plan; execute one wave at a time from the batch files it links.

| Doc | What it is |
|---|---|
| [program/00_master_plan.md](program/00_master_plan.md) | **Read first**: the execution order from now to the v0.7.0 tag: waves W2.1-W6, lanes and file ownership, folded-in decided items, sweeps and exit gates, the GPU-window budget, how a session runs one wave, the owner questions |
| [program/00b_traceability.md](program/00b_traceability.md) | Every task and ID of the original program plans and every open issue mapped to its wave and lane (or retired with a reason); the fold-in record; the proposed GitHub actions |
| [program/00c_plan_review.md](program/00c_plan_review.md) | The independent review of the plan set (3 blockers, 9 high, 15 medium, 11 low) and the planner's response to each finding |
| [program/00_index.md](program/00_index.md) | The decision register, the plain-words map, wave exit gates, the production-ready definition, branch and lane model, test and GPU budget rules, legends, superseded plans |
| [program/01_b0_close_out.md](program/01_b0_close_out.md) | B0: close out the R2 round, gate cycle, push, merge to main |
| [program/02_control_plane_database.md](program/02_control_plane_database.md) | The control plane on PostgreSQL: why, the data-ownership table, the inventory with a destination for every item, the open decisions |
| [program/02a_postgres_schema.md](program/02a_postgres_schema.md) | The PostgreSQL schema: every table, key, constraint, index and row-level-security policy |
| [program/02b_jobs_locking_events.md](program/02b_jobs_locking_events.md) | Jobs: state machine, start, claim and fencing, cancel, the reaper, events to the browser, Celery and Valkey settings, connection pools |
| [program/02c_migration_ops_tests.md](program/02c_migration_ops_tests.md) | Alembic, the 0.6.0 upgrade and rollback, compose services, the storage seam, backup and restore, tests, the trap register |
| [program/03_b2_overview.md](program/03_b2_overview.md) | B2: the control plane on PostgreSQL, Valkey and Celery, six stages, lanes, the sweep |
| [program/03a_b2_w1_queue_infra.md](program/03a_b2_w1_queue_infra.md) | B2 stage 1: foundation (Postgres, Valkey, the Celery app, the schema, the storage seam, the test database) |
| [program/03d_b2_s2_control_data.md](program/03d_b2_s2_control_data.md) | B2 stage 2: control data moves to Postgres; the 0.6.0 upgrade step |
| [program/03e_b2_s3_job_runtime.md](program/03e_b2_s3_job_runtime.md) | B2 stage 3: the job runtime, the reaper and the event feed |
| [program/03b_b2_w2_job_families.md](program/03b_b2_w2_job_families.md) | B2 stage 4: the API-hosted job families move to Celery |
| [program/03f_b2_s5_gpu_queues.md](program/03f_b2_s5_gpu_queues.md) | B2 stage 5: training, model export and the evaluator on GPU queues; promotion and the model registry |
| [program/03c_b2_w3_beat_reaper_workers.md](program/03c_b2_w3_beat_reaper_workers.md) | B2 stage 6: beat and the periodic tasks, the deletions, scale-out compose, hardening, docs |
| [program/04_b3_ultralytics_one_version.md](program/04_b3_ultralytics_one_version.md) | B3: one Ultralytics version and one detection contract |
| [program/05_b4_stateless_api.md](program/05_b4_stateless_api.md) | B4: stateless API, the worker count becomes a tuning knob; the 1-vs-N workers and replicas check |
| [program/07_b5_overview.md](program/07_b5_overview.md) | B5 speed on the final stack: overview, shared measurement protocol, #42 side track |
| [program/07a_b5_baseline.md](program/07a_b5_baseline.md) | B5a: the public baseline, the Triton sweeps and the `/detect` recall question |
| [program/07b_b5_decode_once.md](program/07b_b5_decode_once.md) | B5b: decode once (GPU-3b, the rest of #40, #407, GPU-12) |
| [program/07c_b5_crops_ocr_faces.md](program/07c_b5_crops_ocr_faces.md) | B5c: work crops at model size (#219) and OCR on the GPU (#220) |
| [program/07d_b5_processing_plans.md](program/07d_b5_processing_plans.md) | B5d: request-driven processing plans (#67) |
| [program/06_release_acceptance.md](program/06_release_acceptance.md) | Release prep, R-final acceptance run, and the v0.7.0 tag procedure |
| [program/08_environment_and_setup.md](program/08_environment_and_setup.md) | Cross-cutting: environment, setup, configuration and CI parity |
| [program/09_github_actions.md](program/09_github_actions.md) | Issues, re-scopes, milestones, closes and dependabot; section 7 is the issue map |
| [program/10_standards_and_libraries.md](program/10_standards_and_libraries.md) | Cross-cutting: code standards and the "no hand-rolling" library register |
| [program/11_risks_cleanup_host.md](program/11_risks_cleanup_host.md) | Risk register, cleanup, host, GPU and peer rules |
| [program/12_prior_plans_reconciliation.md](program/12_prior_plans_reconciliation.md) | What was unfinished in the older plans and where it went; the removal waves |
| [program/13_package_pins.md](program/13_package_pins.md) | Cross-cutting: exact pins for every Python dependency, new control-plane packages and images, bump order |

## Plans after v0.7.0 (research and the next release)

| Doc | Status | What it is |
|---|---|---|
| [research_gpu_pipeline_plan.md](research_gpu_pipeline_plan.md) | plan (research; after B5) | Where image-processing time goes per stage, a hypothesis register (batched GPU decode, per-card replicas, native resident-frame backend, faces in-graph, a last INT8 round), experiments in three GPU windows, the target architecture and stop rules |
| [research_clustering_plan.md](research_clustering_plan.md) | plan (research; after B2/B5) | Clustering for curation on real data: detector-class roots, near-duplicate thresholds, judge calibration, crop preprocessing and encoders, global graph refinement, face grouping; libraries, evaluation protocol, pass bars and stop rules |
| [research_plate_text_plan.md](research_plate_text_plan.md) | plan (research first; product in W5e, #401) | Registration-plate text as a per-box attribute: reader bake-off, per-character confidence, burst consensus, normalized exact/fuzzy/fragment search, identity groups, privacy modes and purge; the seams the v0.7.0 rebuild must include, the prototype protocol and the landing lanes |
| [v080_platform_breadth_plan.md](v080_platform_breadth_plan.md) | plan (v0.8.0, not started) | Platform breadth after v0.7.0: COCO segmentation import, search by image, model-family training, API authentication, a second segmenter, video files as a source, the wheel example and the embedding swap, in waves with a test and GPU budget |

## Per-issue plans (published as gists on the issues)

The repository copy of each plan gist linked from its GitHub issue. The plan is the reference; the
gist is the announcement. Each file is deleted with the rest of its batch's plans once the issue
closes.

| Doc | Issue | What it is |
|---|---|---|
| [issue-42-plan.md](issue_plans/issue-42-plan.md) | #42 | Smaller-workspace TensorRT builds and 12 GB GPU guidance |
| [issue-346-plan.md](issue_plans/issue-346-plan.md) | #346 | Clustering research program, umbrella plan |
| [issue-347-plan.md](issue_plans/issue-347-plan.md) | #347 | Evaluate a second public embedding model for similarity and text search (the earlier protocol is [`docs/research/embedding_model_eval_plan.md`](../research/embedding_model_eval_plan.md)) |
| [issue-348-plan.md](issue_plans/issue-348-plan.md) | #348 | Spatial scene discovery and photogrammetry: feasibility study and staged plan |
| [issue-398-plan.md](issue_plans/issue-398-plan.md) | #398 | Detector-class roots with a hold-back guard as the auto-label loop partition |
| [issue-400-plan.md](issue_plans/issue-400-plan.md) | #400 | Attribute-signal layer for clustering |
| [issue-406-plan.md](issue_plans/issue-406-plan.md) | #406 | Finish the visual-judge calibration, the blind 5 % human re-judge and the label check |
| [issue-416-plan.md](issue_plans/issue-416-plan.md) | #416 | Evaluate the near-duplicate grouping threshold (default 0.99) |

## Reference docs

| Doc | Status | Issue | What it is |
|---|---|---|---|
| [curation_api_contract.md](curation_api_contract.md) | implemented (reference) | n/a | Semantics of the `/curation` HTTP API: ordering, locking, error codes, concurrency tokens, which route for which job |
| [curation_design_rationale.md](curation_design_rationale.md) | implemented (reference) | n/a | Why the curation subsystem is built the way it is, and the known tracked gaps |
| [engineering_standards.md](engineering_standards.md) | implemented (reference) | n/a | Portable standards and lessons from the repo cleanup and the cluster research: config that worked, every guard with its baseline and proof recipe, the ordered port checklist for a sister repo, pitfalls hit, before/after numbers |

## Source plans for the program (task detail by ID)

| Doc | Status | What it is |
|---|---|---|
| [task_queue_plan.md](task_queue_plan.md) | plan (task detail by ID for `program/`; its job-state store and lease design are superseded by `program/02_control_plane_database.md`) | Background and concurrent work across processes (Q- IDs) |
| [codebase_audit_plan.md](codebase_audit_plan.md) | plan (task detail by ID for `program/`) | Codebase audit fix plan (BA-, OS-, GPU-, FE-, T-, OPS-, CC-, SEC- IDs) |
| [model_family_versions_plan.md](model_family_versions_plan.md) | plan (task detail by ID for `program/`) | Model families and Ultralytics versions: one toolchain, tested (U- IDs) |

## Implemented plans kept as reference (as-built design)

| Doc | Status | Issue | What it is |
|---|---|---|---|
| [generic_detector_and_selective_embedding_plan.md](generic_detector_and_selective_embedding_plan.md) | implemented (reference; delete guard done; arbitrary-filter clustering parked) | #52 | Default detector, store every detection, per-project embedding policy |
| [auto_label_loop_plan.md](auto_label_loop_plan.md) | implemented (reference; S0 not run, pause control, events and token budgets in N16, v0.8.0) | #286 | Cluster-sampled VLM auto-labeling loop over assist runs: pinned partition, exact sampled-agreement gate, propagate or split and recurse, budgets, resume, audit, rollback |
| [cq3_label_operations.md](cq3_label_operations.md) | implemented (reference) | #404, #419 | Server-side label operations: op record, undo by operation, per-item restore that keeps later edits and never touches the holdout |
| [selectable_assist_plan.md](selectable_assist_plan.md) | implemented (reference) | n/a | Selectable VLM and segmenter assist per run, cluster, selection, box and crop; the "Deviations" section is the as-built record |
| [sam3_full_image_detection_plan.md](sam3_full_image_detection_plan.md) | implemented (reference; waves 1 to 8, as-built deviations in section 16) | #30 | Open-vocabulary full-image detection with SAM 3 |

## Open plans that program batches read as their design input

Each is removed in the PR of the batch named (removal waves R-2 and R-3 of `program/12_prior_plans_reconciliation.md`).

| Doc | Status | Issue | What it is |
|---|---|---|---|
| [triton_pipeline_optimization_plan.md](triton_pipeline_optimization_plan.md) | partially done (Waves 0, 1, 5, 7 shipped; the rest in `program/07`-`07d`, v0.7.0 B5; Waves 8 and 10 are v0.8.0); remove after B5d | #40 | Decode once, stay on GPU, Triton pipeline optimization with before/after numbers; design detail and accuracy gates |
| [gpu_decode_plan.md](gpu_decode_plan.md) | implemented (follow-ups in `program/07b`, v0.7.0 B5b); remove after B5b | #217 | GPU decode once and a GPU-resident ingest: dataflow map, decode and resize costs, decoder coverage, parity, size gate, build record in section 13 |
| [sp_gpu_prep_concurrency.md](sp_gpu_prep_concurrency.md) | partially done (C1 shipped in #392; C2-C5 in `program/07b`, v0.7.0 B5b); remove after B5b | #392, #40, #337, #219, #42 | Why `gpu_image_prep` slows under concurrency: diagnosis, byte-budget admission, fewer syncs, concurrency shape |
| [faces_ocr_resident_crops_plan.md](faces_ocr_resident_crops_plan.md) | open (step 1 done, step 2 rejected; steps 3-4 in `program/07c` and `07d`, v0.7.0 B5c and B5d); remove after B5c | #220 | uint8 OCR input, width-bucketed recognition, face alignment on resident frames |
| [processing_plans_plan.md](processing_plans_plan.md) | open (P1 done; P0, P2-P6 in `program/07d`, v0.7.0 B5d); remove after B5d | #67 | Request-driven processing plans: closed branch vocabulary, per-route defaults, fail-closed validation, one resolved plan driving the CPU route and the `gpu_image_prep` BLS, per-branch metrics, skip-proof tests |
| [release_acceptance_plan.md](release_acceptance_plan.md) | open (method for R-final, `program/06`; the run is mostly scripted in `acceptance_run.sh`); remove at R-final RP-9 | #54 | The scripted clean-install-to-uninstall run that gates each release |
| [cluster_search_makemodel_plan.md](cluster_search_makemodel_plan.md) | open (parked research; W2, the W3 privacy part and W4 shipped; v0.8.0 and later) | #341 to #346, #347, #353, #354, #356, #372, #379 | Clustering and semantic search for faces, objects, whole images and items; fine-grained make/model workflow; attribute signals and plate identity text; measurement standard |

## Issues without a plan doc

See [`program/09_github_actions.md`](program/09_github_actions.md) section 7 (the issue map).

## Research notes

Measured snapshots and external-reference research, including the notes moved from this folder
(`ingest_write_path_investigation.md`, `storage_sizing_and_ingest_baselines.md`,
`v060_vector_db_evaluation_plan.md`, `v060_serving_backend_trade_study.md`): see
[`docs/research/README.md`](../research/README.md).
