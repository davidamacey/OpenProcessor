# Design documents

Index of the design docs and plans in `docs/design/`. The research notes have their own index in
[`docs/research/README.md`](../research/README.md). Status values: **implemented** (reference doc, code
matches), **open** (not started), **partially done** (evidence in the doc), **in progress**. Work is
announced and tracked in GitHub issues; the repository file is the canonical, versioned copy of each plan.

Plan lifecycle: a plan file is deleted once its work is done, in the same PR that closes its batch,
after the lasting knowledge moves to the reference docs ([`v070_rebuild/reference/decision_register.md`](v070_rebuild/reference/decision_register.md),
"Plan lifecycle"). Only `curation_api_contract.md`, `curation_design_rationale.md` and
`engineering_standards.md` stay as living reference.

## v0.7.0 rebuild plans (current): `v070_rebuild/`

Every plan for the v0.7.0 backend rebuild is in one folder, [`v070_rebuild/`](v070_rebuild/00_master_plan.md).
Start at [`v070_rebuild/00_master_plan.md`](v070_rebuild/00_master_plan.md): its "Start here" table lists
every file in the order it is used (01 pins, 02 database, 03 job-queue stages, 04 Ultralytics, 05
stateless API, 06 speed and plate text, 07 release, 08 prototype track, 09a-09h research designs, 10
v0.8.0, 11 the #42 side track) and `reference/` (decisions, owner questions, traceability, reviews,
standards, environment, risks). Plans are deleted only after their wave merges and their lasting
knowledge has moved to the reference docs.



## Reference docs

| Doc | Status | Issue | What it is |
|---|---|---|---|
| [curation_api_contract.md](curation_api_contract.md) | implemented (reference) | n/a | Semantics of the `/curation` HTTP API: ordering, locking, error codes, concurrency tokens, which route for which job |
| [curation_design_rationale.md](curation_design_rationale.md) | implemented (reference) | n/a | Why the curation subsystem is built the way it is, and the known tracked gaps |
| [engineering_standards.md](engineering_standards.md) | implemented (reference) | n/a | Portable standards and lessons from the repo cleanup and the cluster research: config that worked, every guard with its baseline and proof recipe, the ordered port checklist for a sister repo, pitfalls hit, before/after numbers |

## Source plans for the program (task detail by ID)

| Doc | Status | What it is |
|---|---|---|
| [task_queue_plan.md](task_queue_plan.md) | plan (task detail by ID for `v070_rebuild/`; its job-state store and lease design are superseded by `v070_rebuild/02_control_plane_database.md`) | Background and concurrent work across processes (Q- IDs) |
| [codebase_audit_plan.md](codebase_audit_plan.md) | plan (task detail by ID for `v070_rebuild/`) | Codebase audit fix plan (BA-, OS-, GPU-, FE-, T-, OPS-, CC-, SEC- IDs) |
| [model_family_versions_plan.md](model_family_versions_plan.md) | plan (task detail by ID for `v070_rebuild/`) | Model families and Ultralytics versions: one toolchain, tested (U- IDs) |

## Implemented plans kept as reference (as-built design)

| Doc | Status | Issue | What it is |
|---|---|---|---|
| [generic_detector_and_selective_embedding_plan.md](generic_detector_and_selective_embedding_plan.md) | implemented (reference; delete guard done; arbitrary-filter clustering parked) | #52 | Default detector, store every detection, per-project embedding policy |
| [auto_label_loop_plan.md](auto_label_loop_plan.md) | implemented (reference; S0 not run, pause control, events and token budgets in N16, v0.8.0) | #286 | Cluster-sampled VLM auto-labeling loop over assist runs: pinned partition, exact sampled-agreement gate, propagate or split and recurse, budgets, resume, audit, rollback |
| [cq3_label_operations.md](cq3_label_operations.md) | implemented (reference) | #404, #419 | Server-side label operations: op record, undo by operation, per-item restore that keeps later edits and never touches the holdout |
| [selectable_assist_plan.md](selectable_assist_plan.md) | implemented (reference) | n/a | Selectable VLM and segmenter assist per run, cluster, selection, box and crop; the "Deviations" section is the as-built record |
| [sam3_full_image_detection_plan.md](sam3_full_image_detection_plan.md) | implemented (reference; waves 1 to 8, as-built deviations in section 16) | #30 | Open-vocabulary full-image detection with SAM 3 |

## Open plans that program batches read as their design input

Each is removed in the PR of the batch named (removal waves R-2 and R-3 of `v070_rebuild/reference/prior_plans_reconciliation.md`).

| Doc | Status | Issue | What it is |
|---|---|---|---|
| [triton_pipeline_optimization_plan.md](triton_pipeline_optimization_plan.md) | partially done (Waves 0, 1, 5, 7 shipped; the rest in `v070_rebuild/06_b5_overview.md`-`07d`, v0.7.0 B5; Waves 8 and 10 are v0.8.0); remove after B5d | #40 | Decode once, stay on GPU, Triton pipeline optimization with before/after numbers; design detail and accuracy gates |
| [gpu_decode_plan.md](gpu_decode_plan.md) | implemented (follow-ups in `v070_rebuild/06b_b5_decode_once.md`, v0.7.0 B5b); remove after B5b | #217 | GPU decode once and a GPU-resident ingest: dataflow map, decode and resize costs, decoder coverage, parity, size gate, build record in section 13 |
| [sp_gpu_prep_concurrency.md](sp_gpu_prep_concurrency.md) | partially done (C1 shipped in #392; C2-C5 in `v070_rebuild/06b_b5_decode_once.md`, v0.7.0 B5b); remove after B5b | #392, #40, #337, #219, #42 | Why `gpu_image_prep` slows under concurrency: diagnosis, byte-budget admission, fewer syncs, concurrency shape |
| [faces_ocr_resident_crops_plan.md](faces_ocr_resident_crops_plan.md) | open (step 1 done, step 2 rejected; steps 3-4 in `v070_rebuild/06c_b5_crops_ocr_faces.md` and `07d`, v0.7.0 B5c and B5d); remove after B5c | #220 | uint8 OCR input, width-bucketed recognition, face alignment on resident frames |
| [processing_plans_plan.md](processing_plans_plan.md) | open (P1 done; P0, P2-P6 in `v070_rebuild/06d_b5_processing_plans.md`, v0.7.0 B5d); remove after B5d | #67 | Request-driven processing plans: closed branch vocabulary, per-route defaults, fail-closed validation, one resolved plan driving the CPU route and the `gpu_image_prep` BLS, per-branch metrics, skip-proof tests |
| [release_acceptance_plan.md](release_acceptance_plan.md) | open (method for R-final, `v070_rebuild/07_release_acceptance.md`; the run is mostly scripted in `acceptance_run.sh`); remove at R-final RP-9 | #54 | The scripted clean-install-to-uninstall run that gates each release |
| [cluster_search_makemodel_plan.md](cluster_search_makemodel_plan.md) | open (parked research; W2, the W3 privacy part and W4 shipped; v0.8.0 and later) | #341 to #346, #347, #353, #354, #356, #372, #379 | Clustering and semantic search for faces, objects, whole images and items; fine-grained make/model workflow; attribute signals and plate identity text; measurement standard |

## Issues without a plan doc

See [`v070_rebuild/reference/github_actions.md`](v070_rebuild/reference/github_actions.md) section 7 (the issue map).

## Research notes

Measured snapshots and external-reference research, including the notes moved from this folder
(`ingest_write_path_investigation.md`, `storage_sizing_and_ingest_baselines.md`,
`v060_vector_db_evaluation_plan.md`, `v060_serving_backend_trade_study.md`): see
[`docs/research/README.md`](../research/README.md).
