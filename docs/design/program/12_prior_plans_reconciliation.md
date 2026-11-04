# 12: reconciliation of the older plan docs with the program

Status: plan.
Depends on: `00_index.md` (the program), and the batch files it lists.
Verified against `fix/411-r2-round` on 2026-11-03 (head `d1c6bab0`), with read-only `rg`, `git log`,
`gh issue view`, `gh pr view` and `gh issue list` (37 open issues).

Consolidated 2026-11-03.

**Status after consolidation (read this before the tables).**
- **Milestone words (owner decision D-37 = yes).** This file was written when B5 was a second release.
  There is now one release, v0.7.0 = B0, B1, B2, B3, B4, B5 and R-final last; `v0.8.0` is platform
  breadth. Wherever a table below says `v0.8.0` for B5 work read **v0.7.0 (B5)**, and wherever it says
  `v0.9.0` read **v0.8.0 (platform breadth)**. `00_index.md` and `09_github_actions.md` hold the
  authoritative milestone map.
- **Removal wave R-1: done in this consolidation pass** (the docs are moved or deleted, the content
  moves and the reference fixes are made; section 6 marks each row). **R-2** (`release_acceptance_plan.md`)
  is "remove at R-final RP-9". **R-3** docs are "remove after B5b", "after B5c" or "after B5d". R-2 and R-3 follow
  the plan lifecycle of [00_index.md](00_index.md) section 8.2: the closing PR of the batch or
  sub-batch moves the lasting knowledge and deletes the plan files in the same PR.
- **ADD items (section 3) are folded** into the target files; each ADD heading below carries a
  "Folded into" line. The counts in section 3.7 are final.
- **Master plan.** `docs/design/master_plan.md` is deleted; the C14 note in section 3.0 is resolved
  (the program cites it only as "the former master plan (rev 4, in git history)").
- **README.** Section 5 is applied: `docs/design/README.md` has the "Program plan (current)" section and
  the updated statuses.

This file reviews every older doc in `docs/design/` (all except the retired master plan, the
three source plans `task_queue_plan.md`, `codebase_audit_plan.md`, `model_family_versions_plan.md`,
and `README.md`), plus the `docs/research/` notes that still carry action items. For each one it
says what is unfinished, where that work now lives, what must be added to the program, and whether
the doc can be removed. Nothing here edits another file; the coordinator applies sections 3, 5
and 6.

## 0. How to read this file

**Old B2 and B4 task IDs.** This file was written before B2 was re-planned on PostgreSQL. Its
references to `03b` 2A-2C, `03c` 2D-2F and 2E-n, the `2G` lane and B4 `4C` are the former IDs; their
current lanes are in the table of [03_b2_overview.md](03_b2_overview.md) section 14 and, task by
task, in [00b_traceability.md](00b_traceability.md) section 1.

**Dispositions** (one per unfinished item):

| Word | Meaning |
|---|---|
| covered | the program already does it; the cell names the file and the task ID or issue |
| ADD | the program must gain a task; section 3 has the paste-ready text |
| defer | out of this program; the cell names the milestone (v0.7.0 B5 for speed, v0.8.0 for platform breadth) or "parked research" |
| obsolete | already done, or no longer needed; the cell gives the evidence |
| conflict | the old text breaks an owner rule or the new design; section 4 says how the program resolves it |

**Verdicts** (one per doc, section 6):

| Verdict | Meaning |
|---|---|
| DELETE | finished or fully superseded; nothing in it is still needed |
| DELETE-AFTER-MOVE | finished, but some content is still needed; the target and the sections are named |
| KEEP-REFERENCE | a living reference that describes what the code does (as-built design the code and docs cite) |
| KEEP-OPEN | holds unfinished work that a program batch or parked research still reads as its design source |

**The rule used to pick a verdict.** Plans whose content is only sequencing, status or a finished
work breakdown are deleted. As-built designs of a shipped subsystem that code comments, user docs
or the white paper cite as the design source stay as reference. Finished docs that are mostly
measurements move to `docs/research/` (same file name, `git mv`), because the research index is
where measured snapshots live. Docs that a B5 sub-batch names as its design input stay open until
that sub-batch merges. When unsure, the doc is kept.

## 1. Summary table

"Open" counts the unfinished items found (section 2), not the conflicts.

| # | Doc | README status today | Real status found | Open | Recommended status | Verdict |
|---|---|---|---|---|---|---|
| 1 | `auto_label_loop_plan.md` | in progress (S0, S11 open) | implemented; S11 live run done 2026-11-02 (#286 closed); S0 never run | 4 | implemented (reference) | KEEP-REFERENCE |
| 2 | `cluster_search_makemodel_plan.md` | open (nothing implemented) | partly done: research waves W0, W0b, W1, WB, W8 run; W2 (#353, #354) and the privacy part of W3 (#379) shipped; W4 shipped similar, image and `target` search (#402) | 10 | open (parked research; v0.9.0 and later) | KEEP-OPEN |
| 3 | `combine_preview_warnings_plan.md` | implemented | implemented (#55 closed) | 0 | removed | DELETE |
| 4 | `cq3_label_operations.md` | open (design review; nothing implemented) | implemented (label-ops routes, store, undo, domain rules; UI #404 and paging #419 closed) | 0 | implemented (reference) | KEEP-REFERENCE |
| 5 | `curation_api_contract.md` | implemented | reference, kept current by each batch | 0 | implemented (reference) | KEEP-REFERENCE |
| 6 | `curation_design_rationale.md` | implemented | reference; three sections go stale in B1 and B2 | 0 | implemented (reference) | KEEP-REFERENCE |
| 7 | `dataset_import_more_formats.md` | not listed | implemented (`195a7697`) | 1 | removed | DELETE |
| 8 | `dataset_import_voc_openimages.md` | not listed | implemented (PR #426) | 0 | removed | DELETE |
| 9 | `docs_overhaul_plan.md` | in progress | implemented (PR #294; follow-ups F1-F12 filed as #278-#289, all closed) | 0 | removed | DELETE |
| 10 | `engineering_standards.md` | implemented (reference) | reference | 0 | implemented (reference) | KEEP-REFERENCE |
| 11 | `faces_ocr_resident_crops_plan.md` | not listed | step 1 done (`27ebeb45`); step 2 measured and rejected (#220 comment 2026-11-02); steps 3, 4 open | 2 | open (B5c/B5d input, v0.8.0) | KEEP-OPEN |
| 12 | `generic_detector_and_selective_embedding_plan.md` | implemented (deferred: 3 items) | implemented; of the 3 deferred items one is done, one obsolete, one deferred | 1 | implemented (reference) | KEEP-REFERENCE |
| 13 | `gpu_decode_plan.md` | implemented | implemented; follow-ups planned in B5 | 3 | implemented (B5b input) | KEEP-OPEN |
| 14 | `improvement_backlog.md` | not listed | snapshot of 2026-10-30; most rows closed | 6 | removed | DELETE-AFTER-MOVE |
| 15 | `ingest_surfaces_plan.md` | implemented in 0.6.0 | implemented (#264 closed, #263 decided "skip"); own header still says "plan" | 0 | removed | DELETE-AFTER-MOVE |
| 16 | `ingest_write_path_investigation.md` | not listed | fixes merged into 0.6.0; follow-ups mostly done | 2 | moved to research | DELETE-AFTER-MOVE (to `docs/research/`) |
| 17 | `label_confirmation_plan.md` | partially done | implemented (PR 1-5 of its section 6; #119 closed) | 0 | removed | DELETE-AFTER-MOVE |
| 18 | `processing_plans_plan.md` | open (design only for v0.9.0) | P1 done; P0, P2-P6 are B5d (v0.8.0) | 6 | open (B5d input, v0.8.0) | KEEP-OPEN |
| 19 | `program_plan.md` | open (plan) | superseded (`00_index.md` section 13) | 0 in program | removed | DELETE |
| 20 | `release_acceptance_plan.md` | open (full run not scripted) | mostly scripted (`acceptance_run.sh`, 26 phases; #54 closed; R1 and R2 ran); phases D, E, H, parts of I and K not scripted | 7 | open until R-final, then removed | DELETE-AFTER-MOVE (at RP-9) |
| 21 | `repo_cleanup_plan.md` | in progress (PR 1 of 9) | implemented: all nine PRs merged (#300, #301, #302, #309, #310, #303/#313/#317, #323, #324) | 0 | removed | DELETE |
| 22 | `sam3_full_image_detection_plan.md` | implemented | implemented (v0.4.0) | 2 | implemented (reference) | KEEP-REFERENCE |
| 23 | `selectable_assist_plan.md` | not listed | implemented (waves A, B, C); own header still says "plan, not implemented" | 0 | implemented (reference) | KEEP-REFERENCE |
| 24 | `ship_track_plan.md` | open (nothing implemented) | superseded; most waves shipped | 0 in program | removed | DELETE-AFTER-MOVE |
| 25 | `sp_gpu_prep_concurrency.md` | open (nothing implemented) | SP0 counters and C1 (byte budget) shipped (#390, #392 closed); C2-C5 open | 4 | open (B5b input, v0.8.0) | KEEP-OPEN |
| 26 | `storage_sizing_and_ingest_baselines.md` | partially done (quantization open) | done; the quantization benchmark ran in the vector-store evaluation (#56 closed) | 0 | moved to research | DELETE-AFTER-MOVE (to `docs/research/`) |
| 27 | `triton_pipeline_optimization_plan.md` | partially done (v0.7.0, #40) | Waves 0, 1 (FP16 PE), 5, 7 done; 2, 3, 4, 6, 8, 9, 10 open | 7 | open (B5 input, v0.8.0) | KEEP-OPEN |
| 28 | `v060_execution_path.md` | in progress | historical (0.6.0 shipped) | 0 | removed | DELETE |
| 29 | `v060_serving_backend_trade_study.md` | open (nothing implemented) | executed: X-1 to X-5 and X-7 (#226-#230, #232 closed); X-6 (#231) open | 1 | moved to research | DELETE-AFTER-MOVE (to `docs/research/`) |
| 30 | `v060_vector_db_evaluation_plan.md` | implemented | implemented | 0 | moved to research | DELETE-AFTER-MOVE (to `docs/research/`) |
| 31 | `v060_wave_plan.md` | implemented in 0.6.0 | implemented; its gate table is still cited | 0 | removed | DELETE-AFTER-MOVE |
| 32 | `v070_wave_plan.md` | open (waves not started) | superseded; W0, W1, W3, W5 shipped | 0 in program | removed | DELETE |

Totals: 32 design docs. Verdicts: 8 DELETE, 10 DELETE-AFTER-MOVE (4 of them are moves to
`docs/research/`), 8 KEEP-REFERENCE, 6 KEEP-OPEN. Section 2.33 covers the 22 research notes that
carry action items (in 12 rows).

## 2. Per-doc dispositions

### 2.1 `auto_label_loop_plan.md` (#286)

Evidence: `src/services/curation/assist_loop/` (driver, budget, audit, rollback, propagate), the
router `src/routers/curation/assist_loop.py`, the worker `scripts/curation/assist_loop_worker.py`;
#286 closed 2026-11-02 with a live run on public COCO (7 real runs, budget never exceeded,
rollback 138/138, lock rule held). `scripts/bench/vlm_rate.py` does not exist.

| Item (section) | Disposition |
|---|---|
| S0 VLM label-rate measurement (13, S0 row) | defer v0.9.0; folded into new issue N16 (section 3.6). The loop works; the rate only tunes budget defaults, and the VLM is not in B5's Triton scope |
| Pause control and event log (header "Known S9 gaps"; 11) | defer v0.9.0, N16. Build on the B2 consumer (`03c` 2E-2) and the B4 event stream (`05` 4C), not on the trigger file and JSONL events |
| Q9: capture `usage` tokens for remote-endpoint budgets (15) | defer v0.9.0, N16 (needed before any remote primary-VLM loop run; `rg usage src/services/labeling` finds only the endpoint probe) |
| S11: acceptance numbers into `docs/PERFORMANCE.md` (13, S11 row) | ADD `06` RP-8 (copy the #286 closing evidence; docs only) |
| Cost ratio 1 crop per 20 items unproven (#286: about 1 per item at 2,500 images) | parked research with Z5 (#398, v0.9.0) |
| D4 trigger-file worker, `start.lock`, claim by `unlink` (3.1, 8) | conflict C1, resolved by B1 1A 1A-6 and `03c` 2E-2; ADD a doc sync in 2E-2 (section 3.3) |
| `threading.Lock` in `assist_loop/budget.py` and `store.py` | no conflict: it guards in-memory counters of one loop inside its single owner; the lease (1A) makes the owner unique |

### 2.2 `cluster_search_makemodel_plan.md` (#341-#346, #353, #354, #356, #372, #379, #402)

| Item (section) | Disposition |
|---|---|
| W4 bug "object search runs Triton on the event loop and queries crops against the whole-image index" (10, never filed) | ADD B1 1C 1C-15 (section 3.1). Verified: `SearchMixin.search_by_object` in `src/services/visual_search/search.py` calls the sync `infer_yolo_clip_cpu` inside `async def`, and sends a box vector to `visual_search_global` for every class that is not a vehicle or a person |
| W3 rest: incremental face assignment with abstention, merge and split suggestions (3.2, W3) | defer v0.9.0 (faces feature); tracked under #346 until its wave starts |
| W5, W7, W9, W10 research (fine-grained measurement, fine groups, adapter, set P pilot) | parked research (#346) |
| W6 taxonomy as data and a classification-crop export | defer v0.9.0; must use the B1 registry document by name (B1 1B) |
| W8 visual judge | #406: `clu:CL-1` in research wave RW-3 ([research_clustering_plan_b_protocol.md](../research_clustering_plan_b_protocol.md) section 5) |
| W-UI review screens | defer v0.9.0, with each feature |
| W12 identity text (plates) | #401: superseded by [research_plate_text_plan.md](../research_plate_text_plan.md); lands as wave W5e ([03g](03g_w5e_plate_text.md), pending RD-3) |
| 6.7 attribute signals | #400: superseded by [research_spatial_attributes_plan.md](../research_spatial_attributes_plan.md); research wave RW-5 |
| `target=frame`, `group_by=image` | #435, #436 (v0.9.0) |
| 10: eleven issues "to create" | do not file now (owner rule: no growing issue list); file each when its wave starts |
| `GPU2='flock ... gpu2.lock'` shorthand (9) | conflict C9 (host procedure, not app code); use `11_risks_cleanup_host.md` section 1 when a wave resumes |

### 2.3 `combine_preview_warnings_plan.md` (#55)

Implemented in `src/services/projects/combine/warnings.py`; #55 closed. No open item.

### 2.4 `cq3_label_operations.md`

Evidence: `src/routers/curation/label_ops.py`, `src/services/curation/label_ops/{record,store,undo,views}.py`,
`src/domain/label_ops.py`, `src/services/curation/class_undo.py`; frontend route
`p/[project]/label-ops`; #404 and #419 closed. Gate G7 was answered by building it.

| Item (section) | Disposition |
|---|---|
| `write.lock` and `undo.lock` through `exclusive_start_lock` (3, 6.4, 9) | conflict C2, resolved by B1 1A 1A-8 (leases `label_op`, `label_undo`); ADD a doc sync in 1A-8 (section 3.1) |
| Region undo bounded by `Semaphore(32)` around a `gather` (6.4; `src/routers/curation/edit_undo.py`) | covered by `05` 4A-2 (bulk merger, one bulk) |
| Header "design, not implemented" | stale; fix with the README update (section 5) |

### 2.5 `curation_api_contract.md`

Reference. Every batch that changes a route or wire model updates it (the docs lists of `02`, `03`,
`04`, `05`). No open item.

### 2.6 `curation_design_rationale.md`

Reference. It will be wrong after B1 and B2 unless updated:

| Item (section) | Disposition |
|---|---|
| 2.1 says the class registry lives on disk; 12 describes class identity over the file | ADD doc sync to B1 1B 1B-14 (section 3.1) |
| 6 "Known gaps" lists the workers and says "experimental for v0.4.0" | ADD doc sync to `03_b2_overview.md` section 17 (section 3.3) |

### 2.7 `dataset_import_more_formats.md` (#370 part 2)

Implemented in `195a7697` (`cvat.py`, `labelstudio.py`, `kitti.py`, `createml.py`, `labelme.py`,
YOLO names files) and YOLO-seg/OBB rows as boxes in `yolo.py` (`yolo_polygon_to_box`). User docs:
`docs-site/docs/guides/dataset-import.mdx`.

| Item | Disposition |
|---|---|
| COCO segmentation-only annotations as boxes (out of scope there; the last unchecked box of #370) | defer v0.9.0; ADD re-scope comment C-370 in `09` (section 3.6): #370 is open with the program calling it "importers not built", but only this format remains |

### 2.8 `dataset_import_voc_openimages.md` (#370 part 1)

Implemented (`voc.py`, `openimages.py`; PR #426). No open item.

### 2.9 `docs_overhaul_plan.md`

Implemented: PR #294 merged T1-T8; the new pages exist (`choose-your-path.mdx`,
`ingest-large-folder.mdx`, `pipeline-modes.mdx`, `upgrade-and-backup.mdx`, Cropwright `audit.md`),
`make docs-screenshots` exists, `docs/CURATION.md` is cut to a map. Section 8 F1-F12 became
#278-#289, all closed. No open item.

### 2.10 `engineering_standards.md`

Reference. `05` 4A adds the request-scoped `TaskGroup` exemption (Q-J7). It receives the lessons of
`improvement_backlog.md` section 6 (section 6 of this file). No open item.

### 2.11 `faces_ocr_resident_crops_plan.md` (#220)

| Item (section) | Disposition |
|---|---|
| Step 2 width-bucketed recognition | obsolete: measured and rejected (#220 comment 2026-11-02; P-ocr 86 % with padding, no gain without) |
| Step 3 argmax in the recognition graph | covered by `07c` track `ocr` |
| Step 4 face alignment on resident frames | covered by `07d` (#67 P5) |

### 2.12 `generic_detector_and_selective_embedding_plan.md` (#52)

| Item (status line, 5) | Disposition |
|---|---|
| Model-delete guard for a project's own detector | obsolete (done): `src/routers/curation/models.py` answers 409 `detector_in_use`; `src/routers/models_lifecycle.py` refuses a project-promoted model with 409 `project_owned_model` |
| Optional `backfill_embedding_state.py` | obsolete: optional by design; vectorless items are covered by the embed-missing reprocess; no report of a project needing it |
| Arbitrary-filter clustering | defer: parked research (#346) |
| Case table 1 and 2: re-embed a human-drawn or moved box | obsolete (done): `refresh_box_embeddings` in `src/routers/curation/regions_boxes_edit.py` |

### 2.13 `gpu_decode_plan.md` (#217)

| Item (13 "Not reached") | Disposition |
|---|---|
| A second card | covered by `07a` 5a-S1 (two-card set B rows) |
| GPU decode of progressive JPEGs (11 % of set B) | covered by `07b` 5b-C1 (b) |
| Crop JPEG encode on the GPU | covered by `07b` 5b-C1 (a) |

### 2.14 `improvement_backlog.md`

| Row | Disposition |
|---|---|
| Z5 detector-class roots | #398 (v0.9.0, research) |
| Fine-grained make/model | #385, #399 closed; further work parked (#346) |
| Faces plan (#353, #354, #379) | obsolete (all closed); the rest is `cluster_search_makemodel_plan.md` W3 (2.2) |
| Attribute signals, plate identity, Leiden/GPU clustering | #400 (RW-5), #401 (W5e), #346 (seams in W2.3b 4F, research in RW-3), per the research plans of 2026-11-04 |
| PE-Core-L stays; embedding swap | #347, #365 (v0.9.0) |
| Semantic search over crops; filter cost | obsolete for the built parts (#402 phases A and B, #384, #356 closed); #435, #436 v0.9.0 |
| Ingest-speed work | obsolete (#390, #392, #393 closed) |
| C2 batched kernels; staged image-prep ideas | covered by `07b` 5b-P1, 5b-G2-G4 (#407) |
| CUDA MPS opt-in | #403, parked inside B5 (`07` section 10) |
| Triton memory at start: "the installer could offer the profile by card size" | defer v0.9.0 (installer UX); #373's fix already sizes the default profile by card |
| Geometry stage of the cluster pass (#337 leftover) | ADD a row to `07_b5_overview.md` section 10 (section 3.5): act only if the 2.5 ms/crop bar tightens |
| Remaining stage 2 items (#268) | covered by `07a` track `sweeps` |
| Loop pause and event log | N16 (2.1) |
| Label operations UI, thumbnails 404, live write-path failures | obsolete (#404, #405, #397 closed) |
| 5 judge protocol | #406 (v0.9.0) |
| 6 lessons learned | still needed: move to `engineering_standards.md` section 5 (section 6) |

### 2.15 `ingest_surfaces_plan.md` (#264, #263)

| Item | Disposition |
|---|---|
| Per-project trigger loop "with a lock, like the open-vocabulary sweeper", `dispatch.lock`, bulk-ingest start lock (overview, 9) | conflict C5, resolved by B1 1A 1A-6 (bulk ingest leases) and `03c` 2D-1 (beat tick plus a per-project lease) |
| 8 gRPC decision rule | still needed (cited by `docs/PERFORMANCE.md` "Ingest transports"): move (section 6) |
| `error_kind` vocabulary (cited by `src/services/curation/bulk_ingest/object_fetch.py`) | still needed: move to `docs-site/docs/guides/ingest-sources.mdx` (section 6) |

### 2.16 `ingest_write_path_investigation.md`

| Item (7) | Disposition |
|---|---|
| (1) source images and the OpenSearch volume on different devices | obsolete (documented in `docs-site/docs/operations/storage.mdx`) |
| (2) sample IO pressure and merge/flush counters in the scale driver | IO pressure done (`scripts/bench/host_pressure.py`, #337); merge/flush counters: ADD `07a` 5a-A5 |
| (3) super-linear cluster pass | obsolete (#337 closed: fit 388 s to 50 s on 91k crops) |
| (4) force merges between legs, not during ingest | ADD `07a` 5a-A5 (one arm, only if a leg shows a merge-driven dip) |
| (5) admission charge when a GPU call falls back to the CPU | obsolete: #392 replaced the memory-pressure fallback with a byte budget; remaining fallbacks are counted (`route="cpu_call_failed"`) |
| A 100k repeat on private 20 MP photos | conflict C10 (public-only), resolved by `07` OQ-B5-1 (public set B) |

### 2.17 `label_confirmation_plan.md` (#119, #192)

All of section 6 shipped: `vlm_scope` (PR 1), detector class fields (PR 2,
`src/services/curation/ingest_detect.py`, `_item_models.py`), the `detector_disagreements` tab (PR 3),
the audit routes with `src/domain/audit_math.py` (PR 4), the auto-promote audit gate (PR 5,
`AuditGateError`). #119 and #192 closed. No open item. The scope semantics the frontend cites are in
`docs-site/docs/guides/vlm-selection.mdx` (scope table).

### 2.18 `processing_plans_plan.md` (#67)

| Item (10) | Disposition |
|---|---|
| P0, P2, P3, P4, P5, P6 | covered by `07d` |
| "v0.9.0 delivers this" (header) | conflict C8 (release mapping), resolved by `00_index.md` (B5 is v0.8.0) |
| Q2 "parity on set A and the private set" | conflict C10, resolved by `07d` (public only) |

### 2.19 `program_plan.md`

Superseded by `00_index.md` (section 13 there).

| Item (section) | Disposition |
|---|---|
| Phase A winter run on set P (A6, R8) | out of the program: an owner operation on private data, not a release gate |
| Phase B Triton and serving | covered by B5 (`07`-`07d`) |
| Phase C clustering | parked (#346) |
| Phase D optional embedding model | #347 (v0.9.0) |
| Phase E spatial memory | #348 (parked) |
| Phase F auth, video | #369 (blocked), #8 (parked) |
| Phase G white paper | obsolete (done: #412 closed, `docs/WHITEPAPER.md`) |
| 6 NVMe budget, NAS scan | host rules in `11` section 1; NAS scan with #348 |
| 7 public multi-view datasets | obsolete (already in `docs/research/spatial_datasets.md`) |
| 0.1 DoD, 2 design reviews and audit gates | covered by `00_index.md` sections 4, 7, 8 and `10` |

### 2.20 `release_acceptance_plan.md` (#54)

`scripts/release/acceptance_run.sh` drives 26 phases (health through teardown); `06` RF-01 to
RF-63 add the procedure and the batch checks. Not covered by either:

| Phase of the old plan | Disposition |
|---|---|
| D: two projects, nothing of one visible in the API, event stream, logs and metrics of the other | ADD `06` RF-72 |
| E: YOLO and COCO dataset imports, re-import is a no-op | ADD B1 1B 1B-12b (script) and `06` RF-70 (run) |
| H: combine preview and combine; sources unchanged | ADD B1 1B 1B-12b and `06` RF-71 |
| I: reprocess skips locked items; import undo keeps human edits, a second undo dry run is zero; archive and delete guards; dashboards, log search and experiment tracking show live data | ADD `06` RF-73 (import undo is in 1B-12b) |
| Notes: train mode frees the GPU before training and restarts the tiers | ADD `06` RF-74 |
| Notes: the shipped default monitoring credential | ADD `06` RF-75 (record as a neutral security finding unless the installer prompts) |
| K: every docs page captured and viewed | ADD `06` RF-76 (docs-site build plus page checks) |
| 4 findings document, 5 go/no-go | covered by `06` sections 5 and 7 |
| "one run per release" (implied R3, R-B1, R-0.9) | conflict C7, resolved by `06` (one R-final) |

### 2.21 `repo_cleanup_plan.md`

All nine PRs merged: #300 (guards), #301 (dead code), #302 (fail-open), #309 (layering), #310
(duplicates), #317, #303, #313 (splits), #323 (strict mypy), #324 (`engineering_standards.md`),
plus #314 (env and infra). `scripts/codegen/baselines/file_size.txt` holds one `cohesive:` entry;
`process_ids.txt` and `vulture.txt` are empty. The remaining baselines (`broad_except.txt` 226
lines, `complexity.txt` 26, `cited_paths.txt` 188) are shrink-only ratchets owned by
`10_standards_and_libraries.md` section 2. No open item.

### 2.22 `sam3_full_image_detection_plan.md` (#30)

| Item (16) | Disposition |
|---|---|
| Recall against COCO ground truth and the batch-endpoint gain "still need a live run" | defer v0.9.0 (open-vocabulary quality; not a correctness bug and not on the Triton path) |
| "agree" note on the surviving detector item not built | obsolete: agreement is counted as `dropped_agree_existing`; nothing reads a note |
| Ingest-time pass as an API background task, serialized per process; sweeper per worker | conflict C4, resolved by `03b` 2C (`open_vocab_drain`, `-c 1`, `_PASS_LOCK` deleted) and `03c` 2D-1 (sweeper on beat); ADD a doc sync in 2C (section 3.3) |

### 2.23 `selectable_assist_plan.md`

Implemented: the routes in `src/routers/curation/assist.py`, the service in
`src/services/curation/assist/`, the frontend wave (dialog screenshot
`docs-site/static/img/screenshots/cropwright/assist-dialog-1600.png` exists). Its "Deviations"
section is the as-built record.

| Item | Disposition |
|---|---|
| Assist runs as in-API asyncio tasks with a start lock and `reconcile_orphaned_jobs` | conflict C3, resolved by B1 1A 1A-6 and `03b` 2A (`assist_run` family); ADD a doc sync in 2A (section 3.3) |
| Header "plan, not implemented" | stale; fix with the README update |

### 2.24 `ship_track_plan.md`

| Item (section) | Disposition |
|---|---|
| SP0, SP1, SP2, SP6, CQ1-CQ3, UI0-UI5, L1, L2, D1 | obsolete (shipped; their issues are closed) |
| SP3 #219 | covered by `07c` track `crops` |
| SP4 #40 high-resolution row | covered by `07b` |
| SP5 #45 ladders | covered by `07a` track `ladders` |
| 7.3 Z5, 7.4 Z5-I | #398 (v0.9.0) |
| 3 measurement protocol | still needed: `07_b5_overview.md` line 78 "builds on" it; move (section 6) |
| 10.3 GPU locks with `flock` on a host lock file | conflict C9, resolved by `11` section 1 |

### 2.25 `sp_gpu_prep_concurrency.md` (#392, #407)

| Item (5) | Disposition |
|---|---|
| C1 memory pressure waits and is counted | obsolete (done: #392, PR #408) |
| C2 fewer syncs and allocations | covered by `07b` 5b-G2 (CV-CUDA first) |
| C3 concurrency shape | covered by `07b` 5b-P1 (the profile decides) and `07a` 5a-S4 (instance count) |
| C4 JPEG encode off the critical path | covered by `07b` 5b-C1 (a) |
| C5 Triton flags | covered by `07a` 5a-S3 and ADD 5a-S8 |
| E8 MPS check | #403 (parked inside B5, `07` section 10) |

### 2.26 `storage_sizing_and_ingest_baselines.md` (#45, #53, #56)

Done: the measurements, the public re-measure (PR #206) and the quantization benchmark (vector-store
evaluation, #56 closed). The baseline it says to repeat is `07a` 5a-R0 and 5a-A3, which cite it.
No open item.

### 2.27 `triton_pipeline_optimization_plan.md` (#40)

| Item (6) | Disposition |
|---|---|
| Wave 2 CPU decode and letterbox on OpenCV | ADD `07b` 5b-D2 (section 3.5); `rg ingest_decode src` finds nothing |
| Wave 3 crops at model size | covered by `07c` 5c-K1-K3 (#219) |
| Wave 4 `max_queue_delay_microseconds` | covered by `07a` 5a-S3 |
| Wave 4 `model_warmup`, CUDA and pinned memory pool sizes | ADD `07a` 5a-S8 |
| Wave 6 GPU decode for the stateless routes | covered by `07b` 5b-W1 |
| Wave 8 region stage on resident crops, segmenter on TensorRT | defer v0.9.0; needs its own plan file first (owner answer to its Q8); ADD a row to `07` section 10 |
| Wave 9 tuning pass and GPU layout | covered by `07a` track `sweeps` |
| Wave 10 embedding-space migration | defer v0.9.0 with #365 (one re-embed with a version field) |
| Q1, Q7 "gate on the private set" | conflict C10, resolved by `07` OQ-B5-1 |
| Header "baseline numbers not yet recorded" | stale (Wave 0 is in its section 11 and `docs/PERFORMANCE.md`) |

### 2.28 `v060_execution_path.md`

Historical status table of 0.6.0. No open item.

### 2.29 `v060_serving_backend_trade_study.md`

| Item (11) | Disposition |
|---|---|
| X-6 head-to-head with the alternative serving runtime (#231) | covered by `07` section 10 (after 5b, with its own decision rule) |
| MPS entries | #403 (parked) |

### 2.30 `v060_vector_db_evaluation_plan.md` (#221, #56)

Implemented; results in `docs/PERFORMANCE.md`; the Lucene engine is the default for new projects.
No open item.

### 2.31 `v060_wave_plan.md`

Implemented in 0.6.0. Its section 0 rule 3 (benchmark protocol) and sections 2 and 2.1 (parity chain
and gate table) are still cited by `docs/PERFORMANCE.md` line 1008 and
`scripts/bench/parity_gates.json`. No open item; content moves (section 6).

### 2.32 `v070_wave_plan.md`

| Item | Disposition |
|---|---|
| W0 #341, W1 #337/#259, W3 #286, W5 #41, #335 | obsolete (closed) |
| W2 #219 | covered by `07c` |
| W4 #268, #45 | covered by `07a`; #84 is v0.9.0 |
| 3.1 rehearsal budgets for the winter run | out of the program (owner operation, 2.19) |

### 2.33 Research notes with action items (`docs/research/`)

These stay as research notes (no verdict change). Only their action items are reconciled.

| Note | Action items | Disposition |
|---|---|---|
| `triton_deep_research_2026-09.md` "Short Action List" 1-14 | 1-2 baseline and perf tools; 3 bounded async inference in ingest; 4-5 concurrency and pipelining; 6-7 DALI; 8-9 ArcFace batching and GPU alignment; 10 decouple OCR and OpenSearch; 11 Model Analyzer; 12 shared memory; 13 Triton 26.08; 14 micro-optimization last | 1, 2, 4, 5, 12 obsolete (0.6.0: Wave 0, `gpu_image_prep`, bulk runner #229, shared memory measured and not adopted #215); 3 covered by `05` 4A (TaskGroup exemption, C11); 6-7 `07b` 5b-G3, 5b-W1; 8-9 `07d` P5; 10 `07c`, `07d`; 11 `07` section 7; 13 D-38 (26.09, B3 optional task); 14 the `07` rule |
| `v070_measurement_log.md` "what to measure next" | 2 instances of `gpu_image_prep`; profile Python/GIL time; a reduced (DCT-scaled) decode; search tail under ingest plus parallel search clients; higher crops per image and a cold set | instances obsolete (#392, #393); profile `07b` 5b-P1; reduced decode ADD `07c` 5c-K4; search tail covered by `07a` 5a-S6 (interactive p99 under bulk load); crops and cold set covered by `07a` 5a-A3 (record crops per image) |
| `cluster_config_decision.md`, `cluster_config_z4.md`, `cluster_config_loop_cost.md`, `cluster_program_outcome.md` | Z5 repeat on 1,200+ open-set crops, then Z5-I | #398 (v0.9.0) |
| `clustering_research_consolidated.md`, `cross_domain_clustering_lessons.md`, `cluster_research_9oct.md`, `leiden_refinement_prototype.md`, `gpu_clustering.md` | experiments E0-E10, graph refinement phases | #342-#345 closed; the rest parked under #346 |
| `embedding_model_eval_plan.md` | evaluate a newer text-and-image embedding model | #347 (v0.9.0) |
| `fine_grained_benchmark_results.md`, `wb2_standard_benchmarks.md`, `labelled_benchmarks.md` | person-checked labels (FG-0); swap only with a text tower | #399 closed; swap is #365 (v0.9.0) |
| `visual_judge_e0.md` | 5 % human re-judge, layout calibration | #406 (v0.9.0) |
| `OpenProcessor_Spatial_Memory_Photogrammetry_Research_README.md`, `spatial_datasets.md` | E0-E6 | #348 (parked) |
| `vlm_serving_sizing.md` | loop label rate (auto-label S0) | N16 (v0.9.0) |
| `cropwright_health_2026-10-29.md` | #380, #381, #382, #405 | obsolete (closed) |
| `installer_drive_log.md` | "SIGINT mid-run: no message on interrupt (open)" | obsolete (fixed: `_op_on_signal` in `setup-openprocessor.sh` prints the resume hint); the row is stale, ADD to `06` RP-7 |
| `audits/phase_a1_audit.md` | findings for #357, #362, #373-#376 | obsolete (closed) |

## 3. The ADD list (paste-ready, by target file and wave)

Every task follows the program's lane rules (`00_index.md` section 7): own worktree and branch, a
red test seen failing on the base plus a paired control, pre-commit before each commit, `make
contracts` with any wire change.

### 3.0 `01_b0_close_out.md` (B0)

**B0-1b Retire the finished plans (docs only, one commit).** *Folded into
[01_b0_close_out.md](01_b0_close_out.md) B0-1b; wave R-1 itself was executed in the consolidation pass.* After B0-1, in the same worktree:
apply the removal wave R-1 of section 6 (deletes, moves to `docs/research/`, the content moves and
every reference fix listed there), then the README rows of section 5.
- Commit: `docs(design): retire finished plans and move measured notes to research`.
- Checks: `.venv/bin/python scripts/docs/check_docs_vs_code.py`,
  `.venv/bin/python scripts/codegen/check_naming_leaks.py`, `.venv/bin/pre-commit run --all-files`
  (the cited-paths guard checks comments in `.py`, `.ts`, `.svelte`, `.js`, `.mjs` and `.sh` files),
  `.venv/bin/python -m pytest tests/test_doc_links.py tests/codegen/test_check_cited_paths.py -q --no-cov`.
- Done when: for every removed file, `rg -l '<file name>' --glob '!CHANGELOG.md'` lists only
  historical review files that cite it in plain text (section 6 says which may stay), and both checks
  pass.
- If B0 is already past its gate, this commit is the second commit of B1 instead (after the plan
  set is committed).

**Note for the coordinator (plan-set inconsistency; RESOLVED: the master plan is deleted and the
program cites it only as "the former master plan (rev 4, in git history)").** B0-1's done-when says the master-plan file
name must have no hit under `docs` after B0-1. Today nine program files cite that file name
(`00_index.md`, `01`, `02`, `03`, `06`, `08`, `09`, `10`, `11`). Either reword the done-when to
"no hit outside `docs/design/program/`" or reword those citations to "the retired master plan
(rev 4)".

### 3.1 B1 (B1 (merged; see CHANGELOG.md) wave)

**B1 1C (merged; see CHANGELOG.md): 1C-15 Legacy visual search: Triton off the event loop, no box
vector against the whole-image index (S).** *Folded into B1 1C (merged; see [CHANGELOG](../../../CHANGELOG.md)) 1C-15.* Run after 1C-9 in the same lane (both touch legacy
visual search).
- Evidence: `src/services/visual_search/search.py` (`SearchMixin`): `search_by_object` builds a
  sync client (`get_sync_triton_client`) and calls `infer_yolo_clip_cpu(image_bytes)` inside
  `async def`; `search_by_image` and `search_by_text` call `self.inference.encode_image` and
  `encode_text`, which are plain `def` in `src/services/inference.py`. The routers in
  `src/routers/search/` await them, so each call blocks the worker's event loop for a full
  detect-and-embed. `src/routers/search/objects.py` also reads the upload with the sync
  `image.file.read()`. For a detected class that is neither a vehicle nor a person,
  `search_by_object` queries `visual_search_global` (whole-image vectors) with a box vector.
  `SearchMixin.search_vehicles` and `search_people` have no router caller (check with `rg`; delete
  them if the unused-code ratchet agrees).
- Fix: one offload per request for the blocking inference, with the same helper 1D-7 (CC-5) uses
  (`asyncio.to_thread`, Python docs "Running in Threads"; or the repo's CPU executor helper);
  `await image.read()` (Starlette `UploadFile` is async). For an unindexed category, fail closed:
  answer 422 with a new code (recommendation: `object_class_not_indexed`) instead of a cross-space
  query. OC at kick-off: fail closed (recommended) or document it as "find images like this crop".
  Contracts and the README route table in the same commit.
- Red: (a) a stub sync Triton client that sleeps 0.3 s; two concurrent `/search/object` requests
  through `httpx.AsyncClient` with `ASGITransport` on one loop take at least 0.6 s today and under
  0.45 s after; (b) a detection of a class outside both categories never calls `search_global`
  (fails today). Control: a vehicle query returns the same hits before and after.
- Origin: `cluster_search_makemodel_plan.md` section 10 (planned, never filed).

**B1 1B (merged; see ARCHITECTURE.md): 1B-12b Acceptance phases `dataset_import` and `combine` (S).** The
registry migration touches import and combine (`dataset_import/prepare.py`, `combine/execute.py`),
and no scripted phase covers either today. *Folded into B1 1B (merged; see [ARCHITECTURE](../../ARCHITECTURE.md)) 1B-12b.*
- File: `scripts/release/acceptance_phases.sh` (two new `ph_` functions, registered for `--list`).
- `dataset_import`: build a YOLO dataset from the run's own `export` phase output and a COCO
  dataset from the license-filtered COCO subset (`scripts/datasets/fetch_coco_subset.py` already
  parses `instances_val2017.json`); for each, call `/curation/projects/{project}/datasets/preview`
  then `/curation/projects/{project}/datasets/imports`; assert classes map by name (compare the
  project's class names and ids before and after); a second import of the same files reports
  no new items; `/curation/projects/{project}/datasets/imports/{import_id}/undo` keeps a label a
  person set after the import; a second undo dry run changes nothing.
- `combine`: `/curation/projects/combine/preview` then `/curation/projects/combine` into a new
  throwaway project from two sources; assert counts, mapping by name, the test split preserved, and
  both sources unchanged (index document counts and a checksum of each source's registry before
  and after).
- Red: `acceptance_run.sh --list` names both phases (fails today); `bash -n` and shellcheck pass;
  the live run is R-final RF-70 and RF-71.
- Origin: `release_acceptance_plan.md` phases E, H and the import-undo part of I.

**B1 1B (merged; see ARCHITECTURE.md) 1B-14 Docs: add** `docs/design/curation_design_rationale.md` sections
2.1 and 12 (the registry is an OpenSearch document; the file is an export snapshot). *Folded into
B1 1B (merged; see [ARCHITECTURE](../../ARCHITECTURE.md)) 1B-14.*

**B1 1A (merged; see ARCHITECTURE.md) 1A-8 Label operations: add** *(folded into B1 1A (merged; see [ARCHITECTURE](../../ARCHITECTURE.md)) 1A-8)* "update
`docs/design/cq3_label_operations.md` sections 3 (the locks row), 6.4 and 9 (crash during undo) to
name the leases `label_op` and `label_undo` instead of `write.lock`, `undo.lock` and `flock`".

### 3.2 R-final (`06_release_acceptance.md`)

*Folded into [06_release_acceptance.md](06_release_acceptance.md) (RF-70 to RF-76, RP-7 to RP-9).*
Add a section "4.5 Checks carried from `release_acceptance_plan.md`":

| ID | Check | How | Pass |
|---|---|---|---|
| RF-70 | Dataset import | `acceptance_run.sh --only dataset_import` (1B-12b) on install A | PASS |
| RF-71 | Combine | `acceptance_run.sh --only combine` (1B-12b) | PASS; sources unchanged |
| RF-72 | Project isolation, live | two projects with data on install A; follow the event stream of one while ingesting into the other; `docker compose -p op-rfinal logs api` and the metrics endpoint filtered by the first project's slug | nothing of the second project appears in the first's stream, its log lines or its labelled metrics |
| RF-73 | Day-2 operations | a reprocess over items with a human label; archive the protected default project; delete a project while a job runs; with the monitoring tier on, open a dashboard, a log search and the experiment-tracking run of the `train` phase | reprocess reports the skipped locked items and changes none; the guards refuse; the three UIs show this run's data |
| RF-74 | Train mode frees the GPU | `./openprocessor train-mode on` before the `train` phase, `nvidia-smi` before and after; `train-mode off` after | the segmenter and VLM stop and at least 16 GB is free on the training GPU; both come back healthy |
| RF-75 | Monitoring credential | install with the monitoring tier | if the installer does not ask for a new admin password, record a neutral security finding (details in private notes) |
| RF-76 | Docs site | `cd docs-site && npm ci && npm run build`; serve it; open getting-started, quick-start, ingest-large-folder, pipeline-modes, release-acceptance and one page per guide group at 1440 and 430 px | build passes; pages render; screenshots viewed |

Add to section 3 (release prep):

| ID | Task | Files | Done when |
|---|---|---|---|
| RP-7 | Stale "known issue" notes | `docs-site/docs/operations/curation-workflow.mdx` (caution "a gated promote stops the retrain", #280, fixed by #302), `docs-site/docs/guides/pipeline-modes.mdx` section 4 (same), `docs/research/installer_drive_log.md` (the SIGINT row) | every "Known issue" note and every "(issue #N)" in `docs-site/docs`, `docs/` and `README.md` names an open issue (`gh issue list --state open`) or is removed |
| RP-8 | Assist-loop live numbers | `docs/PERFORMANCE.md` (loop section) | the #286 closing evidence (runs, budget, rollback, reproducibility, lock rule, yield) is recorded with its stack facts |
| RP-9 | Retire `release_acceptance_plan.md` | removal wave R-2 (section 6) | the doc is gone, its rules and go/no-go live in `docs-site/docs/operations/release-acceptance.mdx`, and the references are fixed |

### 3.3 B2 (`03_b2_overview.md` stages)

*Folded into [03b](03b_b2_w2_job_families.md) (2A and 2C doc syncs), [03c](03c_b2_w3_beat_reaper_workers.md)
2E-2 and [03_b2_overview.md](03_b2_overview.md) section 17.*

- **`03b_b2_w2_job_families.md` track 2A, the `assist_run` family commit: add** "update the
  `selectable_assist_plan.md` Deviations line that says a restart ends a run `interrupted` through
  `reconcile_orphaned_jobs` to the reaper rule".
- **`03b` track 2C, `open_vocab_drain`: add** "update `sam3_full_image_detection_plan.md` section 16
  Wave 5 (API background task, serialized per process, sweeper) to the `open_vocab` queue and the
  beat task".
- **`03c_b2_w3_beat_reaper_workers.md` 2E-2: add** "update `auto_label_loop_plan.md` sections 0
  (D4), 1 (worker pattern), 3.1 (`start.lock`, `trigger.json`) and 8 (one loop per project) to the
  lease and the `-Q assist_loop -c 1` consumer".
- **`03_b2_overview.md` section 17: add** `docs/design/curation_design_rationale.md` section 6 (the
  worker list and the "experimental for v0.4.0" line).

### 3.4 B4 (`05_b4_stateless_api.md`)

No ADD. The old plans' fan-out and event-log items are already in 4A-2 (region undo) and 4C.

### 3.5 B5 (v0.7.0, D-37)

*Folded into [07a](07a_b5_baseline.md) (5a-S8, 5a-A5), [07b](07b_b5_decode_once.md) (5b-D2),
[07c](07c_b5_crops_ocr_faces.md) (5c-K4) and [07_b5_overview.md](07_b5_overview.md) section 10 (the three
deferred rows; Waves 8 and 10 are v0.8.0).*

**`07a_b5_baseline.md` track `sweeps`: 5a-S8 Triton warmup and memory pools.**
- From `triton_pipeline_optimization_plan.md` Wave 4 (not covered by 5a-S1-S7): `model_warmup` at
  batch 1 and the profile's largest batch for the detector, PE and OCR models; explicit
  `--cuda-memory-pool-byte-size` and `--pinned-memory-pool-byte-size` sized from the measured use.
- Docs: Triton model configuration "Model Warmup" (warmup blocks READY until done) and the
  `tritonserver --help` pool flags.
- Measure: time from start to READY, first-request latency after a model load, p99 at 32 clients,
  GPU memory per card. Pass: the 5a rule (gain beyond 2x the spread, outputs byte-identical).

**`07a` track `ladders`: 5a-A5 OpenSearch merge counters per leg (XS).**
- Record `_nodes/stats/indices` merge, refresh and flush totals before and after each ladder leg,
  beside the `host_pressure.py` IO samples; never force-merge during a leg. If a leg shows a
  merge-driven dip, run one arm with a `_forcemerge` between legs.
- Docs: OpenSearch nodes stats API and force merge API.
- Origin: `ingest_write_path_investigation.md` section 7 items 2 and 4.

**`07b_b5_decode_once.md` track `decode`: 5b-D2 CPU letterbox on OpenCV (M).** After 5b-D1.
- From `triton_pipeline_optimization_plan.md` Wave 2: an ndarray path in `letterbox_to_square`
  (`src/services/detection/geometry.py`) with `cv2.resize`, `INTER_AREA` for downscales (OpenCV docs
  recommend it for shrinking; PIL bilinear antialiases, `INTER_LINEAR` does not), and the 640
  letterbox from the 1280 intermediate. Keep the documented PIL-versus-cv2 split between ingest and
  `/detect` unless the gate says otherwise. A setting, default off, until the gate passes
  (`env.template` and `tests/test_env_surface.py` in the same commit).
- Measure: API CPU-seconds per image on the CPU route (set A, and set B below the 3 MP GPU gate).
- Gate: P-box against `ref-b5a`; adopt only with a gain beyond 2x the spread.

**`07c_b5_crops_ocr_faces.md` track `crops`: 5c-K4 Reduced decode once K3 sets the cap
(measurement first, M).**
- From `v070_measurement_log.md` ("measure a second decode route"): with work crops capped at
  `crop_max_side`, a DCT-scaled decode (scale chosen so the largest work crop still has at least the
  cap in pixels) can feed both the detector input and the crops of a large photo, so one smaller
  decode replaces the full-frame decode.
- Libraries: OpenCV `IMREAD_REDUCED_COLOR_2/4/8`, Pillow `Image.draft`; on the GPU route check
  whether nvImageCodec decode parameters allow a scaled decode before writing anything.
- Measure: decode ms per image on set B; crop pixels per work crop. Gate: P-box (boxes within 1 px
  at the original resolution), P-vlm, P-sam, P-region, P-pe. Adopt only with a gain beyond 2x the
  spread; otherwise record and close.

**`07_b5_overview.md` section 10 "Deferred inside the milestone": add three rows.**

| Item | Rule |
|---|---|
| Cluster-pass geometry stage (#337 leftover: 0.8 ms per crop, the largest stage since the fit change) | act only if the 2.5 ms/crop bar tightens; open an issue then |
| `triton_pipeline_optimization_plan.md` Wave 8 (region stage on resident crops; segmenter on TensorRT) | v0.9.0; needs its own plan file before work (owner answer to its Q8) |
| Wave 10 embedding-space migration | v0.9.0 with #365 (one re-embed with a version field) |

### 3.6 GitHub (`09_github_actions.md`)

*Folded into [09_github_actions.md](09_github_actions.md) (N16, C-402, C-370 and the moved-doc
comments; the milestone of N16 is v0.8.0 platform breadth).*

**N16 body (new issue, milestone v0.9.0).** Title: `feat(loop): pause control, event stream and
token budgets for the assist loop`.
- Problem: the loop can start, watch, cancel, resume and roll back, but has no pause control and no
  event log, because the API serves neither (`LoopStore` already has a `pause.flag` and the state
  has `pause_reason`). Remote VLM endpoints have no token budget because the client does not read
  the `usage` object. The label rate that sizes the budgets was never measured (S0).
- Scope: measure the label rate first (S0 design in `auto_label_loop_plan.md` section 13);
  a pause route on the B2 consumer; events from the Postgres event feed (B2 stage 3); `usage` tokens read from the
  OpenAI-compatible response; Cropwright pause button and event log.
- Depends on: B2 2E-2, B4 4C.

**C-402 (comment, then close #402).** Phases A and B shipped: `/curation/projects/{project}/search/similar`,
`/curation/projects/{project}/search/image` and the `target` option (`src/routers/curation/search.py`,
`search_image.py`). The rest is #435 and #436. Add the close to section 6.

**C-370 (comment and re-scope #370).** Every format in the owner's 2026-10-31 list shipped (#426,
`195a7697`) except COCO segmentation-only annotations as boxes. Re-scope #370 to that item
(v0.9.0), or close it and file one small issue. Fix the `00_index.md` section 10 and `09` section 7
rows that say "importers ... defer".

**Moved-doc comments.** When removal wave R-1 runs, post one comment on each open issue whose body
cites a moved or deleted doc, giving the new path (from the scan of open-issue bodies and comments):
`v060_wave_plan.md` #40, #67, #219, #231, #268; `improvement_backlog.md` #398, #400, #401, #402,
#403, #406; `program_plan.md` #365-#370; `ingest_write_path_investigation.md` #40;
`storage_sizing_and_ingest_baselines.md` #45; `v060_vector_db_evaluation_plan.md` #40, #67;
`v060_serving_backend_trade_study.md` #40, #67, #231. Docs that stay (`triton_pipeline_optimization_plan.md`,
`sp_gpu_prep_concurrency.md`, `faces_ocr_resident_crops_plan.md`, `gpu_decode_plan.md`,
`cluster_search_makemodel_plan.md`, `selectable_assist_plan.md`,
`generic_detector_and_selective_embedding_plan.md`) need no comment now.

### 3.7 Count of ADD items

| Target | Items |
|---|---|
| B0 (`01`) | 1 (B0-1b) |
| B1 (B1 1A, B1 1B, B1 1C) | 4 (1C-15, 1B-12b, two doc syncs in 1B-14 and 1A-8) |
| B2 (`03`, `03b`, `03c`) | 4 doc syncs |
| B4 (`05`) | 0 |
| R-final (`06`) | 10 (RF-70 to RF-76, RP-7 to RP-9) |
| B5 (`07`, `07a`, `07b`, `07c`) | 5 (5a-S8, 5a-A5, 5b-D2, 5c-K4, one block of three deferred rows) |
| GitHub (`09`) | 4 (N16, C-402, C-370, moved-doc comments) |

## 4. Conflicts found and how the program resolves them

| ID | Old doc (section) | Conflict | Resolution in the program |
|---|---|---|---|
| C1 | `auto_label_loop_plan.md` (0 D4, 3.1, 8) | start by `exclusive_start_lock` (`start.lock`), worker claims a `trigger.json` by `unlink` (file locks; at-most-once claim) | B1 1A 1A-6 lease `assist_loop`; `03c` 2E-2 makes the worker a `-Q assist_loop -c 1` Celery consumer and deletes the trigger files; doc sync ADD in 2E-2 |
| C2 | `cq3_label_operations.md` (3, 6.4) | `write.lock`, `undo.lock` through `fcntl.flock` | B1 1A 1A-8 leases `label_op`, `label_undo`; doc sync ADD in 1A-8 |
| C3 | `selectable_assist_plan.md` (Deviations) | assist runs as in-API asyncio tasks with a start lock and startup reconcile | B1 1A 1A-6 lease; `03b` 2A moves the family to Celery; the reaper replaces the reconcile |
| C4 | `sam3_full_image_detection_plan.md` (16, Wave 5) | in-API background pass serialized by a per-process lock; sweeper in every worker | `03b` 2C `open_vocab_drain` on `-c 1` (deletes `_PASS_LOCK`); `03c` 2D-1 sweeper on beat |
| C5 | `ingest_surfaces_plan.md` (overview, 9) | trigger loop in every API worker, `dispatch.lock`, bulk-ingest start lock | B1 1A 1A-6; `03c` 2D-1 (beat tick plus a per-project lease) |
| C6 | `cq3_label_operations.md` (6.4), `selectable_assist_plan.md` | per-item `gather` in request handlers (bounded by a semaphore) | `05` 4A-2 (one `mget` plus one bulk with per-item OCC) |
| C7 | `release_acceptance_plan.md` (5, implied cadence) | a full acceptance run per release (R3, R-B1, R-0.9) | owner decisions 2-3 of 2026-11-03: one R-final before the single v0.7.0 tag (`06`) |
| C8 | `program_plan.md`, `ship_track_plan.md`, `v070_wave_plan.md`, `processing_plans_plan.md` | release mapping: v0.7.0 winter pipeline, v0.8.0 Triton and clustering, v0.9.0 spatial; processing plans in v0.9.0 | `00_index.md` (D-37 = yes): v0.7.0 = B0-B5 with R-final last, v0.8.0 = platform breadth; milestone moves in `09` |
| C9 | `ship_track_plan.md` (0, 10.3), `cluster_search_makemodel_plan.md` (9) | GPU use guarded by `flock` on host lock files | not an app-code rule; the program's host procedure replaces it: ask the peer sessions and wait, `-p` names, GPUs 0 and 2 (`11` section 1) |
| C10 | `triton_pipeline_optimization_plan.md` (Q1, Q7), `processing_plans_plan.md` (Q2), `ingest_write_path_investigation.md` (7) | gates and repeats on the private photo set | public-only rule: set B replaces it (`07` OQ-B5-1, `07d` Q2) |
| C11 | `triton_deep_research_2026-09.md` (action 3) | "replace ingest_batch's sync threads with bounded async inference" could read as request-path fan-out | `05` 4A: request-scoped parallelism only as the documented `asyncio.TaskGroup` exemption (Q-J7, D-29); heavy work goes to Celery (B2) |
| C12 | `generic_detector_and_selective_embedding_plan.md` (5, case 6), `triton_pipeline_optimization_plan.md` (Wave 10) | full re-embed in place when the model changes | #365 dual-field migration (v0.9.0); no in-place swap |
| C13 | `00_index.md` section 10 and `09` section 7 (the program itself) | #370 called "importers, defer" and #402 "defer after B4", but both are almost fully shipped | C-370 and C-402 (section 3.6) |
| C14 | `01_b0_close_out.md` B0-1 done-when versus nine program files | the done-when requires no hit for the master-plan file name under `docs`, but the program cites it | resolved: the master plan is deleted and the citations read "the former master plan (rev 4, in git history)" |

No old doc carries a class-index-across-a-boundary rule or a compatibility shim that the program
would keep. The `threading.Lock` objects in the assist loop are single-owner state (2.1), not a
conflict.

## 5. README status updates (`docs/design/README.md`)

**Applied in the consolidation pass** (`docs/design/README.md` rewritten with the "Program plan
(current)" section; the stale status lines inside the kept docs fixed; milestone words follow D-37, so
read `v0.8.0` below as v0.7.0 B5 and `v0.9.0` as v0.8.0). The "Status" column became:

| Doc | New status |
|---|---|
| `program/00_index.md` (new row, added by B0-1) | plan (the 0.7.0 and 0.8.0 program) |
| `auto_label_loop_plan.md` | implemented (reference; S0 not run, pause and events in N16) |
| `cluster_search_makemodel_plan.md` | open (parked research; W2, the W3 privacy part and W4 shipped; v0.9.0 and later) |
| `cq3_label_operations.md` | implemented (reference) |
| `curation_api_contract.md`, `curation_design_rationale.md`, `engineering_standards.md` | implemented (reference) (unchanged) |
| `faces_ocr_resident_crops_plan.md` (new row) | open (step 1 done, step 2 rejected; steps 3-4 in `program/07c` and `07d`, v0.8.0) |
| `generic_detector_and_selective_embedding_plan.md` | implemented (reference; delete guard done; arbitrary-filter clustering parked) |
| `gpu_decode_plan.md` | implemented (follow-ups in `program/07b`, v0.8.0) |
| `processing_plans_plan.md` | open (P1 done; P0, P2-P6 in `program/07d`, v0.8.0) |
| `release_acceptance_plan.md` | open (method for R-final, `program/06`; removed at RP-9) |
| `sam3_full_image_detection_plan.md` | implemented (reference) (unchanged) |
| `selectable_assist_plan.md` (new row) | implemented (reference) |
| `sp_gpu_prep_concurrency.md` | partially done (C1 shipped in #392; C2-C5 in `program/07b`, v0.8.0) |
| `triton_pipeline_optimization_plan.md` | partially done (Waves 0, 1, 5, 7 shipped; the rest in `program/07`-`07d`, v0.8.0, and v0.9.0 for Waves 8 and 10) |
| `task_queue_plan.md`, `codebase_audit_plan.md`, `model_family_versions_plan.md` (rows added by B0-1) | plan (task detail by ID for `program/`) |

Rows removed (the docs are deleted or moved): `combine_preview_warnings_plan.md`,
`docs_overhaul_plan.md`, `label_confirmation_plan.md`, `program_plan.md`, `repo_cleanup_plan.md`,
`ship_track_plan.md`, `v060_execution_path.md`, `v060_wave_plan.md`, `v070_wave_plan.md`,
`ingest_surfaces_plan.md`, `storage_sizing_and_ingest_baselines.md`,
`v060_serving_backend_trade_study.md`, `v060_vector_db_evaluation_plan.md`.

Rows added to the research table (moved notes): `ingest_write_path_investigation.md`,
`storage_sizing_and_ingest_baselines.md`, `v060_vector_db_evaluation_plan.md`,
`v060_serving_backend_trade_study.md`. Add them to `docs/research/README.md` too. The research table
also lacks rows for existing notes: `cluster_config_baseline.md`, `cluster_program_outcome.md`,
`cluster_research_9oct.md`, `cropwright_health_2026-10-29.md`, `labelled_benchmarks.md`,
`retrieval_baseline.md`, `spatial_datasets.md`, `visual_judge_e0.md`, `audits/phase_a1_audit.md`.

The section "Issues without a plan doc" lists only closed issues (#38, #39, #41, #43, #46, #53,
#55, #57-#66). Replace it with one line pointing to `program/09_github_actions.md` section 7 (the
issue map).

Each kept doc whose own status line is stale gets it fixed in the same commit:
`auto_label_loop_plan.md`, `cq3_label_operations.md`, `selectable_assist_plan.md`,
`cluster_search_makemodel_plan.md`, `sp_gpu_prep_concurrency.md`,
`triton_pipeline_optimization_plan.md`, `processing_plans_plan.md` (v0.9.0 -> v0.8.0),
`faces_ocr_resident_crops_plan.md`, `generic_detector_and_selective_embedding_plan.md`.

**Wrong in the README today:** `repo_cleanup_plan.md` (all 9 PRs merged, not "PR 1 of 9"),
`docs_overhaul_plan.md` (merged in #294), `cq3_label_operations.md` (implemented),
`auto_label_loop_plan.md` (S11 done), `label_confirmation_plan.md` (fully implemented),
`release_acceptance_plan.md` (the run is scripted), `v070_wave_plan.md`, `ship_track_plan.md`
(shipped or re-planned), `v060_execution_path.md` (0.6.0 shipped), `cluster_search_makemodel_plan.md`
(partly built), `storage_sizing_and_ingest_baselines.md` (quantization done),
`sp_gpu_prep_concurrency.md` (C1 shipped), `v060_serving_backend_trade_study.md` (six of seven
experiments ran), `triton_pipeline_optimization_plan.md` and `processing_plans_plan.md`
(milestones). Missing rows: `selectable_assist_plan.md`, `dataset_import_more_formats.md`,
`dataset_import_voc_openimages.md`, `faces_ocr_resident_crops_plan.md`, `improvement_backlog.md`,
`ingest_write_path_investigation.md`.

## 6. Removal plan

**Waves.** R-1 runs in B0-1b (section 3.0). R-2 runs at RP-9 (after the R-final run). R-3 runs in
the docs commit of the named B5 sub-batch. KEEP-REFERENCE docs are never removed by this program.
R-2 and R-3 follow the plan lifecycle of [00_index.md](00_index.md) section 8.2 (the closing PR moves
the lasting knowledge and deletes the plan file in the same PR).

**Status of the waves (2026-11-03).**
- **R-1: DONE in the consolidation pass (every R-1 row below).** Deleted with `git rm`:
  `combine_preview_warnings_plan.md`, `dataset_import_more_formats.md`, `dataset_import_voc_openimages.md`,
  `docs_overhaul_plan.md`, `repo_cleanup_plan.md`, `program_plan.md`, `v070_wave_plan.md`,
  `v060_execution_path.md`, `label_confirmation_plan.md`, `ship_track_plan.md`, `v060_wave_plan.md`,
  `improvement_backlog.md`, `ingest_surfaces_plan.md`. Moved with `git mv` to `docs/research/`:
  `ingest_write_path_investigation.md`, `storage_sizing_and_ingest_baselines.md`,
  `v060_vector_db_evaluation_plan.md`, `v060_serving_backend_trade_study.md`. Content moved first: the
  benchmark protocol, parity chain, gate table and cluster-agreement metric to `docs/PERFORMANCE.md`
  ("Wave 0 parity and FP16 gates"); the gRPC decision rule to `docs/PERFORMANCE.md` ("Ingest
  transports"); the `error_kind` table to `docs-site/docs/guides/ingest-sources.mdx`; the lessons
  learned to `engineering_standards.md` section 5 (items 19 to 27); the ship-track measurement rules to
  `07_b5_overview.md` section 5; the `representatives` rule to `vlm-selection.mdx`. Issue comments
  (section 3.6) are still to post when the owner lets `gh` run.
- **R-2: remove at R-final RP-9** (`release_acceptance_plan.md`).
- **R-3: remove after B5b** (`gpu_decode_plan.md`, `sp_gpu_prep_concurrency.md`), **after B5c**
  (`faces_ocr_resident_crops_plan.md`), **after B5d** (`processing_plans_plan.md`,
  `triton_pipeline_optimization_plan.md`).

**Reference rules.**
- A markdown link (bracketed text followed by a parenthesised path) to a removed file fails `check_docs_vs_code.py` (every `.md` and
  `.mdx` except `CHANGELOG.md`): re-point it or drop it.
- A path in a code comment or docstring (`.py`, `.ts`, `.svelte`, `.js`, `.mjs`, `.sh`) fails the
  cited-paths guard (`scripts/codegen/check_cited_paths.py`): re-point it.
- Plain-text mentions in `docs/reviews/*.md` and in released sections of `CHANGELOG.md` and
  `docs/releases/*.md` are historical records and may stay, except where the file only moved (then
  re-point, it costs nothing).
- `CHANGELOG.md` `[Unreleased]` text ships with 0.7.0: re-point it.
- Open issues: a comment with the new path (section 3.6).

| Doc | Verdict | Wave | Content to move (section -> target) | References to fix |
|---|---|---|---|---|
| `combine_preview_warnings_plan.md` | DELETE | R-1 | none | `docs/design/README.md` row: drop |
| `dataset_import_more_formats.md` | DELETE | R-1 | none (the format table and issue codes are in `docs-site/docs/guides/dataset-import.mdx`) | `CHANGELOG.md` line 134 (`[Unreleased]`, "Design: ..."): re-point to `docs-site/docs/guides/dataset-import.mdx` |
| `dataset_import_voc_openimages.md` | DELETE | R-1 | none | only `dataset_import_more_formats.md` (deleted together) |
| `docs_overhaul_plan.md` | DELETE | R-1 | none | README row: drop |
| `repo_cleanup_plan.md` | DELETE | R-1 | none (`engineering_standards.md` is the record) | `docs/design/engineering_standards.md` lines 4 and 600 (links): replace with "the nine cleanup PRs, #300 to #324" in plain text; README row: drop; `CHANGELOG.md` lines 550, 705 (0.6.0): may stay |
| `program_plan.md` | DELETE | R-1 | none (spatial datasets already in `docs/research/spatial_datasets.md`) | `docs/design/cluster_search_makemodel_plan.md` line 28 (plain): reword to "the retired program plan (git history)"; `ship_track_plan.md` (deleted in the same wave); README row: drop; `00_index.md` section 13 row: keep as a plain-text record; issue comments #365-#370 |
| `v070_wave_plan.md` | DELETE | R-1 | none | `docs/design/auto_label_loop_plan.md` line 6 (link) and lines 23, 557 (plain): replace with "the 0.7.0 wave plan (retired)"; `program_plan.md`, `ship_track_plan.md` (deleted); README rows (also the `auto_label_loop_plan.md` row text): update; `00_index.md` row: keep as text |
| `v060_execution_path.md` | DELETE | R-1 | none | README row: drop; `docs_overhaul_plan.md` (deleted); `00_index.md` row: keep as text |
| `label_confirmation_plan.md` | DELETE-AFTER-MOVE | R-1 | none beyond what `docs-site/docs/guides/vlm-selection.mdx` already holds (the scope table); check that 4's scope definitions match it, add any missing sentence there | `frontend/src/lib/labelConfirmation/vlmScopeCopy.ts` line 4 and `frontend/e2e/stubbed/test_label_confirmation.py` line 1 (cited-paths guard): re-point to `docs-site/docs/guides/vlm-selection.mdx`; `docs/design/selectable_assist_plan.md` line 10 (plain): re-point to the same guide; README row: drop |
| `ship_track_plan.md` | DELETE-AFTER-MOVE | R-1 | 3 measurement protocol -> `docs/design/program/07_b5_overview.md` section 5 (inline the rules it "builds on") | `07_b5_overview.md` line 78: drop "builds on" once inlined; `docs/research/cluster_program_outcome.md` lines 40, 87, 149 (links): re-point to #398 (Z5) and `07_b5_overview.md`; `docs/design/cq3_label_operations.md` line 5 and `sp_gpu_prep_concurrency.md` line 11 (plain): reword to "the retired ship-track plan"; `docs/reviews/cq1_review.md`, `cq3_review.md`, `ui1_ui2_review.md` (plain, historical): may stay; README row: drop; `00_index.md` row: keep as text |
| `v060_wave_plan.md` | DELETE-AFTER-MOVE | R-1 | 0 rule 3 (benchmark protocol) and 2, 2.1 (parity chain, gate table, cluster-agreement metric) -> `docs/PERFORMANCE.md` section "Wave 0 parity and FP16 gates" | `docs/PERFORMANCE.md` line 1008: point at its own section; `scripts/bench/parity_gates.json` `source`: re-point to `docs/PERFORMANCE.md` and `triton_pipeline_optimization_plan.md` section 7; `docs/design/gpu_decode_plan.md` line 4 (link): drop the link, keep "WP-2.4"; moved notes `v060_vector_db_evaluation_plan.md` line 4 and `v060_serving_backend_trade_study.md` line 4 (links): re-point to `../PERFORMANCE.md`; `v060_execution_path.md`, `v070_wave_plan.md`, `docs_overhaul_plan.md` (deleted); README row: drop; issue comments #40, #67, #219, #231, #268 |
| `improvement_backlog.md` | DELETE-AFTER-MOVE | R-1 | 6 lessons learned -> `docs/design/engineering_standards.md` section 5 (pitfalls); every other row is in section 2.14 of this file | `README.md` line 110 (link, "What to do next"): re-point to `docs/design/program/00_index.md`; `docs/handoff/0.7.0-next-steps.md` (superseded and removed; its link is gone); `docs/reviews/rc_summary.md` line 16 (link): re-point to `docs/research/cluster_program_outcome.md`; `docs/design/program/10_standards_and_libraries.md` lines 146, 175 (plain): reword to cite #407; issue comments #398, #400-#403, #406 |
| `ingest_surfaces_plan.md` | DELETE-AFTER-MOVE | R-1 | 8 gRPC decision rule -> `docs/PERFORMANCE.md` "Ingest transports"; the `error_kind` vocabulary -> `docs-site/docs/guides/ingest-sources.mdx` | `docs/PERFORMANCE.md` line 3494: point at its own text; `scripts/bench/ingest_transport_bench.py` line 2 and `src/services/curation/bulk_ingest/object_fetch.py` line 12 (cited-paths guard): re-point to `docs/PERFORMANCE.md` and the guide; `repo_cleanup_plan.md` (deleted); README row: drop |
| `ingest_write_path_investigation.md` | DELETE-AFTER-MOVE (`git mv` to `docs/research/`) | R-1 | whole file -> `docs/research/ingest_write_path_investigation.md`; add the research index rows | `docs-site/docs/operations/storage.mdx` line 19, `docs/releases/v0.6.0.md` line 73 (plain): re-point to the new path; `CHANGELOG.md` line 761 (0.6.0): may stay; `v070_wave_plan.md` (deleted); issue comment #40 |
| `storage_sizing_and_ingest_baselines.md` | DELETE-AFTER-MOVE (`git mv` to `docs/research/`) | R-1 | whole file -> `docs/research/storage_sizing_and_ingest_baselines.md` | `scripts/bench/suite_report.py` line 31 (cited-paths guard), `docs/PERFORMANCE.md` lines 400, 648, `docs-site/docs/guides/ingest-large-folder.mdx` line 505, `docs/design/program/07a_b5_baseline.md` line 11: re-point; `docs/releases/v0.4.0.md` line 118: re-point (moved file); README row: move to the research table; issue comment #45 |
| `v060_vector_db_evaluation_plan.md` | DELETE-AFTER-MOVE (`git mv` to `docs/research/`) | R-1 | whole file -> `docs/research/v060_vector_db_evaluation_plan.md` | `docs/PERFORMANCE.md` line 2204 (link): re-point to `research/v060_vector_db_evaluation_plan.md`; `docker/bench/vectordb/compose.yml` line 1 comment: re-point; the moved note's own links (`v060_wave_plan.md`, `storage_sizing...`): fix with the moves; README row: move; issue comments #40, #67 |
| `v060_serving_backend_trade_study.md` | DELETE-AFTER-MOVE (`git mv` to `docs/research/`) | R-1 | whole file -> `docs/research/v060_serving_backend_trade_study.md` (#231 and #403 still read its rooflines and decision rules) | `docs/design/processing_plans_plan.md` line 395 (plain): re-point; `program_plan.md`, `improvement_backlog.md`, `v060_wave_plan.md` (deleted); README row: move; issue comments #40, #67, #231 |
| `release_acceptance_plan.md` | DELETE-AFTER-MOVE | R-2 (RP-9) | 1 run rules and 5 go/no-go -> `docs-site/docs/operations/release-acceptance.mdx`; phases D, E, H, I, K -> `06` RF-70 to RF-76 and B1 1B 1B-12b (section 3) | `scripts/release/acceptance_run.sh` line 6 (cited-paths guard): re-point to `docs-site/docs/operations/release-acceptance.mdx`; `docs/design/program/00_index.md` line 411, `01_b0_close_out.md` line 178, `06_release_acceptance.md` lines 11, 148: re-point to `06` and the docs-site page; `docs/reviews/rc_summary.md` line 15 (link): re-point to the docs-site page; README row: drop |
| `gpu_decode_plan.md` | KEEP-OPEN, then DELETE-AFTER-MOVE | R-3 after 5b | 13 build record and 6.1 decoder notes -> `docs/PERFORMANCE.md` "Decode once and the GPU decode" and `docs/ARCHITECTURE.md` (GPU pipeline) | at that time: `scripts/bench/gpu_decode_bench.py` line 2 (guard), `docs/PERFORMANCE.md` lines 2976, 3215, 3316, `07b`, `07`, `00_index.md`, `10`, `processing_plans_plan.md`, `faces_ocr_resident_crops_plan.md`, `CHANGELOG.md` line 409 (0.6.0, may stay); issue comments #40, #220 |
| `sp_gpu_prep_concurrency.md` | KEEP-OPEN, then DELETE-AFTER-MOVE | R-3 after 5b | 2-4 diagnosis -> `docs/research/` (new note) or `docs/research/v070_measurement_log.md` | at that time: `docs/benchmarks/v070/sp0/gpu_prep_concurrency_matrix.md`, `docs/reviews/sp_fixes_review.md`, `docs/WHITEPAPER.md`, `07`, `07b`, `00_index.md`, `processing_plans_plan.md`; issue comments #403, #407 |
| `faces_ocr_resident_crops_plan.md` | KEEP-OPEN, then DELETE | R-3 after 5c | none (07c and 07d hold the steps) | at that time: `07`, `07c`, `07d`, `processing_plans_plan.md`; issue comment #220 |
| `processing_plans_plan.md` | KEEP-OPEN, then DELETE-AFTER-MOVE | R-3 after 5d | the branch vocabulary and per-route defaults -> `docs/ARCHITECTURE.md` and the API docs | at that time: `CHANGELOG.md` (`[Unreleased]` mention, re-point), `00_index.md`, `07`, `07b`, `07d` |
| `triton_pipeline_optimization_plan.md` | KEEP-OPEN, then DELETE-AFTER-MOVE | R-3 after 5d | 7 accuracy gates -> `docs/PERFORMANCE.md` parity section; 11 verification log is already in PERFORMANCE | at that time: `docs/PERFORMANCE.md` lines 7, 421, 516, 748, 1007, `docs/VISION_AND_GOALS.md` line 115, `docs/research/triton_deep_research_2026-09.md` line 1 (link), `scripts/bench/parity_gates.json`, the `07` files, `00_index.md`; issue comments #40, #67, #84, #219, #220 |
| `cluster_search_makemodel_plan.md` | KEEP-OPEN | none | none; it is the only design for the parked waves (2.2), too large for an issue body | none now |
| `auto_label_loop_plan.md` | KEEP-REFERENCE, partly superseded | none | none; it is the as-built design (statistics in 6, lock-rule interplay) cited by `docs/WHITEPAPER.md`, the docs-site guide and reviews; its storage (3.1), event file (3.2), budget file (5.8), worker and trigger protocol (D4, 8), control routes (10) and Run panel controls (11) are superseded for the rebuild by [research_auto_label_assist_plan.md](../research_auto_label_assist_plan.md) (lane 6D writes the header note) | links to `v070_wave_plan.md` fixed in R-1 (row above) |
| `issue_plans/issue-346-plan.md`, `issue-406-plan.md`, `issue-347-plan.md`, `issue-348-plan.md`, `issue-398-plan.md`, `issue-400-plan.md`, `issue-401-plan.md`, `issue-416-plan.md` | DELETED by the research branches (2026-11-04) | none | absorbed into the research plans: clustering (346, 406), embeddings (347), spatial (348, 400), auto-label (398), plate text (401), near-duplicates (416); their content lives there | none |
| `cq3_label_operations.md` | KEEP-REFERENCE | none | none; cited by `docs-site/docs/guides/label-operations-and-undo.mdx` and `docs/WHITEPAPER.md` | line 5 fixed in R-1 (ship-track row) |
| `selectable_assist_plan.md` | KEEP-REFERENCE | none | none; cited by `src/services/curation/assist/__init__.py` | line 10 fixed in R-1 (label-confirmation row) |
| `generic_detector_and_selective_embedding_plan.md` | KEEP-REFERENCE | none | none; defines the embedding-state vocabulary that `docs/PERFORMANCE.md` cites | none |
| `sam3_full_image_detection_plan.md` | KEEP-REFERENCE | none | none; 16 is the as-built record | none |
| `curation_api_contract.md`, `curation_design_rationale.md`, `engineering_standards.md` | KEEP-REFERENCE | none | none | none |

**Order inside R-1.** Move content first, then `git mv` the four research notes, then delete. Fix the
links of the moved notes to each other (`v060_vector_db_evaluation_plan.md` cites
`storage_sizing_and_ingest_baselines.md` and `v060_wave_plan.md`) in the same commit. Run
`rg -n '<file name>' --glob '!.git'` for every removed name after the edits; only the plain-text
history listed above may remain.
