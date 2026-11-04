# 00e (review): independent review of the research integration (before any code)

| Field | Value |
|---|---|
| Status | review, written 2026-11-04 on `research/review` (from `research/master`) |
| Reviewer | an independent reviewer (production operations, PostgreSQL, Celery, OpenSearch, GPU inference, ML data curation) who did not write the plans |
| Scope | the seven research plans (`docs/design/research_*_plan*.md`); the integrated program: [00_master_plan.md](00_master_plan.md), [00_index.md](00_index.md), [00b_traceability.md](00b_traceability.md) section 5, [00d_owner_questions.md](00d_owner_questions.md), [00e_research_track.md](00e_research_track.md), [02](02_control_plane_database.md) to [02d](02d_research_seam_schema.md), [03](03_b2_overview.md) to [03g](03g_w5e_plate_text.md), [05](05_b4_stateless_api.md), the 07 files, [08](08_environment_and_setup.md), [10](10_standards_and_libraries.md), [11](11_risks_cleanup_host.md), [13](13_package_pins.md), [v080_platform_breadth_plan.md](../v080_platform_breadth_plan.md) |
| Ground truth | the code on `research/master` (`src/`, `scripts/`, `models/`, compose files, `tests/`) and the official docs or primary sources cited in each finding (read 2026-11-04) |
| Not repeated | anything [00c_plan_review.md](00c_plan_review.md) closed. None of its fixes is reopened by the research integration (checked: R-5, R-13, R-18, R-19, R-20, R-24, R-33, RC-3, RC-9) |
| Method | every claim checked against the code, the DDL text or a primary doc, not against a plan's description of itself; line numbers are from `research/master` and move, so re-find by symbol |

---

## 1. Summary in plain words

**Verdict: yes, with fixes.** W2.0 (the pins PR) can start now: nothing here touches it. Fix
RR-1 to RR-4 in the plan text before the W2.1 lanes are cut, because they change what lanes 1A,
1B, 1F and 3A build. RR-5 must be fixed before the plate prototype window PW-2. There are no
blockers in the strict sense: every fix stays inside the plan's own design.

The research is good. I checked 24 cited facts against their sources (section 3.1). 22 hold
exactly. One is stale: PP-OCRv6 shipped in June 2026, and the plate plan still treats PP-OCRv5 as
current. One more has small wording slips. The code facts the plans cite are accurate:
`nvjpeg_lanes.py`, the Lucene `_knn_field`, `hnsw_m` 16 and `ef_construction` 128, the faiss core
indexes, the "matches Immich" comment and the 204-field items mapping. The cross-topic resolutions
(X-1..X-20) are sound in shape: separate tables with one shared pattern, and one model contract
(X-5). One rationale is wrong: X-1b says PostgreSQL checks `RESTRICT` "at once" and `NO ACTION`
"at the end of the statement" (RR-6).

What is wrong falls into four groups:

1. **Three schema and job-control rules contradict each other** (RR-1, RR-2, RR-3):
   - A paused-then-resumed job is sent again under a task id the workers already hold as revoked.
   - The new immutability trigger refuses the `ON DELETE SET NULL` writes that job pruning and
     partition retention make.
   - The new NULL-project rows (core faces, core near-duplicates, deployment-default calibrations)
     are readable only under a deployment scope that the request paths are not allowed to open.
2. **One lane-ownership rule breaks** (RR-4): three W2.3 research lanes add task modules that lane
   1B did not create, so they would all edit `src/tasks/app.py` at once.
3. **One fact is stale** (RR-5): the plate reader study and the B5c OCR work never mention
   PP-OCRv6.
4. **Scope.** About a third of the 0002 seams have no real v0.7.0 reader. Adding them later would
   cost one new-table revision, so they are deferral candidates (RR-12). The plate normalizer forces
   a close and reopen of every upgraded items index, only for "defence in depth" (RR-8).

**Counts:** blocker 0, high 5, medium 11, low 8.

## 2. Findings

### 2.1 High

**RR-1. A resumed job can be silently skipped by the workers (pause revokes the task id, resume reuses it).**
- *Where:* [02b](02b_jobs_locking_events.md) section 1 (`queued -> paused` row) and section 3.3
  (Pause); [03e](03e_b2_s3_job_runtime.md) 3A-13; [03a](03a_b2_w1_queue_infra.md) 1B-4.
- *What:* pausing a `queued` job does a "best-effort `revoke` of the message"
  (02b:20, 02b:273). Resume sets `queued` and dispatches again. Every dispatch uses
  `task_id = job_id` (02a:31, 03a 1B-4 "`task_id = job_id`"). A Celery worker that got the revoke
  "will skip executing the task", and the revoked-id set lives in worker memory for 10,800 s
  (`CELERY_WORKER_REVOKE_EXPIRES`), capped at 50,000 ids (`CELERY_WORKER_REVOKES_MAX`). So a pause
  then resume within about three hours drops the new message on every worker that saw the revoke.
  The job then sits in `queued` until the reaper fails it with `queue_wait_exceeded` (24 h for
  `assist_loop`). The research made this likely. The auto-label loop's yielded slices are `queued`
  between slices, so a user pause between slices hits exactly this path. Before the integration a
  revoke only went with cancel, which is terminal. Spike S1 item 5 checks only that a revoked queued
  task is skipped, not that a later message with the same id runs.
- *Evidence:* [Celery workers guide, Revoking tasks and its limits](https://docs.celeryq.dev/en/stable/userguide/workers.html#revoking-tasks)
  ("When a worker receives a revoke request it will skip executing the task"; the in-memory list,
  `CELERY_WORKER_REVOKES_MAX` 50,000 and `CELERY_WORKER_REVOKE_EXPIRES` 10,800 s).
- *Fix:*
  - Do not revoke on pause. The claim already refuses any row that is not `queued` (02b section
    3.3: "the row's status is the truth"), and a stale message for a resumed job is harmless,
    because the claim admits one. Keep revoke only for cancel, which is terminal, so the id never
    comes back.
  - If a revoke on pause is still wanted, give each dispatch its own Celery id: add a
    `jobs.dispatch_id uuid` column, regenerate it on every enqueue, and have the claim compare it.
  - Add a test: `queued -> paused -> queued` with a revoke registry that remembers ids (a fake
    worker control) still claims and runs. Add S1 item 10: revoke, then `send_task` with the same
    id, must still execute.

**RR-2. The partition immutability trigger refuses the `SET NULL` writes that retention makes.**
- *Where:* [02d](02d_research_seam_schema.md) section 2.2 (`cluster_partitions`) and section 6;
  [02a](02a_postgres_schema.md) section 9.
- *What:* `cluster_partitions` has `parent_partition_id ... ON DELETE SET NULL` and
  `job_id ... ON DELETE SET NULL` (02d:120-121). The trigger `cluster_partitions_ready_immutable`
  allows only `status` to change once a partition is `ready` (02d:136).
  - The jobs prune deletes terminal `cluster_run` and `persons_cluster` jobs after 90 days. Neither
    kind is on the keep-list (02a section 9). Each delete is a cascade `UPDATE ... SET job_id = NULL`
    on a `ready` partition, which the trigger refuses, so the whole prune batch fails on every
    reaper pass.
  - The same happens when the partition retention deletes a superseded parent of a ready child.
  - The planned test `test_cluster_partition_immutable.py` only deletes a *discarded* partition,
    so it does not catch either case.
- *Evidence:* the DDL lines above; [CREATE TRIGGER](https://www.postgresql.org/docs/current/sql-createtrigger.html)
  (row triggers fire on the referential action's `UPDATE` like on any other).
- *Fix:*
  - Let the trigger allow `status`, and also `job_id` and `parent_partition_id` changing to NULL.
  - Extend the test to cover four cases:
    - prune a `cluster_run` job referenced by a ready partition;
    - delete a ready parent of a ready child;
    - delete a superseded `ready` partition through retention (the children trigger);
    - delete a project that holds ready partitions.

**RR-3. NULL-project rows are only visible in deployment scope, but the code that uses them runs in request or project scope.**
- *Where:* [02a](02a_postgres_schema.md) section 1.4; [02d](02d_research_seam_schema.md) sections
  2.2 and 3; [03b](03b_b2_w2_job_families.md) 4C-7; [05](05_b4_stateless_api.md) ND-C1..ND-C4.
- *What:* the policy lets `project_id IS NULL` rows through only when `op.scope = 'deployment'`,
  and `deployment_tx` is allowed only in "the reaper, beat tasks, the project registry, the upgrade
  step, admin CLIs" (02a section 1.4). The research adds NULL-project rows that other code must
  read or write:
  - Core faces in `cluster_partitions`, `partition_clusters`, `cluster_memberships`,
    `cluster_identities` and `cluster_activations`. 4C-7 says "`GET` reads the result" through
    `src/routers/persons.py`.
  - Core `image_contents` and `near_dup_*` rows at level `core`, written by the core `/ingest` path
    in W4.
  - Deployment-default `cluster_calibrations`, which a *project-scoped* `cluster_run` must read.

  None of these readers is on the allow-list, and the list of nullable tables in 1.4 still names
  only `jobs`, `events`, `config_documents` and `models`. A lane will hit `scope_not_set` or an
  empty result and have to invent a fix.
- *Evidence:* 02a section 1.4 (policy text, allow-list); 02d:101 ("NULL: core faces");
  03b 4C-7; 02d:85 ("NULL: deployment default").
- *Fix:* decide it once, in 02a section 1.4:
  - Add a third scope, `core` (`op.scope = 'core'`), that sees only `project_id IS NULL` rows.
    Open it with `core_tx(session)`, allowed in the core routers and services (persons, core
    ingest, core near-duplicates).
  - Give `cluster_calibrations` a second, `FOR SELECT` policy,
    `USING (project_id IS NULL OR op_scope_allows(project_id))`, the same pattern as
    `p_model_shares_consumer`.
  - Update the list of nullable tables.
  - Extend `test_row_level_security.py`: project scope cannot see core rows, core scope cannot see
    project rows, and a project-scoped run reads a deployment calibration.
  - Alternative (heavier): a sentinel "core" project row, which removes NULL `project_id` from the
    research tables altogether. It also closes RR-18.

**RR-4. The W2.3 research lanes must edit `src/tasks/app.py`, which the plan says no later lane edits.**
- *Where:* [03a](03a_b2_w1_queue_infra.md) lane 1B Owns and 1B-2/1B-3;
  [03b](03b_b2_w2_job_families.md) lanes 4F, ES-B, ND-A.
- *What:* 1B creates empty task modules and kind and route modules "so no later lane edits
  `app.py`" (03a:70-71): `jobs_a`, `jobs_b`, `jobs_c`, `trigger`, `promote`, `periodic`, `b4`, and
  the job-kind and route modules `families_a`, ..., `b4_project`. The research lanes add new ones:
  - 4F: `src/tasks/clustering.py`, `job_kinds/clustering.py`, `task_routes/clustering.py`;
  - ES-B: `src/tasks/embedding_reindex.py`, `job_kinds/b2_embedding.py`;
  - ND-A: `src/tasks/near_dup.py`, `job_kinds/near_dup.py`;
  - later W4 (`capture_backfill`, `near_dup_core_group`) and W5e (`text_read`).

  These three W2.3 lanes code at the same time, so all three would edit `app.py`'s `include=` and
  the aggregating `__init__` files. The 1B text is also inconsistent with itself: `src/tasks/b4.py`
  in Owns, but `b4_fanout.py` and `b4_project.py` in 1B-2.
- *Evidence:* 03a:68-72, 03a:84; 03b:199, 03b:224-225, 03b:238-239.
- *Fix:* choose one of two ways.
  - 1B creates the empty modules for every planned family now: `clustering`, `near_dup`,
    `embedding_reindex` / `b2_embedding`, `capture`, `near_dup_core`, `text_read`.
  - Or `app.py` and the two aggregators discover modules with `pkgutil.iter_modules`, as 2E-8
    already does for upgrade steps. The 1B-5 completeness test (every task has a route and a kind)
    keeps that safe.

  Either way, reconcile `b4.py` with `b4_fanout` and `b4_project`.

**RR-5. The plate-text research misses PP-OCRv6 (released June 2026).** High for research quality; it does not block W2.
- *Where:* [research_plate_text_plan.md](../research_plate_text_plan.md) sections 3.1 and 4.1
  (readers R1-R7), 00e `pt:P1`; [07c](07c_b5_crops_ocr_faces.md) (OCR on the GPU);
  [10](10_standards_and_libraries.md) plate-text row ("PP-OCRv5 on Triton stays the reader").
- *What:* PaddleOCR 3.7.0 (2026-06-11) released PP-OCRv6 under Apache-2.0:
  - three sizes, tiny, small and medium (1.5M to 34.5M parameters);
  - one model for Chinese, English, Japanese and 46 Latin-script languages;
  - vendor-reported gains of +5.1 % recognition and +4.6-4.9 % detection over PP-OCRv5 server,
    plus better digital-display and dot-matrix text.

  PaddleOCR's own model page says the PP-OCRv5 and PP-OCRv6 metrics use different internal sets. The repository
  never mentions PP-OCRv6 (`rg -i ocrv6` finds nothing). The program's goal is that everything is
  researched to current practice before coding. W5c converts the OCR models to TensorRT and W5e
  builds a reader around them, so both could spend GPU-window time on a superseded model.
- *Evidence:* [PaddleOCR 3.7.0 release](https://github.com/PaddlePaddle/PaddleOCR/releases/tag/v3.7.0);
  [PP-OCRv6 blog](https://huggingface.co/blog/PaddlePaddle/pp-ocrv6);
  [text recognition module docs](https://www.paddleocr.ai/main/en/version3.x/module_usage/text_recognition.html)
  (v5 numbers 81.29 / 85.25 confirmed; note on the PP-OCRv6 evaluation sets).
- *Fix:*
  - Add R8 "PP-OCRv6 small and medium recognition (and detection)" to `pt:P1`, measured with the
    repo's own pre- and post-processing.
  - Add a line to 07c: the TensorRT work targets whichever OCR generation `pt:P1` and the 5c gate
    pick.
  - Update the plate row in 10, and add owner question Q-B in section 4.
  - Treat vendor numbers as vendor numbers, as the plan already does.

### 2.2 Medium

**RR-6. X-1b's reason for `NO ACTION` misstates PostgreSQL; keep the choice, fix the reason, and let the test decide.**
- *Where:* [00b](00b_traceability.md) X-1b; [02a](02a_postgres_schema.md) section 1.2;
  02d section 6 `test_profile_fk_cascade.py`.
- *What:* the plan says "`RESTRICT` is checked at once while `NO ACTION` is checked at the end of
  the statement", so "only `NO ACTION` lets the cascade pass whatever order the cascade triggers
  fire in".
  - The docs give deferrability as the difference ("`RESTRICT` does not allow the check to be
    deferred"), plus the rule that `RESTRICT` refuses even an update whose final state would be
    valid.
  - Both forms are `NOT DEFERRABLE` by default, and "a constraint that is not deferrable will be
    checked immediately after every command".
  - So nothing in the docs promises that a non-deferrable `NO ACTION` passes a cascade in every
    trigger order. Whether a pair passes depends on the order the cascade triggers fire.
  - `NO ACTION` is still the right default (never worse than `RESTRICT`).
- *Evidence:* [ddl-constraints, foreign keys](https://www.postgresql.org/docs/current/ddl-constraints.html#DDL-CONSTRAINTS-FK);
  [CREATE TABLE, referential actions and `DEFERRABLE`](https://www.postgresql.org/docs/current/sql-createtable.html).
- *Fix:*
  - Restate the reason: the documented difference is deferrability.
  - Have `test_profile_fk_cascade.py` enumerate every pair that one project delete reaches,
    especially pairs whose referencing row sits deeper in the cascade than the referenced row (the
    cluster, near-duplicate, loop and plate references added here), and generate the list from the
    catalog so new pairs cannot be missed.
  - For any pair that fails, declare the foreign key `DEFERRABLE INITIALLY DEFERRED` (checked at
    commit), or delete in an explicit order in the `project_delete` job (RR-7).

**RR-7. A project delete is now one statement that cascades through millions of rows.**
- *Where:* 02a section 1.2 ("removes all its control data in one statement"); 02a section 9;
  02d sections 2-4.
- *What:* for a 1M-item project the research tables add, in one transaction:
  - up to four partitions of 1M membership rows (about 150 MB each);
  - near-duplicate members per image and per item for three runs and two levels;
  - `image_contents`, loop proposals and selections.

  That is a long transaction with locks on every child table and WAL in the gigabytes. The
  retention rules elsewhere already delete in 10,000-row batches for this reason. A crash midway
  rolls everything back and starts again.
- *Fix:*
  - The `project_delete` job (4C) first deletes the big children in 10,000-row batches, each batch
    in its own transaction, re-runnable: `cluster_memberships`, `partition_clusters`,
    `near_dup_members`, `near_dup_groups`, `near_dup_edges`, `image_contents`,
    `assist_loop_proposals`, `assist_loop_selections`.
  - Then it deletes the project row, whose cascade is now small.
  - Test: a seeded project with 200k membership rows deletes under a `statement_timeout` of 30 s
    per statement.

**RR-8. The plate normalizer forces a close and reopen of every upgraded items index, only for defence in depth.**
- *Where:* [03d](03d_b2_s2_control_data.md) 2F-6; [02c](02c_migration_ops_tests.md) section 2.2;
  X-4; plate plan section 5.1.
- *What:*
  - The plate plan says the domain function already computes `text_norm` and `text_fold`, and that
    "the normalizer is defence in depth so a lower-case or spaced query still matches".
  - Its only reader lands in W5e, yet it costs:
    - a `_close`, `PUT _settings`, `_open` of every 0.6.0 items index in the upgrade window;
    - a new failure mode: an index left closed if the settings call is refused;
    - a mapping-module dependency in W2.2.
  - It also contradicts 2C-7's own rule: plate `region_profile` fields are "not added here: their
    first reader is W5e ... (no unused fields now)".
  - The OpenSearch facts behind it are correct: normalizers accept `pattern_replace`,
    `asciifolding` and `uppercase`; `wildcard` supports a normalizer and not `fuzzy`. The fix
    changes no fact, only whether the seam is needed.
- *Evidence:* [normalizers](https://docs.opensearch.org/latest/analyzers/normalizers/);
  [wildcard field](https://docs.opensearch.org/latest/mappings/supported-field-types/wildcard/);
  plate plan section 5.1 ("defence in depth").
- *Fix (recommended):*
  - Drop 2F-6 and the upgrade close and reopen.
  - The W5e query builder runs every query through the same domain fold, which is the one source of
    truth, plain `keyword` fields carry the folded value, and the fragment field is a `wildcard`
    sub-field with no normalizer.
  - `test_fold_matches_normalizer.py` becomes a test that the query builder and the writer call
    the same function.
- *If kept:* run close, settings and open in a `try/finally` that always reopens; wait for at least
  yellow health per index; verify with `_analyze`; and add a failure-path test (settings refused
  leaves the index open and the upgrade stopped).

**RR-9. `gpu_devices` rows expire under live workers, and some cards never get a row.**
- *Where:* 02a section 9 (`gpu_devices`: "rows not seen for 1 day deleted"); 03a 1B-8 (written
  "at `worker_ready`"); 02d section 1.
- *What:*
  - A GPU worker that runs longer than a day never refreshes `seen_at`, so the reaper deletes its
    card's row unless a reservation holds it. After that the serving planner and `gpu plan` lose
    the card.
  - A card that runs no Celery GPU worker gets no row at all, for example the card the arbiter
    frees for training while the VLM holds it. A `gpu_reservations` insert for that card then fails
    its foreign key.
- *Fix:*
  - Refresh `seen_at` from the worker heartbeat (or from `op.periodic.serving_inventory`, which
    already reads every server), and have `openprocessor gpu plan` and the arbiter upsert all
    NVML-visible cards.
  - Write the retention as `DELETE ... WHERE seen_at < now() - interval '1 day' AND NOT EXISTS
    (reservation)`, so one referenced row never fails the batch.
  - Test: a worker alive for over 24 h keeps its row; a reservation for a card without a worker
    succeeds.

**RR-10. After the upgrade the exact-duplicate key covers only new images.**
- *Where:* [research_near_duplicates_plan.md](../research_near_duplicates_plan.md) sections 5.1
  and 7.2 (ND-8); 02c section 2.2 (the `near_duplicates.py` upgrade step).
- *What:*
  - `image_contents` gets rows from new curation ingest only.
  - The upgrade step clears the `dup_*` fields and marks the levels `rebuild_required`, but writes
    no `content_sha256` for the existing images.
  - `imohash` stops being written (RD-43).
  - So re-ingesting a file that a 0.6.0 project already holds is not caught as an exact duplicate.
- *Fix:*
  - Add a pausable `content_backfill` job (one full read per stored image, measured by `nd:P6`),
    started by the upgrade report like `capture_backfill` (RD-53). Both could share one pass over
    the files.
  - Confirm in ND-8 that the stored bytes equal the uploaded bytes (no re-encode at ingest),
    otherwise the hash does not identify the upload.
  - Test: an upgraded fixture image re-ingested after the backfill is refused as a duplicate.

**RR-11. The dataset list for owner approval (RD-10) misses sets used before the tag, and the request-only sets need lead time.**
- *Where:* [00d](00d_owner_questions.md) RD-10; 00e section 3; the plate, clustering, spatial and
  embeddings plans' dataset tables.
- *What:*
  - RD-10 lists LFW, CO3D, Copydays, UFPR-ALPR, LPLC and others. `pt:P1` (PW-2) also uses the
    OpenALPR benchmark (AGPL repository, image licence not stated) and CCPD (README says MIT).
  - The post-tag windows also use sets RD-10 does not name, several of them under non-commercial
    terms: CompCars, Stanford Cars (CARS196), CUB, SOP, Cambridge Landmarks, Aachen Day-Night,
    RobotCar Seasons, GLDv2, IMC, Deep1B, Text-to-Image and the VIBE sets.
  - UFPR-ALPR, LPLC and RodoSol are "academic, non-commercial, on request": a person must sign and
    send a form, and approval takes days to weeks.
- *Fix:*
  - Make one dataset register in 00e, one row per set: licence or terms with URL, local-only flag,
    the prototypes that use it, the window.
  - Have RD-10 approve the register as a whole.
  - File the access requests now, so PW-1 and PW-2 do not stall.
  - Record in each result note that only aggregates are published.

**RR-12. Several 0002 seams have no real v0.7.0 reader; defer them.**
- *Where:* [02d](02d_research_seam_schema.md) sections 1-2; 00d RD-1;
  [v080_platform_breadth_plan.md](../v080_platform_breadth_plan.md) F9.
- *What:* the test for each seam is whether it is cheap now *and* expensive to add later.
  Adding a new table later is one revision with no backfill. Adding a nullable column later is an
  instant `ALTER TABLE ... ADD COLUMN`. These seams fail that test (section 3.6 gives the reasons):
  - `assist_loop_reviews`: its "0.7.0 reader" is a `CHECK` test, and its routes are v0.8.0.
  - `embedding_evaluation_runs`, `embedding_reindex_runs` and lane ES-B: no candidate model exists
    before RW-2. v0.8.0 F9's own gate says "without a candidate the machinery would be dead code".
    The registry (ES-A) and the readers (ES-C) are the part worth having now.
  - `cluster_backend_validations`: GPU clustering comes only after `clu:CL-9` (RG-4). In 0.7.0 a
    `CHECK (backend = 'cpu')` says the same thing.
  - `cluster_calibrations`: none of the 0.7.0 methods (`ivf`, `minibatch_kmeans`, `union_find`,
    `detector_class`, `edit`) reads one. `calibration_id` can be added later as a nullable column.
  - `project_settings.search`: RRF fusion and `group_by` read it only after RW-2 and v0.8.0 F2.
  - `serving_reports` and `op.periodic.serving_inventory`: with one Triton server the router has
    nothing to choose between. The two-server router lands in RW-1.
  - The plate normalizer (RR-8).
- *Fix:* take the plan's own per-topic fallbacks (00d RD-1: "the re-index job moves to v0.8.0 while
  the registry and readers stay") and move the items above to the wave whose lane first reads them.
  This also removes lane ES-B from W2.3 (RR-13) and removes RR-2's `parent_partition_id` case only
  if lineage is cut too. It is kept, because carry-over needs it.

**RR-13. W2.3 is now 12 concurrent lanes plus 2 follow-ons on one integration branch.**
- *Where:* [00_master_plan.md](00_master_plan.md) W2.3 table; 03b and 03f.
- *What:*
  - Before the integration W2.3 had 9 lanes (4A-4E, 5A-5D). Now it has 4A, 4B, 4C, 4D, 4F, ES-B,
    ND-A and 5A-5D at order 1, then ND-B and 4E.
  - The merge order is 13 steps long, with cross-lane signatures (4F's `record_partition` used by
    4C-7, ND-A's `enqueue_pending` used by 4A, 4B and 4C, 4D's `vlm_client` keyword used by every
    VLM caller).
  - Each merge needs `-m db` and `make contracts-check`.
  - On one shared host with peer sessions this is the program's biggest schedule risk.
- *Fix:*
  - Cut ES-B (RR-12).
  - Split W2.3 into two parts, each closed by a part check: W2.3a = 5D, 4A-4D and 5A-5C (the B2
    families), then W2.3b = 4F, ND-A, ND-B and 4E (the research families, which already depend on
    the W2.3a signatures).
  - The ownership lists hold under the split; ND-B already waits for 4F.

**RR-14. ES-A deletes `_knn_field` but owns none of its callers.**
- *Where:* [03d](03d_b2_s2_control_data.md) ES-A Owns and ES-A3.
- *What:* ES-A3 says "`_knn_field` deleted". Its callers are:
  - `src/clients/curation_opensearch/bodies_core.py:9`, `:97` and `:105`, which lane 2F owns in
    W2.2 "for 2F-6 and 2F-8 only";
  - `src/clients/curation_opensearch/ensure_fields.py:11` and `:441`, which no lane owns.

  2F is cut after ES-A merges, so the conflict is not two lanes editing at once. The problem is an
  ownership gap that a lane brief would have to resolve by itself.
- *Fix:* add "the `_knn_field` call sites in `bodies_core.py` and `ensure_fields.py`" to ES-A's
  Owns, and state that 2F cuts from the head that includes them.

**RR-15. RD-1 bundles ten lanes into one yes or no, and one W2.1-blocking question is not marked.**
- *Where:* [00d](00d_owner_questions.md) RD-1, RD-25.
- *What:*
  - RD-1 asks one yes or no for 1F, ES-A, ES-B, ES-C, 4F, ND-A, ND-B, ND-C and the W4 capture
    lane, while its own text lists per-topic fallbacks. The owner cannot accept the registry and
    reject the re-index job without overriding the recommendation.
  - RD-25 (the `cluster` queue and `cluster-worker`) changes W2.1 lanes 1B-8 and 1C-8, but it is
    not marked "answer before coding".
- *Fix:* split RD-1 into one row per topic, each with the deferral of RR-12 as its default, and mark
  RD-25 "answer before coding". The other seven before-coding rows are the right ones:
  - RD-2, RD-7, RD-8 and RD-10 gate PW-1;
  - RD-23 and RD-51 change 0001 or 0002;
  - RD-4 only moves tables between revisions; it could be dropped from the list (its default needs
    nothing in W2.1).

**RR-16. The research plans' own landing sections and lock files contradict the program they were folded into.**
- *Where:* research_near_duplicates_plan.md section 8 line 860; research_embeddings_search_plan.md
  line 638; research_auto_label_assist_plan.md line 715; X-14; 03b ND-A and 4F.
- *What:*
  - The near-duplicate plan's ND-A row says the lane owns `bodies_core.py`, `semantic_search.py`
    and `review_request.py` (the filter fields). 03b gives `semantic_search.py` and
    `review_request.py` to 4F in W2.3 and the near-duplicate filter to ND-B after 4F.
  - The embeddings plan's GPU wrapper takes `gpu_research.lock` and the auto-label plan takes
    `gpu0.lock`, while X-14 sets one `gpu<N>.lock` per card shared by every topic.
  - A prototype or lane agent that loads its research plan alone (the plan files are written for a
    "fresh agent") follows the old text.
- *Fix:* put a header row in each research plan, "section 8 lanes and the GPU lock are superseded
  by 03b, 03d, 03c and 00b X-14", and correct the three lock lines.

### 2.3 Low

**RR-17. Schema conventions.**
- `cluster_activations` has no primary key: only `UNIQUE NULLS NOT DISTINCT`, and `project_id` is
  nullable, so it cannot be in a primary key. The SQLAlchemy ORM needs one. Add
  `id uuid PRIMARY KEY DEFAULT uuidv7()`.
- `cluster_identities.target` and `level`, and `cluster_activations.target` and `level`, have no
  `CHECK`, unlike `cluster_partitions`. Add them, and add them to `test_clustering_vocab_matches_check.py`.
- Name explicitly every `CHECK` that a later revision alters (`config_documents.kind`,
  `config_activations` axis and kind, `search_projection_tasks.kind`, `project_state.key`), so
  0003 and 0004 drop and recreate them by a stable name under the `ck_` convention of 02a section 1.1.

**RR-18. A project partition can get a NULL-project membership row.**
- *What:* the composite foreign key is `MATCH SIMPLE`, so a row with `project_id` NULL skips it.
  Such a membership is then invisible to the project (RLS) but still cascades with its partition.
- *Fix:* add a `BEFORE INSERT` trigger that copies `project_id` from the partition (one lookup per
  `COPY` batch is cheap), or use the sentinel project of RR-3.

**RR-19. Wording slips in otherwise verified research text.**
- Near-duplicates 3.3 calls DINOv2's 0.45 relative-dedup cut "the stricter" and then "looser". It
  is the lower threshold: a looser match, a stricter cleanup.
- Plate 3.2 gives OpenALPR "70 % with redundancy" on UFPR-ALPR, but the abstract says commercial
  systems were "below 70 %". Quote the paper's table.
- Plate C3 calls a star alignment to the medoid "ROVER on characters". ROVER (Fiscus 1997) builds
  a word transition network by progressive alignment. Either define how insertion slots from
  different readings merge, or name it "medoid alignment vote".
- Near-duplicates 3.1 names Immich's `config.dto.ts`; the defaults live in `system-config.dto.ts`.

**RR-20. `embedding_reindex` (24 h) holds a `jobs` slot without time slices.**
- *What:* X-3's own rule is that a long job must not hold a slot that request-started work needs.
- *Fix:* give it `slice_s` (the mechanism exists after 3A-13) or route it to the `cluster` queue.
  This is moot if ES-B is deferred (RR-12).

**RR-21. Beat details.**
- `op.periodic.cluster_refresh` gates on `inspect(timeout=1).active_queues()`, a broadcast to every
  worker on every tick. Read the TTL'd `worker_capabilities` rows instead (00c already relies on
  them for replica counts).
- `near_dup_rescore` uses an "`INGEST_DISPATCH`-style" lock. Give it its own advisory-lock class in
  `src/config/advisory_locks.py`.

**RR-22. The near-duplicate prune can orphan a freeze's audit link.**
- *What:* `holdout_freezes.near_dup_*_run_id ... ON DELETE SET NULL` loses the run that a freeze
  moved items with once the prune keeps only the current and two superseded runs.
- *Fix:* the prune keeps runs referenced by any freeze.

**RR-23. 02c T-6 is stale.** It still says an unscoped query "returns zero rows", while 02a
section 1.4 (R-13) raises `scope_not_set`. Align the text.

**RR-24. `pt:P2` looks like it waits for W5c.** 00e says it "needs `rec_char_min`, W5c", but it is
scheduled in PW-2, before W5c. The research script can compute the per-character minimum from the
recognition logits in its own venv. Say so, so that PW-2 does not wait for W5c.

## 3. Notes per review dimension

### 3.1 Research quality (24 facts spot-checked)

| # | Claim (plan) | Source checked | Result |
|---|---|---|---|
| 1 | `nvjpegDecodeBatched` uses GPU Huffman for baseline interleaved JPEGs only when the batch is greater than 50 (GPU) | [nvJPEG docs](https://docs.nvidia.com/cuda/nvjpeg/index.html), `GPU_HYBRID` backend | holds |
| 2 | Hardware JPEG decoder on A100, A30, Hopper, Ada, Blackwell, Jetson Thor, not GA102 (GPU) | same, Hardware Acceleration | holds |
| 3 | Scaled decode only with the `HARDWARE` backend; batched ROI only `HARDWARE` (GPU) | same | holds (batched ROI also allows `LOSSLESS_JPEG`) |
| 4 | DALI `jpeg_fancy_upsampling` default False, `hybrid_huffman_threshold` 1,000,000, `adjust_orientation` True, `hw_decoder_load` 0.65 (GPU) | [decoders.image](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/operations/nvidia.dali.fn.decoders.image.html) | holds |
| 5 | Immich near-duplicates: `<=>` cosine distance, `LIMIT 64` (`DUPLICATE_SEARCH_LIMIT`), `distance <= maxDistance` (ND) | [duplicate.repository.ts](https://github.com/immich-app/immich/blob/main/server/src/repositories/duplicate.repository.ts) | holds |
| 6 | Immich `maxDistance` default 0.01, admin range 0.001-0.1 (ND) | [system settings docs](https://docs.immich.app/administration/system-settings/), [config file](https://docs.immich.app/install/config-file/) | holds |
| 7 | Immich single writer `withLock(DatabaseLock.DuplicateDetection)`, transitive merge, stacked/hidden/locked skipped, `duplicatesDetectedAt` (ND) | [duplicate.service.ts](https://github.com/immich-app/immich/blob/main/server/src/services/duplicate.service.ts) | holds (also scoped to the owner's assets) |
| 8 | PDQ match at Hamming <= 31, discard quality <= 49, no exact rotation invariance (ND) | [PDQ README](https://github.com/facebook/ThreatExchange/blob/main/pdq/README.md) | holds (256 bits and the 64x64 Jarosz / 16x16 DCT come from `hashing.pdf`, not the README) |
| 9 | SSCD: `sscd_disc_mixup` recommended, short side 288, cosine > 0.75 is a copy at 90 % precision on DISC, code MIT, weights licence unstated (ND) | [sscd-copy-detection README](https://github.com/facebookresearch/sscd-copy-detection) | holds |
| 10 | Lucene HNSW defaults `m` 16, `ef_construction` 100, ignores `ef_search` (uses `k`); faiss `ef_search` 100; nmslib deprecated (EMB) | [methods and engines](https://docs.opensearch.org/latest/mappings/supported-field-types/knn-methods-engines/) | holds |
| 11 | `half_float` vectors and 2/4-bit SQ new in OpenSearch 3.9 (EMB) | [3.9 release notes](https://github.com/opensearch-project/opensearch-build/blob/main/release-notes/opensearch-release-notes-3.9.0.md), [blog](https://opensearch.org/blog/get-to-know-opensearch-3-9/) | holds (`half_float` only at 1x and 16x compression) |
| 12 | Normalizer may use `pattern_replace`, `mapping`, `asciifolding`, `uppercase`, `lowercase`, `trim` (PT) | [normalizers](https://docs.opensearch.org/latest/analyzers/normalizers/) | holds |
| 13 | `wildcard` since 2.15, supports a normalizer, no `fuzzy`, `term` slower than on `keyword` (PT) | [wildcard](https://docs.opensearch.org/latest/mappings/supported-field-types/wildcard/) | holds |
| 14 | leidenalg GPL-3.0-or-later, igraph GPL-2.0-or-later, GPLv3 section 13 allows combining with AGPLv3; research dependencies only (CLU) | [PyPI leidenalg](https://pypi.org/project/leidenalg/), [PyPI igraph](https://pypi.org/project/igraph/) | holds (leidenalg 0.12.0 `license_expression` GPL-3.0-or-later; igraph 1.0.0 "GPL") |
| 15 | HDBSCAN through scikit-learn (BSD-3) and cuML (Apache-2.0) (CLU) | [scikit-learn HDBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html) | holds |
| 16 | fast-plate-ocr MIT, 1.1.0 on 2026-03-14 (PT) | [PyPI JSON](https://pypi.org/pypi/fast-plate-ocr/json) | holds (`license_expression` MIT; no classifier) |
| 17 | PP-OCRv5 mobile rec 81.29 % average, English 66.00 %; `en_PP-OCRv5_mobile_rec` 85.25 % (PT) | [text recognition docs](https://www.paddleocr.ai/main/en/version3.x/module_usage/text_recognition.html) | numbers hold; **the claim that v5 is current is stale** (RR-5) |
| 18 | Celery: `terminate=True` "you must never call this programmatically"; revokes in worker memory; pause must be cooperative (AL) | [workers guide](https://docs.celeryq.dev/en/stable/userguide/workers.html#revoking-tasks) | holds; the consequence for resume is RR-1 |
| 19 | Multi-model fusion raises the mean plate recognition rate from 92.4 % to 97.6 %, 4-6 models recommended (PT) | [arXiv 2309.04331 HTML](https://arxiv.org/html/2309.04331), section 4 and conclusions | holds |
| 20 | ROVER (Fiscus, ASRU 1997) is the standard align-then-vote for unequal-length outputs (PT) | the cited DOI | holds as a citation; C3 is not literally ROVER (RR-19) |
| 21 | UFPR-ALPR: 78.33 % with temporal redundancy (PT) | [arXiv 1802.09567](https://arxiv.org/abs/1802.09567) abstract | holds; the OpenALPR "70 %" differs from the abstract's "below 70 %" (RR-19) |
| 22 | DINOv2 curation: SSCD, k = 64, > 0.6 self-dedup, > 0.45 relative dedup (ND) | [arXiv 2304.07193](https://arxiv.org/abs/2304.07193) appendix | holds; wording slip (RR-19) |
| 23 | Code facts: `DEFAULT_SIMILARITY_THRESHOLD = 0.99` "matches Immich" (`src/services/duplicate_detection/models.py:5-9`), Lucene `_knn_field` (`src/clients/curation_opensearch/base.py:27-49`), `hnsw_m` 16 / `ef_construction` 128 (`src/config/curation.py:159-160`), faiss core indexes (`src/clients/opensearch/indexes.py:125`), `models/gpu_image_prep/1/nvjpeg_lanes.py` | the tree | holds |
| 24 | Items mapping far below `index.mapping.total_fields.limit` 1000 (EMB) | built `_items_body()`: 204 fields, 2 nested | holds |

No finding on the GPU, near-duplicate, embeddings or clustering literature beyond RR-19. The auto-label
plan's vendor claims (SAM 3 data engine) are marked vendor-reported, as they should be.

### 3.2 Schema coherence

- **Names and conventions:** consistent with 02a 1.1 (`uuidv7()`, `timestamptz`, `text` plus `CHECK`,
  one vocabulary module per closed list). Gaps: RR-17.
- **Foreign keys and project delete:**
  - every project-scoped research table cascades from `projects`;
  - profile pins reference `config_revisions` with `NO ACTION`, and job references are
    `SET NULL` or on the keep-list. That is coherent, except RR-2 (`SET NULL` against the
    immutability trigger) and RR-6 (the stated reason).
  - Two more checks hold:
    - the single-column `partition_id` foreign key next to the composite one is correct
      `MATCH SIMPLE` handling;
    - the keep-list (02a section 9) covers `assist_loop_runs` and `embedding_reindex_runs`.
- **RLS:** every table with a `project_id` column gets the policy, through 1A's introspection test.
  The policy's NULL handling does not fit the new core and default rows (RR-3). The loop children's
  `EXISTS` policy is acceptable because budgets bound them (02d 2.4).
- **Indexes for the stated queries:**
  - `ix_cluster_partitions_recent`, `ix_cluster_memberships_cluster`, the one-primary partial
    index, the `near_dup_runs` current and building partial indexes, and `uq_jobs_one_active` cover
    the queries the plans name.
  - Missing: none. The retention deletes go by `partition_id` (the leading primary-key column).
- **Duplicate concepts:** none. X-1 keeps four grouping tables apart for real lifecycle reasons, and
  X-5 joins the two model registries through one contract class.
- **Revision order:**
  - `0001_control_plane` (with the in-place changes), then `0002_research_seams` (1F, after 1A
    merged), then `0003_near_duplicates` (ND-A, the only W2.3 writer), then `0004_region_text`
    (W5e, pending RD-4).
  - The chain is linear, with one writer per wave and fixed `--rev-id` values. It is coherent.
  - With the RR-12 cuts, 0002 shrinks; the later tables come in a new revision of their landing
    wave.

### 3.3 Jobs and queues

- **Kinds and states:**
  - `retry_count` (G-1), `pause_reason`, `Yield` (`running -> queued`, outcome `yielded`) and
    cancel-is-terminal are sound and closed by tests (02b 3.4).
  - The one hole is revoke plus id reuse (RR-1).
- **Queues and worker shapes:**
  - `cluster` (CPU) and later `text_read` are each declared with their consumer.
  - No GPU-prep queue is right while Triton owns the GPU (S-11): the jobs only call Triton, and
    `gpu_affinity` only picks the server.
  - The GPU queues stay the trainer, export and evaluator: they use the card directly under
    `gpu_reservations` by UUID, which the research improves (X-6), apart from RR-9.
- **Time-sliced loops against `acks_late` and the visibility timeout:**
  - A 600 s slice under a 30 / 35 min limit stays far inside the 6 h visibility timeout.
  - The re-dispatch after the commit plus the status claim makes a duplicate harmless.
  - Fairness holds on a FIFO list with prefetch 1.
  - No finding beyond RR-1 and RR-20.
- **Beat:** the research entries (`serving_inventory`, `near_dup_rescore`, `near_dup_full_build`,
  `near_dup_prune`, and `cluster_refresh` starting `cluster_run`) are idempotent and expire like
  the others. Minor: RR-21.
- **Events and durable records:** X-9 is right. A 24-hour `events` table is a feed, not an audit;
  every topic keeps its own record. No findings.

### 3.4 OpenSearch

- **Field families:** one mapping module per family through 2F-8's additive registry keeps the W2.3
  lanes file-disjoint. No findings.
- **Creation-time settings:** the `index.knn = true` door test (2F-7) and `KnnMethodSpec` per field
  (immutable in OpenSearch) are right.
- **The 0.6.0 close, settings and reopen step (2F-6):**
  - It is safe only inside the stop-the-world window, with an always-reopen `finally`.
  - Downtime is per index (seconds each on one node, more for large shards on reopen recovery).
  - Old faiss-engine project indexes reload their native graphs on the first search after reopening.
  - Recommendation: remove the step (RR-8).
- **Mapping size:** today's items body has 204 fields and 2 nested. The plate keys, `cluster_*`,
  `near_dup_*`, `capture_*` and up to three spaces per target stay far below the default limits
  (1,000 fields, 50 nested). No findings.

### 3.5 Lanes and waves

Each wave's lanes and their owned paths (03a, 03d, 03e, 03b, 03f, 03c Owns lines):

| Wave | Lanes at the same time | Overlaps found |
|---|---|---|
| W2.1 | 1-0, then 1A, 1B, 1C, 1E; then 1F (after 1A), 1D (after 1A, 1B) | none in files. 1F adds import lines to `src/db/models/__init__.py` after 1A merges, which is sequential. RR-4 is a gap in 1B's list, not an overlap |
| W2.2 | 2A, 2B, 2C, 2D, 2E, ES-A, 3B, 3A (after 2A); 2F (after 2A-2E, ES-A) | ES-A's `_knn_field` deletion touches 2F's and an unowned file (RR-14). `refresh_state_store.py` (2C) and `delete.py` (2A) pass to W2.3 lanes later, which is sequential |
| W2.3 | 4A, 4B, 4C, 4D, 4F, ES-B, ND-A, 5A, 5B, 5C, 5D; then ND-B (after ND-A, 4F), 4E | `src/tasks/app.py` and the aggregators (RR-4). `env.template`, `retired_env.py` and `contracts/` are shared under the documented append and regenerate rule (03 section 4), so not a defect. `wire.py`, `semantic_search.py` and `review_request.py` go 4F first, then ND-B, which is sequential and correct. The pattern-defined sets (4A routers, 4D VLM call sites, ND-A pending writers) are listed by the coordinator at the cut, which is acceptable |
| W2.4 | 6A, 6B, ES-C; then `b2/frontend-spaces`, 6C, 6D | ES-C's 40 files today include `item_scores/near_dup.py` and `curation/clustering/*`, which W2.3 deletes or moves. Recompute the list on the W2.3 head (the plan says "at the cut"; this note only confirms it must be the W2.3 head) |

- **Every seam has a lane with a red and a green test:** yes, by 00b 5.2 and 5.4, with three gaps:
  - resume after a revoke (RR-1);
  - ready-partition retention (RR-2);
  - the core and default reads under RLS (RR-3).
- **Wave size:** W2.3 is too large as one part (RR-13).
- **The prototype track's GPU time:**
  - 31 GPU hours over W2.1 to W4, on idle cards, never during a part check, a live run or the 4B-5
    benchmark, under one per-card lock (X-14). That is realistic only if:
    - the lock files in the research plans are corrected (RR-16);
    - the dataset requests are filed in time (RR-11);
    - each prototype keeps its CPU threads capped, because PW-2 overlaps the busiest coding wave
      (W2.3) on the same host.
  - The program's own windows stay separate (W3 window 1 on GPU 0, windows 2 and 3 program-only).

### 3.6 Scope creep against value

| Seam | Verdict | Why |
|---|---|---|
| `retry_count`, `pause_reason`, `Yield`, cancel-is-terminal (3A-12..3A-15) | keep | fixes real bugs (G-1, G-2) in a file being rewritten anyway |
| `gpu_devices` by UUID, `gpu_reservations(gpu_uuid)`, `training_runs.gpu_uuids` | keep (with RR-9) | removes the measured index-order trap; changing a primary key later is a real migration |
| `model_engines` contracts, `EngineInputContract` (X-5) | keep | one pixel contract for the CPU and GPU routes and the registry |
| embedding registry and spaces (ES-A), readers (ES-C) | keep | vector identity by space, not dimension; deletes the field literals |
| re-index job, `embedding_reindex_runs`, `embedding_evaluation_runs` (ES-B) | **defer to RW-2 / v0.8.0 F9** | no candidate model before the tag; F9's own gate calls it dead code without one (RR-12) |
| clustering S-core: partitions, memberships, identities, activations, events (4F) | keep | replaces the `cluster_id` namespace and region clustering in the same release; the upgrade writes the first partition |
| `cluster_calibrations`, `cluster_backend_validations` | **defer to RW-3** | no 0.7.0 method reads a calibration; `backend` is `cpu` only (RR-12) |
| near-duplicate runs, holdout guard, exact key (ND-A, ND-B) | keep (with RR-10) | closes #473 and #416, real correctness bugs |
| `assist_loop_*` record, budgets, ledger, `vlm_usage_hourly` (4D) | keep | usage history cannot be rebuilt later; pause and budgets ship in 0.7.0 |
| `assist_loop_reviews` | **defer to v0.8.0 F3** | no route or reader in 0.7.0 (RR-12) |
| `serving_reports`, `serving_inventory`, the serving router | **defer to RW-1** | one server in 0.7.0 |
| `project_settings.search` | **defer to RW-2 / F2** | nullable column added in one statement later |
| plate normalizer and the upgrade close and reopen (2F-6) | **cut** | defence in depth only; the domain fold is authoritative (RR-8) |
| capture metadata at ingest (W4) | keep, but restate the reason | RD-53's backfill from stored files contradicts "the only signal that waiting can lose"; the real reason is the capture-date filter and the plate consensus basis |

### 3.7 Privacy and compliance

- **Set P:**
  - X-13 and RD-8 are strict and consistent: a seeded local copy; public text "met / not met";
    aggregates only with consent; never to a hosted model.
  - One gap: seven topics each copying "a seeded sample" makes seven private copies. Use one shared
    local copy with one manifest, and record when it is removed after its wave.
- **Location:** RD-51 (no coordinates in 0.7.0, only `has_location`), RD-52 (no raw body serial),
  RD-54 (strip GPS on export) and the location canary are sound. No findings.
- **Plate strings:** plate plan 6.8 and 7 keep every plate string out of committed files, with
  outputs under the gitignored `artifacts_local/`, no plate text in PostgreSQL, and a log redaction
  processor plus a sink-inventory test in W4. Minor: "made-up canaries that no jurisdiction issues"
  cannot be guaranteed. Use strings that fail every plate format, for example with a character
  outside `[A-Z0-9]`.
- **Dataset licences:** RR-11. The non-commercial sets are fine for local evaluation with published
  aggregates. The register should say so per set.
- **Names and paths:** the research text uses `$SCRATCH` and `<scratch>`, never a machine path. The
  naming-leak scan passes on this branch (section 5).

### 3.8 Owner question list (00d)

- Deduplication from 112 to 80 is honest. Every merged row keeps its sources, and I found no row
  that asks the same thing twice.
- The eight "answer before coding" rows are right in kind. Two changes: add RD-25, and RD-4 could
  leave the list (RR-15).
- Missing rows that block work:
  - PP-OCRv6 as a reader candidate (RR-5), before PW-2;
  - the RD-1 split per topic with the RR-12 deferrals, before W2.1;
  - who files the academic dataset requests (RR-11), before PW-1.

## 4. Questions that need the owner

1. **Q-A (before W2.1).** Accept the deferrals of section 3.6? They mean: no re-index job,
   evaluation runs, calibrations, backend validations, loop reviews, serving reports or
   `project_settings.search` in 0.7.0, and no plate normalizer. Each moves to the wave that first
   reads it. Recommended: yes. This splits RD-1 into per-topic answers.
2. **Q-B (before PW-2).** Add PP-OCRv6 (small and medium) as a reader arm in the plate study and as
   a candidate for the B5c OCR work? Recommended: yes. It is Apache-2.0, like PP-OCRv5.
3. **Q-C (before PW-1).** The UFPR-ALPR and LPLC datasets need a signed academic request and allow
   non-commercial use only. Will you (or a delegate) file the requests now, and do you accept
   local-only evaluation with published aggregates under those terms? Recommended: yes.
4. **Q-D (before W2.1).** Answer RD-25 (the `cluster` queue and `cluster-worker`) with the other
   before-coding rows, because lanes 1B-8 and 1C-8 build it in W2.1.

## 5. Checks run on this branch

- `.venv/bin/python scripts/codegen/check_naming_leaks.py`: pass.
- `.venv/bin/python scripts/docs/check_docs_vs_code.py`: pass.

## 6. Planner responses (2026-11-04, final pass on `research/master`)

Every finding was checked against the tree and the plan files before the fix. The owner rejected
the review's scope cuts (Q-A: "keep all seams now and ensure the plans are updated accordingly to
do in the right order and waves"); its correctness findings all apply. Owner answers of the same
day are recorded in [00d](00d_owner_questions.md) section 0 and the [00_index.md](00_index.md)
register; the plan changes are mapped in [00b](00b_traceability.md) section 5.6.

| Finding | Verdict | Fix (file, lane, test) |
|---|---|---|
| RR-1 pause revokes, resume reuses the id | fixed | pause never revokes (revoke only on cancel; a stale message is acked by the claim): [02b](02b_jobs_locking_events.md) state table and pause text; [03e](03e_b2_s3_job_runtime.md) 3A-13; [03a](03a_b2_w1_queue_infra.md) spike S1 item 10. Test `tests/tasks/test_pause_resume_after_revoke_registry.py` |
| RR-2 immutability trigger vs `SET NULL` | fixed | `cluster_partitions_ready_immutable` lets `job_id` and `parent_partition_id` go to NULL and nothing else: [02d](02d_research_seam_schema.md) section 2.2; 03a 1F-1; four test cases in 02d section 6 |
| RR-3 NULL-project rows invisible in request or project scope | fixed | a third RLS scope `core` (`op_scope_is_core()`, `core_tx` with a ratchet allowlist) for core near-duplicate rows; a read-only `FOR SELECT` policy for the default calibrations: [02a](02a_postgres_schema.md) section 1.4; 03a 1A-13; ND-C writes under `core_tx` ([05](05_b4_stateless_api.md) 4A-4b). RLS test cases added |
| RR-4 later lanes would edit `app.py` | fixed | 1B pre-creates every planned research task module, its `include=` entry and both aggregator entries (`clustering`, `embedding_reindex`, `near_dup`, `near_dup_core`, `source_backfill`, `region_text`; `b4_fanout`, `b4_project`): 03a 1B Owns, 1B-8. Test `tests/tasks/test_planned_modules_included.py` |
| RR-5 PP-OCRv6 missing | fixed (owner Q-B) | one short PP-OCRv6 arm against PP-OCRv5 in `pt:P1` (PW-2), also deciding the B5c OCR model: [00e_research_track.md](00e_research_track.md) section 3; [07c](07c_b5_crops_ocr_faces.md) research amendments; [03g](03g_w5e_plate_text.md); [10](10_standards_and_libraries.md) |
| RR-6 X-1b reason misstated | fixed | the reason now says the documented difference is deferrability; the project delete no longer relies on one cascade (RR-7) and the catalog-enumerated test decides: 02a section 1.2; 00b X-1b |
| RR-7 one-statement cascading project delete | fixed | 4C-5 deletes per table in reverse foreign-key order read from the catalog, 10,000 rows per transaction, the `projects` row last: [03b](03b_b2_w2_job_families.md) 4C-5. Test `tests/projects/test_project_delete_batched.py` (200,000 rows under a 30 s statement timeout, resume after a kill) |
| RR-8 the normalizer forces close and reopen | **rebutted on scope (owner Q-A keeps it); the safety part fixed** | the step is per index with a dry run and stated downtime, progress per index, a `try/finally` reopen with a wait for yellow, an `_analyze` check, stop-and-report rollback and an idempotent re-run: [02c](02c_migration_ops_tests.md) section 2.2; [03d](03d_b2_s2_control_data.md) 2F-6. Test `tests/opensearch/test_analysis_settings_failure.py` |
| RR-9 `gpu_devices` rows expire, some cards never get one | fixed | `seen_at` refreshed by every GPU worker heartbeat and by `op.periodic.serving_inventory` (every card Triton reports); a worker registers every NVML-visible card; every reservation upserts its card in the same transaction; retention skips referenced rows: 02a sections 5 and 9; 03a 1B-8; [03c](03c_b2_w3_beat_reaper_workers.md) beat table; [03f](03f_b2_s5_gpu_queues.md) 5A-6. `openprocessor gpu plan` stays a reader (bash, no database client; 00b section 5.5). Tests in 1B-8 and `tests/training/test_reserve_card_without_worker.py` |
| RR-10 the exact key covers only new images | fixed | one pausable `source_backfill` job (02b) with a content step (ND-A, ND-11) and the W4 capture step in the same pass; ND-8 first proves stored bytes equal uploaded bytes: 03b ND-A; 02c section 2.2; 05 CM-3; [06](06_release_acceptance.md) RF-82. Tests `tests/near_duplicates/test_stored_bytes_equal_upload.py`, `test_backfill_then_reingest_refused.py` |
| RR-11 the dataset list | fixed (owner RD-10, Q-C) | one dataset register, 00e_research_track section 4.1; UFPR-ALPR, LPLC and RodoSol skipped, so no request is needed; post-tag sets get their row before their window |
| RR-12 defer seams without a 0.7.0 reader | **rebutted** | the owner keeps every seam now (Q-A, RD-1). Each kept seam has a writer and a red test in its lane (00b section 5.2); the re-index job is time-sliced (RR-20) |
| RR-13 W2.3 too large | fixed (owner Q-A) | W2.3a = 5D, 4D, 4A, 4B, 4C, 5A, 5B, 5C (ports, one cross-lane signature); W2.3b = 4F, ND-A, ES-B, then ND-B, then 4E. Every research call W2.3a lanes made moved to W2.3b (the `near_dup` scorer deletion and the `enqueue_pending` calls to ND-A; person grouping as a partition to 4F-9; the IVF retrain and the loop's cluster reader to 4F-10). ES-B is kept, not cut: 03b header and sections 2-8; [00_master_plan.md](00_master_plan.md) sections 2-4 |
| RR-14 ES-A owns none of `_knn_field`'s callers | fixed | ES-A owns the call sites in `bodies_core.py` and `ensure_fields.py`; 2F cuts from the head that includes them: 03d ES-A |
| RR-15 RD-1 bundled; RD-25 unmarked | fixed by the answers | the owner approved RD-1 for every topic and decided RD-25 (Q-D); no row blocks coding now: 00d sections 0 and Count |
| RR-16 research plans contradict the program | fixed | a "Superseded parts" header in all eight research plan files (lanes, waves, lock, GPU hours point to the program); the embeddings and auto-label lock lines use `gpu<N>.lock` (00b X-14) |
| RR-17 schema conventions | fixed | `cluster_activations.id` primary key; `target` and `level` `CHECK`s on identities and activations; named `ck_` constraints for every `CHECK` a later revision alters: 02a, 02d; 03a 1A-13, 1F-1 |
| RR-18 NULL-project membership | fixed | `BEFORE INSERT` trigger `cluster_children_project_from_parent` (and the same on near-duplicate groups and members): 02d; test `tests/db/test_children_project_from_parent.py` |
| RR-19 wording slips | fixed | near-duplicates 3.3 (the 0.45 cut is the lower threshold: a looser match, a stricter cleanup) and 3.1 (`system-config.dto.ts`); plate 3.2 quotes "below 70 %"; plate C3 is named the "medoid alignment vote" |
| RR-20 `embedding_reindex` without slices | fixed | `slice_s` 600 with a 30 / 35 minute limit: 02b kind table; 03b ES-B1 |
| RR-21 beat details | fixed | `cluster_refresh` reads a fresh `worker_capabilities` row instead of `inspect()`; `near_dup_rescore` has its own advisory-lock class `NEAR_DUP_RESCORE`: 03c beat table; 02b section 4 |
| RR-22 prune orphans a freeze's link | fixed | the prune keeps every run a freeze references: 02a section 9 |
| RR-23 T-6 stale | fixed | T-6 now raises `scope_not_set`: 02c |
| RR-24 `pt:P2` waits for W5c | fixed | the script computes the per-character minimum from the recognition logits in its own venv: 00e_research_track `pt:P2` |
| 3.5 ES-C's file list | fixed | recomputed on the merged W2.3b head: 03c ES-C |
| 3.6 capture reason | fixed | restated as the capture-date filter and the plate consensus basis: 05 section 6c |
| 3.7 seven set-P copies; plate canaries | fixed | one shared seeded copy and manifest, removal recorded ([11](11_risks_cleanup_host.md) rule 10); canaries fail every plate format (03g, plate plan 6.8) |
| 3.5 the prototype GPU time | superseded by the owner | RD-2 as answered: short tiered runs, about 10 GPU hours before the tag (was 31), post-tag windows only on the owner's OK: 00e_research_track sections 2-4 |

## 7. Re-check (2026-11-04)

Re-checked on `research/recheck`, cut from `research/master` at 77ee46823 (the merge a005ec037 and
the fixes 2137d0ceb..77ee46823), with the same rules as sections 1-5: the plan text checked against
the code and the docs, not against the planner responses.

### 7.1 Verdict

**Yes, with fixes.** RR-1 to RR-5 are resolved in the plan text. The W2.3a and W2.3b split is
file-disjoint and sensibly sized. The 10-hour prototype track adds up and its windows are coherent.
The bounded close and reopen step is sound in itself, but the plan puts it in the wrong place in the
upgrade order.

The fixes introduced **one new high finding (RN-1)**: the batched project delete of RR-7 runs into
the immutability triggers. It must be fixed in 02a and 02d before W2.1 lane 1F writes the
triggers. The rest are three mediums and four lows.

**Counts:** blocker 0, high 1, medium 3, low 4.

### 7.2 RR-1 to RR-5, checked

| Finding | Resolved? | Evidence |
|---|---|---|
| RR-1 pause revokes, resume reuses the id | yes | 02b:20 and 02b:273 say "no revoke"; revoke stays only for cancel (02b:259). 03e 3A-13 matches. New red test `test_pause_resume_after_revoke_registry.py` (02b:294). The claim refuses a non-`queued` row, so a stale message is acked as `claim_skipped` (02b:293). Leftover: RN-5 |
| RR-2 immutability trigger vs `SET NULL` | yes, for the `SET NULL` updates | 02d:139-141 lets `job_id` and `parent_partition_id` go to NULL; 03a 1F-1; 02d:434 adds the four test cases. The new batched delete still hits the triggers (RN-1) |
| RR-3 NULL-project rows | mostly | 02a 1.4: a third scope `core`, `core_tx` with an allow-list naming files that exist (`src/routers/persons.py`, `src/routers/query_duplicates.py`, `src/services/face_identity/`, `src/services/duplicate_detection/`), the list of nullable tables, the read-only `p_cluster_calibrations_default` policy, and the RLS test cases. Gap: RN-2 |
| RR-4 later lanes would edit `app.py` | yes | 03a lane 1B Owns pre-creates `clustering`, `embedding_reindex` / `b2_embedding`, `near_dup`, `near_dup_core`, `source_backfill`, `region_text`, `b4_fanout`, `b4_project`, each with its `include=` and aggregator entries. Every later lane's Owns uses the same names (03b:200, 231-232, 245-250; 05:152-153, 310, 529; 03g:53-54). New test `test_planned_modules_included.py` |
| RR-5 PP-OCRv6 | yes | `pt:P1` runs PP-OCRv5 as the control, one PP-OCRv6 arm and one dedicated reader (00e_research_track section 3). 07c:241 makes the same run decide the B5c OCR model. 10:133 and 03g:85 agree. Leftover: the plate plan's own reader table (RN-5) |

### 7.3 The bounded close and reopen step (RR-8), on its merits

[02c](02c_migration_ops_tests.md) section 2.2 and 03d 2F-6.

- **Safety: good.**
  - One index at a time; a `try/finally` that always reopens; a bounded wait for yellow health; an
    `_analyze` check.
  - An idempotent re-run that skips indexes that already have the settings.
  - A failure-path test.
  - An added normalizer is inert until W5e gives a field to it, so a partly applied run is safe.
- **Downtime: acceptable.** It falls inside the stop-the-world window; the dry run prints a
  measured estimate per index; the plan notes that a faiss index reloads its graph on its first
  search.
- **Rollback: one contradiction (RN-3).**

### 7.4 Lanes and waves

- **W2.3a** (5D, 4D, 4A, 4B, 4C, 5A, 5B, 5C) is file-disjoint. The pattern-defined sets (4D's VLM
  call sites, 4A's routers) are listed by the coordinator at the cut, as before. It is 8 lanes, the
  size the pre-research plan carried.
- **W2.3b** (4F, ND-A and ES-B together, then ND-B, then 4E) is file-disjoint:
  - Every file a W2.3b lane takes from a W2.3a lane is edited only after W2.3a has merged:
    `autolabel/job.py` and `assist_loop/partition.py` (4F); `persons.py` and the face grouping
    (4F-9); `item_scores/near_dup.py`; the reprocess, import, combine and ingest
    `enqueue_pending` call sites (ND-A).
  - Inside W2.3b, `wire.py`, `semantic_search.py` and `review_request.py` pass from 4F to ND-B in
    sequence.
  - `env.template`, `retired_env.py` and `contracts/` follow the documented append and regenerate
    rule.
- **W2.4 and W4:** W2.4 is unchanged; ES-C's file list is recomputed on the W2.3b head (03c). W4's
  lanes run in sequence (05 section 3). The capture lane fills ND-A's `source_backfill` step list
  and a hook in 2C's settings service, both after their owners merged. No overlap found.

### 7.5 The 10-hour prototype track

- **The hours add up:**
  - PW-1 3.4 h, PW-2 2.9 h, PW-3 1.5 h, plus a 2.2 h reserve for confirmations only: 10.0 GPU hours
    (00e_research_track section 4).
  - At most three arms plus a control, early stop, and today's approach ships unless the tier-2
    confirmation wins.
- **Cut prototypes:** the program lanes take each decision of a cut prototype on today's behaviour.
  07b 5b-G4 is "not opened"; 07_b5:284 matches; 03g and the master plan use the segmenter instead of
  `pt:P6`.
- **Datasets:** every set used before the tag is in the dataset register (section 4.1). Dropping
  UFPR-ALPR, LPLC and RodoSol leaves no set that needs a signed request.
- **Leftovers:** 00b still cites two cut prototypes as feeding v0.7.0 decisions (RN-6).

### 7.6 Five mediums, spot-checked

| Finding | Holds? | Note |
|---|---|---|
| RR-6 | yes | 02a 1.2 states deferrability as the documented difference and adds the catalog-enumerated test plus the `DEFERRABLE INITIALLY DEFERRED` fallback. Its first bullet still says the project row "removes all its control data in one statement" (RN-7) |
| RR-7 | yes as text, but see RN-1 | 03b 4C-5: per table, in reverse foreign-key order read from `pg_constraint`, 10,000 rows per transaction, resumable; test with 200,000 rows under a 30 s timeout |
| RR-9 | yes | `seen_at` refreshed on the worker heartbeat and by `serving_inventory`; the reservation upserts its card; retention uses `NOT EXISTS` (02a:621, 02a:905; 03a 1B-8 test). Wording nit: RN-8 |
| RR-10 | yes | `source_backfill` with a content step (ND-A, 03b:250); the stored-bytes-equal-upload test; the upgrade report offers the job (02c, the `near_duplicates.py` row) |
| RR-14 | yes | 03d:163-164: ES-A owns every `_knn_field` call site, and 2F cuts from that head |

### 7.7 Remaining findings

#### High

**RN-1. The batched project delete (RR-7) deletes rows that the immutability triggers refuse to delete while their parent exists.**
- *Where:* [03b](03b_b2_w2_job_families.md) 4C-5; [02d](02d_research_seam_schema.md):190-191
  (`cluster_children_ready_immutable`: "UPDATE or DELETE raises while the parent partition is
  'ready' (a DELETE by cascade passes: parent gone)"), 02d:86 (`cluster_calibrations_immutable`);
  [02a](02a_postgres_schema.md):362 (`config_revisions_immutable`: "a DELETE only by cascade: the
  trigger allows it when the parent row is gone").
- *What:* 4C-5 deletes "per table in reverse foreign-key order ... the `projects` row last". In that
  order the children go before their parents:
  - `cluster_memberships` and `partition_clusters` of a `ready` partition before the partition;
  - `config_revisions` before their `config_documents`;
  - project `cluster_calibrations` before the project row.

  Each of these trigger guards raises on exactly that delete. So every project that has run a
  clustering job or holds a config document fails to delete.

  The 02a section 9 partition retention ("10,000 rows per statement") has the same conflict if it
  batches the children of a superseded `ready` partition. The RR-2 test case "deleting a project
  that holds ready partitions passes" (02d:434) will be red against 4C-5.
- *Fix:* one rule for every guarded parent and child pair, in 02a section 1.2 and 4C-5. Batched
  deletes go top-down where a guard exists:
  - Mark a partition `discarded` first (a status change, which the trigger already allows), then
    batch-delete its children. Change the children trigger to allow DELETE when the parent is not
    `ready`.
  - Delete `config_documents` one document at a time, so the revisions go by the guarded cascade
    (small per document).
  - Delete project calibrations after their referencing partitions, through a trigger exception for
    a project whose `status` is `deleting`.

  Then add 4C-5 cases: a project with a ready partition, config revisions and a calibration deletes
  in batches. Do the same for retention.

#### Medium

**RN-2. `core` scope does not reach the tables whose policy goes through `op_scope_allows`.**
- *Where:* 02a:116 (`op_scope_allows(pid)` = `op_scope_is_deployment() OR pid = op_scope_project()`),
  02a:630 (`job_attempts`, `job_items` and `gpu_reservations` use it through `jobs`), 02a:167 (the
  `op_app` grants).
- *What:*
  - In `core` scope a NULL-project job's `job_attempts` and `job_items` evaluate
    `NULL = NULL`, so a core handler running under `core_tx` (`persons_cluster`,
    `near_dup_core_group`) cannot write its attempt or items. The same holds for the core rows of
    any other `EXISTS`-policy child.
  - `op_app` is granted `EXECUTE` on `op_scope_project`, `op_scope_is_deployment` and
    `op_scope_allows`, but not on the new `op_scope_is_core`.
- *Fix:* define
  `op_scope_allows(pid) = op_scope_is_deployment() OR pid = op_scope_project() OR (pid IS NULL AND op_scope_is_core())`,
  add `op_scope_is_core` to the grant, and add an RLS test case: a core job's attempt row written
  under `core_tx`.

**RN-3. The analysis-settings step runs after `delete_sources`, so its "stops before `delete_sources`" rollback cannot happen.**
- *Where:* [02c](02c_migration_ops_tests.md) section 1 (`openprocessor upgrade` order: the
  control-data step, then "`run --rm db-migrate` again, which now finds them and migrates and stamps
  their indexes"); section 2.2 step 4 (`delete_sources` sets `upgrade_state = completed`); the RR-8
  rollback row.
- *What:* the close and reopen belongs to 2F's OpenSearch migrate (R-7), which runs in the second
  `db-migrate`, after `upgrade_from_060` has already deleted the control indexes and written
  `completed`. A failure there cannot "stop before `delete_sources`".
  - It stays fail-closed: the missing `_meta` stamp makes every process refuse with
    `opensearch_mapping_not_current`.
  - The 0.6.0 rollback still works from the bundle.
  - But the stated rollback and the `upgrade_state` gate are wrong for this step.
- *Fix:* choose one of two ways.
  - Run the analysis-settings step as a step of `upgrade_from_060` before step 4, so a failure
    leaves the sources in place.
  - Or keep it in `db-migrate`: state that a failure leaves `upgrade_state = completed` with the
    index unstamped, fail closed, fixed by re-running `db-migrate`, and correct the rollback row.

**RN-4. `location_policy = 'drop'` does not keep GPS out of exports.**
- *Where:* 05 CM-4 ("the former export GPS strip is dropped (RD-54: exports keep the embedded GPS)");
  02a:423; 00b:602.
- *What:* the owner chose to store GPS now and to keep it in exports. Under a project that sets
  `drop`, the purge clears the index values only. The stored originals and every export still carry
  the embedded GPS, so the policy name suggests more privacy than it gives. This is the location
  privacy text that `b4/privacy-hardening` writes in `SECURITY.md`.
- *Fix:* either strip GPS on export only for `drop` projects (ExifTool or a lossless EXIF rewrite in
  the export path), or state in `SECURITY.md`, the settings wire description and `docs/CURATION.md`
  that `drop` covers the index and wire only, never the stored files or exports. Pick one and test it
  (`test_location_policy.py`: an export under `drop` either has no GPS or the documented warning).

#### Low

**RN-5. Three research-plan sections still say the old thing.** The "Superseded parts" header covers
only section 8, the lock and the GPU hours:
- the auto-label plan sections 5.2 and 5.3 (lines 473 and 507) still revoke on a queued pause, and
  02b section 3.3 links to section 5.3 "for the loop";
- the spatial plan section 5.1 (line 272) still defaults `location_policy` to `drop`;
- the plate plan's reader table (section 4.1) has no PP-OCRv6 row.

*Fix:* correct the three lines, or widen the header to name them.

**RN-6. 00b still cites cut prototypes as feeding v0.7.0 decisions.**
- 00b:196 (#403 MPS "adopt only on the `gpu:PR-4` evidence")
- 00b:198 and 00b:552 (#407 "reads ... `PR-4` (the C++ bar) first")
- 00b:551 (#40 "reads `gpu:PR-1..PR-4`")
- 00b:556 (#403 "W5b conditional on `gpu:PR-4`")

*Fix:* use 07b's wording: MPS off, 5b-G4 not opened, the post-tag prototypes in RW-1.

**RN-7. 02a section 1.2's first bullet** still says deleting the project row "removes all its
control data in one statement". It should say that the cascade is the safety net behind 4C-5's
batched delete.

**RN-8. 02a:621 says GPU workers upsert `gpu_devices` "under `deployment_tx`"**, but `deployment_tx`
is allow-listed to the reaper, beat, registry, upgrade and CLIs. `gpu_devices` has no RLS, so the
upsert needs no scope. Say that, or add the GPU runtime module to the allow-list.

### 7.8 Checks

- `.venv/bin/python scripts/codegen/check_naming_leaks.py`: pass.
- `.venv/bin/python scripts/docs/check_docs_vs_code.py`: pass.

### 7.9 Planner responses to the re-check (2026-11-04)

| Finding | Verdict | Fix (file, lane, test) |
|---|---|---|
| RN-1 batched delete vs immutability triggers | fixed, settled before lane 1F writes the triggers | one guarded-delete rule in [02a](02a_postgres_schema.md) section 1.2: every immutability trigger allows a DELETE by cascade or when `op_project_delete_allows(project_id)` is true, which needs the transaction-local setting `op.project_delete` naming the project **and** the project row in `deleting`; one role (`op_app`) runs every process, so the status, not a role, is the guard, and a ratchet test allows the setting's literal only in `src/services/projects/delete.py`. Retention sets a superseded partition `discarded` before batch-deleting its children. 02d trigger comments and tests; 03a 1A-13, 1F-1; 03b 4C-5. Red test `tests/projects/test_project_delete_guarded.py` (a project with a `ready` partition, a config document with two revisions and a calibration deletes; the same child deletes without the setting, for another project, or for a project not `deleting` raise) |
| RN-2 core scope and `op_scope_allows` | fixed | `op_scope_allows(pid)` gains `OR (pid IS NULL AND op_scope_is_core())`; `op_app` gets `EXECUTE` on `op_scope_is_core` and `op_project_delete_allows`: 02a sections 1.4, 1.5; 03a 1A-13. Test case in `tests/db/test_row_level_security.py`: a core job writes its `job_attempts` and `job_items` rows under `core_tx` |
| RN-3 analysis step after `delete_sources` | fixed (the first option) | the bounded close, settings and reopen is step 3a of `upgrade_from_060`, after verify and before `delete_sources`, recorded in `upgrade_runs`; step 4 refuses until it completed, so "stop before `delete_sources`" is real; the later `db-migrate` OpenSearch migrate only skips indexes that have the settings: [02c](02c_migration_ops_tests.md) section 2.2; [03d](03d_b2_s2_control_data.md) 2F-6 |
| RN-4 `drop` does not keep GPS out of exports | fixed, no new question | exports keep the embedded GPS by default (RD-54); a `drop` project's exports strip the GPS IFD losslessly with ExifTool in the jobs image; the docs say `drop` covers index, wire and exports, not the stored originals: [05](05_b4_stateless_api.md) CM-5; [06](06_release_acceptance.md) RF-84; [08](08_environment_and_setup.md); [10](10_standards_and_libraries.md); 00b, 00d, 00_index. Test `tests/curation/test_export_gps_by_policy.py` |
| RN-5 three research-plan sections | fixed | the auto-label plan's queued pause and cancel rows (no revoke on pause; revoke only on the terminal cancel); the spatial plan's `location_policy` default `store`; the plate plan's reader table gains R9, PP-OCRv6 |
| RN-6 cut prototypes cited as feeds | fixed | 00b #40, #403, #407 rows (sections 2 and 5.3) use 07b's wording: MPS off, 5b-G4 not opened, `PR-3` and `PR-4` post-tag in RW-1 |
| RN-7 02a first cascade bullet | fixed | the cascade is the safety net behind 4C-5's batched delete: 02a section 1.2 |
| RN-8 `gpu_devices` under `deployment_tx` | fixed | `gpu_devices` has no RLS, so its upsert needs no scope; the worker start writes it under `deployment_tx` from `src/tasks/runtime/gpu_app.py`, now on the `deployment_tx` allow-list (it touches only this table and `worker_capabilities`): 02a section 5 |
