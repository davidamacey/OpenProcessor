# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- **W10 finish and combine review fixes** (review `w10_p4_review_2026-10-01`).
  - **Delete-time lock re-check.** `item_delete.delete_items` is the one delete
    path: it re-reads each document, asks the caller whether it is still
    deletable, and deletes with `if_seq_no`/`if_primary_term` (retrying on a
    version conflict), reporting `skipped` ids. Import undo (items and the
    images it created), reprocess `detect` and reconcile use it, so a human edit
    made after the decision is no longer destroyed. Undo applies its update
    path to a skipped item.
  - **YAML alias bomb.** `data.yaml` is parsed by `safe_yaml.load_bounded_yaml`,
    which rejects every alias and caps nodes (50,000) and depth (32); class
    names and split entries must be scalars. A hostile file is a
    `data_yaml_invalid` issue. Only the dataset-import reader changed; the
    operator scripts under `scripts/curation` still use `yaml.safe_load`.
  - **Stale imports.** `FileJob.repair_if_stale` lazily rewrites a job whose
    heartbeat aged out to `interrupted`; imports (on open and list), the resume
    check, `job_wire`, reprocess jobs and combine jobs use it, so a stale
    `running` import is resumable and undoable.
  - **Undo vs import.** Undo claims the import under the project start lock and
    returns 409 `import_busy` while another import is live (dry runs only check
    that the import is undoable).
  - **Combine resume** takes a per-job claim (a second concurrent resume is
    409), runs the same `resolve_sources` gate as preview and start, and
    refuses with 409 `preview_stale` when a source changed since the preview.
    The duplicate-hash scan and the file copy run off the event loop.
  - **Combine duplicates.** A duplicate image's region boxes are attached to the
    target copy instead of dropped; a box the copy already holds is not added
    twice, and a validated region set is never rewritten (the box becomes a
    standalone region item). The same rule now applies to a first copy.
  - **Archive upload.** Tar and zip directory members count against the member
    cap and nesting is capped at 64; the sweep removes `.incoming` leftovers
    idle for an hour; a re-upload restarts the TTL; two uploads of one archive
    no longer remove each other's extraction; `__MACOSX` no longer hides a
    single root folder. The COCO annotation cap is 128 MiB (was 1 GiB).
  - **Smaller fixes.** Combine resolves a stored image path before judging it
    against the upload root (a `..` could hard-link outside it). Undo clears
    only a holdout flag the import set (the ledger row records `froze_test`).
    A reprocess of `detect` over more than `OP_REPROCESS_SYNC_MAX` images
    together with `region` or `vlm` is a 422 instead of reordering them.
    `is_human_marker` is public; the region requeue uses `region_locked_clause`
    and `region_set_locked`.
  - **Project dir marker.** Import, upload and reprocess-job dirs carry a
    `.project` marker; a dir marked as another project's raises
    `ProjectDirMismatchError`.
  - **Tests** now fail when removed: undo of a human verdict on an untouched
    import box, combine embedding-dimension mismatch (`embeddings_dropped`) and
    unclassed items.
  - Not changed: `images_already_indexed`, the `near_duplicate_pairs_estimate`
    stub, sequential detector calls in `propose`, the read-time export-root
    check in `image_load.stored_path`, and pruning of combine job dirs.
- **W8-cleanup items 3-6 review fixes.**
  - **Breaking (response keys):** every region clustering count names its
    unit. `POST /regions/cluster` job result: `n_regions` -> `n_boxes`,
    `assigned` -> `n_boxes_changed` (boxes whose cluster changed; `0` on a
    re-run over unchanged data) plus `n_items_written`. Refine
    (`POST /regions/clusters/refine/{id}` -> `n_boxes`, `n_boxes_updated`;
    `POST /clusters/refine/{id}` -> `n_items`, `n_items_updated`; was
    `n_members` / `n_updated`, which counted boxes and items respectively
    under the same names). FP centroid build: `n_members` -> `n_boxes` (also in
    the persisted centroid metadata and `GET /regions/fp_centroids/status`);
    auto FP pull: `n_scanned` / `n_moved` -> `n_boxes_scanned` /
    `n_boxes_moved`. `write_box_edits` reports `items_*` / `boxes_changed`, and
    its `unchanged` tally no longer double counts conflict retries.
  - `GET /regions*` pages carry `rows_truncated` (an item matched more boxes
    than `index.max_inner_result_window`, so some of its rows are missing).
  - Cluster-only box writes (partition, refine, FP sub-typing) no longer
    advance `region_revision`, so a recluster no longer 409s an open editor;
    any write that changes a box's state, geometry or text still does.
  - `GET /crops/{id}/region_thumbnail` is `no-cache` with an `ETag` (a moved
    box is no longer shown from a one-hour public cache; an unmoved one is a
    304).
  - A human move or delete of a box (`PUT .../regions`, `PUT
    /crops/batch_regions`) and `requeue --clear-detection` prune that box's
    `region_box_embeddings` entry immediately; a stale (moved-box) entry is
    now dropped by every embedding write too.
  - A whole-set confirm that no box can satisfy answers an actionable 422
    (`no_accepted_box: ... accept ... one by one with PATCH
    /crops/{crop_id}/regions/{box_id}`) instead of the bare code.
  - Live harness: seeded `region_box_embeddings` entries carry `bbox_norm`
    (built with the production `entry_for`), so the live clustering and FP
    scenarios no longer see every vector as stale; the live region tests send
    `region_label_source` (the request models forbid `label_source`) and
    un-mark a false-positive item per box.
  - Docs: dropped the deleted `OP_REGION_FIELD_BBOX_NORM` override and the
    removed `region_text` meta field.
- **W8-cleanup Items 4-6 fixes** (also the open minors from the Items 1-2
  review):
  - `GET /regions`: every box filter (`detector`, `min_score`,
    `max_score`, `text`, `region_cluster_id`, `region_cluster_subid`,
    `box_state`) now applies to the SAME box in one nested clause
    (`m2`; before, each was its own nested query, so `detector=a` AND
    `min_score=0.8` matched an item whose a-box scored 0.5 and whose
    b-box scored 0.9).
  - `region_eval` scores a rejected box whose reason is `sanity_reject:*`
    as `detection_failed`, not `verify_rejected` (`m3`).
  - A refine, FP sub-typing or auto FP pull can no longer overwrite a box
    a human moved, re-stated or locked while the model was fitting: every
    cluster write is an OCC merge over the live box list, re-merged on a
    version conflict (up to 3 attempts) instead of dropped.
  - `GET /crops/{id}/region_thumbnail` requires `?box_id=`
    (`422 box_id_required`, `404 unknown_box_id`); it no longer serves "the"
    item region.
  - `/vlm/verify_regions` verifies every open (proposed/accepted, not
    human-owned) box and writes each verdict onto its own box (it used to
    read one item-level box); a verdict is dropped when its box moved or was
    locked while the VLM call was in flight.
  - `occ_skip_on_conflict_bulk` returns `skipped_ids` next to
    `skipped_due_to_conflict`.
  - `ensure_items_inner_result_window` reads `GET /{index}/_settings` (the
    project guard allowlists that shape only).
- **W8-cleanup Item 3: bugs the PUT-region test migration surfaced in the
  box routes** (`region_box_edits.py`, new; shared by every human box
  writer): a parent-frame box on `PUT /crops/{id}/regions` was projected
  through the item's region box instead of the item crop's own
  `bbox_norm`; a moved box kept the machine's detector/score instead of
  becoming human geometry (and an unmoved one, within float noise, now
  keeps its stored coordinates and provenance); a human "no region
  visible" (`boxes: []`) wrote the pipeline status `no_region_box`
  instead of `no_region_visible`; bbox range/degenerate validation, a
  required `bbox_norm` on a new box and duplicate `box_id` are now 422s;
  a box moved into / out of `false_positive` is parked in / released from
  the FP cluster per box on every writer, and any state change clears a
  stale per-box `rejection_reason` (review minor m5); `POST
  /crops/{id}/region/undo` now bumps `region_revision` (a stale
  `expected_region_revision` across an undo is a 409); the request bodies
  are `extra='forbid'` and carry `region_label_source`.
- **W9 review fixes (VLM model selection).**
  - A probe is stored per `name@revision`: probing a new revision no longer
    erases the running revision's probe (which flipped its JSON mode and
    provenance stamps), and rolling back to the previous revision of the same
    endpoint works. The worker's rebuild marker reads the same key.
  - `POST /vlm/endpoints/validate?probe=true` and `.../{name}/probe` refuse to
    contact (or resolve a key for) an endpoint outside the deployment that has
    no `allow_external`, one denied by `OP_VLM_EXTERNAL_POLICY`, or a body
    with an out-of-range field (a huge `max_images_per_call` built the
    request on the event loop); the probe also clamps the cap itself. A draft
    named `env` with the `env:` key reference is a validation error, not a
    500, and a draft's probe is recorded only for the revision it equals.
  - A credential in `OP_VLM_URL` is dropped when the `env` endpoint is built,
    so no route, log line or labeler sees it (use `OP_VLM_API_KEY`).
  - Profile validation asks the endpoint registry whether a VLM is
    configured, not `OP_VLM_URL`.
  - Every labeler re-checks its endpoint (never-allowed addresses,
    `OP_VLM_EXTERNAL_POLICY`, `allow_external`) when built and before sending,
    at most every 30 s, and fails closed; this covers the worker's long-lived
    labeler and cached ones. The connection itself is still not pinned to the
    checked address (`SECURITY.md`).
  - URL policy: 6to4 (`2002:...`) addresses are judged by the IPv4 address
    they carry, and host names are NFKC-folded (full-width names).
  - Tests: `tests/test_compose_contract.py` is restored to its 41 contract
    tests plus the vlm argv tests (the VLM change had replaced the file);
    new tests cover the per-run acknowledgement and the clone refusal, which
    no test failed without.
  - Docs: "Choosing a VLM" in `docs/CURATION.md`, the VLM routes in
    `docs/design/curation_api_contract.md`, the catalog in `ATTRIBUTION.md`.
- **W10 dataset-import foundation fix pass (post-review).** An
  independent review of the W10 foundation slice below found 4 majors;
  all fixed before any route is wired to `import_dataset()`:
  - **Sparse YOLO `names` dict crossed class indexes** (`yolo.py`): a
    `data.yaml` `names` dict with a gap (e.g. `{0: car, 2: truck}`, a
    class pruned from training) built a dense list by sorted-position, so
    index `1` silently resolved to `truck` instead of being rejected —
    the exact index-crossing bug this wave exists to close. Now parsed
    into `dict[int, str]` and looked up by key; a missing index is a
    clean `label_class_out_of_range` (new info issue
    `data_yaml_names_sparse` flags the gap).
  - **Re-import over an existing item could overwrite a locked (human or
    validated-import) class, keeping `class_source: human`** (`job.py`):
    `import_dataset()` now mgets existing docs first and drops any
    `is_locked_item()` match from the write set, reported as a conflict
    (`DatasetImportReport.conflicts`/`items_locked_skipped`), never
    applied. `import_ids` is appended, not replaced. The region-box
    write path had the same hole (`_region_merger` unconditionally
    overwrote the box list) and is fixed the same way
    (`regions_locked_skipped`).
  - **The class-identity E2E test could not detect index-crossing**
    (`tests/integration/test_class_identity_e2e.py`): every hop compared
    values derived from the same map, so a deliberate car/truck name
    swap still passed. Rewritten to tie each box's specific geometry to
    its specific class name at every hop (import, export through the
    real `GenericYoloExportService`, promote, predict) — verified to
    fail against the reversed-names reproduction.
  - **Lock-rule call sites**: `exclusion.py`'s legacy un-exclude branch
    now checks `_is_human_marker` instead of the broader `is_locked_class`
    (a `test_holdout` item with a machine class was incorrectly restored
    as validated); `_merge_preserving_human` (`occ.py`) now gates the
    `class_source`/`label_source` guards on `is_locked_class`, not a bare
    per-value marker check, so an unvalidated ("suggestion") import is no
    longer incorrectly locked on re-ingest — correcting the "Re-ingest ...
    now respects it" claim below, which previously only preserved
    provenance strings, not the class value; `revert_class_cluster_
    promotions.py` now reports frozen-`test_holdout` skips as their own
    counter instead of silently folding them in.
  - **`scripts/curation/import_labeled_dataset.py`'s default (labeled)
    mode was broken two ways** (posts forbidden fields to
    `/ingest/batch`, and to the deleted `/import_labels/batch`) with its
    own test deleted and no replacement. It now fails loudly and
    immediately when invoked without `--images-only` instead of 422ing
    deep in a request; `--images-only` is unaffected. Fixed every stale
    doc pointing at the removed `/import_labels(/batch)` routes.
  - Narrowed the `auto_promote.py` `class_validated: True` AST-gate
    allowlist (`test_class_label_single_writer.py`) to the specific
    `_merge_promote` function, not the whole file.
- **W10 dataset-import R2 fix pass (second confirmation round).** A
  follow-up review of the fix pass above found the region-box fix
  introduced one new major (R2-M1) plus minors; all fixed:
  - **R2-M1: a first validated import wrote zero region boxes**
    (`job.py`): `_region_merger`'s lock check read `is_locked_class` off
    the parent item's *current* OpenSearch state — which, on a first
    import, is the `class_source: external_label` /
    `class_validated: True` this same import just wrote to the parent
    moments earlier via `class_label_fields`. Under the default
    `label_trust='validated'`, every region box was therefore dropped
    and reported as a false "region boxes locked" conflict against the
    importer's own write (`boxes_written=0` on a fresh dataset).
    `_region_merger` now decides the lock from pre-import state only:
    whether the parent item was already in `_split_locked_items`'s
    `locked_ids` (locked by something other than this import), plus any
    already-locked existing region boxes — never `is_locked_class` on
    the current doc. New regression test
    `test_first_validated_import_writes_region_boxes`
    (`test_job_import.py`) seeds fresh items with no prior human/
    validated state and asserts `boxes_written` is nonzero.
  - **Fixed the wrong-items-index test bug that let R2-M1 slip through**
    (`test_job_import.py`): tests passed a hardcoded
    `items_index='op_curation_items'` string instead of the bound
    project's real `get_curation_config().items_index` — `FakeOpenSearch`
    routes bulk/update writes by comparing against the real config value,
    so a mismatched literal silently misrouted writes. All three existing
    tests plus the new one now use `get_curation_config().images_index` /
    `.items_index`.
  - `import_labeled_dataset.py`'s labeled-import mode: the dead
    `_relabel` method, `label_txt_path` posting branch, disagreement-
    report generation, and their CLI flags (`--relabel-duplicates`,
    `--label-source`, `--no-detect-mismatches`, `--no-verify-labels`,
    `--skip-class-check`) are deleted, not just gated off — the guard
    that fails loudly for non-`--images-only` invocations is the only
    labeled-mode-related code left. The module docstring and `--help`
    text now describe `--images-only` as the only working path instead
    of still documenting the disabled labeled mode as if it worked.
  - `main()` now skips `bind_script_project()` (which contacts
    OpenSearch to resolve the project) when `--images-only` was not
    passed, so the disabled-mode guard fails immediately/cheaply instead
    of after an OpenSearch round-trip.
- **W8-cleanup Items 1-2 confirmation-review fix pass (round 3).** A
  third independent confirmation review of the round-2 N1/N2/N3 fixes
  found one residual: `region_writes.reason_only_box_write`'s N3 fix
  restored the item-level `region_rejection_reason` whenever the item
  had no *rejected* box, instead of when it had no box at all — so a
  reason-only PATCH on an accepted-only `detected` item, an FP-only
  item, or a proposed-only item incorrectly stored the reason, bringing
  back N1's exact symptom. The condition is now `not new_boxes`, the
  same box-less guard `human_status_box_write` uses for M1(a). Also
  updated `human_status_box_write`'s docstring, which still described
  the pre-N1/N2 mirror rules.
- **W8-cleanup Items 1-2 confirmation-review fix pass (round 2).** An
  independent confirmation review of the round-1 fix pass found the M2
  mirror redesign (moving the legacy per-item mirror into
  `region_boxes.boxes_write_fields`) introduced three new regressions;
  all fixed:
  - **N1: a `detected` item with one accepted box and a rejected sibling
    stored a `region_rejection_reason`**, so the labeler rendered a red
    "Rejection" row instead of "needs human confirmation" on ordinary
    multi-box items. `rejection_reason` is now only mirrored when there
    is no accepted-or-false_positive representative on the item.
  - **N2: a higher-scoring false-positive box could outrank an accepted
    box for the mirror.** `_mirror_representative` now always prefers
    the best accepted box, falling back to the best false-positive box
    only when no accepted box exists.
  - **N3: a reason-only PATCH on a box-less item (`no_region_visible`)
    wiped the reason M1(a) had just stored**, because
    `boxes_write_fields([])` always re-derives `rejection_reason=None`
    from an empty box list. Added `region_writes.reason_only_box_write`,
    which restores the item-level reason when there's no rejected box to
    carry it; `regions_edit.py`'s reason-only PATCH branch now routes
    through it instead of writing `F.rejection_reason` directly.
  - **Minor:** documented, at the `no_accepted_box` raise in
    `boxes_with_status`, that a whole-set human reject followed by
    CONFIRM now 422s (pre-W8: 200) — intentional, matching M3's "never
    reopen a human-rejected box" rule; a per-box and whole-set human
    reject share the same reason and can't be told apart.
- **W8-cleanup Items 1-2 review fix pass.** An independent review of the
  items-1-2 port (see `docs/design/openprocessor_internal/
  w8_cleanup_items1_2_review_2026-09-28.md`) found the default regions-tab
  sort silently broken on any real W8-written index plus five majors; all
  fixed:
  - **Blocker: `review_sorts.py`'s `region_score` sort never matched what
    the worker writes.** Nothing in the W8 worker writes `region_score`/
    `region_candidate_score` (only dead code and human single-box paths
    do), so every W8 item tied on `missing: '_last'` and fell back to the
    `crop_id` tiebreak — a 0.95-score item could sort dead last behind a
    0.2-score legacy row. Now sorts on the flat `region_max_score` (every
    box writer maintains it via `boxes_write_fields`, covering a
    rejected-only item's own score too, so no second sort key is needed).
    Removed the `region_score`/`region_candidate_score` fixture
    dual-writes in `test_review_locate.py` / `test_review_regions_rejected_router.py`
    that hid this in production-shaped-looking tests.
  - **Major: a `no_region_visible` reject with a reason dropped the
    reason entirely**, and a reason-only PATCH updated the box but not
    the item-level mirror the labeler reads as authoritative. Both fixed
    in `human_status_box_write`.
  - **Major: the legacy per-item mirror (`bbox_norm`/`score`/`detector`/
    ...) went stale in three ways** — a `verify_rejected` write kept a
    prior confirm's box/score, every writer other than
    `human_status_box_write` (per-box PATCH, `PUT .../regions`,
    `POST regions/batch_box_state`, requeue, the worker) never touched
    it, and it mirrored the first accepted box instead of the
    best-scoring one. Fixed by moving the mirror computation into
    `region_boxes.boxes_write_fields` itself (every box writer routes
    through it), always deriving from the highest-scoring
    accepted-or-false_positive box (never a rejected one — `bbox_norm` is
    an accepted region to every reader) with `rejection_reason` mirrored
    from the highest-scoring rejected box independently.
  - **Major: a whole-set CONFIRM reopened every rejected box**, including
    ones a human rejected per-box or the sanity gate rejected — overriding
    a human's own earlier verdict and turning degenerate sanity-rejected
    geometry into an accepted training box. Now only reopens a box the
    *verifier* rejected (`region_visible_elsewhere` / `verifier_no_verdict`);
    a `detection_failed` item whose only box is sanity-rejected 422s
    again, matching pre-W8.
  - **Major: the `low_conf_correct` training cohort admitted a primary
    box the verifier rejected**, as long as some other accepted box
    existed on the item. Added the same same-box `state == 'accepted'`
    nested filter `detector_blind_spots` already used.
  - **Major: the region-embeddings backfill stopped selecting
    false-positive items**, starving the FP centroid store
    (`build_region_fp_centroids`) of its inputs — the classic
    VLM-rejected/human-marked-FP hard negative was never embedded.
    Selection now matches `state in [accepted, false_positive]` (like
    every other ported W8 reader); the representative crop falls back to
    the highest-scoring FP box when there's no accepted one.
  - **Minor: `/stats/dataset`'s `region_detectors` agg counted boxes, not
    crops** — an item with 2 accepted boxes from the same detector
    counted twice. Added `reverse_nested` so a crop counts once
    regardless of how many of its boxes match.
  - **Guard test gaps:** `tests/test_no_legacy_region_scalars.py` was a
    no-op for `export_single_class_rows.py`'s `f = self.fields` local
    alias, and missed `get_region_fields().<attr>` inline calls and bare
    wire-name string literals. Strengthened to catch all three (still
    excluding a legitimate wire-name collision with a Pydantic model
    field name in `regions_edit.py`).
- **W8-cleanup Items 1-2: seven production files silently read retired
  item-level region scalars instead of `region_boxes`.** The current
  box-list worker only ever writes per-item `region_bbox_norm`/
  `region_score`/`region_detector`/`region_text*`/`region_detected_at`/
  `region_candidate_*` on the pre-W8 `PUT /crops/{id}/region` write chain
  (unchanged this pass); every other reader of those fields was silently
  scoring/filtering/aggregating against permanently-empty data for any
  item processed under the current pipeline. Ported off them, onto
  `region_boxes` nested queries/reads:
  - `scripts/curation/backfill_region_embeddings.py` (selection + crop
    bbox now from an accepted box; picks the highest-score one as the
    interim single-embedding representative — per-box embeddings are
    Item 5, not done this pass).
  - `region_eval.py` (`region_record` → `region_records`, one record per
    box; a box-less item falls back to the item-level `region_status`,
    which the worker still writes). Removed
    `test_unknown_frame_is_refused` / `test_cli_unknown_frame_exits_3`:
    `region_boxes` entries are always source-frame (every writer
    projects before appending), so the frame-refusal scenario they
    exercised is now structurally impossible.
  - `review_queries.py`'s `regions` review tab (has-a-box / rejected-
    candidate / text-search clauses; `region_reason()`'s rejection
    reason now reads the item's rejected box).
  - `regions_fp.py`'s `GET /regions/clusters` cluster-membership filter.
  - `export_single_class_rows.py`'s region-mode row collector (every
    accepted box on an item now becomes its own label line/row entry,
    not just the first — a natural side effect of reading the list).
  - `stats.py`'s `/stats/dataset`: `region_detectors` (now a nested agg),
    `region_boxed`, and `regions_validated_by_human`'s "drew a box"
    clause.
  - `regions.py`'s `GET /regions` browse + `GET
    /regions/training_candidates` (default filter, min/max score,
    detector, text search, all five training-cohort modes, and the
    `region_detected_at` sort — now a nested sort with `mode='max'`).
    `cluster_id`/`cluster_subid`/`cluster_distance` deliberately left
    item-level and unchanged: still actively written by the region-FP
    clustering job (Item 5's per-box clustering wasn't attempted this
    pass, so there is nothing stale to port here).
  - `PATCH /crops/{id}/region_meta` / `POST /regions/batch_status`:
    ported onto `region_boxes.boxes_with_status` (new
    `region_writes.human_status_box_write`) rather than deleted — both
    keep a legitimate whole-item purpose. Reopens a verifier-rejected
    candidate box on a human CONFIRM (the pre-W8 candidate-promotion
    reversal `boxes_with_status` doesn't do on its own) and keeps the
    legacy per-item mirror fields (`bbox_norm`/`score`/`detector`/...)
    `wire.py` still serves additively in sync with the box list instead
    of letting them go stale.
  New `tests/test_no_legacy_region_scalars.py`: a scoped ratchet guard
  (see its module docstring for exactly what it does and does NOT cover)
  against these eight files regressing back to the scalars just removed
  from each.

  (The deletions this entry deferred, and the legacy mirror fields, landed in
  the Items 3-6 entries below.)

### Added
- **W8-cleanup Items 5-6: per-box embeddings, clustering and FP, and row
  shapes** (`docs/design/openprocessor_internal/any_domain_plan.md`
  W8.10 / W8.12):
  - `region_box_embeddings` (nested sibling of `region_boxes`):
    `[{box_id, bbox_norm, embedding}]`, one PE vector per accepted or
    `false_positive` box; `bbox_norm` is the geometry the vector was
    computed from, so a box a human moved is recognised as stale and
    re-embedded. Written only by the worker embed stage (after the bulk
    write, keyed to the final box ids) and `scripts/curation/
    backfill_region_embeddings.py` (every missing box; resumable), through
    `write_box_embeddings` (OCC; never touches the box list or revision).
  - Region clustering is over boxes (`clustering/region_box_rows.py`,
    `region_box_clustering.py`, `region_cluster_jobs.py`):
    `cluster_id` / `cluster_subid` / `cluster_distance` live on each box;
    the KMeans partition reads accepted boxes, `build_region_fp_centroids`
    is fed by false-positive boxes only, `auto_assign_fp_from_centroids`
    flips matching boxes to `false_positive` and re-derives the item
    status (a sibling accepted box keeps the item `detected`). The AHC
    core is shared with item refine (`refine_members`). The guard on
    automated writers is one definition (`human_final_clauses`): a
    human-final item (human label source / verifier, or validated) or a
    locked box is never touched. Deviation from the plan: these writes are
    Python OCC mergers over `boxes_write_fields`, not painless scripts, so
    one `derive_status` stays the only status implementation.
  - `ensure_items_region_boxes_fields` (one `_region_boxes_mapping()` behind
    both the index body and the ensure step) and
    `ensure_items_inner_result_window` (raises
    `index.max_inner_result_window` to `limits.max_boxes_per_write`, never
    lowers it).
  - Row shapes (`region_rows.py`, `RegionRowPage`): `GET /regions`,
    `/regions/training_candidates`, `/regions/suspected_false_positives`,
    the representatives of `GET /regions/clusters` and the items of the
    batch routes are rows — the full wire item plus `region_box_id` (`null`
    for an item-level row). A box-selecting request is one row per matching
    box (matched boxes are asked of OpenSearch via nested `inner_hits`);
    `total` counts items, new `total_rows` counts rows. `GET /regions`
    gains the `box_state` filter; cluster cards gain `size` (items),
    `box_count` (boxes), `representatives` and `representative_box_ids`;
    `POST /regions/batch_box_state` returns one row per targeted box.
    The training modes `detector_blind_spots`, `low_conf_correct` and
    `false_positives` are per box; `disagreement` and `human_corrected`
    per item. The suspected-FP route scores and pages boxes.
  - The review `regions` tab gains the `has_rejected_box` option
    (`region_rejected_count >= 1`, whatever the item status); `verify_rejected`
    is relabelled "Items with only rejected boxes".
  - Export: the region stratum key is the first accepted (positive) or
    `false_positive` (hard negative) box's cluster.

- **W9 VLM model selection: a registry of VLM endpoints, a per-project `vlm`
  axis, a local model catalog, and one place every model choice is served.**
  See `docs/design/openprocessor_internal/any_domain_plan.md` W9.
  - **Endpoint registry** (deployment-wide, stored in `op_global_configs`,
    never per project): `GET/POST /curation/vlm/endpoints`,
    `GET /vlm/endpoints/schema`, `POST /vlm/endpoints/validate`,
    `GET/PUT/DELETE /vlm/endpoints/{name}`, `.../revisions[/{revision}]`,
    `POST .../{name}/clone`, `POST .../{name}/probe`. Every save is a new
    immutable revision (numbers are never reused); a probe (synthetic
    images only) records the served model root, context length, image
    cost, image cap and JSON-mode support, and belongs to the revision and
    body it tested. The `env` built-in (`OP_VLM_URL`/`OP_VLM_MODEL`) is listed
    first, read-only.
  - **Activation per project** (`/curation/projects/{project}/vlm/endpoints/
    active`, `.../{name}/activate`, `.../active/rollback`, `.../deactivate`),
    `PUT /settings` `defaults.vlm`, and a per-run `?vlm=`
    (+ `acknowledge_external`) on `/vlm/label_batch`, `/vlm/verify_regions`,
    `/vlm/verify_region_batch`, `/vlm/region_visible_batch`,
    `/vlm/label_cluster/{cluster_id}`, `/pipeline/auto_label` and
    `/pipeline/auto_label/start`. All of them go through ONE gate
    (`enforce_vlm_gate`); `tests/curation/test_vlm_selection_paths.py`
    walks each with an input that must be refused. `/start` pins the
    resolved `(name, revision)` into the job.
  - **Hot switch**: the detection worker follows a project's VLM at its
    quiesce points (queues drained) from one pinned registry store shared by
    every project; an activation, a re-probe or a rollback swaps the
    labeler, a new revision of the active endpoint changes nothing until it
    is activated.
  - **Safety**: URL policy (`vlm_url_policy`) refuses this stack's own
    services, link-local / metadata / unspecified addresses in any notation,
    for the literal host and every resolved address; keys are references
    only (`secret:<slug>` files under `./secrets/vlm`, written by
    `openprocessor vlm key set <slug>`), never stored, served or logged;
    no client to an endpoint follows a redirect; an endpoint outside this
    deployment needs an acknowledgement (`OP_VLM_EXTERNAL_POLICY=deny`
    refuses them outright) that is recorded per `name@revision`.
  - **Pairing** (`vlm_pairing_issues`): context-size estimate per labeler
    call, the server's own image cap, multi-box and text-reading
    verification, JSON mode; it runs on VLM activation, pack activation,
    profile activation, per-run selection, and a combined `PUT /settings`
    pairs with what the request will change (not what it replaces).
  - **Local model catalog** `examples/vlm/catalog.tsv`, read by the
    installer/CLI's bash and by `src/services/labeling/vlm_catalog.py`;
    `GET /curation/vlm/catalog`, `GET /vlm/local`,
    `POST /vlm/local/select` (records the DESIRED model; the API never
    restarts vLLM), `DELETE /vlm/local/select`; and the host CLI
    `openprocessor vlm status|use <id> [--force] [--yes]|apply|probe`
    (fit check, training-lock and pause handling, `.env` rewrite with
    restore on failure, wait for the new model, probe, unpause).
  - `GET /config/vocabulary` serves a `model_choices` table
    (`src/services/curation/model_choices.py`) and its `vlm` block comes
    from the registry; `GET /methods` gains a `vlm` axis with each
    endpoint's status, locality and acknowledgement flags;
    `GET /regions/vocabulary` offers every endpoint's resolved model as a
    verifier.
  - Items gain `vlm_endpoint` and `vlm_model` (which endpoint and resolved
    model answered), stamped per task at call time; `vlm_prompt_pack` is
    now on the item wire.
  - `clone_settings` gains the `vlm_activation` axis (an acknowledgement is
    never copied); global SSE event `vlm.changed`.
  - Compose: the `vlm` service is parameterised by `VLM_*` (defaults
    reproduce the previous command argument for argument);
    `yolo-api`, the detection worker and the auto-label worker mount
    `./secrets/vlm` read-only; `SECURITY.md` records the SSRF residual risk.
- **W10.11 reconcile.** Importing a different dataset version over images an
  earlier import labeled removes the earlier import's items the new version no
  longer has, unless a human or a holdout freeze touched them, a box the
  dataset still has (even one mapped to `skip`) matches them, or the frame has
  no label file. The whole document is kept in the new import's ledger row and
  undoing that import reinstates it. Region boxes written by an earlier import
  are not reconciled. Wire: `items_reconciled_removed` on the import report and
  `items_reinstated` on the undo report (`contracts/openapi/curation.json`).

- **P4 combine projects: `POST /projects/combine` builds a new project from
  1 to 8 existing ones.** `/projects/combine/preview` (writes nothing; returns
  errors, warnings, `suggested_mapping`, counts, duplicate and conflict
  numbers, bytes to link and a `preview_sha`), `/projects/combine` (start,
  `expected_preview_sha`, `202 {job_id, target}`), `/projects/combine/{job_id}`
  and `/cancel`, `/resume`; progress is also published as `combine.progress`.
  The sources are only read (bound read-only); the target is `building` while
  it fills, then `active` (`failed` on an error; deleting it is a complete
  undo). `src/services/projects/combine/` plus
  `dataset_import/project_source.py`, W10's project reader (a source project
  as a `ScanEntry` stream).
  - Class identity: each source class maps to a target class BY NAME with
    W10's `map` / `create` / `skip` / `region` rows and completeness rule
    (`unmapped_class`, `mapping_target_invalid`); the target owns its ids and
    nothing numbered in a source (class ids, class-id history, clusters)
    crosses. `tests/integration/test_class_identity_combine.py` extends the
    identity E2E through combine, export, remap, promote and predict.
  - Dedup by content: byte-identical images (imohash candidate, full sha256
    confirmed) are copied once from the first-listed source; boxes of the
    other copy merge by IoU and target class (human > import > VLM > model;
    ties keep the priority source), and a box with a different class keeps the
    priority label and is flagged `combine_conflict` (new
    `GET /review/*?combine_conflict=true` filter; the `all` tab includes them).
  - Provenance: `import_ids` = the job, `origin_project` / `origin_item_id` /
    `origin_image_id`, `label_source` preserved; upload files are hard-linked
    into the target. Frozen test splits are kept by union
    (`holdout: preserve_union`, with a freeze record) or recomputed (warns).
  - Jobs are file-backed under `OP_COMBINE_JOBS_DIR`, chunked
    (`OP_COMBINE_PAGE_SIZE`), reconciled to `interrupted` on startup and
    resumable from the persisted plan; a live combine marks its source and
    target projects busy.
  - `create_project` gains `origin` and `activate=False`, and a public
    `finish_building`.
- **W10 dataset import: `/datasets/imports` (preview, start, status, cancel,
  list, resume, undo), archive uploads and the OpenProcessor-export reader.**
  A chunked, persisted, resumable importer on the shared jobs volume
  (`dataset_import/{prepare,runner,chunk,store,undo,upload,op_export}.py`):
  - Class identity is the name in the project's registry; every dataset class
    needs a `map`/`create`/`skip`/`region` decision, pinned in `mapping.json`
    and consumed (not re-resolved) on resume. `import_key` (project, source
    hash, name-based mapping, write-affecting options) makes a repeated
    request idempotent (`reused`); one import per project at a time.
  - Labels are written through `class_label_update` / `ItemLabel.imported`; a
    human edit between plan and write still wins (the merger re-checks the
    lock inside the OCC write). A write-ahead ledger keeps `created` vs
    `updated` truthful across a mid-chunk crash; undo restores each class
    snapshot or region `edit_history`, deletes created items (and crops),
    deprecates created classes, and a second undo reports zeros.
  - Negatives, splits, `test` holdout freeze, `processing: propose` (the
    shared `detect_items` path), region class labels with `parents: detect`,
    and backpressure on the region worker (`OP_DATASET_IMPORT_MAX_PENDING`).
  - Uploads: only regular files and directories, member-count / bytes-written
    / compression-ratio caps, content-addressed extraction, 413 on the stream
    cap; every path is checked after symlink resolution.
  - A thin `scripts/curation/import_labeled_dataset.py` client replaces the
    old labeled mode. New env vars are documented in `env.template`.
- **W10 unified `POST /reprocess`** (scopes `detect|region|vlm|embed`)
  replaces the requeue route, `clear_detection` and the retry endpoints;
  larger detect/embed runs are file-backed jobs. `ActivationImpact` gains
  `suggested_reprocess`.
- **W10 export**: negative frames, split pinning and the frozen test-holdout
  record; items and regions carry `locked` / import provenance on the wire.
- **W10 dataset import (earlier slice): the lock rule, class-name mapping.**
  See `docs/design/openprocessor_internal/any_domain_plan.md` W10 for the spec.
  - `src/clients/occ_locks.py` (new): `is_locked_class` / `is_locked_box`
    / `is_locked_item` / `_is_locked_marker` — the lock rule. Replaces
    `is_human_owned_class` (superset semantics: also locks validated
    imported labels and `test_holdout` items). Re-ingest and
    `/vlm/label_batch` now respect it.
  - `src/services/curation/class_label.py` (renamed from
    `human_label.py`): `ItemLabel` (`.human()`/`.imported()`),
    `class_label_fields()`, `class_label_update()` — the single writer
    every human AND dataset-import class-label write goes through
    (`tests/test_class_label_single_writer.py` gates it).
  - New `src/services/curation/dataset_import/` package: `scan.py` /
    `yolo.py` / `coco.py` (format detection + reading), `mapping.py`
    (`suggest_mapping`/`resolve_mapping` — class mapping is always by
    NAME, never by index; exported for P4's combine-projects wave),
    `issues.py` (the served issue catalog).
  - `tests/integration/test_class_identity_e2e.py`: two YOLO fixtures
    with the same class names in different `data.yaml` index orders (one
    with an extra class) import → export (dense remap) → stub-train
    (`class_remap.json`) → promote (`labels.txt`) → predict, asserting
    `(class_id, class_name)` pairing at every hop.
  - `src/services/projects/busy.py`'s `_dataset_import_jobs()` (the P2
    busy-inventory hook) reports a project's live import.

### Changed
- **W9 (wire changes; see the Cropwright delta list).** `region_verifier`,
  `text_engine_version` and the class `detector`/`labeler` provenance now
  record the resolved model of the endpoint that ran (the probe's model
  root), not the process's `OP_VLM_MODEL`. `GET /models/status` VLM rows are
  one per registered endpoint with `kind: "vlm"` (was `external`), named by
  endpoint, with `active` and `active_in` (the bound project only; other
  projects' use is on `GET /vlm/endpoints`). The `vlm` compose service
  starts through a shell entrypoint so one file serves every catalog model.
  `POST /pipeline/auto_label` now also refuses a bad `?vlm=` when no VLM
  stage runs, as `/start` does. `GET /health` and `/curation/health` report
  the active endpoint.
- `DetectedItem` (`item_doc.py`) gains an optional `label: ItemLabel`
  field; `build_item_doc()` applies `class_label_fields(item.label)` on
  top of the detector-class defaults when set (`None` is a no-op for
  every existing caller).
- `confident_class_sources()` (`ingest_class_sources.py`) now includes
  `LABEL_IMPORT_CLASS_SOURCE` (`external_label`).
- Re-ingest keeps a locked item's whole class state (`class_source`,
  `class_validated`, `test_holdout`) -- fixes #31; a pruned `names` map with
  `nc == max + 1` is accepted.

### Removed
- **W8-cleanup Items 4-6 (breaking, no back-compat): every item-level
  per-box region scalar is gone** — from `RegionFields` (`bbox_norm`,
  `bbox_frame`, `bbox_correct`, `score`, `confidence`, `source`, `detector`,
  `detector_version`, `text*`, `candidate_*`, `embedding`, `cluster_id`,
  `cluster_subid`, `cluster_distance`, `*_legacy` bbox/score), from the
  items mapping, from `boxes_write_fields` (it no longer mirrors the
  representative box onto the item; the item-level `rejection_reason`
  mirror stays, only for an item with no accepted or false-positive box),
  and from the wire: `region_bbox_norm`, `region_bbox_frame`,
  `region_bbox_correct`, `region_score`, `region_confidence`,
  `region_detector`, `region_detector_version`, `region_source`,
  `region_text` / `region_text_raw` / `region_text_confidence` /
  `region_text_source` / `region_text_engine_version` / `region_text_vlm` /
  `region_text_ocr` / `region_text_disagreement` / `region_text_choice` /
  `region_text_vlm_invalid`, `region_candidate_*`, `region_cluster_id` /
  `region_cluster_subid` / `region_cluster_distance`,
  `region_bbox_in_parent`, `region_candidate_bbox_in_parent` and the
  item-level `region_thumbnail_url` (88 item keys now). Nothing in this
  repository reads them: the data is an element of `region_boxes[]`
  (`bbox_in_parent`, `thumbnail_url` per box). The `crop.region_verified`
  event carries `region_count` instead of `region_text`. The worker no longer
  mirrors a candidate onto the item, and its legacy detector fallback is
  gone.
- The legacy repair tools `region_provenance_restore` /
  `scripts/curation/restore_region_provenance.py`, `region_validation_repair`
  / `scripts/curation/repair_region_validation.py` and
  `cascade_detect.region_provenance` (fresh build, nothing to repair);
  `rederive_region_text.py` now re-derives per box.
- The VLM reply keys `region_bbox_correct` / `region_confidence` /
  `region_text` are unchanged: a fixed protocol of the prompt packs
  (`REPLY_*_KEY`).
- `tests/test_no_legacy_region_scalars.py` is now a full-repo guard over
  `src/` and `scripts/` (AST: attribute access on any `RegionFields` value
  including locals, parameters and inline `get_region_fields()`, plus
  wire-name literals), with a two-entry allowlist: the VLM reply keys in
  `region_overlay.py` and the served review-sort id `region_score`.
- **W8-cleanup Item 3 (breaking, no back-compat): the legacy single-box
  routes `PUT /crops/{id}/region` and `PUT /crops/batch_region` are
  deleted outright (no 410), with their whole write chain in
  `region_writes.py` (`region_box_write`, `region_box_doc`,
  `region_confirm_doc`, `same_box`, `candidate_promotion`,
  `human_status_fields`, the `candidate_*` helpers, `fp_cluster_fields`,
  `RegionWriteError`) and the dead pre-W8 worker write builders
  (`verify._region_write_doc` / `_region_reject_doc` /
  `candidate_reject_doc` / `_verify_with_vlm`, `no_verdict.
  cascade_no_verdict` / `no_verdict_reject_doc` and the process-wide
  cascade counter). Box edits go through `PUT /crops/{id}/regions` /
  `PUT /crops/batch_regions` / the per-box routes. `ItemRegionRequest` /
  `ItemBatchRegionRequest` are removed from the OpenAPI contract.
  The one-off same-box-confirm repair tool (`region_provenance_restore.py`
  + `scripts/curation/restore_region_provenance.py`) is deleted with the
  route whose bug it repaired: it only restored detector provenance on
  legacy scalar docs, and a same-box confirm on the box routes now keeps
  provenance (tested below); a fresh build has no such data to repair.
  Coverage migrated, each test proven red against a mutation: same-box
  confirm (provenance kept, stored coordinates kept, within float noise,
  in the parent frame), a moved box is human geometry, accepting a
  false-positive box, the rejected-box reversal (whole-set confirm and
  per-box accept, provenance kept, undoable), parent-frame projection,
  bbox validation, undo.
- **W9: the detection worker's `--vlm-url` flag and the process-wide
  `VlmLabeler` singleton** (`_get_vlm_labeler._insts`). The `env` built-in
  endpoint (`OP_VLM_URL`) is the one remaining environment path; every
  labeler is built by `vlm_factory`.
- **W10 (breaking, no back-compat): `POST /import_labels` and
  `/import_labels/batch`, deleted outright (no 410).** Importing an
  already-labeled dataset is `POST /datasets/imports`. `src/services/curation/
  label_import.py` (the parallel, non-OCC item writer these routes used)
  is deleted entirely, along with `IngestBatchItem.label_txt_path`,
  `IngestBatchRequest.label_source`/`detect_mismatches`, and the
  matching response fields (`labels_imported`, `mismatches`,
  `missed_labels`, `unmatched_detections`, `disagreements`). The
  `labels_confirmed` OpenSearch index/mapping stay (mappings are never
  dropped) as documented legacy — nothing writes it anymore.
- **W4 region-profile CRUD and vocabulary.** Full per-project region-profile
  lifecycle mirroring W3's pack CRUD:
  `src/services/config_store/{profiles,profile_validation}.py`,
  `src/routers/curation/{_region_profile_models,region_profiles,
  config_vocabulary}.py`, `src/services/curation/region_impact.py`.
  Routes: `GET/POST /region_profiles`, `/schema`, `/validate`,
  `/validate_segmenter_prompt`, `/test`, `/{name}`,
  `/{name}/revisions[/{revision}]`, `/{name}/clone` (`from_project`
  read-only, pinned by a test mirroring W3's), `PUT/DELETE /{name}`,
  `/active`, `/active/impact`, `/{name}/activate`, `/active/rollback`,
  `/deactivate`; `GET /config/vocabulary`.
  `src/services/training/promoted_models.py` (moved off
  `routers/curation/models.py` in the W3 pass) backs the new
  `detector_model_not_shared` / `detector_model_other_project` /
  `detector_model_classes_unmapped` sharing checks in
  `profile_validation.py`, using `model_classes.py`'s by-name matching.
  `region_impact.py` aggregates the items index by `region_profile` (+
  revision) plus validated/pending counts for the activation-impact
  response; per glue G2, `ActivationImpact` ships WITHOUT
  `suggested_reprocess` (W10 had not merged when this wave was built --
  whichever of W4/W10 merges second adds the field per the coordinator's
  merge-order call). `regions_requeue.py` / `region_requeue.py` /
  `reprocess.py` are untouched, per G2.
  `pipeline_params.py`'s `DETECTION_PROFILE_REJECTED` reworded off
  `OP_REGION_PROFILE` (E7) to name the config-store activation route.
  Every new route registered in the leak-sweep, with its own
  PREPARE-resets-to-revision-1 hook mirroring W3's.
- **W3 prompt-pack CRUD.** Full per-project prompt-pack lifecycle on top
  of W2's config store: `src/services/config_store/{packs,pack_validation}.py`,
  `src/services/config_store/activation_view.py` (shared active-config
  response builder for packs and, later, region profiles),
  `src/routers/curation/{_prompt_pack_models,prompt_packs}.py`. Routes:
  `GET/POST /prompt_packs`, `/schema`, `/validate`, `/test`, `/{name}`,
  `/{name}/revisions[/{revision}]`, `/{name}/clone` (with the one
  legitimate cross-project read in this wave, `from_project`, bound
  read-only and pinned by a test), `PUT/DELETE /{name}`, `/active`,
  `/{name}/activate`, `/active/rollback`. `REPLY_KEY_CONTRACT` and
  `FORMATTED_PLACEHOLDERS` added to `vlm_prompts.py`; the W8 list-shape
  multi-box pairing check (`pack_multi_region_keys_missing`) is a
  warning everywhere except activation pairing, where it is a
  never-bypassable error. `CLONEABLE_AXES` gained `prompt_packs`
  (every stored pack, current revision only) with a matching
  `_clone_prompt_packs` in `src/services/projects/clone.py`.
- **W4 prep: promoted-model discovery moved to a service module.**
  `src/services/training/promoted_models.py` (`discover_promoted_models`,
  `project_owns_model`) moved out of `src/routers/curation/models.py` so
  service code (region-profile validation) can use it without importing
  a router; `models.py` re-imports both under their old private names.
- **W8 multi-box regions (partial, foundational slice).** Laid the core
  storage primitives for the per-item region-box list
  (`src/services/curation/region_boxes.py`): `RegionBox`, `read_boxes`,
  `boxes_write_fields`, `next_box_id` (never reuses an id after a delete),
  `derive_status` (fixed accepted > false_positive > proposed > rejected >
  empty precedence), `box_query`/`has_any_box_query` nested-query helpers,
  and `BOX_STATES`. Added the new `RegionFields` attributes for the list
  and its item-level summary fields (`region_boxes`, `region_box_embeddings`,
  `region_count`, `region_rejected_count`, `region_max_score`,
  `region_set_complete`, `region_revision`, `region_box_seq`,
  `region_boxes_migrated_at`, `region_legacy_scalars`).
  Added the served per-box `box_states` vocabulary with a semantic `tone`
  field (W8.7 pin) to `GET /regions/statuses`. Added the served,
  operator-tunable write-size abuse guard `region_max_boxes_per_write`
  (env `OP_REGION_MAX_BOXES_PER_WRITE`, default 500), served on
  `region_profile.limits.max_boxes_per_write` (`/health` and
  `/regions/vocabulary`).
  The worker pipeline rewrite (candidate selection, numbered VLM
  overlay, verdict-to-storage) landed in the later "W8 pipeline wiring"
  entry below; removal of the old scalar routes/fields, embeddings, and
  clustering remain open (W8c) — see the handback report.
- **W8a multi-box region human edit routes.** New, additive routes
  alongside the existing single-scalar ones (legacy fields/routes NOT
  removed this pass -- the worker pipeline still writes them
  exclusively; see the handback report):
  - `PUT /crops/{crop_id}/regions` -- sets the full per-item box list,
    sibling-preserving (an element with only `box_id` leaves that box
    untouched); `box_id: null` mints a new box defaulting to `accepted`
    when `state` is omitted (W8 pin 2); `frame: "parent"` projects into
    the source frame server-side (W8 pin 1); optional `region_status`
    applies a whole-set status to the built list in the same write (W8
    pin 3; Enter confirms only `proposed` boxes, never boxes already
    settled); a stale `expected_region_revision` is 409
    `region_conflict` (current revision + box ids + item); over
    `region_profile.limits.max_boxes_per_write` is 422 `too_many_boxes`.
  - `PUT /crops/batch_regions` -- same new-boxes-only semantics across
    many crops (`box_id` must be `null`, else 422 `box_id_in_batch`).
  - `PATCH /crops/{crop_id}/regions/{box_id}` -- per-box `state`/`text`
    patch; every sibling box is left untouched (per-box states persist
    independently).
  - `POST /regions/batch_box_state` -- one state on many `{crop_id,
    box_id}` targets across items (region-gallery triage), never
    touching a target's sibling boxes (contrast `batch_status`, which
    flips every box of each item).
  - `src/services/curation/region_boxes.py` gained `apply_put_boxes`
    (the sibling-preserving merge) and `boxes_with_status` (the W8.7
    whole-set table: `detected` accepts every `proposed` box and 422s
    `no_boxes`/`no_accepted_box` when empty/still-empty-of-accepted;
    `false_positive` flips every box; `verify_rejected` rejects every
    box with `rejection_reason: human`; `no_region_visible` clears the
    list).
  - `serialize_item`/`ItemDoc` now carry `region_boxes` (list) plus the
    item-level summary fields `region_count`, `region_rejected_count`,
    `region_max_score`, `region_set_complete`, `region_revision`,
    additive alongside the existing per-box scalar wire keys.
  - New module `src/routers/curation/regions_boxes_edit.py` (kept
    separate from `regions_edit.py` to stay under the 700-LOC module
    ceiling); added to `tests/curation/test_cross_project_leak.py`'s
    route/body/prepare maps.
  - **Not done** (deferred to W8b/W8c, see the handback report):
    deleting the legacy scalar `RegionFields` attrs/mappings/routes,
    `test_no_legacy_region_scalars.py`, and updating
    `POST /crops/{id}/region/undo` to restore the box list (pin 4) --
    undo still only restores the legacy scalar snapshot today.
- **W8b multi-box region infrastructure (candidate selection, VLM
  overlay, verdict-to-box mapping) — standalone, NOT yet wired into the
  worker's streaming pipeline.** `src/services/detection/
  region_candidates.py` (new): `select_region_candidates()` — floor /
  deterministic tie-break / greedy class-agnostic NMS / cap over a list
  of `RegionCandidate`, the one function every candidate leg (detector,
  segmenter, text-hint re-pass) and `POST /region_profiles/test` will
  use once wired. `src/services/labeling/region_overlay.py` (new):
  `draw_region_overlay` (numbered red-rectangle tags, 1-based, one code
  path for N=1..N), `overlay_description`, `render_region_block`,
  `VlmBoxVerdict`, `box_verdicts`. **D-B (owner decision, 2026-09-26):**
  the pre-W8 flat VLM reply shape is dropped — list shape only (N=1 is a
  list of one); a reply lacking the list key raises
  `MultiRegionKeysMissingError` (`code=pack_multi_region_keys_missing`),
  not a flat-shape fallback. `scripts/curation/worker/verify.py` gains
  `TaskBoxInput` and `verdicts_to_boxes` — maps a combined reply's
  per-box verdicts onto a `list[RegionBox]` (every entry carries its own
  `box_id`, Cropwright C3/Q15), applying the sanity gate per box and the
  no-verdict retry/force-resolve split from the single-box cascade.
  **Superseded by the W8 pipeline-wiring pass below** — these primitives
  are now the live worker's only code path; this bullet is kept for the
  historical record of what W8b added standalone.
  `src/config/region_rejection.py` gains `REJECT_REASON_HUMAN`
  (a human reviewer's per-box rejection is now a labelled catalog
  entry); `src/config/region_state.py` gains `BOX_STATE_ROUTES`, served
  as `box_state_routes` on `GET /regions/statuses`.
- **W8 pipeline wiring: the multi-box primitives now drive the live
  detection worker end to end.** `scripts/curation/worker/runner.py`'s
  streaming stages (`stage_a_consumer`, `stage_a_sam_consumer`,
  `stage_b_combined`) now select N candidates per item
  (`select_region_candidates`, new `DetectionProfile` fields
  `region_nms_iou`/`max_regions_per_item`, the latter renamed from
  `region_max_candidates` and defaulted to 1 by the correctness pass
  below -- multi-box is opt-in per profile, never silently on), render
  one numbered VLM
  overlay per crop (`render_region_block`) instead of a single-box
  prompt, and map the reply's per-box verdicts onto `RegionBox` entries
  (`verdicts_to_boxes`) written via `boxes_write_fields` — every write
  path (`stage_b_combined`, the segmenter high-confidence skip,
  `accept_without_vlm`) now produces `region_boxes`/`region_count`/
  `region_revision`/`region_box_seq`, not the legacy scalar fields.
  `RegionDetector` gained `detect_multi`/`detect_batch_multi` (decode
  every anchor above the confidence floor, not just top-1, capped to
  300 before NMS); `SegmenterClient` gained `segment_multi` (the
  segmenter's HTTP response already returned every candidate; `segment`
  only ever kept the top one). `VlmLabeler.label_combined`/
  `label_combined_batch` take `region_bboxes_norm: list[...]` (was a
  single `region_bbox_norm`) and `VlmCombinedReply.region_boxes:
  list[VlmBoxVerdict]` (was flat `region_bbox_correct`/`region_text`/
  `region_confidence`); `vlm_prompts.py`'s built-in packs
  (`GENERIC_ITEM_PACK`, `GENERIC_REGION_PACK`) and the
  `examples/prompt_packs/vehicle_wheel.json` example were rewritten to
  ask for the nested `region_boxes` shape the parser now requires (a
  real gap from the W8b pass: the prompts still asked for the old flat
  shape while the parser demanded the new one).
  Deleted `scripts/curation/worker/combined.py` and
  `cascade.py::_process_crop` (dead: the streaming pipeline never had a
  legacy two-call cascade fallback; every candidate always went through
  one combined VLM call). Their test coverage was ported onto the real
  pipeline (`tests/curation/test_region_cascade_integrity.py`'s
  `_drive_worker` harness) rather than deleted outright, including a
  full re-port of `tests/curation/test_region_worker.py`'s per-routing-
  row cascade tests; a few rows describing pre-W8 intra-pass fallback
  behaviour (detector/verify reject -> immediately try the secondary
  segmenter in the same pass) were replaced with tests asserting the
  new, correct invariant instead: a combined-verify reject is terminal
  (`verify_rejected`) for that pass, not a same-pass fallback trigger.
  **Not done this pass** (see the handback report): deleting the legacy
  scalar `RegionFields` routes/fields (W8c), per-box embeddings/
  clustering/FP matching, undo-snapshot simplification to box-list-only,
  review-queue/stats/export per-box row shapes, and segmenter service
  `min_score`/candidate-cap config.
  **Correctness note (2026-09-27):** an independent review of this exact
  wiring found a blocker (a `pending_verification` item's stored
  candidate/sibling boxes were read from the wrong source and then
  discarded on write) and 8 majors (stale-revision/box-id reuse under
  concurrent writes, item-level verify/auto-confirm fields silently
  dropped, the region embedding could source a rejected box, multi-box
  silently on by default, a VLM text-echo filter and a text-free-profile
  leak fix each ported incompletely, two more packs still on the flat
  reply shape, and weak N>1 test coverage that let 5 of 6 targeted
  mutations survive the suite). All fixed in the pass documented under
  `### Fixed` below — this bullet's "every write path... now produces"
  claim above was accurate for the happy path only.
- **W8: explicit OpenSearch mapping for `region_boxes` /
  `region_box_embeddings`.** `_items_body()` now maps the W8 nested list
  and its sibling per-box-embedding field explicitly (fixed element-key
  properties, not dynamic-inferred), plus the item-level summary fields.
  No separate `ensure_items_*` step (stacks are re-created).
- **W8: `RegionBoxWire` gains `bbox_in_parent` and `thumbnail_url` per
  box** on the item wire (`region_boxes_to_wire`), so a client can render
  and link a box without a second geometry projection or an extra
  `crop_id` round-trip.
- **W2b: per-project configurable keymap.** `src/config/keymap_actions.json`
  (+ a pydantic model) is the action registry: contexts, groups, defaults,
  `modifiable`, and the wire grammar (combo syntax, locked keys, browser-
  reserved combos, `max_combos_per_action: 3`). Every project stores its
  own override map in its `configs` index; a project with no stored doc
  serves the built-in defaults (`is_default: true`). Routes (all
  project-scoped, no global alias): `GET/PUT {prefix}/keymap`,
  `POST {prefix}/keymap/validate` (dry-run, always 200), `POST
  {prefix}/keymap/reset` (all-or-listed actions). Every write is OCC'd
  (`expected_revision`/`If-Match`). The validator enforces the CW-K §3.1
  codes (unknown action, locked action/key, grammar, too-many-combos,
  browser-reserved, context collision, overlay-must-have-a-key) as 422
  `validation_failed`; a clash with a class hotkey is a separate 409
  `class_hotkey_conflict` with an explicit `unbind_conflicting_class_hotkeys`
  opt-in that clears the conflicting classes' hotkeys in the same write and
  echoes `unbound_class_hotkeys`. `reserved_hotkeys` is now derived from the
  active keymap (`RESERVED_HOTKEY_LETTERS` deleted); `PUT /classes/{id}`
  reports `hotkey_reserved` (422, with the blocking `actions[]`) and
  `hotkey_taken` (409) as typed `ConfigErrorDetail`. `config.changed` gained
  the `keymap` axis (event-only). `CLONEABLE_AXES` gained `keymap`
  (`clone_settings` copies the source's overrides, validated against the
  target's class registry -- a clash is dropped from the copy, never a
  silent unbind). W8's per-box actions
  (`review.region.accept_box`/`reject_box` on `y`/`r`, `box_edit.next_box`
  on `tab`, `box_edit.delete_box`) ship now, served `available: false` on
  a project with no region profile. Exported to
  `contracts/json/keymap_actions.json` via `generate_contracts.py`.
- **`op_global_configs`: the global (non-project-scoped) config store
  (W2 review M3, 2026-09-27).** `src/services/config_store/store.py`
  gains a sibling to the per-project `ConfigStore`: `global_configs_index()`
  (env `OP_GLOBAL_CONFIGS_INDEX`, default `op_global_configs` --
  mirrors `src.services.projects.registry.projects_index()` for
  `op_projects`), `GLOBAL_CONFIGS_INDEX_BODY` + `ensure_global_configs_index()`
  (idempotent create-with-mapping, wired into
  `startup_bootstrap_config_store_safe()` alongside the existing
  per-project index bootstrap), and `get_global_config_store()` -- a
  process singleton cached in its own dict (`_GLOBAL_STORE`, never
  conflated with the per-project `_STORES` cache), reusing
  `src.services.config_store.index`'s existing doc-id/OCC/revision
  primitives unchanged (they were already index-parameterized, not
  project-specific). It never consults
  `src.config.project_context.current_project` -- no project binding is
  required, and calling it while a project happens to be bound has no
  effect on which store it comes back as. This is the foundation W9's
  VLM endpoint registry (a `local_vlm:desired`-style global axis) builds
  on; no CRUD routes exist yet. The project guard needed no code change:
  `op_global_configs` is an unowned index by construction (not
  `op_projects`, not `op_prj_*`-prefixed), so it already passes the
  existing unowned-index rule -- readable/writable unbound, refused for
  a request already bound to a project -- the same shape
  `visual_search_*` already uses and the shape W9's global-router routes
  will run under.
  Tests: `tests/curation/test_global_config_store.py` (new) -- a global
  write is invisible through any project's own `ConfigStore` and vice
  versa (sharing one fake OpenSearch client), the global store requires
  no `bind_project`, it is a singleton regardless of bound state, and
  it's never the same object/index as a project store;
  `tests/projects/test_opensearch_guard.py::test_global_configs_index_is_a_legitimate_unowned_index`
  (new) covers the guard recognition explicitly, alongside the existing
  `op_projects` case. `tests/curation/_fake_config_opensearch.py`'s
  `_FakeIndices` gained `exists`/`create` for the new index-bootstrap
  test.

### Changed
- **W8 (breaking): `region_text` removed from `PATCH
  /crops/{id}/region_meta`.** D decision (owner, 2026-09-26): it was
  always a per-box value riding on an item-level route.
  `ItemRegionMetaRequest` no longer has the field at all
  (`extra='forbid'` 422s a stale client that still sends it, whether or
  not the profile reads text) — per-box text now goes through `PUT
  /crops/{crop_id}/regions` / `PATCH /crops/{crop_id}/regions/{box_id}`
  (W8a), which gained the same text-free-profile guard (422
  `region_text_disabled`) and human-provenance stamp
  (`text_source='human'`, `text_confidence=1.0`,
  `text_choice='human'`) per box that `region_meta` used to apply at the
  item level. No back-compat window.
- **W8 pin 4: `POST /crops/{id}/region/undo` restores the box-list
  snapshot, not just the legacy scalar.** `edit_history.py`'s snapshot
  set gained the per-item box-list fields alongside the pre-W8 per-box
  scalars (additive, not a replacement — the worker still writes only
  the legacy scalars this pass).

### Fixed
- **W3+W4 prompt-pack/region-profile CRUD fix pass, round 7 (independent
  Opus review, 2026-09-28): 1 blocker + 1 major (new, both introduced by
  round 6's `pipeline.py` 3-way split) + 1 minor test-quality gap + 1
  major (pre-existing, mirror of R6-2).** Fixes every finding of the
  round-7 section appended to
  `docs/design/openprocessor_internal/w3_w4_review_2026-09-28.md`:
  - **Blocker (R7-1, the worker's IVF auto-retrain self-trigger crashed
    on every run):** `scripts/curation/auto_label_worker.py`'s
    `_IVF_PIPELINE_PATH` still pointed at `pipeline:pipeline_auto_label`
    — round 6's split made that name the thin PUBLIC route wrapper,
    which has no `progress` parameter, while the worker always calls
    `pipeline_fn(opensearch=, progress=, **args)`. Every IVF centroid
    auto-retrain the idle worker fired ended `status: failed`, so
    centroids never refreshed and the worker re-fired (and re-failed) on
    every idle check. Fixed: point the constant at the internal
    `pipeline:_run_auto_label` instead. Regression:
    `tests/curation/test_r7_fixes.py` binds every pipeline path the
    worker can resolve (the IVF constant and `_run_auto_label`'s own
    `_pipeline_import_path`) against the worker's real call shape.
  - **Major (R7-2, with `prompt_pack` omitted and a legacy settings-doc
    default, the job ran a different pack than it echoed):** R6-1b's
    `(None, None)` re-resolution (dodging the store-active-pack TOCTOU)
    fired for EVERY omitted-pack case, including a legacy
    `settings.defaults.prompt_pack` override or a plain env/file
    default — neither of which has any staleness to dodge. The job's
    `summary`/echoed `prompt_pack` kept reporting the settings-doc name
    while the VLM labeler silently ran `active_prompt_pack()` (the
    env/file default) instead. Per-item `vlm_prompt_pack` stamps stayed
    correct (they read `labeler._pack` directly); only the job-level
    summary and operator expectation were wrong. Fixed: new
    `omitted_pack_is_store_active()` (`pipeline_params.py`) scopes the
    omitted signal to "the echoed name really is the config store's own
    active pack" — both `/start` and the direct-call branch in
    `_run_auto_label` now gate `prompt_pack_omitted` through it before
    passing it to `labeler_resolution_args`. Regression:
    `tests/curation/test_r7_fixes.py` (legacy-settings-doc-default case
    at both call sites, plus the store-active TOCTOU case still holds).
  - **Minor (R7-3, the production line applying R6-1b's `labeler_
    resolution_args` inside the job had no test driving the real
    execution path):** the landed R6-1b tests checked that `/start` sets
    the flag and that the helper maps it in isolation, but nothing drove
    the actual VLM stage through it — a re-implementation of the
    selection logic in `test_r6_worker_process_cold_store` stayed green
    even with the production call site reverted. Regression:
    `tests/curation/test_r7_fixes.py` runs `/start`'s real trigger args
    through `_run_auto_label(..., run_vlm=True)` for both the
    active-pack-switch and direct-call shapes, spying on the actual
    labeler instance obtained, plus a unit-level spy confirming the
    direct-call branch invokes `omitted_pack_is_store_active` at all.
  - **Major (R7-4, pre-existing, mirror of R6-2: clone copies the
    profile but not an env/file pack, and the target's fallback pack was
    never gated):** R6-2 fixed clone skipping PROFILE validation when
    the source has no real stored profile. This round's reviewer found
    the mirror: when the PROFILE axis is cloned but the source's PACK is
    an env/file default (`revision=None`, never written to the store —
    `_clone_activations` skips it), the gate's `pending_sibling` for the
    cloned profile was still the SOURCE's pack body, not the TARGET's
    actual post-clone fallback pack — so a valid source pairing (e.g. a
    full multi-box pack + a 3-region profile) could pass the gate while
    the target actually goes live with ITS OWN stripped pack under that
    same cloned 3-region profile, a pairing nobody checked. Fixed:
    `clone_activation_gate.py` gained `pack_will_be_cloned`, mirroring
    `profile_will_be_cloned`'s exact condition; when the pack won't be
    cloned, the profile-cloned branch validates against
    `resolve_prompt_pack()` bound to the target instead of the source
    body. Also closes Item 4's over-rejection Nit: the pack-only branch
    now returns early when the pack itself won't be cloned either (was
    previously always validating a pack body that would never land).
    Regression: `tests/projects/test_r7_clone_pack_mirror.py`.
  - Nits: `pipeline_public.py`'s OpenAPI `description` restored to
    describe the public contract (it had regressed to describing
    internals after the round-6 split); the internal notes moved to a
    comment. CHANGELOG's round-5 R5-2 entry corrected (it said the fix
    keyed off `prompt_pack is None`; the landed fix actually used the
    dedicated `prompt_pack_resolved` signal). Round-5's m-c/m-d gaps,
    previously undocumented, now noted under the round-5 entry below.
  - Landed as permanent tests: `tests/curation/test_r7_fixes.py`,
    `tests/projects/test_r7_clone_pack_mirror.py`.
- **W3+W4 prompt-pack/region-profile CRUD fix pass, round 6 (independent
  Opus review, 2026-09-28): 1 blocker (two halves) + 1 major + 4
  test-quality fixes.** Fixes every finding of the round-6 section
  appended to
  `docs/design/openprocessor_internal/w3_w4_review_2026-09-28.md`:
  - **Blocker (R6-1, both halves required together):** `/start` jobs run
    in the `auto_label_worker` container, a SEPARATE process from the
    API that received `/start` — round 4/5's fixes were validated only
    in-process, where the config-store snapshot is naturally warm/shared,
    which masked this entirely.
    - **(a) Cold store in the worker:** nothing on the job path ever
      refreshed that process's config-store snapshot, so a pinned
      `(name, revision)` resolved at `/start` time 404'd
      (`ValueError: unknown prompt pack`) the moment `_run_auto_label`
      (the renamed internal implementation, split from the public route
      below) called `_get_vlm_labeler` — for every VLM auto-label run on
      a stored pack, including an explicit `@rev` pin (broken since
      round 4's fix, not just this round). Fixed with a
      `get_config_store().ensure_fresh(opensearch)` at the top of
      `_run_auto_label`, scoped to `prompt_pack_resolved=True` (a
      `/start`-originated job trigger) — the synchronous public route
      runs entirely in-process and doesn't need it; adding it
      unconditionally there was proven, by a genuine
      `test_cross_project_leak.py` regression, to add a real (if
      TTL-gated) extra OpenSearch call with no correctness benefit.
    - **(b) The omitted-pack case still served an un-activated draft
      once the store is warm:** the job's `(active_name, None)`
      placeholder for an omitted `prompt_pack` only redirects to the
      pinned body while `active_name` is STILL the active pack —
      reactivating a different pack between `/start` and the VLM stage
      actually running (a window spanning the whole pre-VLM pipeline,
      not a tight race) makes it silently serve the old name's
      un-activated CURRENT doc (round-1 B1, reachable again once (a) is
      fixed). Fixed with a new `prompt_pack_omitted` signal, computed
      from the raw query param before resolution and threaded through
      the job trigger separately from the echoed `(prompt_pack,
      prompt_pack_revision)` (kept for `summary`/job-status display);
      `labeler_resolution_args()` (`pipeline_params.py`, directly
      unit-tested) resolves the VLM labeler against `(None, None)` —
      always "whatever is active right now" — whenever this is set,
      instead of the echoed name.
    - Regression: `test_r6_worker_process_cold_store` (both the omitted
      and explicit-`@rev`-pin shapes, cold-store simulated via
      `reset_config_stores()` between `/start` and the job) and
      `test_r6_omitted_job_after_active_switch_serves_draft`
      (in-process, an activation change during the run).
    - `pipeline.py`'s `POST /pipeline/auto_label` and
      `POST /pipeline/auto_label/start` routes split into
      `pipeline_public.py` / `pipeline_start.py` (700-LOC ratchet,
      needed once this fix's code landed) — `pipeline.py` keeps only the
      internal `_run_auto_label` implementation, re-exporting
      `pipeline_auto_label` for existing direct-call test sites.
  - **Major (R6-2, the clone pack axis was ungated in the target's own
    context):** R5-3's gate skipped ALL cross-axis validation whenever
    the source's `detection_profile` activation wasn't itself going to
    be cloned (`off`, or an env/registry profile with no stored
    revision) — `_clone_activations`'s own skip condition requires a
    NON-`None` activation revision (env/registry ids never carry one);
    the gate's condition didn't check that, so it also (wrongly) treated
    "a body merely resolved for validation" as "will be cloned." In both
    trigger cases, the target falls back to its OWN existing/default
    profile, and nothing validated the cloned PACK against that. Fixed:
    `check_activation_pair_in_target_context` now matches
    `_clone_activations`'s exact "will this axis be copied" condition,
    and when the profile axis won't be copied, runs the pack's
    cross-axis-only checks (`check_multi_region_keys`, `_check_text_mode`
    — deliberately NOT the full `for_activation` gate, which would also
    re-run intrinsic completeness checks against pre-existing source
    packs never validated end-to-end, the same scoping reasoning R5-3
    itself used) against the target's real post-clone profile
    (`_resolve_profile(None)`, bound to the target project context this
    function already runs in). Regression: `test_r6_clone_pack_only_gap`
    (both trigger variants).
  - **Test-quality fixes** (all previously vacuous — mutation-proven to
    stay green when the fix they claimed to guard was reverted):
    1. The walk-all-writers test's combined-PUT entry now asserts 422
       specifically and that the bad pairing never went live, instead of
       accepting either 409 or 422 for ANY reason.
    2. The landed R5-3 clone test now patches Triton READY (matching
       `test_r6_clone_cause.py`'s pattern) and asserts the exact error
       code set, so it can no longer pass because Triton was simply
       unreachable rather than because the ownership check fired.
    3. Landed guards for 3 previously-untested fixes: the pack-side
       `pending_sibling` override (`{pack, off}` over-rejection,
       `test_r6_combined_stripped_with_profile_off_is_accepted`), m-a's
       rollback `ensure_fresh`
       (`test_r6_rollback_refreshes_snapshot_before_gating`), and the
       `GET /active` source-label fix for a deleted-but-still-activated
       pack (`test_r6_active_label_stored_for_deleted_but_still_
       activated_pack`).
    4. `prompt_pack_resolved`/`prompt_pack_revision` are no longer
       reachable from an HTTP request at all (previously hidden
       `Query(include_in_schema=False)` params directly on the combined
       route, settable via `?prompt_pack_resolved=true` to skip the
       unknown-pack 422) — real Python-only params on `_run_auto_label`
       now, set only by `/start`'s resolved pin and the worker's direct
       call; the public `pipeline_auto_label` route always forces them
       off.
  - Landed as permanent tests: `tests/curation/test_r6_probes.py`,
    `tests/projects/test_r6_clone_cause.py`; `tests/curation/
    test_r5_probes.py` and `tests/projects/test_r5_clone_probes.py`
    updated in place for items 1 and 2 above.
- **W3+W4 prompt-pack/region-profile CRUD fix pass, round 5 (independent
  Opus review, 2026-09-28): 2 blockers + 1 major + 1 structural test +
  3 minors.** Fixes every finding of the round-5 section appended to
  `docs/design/openprocessor_internal/w3_w4_review_2026-09-28.md`:
  - **Blocker (R5-1, introduced by round 4's own m1 fix — combined
    pack+profile `PUT /settings` checked each axis against the OTHER's
    OLD stored value, not its PENDING one):** `run_activation_gate` gained
    an optional `pending_sibling` override; `PUT /settings` now resolves
    BOTH axes' target bodies first (`_resolve_config_store_axis`, the
    gate-free half of the old `_prepare_config_store_axis`), then gates
    each axis using the OTHER axis's PENDING target from the SAME
    request (`_pending_sibling_for_gate`) when both are set in one call.
    Before this fix, a single `PUT /settings` could pair a multi-box-
    stripped pack with a multi-region profile even though `/activate`
    (with `force`) still correctly 422s that exact pairing standalone.
    Regression: `test_r5_two_axis_put_cannot_pair_stripped_pack_with_
    multi_profile` (both key orders).
  - **Blocker (R5-2, round-4's own R4-3 fix missed the no-pack-specified
    default path — the common/Cropwright case):** `pipeline_auto_label`
    now re-resolves on the dedicated `prompt_pack_resolved` signal (`/start`
    sets it `True` unconditionally, for every shape of its request), not
    `prompt_pack_revision is None`. `/start`'s truly-omitted-pack path resolves to `(active_
    name, None)` at request time — `revision=None` there means "follow
    the active pack's PINNED body dynamically" (`get_prompt_pack`'s
    active-name redirect, B1 round-2), not "unresolved." Keying the job's
    re-resolution off the revision re-entered the N7 "bare name -> latest
    saved revision" branch on that already-resolved name, silently
    running an un-activated draft — the original round-1 B1 bug, reachable
    again via the default path since the job no longer crashes.
    Regression: `test_r5_start_job_runs_what_the_request_resolved[-1]`
    (the omitted-query case; the two explicit-pack cases were already
    covered by R4-3's own tests).
  - **Major (R5-3, project clone never routed through the gate at all):**
    `run_activation_gate` gained an optional `body` override (skips the
    name-based `build_record` lookup, validates a caller-supplied body
    instead) so a caller can validate one project's body bound to a
    DIFFERENT project's context. `_validate_clone` now re-resolves the
    source's active `detection_profile` body and runs the gate bound to
    the TARGET (its own class registry, Triton state, `project_slug`)
    before any write, using the source's active pack as
    `pending_sibling` when the pack axis is cloned alongside it. Scoped
    to `detection_profile` (the axis with today's actual target-specific
    condition, `detector_model_not_shared`) rather than also re-running
    full `prompt_pack` `for_activation` validation, which would reject
    pre-existing source packs on unrelated completeness checks with no
    reported gap behind it. Regression:
    `test_r5_clone_activates_source_private_detector_in_target`.
  - **Structural test (asked for in round 4, still missing per round 5):**
    `test_r5_walk_every_activation_writer_rejects_gate_failing_pair`
    enumerates every real activation-writing call site — direct activate
    ×2, settings-bridge ×2 axes + the combined-request case, rollback ×2,
    and clone — against the SAME gate-failing pack+profile pair, so a
    future caller that skips the gate on any of these paths turns it red.
    Confirms `detection_profile` rollback rejects the pair too (round 5
    found it worked but had no landed test — see the m4 correction
    below).
  - **Minor (m-b, rollback to a deleted pack/profile resurrected it as
    active, and `GET /active` mislabeled its `source` as `'env'`):** the
    gate validates the immutable `<kind>:<name>@<rev>` revision-copy doc,
    which survives a `DELETE` of the CURRENT doc, so a target deleted
    while it was `previous` used to come back active. `rollback_axis` now
    refuses (`LookupError('previous_deleted')` -> 409 `previous_deleted`,
    new `ConfigErrorDetail` code) when the target's revision was
    genuinely stored (M6: env/file ids never carry a revision) but is no
    longer in the store's current names. `activation_view.py`'s `source`
    field fix (a non-`None` revision alone now implies `'stored'`) is
    kept as defense in depth for any activation state the rejection
    doesn't cover. Regression: `test_r5_rollback_to_deleted_pack`.
  - **Minor (m-a, rollback never refreshed its snapshot before gating):**
    `rollback_axis` now calls `store.ensure_fresh(client)` before reading
    the cross-axis gate inputs or the deleted-target check above.
  - **Known limitation (m-c, residual): a two-axis `PUT /settings` can
    still half-write if the SECOND axis's *apply* hits an
    `ActiveConflictError` after the first axis's already committed.**
    The gate (R5-1) now runs both axes' checks against each other's
    pending target before either write, so a genuinely invalid pairing
    is rejected up front — this is only the narrower window where the
    second axis's OWN OCC check fails after that. Documented, not fixed:
    apply both axes under one shared OCC token to close it fully.
  - **Known limitation (m-d, inherent): concurrent single-axis
    activations on different axes can each pass against the other's old
    (not pending) state.** OCC here is per axis, so a `prompt_pack`
    activation racing a `detection_profile` activation (each via its own
    direct `/activate` route, not the combined `PUT /settings` R5-1
    covers) can land a pairing neither request's own gate check saw
    together. This needs a genuine race to trigger, so it's rated minor.
    Documented, not fixed: either axis's write condition would need to
    include the other axis's activation `etag`.
  - **Minor (CHANGELOG over-claims from round 4):** corrected — see the
    round-4 entry above, now flagging that clone was not routed through
    the gate and that only `prompt_pack` rollback had a landed test.
  - Landed as permanent tests: `tests/curation/test_r5_probes.py`,
    `tests/projects/test_r5_clone_probes.py`.
- **W3+W4 prompt-pack/region-profile CRUD fix pass, round 4 (independent
  Opus review, 2026-09-28): 1 blocker + 2 majors + 5 minors.** Fixes every
  finding of the round-4 section appended to
  `docs/design/openprocessor_internal/w3_w4_review_2026-09-28.md`:
  - **Blocker (R4-1, rollback bypassed the activation gate — the 4th
    round of "fixed the probed caller, missed a sibling"):** consolidated
    the `for_activation` gate into ONE shared function,
    `run_activation_gate` (new `src/services/config_store/
    activation_gate.py`), and routed every DIRECT activation writer
    through it: `POST /prompt_packs/{name}/activate`, `POST
    /region_profiles/{name}/activate`, `PUT /settings` (the round-3 N1
    fix, now delegating to the shared function instead of its own copy),
    and — the previously ungated path — `POST /prompt_packs/active/
    rollback` / `POST /region_profiles/active/rollback` (wired into the
    shared `rollback_axis` in `activation_apply.py`, so both axes'
    rollback get it in one place). Rollback runs the gate with no
    `force`, mirroring the settings bridge. A rollback target naming a
    revision that is no longer the STORED CURRENT one (e.g. superseded by
    a later, never-activated PUT) is still resolvable: the gate accepts
    an optional OpenSearch client and falls back to the immutable
    `<kind>:<name>@<rev>` revision-copy doc when the in-memory snapshot
    only has the current revision. One test,
    `test_r4_rollback_bypasses_multibox_gate`, plus the settings-bridge
    N1 probes, now walk activate/settings/rollback with a gate-failing
    revision — **round-5 review found this was still incomplete: project
    clone (a separate activation writer, `_clone_activations` ->
    `index.activate`) was never routed through the gate, and only the
    `prompt_pack` axis of rollback had a landed test, not
    `detection_profile`. Both closed in the round-5 fix pass below.**
  - **Major (R4-2, `expected_active` shape bug, same class as round-3's
    N3a but in `settings.py`):** `_activate_config_store_axis` (now split
    into `_prepare_config_store_axis`/`_apply_config_store_axis`, see
    m1 below) folded an explicit deactivation (`'off'`) into Python
    `None` when building `expected_active` — but `index.activate`'s OCC
    only treats `None` as "no doc has ever been written"; an explicit
    deactivation stores a real `{'name': None, 'revision': None}` doc.
    Every `PUT /settings` after one deactivation 409'd `active_conflict`
    on both axes. Fixed to map `'off'` to the real doc shape.
  - **Major (R4-3, predates this branch; `/pipeline/auto_label/start`
    jobs crashed, and the pin was discarded):** `pipeline_auto_label`
    (`routers/curation/pipeline.py`) now accepts a hidden
    `prompt_pack_revision` param — `/start`'s background job trigger was
    already putting it in the job args, and the function had no such
    parameter, so every `/start` job raised `TypeError`. When the caller
    supplies it (i.e. `/start` already resolved-and-pinned), the function
    uses it as-is instead of re-running `resolve_run_prompt_pack` on the
    bare name, which (after N7) would silently re-pin to the LATEST
    saved revision instead of the one the request actually pinned.
  - **Minor (m1, two-axis `PUT /settings` half-applies):** resolving +
    gating both config-store axes now happens BEFORE either is activated
    (`_prepare_config_store_axis` / `_apply_config_store_axis` split), so
    a second axis's 422 can no longer leave the first axis's activation
    committed.
  - **Minor (m2, N3 name-collision check read a stale/cached target
    store):** the activations-axis "target already has a stored config
    under this name" check in `clone.py` now reads the specific doc
    straight from the client instead of the 1s-TTL `ConfigStore`
    snapshot, closing (most of) the narrow cross-worker race window.
  - **Minor (m3, the landed N4 test was vacuous):**
    `test_r3_rollback_transient_error_status` now asserts on the STORED
    `activation:*` doc (`get_activation`) instead of `GET /active`'s
    in-process snapshot, which was stale in both the pre-fix and
    post-fix worlds and so passed either way.
  - **Minor (m4, missing test coverage):** landed
    `test_r4_n1_profile_settings_gate` (profile axis of the settings-
    bridge gate), `test_r4_n5_profile_pinned_copy_missing` (profile side
    of the pinned-copy fail-open fix), and
    `test_r4_settings_reactivate_after_off` (on→off→on through
    `PUT /settings`, both axes).
  - **Minor (m5, test hygiene):** fixed stale "Round-3" docstrings on the
    now-permanent round-3 probe files; gave
    `test_r3_r1_far_past_revision` real assertions (both a
    revision-that-once-existed-but-is-no-longer-current/active AND a
    never-saved revision now assert the specific 422 wording, instead of
    only printing the outcome); tightened
    `test_r3_existing_target_previously_deactivated` to require success
    (not "success or a clean 409"); made `_apply_clone`'s
    `target_activations` parameter required (no more silent
    "assume the target is empty" default a future caller could
    reintroduce N3a through).
  - Landed as permanent tests: `tests/curation/test_r4_probes.py`,
    `tests/projects/test_r4_clone_probes.py`.
- **W3+W4 prompt-pack/region-profile CRUD fix pass, round 3 (independent
  Opus review, 2026-09-28): 1 blocker + 2 majors + 4 minors + 1 nit.**
  Fixes every finding of the round-3 section appended to
  `docs/design/openprocessor_internal/w3_w4_review_2026-09-28.md`:
  - **Blocker (`PUT /curation/settings` bypassed activation validation):**
    `_activate_config_store_axis` (`routers/curation/settings.py`) now
    runs the same `for_activation` `validate_pack`/`validate_profile`
    gate `POST /{name}/activate` runs, before activating a
    `defaults.prompt_pack`/`defaults.detection_profile` value — a
    multi-box-stripped revision that `/activate` correctly 422s (even
    with `force`) can no longer go live through the Cropwright
    default-pack dropdown's settings route instead. No `force` support on
    this bridge — any blocking error always 422s.
  - **Major (wrong provenance stamp on a per-run draft pin):**
    `get_prompt_pack(name, revision=N)` now tags the resolved
    `PromptPack` instance with the exact revision it served;
    `prompt_pack_stamp` prefers that tag over re-deriving from the
    store's *active* ref, so a per-run `name@<draft rev>` pin (never
    activated) stamps VLM writes with the draft's own revision, not
    whatever happens to be currently active. Also fixes the same-shaped
    bug for a per-run bare `name` (no `@rev`), which now correctly
    resolves to the *latest* saved revision per any_domain_plan.md §3.7
    instead of the active one (`resolve_run_prompt_pack` pins the current
    stored revision explicitly for that case) — the "default/omitted"
    path (no `prompt_pack` param at all) is unchanged and still serves
    the active pinned body.
  - **Major (`clone_settings_into` half-writes before a 409):**
    `_validate_clone`'s `activations` axis check now (a) captures the
    target's REAL existing activation doc (including a
    previously-deactivated `{'name': None, 'revision': None}` one) and
    threads it through as `_clone_activations`'s `expected_active`
    instead of assuming an empty target, and (b) refuses up front when
    the target already has a STORED (even never-activated) pack/profile
    sharing the source's active name for either axis — both cases the
    old `existing.get('name')`-only check missed, which used to write
    settings/keymap/pack/activation docs before a spurious 409.
  - **Minor (500 after a committed activation):** `activate_axis`
    (`store.py`) and a new shared `rollback_axis` (replacing
    `rollback_pack`/`rollback_profile`'s own write-then-resolve calls
    into `index.rollback`) now resolve the pinned body BEFORE writing the
    activation doc, so a transient error on that read aborts cleanly with
    nothing committed instead of surfacing a 500 after the write already
    landed.
  - **Minor (pinned-copy 404/decode failure fails open):**
    `active_prompt_pack()` / `get_active_region_profile()` no longer fall
    back to the unvalidated current doc when the ref names a revision and
    the pinned copy is missing or malformed — only when the current doc
    genuinely IS that same revision. Otherwise falls back to the env/file
    default and logs an error, closing the other half of round-2's B1
    fix.
  - **Minor (circular import in the `vlm_prompts`/`vlm_prompt_resolution`
    split):** `vlm_prompt_resolution.py`'s module-level import of
    `PromptPack`/`BUILT_IN_PACKS`/`_BUILT_IN_NAMES` is now `TYPE_CHECKING`
    plus per-call-site lazy imports, so importing it before `vlm_prompts`
    in a cold interpreter no longer raises `ImportError`.
  - **Nit (misleading 422 message):** the per-run `name@<far-past
    revision>` 422 no longer says "unknown revision" (the revision can
    genuinely exist in history) — it now says the revision isn't
    resolvable in this process, and names the actual scope: the current
    saved revision or the currently-activated revision only, not a full
    historical lookup.
  - Landed as permanent tests: `tests/curation/test_r3_activation_probes.py`,
    `tests/projects/test_r3_clone_probes.py`.
- **W3+W4 prompt-pack/region-profile CRUD fix pass (independent Opus
  review, 2026-09-28): 2 blockers + 5 majors + isolation gaps.** Fixes
  every finding of
  `docs/design/openprocessor_internal/w3_w4_review_2026-09-28.md`:
  - **Blocker 1 (activation revision pin ignored):** `ConfigSnapshot`
    gained `active_pack_body`/`active_profile_body` (`store.py`), resolved
    at activation/rollback time from the immutable `<kind>:<name>@<rev>`
    revision copy, not the current-by-name doc. `active_prompt_pack()`
    (`vlm_prompts.py`) and `get_active_region_profile()`
    (`profile_registry.py`) now serve that pinned body. A `PUT` on the
    active pack/profile still writes a new revision (per
    any_domain_plan.md §4.4) but no longer changes what's running until a
    separate `/activate` call re-validates it — closing the path where a
    PUT could silently bypass the "never-bypassable" multi-box check.
  - **Blocker 2 (default clone 409s after partial writes):** `_clone_activations`
    (`services/projects/clone.py`) now reuses the revision
    `_clone_prompt_packs`/`_clone_regions` already wrote for the same
    name instead of re-`save_config`-ing with `expected_revision=None`,
    so the default (all-axes) clone from a source with an active stored
    pack no longer 409s `target_not_empty` partway through.
  - **Major (name validation bypassed by PUT/clone):** PUT now 404s an
    unknown name (creation is POST's job) and runs `_check_name`; clone
    validates `new_name` the same way, both for packs and region profiles.
  - **Major (template pack activation lies):** `POST /prompt_packs/{name}/activate`
    now 403s a template, mirroring the region-profile route's existing guard.
  - **Major (`ActivationImpact.items_total` capped at 10k, untested):**
    `region_impact.py` sets `track_total_hits: True`;
    `tests/curation/test_region_impact.py` pins every field's arithmetic
    against a seeded fake response (previously no test asserted any
    number).
  - **Major (`from_project` clone 500s on a stale-target name collision):**
    both clone routes now refresh the target's own store after the
    source's read-only bind exits, and map `RevisionConflictError` to 409
    `name_conflict` instead of a bare 500.
  - **Isolation gaps:** `test_cross_project_leak.py`'s marker set now
    includes `f'{slug}-model'` (the pattern `_stored_prompt_pack`/
    `_stored_region_profile` already seed), so a config-store *content*
    leak into a list response is now detectable; a new
    `test_from_project_clone_reads_source_index_only_under_the_real_guard`
    drives `from_project` through the real guarded transport for both
    clone routes, pinning that the source `configs` index is only ever
    read, never written.
  - **Refactor:** the duplicated from-project clone-source resolution in
    both routers is now one shared `services/config_store/clone_shared.py`.
    `src/routers/curation/models.py`'s compat re-import of
    `discover_promoted_models`/`project_owns_model` under old private
    names is gone (no-shims rule) — `_core_models()` moved down into
    `services/training/promoted_models.py` so the service no longer
    depends back on the router it was extracted from; all call sites
    updated. Removed dead code: `packs.RESERVED_NAME_ERROR`,
    `pack_source`/`profile_source`, the `get_activation` re-export from
    `packs.__all__`, the `del check_multi_region_keys` breadcrumb, and
    `region_profiles.py`'s duplicate `store2`.
  - **Deferred (documented, not fixed this pass):** `GET
    /region_profiles/schema` is left as an explicit `TODO` (still a
    placeholder — every field types as `'string'`/`'advanced'`/`enum:
    None`) rather than implemented, since it's out of scope for a
    blockers/majors fix pass; activating a stored pack/profile by an
    explicit past `revision` number remains unreachable (`build_record`
    returns `None` for a non-current revision) — request field kept,
    not wired; deleting a pack/profile that is only an activation's
    `previous` (not the current active) is still allowed.
- **W3+W4 prompt-pack/region-profile CRUD fix pass, round 2 (independent
  Opus review, 2026-09-28): the round-1 fix pass above left B1 and B2
  only partly fixed.** Fixes every round-2 finding of
  `docs/design/openprocessor_internal/w3_w4_review_2026-09-28.md`:
  - **Blocker 1 remaining (activation pin bypassed by other callers):**
    round 1 pinned `active_prompt_pack()`/`get_active_region_profile()`,
    but the API's own VLM write routes (`/vlm/label_batch` and 3
    siblings, via `_get_vlm_labeler(await _default_pack_name(...))`) and
    the pipeline's omitted-`prompt_pack` default still resolved the
    active pack BY NAME, serving the un-activated current doc under the
    activated revision's stamp — the original bug via a new call site.
    `get_prompt_pack()` (`vlm_prompts.py`, resolution logic split into
    the new `vlm_prompt_resolution.py` to stay under the 700-LOC ratchet)
    now redirects a by-name lookup of the store's currently active pack
    to the same pinned body `active_prompt_pack()` serves, and resolves
    an explicit `name@<rev>` against the pinned-active revision's copy
    too when it isn't the current one (fixes the R1 regression below in
    the same change). A transient pinned-revision fetch failure in
    `_resolve_active_body` (`store.py`) no longer falls open to the
    unvalidated current doc: only a genuine 404 means "nothing to pin"
    now — any other exception propagates so `refresh()` marks the
    snapshot `stale` and keeps the last known-good pinned body.
  - **Blocker 2 (clone partial-write 409 + activation-skip regression):**
    the round-1 fix only passed its own test because that test called
    `_apply_clone` directly, skipping `_validate_clone`'s store warm-up —
    in the real `clone_settings`/`clone_settings_into` flow the target
    store's 1s cache TTL hid `_clone_prompt_packs`'s just-written pack
    from `_clone_activations`'s `already_cloned` check, so a clone
    finishing inside that window (the typical case) still 409'd
    `target_not_empty` after settings/classes/keymap/packs had already
    landed. `_clone_prompt_packs` now returns what it wrote
    (`{name: StoredConfig}`) and `_clone_activations` takes that as an
    explicit argument instead of re-reading the target store's cache —
    never needs to see what was just written, so the race is gone
    entirely (no sleep needed to make the real flow pass). The round-1
    fix also activated the source's *current* pack body instead of its
    *activated* one when a sibling axis had already written that name,
    undoing W2 Minor 2 and opening a clone-shaped activation-gate bypass;
    `_clone_activations` now always activates the source's fetched
    ACTIVATED body, reusing the sibling axis's write only when it's
    byte-identical, else saving the activated body as an additional
    revision so the target lands in the same current-vs-activated split
    state as the source.
  - **Regression (R1, introduced by this round's own B1 fix):**
    `resolve_run_prompt_pack` (`pipeline_params.py`) now validates
    `name@<rev>` at request time via `get_prompt_pack` (422 for a
    genuinely unknown revision) instead of accepting any syntactically
    valid pin and only failing the background job later — fixes
    `?prompt_pack=name@<the-still-active-but-superseded-revision>`
    (exactly B1's motivating case) 202-then-job-failing.
  - **M1b (missing safety-net test for the read-only clone-source
    guard):** confirmed live-correct but untested — removing
    `read_only=True` from any of the four `bind_project(source,
    read_only=True)` binds in `clone.py` or the one in
    `clone_shared.py`'s `read_source_record` left the whole suite green.
    New `tests/projects/test_source_read_only_write_guard.py` plants a
    real write attempt (via the guard's own `check_request`) inside each
    of the five source binds and asserts `ProjectReadOnly` stops it;
    manually confirmed each test goes red when its corresponding
    `read_only=True` is removed.
  - Round-1's probe tests (`test_p*`/`test_c*`) are landed as permanent
    regression tests: `tests/curation/test_w34_round2_activation_regression.py`,
    `tests/projects/test_w34_round2_clone_regression.py`.
- **W8 pipeline-wiring correctness fixes (independent Opus review,
  2026-09-27): 1 blocker + 8 majors.** Fixes every finding of
  `docs/design/openprocessor_internal/w8_pipeline_review_2026-09-27.md`
  against the W8 pipeline-wiring pass above.
  - **B1 (blocker, data loss):** `pending_verification` re-verification
    now reads its candidate from the item's stored `region_boxes` list
    (whichever boxes are `state=='proposed'`, keeping their `box_id`),
    never the legacy single scalar. The write path merges the resolved
    verdict boxes back into the CURRENT stored list (`region_boxes.
    merge_boxes_for_write`) instead of replacing it outright, so an
    untouched sibling box (already accepted/rejected, or a second
    proposed box) is never silently discarded (`_ItemTask.pending_merge`,
    opt-in only for this path — every fresh-detection write keeps its
    pre-existing replace behaviour, unchanged).
  - **M1 (stale revision/box-id reuse):** the box-list write
    (`region_boxes`, `region_count`, `region_revision`, `region_box_seq`,
    ids) is now computed inside `bulk_writer._merge`, against the live
    doc `occ_skip_on_conflict_bulk` re-reads immediately before the
    write, never the task's own fetch-time snapshot. Fresh candidates
    get a placeholder id (`region_boxes.new_box_placeholder`) at
    selection time and a real one only at write time
    (`region_boxes.finalize_box_ids`, minted against the CURRENT
    `region_box_seq`), so a concurrent write's ids can never collide and
    the revision only ever increments forward.
  - **M2 (item-level verify fields dropped):** `region_verified`/
    `region_verifier`/`region_verifier_version`/`region_verified_at`/
    `region_validated`/`region_auto_confirmed` are written again on
    every box-list write (`verify.item_verification_fields`).
    `region_auto_confirmed`'s box-aware rule: at least one accepted box,
    and every accepted box independently passes the pre-W8 2-of-2
    auto-confirm policy (`verify.boxes_auto_confirmed`). The skip-verify
    and no-VLM-configured paths stay `verified=False`/
    `auto_confirmed=False`, matching pre-W8 behaviour.
    **Correctness note (2026-09-28 re-review confirmation):** the fields
    were genuinely restored, but `verified`'s reconstructed rule
    (`reply is not None`) was wrong -- true even for a VLM rejection or
    a not-visible answer, contradicting the pre-W8 write and the
    existing `test_region_status_invariants.py` invariant. Corrected in
    the fix pass below (R-M4): `verified` now requires at least one
    ACCEPTED box.
  - **M3 (embedding from a rejected box):** the region-embedding source
    (`_ItemTask.candidate_in_crop`) now syncs to the first ACCEPTED box
    (`runner._sync_accepted_candidate`), never `candidates[0]` (the
    top-scored candidate, which the VLM may have rejected while
    accepting a lower-scored sibling).
    **Correctness note (2026-09-28 re-review confirmation):** the
    selection logic itself was correct, but moving the box-list's status
    write into `bulk_writer._merge` (M1's fix, same commit) removed
    `F.status` from `t.update_doc` before `region_embed_stage.
    _eligible_tasks` and `bulk_writer._publish_region_events` ever read
    it, so this fix had NO OBSERVABLE EFFECT: no region embedding and no
    `crop.region_verified` event were produced for ANY worker output,
    including the single-box case that worked before this whole pass.
    The gate didn't catch it because the embed stage is disabled by
    default in `_drive_worker` and the M3 test spied on
    `_sync_accepted_candidate` instead of asserting the real written
    output. Fixed below (R-M1), with a real end-to-end test replacing
    the spy.
  - **M4 (multi-box silently on by default):** `DetectionProfile.
    region_max_candidates` renamed to `max_regions_per_item`
    (any_domain_plan.md W8.4/W8.9's name), default changed 3 -> 1.
    Multi-box is now opt-in per profile.
  - **M5 (echo-suppression filter dropped):** `region_overlay.
    box_verdicts`/`_clean_text_reply` gained an `echoes` parameter (the
    reply's picked class name, `make`, `model`, and their join);
    `vlm_labeler._combined_reply_from_entry` computes and passes it,
    restoring the pre-W8 flat parser's `_clean_combined_region_text`
    suppression on the new per-box path. Also restored the full 7-value
    sentinel set (`unreadable`/`-` were missing from the W8 rewrite's
    4-value set).
  - **M6 (text-free leak, rejected boxes):** the prior pass's
    `_box_with_resolved_text` leak fix only ran on accepted boxes. New
    `region_text_stage.resolve_rejected_box_text` runs on every
    rejected/no-verdict box too (both the ordinary reject path and the
    no-verdict-cap-reached path): drops the raw VLM text entirely on a
    text-free profile, and holds it to the same `region_text_rules` an
    accepted box's reading is held to on a text-reading profile.
  - **M7 (flat-shape prompt bug, 2 more files):** `data/
    prompt_pack.example.json` and `docker/test/fake_vlm.py` rewritten to
    the nested `region_boxes` shape (the built-in packs and
    `examples/prompt_packs/vehicle_wheel.json` were already fixed by the
    prior pass).
  - **M8 (weak N>1 test coverage):** `_drive_worker`'s `primary`/
    `segmenter` params now accept a list of raw candidates (not just 0 or
    1), and new tests drive 2-3 candidates through the detector leg, the
    segmenter leg, and the combined VLM stage, asserting per-box outcomes
    (state, score, id, text) distinctly — confirmed to catch the review's
    "cap forced to 1" and "skip-verify guard removed" mutations by
    reproducing them against the new tests.
  - **Flaky test fix:** `test_region_no_verdict_cap.py::
    test_real_verdict_before_the_cap_writes_normally_and_clears_the_count`'s
    harness (`_drive_worker`'s `_stopper`) used a FIXED 500 x 10ms poll
    budget (5s) regardless of actual wall-clock elapsed; under load
    `asyncio.sleep(0.01)` can itself take longer than 10ms, so a
    multi-retry scenario (6 combined-VLM round trips across 2 write
    cycles) could run out of budget one retry short of the cap and stop
    the worker early (~1/12 standalone failure rate, reproduced). Changed
    to a deadline-based wait (18s, still well under the outer 30s
    timeout) — 0/12 failures after the fix; this was a harness
    time-budget issue, not a pipeline logic bug (the B1/M1 fixes above
    were unrelated, separately reproduced and fixed bugs found while
    investigating).
  New tests: `tests/curation/test_region_pending_verification_b1.py`,
  `tests/curation/test_region_write_occ_m1.py`,
  `tests/curation/test_region_multi_box_pipeline.py`; extended
  `test_region_auto_confirm.py`, `test_text_free_worker.py`,
  `test_region_cascade_integrity.py`, `test_vlm_prompts.py`,
  `test_verdicts_to_boxes.py`, `test_detection_profile.py`.
- **W8 pipeline-wiring fix pass 3 (re-review confirmation, 2026-09-28):
  1 blocker + 3 majors introduced/left open by the previous fix pass.**
  Fixes every new finding of the "Re-review 2026-09-27, fix-pass
  confirmation" section appended to
  `docs/design/openprocessor_internal/w8_pipeline_review_2026-09-27.md`.
  - **R-B1 (blocker, livelock):** in merge mode (Path 1 re-verifying a
    stored `proposed` box), a `region_visible=False` combined-VLM reply
    now resolves each re-verified candidate to a `rejected` box (keeping
    its id, reason `region_visible_elsewhere`) via `derive_status` over
    the full merged list, instead of writing an empty box list.
    Previously `merge_boxes_for_write(stored, [])` never touched the
    stored box's id, so it stayed `proposed` forever: every poll made
    another VLM call and bumped the revision, unbounded --
    `docker/test/fake_vlm.py` defaults `region_visible` to `false`, so
    this would have hit immediately in the dev/E2E stack. Fresh-detection
    writes (no stored box to lose) are unchanged: still an empty list,
    terminal `no_region_visible`.
  - **R-M1 (major regression, silent):** restored a PROVISIONAL
    `F.status` directly onto `t.update_doc` in `runner._box_list_doc`
    and `region_text_stage.accept_without_vlm` (computed the same way
    `derive_status` would from this task's own boxes, so it can only
    ever under-report eligibility, never over-report it) so
    `region_embed_stage._eligible_tasks` -- which runs BEFORE
    `bulk_writer._merge` -- can see it again. `bulk_writer._merge` now
    also corrects `task.update_doc[F.status]` to the REAL merged status
    once it's known, so `_publish_region_events` (which runs AFTER the
    merge) reads the accurate value. Region embeddings and
    `crop.region_verified` events are written/published again for every
    worker output, not just the ones this pass happened to also touch.
  - **R-M2 (major regression):** `region_text_stage.accept_without_vlm`
    now reuses `cand.box_id` (the stored box's own id, set when the
    candidate came from a stored `proposed` box) instead of always
    minting `new_box_placeholder(0)`. The no-VLM Path-1 accept used to
    leave the human's stored box `proposed` forever AND mint a brand-new
    duplicate `accepted` box for the same geometry.
  - **R-M3 (M1 residual):** `region_boxes.merge_boxes_for_write` gained
    an optional `baseline` parameter (the task's fetch-time
    `stored_boxes` snapshot); `bulk_writer._merge` now passes it. A box a
    human moved or deleted DURING its own VLM re-verification call is
    detected per-box (baseline vs. the live re-read doc) and the human's
    newer state wins -- a move is no longer silently reverted to the
    stale geometry the VLM verified against, and a delete is no longer
    resurrected by the pass's now-stale verdict.
  - **R-M4 (M2 semantics):** `verified` is `reply is not None AND at
    least one box accepted`, never `reply is not None` alone -- a VLM
    rejection or a not-visible answer is a real reply but never a
    confirmed region. Restores the pre-W8 semantics and the
    `test_region_status_invariants.py` invariant this contradicted.
  - **Nit:** corrected the `_stopper` deadline-fix comment
    (`test_region_cascade_integrity.py`) -- it described the flake's
    cause backwards (a slower `asyncio.sleep(0.01)` makes a fixed
    500-iteration budget take MORE wall time, not less); the real cause
    is plain wall-clock variance against a fixed ~5s budget under xdist
    CPU contention.
  New tests: `tests/curation/test_region_not_visible_terminal_r_b1.py`,
  `tests/curation/test_region_no_vlm_reuses_box_id_r_m2.py`,
  `tests/curation/test_region_merge_concurrent_edit_m1.py`; rewrote
  `test_region_multi_box_pipeline.py`'s M3 embedding test to drive the
  real embed stage + event publisher end-to-end (asserting the written
  `F.embedding` and the published event) instead of spying on
  `_sync_accepted_candidate`; extended `test_region_rejected_candidate.py`
  and `test_region_no_verdict_cap.py` with `verified is False` assertions
  on rejected writes. **Not fixed this pass, confirmed pre-existing and
  left for W8c** (per the review's r1): a requeue from a terminal status
  to `pending_detection`/`pending_verification` still runs in "replace"
  mode against any stored boxes, including human-sourced ones -- the
  requeue boundary itself, plus the legacy-scalar readers listed in the
  original W8 pipeline review, remain W8c scope.
- **W8c pass 1 (r1 requeue fix, per-route box-state validation,
  segmenter `min_score`/128-candidate config).** Partial pass -- the
  legacy scalar field/route deletion and the per-box embeddings/
  clustering/FP-matching work this wave was scoped for are NOT done
  this pass; see the handback report for the exact remaining list.
  - **Requeue query ported off the deleted-in-spirit legacy scalars.**
    `region_requeue.requeue_query` filtered on item-level `F.detector` /
    `F.rejection_reason` / `F.bbox_norm`, none of which a W8 worker
    write ever populates -- so requeueing a W8-written cohort by
    detector/reason, or to `pending_verification`, silently matched
    nothing. Now a nested `region_boxes` query
    (`region_boxes.box_query`/`has_any_box_query`); `requeue_breakdown`'s
    aggregation is a nested agg over the same path (counts BOXES, not
    items, for the by-detector/by-reason breakdown only -- `total`
    still counts items).
  - **r1 wipe-on-replace fix, both ends (CORRECTED below -- see "W8c
    slice 1 fix pass"; this bullet's "never a human's" claim only ever
    covered a human-CREATED box, `source == 'human'`, not a human's
    per-box accept/reject VERDICT on a machine-created box, which this
    pass's own edit routes leave no trace of -- that gap, M3, is closed
    by the later pass, not this one).** `region_requeue.apply_requeue`
    with `clear_detection=True` drops only this pass's MACHINE-sourced
    boxes from `region_boxes` (`source != 'human'`). Separately (and
    required regardless of the requeue tool, since a fresh-detection pass
    can also follow a raw/direct status edit): every fresh-detection path
    in the streaming worker (`runner.py` Path 2/Path 3 and the text-hint
    re-pass they can fall into) now sets `_ItemTask.pending_merge = True`,
    so its own candidates MERGE onto whatever is live at write time
    (`region_boxes.merge_boxes_for_write`, the same primitive Path 1's
    B1 fix already uses) instead of replacing the box list wholesale --
    a no-op for the common case (no stored boxes at all). **This
    "merge" semantic for a fresh detection was itself wrong (M1) and is
    replaced by the later pass below.** Red-then-green on
    `tests/curation/test_region_write_occ_m1.py` (a concurrently
    human-added box now survives a fresh-detection write instead of
    being silently discarded) and a new
    `query_fakes.py` `nested` query/aggregation double (query + agg;
    additive, every existing test unaffected).
  - **Per-route box `state` validation (prior-pass gap).**
    `BOX_STATE_ROUTES` (`src/config/region_state.py`) was served on `GET
    .../regions/statuses` but never enforced on write. New
    `region_boxes.validate_box_state(route, state)`, wired into all four
    W8a box-write routes (`PUT /crops/{id}/regions`, `PUT
    /crops/batch_regions`, `PATCH /crops/{id}/regions/{box_id}`, `POST
    /regions/batch_box_state`) -- an unrecognized `state` is now a 422
    instead of being written verbatim into `region_boxes`.
  - **Segmenter `min_score` + the 128-candidate ceiling
    (`docker/segmenter/`).** Verified against the upstream source (no
    `docker run`; the `sam3` package isn't installed in this
    environment, so this is a read of the public
    `facebookresearch/sam3` GitHub source, not a live probe):
    `Sam3Processor.__init__(..., confidence_threshold=0.5)` is SAM 3's
    only score threshold, read as a plain instance attribute by both the
    upstream `_forward_grounding` and this repo's mask-disabled patch;
    `build_sam3_image_model`'s decoder defaults to `num_queries=200`, so
    a 128-candidate top-K ceiling never asks for more than the model can
    produce. `SegmentRequest`/`BatchSegmentRequest` gain `min_score:
    float | None` (sent to the server; `None` = the processor default);
    `max_candidates`'s `le` rises from 32 to 128
    (`sam3_backend.MAX_CANDIDATES_CAP`). `ProcessorPool.acquire(
    min_score=...)` sets the leased processor's `confidence_threshold`
    for just that call and restores the prior value on release (even on
    error), so no per-instance state leaks between callers. `GET
    /health` now serves `max_candidates` and `default_min_score`. Scope
    note: this is the segmenter-server half of `any_domain_plan.md`
    W8.4 only -- the client-side half (`DetectionProfile.
    segmenter_min_score`, `SegmenterClient` sending `min_score`,
    `region_candidates.py`, the detector-leg port) is separate, larger
    W8.4 scope and is NOT done this pass.
  New/extended tests: `tests/curation/test_region_requeue.py` (rewritten
  onto a `region_boxes`-shaped corpus), `tests/curation/query_fakes.py`
  (`nested` query/agg support), `tests/curation/test_region_write_occ_m1.py`,
  `tests/curation/test_region_no_verdict_cap.py` (one scenario's manual
  requeue helper updated to clear `region_boxes`, matching what the real
  `apply_requeue(clear_detection=True)` now does), `tests/curation/
  test_regions_boxes_edit.py` (4 new per-route invalid-`state` tests),
  `tests/curation/test_segmenter_service.py` (2 new: `min_score`
  filters-and-restores, `max_candidates` at/over the new cap).
- **W8c slice 1 fix pass (independent Opus review response, 2026-09-28):
  reverify-vs-fresh-detection semantics, requeue-to-pending_verification,
  human-touch protection scope, empty-batch validation, requeue breakdown
  reconciliation.** Fixes every finding in
  `docs/design/openprocessor_internal/w8c_slice1_review_2026-09-28.md`
  (1 blocker, 3 majors, 1 minor, 1 nit).
  - **B1 (blocker) + M1: `pending_merge` was overloaded across two
    unrelated meanings.** The prior pass's `_ItemTask.pending_merge = True`
    on every fresh-detection task (see the corrected bullet above) was
    also read by `runner.py`'s `region_visible=False` combined-verify
    branch to mean "this is Path 1's re-verify" -- so a FRESH item (no
    stored boxes) that got a not-visible VLM reply was misrouted into the
    re-verify branch and wrote a phantom `rejected` box with the
    misleading reason `region_visible_elsewhere` instead of the correct
    empty `no_region_visible`. The dev/test stack's `fake_vlm` defaults to
    `region_visible=False`, so this was not an edge case. Separately (M1),
    "merge" was the wrong semantic for a fresh detection in the first
    place: a fresh detection is a new answer to "where are the regions?",
    but merging let a stale MACHINE box from a prior pass (left behind by
    the documented default `clear_detection=False` requeue) accumulate
    forever and keep overriding the new pass's own derived status (e.g. a
    requeued `detection_failed` item that now finds nothing landed in
    `verify_rejected` from the stale box instead of `no_region_box`).
    Fix: a new `_ItemTask.reverify: bool = False` (`state.py`), set ONLY
    at Path 1 (`runner.py`, next to `pending_merge = True`), and read at
    the not-visible branch instead of `pending_merge`. `bulk_writer._merge`
    now branches three ways: `reverify` -> `merge_boxes_for_write`
    (unchanged Path 1 behavior); `pending_merge` (fresh detection) -> keep
    only stored boxes `region_boxes.is_human_owned` recognizes and REPLACE
    every machine-sourced one with this pass's own findings; neither ->
    plain replace (unchanged for every other write path, e.g.
    `accept_without_vlm`'s sanity-reject branch). Red-then-green: reverted
    `runner.py`'s `if t.reverify:` back to `if t.pending_merge:` turns
    `test_region_fresh_detection_replaces_machine_b1_m1.py`'s B1 test red;
    reverted `bulk_writer._merge`'s `pending_merge` branch back to
    `merge_boxes_for_write` turns 3 of that file's 4 tests red.
  - **M2: the ported `pending_verification` requeue query selected items
    with nothing to re-verify.** A `REQUEUEABLE_STATUSES` item's box(es)
    are always `rejected` (never `proposed` -- `derive_status` would
    already report `pending_verification` if one were), so the worker's
    Path 1 guard (`proposed_stored or t.detector_region_in_source is not
    None`) always failed for a requeued item and it silently ran a fresh
    detection pass instead -- the documented operator action ("Re-verify
    boxes the previous verify prompt rejected") did something else
    entirely. Fix: `apply_requeue`, for `target=pending_verification`,
    now rewrites each non-human `rejected` box to `state='proposed',
    rejection_reason=None` before flipping the item's status; an item
    left with no `proposed` box after that (its only box(es) are
    human-owned, or it has none at all) is skipped rather than moved to a
    mismatched `pending_verification` with nothing to re-verify. The
    false claim at `runner.py` ("reaching this line already proved
    `region_status` is a pending_detection alias") is corrected and now
    logs `region_pending_verification_fallthrough` if that invariant is
    ever violated (stale/pre-fix data, a direct write). Red-then-green:
    `tests/curation/test_region_requeue.py` (3 tests) and a new
    `tests/curation/test_region_requeue_then_worker_m2.py` (requeue, then
    drive the real streaming worker end-to-end and confirm Path 1 runs,
    not a fresh detection) all go red against the un-fixed query/merge.
  - **M3: `clear_detection` (and the fresh-detection replace above) only
    protected human-CREATED boxes, not a human's accept/reject VERDICT on
    a machine-created box.** The W8a per-box edit routes (`PATCH
    /crops/{id}/regions/{box_id}`, `POST /regions/batch_box_state`, and a
    `PUT .../regions` patch of an existing box) change `state` but left
    `source`/`detector` exactly as the machine wrote them, so
    `source == 'human'` alone never recognized the human's action.
    Fix: those three write paths now also stamp
    `rejection_reason=REJECT_REASON_HUMAN` when the human sets `state=
    'rejected'` (matching `boxes_with_status`'s whole-set path); new
    `region_boxes.is_human_owned(box)` (`source == 'human' or
    rejection_reason == REJECT_REASON_HUMAN or text_source == 'human'`)
    is now the shared criterion both `apply_requeue`'s `clear_detection`
    box-drop and `bulk_writer._merge`'s fresh-detection replace use,
    replacing the narrower `source == 'human'` check both had. The
    CHANGELOG wording above is corrected accordingly -- "never a human's"
    was only ever true for a human-created box.
  - **m4: an empty batch skipped validation on two routes.**
    `batch_set_crop_regions` (`PUT /crops/batch_regions`) and
    `batch_set_region_box_state` (`POST /regions/batch_box_state`) both
    returned 200 for an empty `crop_ids`/`targets` list before their
    `_check_box_states`/`validate_box_state` call ever ran, so a bogus
    `state` alongside an empty target list was silently accepted. Fix:
    validation now runs unconditionally, before the empty-input early
    return (which still short-circuits to a 0-updated no-op once
    validation passes).
  - **Nit: the requeue breakdown's numbers didn't reconcile with its own
    item total, and a boxless cohort rendered an empty breakdown.** A
    nested aggregation has no element to bucket a zero-box item under (not
    even `NONE_BUCKET`), so a cohort of entirely `no_region_box`/
    `no_region_visible`/unseeded items showed nothing where the
    pre-nested-query version showed an explicit `(none)` bucket -- a real
    regression, now fixed with a new `no_box` field (a sibling, non-nested
    `filter` aggregation counting ITEMS, so `total - no_box` is exactly
    the item count with at least one box). The by-detector/by-reason
    buckets still count BOXES and can still exceed `total - no_box` for a
    genuinely multi-box item with boxes in different buckets -- documented
    in `requeue_breakdown`'s docstring as inherent to box-level bucketing
    (an exact per-bucket item count would need `reverse_nested`, not worth
    the complexity for a dry-run report), not a bug. The CLI's dry-run
    header is relabeled "items selected" (was "regions selected", which
    read as boxes given the per-detector/reason lines directly below it).
  New/changed tests: `tests/curation/test_region_fresh_detection_replaces_
  machine_b1_m1.py` (new -- B1, M1 stale-machine-replace, and the combined
  human+machine case), `tests/curation/test_region_requeue_then_worker_m2.py`
  (new), `tests/curation/test_region_requeue.py` (3 new + 1 extended +
  `no_box`/relabel assertions), `tests/curation/test_region_boxes.py`
  (`is_human_owned` unit tests + an `apply_put_boxes` reject-stamp test),
  `tests/curation/test_regions_boxes_edit.py` (6 new: 2 empty-batch
  validation, 2 empty-batch no-op, 2 human-reject-reason stamping),
  `tests/curation/test_region_write_occ_m1.py` (seeded box now carries
  `source='human'`, matching what a real concurrent PUT stamps -- the
  fresh-detection fix now keys survival off that, not off being merely
  unrecognized as machine-owned).
- **W2b-finish: independent re-verification of the Opus review fix pass
  (2026-09-27), plus merging in W2's reviewed config-store hot reload.**
  Merged `main` (W2 config store hot reload, `op_global_configs`, P3F
  finish passes 2-4, P3 review fixes) into `cutover/keymap`; confirmed
  `scripts/curation/worker/{runtime.py,runner.py}` now match `main`
  exactly (no diff), so this branch inherits W2's real per-project
  hot-reload worker unchanged. Then independently reproduced every one
  of `w2b_review_2026-09-27.md`'s own probes against commit `f418a357`
  ("W2b Opus review fixes -- B1/B2/B3 blockers + majors") instead of
  trusting its commit message -- B1 (atomic unbind-then-save
  rollback), B2 (all-or-nothing keymap clone), B3
  (`get_active_region_profile()`-backed `available`), M1 (reserved-
  hotkey `actions[]` from the project's own overrides), M2 (canonical
  combo grammar, rejects `shift+ctrl+z`/`ctrl+ctrl+z`/`alt+meta+x`), M3
  (`keymap_class_hotkey_conflict` as a real `ValidationIssue`, folded
  into `POST /keymap/validate`'s `ok`), M4 (validate uses PUT's replace
  semantics), M5 (`If-Match: "keymap:N"` accepted), M6 (`save_keymap_doc`
  calls the real atomic `bump_config_revision`, no parallel bump path)
  and M7 (`test_put_keymap_422_on_collision` implemented, not a `pass`
  stub; `keymap_combo_invalid`/`keymap_focus_key`/
  `keymap_class_hotkey_conflict` all have real validator tests; the
  alpha/beta keymap and class-hotkey isolation probes are permanent
  `leak_env` tests) were all genuinely fixed, not just claimed. Found
  and fixed two real gaps the merge with `main` surfaced that the
  original fix pass's narrower test run never caught:
  - `tests/projects/conftest.py`'s `FakeLifecycleOpenSearch.update()`
    (used by the project-lifecycle/clone unit tests, a different fake
    from `tests/curation/_fake_config_opensearch.py`'s
    `FakeConfigOpenSearch`) had never been taught M6's painless-script
    `bump_config_revision` shape, so any `create_project(...,
    clone_settings_from=...)` that clones the `keymap` axis raised
    `TypeError('unexpected keyword argument retry_on_conflict')`. Taught
    it the same real create-vs-bump distinction the other fake already
    had (see M6's own guidance: the fix belongs in the fakes, not a
    parallel production path). Red-then-green:
    `test_create_with_bad_clone_source_burns_no_slug_and_leaves_no_indexes`
    failed with that `TypeError` before this fix.
  - `tests/projects/test_clone_settings.py`'s
    `test_clone_keymap_axis_copies_overrides_and_reports_class_conflicts`
    asserted the pre-B2 partial-copy behavior (drop only the
    conflicting action, copy the rest) -- stale since B2's fix (landed
    in the same `f418a357` commit) made the axis all-or-nothing.
    Renamed to `test_clone_keymap_axis_all_or_nothing_reports_class_conflicts`,
    fixed its assertion, and added a sibling
    `test_clone_keymap_axis_copies_overrides_when_no_conflicts` for the
    clean-copy case its old docstring described but never actually
    exercised post-B2.
  - Resolved the `main` merge's textual conflicts (`CHANGELOG.md` --
    both sides' distinct entries kept; `contracts/openapi/curation.json`
    and `_config_common_models.py`'s shared `ErrorCode` Literal --
    both sides' new codes kept; `projects.py`'s `clone_settings_route`
    -- now both publishes `project.updated` (`main`) and returns
    `keymap_clone_conflicts` (this branch); `test_cross_project_leak.py`
    -- kept `main`'s more correct fake `_update` handler, which
    distinguishes a fresh upsert from an existing doc's script-driven
    increment matching real OpenSearch upsert semantics, over this
    branch's own less-correct version; `test_project_lifecycle_routes.py`
    -- kept both branches' new tests).
- **Review focus-item #5: create-time keymap clone conflicts are now a
  `ProjectWarning`, not just a log line.** `create_project`'s
  `clone_settings_from` clone could silently drop a keymap override on
  a class-hotkey conflict, logging
  `project_create_keymap_clone_conflicts` but leaving the 201 response's
  `warnings` empty (`ProjectLifecycleResponse.keymap_clone_conflicts` is
  the standalone `POST clone_settings` route's field, always `[]` on
  create). `create_project` now appends one `ProjectWarning` (code
  `keymap_clone_conflict`) per dropped conflict to the `(record,
  warnings)` pair every caller already unpacks. Red-then-green:
  `test_create_project_surfaces_keymap_clone_conflicts_as_warnings`
  (new) failed with an empty `warnings` list before this change.
- **W2-finish review fix-on-fix pass (2026-09-27), including a
  fix-on-fix confirmation re-review.** Addresses the independent review
  of the W2-finish pass (`w2_finish_review_2026-09-27.md`), which came
  back MERGE AFTER FIXES on commits `8f472157`/`707e7ee7` (MJ1, MJ2, m1,
  m2, m3 below), and the follow-up re-review at `bbc82fe8` (range
  `707e7ee7..bbc82fe8`), which confirmed all five of those closed and
  found one new major left over from MJ1 (MJ3, below) before returning
  MERGE:
  - **MJ1** (major): the config-store poll task never actually started
    in production. `startup_bootstrap_config_store_safe()`
    (`src/services/config_store/store.py`) used to also call
    `get_config_store(mode='live')` + `store.refresh(client)` to "warm
    the bound project's store" -- but `src.main`'s lifespan runs
    unbound by design, so that call always raised `ProjectNotBound`,
    which the function's own broad `except` swallowed, silently
    returning `None` (no poll task) on every real deployment. Dropped
    the unreachable warm-up; `ensure_global_configs_index` still runs
    first, and the function now returns
    `_poll_all_active_projects(...)`'s task directly -- its own first
    tick binds and refreshes every active project. Red-then-green:
    `test_startup_bootstrap_config_store_safe_starts_poll_task_unbound`
    (new, `tests/curation/test_config_store.py`) reproduced the
    reviewer's exact probe (`assert task is not None` failed with
    `config_store_bootstrap_skipped` logged) against the unfixed code.
  - **MJ3** (major, found in the re-review of this same pass; fix-on-fix):
    MJ1's fix still ran `ensure_global_configs_index` inline before
    `create_task`, inside a broad `except` that returned `None` on any
    failure there -- an unreachable OpenSearch at startup, or a lost
    index-create race surfacing as `resource_already_exists_exception`
    before `indices.exists` sees the winner's index. Both are realistic
    on a cold stack start (production `yolo-api` has no
    `opensearch: service_healthy` gate ahead of its 32 workers), and
    nothing ever retried, so that worker had no poll task for its entire
    lifetime -- M4 stayed inert there, contrary to what MJ1's own
    docstring claimed ("`_poll_all_active_projects`'s own first tick
    still runs once OpenSearch recovers"). Fixed by mirroring
    `src.services.projects.bootstrap.startup_bootstrap_project_registry_safe`'s
    retry-then-poll shape: `startup_bootstrap_config_store_safe()` now
    always returns a real task -- it tries the index-ensure once inline,
    and on any failure hands off to a task that retries with backoff
    (1s, doubling to a 30s cap) until it succeeds, then runs
    `_poll_all_active_projects` forever. Corrected the function's
    docstring to state this plainly instead of asserting a recovery path
    that didn't exist. Red-then-green:
    `test_startup_bootstrap_config_store_safe_retries_until_opensearch_reachable`
    (new, `tests/curation/test_config_store.py`) makes `indices.exists`
    raise a connection error on the first 3 calls, then succeed --
    failed with `assert task is not None` against the unfixed code
    (`None`, no task), passed once the retry-then-poll wrapper landed.
  - **MJ2** (major): the new unprefixed `op_global_configs` index broke
    the live verify harness's `verify_`-prefix safety guard
    (`tests/live/conftest.py`'s `harness_safety_guard`). The code
    already read the index name from `OP_GLOBAL_CONFIGS_INDEX`
    (`global_configs_index()`, `store.py`) rather than hardcoding it, so
    this was purely a compose-file gap: `docker/test/compose.yml` now
    sets `OP_GLOBAL_CONFIGS_INDEX=verify_global_configs` alongside the
    existing `OP_PROJECTS_INDEX=verify_projects`, and
    `tests/live/conftest.py`'s `INDEXES` map gained a `global_configs`
    entry (a hardcoded `verify_global_configs` literal, so its own
    static `verify_`-prefix check always passes and does not itself
    verify the compose file and the map agree -- the real protection is
    the harness's live stray-index scan against `_cat/indices`, which
    does check the two agree).
  - **m1** (minor): the guard test for `op_global_configs` isolation
    (`tests/projects/test_opensearch_guard.py::test_global_configs_index_is_a_legitimate_unowned_index`)
    now builds its URLs from the real `global_configs_index()` resolver
    instead of a hardcoded `'op_global_configs'` literal, so it would
    catch the index resolving into a project's own namespace. Verified:
    mutating `global_configs_index()`'s default to
    `op_prj_default__configs` now turns this test red (it previously
    stayed green against the hardcoded literal).
  - **m3** (minor): minor 5's four `vlm_called = True` call sites had no
    test coverage beyond the bulk-writer gate test -- removing all of
    them left the full suite's pass/fail outcome unchanged except for
    that one test. Added `tests/curation/test_vlm_called_call_sites.py`
    with one test per path (the cascade verify path via
    `verify.py::_verify_with_vlm`, the combined single-crop path via
    `combined.py::_try_combined_class_region`, the Stage A visibility
    batch and the Stage B combined batch, both in `runner.py`) plus one
    negative test (the high-confidence secondary-segmenter auto-skip
    path must NOT set `vlm_called`). Red-then-green: removing the four
    production `vlm_called = True` assignments turned the four positive
    tests red while the negative test and the pre-existing gate test
    stayed green, confirming the new tests close the gap the reviewer
    found.
  - **m2** (minor, documented not fixed -- W9's call): reading the
    global store while a project is bound silently degrades to an
    empty, stale snapshot (the project guard refuses the I/O; `refresh`
    treats that like any other failure). Added a docstring note on
    `get_global_config_store()` making this explicit, per the review's
    guidance that the actual read-while-bound rule is W9's decision, not
    this pass's.
- **W2-finish minors pass (2026-09-27).** Closes 4 of the W2 review's 7
  minors the prior fix pass left open or didn't fully close (`w2_review_2026-09-27.md`):
  - **Minor 2** (G1 copied the current body, not the activated
    revision): `_clone_activations` (`src/services/projects/clone.py`)
    now reads `config_doc_id(kind, name, activation['revision'])` --
    the immutable revision copy that was actually active -- instead of
    `config_doc_id(kind, name)` (whatever the source has saved since,
    which diverges once a pack/profile is saved again after being
    activated). Red-then-green:
    `test_clone_activations_copies_the_activated_revision_not_the_current_body`
    (new) failed with the stale body (`{'v': 2}` instead of `{'v': 1}`)
    against the unfixed code.
  - **Minor 3** (the `except (NotFoundError, KeyError)` test-double
    accommodation in `clone.py`): re-assessed and fixed, reversing the
    prior pass's "too risky" call. `tests/projects/conftest.py`'s
    `FakeLifecycleOpenSearch.get()` now raises `NotFoundError` on a
    missing doc like the real client (and like
    `tests/curation/_fake_config_opensearch.py`'s `FakeConfigOpenSearch`
    already did) instead of returning a `{'found': False}` body --
    contained, because every production caller of `client.get()` in
    `registry.py`/`bootstrap.py` already handles BOTH shapes
    defensively (checked by running the full `tests/projects/` suite,
    367 passed, after the fake change alone). `clone.py`'s three
    `except (NotFoundError, KeyError)` sites are now real-404-only:
    two collapse entirely (`get_activation` already maps a 404 to
    `None` itself, so nothing there could still raise), the third
    (a raw `client.get()` for the activated revision copy) keeps
    `except NotFoundError`, dropping `KeyError`. Verified by the full
    `tests/projects/` suite (367 passed) and `tests/projects/test_clone_activations.py`
    (5 passed) after the change.
  - **Minor 4** (`GET /settings` hid `detection_profile: off`):
    `_config_store_axis_defaults` (`src/routers/curation/settings.py`)
    now reports `'off'` explicitly instead of skipping the axis --
    "never activated" (absent from `defaults`) and "explicitly turned
    off" (`'off'`) are distinct per the store's own `AxisRef` docstring.
    Applies to both config-store axes uniformly (`prompt_pack`'s `null`
    deactivation surfaces as `'off'` too, not just `detection_profile`'s).
    Red-then-green:
    `test_put_detection_profile_off_and_on`/`test_put_prompt_pack_null_deactivates`
    (updated) failed against the unfixed code (`'detection_profile' not in
    defaults` / `'prompt_pack' not in defaults` no longer held once the
    assertions were flipped to expect `'off'`).
  - **Minor 5** (the worker stamped `vlm_prompt_pack` on every region
    write, even when no VLM call contributed to it): `_ItemTask`
    (`scripts/curation/worker/state.py`) gains `vlm_called: bool = False`,
    set at every point a VLM call actually ran for that task this pass
    (`verify.py::_verify_with_vlm`, `combined.py`'s combined-cohort call,
    and `runner.py`'s two batched VLM stages -- call-site-only diffs per
    non-negotiable 8). `bulk_writer.py`'s `_merge` now gates the
    `vlm_prompt_pack` stamp on the per-task `task.vlm_called`, not just
    the per-batch resolved pack -- a deployment with no VLM configured,
    or a write path that skipped the VLM (e.g. the high-confidence
    secondary-segmenter auto-skip), no longer claims a VLM ran.
    Red-then-green: `test_bulk_write_stamps_region_profile_and_pack`
    (existing) failed with `KeyError: 'vlm_prompt_pack'` once the gate
    landed, until updated to set `task.vlm_called = True`; new
    `test_bulk_write_does_not_stamp_pack_when_no_vlm_call_happened`
    covers the previously-missing case.
  - **Minor 6** (`registry_reclassify.py`'s docstring/code mismatch):
    the docstring said the default pack is "the active pack"; the code
    called `resolve_prompt_pack` (env/file default only, never
    store-aware). Now calls `active_prompt_pack`, matching the
    docstring and W2's B4 fix elsewhere. Red-then-green:
    `test_default_pack_resolves_through_the_store_not_the_env_file_default`
    (new) failed (`active_prompt_pack` never consulted) against the
    unfixed code.
  - **Minor 7** (the leak sweep's `_FakeTransport` couldn't run the
    painless `bump_config_revision` script, so it had no real coverage
    of the config-store write path): re-assessed and fixed.
    `tests/curation/test_cross_project_leak.py`'s `route_bodies()` now
    puts a config-store axis (`prompt_pack: GENERIC_ITEM_PACK.name`) in
    `PUT /settings`'s body instead of an empty `defaults`, and
    `_FakeTransport`'s `_update` action now models the real
    create-with-upsert-vs-script-bump distinction (mirroring
    `_fake_config_opensearch.py`'s `FakeConfigOpenSearch.update`).
    Red-then-green: `test_every_scoped_route_stays_inside_the_bound_project`
    failed with `500 ... KeyError` for both project orderings against
    the unfixed fake, once the route body change alone landed.
  - **Not addressed, left as documented (minor 1):** `name@rev` pinning
    is still a no-op -- explicitly deferred to W3 in the original W2
    commit message; out of scope for this pass per the brief.
- **W2 review fix pass (2026-09-27).** Addresses the independent W2
  review's 5 blockers and 7 majors (`w2_review_2026-09-27.md`):
  - **B1** the real worker never held a runtime per project. The
    producer loop now iterates `project_registry.active_projects()`
    every cycle, binding each in turn (`_sync_project_runtime`) so each
    project's `ConfigStore` and `RegionRuntime` are built/refreshed
    under that project's own context. `ProjectNotBound` is never
    suppressed -- store creation only ever happens inside a real
    `bind_project(record)`. Every per-item stage (`stage_a_consumer`,
    `stage_a_vlm_visible`, `stage_a_sam_consumer`, `stage_b_combined`)
    now resolves `rt = _rt_for(t)` (raises
    `RegionProfileNotConfiguredError`, caught by the stage's own
    exception handler as a drop-and-retry, for a project with no
    runtime yet) instead of reading process-wide `detector`/`segmenter`/
    `ocr_recognizer`/`vlm`/`profile`/`pack`/`text_rules` closure
    variables. `RegionDetector`/`PaddleOcrTextRecognizer`/
    `SegmenterClient`/`VlmLabeler` are passed into `build_runtime` as
    parameters (never imported fresh), so a real
    `test_two_project_worker_alpha_activation_swaps_alpha_only` test
    drives `worker.run()` with two real projects end to end and
    confirms alpha's detector-construction count increases on an
    alpha-only activation while beta's stays put.
  - **B2** pinned mode could never swap past the first cycle
    (`refresh()` only staged `pending_snapshot`; the check compared
    `store.current`, which pin_active() alone moves). `maybe_hot_reload`
    now reads `pending_snapshot or current`, and `quiesce_and_swap`
    pins strictly between the drain and the build.
  - **B3** the store silently ended up in `live` mode in a `--project`
    deployment (an earlier default-mode `get_config_store()` call
    stuck). The store is now always created pinned inside
    `_sync_project_runtime`, the first thing to touch it for a given
    project. The writer's `out_q.task_done()` no longer fires at
    dequeue time -- it fires once per item inside `_flush()`, only
    after that item's write actually completed, so `quiesce_and_swap`'s
    drain (`out_q.join()`) is now a real guarantee that every
    old-runtime item is durably written (and stamped with the store
    state that was current when it was flushed) before the swap
    proceeds.
  - **B4** a pack activation rebuilt with the env/file pack
    (`resolve_prompt_pack`) instead of the activated one. Both the
    initial build and every swap now resolve via `active_prompt_pack`.
    `profile_revision`/`pack_revision` are computed from the resolved
    object's own name matching the activation ref (`_revision_for`), so
    a fallback to the env default never inherits a stale revision.
  - **B5** the snapshot could pair a fresh revision (read via realtime
    `GET`) with a stale near-real-time `_search`. `_load_snapshot` and
    `_next_revision` now force `indices.refresh(index)` before
    searching. `tests/curation/_fake_config_opensearch.py` gained
    `NearRealTimeConfigOpenSearch`, a fake that actually models the lag,
    reproducing the reviewer's probe #8 as a real red-then-green test.
  - **M1** `runtime:detection_worker:<host>` docs are now written (once
    per project, throttled to 60s, immediately on the first sync of a
    project) via `upsert_project_runtime_doc`.
  - **M4** the API's background poll loop (`_poll_all_active_projects`)
    now fans out over every active project's own store each tick,
    not just the one bound at lifespan startup.
  - **M5** `clone_settings`'s `activations` axis now refuses a target
    that already has its own active pack/profile up front
    (`target_not_empty`, before any write), and additionally maps a
    `RevisionConflictError`/`ActiveConflictError` from the write itself
    to a structured 409 as defense in depth.
  - **M6** the settings bridge now resolves a stored pack/profile's own
    current revision before activating it (never `None` for a real
    stored config), and `_axis_ref` no longer coerces a genuinely-`None`
    revision (an env/file id) to `0` -- two processes reading the same
    activation now agree on its revision.
  - **M7** `text_hint_on` (plus `vlm_available`, `item_text_enabled`,
    `item_text_min_conf`) moved onto `RegionRuntime`, computed fresh in
    `build_runtime` from the runtime's OWN profile/segmenter -- a swap
    to a profile with different text-hint settings no longer keeps the
    old gate.
  - **Cropwright W3 UI (C2/Q5).** `ActiveConfigResponse` gained
    `source` (`'stored' | 'env' | 'off'`), `activated_at` and
    `applied: list[AppliedRuntime]` per any_domain_plan.md §7.2/§7.3
    (`AppliedRuntime` is new). No route serves this yet (W3/W4 land the
    CRUD routes); this is the shared model Cropwright's contract already
    expects. Contracts regenerated.
  - **Not done, recorded as remaining work (M3): since closed.** At the
    time of this pass, the `op_global_configs` global store (for W9
    endpoints / `local_vlm:desired`) did not exist yet and
    `ConfigStore`/`get_config_store` were project-scoped only. Built by
    W2 review M3 (2026-09-27) -- see the `op_global_configs` entry under
    Added, above.
  - **Minors not addressed: four of five since closed.** At the time of
    this pass: `name@rev` pinning was still a no-op; `GET /settings`
    hid `detection_profile: off`; the worker stamped `vlm_prompt_pack`
    even when no VLM ran; `registry_reclassify.py` had a docstring/
    behavior mismatch; and `_clone_activations`'s
    `except (NotFoundError, KeyError)` test-double accommodation was
    unchanged. The W2-finish minors pass (above) closed all of these
    except the first: Minor 4 (`detection_profile: off`), Minor 5
    (`vlm_prompt_pack` gating), Minor 6 (`registry_reclassify.py`
    docstring) and Minor 3 (the `except` accommodation, reversing this
    pass's "too risky" call). **Still open, deferred to W3 by design:**
    `name@rev` pinning remains a no-op.
- **Docs site: one-line installer.** New `getting-started/installer` page
  (tiers, `--unattended`, verifying `SHA256SUMS` with integrity-not-
  authenticity wording, the Cropwright LAN default and `--local-only`,
  OpenSearch heap sizing, upgrade / rollback / uninstall, exit codes).
  `quick-start` now leads with the installer, with install-from-source below;
  `deployment/security` covers LAN access.
- **Installer docs.** README Quick Start is now the one-line installer
  (tiers, `--unattended`, verifying `SHA256SUMS`, the LAN/Cropwright
  network decision), with "Install from source" below it. `INSTALLATION.md`
  documents every installer flag and consent variable, upgrade / repair /
  rollback / uninstall, offline `--release-dir` bundles, OpenSearch heap
  sizing and troubleshooting by exit code. `SECURITY.md` states that release
  checksums prove integrity, not authenticity. Static tests pin the network
  wording and that every `--help` flag is documented.

### Changed
- **OpenSearch heap is sized from host RAM in one place.** New
  `scripts/lib/opensearch_heap.sh` (`opensearch_heap_for_host`: RAM/8,
  clamped to 1-8 GB; `opensearch_shard_budget`) is used by both
  `setup-openprocessor.sh` and `scripts/lib/config.sh`. `config.sh` no
  longer takes the heap from the GPU profile, keeps an `OPENSEARCH_HEAP`
  the user already set on a forced regeneration, and its compose override
  interpolates `${OPENSEARCH_HEAP}` instead of a baked value. The
  installer summary prints the heap and the soft shard budget (heap GB x
  `OP_SHARDS_PER_HEAP_GB`, new advanced knob, default 20).

### Fixed
- **P3F finish pass 4 (2026-09-27).** Closes the pass-3 confirmation
  re-review's two remaining small items (F1, F2) plus a nit (n-f):
  - **F1 (the important one -- a genuine data-loss bug under the real
    production topology)**: pass 3's `delete._FINISH_IN_PROGRESS` guard
    is per-WORKER-PROCESS only, and `yolo-api` runs `--workers=32`. A
    retried DELETE that lands on a different worker (31 times out of 32
    in production) had its own, empty copy of that guard and could not
    see that a finish for the same slug was already running elsewhere.
    The review's cross-worker probe showed the exact failure: finish A
    timed out on its drain wait and rolled the record back to `active`,
    while finish B -- on the simulated second worker, unaware of A --
    went on to unload the project's models and delete all 7 of its
    indexes anyway, refusing only at the very last step (the tombstone
    write), by which point the data was already gone. Fix:
    `delete_project_finish` now claims exclusive ownership of the finish
    with a real cross-process primitive, right after the drain succeeds
    and before the first irreversible step (model unload) --
    `_refetch_for_write(expect_status='deleting')` followed by an
    OCC-guarded `write_record`. A peer finish that already moved the
    record off `deleting` (e.g. a sibling's rollback) makes this claim
    raise 409 `invalid_transition` before anything destructive runs; two
    finishes whose claim reads race each other resolve via ordinary
    OpenSearch document-version OCC (`RevisionConflictError` -> 409
    `revision_conflict`). This works across all 32 worker processes
    because it is backed by OpenSearch's own document versioning, not an
    in-memory set any one process can see. The now-inaccurate
    "process-wide" wording describing the pass-3 guard (`delete.py`,
    `_FINISH_IN_PROGRESS`'s docstring, and this file's own pass-3 entry
    above) is corrected to say what it actually protects: one worker
    process, not the deployment.
  - **F2 (known gap, documented + best-effort cleanup)**: a losing
    resurrection attempt -- a create that loses the MA1
    `expect_status='building'` race on its final `active`/`failed` write
    because a concurrent stale-`building` DELETE won and tombstoned the
    slug first -- can leave up to 7 freshly created
    `op_prj_<slug>__*` indexes unreachable under a now-retired slug
    (needs a create running past `_BUILDING_STALE_SECONDS`, 120s, with a
    DELETE landing inside that exact window; rare, and it costs only
    shards, never a correctness bug). This was previously silent.
    `create_project` now logs `project_create_orphaned_after_delete`
    with the exact orphaned index names whenever this fires, and
    best-effort deletes them itself (any failure to do so is logged and
    swallowed -- this is cleanup, not a correctness path) since the
    retired slug can never own them again anyway.
  - **Nit (m5 job.json label parsing)**: `_train_job_label` (`busy.py`)
    guards against a `job.json` that is valid JSON but not an object
    (e.g. a bare list) -- it used to call `.get(...)` unconditionally and
    raise `AttributeError`, failing the busy preflight (and with it
    delete/archive) for a hand-edited or corrupted `job.json`. Now
    treated the same as missing/unreadable: falls back to the job id.

- **P3F finish pass 3 (2026-09-27).** Closes the "MERGE AFTER FIXES"
  re-review's two majors and its m-a path-escape gap:
  - **MA1**: a status-transition write now re-validates the status it
    still owns, not just the storage-level OCC token. `_refetch_for_write`
    takes an `expect_status` argument and raises 409 `invalid_transition`
    if a fresh re-read is no longer in that status -- applied to create's
    `active`/`failed` writes (`expect_status='building'`) and
    `delete_project_finish`'s drain-timeout rollback and tombstone
    (`expect_status='deleting'`). Without this, a re-read taken
    immediately before a write always has a trivially-current seq/term
    (nothing else was writing at that exact instant), so OCC alone never
    caught a slow create's late `active` write resurrecting a slug a
    stale-building delete had already tombstoned. `delete_project` itself
    now reads the record ONCE (`_get_mutable_record`) and runs every
    precondition check plus the write against that same read's seq/term,
    instead of checking against a possibly-stale registry snapshot and
    then re-reading fresh only at write time. A new `delete._FINISH_IN_PROGRESS`
    guard (plus a router-level `_BACKGROUND_DELETE_TASKS` keyed by slug)
    also ensures only one `delete_project_finish` genuinely runs to
    completion per slug at a time -- **within one worker process**. As
    pass 4 below found, this guard is per-worker-process only and does
    NOT protect across `yolo-api`'s `--workers=32`; the real cross-process
    fix landed in pass 4.
  - **MA2**: `delete_project_finish`'s model-unload step (and
    `dry_run_delete`'s `promoted_models` report) now enumerate EVERY
    model a project owns (`_owned_models`, keyed on
    `promote.json.project`), not just the `shared=True` subset
    (`_shared_model_users`, now used only for the `in_use` refusal). A
    project's own models -- private ones included, the common case --
    are always unloaded on a normal delete; `force` only bypasses the
    `in_use` 409 for the shared subset, never whether unload runs.
  - **m-a**: the delete path-escape guard (m1, previous pass) covered
    only `train_jobs_dir`/`autolabel_dir`. The other 6 of the project's
    8 dirs still accepted `path == shared_root` itself (a corrupted
    resources record pointing at the multi-project root could wipe every
    sibling project's dir tree). One guard
    (`_require_project_scoped_path`) now covers all 8, each requiring a
    strict `<shared_root>/<slug>`-rooted path, always raising
    `path_escape`. Path validation (`_validate_delete_paths`) also now
    runs as a preflight in `delete_project_finish`, before the drain
    wait and the irreversible index delete -- previously it ran only
    inside dir removal, itself after indexes were already gone, so a
    `path_escape` left the record wedged `deleting` forever.
  - **m5 (partial, from the prior pass)**: a train job's `JobRef.label`
    is now its submitted `mlflow_run_name` (read from the companion
    `<job_id>.job.json`) when one was set, falling back to the internal
    job id only when it wasn't -- documented explicitly rather than
    always silently treating the job id as a human label.

- **P3F finish pass 2 (2026-09-27).** Closes every item the P3 re-review
  still marked open (verdict FIX-FIRST):
  - **M4 retry**: a re-issued `DELETE ?confirm=<slug>` on a record already
    `deleting` (a prior finish attempt's index or model-unload step
    failed) now answers 202 and re-triggers the finish, instead of 409
    `invalid_transition`.
  - **N1**: a `building` record left by a mid-create failure no longer
    wedges forever. `create_project` catches a failure in its own initial
    `write_record` (distinguishing a genuine storage-level slug conflict,
    propagated untouched, from its own `bump_revision` failing after the
    doc landed, which now flips the record to `failed`). A delete-side
    escape hatch also allows deleting a `building` record whose
    `updated_at` is stale (>120s); a fresh one still 409s.
  - **B2(a) residual**: `create_project` now calls `registry.refresh_strict()`
    (raises) right after the `building` write, and verifies every one of
    its own indexes actually exists before ever writing `active` --
    `_ensure_indexes` is itself fail-open, so refresh_strict alone did
    not close the gap that let a live create return `active` with zero
    real indexes.
  - **M5 step 4**: `delete_project_finish` now unloads the project's own
    promoted, shared models via P2's `unload_triton_model` primitive
    (after the drain wait, before index deletion); `dry_run_delete`
    reports them in `promoted_models` instead of a hardcoded `[]`.
  - **B2(b)**: the `FakeLifecycleOpenSearch`/`_noop_ensure_indexes` test
    stub across `tests/projects/*` now really creates the bound
    project's indexes (`fake_ensure_indexes`), so `create_project`'s
    index-verification check has real state to check, and a genuine
    `indices.create` failure (simulated) is proven to still end the
    create `failed`.
  - **Minors**: m1 (the delete-path directory guard for
    `train_jobs_dir`/`autolabel_dir` was checked against a root derived
    from the same path, which could never refuse anything -- now guards
    against the real shared root with a `path_escape` refusal), m2 (every
    status-transition write rebuilds its doc from a fresh read, not a
    stale closure snapshot, so a concurrent write landing during a
    delete's up-to-60s drain wait is no longer silently discarded), m5
    (`JobRef.started_at` is now a real timestamp or `null`, never an
    always-`''` filler), m7 (the 10s capacity cache is now busted on
    every create/delete), m8 (heap sum excludes non-data nodes; the warn
    message no longer rounds 0.5 GB down to "0 GB"), m10 (dry-run index
    counts report `null`, not `0`, when uncountable). m4 (cross-document
    races on `_last_active_check`) and m12 (`GET /projects`'s per-project
    `validated_count` N+1) are documented as deferred, not fixed --
    both need infra (distributed locking; a cross-index aggregation the
    test fakes don't model) this pass does not add.
- **P3 review fix pass (2026-09-27).** Addresses the independent P3 review's
  blocker and majors:
  - M1: `POST /projects` create is now storage-OCC-safe (`op_type='create'`);
    a concurrent create of the same slug 409s `slug_taken` for the loser
    instead of silently overwriting.
  - M7 / m9: every clone-on-create refusal (unknown axis, clone into itself,
    a source that is not `active`/`archived`) now runs *before* the first
    write, so a refused clone burns no slug, creates no indexes and leaves
    no `failed` record. Clone-into-itself is a clean 422 `combine_invalid`,
    not a 500 `SameFileError`. New error code: `clone_source_not_ready`.
  - M2: a read-only bind (archived project, or a stale registry) now
    refuses every non-safe HTTP method at bind time (409 `project_archived`
    / `project_read_only`) before the route handler runs, covering
    file-backed writes the OpenSearch guard never saw.
  - M6: `tests/curation/test_cross_project_leak.py`'s `leak_env` now seeds
    the shared `op_projects` registry doc, so the leak sweep's lifecycle
    mutations (PATCH/archive/unarchive/clone_settings) exercise a real
    write behind the real guard instead of 404ing before ever reaching it;
    added a create-then-real-delete pass asserting other projects' data is
    untouched.
  - Fixed the test suite's process-wide default project registry stub
    (`tests/conftest.py`) so it is never reported `stale` by M2's new
    read-only gate -- it is a deliberately frozen, authoritative-for-tests
    snapshot, not an instance that fell behind.

### Added
- **W2 finish pass (config store worker wiring, glue G1).**
  `scripts/curation/worker/runtime.py` gains a real `build_runtime`
  (extracted, testable Triton/segmenter/OCR/VLM construction),
  `RuntimeHolder` (one `RegionRuntime` per project slug -- activating a
  profile in one project never touches another's holder entry, backed
  by the config store's existing per-slug `ConfigStore` isolation), and
  `quiesce_and_swap` (drains the given queues, then rebuilds). The
  detection worker's bulk writer (`scripts/curation/worker/bulk_writer.py`)
  now stamps `RegionFields.profile`/`profile_revision` and
  `vlm_prompt_pack` on every region write -- the 4th (worker-side) stamp
  site, alongside the three route-level ones. `CLONEABLE_AXES` gains
  `activations`: cloning a project now optionally copies the source's
  active prompt-pack/region-profile config body and activation into the
  target (`src/services/projects/clone.py::_clone_activations`).
  `tests/curation/test_worker_hot_reload.py` (new) covers
  `RuntimeHolder`/`config_wants_swap` project isolation, `build_runtime`,
  `quiesce_and_swap`, and the write-path stamping (including a
  store-activated profile's revision vs. an env-registered profile's
  `None`); `tests/projects/test_clone_activations.py` (new) covers the
  clone axis. `test_folded_roles_by_id_only.py`,
  `test_configs_mapping_union.py` (including the "6 distinct indexes"
  glue-G1 check) and `test_axis_ids.py` were already present on this
  branch and verified green.
  **Producer-loop wiring (follow-up).** `build_runtime` now takes its
  four heavy-IO constructors (`region_detector_cls`, `ocr_recognizer_cls`,
  `segmenter_cls`, `vlm_cls`) as parameters instead of importing them
  fresh, so `runner.py` passes its own module-level names
  (`RegionDetector`, `PaddleOcrTextRecognizer`) and the
  `region_worker_main` shim's (`_wkr.SegmenterClient`, `_wkr.VlmLabeler`)
  -- the exact names `test_region_worker.py`/`test_region_text_worker.py`
  monkeypatch, so a rebuild honours the patch on every call, not just
  the first. `RuntimeHolder` now tracks synced `AxisRef` pairs
  (`get_synced_refs`/`set_synced_refs`) separately from a runtime's own
  always-populated `profile_ref`/`pack_ref`, and a new
  `maybe_hot_reload(...)` is the producer loop's per-cycle check: it
  refreshes the project's store and only drains + rebuilds when the
  store's served `(active_profile, active_pack)` pair actually differs
  from what was last synced -- never on object identity, never every
  cycle. `runner.py`'s producer loop calls it once per fetch cycle,
  reassigning the `profile`/`pack`/`detector`/`segmenter`/
  `ocr_recognizer`/`text_rules`/`vlm`/`item_text_enabled` closure locals
  via `nonlocal` when it returns a runtime. `test_worker_hot_reload.py`
  gained `test_build_runtime_uses_the_passed_in_constructors_not_fresh_imports`
  and `test_maybe_hot_reload_never_swaps_when_activation_is_unchanged`
  (asserts a constructor's call count stays at 1 across three
  no-activation-change cycles). All 62 targeted worker tests
  (`test_region_worker.py`, `test_region_text_worker.py`, and five other
  worker suites) pass with the wiring live -- no monkeypatch bypass, no
  per-cycle swap spam.
- **P3 finish pass, Cropwright backend asks (2026-09-27).** `GET
  {prefix}/models/status` now serves `owned: bool` (this route's own
  ownership check, never inferred client-side from `project`) and
  `sharing_revision: int | None` (only for an owned entry -- the value
  `PUT .../sharing` needs as `expected_revision`) on every entry; a
  foreign shared entry is served `unloadable: false`. `GET {prefix}/pause`
  now also reports `paused_by: list[str]` (`'project'` / `'gpu_training'`)
  and `reason: str | None` for the global GPU/training claim
  (`gpu_arbiter.read_training_lock`); `ProjectSummary.paused` lets `GET
  /projects` render a per-row paused chip with no extra reads.
  Pause/resume now publish `project.paused` / `project.resumed` on the
  global event stream (BA-P2-1, BA-P2-2, BA-P2-4, BA-P2-5, BA-P2-7).
- **P3 finish pass, final merge.** Merged `cutover/projects-workers`
  (through `fix(projects): refresh detection-worker liveness on a
  timer`) into `cutover/projects-lifecycle`: the detection-worker
  fairness scan now re-reads its liveness file on a timer instead of
  once at process start, plus a per-project worker-runtime regression
  test. No conflicts; `cutover/projects-foundation` had not moved past
  what was already merged. Full suite (4287 passed, 5 skipped),
  pre-commit, and contract generation all verified green post-merge.
- **P3F finish pass (projects lifecycle).** `delete`/`archive`'s busy
  check now runs through `src.services.projects.busy.running_jobs`
  (§5.4's real per-project job inventory) instead of a bespoke file
  scan; the 409 `project_busy` body carries typed `JobRef` objects
  (`kind`, `kind_label`, `id`, `label`, `started_at` -- Cropwright rev-3
  delta 11), not raw ids. `ConfigErrorDetail.jobs` is now
  `list[JobRefWire] | None`.
- Delete's §5.5 shared-promoted-model guard: a project that owns a
  model opted into cross-project sharing (`promote.json.shared`) is
  refused (409 `in_use`) unless `force=true`, which proceeds and logs
  `project_delete_forced_past_shared_models` distinctly. **Known gap**:
  the *dependent project* list this returns is the project's own shared
  model names, not consumer slugs -- there is no reverse index of
  "which project actually uses model X" yet (needs the not-yet-landed
  W4 profile-CRUD wave; see the `TODO(W4/profile_validation)` already in
  `_models_sharing.py`).
- `registry.write_record` raises `RevisionConflictError` on a losing
  OCC race (`if_seq_no`/`if_primary_term` stale); `lifecycle.write_record`
  translates that into the API's 409 `revision_conflict`, so a second
  concurrent writer never silently clobbers the first.
- `ProjectRegistry.stale`: true right after the most recent
  `ensure_fresh()` failed. The request binder (`_project_deps.py`) now
  binds read-only whenever the registry is stale, not just when the
  cached status is `archived` (P1R minor 10: a project flipped to
  `deleting` while OpenSearch is flaky must not bind writable off a
  stale snapshot).
- `src/services/projects/clone.py`: `clone_settings`/`clone_settings_into`
  split out of `lifecycle.py` (700-LOC ratchet); re-exported from
  `lifecycle` for existing callers.
- `ensure_region_class()` now also runs at the end of `create_project`,
  and inside `clone.py`'s `_apply_clone` whenever `'classes'` was not
  one of the cloned axes -- `bootstrap.py`'s startup seed only ever
  covered projects that existed when the process booted, so a project
  created (or cloned without its classes) afterward had an empty
  registry until the next restart.
- Confirmed already-correct and covered with new regression tests:
  delete's 202/background-completing shape (delta 10), and the
  registry's `search_after` pagination past OpenSearch's 1000-hit
  default result window (P1R minor 4).
- `GET /curation/projects/{project}/models/status?include_other_projects=true`
  also lists other projects' promoted models whose owner shared them
  (§5.5 #3). Every Triton entry now carries `project` (owner slug, null
  for base models), `shared` and `class_mapping: {mapped_count,
  unmapped}` (null for a model with no class list); external entries
  carry `project: null, shared: false, class_mapping: null`.
- `GET /curation/projects/{project}/models/{name}/class_mapping`: the full
  name mapping of a model onto the bound project's registry (`model`,
  `model_project`, `project`, `entries[{model_id, model_name, class_id,
  class_name, match}]`, `unmapped`, `not_covered`, `labels.match`), 404
  `model_not_found` for another project's unshared model (Cropwright
  delta 8).
- **Cross-project model sharing (§5.5, owner D1).** New
  `src/services/training/model_classes.py`: `model_classes()` reads a
  model's own classes from `promote.json.classes` (model order), else
  `labels.txt`; `model_class_mapping()` matches a model's classes onto
  the *consuming* project's registry by name (exact, then
  case-insensitive) -- runs for every model, own-project included, so a
  class renamed since training shows up as unmapped there too. Never a
  raw model id crosses a project boundary.
- `PUT /curation/projects/{project}/models/{name}/sharing` -- owner-only
  opt-in/opt-out (404 for a non-owner), optimistic concurrency via
  `promote.json.sharing_revision` (409 `revision_conflict`). The
  used_by/in_use cross-project detector-usage scan is a documented
  `TODO` (needs W4's per-project `DetectionProfile` read, not merged
  here); unsharing is never refused yet.
- `POST /curation/projects/{project}/pause`, `POST .../resume`, `GET
  .../pause` -- the write side of the `pipeline_paused.flag` file
  sentinel the multi-project workers already read.
- `src/services/projects/busy.py`'s `_detection_worker_inflight` reads
  the real per-project `runtime_detection_worker_<host>.json` liveness
  files `fairness.py` writes, instead of a `[]` stub.
- `triton_promote.py`'s `promote()` now writes `promote.json.classes`
  (== `labels.txt`, model order), `shared` (preserved across a
  re-promote) and `class_remap_source`.

### Fixed
- Bake-off queue docs (`docs/CURATION.md`, `env.template`, `bakeoff.py`,
  `bakeoff_runner._pending_job_files`) described
  `$OP_STATE_DIR/bakeoff_jobs` / `OP_BAKEOFF_JOBS_DIR` as the router's
  queue. Every project, `default` included, queues in
  `$OP_STATE_DIR/projects/<slug>/bakeoff_jobs`; the evaluator must watch
  `$OP_STATE_DIR/bakeoff_jobs` on the API's state-dir path to find them.
- `PUT .../models/{name}/sharing` answers its 404s through `api_error`
  (`{"detail": {"error": "model_not_found", "message", "project", ...}}`)
  instead of a bare string detail.
- `PUT .../models/{name}/sharing`'s revision check is atomic: the
  read-compare-write of `promote.json` holds a per-model `flock`
  (`job_lock.exclusive_file_lock`, new blocking sibling of
  `exclusive_start_lock`) and writes via temp file + rename
  (`src/services/training/promote_json.py`). Two concurrent PUTs on the
  same `expected_revision` now give one 200 and one 409
  `revision_conflict` instead of two 200s. A re-promote takes the same
  lock and keeps `sharing_revision` (it used to drop it back to 1).
- `src/services/projects/busy.py`'s `running_jobs()` now reports running
  probe, item-scores, selection and viz jobs (each module's own
  `state.json` + heartbeat busy rule, read for the given project), so a
  P3 delete/archive busy check cannot pass while one runs. A detection
  worker liveness file older than the worker heartbeat window
  (`worker_liveness.DEFAULT_MAX_AGE_S`) no longer counts as busy forever.
- `/train/preflight`, `/train/start` and `/train/start_campaign` refuse a
  `dataset_export_dir` outside the bound project's `export_root` with 422
  `{"detail": {"error": "export_outside_project", ...}}`, before any check
  reads the export; `force=true` does not bypass it. Previously another
  project's manifest, registry and label counts were read back into the
  report, and `start?force=true` queued the job.
- Merged the finished `cutover/projects-foundation` (P1) twice (once
  before, once after its final review-resolution pass): resolved P1's
  worker-script conflicts in P2's favour (already multi-project) and
  P1's guard/registry/context APIs in P1's favour, per R-6.
- Adapted to P1's `default` refactor (an ordinary registered project,
  `resources_for_new`-built, no env-derived special case):
  `gpu_arbiter.py`'s train-jobs-dir/bakeoff-jobs-dir resolution, the
  many job/status/manifest/heartbeat test fixtures that assumed an
  unnested default jobs dir, and `_project_owns_model`'s one deliberate
  exception (`default`'s `model_prefix` stays empty, unlike every other
  project, so pre-projects/core-pipeline models keep resolving as
  default's own).
- `scripts/curation/worker/runner.py`'s multi-project registry
  discovery opened a second, unmockable `AsyncOpenSearch` straight from
  `--opensearch`, so every region-worker end-to-end test silently
  discovered zero projects and processed nothing; now reuses the
  already-built (patchable) client.
- `GpuArbiterConfig.bakeoff_jobs_dir`'s `default_factory` read the
  *bound* project's config, which raises `ProjectNotBound` at API
  startup (before any request binds one) and silently killed the
  reconcile-loop task; reads `default`'s own resources directly instead.
- Adapted P1's cross-project leak sweep's autolabel fixture to P2's
  actual (already per-project-function, no module constants)
  `autolabel/job.py`, and registered `preflight_scan._scan_cache` with
  the sweep's process-cache-clearing fixture.
- P2 review fixes (`projects_p2_review_2026-09-27.md`): the GPU arbiter's
  `bakeoff_active()` now sees a bake-off queued in any project (it only
  watched `default`'s dir and restarted the GPU services under another
  project's running bake-off); the API reads `.trainer_capabilities.json`
  from the trainer's watch root, not the bound project's nested jobs dir
  (where nothing writes it); `_project_owns_model` refuses a model whose
  `promote.json` names another project, so `default` cannot inherit a
  project's models when it drops out of the registry snapshot. The route
  sweep no longer exempts 14 routes from its isolation checks.

### Changed
- **Merged the three projects-lifecycle branches (checkpoint 1: merges +
  adapt only).** `cutover/projects-foundation` (P1, default-is-an-
  ordinary-project + no unscoped alias) and `cutover/projects-workers`
  (P2, multi-project workers/busy.py/fairness runner) merged into
  `cutover/projects-lifecycle` (P3, project lifecycle API). Re-added
  `registry.get_record_with_seq`/`write_record` (P1 dropped them with
  `default_project_record`; P3's lifecycle.py still needs OCC-guarded
  writes, now delegating `bump_revision` to `bootstrap.py`'s OCC
  version). `lifecycle._get_mutable_record` no longer synthesizes a
  default record from env -- a missing registry doc is a genuine 404.
  `_project_owns_model` (model unload/status) no longer assumes
  `default`'s `model_prefix` is `''`; it's `'default__'` like any
  project's, so an unprefixed name (core pipeline models, pre-project
  promotes) is owned by `default` specifically rather than by every
  project. Registered P3's five lifecycle mutations (archive/unarchive/
  clone_settings/PATCH/DELETE) in the cross-project leak sweep's
  request-body and unbound-by-design tables.
- **Triton model names are env-overridable settings, not literals**
  (`TritonModelConfig` in `src/config/settings.py`): `FACE_DETECT_MODEL`,
  `ARCFACE_MODEL`, `CLIP_IMAGE_MODEL`, `CLIP_TEXT_MODEL`, `OCR_DET_MODEL`,
  `OCR_REC_MODEL` now read from `os.environ` like `YOLO_MODEL` already
  did, plus two new fields, `OCR_PIPELINE_MODEL` and `PE_IMAGE_MODEL`/
  `PE_TEXT_MODEL`. `triton_client.py`, `fast_face_client.py`,
  `pe_encoder.py` and `scripts/curation/bakeoff/sample.py` now read these
  instead of hardcoding the model name. Documented (commented, advanced)
  in `env.template`.
- **One probe-architecture registry.** `PROBE_ARCHITECTURES` (renamed
  from the private `_PROBE_ARCHITECTURES`) in
  `src/services/curation/probe_models.py` is now the single source of
  truth, imported by `scripts/curation/run_probe.py`,
  `src/services/curation/probe_predictions.py`,
  `src/services/curation/probe_job.py` and `src/routers/curation/probe.py`
  instead of each redeclaring its own copy of the tuple.
  `start_probe_job` now rejects an unknown `architecture` immediately
  (`ValueError`, `422` at `POST /probe/run`) instead of only failing deep
  inside the background task.

### Removed
- `opensearch_heap` from the GPU profiles (`config_templates/profiles/*.json`)
  and `PROFILE_HEAP` from `scripts/lib/gpu.sh`: the heap is a host-RAM
  fact, not a GPU fact.
- **COCO special-case in class-name resolution.** `class_names.py`'s
  `_STOCK_COCO_MODEL_NAMES` fallback (borrowing COCO's vocabulary for the
  stock YOLO11 detector names if `labels.txt` was ever missing) is gone —
  both stock detectors already ship their own `labels.txt`, so this was
  dead safety net that violated the class-identity invariant (names come
  from the model's own labels, never another model's).

### Changed
- **Project lifecycle finish pass (P3 binding inputs).**
  - `DELETE /curation/projects/{project}` documents typed responses:
    200 `DeleteDryRunResponse` (dry run) and 202
    `ProjectLifecycleResponse` (accepted). `ProjectsResponse.capacity` is
    the typed `ProjectCapacityWire` (was an untyped object).
  - Every projects-route error is the typed `ApiErrorResponse`
    (`{detail: ConfigErrorDetail}`), now in the OpenAPI contract;
    `ConfigErrorDetail.capacity` is `ProjectCapacityWire`, so a 409
    `shard_budget_exceeded` carries the full capacity block on the wire.
  - `ProjectSummary` serves `archivable` (status `active`) and
    `unarchivable` (status `archived`). Archive and unarchive refuse any
    other status with 409 `invalid_transition` (new error code;
    `detail.project_status` / `detail.action` name what was refused), and
    `clone_settings` refuses a non-active target the same way.
  - `POST /projects/{project}/clone_settings` checks status, revision,
    axes, source and target emptiness before writing anything and bumps
    the revision only after the copy; a refused clone never changes the
    revision (it used to bump it first, via PATCH).
  - `counts.validated` (items with `class_validated: true`) is computed
    on every route that serves project counts (list, get, lifecycle
    envelopes, `/stats`) and is `null` on all of them when it cannot be
    counted. `/stats` used to count a `label_validated` field no writer
    sets and served 0 on failure.
  - Deleting `default`: the dry run answers 200 with
    `blocking: ["project_protected"]`; a real delete is 409
    `project_protected` with or without `confirm`/`force` (it used to be
    422 `confirm_mismatch` without `confirm`).
  - The last-active-project rule is one helper used by the delete dry
    run and the real archive/delete guards: only other `active` projects
    count (the dry run used to count archived ones, the guards counted
    building/failed/deleting ones).
  - A malformed project slug in any `/curation/projects/{project}...`
    path is 404 `project_not_found`, not a 422 validation error.

### Fixed
- **Installer review round-3 follow-ups (s1-s5).**
  - The install summary no longer claims "every port is bound to
    127.0.0.1" when Cropwright is on the LAN; it names Cropwright as the
    exception.
  - A specific non-loopback `--bind <ip>` now narrows Cropwright to that
    interface instead of leaving it on `0.0.0.0` (`--local-only` still wins).
  - `build_deploy_bundle.sh` stages Cropwright's release files into
    `<out>/cropwright/<tag>/` when given `CW_RELEASE_DIR` (checked against
    `cropwright.lock`), so a `--release-dir` install of the cropwright tier
    is offline. Without them the installer now says it is fetching Cropwright
    from the network instead of doing so silently.
  - An `images.lock` line whose repo differs from the one
    `scripts/lib/image_keys.sh` names for that key is refused (exit 7, nothing
    pulled). This is a consistency check against a release-script mistake,
    not an authenticity check.
  - `--rollback` restores the newest backup of a *different* version, so a
    same-version re-run after an upgrade no longer makes rollback land on the
    version already installed.
- **Trainer capabilities are read from the trainer volume root.** The
  trainer writes `.trainer_capabilities.json` once at `OP_TRAIN_JOBS_DIR`
  (it serves every project), but preflight's `trainer_gpus` check and the
  heartbeat-based reachability probe read it from the bound project's
  `train_jobs_dir` (`.../projects/<slug>/`), so every project saw "no
  capabilities file" and GPU-scoping could never block. Both now read
  `src.config.projects.trainer_jobs_root()`.
- **Bake-offs are per project end to end.** The GPU arbiter only watched a
  flat `<state_dir>/bakeoff_jobs` (`GpuArbiterConfig.bakeoff_jobs_dir`)
  while the router enqueues into each project's own
  `projects/<slug>/bakeoff_jobs`, so a queued bake-off never kept GPU
  containers stopped. `bakeoff_active()` now scans every project's queue
  (`gpu_arbiter.all_bakeoff_jobs_dirs()`); `GpuArbiterConfig.bakeoff_jobs_dir`
  and the `OP_BAKEOFF_JOBS_DIR` / `OP_BAKEOFF_OUT_DIR` settings are
  removed. Results live in `<project bakeoff_jobs_dir>/out` for every
  project (`default` included), and `prune_training_runs.py` prunes only
  the bound project's results instead of every project pruning the one
  shared `bakeoff_out` dir. The post-export prune pins exports from the
  project's own training/bake-off job dirs, not flat env roots no job is
  written to, so a queued training run's export is never pruned.
- **`SegmenterClient.source_name` has no default.** The constructor no
  longer defaults to `source_name='sam3'`; every caller (the worker
  runner, tests) passes the active profile's `segmenter_name` explicitly,
  so a non-default segmenter name can never be silently mislabeled as
  `sam3` in stored candidate provenance.

### Added
- **Projects foundation (P1).**
  `src/config/projects.py` (`ProjectRecord`/`ProjectResources`,
  `resources_for_new`/`new_project_record`, slug validation),
  `src/config/project_context.py` (`ContextVar`-based `BoundProject`,
  `current_project()`/`bind_project()`/`set_bound_project()`,
  `bind_process_project()` for script entry points,
  `run_in_executor_bound()`, `project_jobs_dir()`, `project_env()`), and
  `src/services/projects/` (`registry.py`: the `op_projects` index
  snapshot, revision-gated `ensure_fresh()`/`poll_loop()`, exactly the
  stored records; `bootstrap.py`: idempotent `default` project create; `guard.py`: transport-level
  OpenSearch project guard — `CrossProjectAccess`/`ProjectNotBound`/
  `ProjectReadOnly`, `make_curation_opensearch()` and
  `make_script_opensearch()` as the only client factories;
  `capacity.py`: read-only shard/heap capacity check, no fixed project
  cap; `script_binding.py`: `--project` for scripts).
- **Every curation route is scoped under
  `/curation/projects/{project}/...`** (`src/routers/curation/_mounting.py`),
  binding the project for the request. There is no unscoped alias: an
  unscoped curation path is a 404. The only routes outside a project are
  `GET /curation/projects[/{project}]`, `GET /curation/health` and
  `GET /curation/events`. Served URLs (thumbnails, region thumbnails,
  training artifacts) are always the scoped form.
- **Global `GET /curation/health` and `GET /curation/events`** for
  project-less screens: deployment facts only, and only `project: null`
  events (`project.*` and `combine.*` with `target`). The scoped
  `{prefix}/health` keeps its shape plus `project`.
- `GET /curation/projects` serves `labels.status` and
  `limits.retired_slugs`; `GET /curation/projects/{project}` serves the
  project's counts and a typed `error`; `ProjectLifecycleResponse`
  (`{project, warnings}`) is the envelope P3's lifecycle routes answer.
- Isolation errors map to structured responses: `CrossProjectAccess` /
  `ProjectNotBound` → 500 `internal_isolation_error`, `ProjectReadOnly` →
  409 `project_read_only`.

- **Multi-project workers (P2).** One detection worker, VLM worker,
  auto-label worker and cluster-refresh daemon serve every active project:
  each cycle they list the active projects, skip paused ones, and bind
  each project only around its own work; `--project SLUG` narrows a
  worker to one project. The detection worker splits each fetch by
  deficit round-robin (equal quota, rotating start, leftover capacity to
  projects that filled theirs), caps each project's in-flight items at
  `ceil(pipeline capacity / active projects)`, backs idle projects off
  from 5 s to 60 s, keeps each batched VLM call to one project, classifies
  against the item's own project registry, and flushes one `_bulk` per
  project. The auto-label worker runs one job at a time across projects,
  oldest trigger first, and checks each project's IVF centroids for a
  retrain on its own interval.
- Per-project pipeline pause: `<project_state_dir>/pipeline_paused.flag`
  stops the workers' fetches for that project only (no route yet). The
  detection worker writes `runtime_detection_worker_<host>.json`
  (`inflight`, `applied`, `paused`) into each project's state dir.
- `prune_exports.py` and `prune_training_runs.py` prune every active and
  archived project under its own binding.

### Changed
- **BREAKING: `default` is an ordinary project.** It is created at first
  boot with the standard naming: indexes `op_prj_default__<role>`, class
  registry/exports/bake-off eval sets under
  `$OP_PROJECTS_DATA_ROOT/default/`, uploads and job state under
  `.../projects/default/`. It can be archived, never deleted. The
  `OP_*_INDEX` env vars, `OP_ITEMS_INDEX_OVERRIDE`, `OP_REGISTRY_PATH`,
  `OP_EXPORT_ROOT`, `OP_UPLOAD_ROOT` and `OP_BAKEOFF_EVAL_ROOT` are gone;
  existing `op_*` indexes are not migrated (re-create and re-ingest).
- **Unbound project-scoped config fails closed.** Reading a project-scoped
  `CurationConfig` field with no project bound raises `ProjectNotBound`
  instead of silently using `default`. Requests bind through their route;
  the API lifespan runs unbound and binds each active project in turn
  only for startup steps that touch project data; every
  `scripts/curation` entry point binds `--project` (default
  `$OP_CURATION_PROJECT`, else `default`) for its whole process, resolved through
  the registry so the stored status applies.
- Index names, the class registry, the index bootstrap flag, the UMAP
  reducer/projection state files, the eval-dataset roots, and the scores /
  probe / selection / projection job dirs resolve per bound project
  (`items_index()` and friends replace the frozen `CURATION_*_INDEX` /
  `ITEMS_INDEX` constants). `default` is now an ordinary project
  (`op_prj_default__*`), created with `resources_for_new('default')`;
  the env-derived special case is gone.
- The event hub stamps every event with the bound `project` and refuses
  an unbound publish or one naming another project; a scoped stream
  delivers only its own project's events, the global stream only
  `project: null` ones (`project.*`, `combine.*`).
- **The OpenSearch project guard fails closed by construction.** It
  allowlists the request shapes the codebase sends and requires every
  index they name (URL, multi-doc line, query body) to belong to the
  bound project; index-less searches, wildcards, `_all`, aliases,
  `_reindex`, `_sql`, unknown `op_prj_` names and cross-index bodies are
  refused. It is installed when the shared client is built.
- The IVF residual centroid store lives in each project's state dir
  (it was one global store shared by every project), the pipeline SSE
  stats cache is kept per project, and the auto-label trigger/state/
  heartbeat/cancel files resolve under the bound project's `autolabel_dir`.
- The pipeline SSE stream polls its own project's auto-label `state.json`
  (1 s) instead of waiting on a process-wide event that nothing signalled.
- Workers read OpenSearch only through the guarded client, against the
  bound project's indexes; `OP_ITEMS_INDEX_OVERRIDE` is gone.

### Removed
- `src/services/curation/autolabel/cli.py` (nothing launched it) and the
  unstarted auto-label `state.json` watcher.

### Added
- **Text-free region mode.** A region profile with `text_reader: "none"`
  stores region boxes and no region text: the region OCR reader never
  runs, a VLM reading is dropped, and `PATCH /crops/{id}/region_meta`
  answers 422 `{"error": "region_text_disabled"}` for a `region_text`
  edit. `GET /regions/vocabulary` then serves `text_rules: null` and
  `text_choices: []`, and the region-profile summary (also on `/health`)
  gains `reads_text` and `text_hint_enabled`. The `regions` review tab
  drops its `text` filter for such a profile, and the text-repair tools
  (`rederive_region_text.py`) exit cleanly with nothing to do.
- **Optional OCR text hint.** New profile fields `text_hint_enabled`
  (default `true`) and `text_hint_require_letters_and_digits` (default
  `false`). The text-hint re-pass after a segmenter miss runs only when it
  is enabled, an `ocr_pipeline_model` is set and the segmenter leg is on;
  otherwise the chain ends at `<segmenter>:miss`. The `OCR text hint`
  actor and the `segmenter_text_hint` region source are only listed in
  the vocabulary when the hint can run.
- **`parent_classes` region-profile field.** Restricts the region stage
  to items whose `class_name` or `proposal_name` matches (case-insensitive;
  empty = every item). Ingest seeds only matching items and the detection
  worker skips non-matching ones already pending.
- **Built-in text-free prompt pack `generic_region_v1`**, plus a public
  car -> wheel example: `examples/region_profiles/vehicle_wheel.json`
  (segmenter-only, text-free) and `examples/prompt_packs/vehicle_wheel.json`.
- **`GET /classes` exposes `merged_into`.** A class merged via `POST
  /classes/merge` has always tracked its `merged_into` target internally
  (`RegistryClassEntry.merged_into`), but the wire model never served it,
  so the frontend had no way to render "-> merged into X" without a
  second `GET /classes/{id}` round trip.
- **Fresh-start gaps batch B: compose and install portability.**
  - `yolo-api` and `curation-detection-worker` both mount a source-image root
    at the same container path (`${OP_SOURCE_ROOT_HOST:-./data/source}:/data/source:ro`,
    `OP_SOURCE_ROOT=/data/source`) and `./examples:/app/examples:ro`, so
    `OP_REGION_PROFILE_PATH=/app/examples/region_profiles/license_plate.json`
    (the env.template example) actually resolves in both containers.
  - `pe_image_encoder` is now in `triton-server`'s default `--load-model`
    list (CURATION.md already called it required, not optional). `make
    export-all`/`./scripts/setup.sh`'s automated export flow now builds it
    (weights download, image-tower ONNX, TensorRT with an ONNX Runtime
    fallback, text-tower ONNX) before Triton's first start — it needs this
    like every other listed model, since Triton's explicit
    `model-control-mode` exits at startup if a listed model fails to load.
  - New optional `vlm` compose profile (`docker compose --profile vlm up -d`):
    a pinned-tag vLLM service serving Gemma 4 E4B, configurable via
    `VLM_IMAGE`/`VLM_MODEL`/`VLM_SERVED_MODEL_NAME`/`VLM_DTYPE`/
    `VLM_MAX_MODEL_LEN`/`VLM_GPU_MEMORY_UTILIZATION`/`VLM_LIMIT_MM_IMAGES`/
    `VLM_GPU_ID`/`VLM_PORT`. `yolo-api` and `curation-vlm-worker` both carry
    `extra_hosts: ["host.docker.internal:host-gateway"]` so the
    `OP_VLM_URL=http://host.docker.internal:<port>/v1` (external VLM) example
    resolves on Linux, not just Docker Desktop.
  - New optional `docker-compose.gpu-arbiter.yml` overlay: mounts
    `/var/run/docker.sock` into `yolo-api` so `OP_GPU_ARBITER_CONTAINERS`
    coordination actually works, documented as an explicit opt-in with its
    security tradeoff spelled out. Without it, the arbiter now logs exactly
    one `arbiter_docker_unavailable` warning per outage (was: one per
    call) when it fails open.
  - `yolo-api` carries a network alias `op-api` so Cropwright's default
    `API_UPSTREAM=http://op-api:8000` resolves without an override; see
    `docs/CURATION.md` "Wiring up Cropwright" and the README's Cropwright
    paragraph for the exact env vars and docker network name.
  - `curation-trainer`'s `OP_TRAIN_GPU_ORDER` and `device_ids` are both
    interpolated from the same `OP_TRAIN_GPU_ORDER` env var (was hardcoded
    to `0` in `environment:`); a multi-GPU order still needs a compose
    override for `device_ids` (documented inline).
  - `scripts/setup.sh` gained `--force` (never overwrites an existing `.env`
    otherwise) and `--curation` (prints the curation-subsystem next steps:
    PE export, class registry, VLM, segmenter, `--profile`); its smoke tests
    and `unload_models_for_export`/`check_triton_container` now read the
    deployment's actual configured ports/compose-service state instead of
    hardcoded `4600`/`4603`/`4607` and a hardcoded `triton-server` container
    name.
  - `tests/test_full_system.py` reads `API_PORT`/`TRITON_HTTP_PORT`/
    `OPENSEARCH_PORT` from the environment; README's Testing section gained
    a Docker-only path (`docker compose exec yolo-api pytest tests/ -q`).
  - `tests/test_compose_contract.py` gained invariants pinning all of the
    above (no fixed project name, no hardcoded host ports, source root +
    examples mounted on both services, `pe_image_encoder` in the default
    load list).
- **S-2: heartbeat-based curation worker healthchecks.** The four
  curation background workers (detection, VLM, auto-label,
  cluster-refresh) now write a heartbeat file on their main loop —
  including while idle — checked by
  `python src/services/curation/worker_liveness.py check <name>
  --max-age 120`, replacing `pgrep -f <module>` (which can't see a
  deadlocked-but-still-running event loop). `yolo-api` gained its own
  `/health`-based healthcheck so `curation-vlm-worker` /
  `curation-cluster-refresh`'s `depends_on` can gate on
  `condition: service_healthy` instead of merely "container started."
  New `OP_HEARTBEAT_DIR` env var (container-local, no mount needed).
- **S-3: cross-process event bus.** `GET /events` SSE subscribers on any
  of `yolo-api`'s 8 uvicorn worker processes now see every published
  event, not just the ones published on the same process. Backed by a
  shared, bounded, rotated JSONL log
  (`{OP_STATE_DIR}/events/events.jsonl`) every process tails; new
  `OP_EVENT_BUS` (`file` default, `process` restores the old
  in-process-only behavior) and `OP_EVENT_LOG_MAX_BYTES` env vars.
  `curation-detection-worker` now sets `OP_EVENT_API_URL` so its
  `crop.region_verified` events reach every subscriber, not one
  arbitrarily-chosen worker; `bulk_writer.py`'s event-publish URL also
  falls back to `OP_API_BASE_URL`/`OP_API` when `OP_EVENT_API_URL` is
  unset. `GET /events/stats` now also reports `bus` and `log_path`.
- **Class deprecate/restore.** `POST /classes/{class_id}/deprecate` flips
  `deprecated` on a class nothing references (idempotent; refuses on a
  still-referenced class); `POST /classes/{class_id}/restore` undoes it
  (`404` unknown id, `409` if a non-deprecated class already uses the
  name) — a lighter-weight alternative to `POST /classes/merge` for a
  class that was never actually used.
- **Deployment-supplied training presets.** `OP_TRAIN_PRESETS_PATH` (a
  JSON list of the same shape as the built-in presets) appends
  deployment-specific `class_subset_presets` entries, served by
  `GET /train/presets` alongside the generic built-ins (`all`, and
  `all_except_region` / `region_only` when the active region profile
  sets `region_class_name`).
- **A background probe-inference job API**: `POST /probe/run` (resolves
  a finished training job's checkpoint, `409` if not `finished` or no
  checkpoint on disk; claims a GPU through the same arbiter
  `POST /train/start` uses), `GET /probe/status`, `POST /probe/cancel`
  — wraps `run_probe_inference` so a probe backfill runs as a tracked
  background job instead of blocking the request; one job at a time.
- **Review queues explain an empty result instead of just serving zero
  rows.** `GET /review/{tab}` computes `empty_reason` from live index
  state (for example, `"no probe predictions — run a probe"`,
  `"item scores never computed"`, `"no unclassified proposals"`, else
  `"no items match"`); `GET /review/tabs` gained
  `empty_state: {has_probe_predictions, has_item_scores}` so a client
  can word any tab's empty state without a per-tab round trip.
- **A naming-leak pre-commit guard** (`scripts/codegen/check_naming_leaks.py`,
  wired into `.pre-commit-config.yaml`): three `git grep` scans over the
  whole tracked tree catch a reintroduced company name, retired
  vendor/domain vocabulary, or private class-registry vocabulary before
  it ships, filtered through a reviewed, per-line allowlist
  (`scripts/codegen/naming_leak_allowlist.txt`) so a deliberate example
  or historical/negative-test mention doesn't need re-justifying on
  every commit.
- **Per-class model comparison.** Every export with a labelled test split is
  an eval dataset (`GET /bakeoff/eval_datasets`, class counts and
  test-split hashes computed from the files); finished training runs are
  contenders directly (`GET /bakeoff/trained_models?dataset_id=` with
  same-export / same-frozen-test / train-test overlap). The harness scores
  per class and overall (COCO mAP50, mAP50-95, micro P/R/F1), maps each
  model's classes onto the dataset's (run class remap, registry ids, names
  or an explicit map) and reports uncovered classes and unmapped
  predictions; rows rank on the classes every model covers.
- **Test-split identity and build identity in run lineage.** Export
  manifests record `frozen_test_sha` (which frames) and `test_label_sha`
  (which boxes); training job specs and run manifests record `dataset_sha`,
  `frozen_test_sha`, `test_label_sha` and `dataset_version_tag` separately,
  plus `code_versions.api_sha`, `trainer_sha` and `trainer_image_id`.
  `OP_BUILD_SHA` is baked into the API, trainer and evaluator images at
  build time (`build.args`, OCI revision label; `make build` passes it); a
  runtime value still overrides it.
- **VLM class-attempt fields** `vlm_class_attempted_at` (date) and
  `vlm_class_empty_reason` (keyword: `no_answer` / `no_match` /
  `invalid_index` / `unparseable`; `null` when the attempt answered) on items
  (mapped, migrated on boot, class-guarded) and on the item wire. The `all`
  review tab surfaces items whose last attempt was empty; the VLM worker and
  the auto-label sweep skip them for 24 h. Every VLM class write (label
  batch, auto-label sweep, the region worker's combined call) now records a
  full, restorable `class_id_history` snapshot, including writes onto a
  proposal with no class yet and `class_source`-only writes.
- `scripts/curation/repair_empty_vlm_answers.py` (dry-run default,
  `--apply`, OCC): restores items stamped `vlm_unmatched` for an empty VLM
  answer to the class source they had before (VLM, ingest proposal or
  classifier, recovered from the untouched class provenance; class history
  as fallback) and records the empty attempt.
- **Served detector/segmenter/VLM vocabulary**:
  `GET {prefix}/regions/vocabulary`
  serves `{detectors, region_sources, chain_actors}` (each entry `{id,
  label, role, filterable}`) built from the active `DetectionProfile` /
  ingest profiles / `OP_VLM_MODEL` — never a hardcoded model id — so the
  frontend stops keying a label/palette map on private ids
  (`lpr_nanov11_640`, `sam3`, `gemma-4-e4b`). `GET {prefix}/review/tabs`
  serves `{id, label, description}` for every review tab.
- `scripts/curation/backfill_region_embeddings.py` (dry-run default) and a
  shared region-embedding encode helper, so region false-positive clustering
  has embeddings to work with.
- `POST /curation/review/new_class_proposals/resolve` — bulk-resolves
  every pending `vlm_new_class_pending` item proposing a term in one call
  (map to an existing class or create one, `?dry_run=` to preview),
  instead of relabeling only the summary endpoint's capped
  `sample_crop_ids` one page at a time. Explicitly mapped
  `vlm_verify_completed_at` (`date`) on the items index — it was being
  written and range-queried but left to dynamic mapping.
- `GET /curation/train/gpus` — served training GPU picker (values, human
  labels, stop advisories, and the resolved default), so the frontend no
  longer hardcodes GPU ids/labels. `OP_GPU_ARBITER_CONTAINERS` entries may
  now carry a GPU scope (`name@2`, `name@0/2`); `OP_GPU_LABELS` and
  `OP_TRAIN_DEFAULT_GPUS` configure the option labels and default value.
- Curation operator tools: `run_probe.py` (probe-inference backfill),
  `reclassify_after_registry_growth.py`, `requeue_regions.py` (incl.
  `--missing-status` backfill), `seed_class_registry.py` (registry from ONNX
  `names`, `--check`), `cluster_raw_labels.py`, `ingest_upload.py` +
  `POST /curation/ingest/upload` (byte ingest with content dedup;
  persists content-addressed uploads server-side under
  `OP_UPLOAD_ROOT`), `GET /curation/ingest/config` (served upload/batch
  limits and accepted extensions so a client stops hardcoding them),
  `import_labeled_dataset.py` (incl. `--images-only`) and
  `eval_regions_vs_gt.py` (region cascade vs ground truth: recall, precision,
  IoU, background false-positive gate).
- `BakeoffProfile` + `GET /bakeoff/profiles` (with `default`/`default_profile`),
  optional `profile` on bake-off runs, a ported quantize leg.
- Ingest writes class provenance on every item, publishes `crop.created` SSE
  events, writes the backbone embedding from the secondary detector's feature
  map, and seeds `pending_detection` region status so the cascade picks new items up.
- `GpuArbiterConfig.from_env` (`OP_GPU_ALLOWED_IDS`, `OP_GPU_ARBITER_*`),
  `OP_VLM_MAX_IMAGES_PER_CALL`, `OP_SOURCE_PATH_ALIASES`, `OP_PROMPT_PACK_PATHS`
  (several selectable prompt packs), `OP_INGEST_PRIMARY_CLASS_IDS`.
- Per-run `?prompt_pack=` on auto_label (422 on unknown id, echoed in job args).
- `vlm_proposed_class_id` / `vlm_proposed_class_name` on every item;
  `GET /curation/class_sources`; `GET /curation/classes/{id}`; `GET /crops`
  `limit`/`sort`/`conf_min`/`conf_max`/`k`; `/export/datasets` `kind`/`profile_name`.
- Generated TypeScript `RegionStatus` contract (`contracts/ts/regionStatus.ts`)
  with a `--check` pre-commit drift hook.
- **Segmenter container (`docker/segmenter/`)**: a reference
  implementation of the detection cascade's segmenter leg — a FastAPI
  service wrapping Meta's SAM 3 that answers `POST
  /sam3/segment_plate` (alias `POST /segment`) and
  `/segment/batch` with candidate boxes in the submitted image's
  normalized frame. `text_prompt` is required per request with no
  server-side default, so the service carries no domain of its own.
  Ships behind its own `segmenter` compose profile (it needs a GPU and
  a HuggingFace token); the leg remains optional — an empty `SAM3_URL`
  still makes it a clean no-op. See
  [`docker/segmenter/README.md`](docker/segmenter/README.md).
- **Trainer container** (`docker/trainer/`, compose service
  `curation-trainer` behind the new `training` profile). Until now the
  API implemented only the control-plane half of the training file
  protocol and the repo shipped nothing that could answer it — a
  `/curation/train/start` had no counterparty outside the test harness's
  shell-script fake. The image watches `/jobs/` for `job.json`, runs
  each through Ultralytics, and writes `status.json` heartbeats,
  `run.log`, `best.pt` + `best.onnx`, and a `manifest.json` lineage
  envelope. Includes the subset/class-remap dataset rewrite,
  Albumentations stage-1 augmentation, cooperative cancel, CUDA-OOM
  batch backoff, multi-GPU AutoBatch, campaign auto-skip/auto-promote,
  and an optional side-by-side comparison against a served Triton
  model. Nothing domain-specific is baked in: dataset, class subset,
  hyperparameters, augmentation preset, orientation-sensitive class
  names and incumbent model all arrive via `job.json` or `OP_*` env.
- **`curation-mlflow`** compose service (port 4609) for optional
  experiment tracking of those runs.

### Changed
- **`docker-compose.yml` is now pull-only and deploy-safe** (one-line
  installer plan, Wave 0). It no longer has any `build:` block or any
  bind mount of `./src`, `./scripts`, `./export`, `./tests`,
  `./benchmarks`, `./test_images`, `./VERSION` or `./examples` — dropping
  it into an empty directory with no git checkout and running
  `docker compose pull && up -d` no longer gets Docker silently creating
  empty host directories that shadow the image's `/app/src`,
  `/app/export`, etc. Every `build:` block and every one of those source
  mounts moved to a new opt-in overlay, **`docker-compose.dev.yml`**,
  which restores today's checkout hot-reload workflow unchanged. `make`
  (via the `COMPOSE` variable), `scripts/setup.sh` and
  `scripts/openprocessor.sh` all detect a checkout (`src/main.py` next to
  the compose file) and add the dev overlay automatically — **no action
  needed for existing checkout users of `make`/`./scripts/setup.sh`.** A
  bare `docker compose` invocation now needs
  `-f docker-compose.yml -f docker-compose.dev.yml` explicitly to build
  from source or hot-reload; `docker compose up -d` alone now only pulls.
- **`Dockerfile` bakes in `export/`, `examples/` and a model-repo seed**
  (`/opt/openprocessor/model_repo_seed`, from the tracked `models/`
  config tree) so the deploy-safe compose file above doesn't need to
  bind-mount any of them. `docker/evaluator/Dockerfile` gains the same
  `examples/` copy (read by the opt-in bake-off baseline path).
- **Published ports default to loopback-only.** Every `ports:` entry in
  `docker-compose.yml` is now
  `"${OP_BIND_ADDRESS:-127.0.0.1}:<host-port>:<container-port>"`. The API
  has no auth and OpenSearch security is off by default, so this is a
  behavior change for anyone who was relying on the old bare
  `${PORT}:<container-port>` binding on `0.0.0.0` — set
  `OP_BIND_ADDRESS=0.0.0.0` (and put a reverse proxy with auth in front;
  see `SECURITY.md`) to restore the old exposure.
- **Custom images are pinned per-service and never fall back to `latest`.**
  `triton-server`, `yolo-api` (and its curation workers), the evaluator,
  segmenter and trainer images each gained their own override var
  (`OP_TRITON_IMAGE`, `OP_API_IMAGE`, `OP_EVALUATOR_IMAGE`,
  `OP_SEGMENTER_IMAGE`, `OP_TRAINER_IMAGE`), falling back to
  `${OP_IMAGE_REPO:-davidamacey}/<image>:${OP_IMAGE_TAG:-<VERSION>}` — the
  fallback tag now tracks the `VERSION` file instead of `latest`
  (`test_compose_default_tag_matches_version` pins this).
- **`env.template` gained a consolidated "Curation quick-config" block**
  (the handful of vars every curation tier actually needs to get
  running) plus `OP_BIND_ADDRESS` and the new per-service `OP_*_IMAGE`
  vars. `docs/CURATION.md`'s environment-variables section now links to
  that block instead of repeating scattered paragraphs.
- **Letters-and-digits text-hint rule is opt-in.** A text-hint candidate no
  longer has to mix letters and digits unless the profile sets
  `text_hint_require_letters_and_digits: true`
  (`examples/region_profiles/license_plate.json` does).
- **`examples/region_profiles/license_plate.json` is segmenter-only**
  (`detector_model: ""`) and sets its text-hint flags explicitly.
- Segmenter candidates carry the profile's `segmenter_name` as their
  source instead of a hardcoded `sam3`. New geometry rejects are recorded
  as `parent_bbox_unpack_failed` / `parent_bbox_degenerate` (were
  `vehicle_bbox_*`). `VlmLabeler.label_vehicle_batch` is renamed
  `label_item_batch`.
- The GPU arbiter now decides which containers to stop by **GPU scope**, not
  claim size: a single-GPU training claim that intersects a scoped
  container's GPU set stops that container (it no longer takes a multi-GPU
  claim to free a GPU that hosts a large sibling service). Unscoped
  containers keep the original "stopped only on a multi-GPU claim" behavior.
  `TrainJobSpec.cuda_visible_devices` / `TrainCampaignSpec.cuda_visible_devices`
  now default to the smallest `OP_GPU_ALLOWED_IDS` entry (or
  `OP_TRAIN_DEFAULT_GPUS` if set) instead of a hardcoded `'0'`, so a
  restricted allowlist that excludes GPU 0 no longer rejects the default spec.

### Fixed
- **Three Grafana/Prometheus monitoring panels/alerts queried metrics
  this Triton version never exports, so they were always empty and could
  never fire.** `monitoring/dashboards/triton-unified-dashboard.json`'s
  "Model Ready" panel and `monitoring/alerts/triton-alerts.yml`'s
  `ModelNotReady` alert both queried `nv_model_ready_state`, which
  Triton's `/metrics` doesn't export (confirmed against a live server's
  actual exposition) — removed; there is no honest Triton or
  DCGM/nvidia-exporter equivalent for per-model readiness (it's a
  Triton-internal concept, not a GPU one), so use `GET
  /v2/repository/index` or `/curation/health` instead. The "GPU
  Temperature" panel queried `nv_gpu_temperature` (also never exported)
  — switched to `DCGM_FI_DEV_GPU_TEMP` from the already-scraped
  `dcgm-exporter` service. Also found and fixed while auditing this: the
  "Model Track Latency Comparison" panel's P95/P99 lines used
  `histogram_quantile(...,
  nv_inference_request_duration_us_bucket)`, but
  `nv_inference_request_duration_us` is a plain counter, not a histogram
  (Triton exposes no `_bucket` series for it) — dropped, keeping only the
  Avg line. New `tests/test_monitoring_metrics.py` pins every
  dashboard/alert metric name against a fixture of metrics actually
  exported by this stack's pinned Triton/dcgm-exporter/node-exporter
  images (confirmed red against the old queries, green against the fix).
- **Alloy's log-collection filters never matched this compose's own
  containers.** `container_name` in `docker-compose.yml` has always been
  `${COMPOSE_PROJECT_NAME:-openprocessor}-triton` /
  `${COMPOSE_PROJECT_NAME:-openprocessor}-api`, never a bare
  `triton-server` or `yolo-api`/`pytorch-api` container, so
  `monitoring/alloy-config.alloy`'s old `/triton-server.*` and
  `/(yolo-api|pytorch-api).*` `discovery.relabel` regexes never matched
  under any `COMPOSE_PROJECT_NAME` — Loki only ever received a different
  stack's logs (or nothing) from the monitoring profile. Both regexes now
  match on the `-triton` / `-api` container-name suffix instead, which is
  independent of the project name. New `tests/test_monitoring_config.py`
  pins the fix (and confirms it fails red against the old patterns).
- **An empty `detector_model` no longer calls Triton.** It used to run
  inference against model `''` on every item, log `region_infer_failed`
  and append a `':miss'` trace tag with an empty actor; the detector leg
  is now skipped entirely.
- **Segmenter never became reachable on a stock install (F-75).** The
  `segmenter` service's `env_file: .env` loaded the host-port variable
  `SEGMENTER_PORT` (env.template default `4611`) straight into the
  container, and `docker/segmenter/main.py` read that same name as its
  uvicorn listen port -- so the container bound to `4611` while the port
  mapping, healthcheck and `OP_SEGMENTER_URL` all still targeted `8000`.
  The in-container variable is renamed `SEGMENTER_LISTEN_PORT` (default
  `8000`, also set explicitly under `environment:` so it beats
  `env_file`), and a new compose-contract test
  (`test_no_service_reads_a_host_port_var_as_its_own_container_config`)
  guards every other `env_file`-loading service against the same class of
  bug. An audit of the remaining host-port vars (`API_PORT`,
  `TRITON_*_PORT`, `PROMETHEUS_PORT`, `GRAFANA_PORT`, `LOKI_PORT`,
  `DCGM_PORT`, `OPENSEARCH_PORT`, `OPENSEARCH_DASHBOARDS_PORT`,
  `MLFLOW_PORT`, `VLM_PORT`) found no other container reading its own
  host-port var name. **Requires rebuilding the segmenter image.**
- **Region-dependency health check never saw a healthy segmenter (V-1
  follow-up).** `check_region_dependencies` looked up the profile's
  segmenter (e.g. `sam3`) in Triton's repository index, but SAM 3 runs as
  the separate HTTP segmenter service (`OP_SEGMENTER_URL`), not in
  Triton, so `stall_reason` never cleared even with a healthy segmenter.
  Triton-served detectors still go through the Triton repository index;
  the segmenter dependency now does a `GET {OP_SEGMENTER_URL}/health`
  with a short timeout, requiring `loaded: true`.
- **Training couldn't start on a stock install (F-72 regression).**
  `OP_GPU_ARBITER_TRAINER_CONTAINER` defaulting to
  `${COMPOSE_PROJECT_NAME}-trainer` (see the F-72 entry below) meant
  `/train/preflight`'s trainer probe now always ran -- but the stock
  `yolo-api` container has no docker socket/SDK, so the probe
  unconditionally reported `block` ("docker SDK/socket unavailable"),
  422ing `/train/start` even with a perfectly healthy trainer. The probe
  (moved to `src/services/training/trainer_reachability.py`) now reads
  the trainer's own heartbeat file (`.trainer_capabilities.json`, which
  the trainer's watch loop refreshes every ~30s) as its primary signal --
  no docker socket needed. A fresh heartbeat is `ok`; a missing or stale
  one is `warn`, never `block`. The docker SDK/socket path (only present
  behind the `docker-compose.gpu-arbiter.yml` overlay) is now a purely
  optional, confirming extra: it's only consulted when the heartbeat
  itself is missing/stale, and only then may it upgrade the warning to a
  definitive `block`.
- **API image builds again.** `perception_models` is installed with `--no-deps`
  at a pinned commit (its requirements exact-pin `timm==1.0.15`, which
  conflicts with `open-clip-torch>=3.2`'s `timm>=1.0.17`); the PE encoder's
  real runtime deps (`einops`, `regex`) are declared in `requirements.txt`.
- **Triton serves a partial model set.** `triton-server` now runs with
  `--exit-on-error=false --strict-readiness=false`, so one missing or failed
  engine (the minimal setup profile skips OCR; setup continues past a failed
  export) leaves only that model unloaded instead of stopping the server.
  The minimal profile's export now also builds the PE-Core image encoder that
  curation ingest needs. `TRITON_GPU_ID` (default `0`) selects Triton's GPU.
- **`vlm` profile image pinned by digest** to the vLLM Gemma 4 build this
  stack is tested against (`vllm/vllm-openai:gemma4-cu130@sha256:0d1525...`);
  the earlier `v0.11.0` default predates Gemma 4.
- Run lineage recorded the frozen test-split hash as `lineage.dataset_sha`
  and nothing for multi-class exports; `code_versions.api_sha` /
  `trainer_image` were always null (read from the trainer's own env, which
  nothing set); the trainer's MLflow dataset tags read keys the job spec
  does not have and were always empty.
- The GPU arbiter did not see queued bake-offs on the default config
  (`bakeoff_jobs_dir` was unset while the router wrote to
  `<state_dir>/bakeoff_jobs`), and the router claimed hardcoded GPUs
  `0,1` and continued when containers could not be stopped: a GPU-resident
  container could be restarted under a running bake-off. The arbiter now
  defaults to the router's dir, and enqueueing answers 409 (job removed)
  when it cannot stop them.
- The bake-off evaluator could not read exports (no mount); compose now
  mounts `./data` read-only on `curation-evaluator`.
- **Most VLM class answers were read as empty and recorded as `vlm_unmatched`**
  (live: 2,927 of 3,102 `vlm_unmatched` items had `vlm_raw_class=''`). The
  class calls sent no JSON-object `response_format`, so against a vLLM server
  with a reasoning parser the answer landed in the reasoning channel and
  `content` was empty or a lone `]`. Class calls now request JSON-object mode
  (with a `{"results": [...]}` envelope) and fall back to the reasoning
  channel; the combined reply accepts a class name / `"3=name"` / numeric
  string instead of rejecting the whole entry. An empty class answer (empty,
  `null`, `-1`, out-of-range, unparseable) no longer becomes `vlm_unmatched`:
  the item's class fields are left untouched and the attempt is recorded;
  `vlm_unmatched` is kept for a real, non-empty label (with `vlm_raw_class`).
  A VLM call that never completed writes nothing.
- Freshly ingested items never reached `/review/all`, the VLM worker or the
  pipeline VLM sweep (queues gated on a nonexistent `embedding` field).
- Region/label fields fell to dynamic `text` mapping on fresh indexes, breaking
  aggregations; every field is now explicitly mapped and queries no longer
  target `.keyword` subfields.
- A generic proposer's class ids were looked up in the domain class registry.
- Labels imported in the same `/ingest/batch` call were silently dropped.
- Secondary-detector NMS no longer depends on an external YOLOv5 checkout
  (native implementation following YOLOv5's documented semantics).
- Fresh `/jobs` volumes are writable by the app user; the evaluator image builds again.
- Bake-off quantize jobs silently scored nothing (missing module).
- `crop.region_verified` events carry `region_status`; `/export/datasets` lists
  single-class exports; server-built URLs and scripts follow `OP_API_PREFIX`.
- Removed host-specific paths and a LAN hostname from public source and docs.
- A subset-trained run now propagates its `class_remap.json` into the
  checkpoint's `weights/` directory *and* the run manifest, and reports
  `class_remap_copy_failed` on the job status when it cannot. This is
  the trainer half of the promote fix already present on the API side
  (`resolve_class_remap`): without it, promoting a subset run silently
  wrote a `labels.txt` from the full class registry, mislabeling every
  class the served model emits.
- The curation VLM worker never processed anything (bare-script import
  failure swallowed); it now runs as a module and exits loudly on an
  unhandled task exception.
- Training runs failed at the final step because MLflow's artifact root was
  a local path the trainer couldn't write; artifacts now proxy through the
  tracking server (`--serve-artifacts`), and the trainer's MLflow client is
  pinned to the server's major version.
- Promoted Triton models could land outside the mounted model repository;
  the repo path/URL are resolved from env at promoter construction, and
  promoted models are reloaded on API startup after a Triton restart.
- Auto-promote validated classifier labels via class clusters (purity 1.0 by
  construction); it now only considers candidate clusters, skips excluded
  items, and reports a correct dry-run count. Operator repair script:
  `scripts/curation/revert_class_cluster_promotions.py`.
- Region false-positive distances were squared L2 read as plain L2, loosening
  every FP-clustering threshold; region k-means centroids are re-normalized.
- The GPU-arbiter pause sentinel writer and readers used different paths.
- `id_normalize` pulled excluded items back into their class cluster.
- The GPU arbiter fell back to a sentinel-only pause when it could not stop a
  container sharing the claimed GPU; `/train/start` and
  `/train/start_campaign` now refuse with 409 and preflight blocks
  (`gpu_arbiter` check). The `docker` SDK is now a dependency.
- cuML kNN graph self-loops dropped (parity with sklearn).

- `POST /crops/move` into a candidate cluster wrote the cluster id as a
  validated class id; it now only sets placement (a human-owned class is
  cleared, nothing is validated), and unassigned or unregistered targets
  get 400. Export manifests record rows dropped for unregistered class ids
  (`dropped_unregistered_class_ids`), and preflight warns on them.
- `POST /crops/batch_unexclude` returns an unvalidated item to the
  candidate cluster it was excluded from while that cluster still has
  members, instead of leaving it outside every cluster until a recluster.

### Changed (BREAKING)
- **Compose/install portability (fresh-start gaps batch B).** `docker-compose.yml`
  no longer hardcodes `name: openprocessor` or any `container_name:` — both are
  now interpolated from `COMPOSE_PROJECT_NAME` (default `openprocessor`, so an
  existing single-stack deployment behaves identically). **Migration hint:**
  if you script against container names directly (e.g. `docker exec yolo-api
  ...`, `docker logs triton-server`), switch to `docker compose exec
  yolo-api ...` / `docker compose logs triton-server` — those already resolve
  by service name regardless of the interpolated container name, and keep
  working the same way after this change. Every host port
  (`API_PORT`, `TRITON_HTTP_PORT`, `TRITON_GRPC_PORT`, `TRITON_METRICS_PORT`,
  `PROMETHEUS_PORT`, `GRAFANA_PORT`, `LOKI_PORT`, `OPENSEARCH_PORT`,
  `OPENSEARCH_DASHBOARDS_PORT`, plus new `MLFLOW_PORT`, `DCGM_PORT`,
  `SEGMENTER_PORT`, `VLM_PORT`) is now interpolated from `.env`/the shell
  instead of hardcoded, so a second isolated stack on the same host only
  needs a `.env` with a different `COMPOSE_PROJECT_NAME` and remapped ports.
  `env.template`'s `TRITON_HTTP`/`TRITON_GRPC`/`TRITON_METRICS` were renamed to
  `TRITON_HTTP_PORT`/`TRITON_GRPC_PORT`/`TRITON_METRICS_PORT` to match the
  Makefile's existing names — update any script/CI reading the old names.
  `Makefile`'s port variables now use `?=` and load `.env` (`-include .env`),
  so both `.env` and `make API_PORT=... TRITON_HTTP_PORT=... <target>` work.
- **One generic curation wire vocabulary.** Every region field is `region_<attr>`
  on the wire, fixed regardless of `OP_REGION_FIELD_*` storage overrides;
  `plate_thumbnail_url` → `region_thumbnail_url`; `gemma_*` → `vlm_*` and `v6_*` →
  `classifier_*` across item fields, `class_source` values, the `vlm_low_conf`
  review tab, auto_label params, `/health` and `/stats/dataset` (`plates` →
  `regions`); `coco_proposal_name` → `proposal_name`; ingest `n_plates` →
  `n_regions`. Region write bodies use `region_*` keys and reject unknown keys.
  Every item-returning endpoint (`/crops`, `/crops/{id}`, `/review/{tab}`,
  `/regions`, training candidates, `/search/text`) returns the same serialized
  item. Full old→new table: `docs/design/curation_api_contract.md` (B3).
- **Region detection is off by default, and no profile ships built in.**
  `src/services/detection/reference_profiles.py` is removed; the
  license-plate example profile is a data file,
  `examples/region_profiles/license_plate.json`, loaded via
  `OP_REGION_PROFILE_PATH=<path>`. `OP_REGION_PROFILE=<name>` now only
  resolves a profile a deployment's own startup code registered.
  `DetectionProfile` gains `region_class_name`, `display_name` and
  `display_name_singular` fields, served on `GET {prefix}/regions/vocabulary`.
- **`OP_DETECTION_*` is retired**; ingest detectors use `OP_INGEST_PRIMARY_*` and
  `OP_INGEST_SECONDARY_*` (leftover `OP_DETECTION_*` vars fail with a rename
  message). The secondary detector is now actually wired into ingest.
- **The ingest primary is a proposer by default** (`OP_INGEST_PRIMARY_ASSIGNS_CLASS=false`):
  its detections are unlabeled `<name>_proposal` items carrying the model's own
  label (`OP_INGEST_PRIMARY_LABELS_PATH`); the secondary assigns the class.
- **`detection_profile` is read-only**: `?detection_profile=` on
  `POST /pipeline/auto_label[/start]` and `PUT /settings` for that axis return
  422. `GET /methods` entries carry `settable: bool`.
- The detector bake-off harness is domain-neutral by default (`generic`
  `BakeoffProfile`; `--backend triton` requires a model); plate baselines moved
  to the `license_plate` example profile; paper-only scripts (a dedup-threshold
  sweep and a LaTeX-number generator that hardcoded a private model id and a
  live-deployment URL) removed from the public tree.
- **Model comparison (bake-off) API v2, generic and multi-class** (clean break,
  no compatibility fields; shapes in `docs/design/curation_api_contract.md`). Every
  `/curation/bakeoff/*` route is typed and result files carry
  `schema_version: 2` (older result files answer 409).
  `POST /bakeoff/run` takes `datasets: [{id}]` (`export:<path>`,
  `external:<group>/<name>`, or `run:<job_id>`) and `models[]` discriminated
  on `source` (`run` / `baseline` / `custom`), plus
  `quantize: {run_id, formats, n_calib, calib_split, throughput}`; removed:
  `dataset`, `datasets[].path/name`, `verify_frozen`, free-form model specs
  (`backend`/`profile`/`gt_class_id`/`gt_class_name`/`pred_class_id`/
  `lpdnet_variant`/`primary_classes`), `quantize.coreml`. Responses: eval
  datasets use `source` + `group` (no `n_test`/`frozen_sha`/`kind`);
  `trained_models` serves `trainer_map50` / `trainer_map50_split` (were
  `map50` / `map50_split`); comparison rows put metrics under `overall` /
  `common` with `per_class` and `coverage`; `results` takes `?dataset_id=`;
  matrix `best` values are lists of tied winners; job state adds `queued`.
- **`BakeoffProfile` loses `target_class_id` / `target_class_name`**: a
  profile scores every class in the eval split (`class_filter` narrows by
  name). `OP_BAKEOFF_PROFILE_TARGET_CLASS_ID` / `_NAME` are retired (startup
  fails with a pointer to `OP_BAKEOFF_PROFILE_CLASS_FILTER`). The
  license-plate profile, baselines, converters and the `lpdnet` /
  `open-image-models` backends moved to `examples/bakeoff/license_plate/`
  and load only by profile path; `GET /bakeoff/profiles` no longer lists
  example profiles and the default baseline registry is empty.
- The trainer's opt-in auto-quantize posts `POST /curation/bakeoff/run`
  (via `OP_API_BASE_URL` + `OP_API_PREFIX`) instead of writing a job file;
  `campaign.py` no longer reads `OP_BAKEOFF_JOBS_DIR` / `OP_BAKEOFF_OUT_DIR`.
- **Stored-data renames** (re-ingest required):
  - Items index kNN field `v6_embedding` → `backbone_embedding`
    (`CurationConfig.BACKBONE_EMBEDDING_FIELD`).
  - Images + items ingest-source field `hdd_source` → `source`; `GET
    /crops`'s `?hdd_source=` query param is removed (use the existing
    `?source=`).
  - Stored `region_source` / `candidate_source` provenance values:
    `sam3` → `segmenter`, `sam3_text_hint` → `segmenter_text_hint`, `lpr` →
    `detector`, `lpr_existing` → `detector_existing`.
  - `class_id_history[].writer` value `sam_worker` → `region_worker`.
  - `GET /curation/ingest/sam_drain` → `GET /curation/ingest/region_drain`;
    its response and `GET /stats/dataset`'s `in_progress.*` drop the legacy
    `pending`/`pending_verify` rollup keys (re-ingested data can never carry
    those short names).
  - Export manifest `dataset_kind` no longer accepts the alias
    `lpr_single_class`; only `single_class` is recognized.
  - No hardcoded model-id defaults: `OP_VLM_MODEL` has no default (was
    `gemma-4-e4b`) and `VlmLabeler` construction fails loudly when a VLM
    URL is configured without one; the reference license-plate profile's
    `detector_model` is the neutral example id `license_plate_detector`
    (was the proprietary Triton id `lpr_nanov11_640`).
  - `DELETE /curation/models/{name}`'s unload guard drops its hardcoded
    `lpr_` name prefix; a model is protected only via the active
    `DetectionProfile`'s configured model ids or the fixed `paddleocr_`
    prefix.
  - `OPENWEBUI_BASE_URL` / `OPENWEBUI_MODEL` / `OPENWEBUI_API_KEY` /
    `VLM_URL` / `GEMMA_URL` are retired; only `OP_VLM_URL` / `OP_VLM_MODEL`
    / `OP_VLM_API_KEY` are read now.
- **Wire surface renames**: `GET /curation/methods`'
  operationId is `get_methods_curation_methods_get` (was a
  company-initialed operation id); its `flags` keys drop the same
  company-initialed prefix (`scores_enabled`, `scores_shadow`,
  `select_diverse_enabled`, `viz_projection_enabled`,
  `semantic_search_enabled`); the
  `coco_blind_spots` review tab id and its default-sort id are renamed
  to `classifier_blind_spots` / `classifier_blind_spots_default`.
- **Env vars, clean break, no aliases.** A
  startup guard (`src/config/retired_env.py`, called from `src/main.py`'s
  lifespan and both worker `main()` entry points) now fails loudly,
  naming the replacement, if any of these are still set:

  | Old | New |
  |---|---|
  | `VLM_URL`, `GEMMA_URL`, `OPENWEBUI_BASE_URL` | `OP_VLM_URL` |
  | `OPENWEBUI_MODEL` | `OP_VLM_MODEL` |
  | `OPENWEBUI_API_KEY` | `OP_VLM_API_KEY` |
  | `VLM_IMAGES_PER_CALL`, `GEMMA_IMAGES_PER_CALL` | `OP_VLM_OPEN_IMAGES_PER_CALL` |
  | `VLM_HTTPX_MAX_CONNECTIONS`, `GEMMA_HTTPX_MAX_CONNECTIONS` | `OP_VLM_HTTPX_MAX_CONNECTIONS` |
  | `VLM_HTTPX_KEEPALIVE`, `GEMMA_HTTPX_KEEPALIVE` | `OP_VLM_HTTPX_KEEPALIVE` |
  | `SAM3_URL` | `OP_SEGMENTER_URL` |
  | `SAM3_URLS` | `OP_SEGMENTER_URLS` |
  | `SAM3_HTTPX_MAX_CONNECTIONS` | `OP_SEGMENTER_HTTPX_MAX_CONNECTIONS` |
  | `SAM3_HTTPX_KEEPALIVE` | `OP_SEGMENTER_HTTPX_KEEPALIVE` |
  | `SAM3_SKIP_VLM_VERIFY_SCORE`, `SAM3_SKIP_GEMMA_VERIFY_SCORE` | `OP_SEGMENTER_SKIP_VERIFY_SCORE` |
  | `SAM_WORKER_VLM_CONCURRENCY`, `SAM_WORKER_GEMMA_CONCURRENCY` | `OP_REGION_WORKER_VLM_CONCURRENCY` |
  | `SAM_WORKER_VLM_VISIBLE_CONCURRENCY`, `SAM_WORKER_GEMMA_VISIBLE_CONCURRENCY` | `OP_REGION_WORKER_VLM_VISIBLE_CONCURRENCY` |
  | `SAM_WORKER_METRICS_PORT` | `OP_REGION_WORKER_METRICS_PORT` |
  | `OP_REGION_DETECTION_SAM_TEXT_PROMPT` | `OP_REGION_DETECTION_SEGMENTER_TEXT_PROMPT` |
  | `GEMMA_CROP_CACHE_DIR` | `OP_CROP_CACHE_DIR` |

  Also: the region worker's `--gemma-url` CLI flag is now `--vlm-url`;
  the segmenter service's `/sam3/segment_plate` and
  `/sam3/segment_plate_batch` path aliases are removed (`POST /segment`
  and `POST /segment/batch` are the only paths now; the shipped client
  posts to `/segment`).
- **Prometheus metric name cleanup.** Every metric
  constant and name in `src/services/curation/metrics.py` moved off the
  legacy metric prefix onto `OP_*`/`op_*`, and
  domain/vendor-named metrics were renamed alongside the prefix swap
  (for example, the combined/separate VLM call counters, the
  segmenter-leg duration and circuit-breaker metrics, and the
  region-detector stage duration). Metrics that carried no domain name
  (`occ_retry_count`, `worker_skip_human_won`, `shm_crop_cache_*`,
  `source_image_*`, `thumbnail_cache_*`, …) kept their name and only
  gained the `op_` prefix.
- **Structured log events use a `curation_` prefix** instead of the
  retired company-initialed one, across the OpenSearch client, ingest,
  index bootstrap, and job/status logging.
- **Training run status fields renamed**: `TrainJobStatus`'s
  `best_metric` / `last_metric` pair is replaced by two distinct rows,
  `last_epoch_metric` (the true last training epoch's metrics) and
  `best_checkpoint_metric` (the best checkpoint's own re-validation
  metrics) — see `docs/design/curation_api_contract.md`'s "Training run
  status" section for why two fields are needed. `Job.migrate_status`
  drops the retired keys from any pre-rename `status.json` on read
  rather than migrating their values, since the two were never the same
  measurement.

### Removed
- `DETECTION_YOLOV5_FORK`; the bake-off CoreML leg and `OP_COREML_HOST`
  (`quantize.coreml` returns 400).

## [0.3.0] - 2026-09-21

### Added
- **Curation subsystem (EXPERIMENTAL)**: a generic active-learning
  curation and labeling stack — class registry, item
  browse/label/move/exclude, clustering (AHC refinement +
  auto-promote), VLM-assisted region labeling/verification,
  review/active-learning queues, YOLO dataset export, and a
  training-job control API — mounted under `CurationConfig.api_prefix`
  (default `/curation`, 109 routes across 25 route groups as of this
  release). Configured through four dataclasses (`CurationConfig`,
  `RegionFields`, `DetectionProfile`, `RegionStatus`) rather than
  forked code. Ships opt-in behind the `curation` Docker Compose
  profile and disabled by default — see
  [`docs/CURATION.md`](docs/CURATION.md) for the user guide,
  `docs/design/curation_design_rationale.md` for the design rationale,
  and `docs/design/curation_api_contract.md` for the HTTP wire
  contract.
- **Curation ingest endpoints**: `POST /curation/ingest/image`,
  `/ingest/batch`, and `/import_labels(/batch)` — duplicate detection,
  a quality gate, crop-cache population, bulk OpenSearch indexing, and
  YOLO-format label import. Includes `scripts/curation/ingest_walker.py`
  for resumable bulk directory ingest.
- **Curation runtime companions**: `curation-detection-worker`,
  `curation-vlm-worker`, `curation-auto-label-worker`,
  `curation-cluster-refresh`, and an on-demand `curation-evaluator`
  bake-off container, all behind `docker compose --profile curation`.
  The detection cascade's segmenter leg is optional — a deployment may
  omit a segmenter entirely.
- **Class-selectable labeling assist**: the VLM prompt pack is
  deployment-supplied and loadable from a JSON file
  (`OP_PROMPT_PACK_PATH`); `GET /curation/methods` advertises the
  active detection profile and prompt pack, and dataset-export
  capability by kind.
- **Shared curation deployment settings**: `GET,PUT /curation/settings`
  — a durable, backend-stored shared default per strategy axis
  (cluster method, sort order, detection profile, prompt pack),
  replacing per-browser client defaults that reset on every reload.
- **Live write-path verification harness**: an isolated, disposable
  Compose stack (`docker/test/compose.yml`, project `op-live-verify`)
  plus a deterministic seed script
  (`scripts/curation/seed_live_harness.py`) and a `tests/live` suite
  (`-m live`) that exercises curation write endpoints against a real
  OpenSearch, a real file-based training protocol, and a fake
  OpenAI-compatible VLM/trainer — the first time any curation write
  endpoint has been run against a live stack rather than mocked
  clients. See `docker/test/README.md`.
- `.github/workflows/ci.yml` — runs the full pytest suite and
  `pre-commit run --all-files` on every PR and push to `main`.
  `requirements-test.txt` provides a CPU-installable dependency subset
  for the CI runner.
- `SECURITY.md`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, GitHub issue
  templates, a pull-request template, `CODEOWNERS`, and `dependabot.yml`.
- `docs/CURATION.md` — the previously-missing curation user guide.

### Fixed
- **Export/training artifact chain**: `class_registry.json` is now
  actually written by dataset export (it was declared but never
  produced), with a dense `export_id_map` so `include_classes`
  filtering on export is no longer inert. Export now does a
  stratified, group-aware split with image copy/resize, replacing an
  unstratified hash split that wrote labels but no pixels.
- **Persistence hardening**: orphaned running-job state is reconciled
  on API startup, and export-task tracking survives a restart instead
  of being lost.
- Restored CORS middleware and several `OP_*` env-var override paths
  (feature flags, crop-cache directory, region-field overrides) that
  had silently diverged between routers and services during the port.
- Removed private absolute host-path defaults from the bake-off
  harness.
- Renamed the reference deployment's company-initialed environment-variable
  prefix to `OP_*` (23 vars) and its matching Prometheus metric-name
  prefix to `op_*`, closing the last reference-deployment naming
  residue in the config surface.
- **Region wire-contract leak**: `GET /curation/crops/{id}`
  returned the raw OpenSearch `_source` (`RegionFields` storage keys,
  `region_*` by default) instead of the frozen `ItemDoc` wire contract;
  `PATCH /crops/{id}/region_meta`'s `updated_fields` echoed
  the same internal keys instead of the request's wire names; and
  `GET /review/{tab}` built its response dict using storage keys as
  literal JSON keys. All three now correctly emit `plate_*`. `ItemDoc`
  gained 11 previously-missing round-trip fields
  (`plate_status`/`plate_text`/`plate_detector`/`plate_verified`/etc)
  that `PATCH .../plate_meta` wrote but no `GET` ever returned. Found
  via a live cross-repo integration test against the Cropwright
  labeler frontend.
- **Shared settings could not be cleared**: `PUT /curation/settings`
  required every submitted value to be a currently-advertised strategy
  id, so once an axis was pinned there was no way back to "each
  endpoint uses its own tuned default" — a `null` value now clears
  that axis's override.

### Removed
- `docs/security/` and `docker/hardened/deepstream/` — an internal
  DeepStream CVE-hardening investigation unrelated to this product
  (the Triton half of that work is kept — see
  `docs/security/triton_cve_hardening.md` and
  `docker/hardened/triton/`).

### Changed
- **License: re-badged MIT → AGPL-3.0-or-later.** This project vendors
  an AGPL-3.0 Ultralytics fork (`src/ultralytics_patches/`); the whole
  repository is now correctly badged to match that copyleft obligation
  instead of the previous (incorrect) MIT badge. See `LICENSE`,
  `README.md`, `ATTRIBUTION.md`, and `pyproject.toml`.
- Test suite hardened: restored dropped OCC-invariant and
  write-guard tests, added coverage for previously-zero-coverage
  detection/clustering leaves, measured `scripts/` for coverage, and
  enforced a coverage floor.

## [0.2.1] - 2026-07-04

### Fixed
- Fresh-install path (`scripts/setup.sh`) on Triton 26.06: trtexec moved
  to `/usr/bin` and its `--fp16` flag was removed in TensorRT 11 — the
  PaddleOCR engine step now works out of the box.
- End2end `config.pbtxt` is written from the built engine's actual output
  dtypes (EfficientNMS_TRT precision varies across TRT releases/builds).
- Health checks in `setup.sh` accept the `/health` -> `/ready` status
  contract.
- CI: valid action pins (trivy-action v0.36.0, checkout v5,
  codeql-action v4) and a scan timeout suited to the image size.
- Endpoint suite: dual-family checks skip gracefully when the optional
  YOLO26 engine is not exported.

## [0.2.0] - 2026-07-04

### Added
- **YOLO26 support served alongside YOLO11** in the same Triton + API
  instance: native NMS-free export (`export/export_yolo26.py`), a
  detection-adapter registry that resolves each model's output contract
  from Triton metadata, `YOLO_MODEL` env for the default detector, and
  per-request selection via the existing `model_name` parameter.
- Dual export toolchains in one image: the proven YOLO11 EfficientNMS
  path keeps its exact pin (`ultralytics==8.3.253`) in an isolated
  `/opt/venv-y11`; `export_models.py` re-execs into it transparently.
- `/live` and `/ready` health endpoints with real per-dependency probes
  (Triton gRPC `is_server_live`, OpenSearch HTTP); `/health` is now an
  alias of `/ready`.
- Prometheus `/metrics` endpoint with an `http_request_duration_seconds`
  histogram labeled by route template.
- dcgm-exporter service + GPU Metrics Grafana dashboard (all host GPUs).
- `make scan` targets and a GitHub Actions Trivy workflow (filesystem +
  API image, SARIF upload, weekly cron).
- Integration test suites: 20 GPU-free pytest tests and a live endpoint
  suite (25 checks) including dynamic YOLO26 load/unload
  (`tests/test_endpoints.sh dual`).
- `docs/MIGRATION_TRITON_26.md` upgrade guide.

### Changed
- **BREAKING: Triton upgraded to 26.06 (CUDA 13.3, TensorRT 11.1)** —
  every existing TensorRT engine must be re-exported; see the migration
  guide. A build-time assertion keeps the server TRT and the
  `tensorrt-cu13` pip pin in lockstep.
- TensorRT 11 is strongly-typed: FP16 is baked into the ONNX at export
  (NVIDIA ModelOpt AutoCast / onnxconverter-common for EfficientNMS
  graphs). Text detection (`paddleocr_det`) defaults to FP32 for
  threshold robustness.
- Monitoring stack pinned (prometheus v3.12.0, grafana 13.1.0, loki
  3.6.12 non-root, node-exporter v1.10.2); Promtail (EOL 2026-03-02)
  replaced by Grafana Alloy v1.17.1.
- OpenSearch upgraded to 3.6.0 (3.0–3.2 carry known HIGH CVEs).
- Triton container runs as the non-root `triton-server` user; both built
  images apply apt security upgrades and scan clean of fixable
  HIGH/CRITICAL CVEs (Nsight Systems CLI removed from the runtime image).
- Triton batching: 25 ms max queue delay; instance counts are a 12 GB
  baseline with scale-up guidance in each `config.pbtxt`.
- gRPC message caps raised to 512 MB for large raw detector heads.
- `/detect` always applies its confidence filter (NMS-free engines emit
  all top-K candidates).
- Request-id context moved to `src.core.logging` (importable by worker
  processes); structlog `foreign_pre_chain` formatter bug fixed.

### Fixed
- `cluster_distance` sort no longer 400s on documents missing the field
  or on freshly created indices.
- Container HEALTHCHECK targets `/live` so a degraded downstream
  dependency cannot cascade restarts through `depends_on`.

## [0.1.0] - 2026-03-19

Initial public release: YOLO11 detection, SCRFD + ArcFace face
recognition, MobileCLIP embeddings, PP-OCRv5 OCR, OpenSearch visual
search, Triton 25.10 TensorRT serving, monitoring stack.
