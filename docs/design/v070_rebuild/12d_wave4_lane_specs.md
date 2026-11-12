# 12d Wave 4 lane specs: scale, the Embedding view, polish and live fakes

| Field | Value |
|---|---|
| Status | spec; wave 4 starts after wave 3 merges and its sweep is green. Ships before the speed review (owner) |
| Interfaces | [12a_workflow_interfaces.md](12a_workflow_interfaces.md) |
| Rules | the "Rules every lane follows" block of [12b_wave2_lane_specs.md](12b_wave2_lane_specs.md) |
| Changed from the plan | cursor paging (plan 4A) moved to wave 2 as lane 2E (12a D-2). Wave 4 lanes: 4A per-image counts, 4B projects at scale, 4C Embedding view, 4D polish, 4E live-harness fakes |

---

## Lane 4A: Per-image label counts (R-13)

**Size:** L, 4 days. **Model:** Sonnet at high effort (a write-path trigger on every label write).

### Goal and outcome

Gallery and Table sort images by "most unlabelled first" and filter `label_state` without a capped join, at
100,000+ images.

### Design

- Images doc fields (mapping migration `ensure_images_count_fields`): `counts.items`, `counts.labeled`,
  `counts.validated`, `counts.unlabeled`, `counts.in_holdout` (integers), `counts_at` (date).
- One coalescing Celery task `refresh_image_counts` (`src/tasks/`), keyed by `image_id`, enqueued after the OCC
  write helpers commit a label, region, exclude, geometry or create write. It is enqueued from one place: the
  label-write and item-write services call `enqueue_image_counts(image_ids)`
  (`src/services/curation/image_counts_refresh.py`), never the routers. Coalescing: a Valkey set of dirty
  image ids plus one beat-driven drain task every 5 s (Celery docs: periodic tasks and task idempotency,
  https://docs.celeryq.dev/en/stable/userguide/tasks.html). The drain runs the same aggregation as
  `image_counts.item_facts` (one terms aggregation per 500 ids) and bulk-updates the images index. It is idempotent:
  recomputed from the items, never incremented.
- `image_view` (`label_state`, the `unlabeled_count:desc` sort option) reads the stored counts when every image
  in scope has `counts_at` (a backfill job sets it). Until then it keeps the join and serves the sort option absent.
  A one-shot backfill job kind `image_counts_backfill` walks all images.
- `ImageCard.counts` keeps coming from `item_facts` for the page (exact at read time); the stored counts serve sort
  and filter only. The served card and the sort never disagree by more than the drain interval; the route
  description says so.

### Files, tests, order

- Create `src/services/curation/image_counts_refresh.py`, the task in `src/tasks/jobs_b.py` (or the module the
  job-family registry names), the job kind in `src/config/job_kinds/`, tests `tests/curation/test_image_counts_refresh.py`,
  a db-tier test for the Valkey coalescing, live `tests/live/test_live_image_counts.py`.
- Red first: `test_label_write_marks_the_image_dirty`, `test_drain_writes_counts_equal_to_an_items_aggregate`,
  `test_unlabeled_sort_is_served_only_after_backfill`, `test_label_state_reads_stored_counts_without_a_join`
  (assert one images-index query, no items aggregation), `test_drain_is_idempotent`.
- Commits: mapping; refresh service and enqueue points; drain task and beat entry; backfill job; `image_view` reads
  stored counts; frontend: nothing beyond the new served sort option (Gallery and Table render served options).
- Live harness: label 3 items on a seeded image, wait for the drain, `GET /images?sort=unlabeled_count:desc` puts
  the right image first.

### Risks and do not

- Do not increment counts; recompute. Do not enqueue from routers. Do not add a per-write Celery task without
  coalescing (one label op of 20,000 items would flood the queue).

### Done when

- [ ] Red tests green; the drain is idempotent; backfill completes on the live harness.
- [ ] Standards gate (12a §13).

---

## Lane 4B: Projects at scale (G-21 remainder)

**Size:** M, 2 days.

- `GET /curation/projects` (wf-1d added `q`, `limit`, `total`): add `status` (repeatable), `selectable` (bool),
  `sort` (`name:asc`, `updated_at:desc`, `items:desc`; served `sort_options`) and `cursor`/`next_cursor` with
  `list_cursor` (Postgres keyset: `ORDER BY <sort>, slug` with a `WHERE (sort_value, slug) > (...)` clause).
- Frontend: `ProjectPickerController` (`frontend/src/lib/projects/projectPickerController.svelte.ts`) sends
  `selectable=true` and deletes its client-side `filter((p) => p.selectable)` (12a C-4); `/projects` page uses the
  cursor pager, a search box, the status filter and the served sort; recent projects in `sessionStorage` (view
  state only).
- Red first: `tests/projects/test_projects_list.py::test_selectable_filter`, `::test_cursor_walks_all_projects`,
  `::test_sort_options_served`; vitest `projectPickerController.test.ts::sends selectable and shows the served total`.
- Visual: the switcher with 60 public test projects at 1600 and 430 (scrolls inside its panel, search narrows).
- Done when: red tests green, screens viewed, standards gate.

---

## Lane 4C: Embedding view (R-23)

**Size:** M, 3 days.

### Goal and outcome

The existing scatter (`frontend/src/lib/components/EmbeddingPlot.svelte`, props `:55-77`) becomes the
Workspace's **Embedding** view (owner answer 6). It follows the shared filter (non-matching points dimmed), colours
by class from a served legend, and a box drawn on it becomes a `viz_box` chip that filters every view.

### Backend (R-23)

- `GET /viz/projection` (`src/routers/curation/viz.py:137-186`) takes `ItemFilterQuery` (replacing its own
  `cluster_uid` and `class_id` params; one mechanism). Points not matching are served with `matched: false`
  (`ProjectionPoint` `:112` gains `matched: bool`). The service `embedding_viz_read.get_cached_projection`
  (`:79-140`) evaluates the filter as a second `terms crop_id` pass over the matched set, capped like
  `max_points`.
- `ItemFilter.viz_box` (12a §3.1): `range viz_x` and `range viz_y`. Validation 422 for an inverted box.
- `ProjectionResponse` gains `legend: [{class_name, colour_index, count}]`, served (the frontend maps
  `colour_index` to a token colour, never a class to a colour itself).
- Red first: `tests/curation/test_viz_projection_filter.py::test_unmatched_points_are_dimmed_not_dropped`,
  `::test_viz_box_count_equals_a_range_query`, `::test_legend_is_served`.

### Frontend

- `frontend/src/lib/workspace/views/EmbeddingView.svelte` mounts `EmbeddingPlot` with the controller's
  `itemQuery()`; box select calls `ws.filter` to set `vizBox` and `applyFilter()`. The plot's mount in
  `ClustersGrid.svelte:226` and the `showEmbeddingViz` props (`:21-47`) are deleted.
- Keymap: no new letters (`cluster` context applies to the details panel; the plot itself is pointer-driven).
- Tests: vitest `EmbeddingView.test.ts::a drawn box becomes a viz_box chip`, `::unmatched points render dimmed`;
  stubbed `test_embedding_view.py` (draw a box, Gallery is filtered, 430 px); a11y row `workspace-embedding`.
- Visual: the view with a class filter (dimmed rest) and a drawn box. "Right": spot mislabels inside the wrong
  cloud; the legend's counts are the served ones.
- Docs: `docs-site/docs/cropwright/user-guide/embedding-view.md`; changelogs.
- Done when: red tests green, `ClustersGrid` no longer mounts the plot, screens viewed, standards gate.

---

## Lane 4D: Polish

**Size:** S each, about 4 days in total; one commit per item. Items:

| Item | What | Backend | Red test |
|---|---|---|---|
| G-19 | Overlay on busy images: item boxes labelled; region boxes drawn only for the selected item unless "All regions" is on (a `BoxLayer` prop) | none (R-24 gave the names) | `BoxLayer.test.ts::region boxes only for the selected item by default` |
| G-20 | Card chips: served `clarity_label` replaces "b0.14"; rank shows "largest", "2nd" from a served `rank_label` | `wire.py` adds `clarity_label`, `rank_label` (thresholds in the backend) | `tests/curation/test_wire_derived_fields.py::test_clarity_and_rank_labels` |
| G-22 (D-22 overridden by the owner on 2026-11-12) | Single label writes recorded as label ops, so undo is one served history. Scope: `PUT /crops/{crop_id}/label` stays synchronous and records its op in the request (`recorded_relabel` with no owning job; `start_relabel` queues a job and would make the write asynchronous); `undoStore.recordWrites` becomes `recordLabelOp`. The editor keeps its own command stack for geometry and undoes a class change through the served op (owner ruling 2026-11-12) | `crops_write.py`, label ops store | `tests/curation/test_label_ops.py::test_single_label_write_is_a_label_op` |
| G-23 | One "VLM" menu in the Workspace view bar (Assist on the selection or scope; loop status; Auto-label run), replacing the three entry points | none | `VlmMenu.test.ts::offers assist only when a VLM is active (served)` |
| G-27 | A failed thumbnail retries once after 1 s, then shows the served source-missing badge (`fetchSourceMissing`) | none | `ImageCard.test.ts::retries a failed thumbnail once` |
| G-28 | Served pages for a finished diverse selection (`GET /select/diverse/{job_id}/items?cursor=`), served counts on `GET /training_cohorts`, `sort` on `GET /stats/classes`; delete W-9, W-10, W-12 client code (plan §5) | three route changes | one per route |
| Minimap | Editor minimap for photos larger than 4 times the viewport (a 160 px `BoxLayer` thumbnail with the viewport rectangle; click to pan) | none | `EditorCanvas.test.ts::minimap pans on click` |
| Dup group | "Label this whole duplicate group" from a near-dup representative (owner answer 3): a `BulkBar` action over `{filter: {near_dup_group}}` with the served dry run | none (2B's filter) | `test_near_duplicates.py::label_the_whole_group_uses_the_group_filter` |

Each item: its visual check at 1600 and 430, a changelog line, standards gate.

---

## Lane 4E: Live-harness fakes for the core inference paths (cov-live gaps)

**Size:** M, 3 days. Test infrastructure; no product code changes.

- The fake Triton (cov-live, merged at `1685723a9`) serves the detector, the PE image and text encoders and OCR.
  Add fakes for SCRFD (faces), ArcFace (face embeddings) and MobileCLIP (image and text embeddings), returning
  deterministic, shape-correct outputs from the models' `config.pbtxt` (`models/<name>/config.pbtxt`), so core
  `POST /ingest` and the OCR search path can be live-tested.
- A fake trainer for the `training` profile that writes a small valid ONNX (built once with `onnx.helper` in the
  test image, a one-layer graph with the detector's output names), so the promote success path is live-tested.
- Tests: `tests/live/test_live_core_ingest.py` (ingest a public COCO image through the fakes: faces, embeddings and
  OCR fields present), `tests/live/test_live_train_promote.py` (train job finishes, promote succeeds, the promoted
  model appears in `/models`).
- Read first: the cov-live fake's module (find it with `rg -l "fake" docker/test`), the Triton model configs, the
  trainer file protocol (`src/services/training/`).
- Done when: both live tests green under `-p op-wf-4e`, torn down; standards gate.

---

## After wave 4: the full-loop acceptance run

Run by the coordinator on the merged tree: [12e_workflow_execution_order.md](12e_workflow_execution_order.md),
"Live acceptance walk".
