# Selectable VLM and segmenter assist: per run, cluster, sub-cluster, selection, box and crop

Status: plan, not implemented. Audience: a fresh implementation agent (Sonnet) with no
conversation memory. Base: `origin/main` at `bbe929a2` **plus** PR #329
(`feat/workflow-controls`, not yet merged). Every line number below is from `bbe929a2`
unless marked `(#329)`. Before starting PR 1, rebase on `main` after #329 merges; if it has
not merged, branch from `feat/workflow-controls` and say so in the PR body.

Read first: `CLAUDE.md`, `docs/VISION_AND_GOALS.md`, `CONTRIBUTING.md` (ratchets, baselines),
`docs/design/label_confirmation_plan.md` (the VLM scope history).

Notation: `{P}` = `/curation/projects/{project}`. Planned routes are written `{P}/assist/...`
because `scripts/docs/check_docs_vs_code.py` rejects any doc that names a route the app
does not serve yet. In user docs (PR 4 onward), write the routes in full once they exist.

## 0. The requirement, and the short answer

Owner requirement:

- Ingest many images, label quickly with clustering.
- The VLM and the segmenter (SAM 3) are **selectable, never hard-coded**, at these scopes:
  a whole project run, a cluster, a **sub-cluster** (from the AHC refine step), a
  user-selected set of crops, a single box and a single crop.
- **VLM-assisted cluster labeling** where the user picks what the VLM sees: the selected
  crops, all crops of the cluster or sub-cluster, or a **random representative sample**
  whose size grows with the scope size (bounded formula, stratified by sub-cluster when
  there are sub-clusters, reproducible with a seed).
- The VLM result (a) proposes the cluster or sub-cluster label while the user refines the
  cluster and labels items, and (b) returns **assist information** (free-form attributes
  declared by the prompt pack, for example car make or material). These are suggestions
  only. They never overwrite a human- or import-locked label (lock rule).
- Segmenter proposals, and VLM validation of the new boxes, are selectable per run and per
  selection.
- Scopes are addressed by stable ids or filters. The assist layer must not depend on the
  clustering method, because the owner will rework clustering later. Re-running after a
  re-refine must not clobber locked labels.

Short answer from the code:

1. A whole-project VLM run exists (auto-label job, `vlm=<name>`, `vlm_scope`). A whole-cluster
   run exists (the route `/vlm/label_cluster/{cluster_id}`, every unvalidated member, no
   sampling). Selected crops and single crops work only through the synchronous
   `/vlm/label_batch` (≤64). Nothing sends a sub-cluster, one box or a sample to the VLM.
2. There is no sample-by-size, no stratified sample and no cluster-level proposal. The
   VLM labels items one by one, and nothing aggregates the answers into a proposal for
   the cluster.
3. There are no free-form attributes. The only extra outputs are the hard-coded `make` and
   `model` keys (`src/services/labeling/vlm_prompts.py:490`). They are written to
   `vlm_item_make` / `vlm_item_model` but never reach the wire, so the UI cannot show them.
4. Segmenter verification has no switch. It runs whenever the project has an active VLM,
   and it runs per item. It cannot be restricted to selected boxes or switched off per run.
5. Sub-clusters are positional labels (`"47a"`, `"47b"`, ...) that a re-refine rewrites
   (`refine.py:318-322`). No id survives a re-refine, and the crop list cannot filter by
   sub-cluster (the frontend filters the loaded page in memory).

The plan adds one selection-scoped **assist run** (a background job, in the API process,
like reprocess). It has one request shape for every scope, a method-independent sampling
function, pack-declared attributes, a cluster proposal with a membership-overlap staleness
check, a per-run segmenter-verify choice, and a Cropwright dialog that renders what the
backend decided. `/vlm/label_cluster/{id}` and `/vlm/verify_regions` are deleted in the PRs
that replace them.

## 1. What exists today, per scope (file:line)

All curation routes are mounted at `/curation/projects/{project}` + path
(`src/routers/curation/_mounting.py:29-61`).

### 1.1 Whole project run

| Piece | Where | Notes |
|---|---|---|
| `POST /pipeline/auto_label/start` | `src/routers/curation/pipeline_start.py:43-168` | Query params only: `run_vlm` (default false), `vlm`, `prompt_pack`, `acknowledge_external`, `vlm_scope`, `max_vlm_crops` (0–100000), `vlm_batch_size` 4–64, `vlm_concurrency` ≤128, `class_id`, `cluster_id`, `embed_missing`, plus `ItemFilterQuery` (`_item_filter_params.py:21-84`). No `item_ids`. |
| Job | `src/services/curation/autolabel/job.py:451` `start_job` writes `trigger.json`; `scripts/curation/auto_label_worker.py:300` claims it; `src/services/curation/auto_label_run.py:224` `run_auto_label` | One job per project (409). States `_JobState` `job.py:115-145`. |
| Selection | `src/services/curation/autolabel/selection.py:51-98` `vlm_selection_query`, `:110-163` `resolve_vlm_selection` | Embedded, not validated, not classifier-confident, VLM policy and daily budget. |
| VLM stage | `src/services/curation/autolabel/vlm_stage.py:323` `run_vlm_stage`, `:242` `label_or_propose_batch`, writes at `:201-226` (`writer_id='vlm_pipeline'`); extras `vlm_item_make`/`_model` at `:159-165` | (#329) adds `failed`/`unreadable` counts and fails a stage that labeled nothing. |
| Continuous worker | `scripts/curation/vlm_worker.py:78-99` posts `/vlm/label_batch`; selection `scripts/curation/vlm_pending.py:34-132` | (#329) skips a project with no VLM activation. |
| VLM policy | `GET/PUT /vlm/policy` `src/routers/curation/vlm_policy.py:21,29`; `VlmPolicyBody` `src/services/curation/vlm_policy.py:32-39` (`scope` all/uncertain/representatives/off, `conf_max`, `per_cluster`, `max_crops_per_day`, `sample_frac`) | Limits automated writers only. |
| Activation | `src/routers/curation/vlm_activation.py`; `src/services/labeling/vlm_endpoints.py:429` `active_vlm_endpoint` | (#329): no activation = no VLM (`source: "off"`); `vlm=<name>` per run still works. |
| Run-time endpoint resolution | `src/services/curation/autolabel/run_vlm.py:67` `resolve_run_vlm`, `:131` `labeler_unavailable`; `src/routers/curation/vlm.py:55` `request_labeler`; `src/services/labeling/run_labeler.py` `get_vlm_labeler`, `resolve_pack`, `vlm_stamp` | Reuse these. Do not write a second resolver. |
| Segmenter, whole project | Region worker `scripts/curation/worker/` (`__main__.py:32-98`), pending query `cascade.py:44-88` (status `pending_detection`/`pending_verification` + `parent_classes`) | Runs whenever a region profile is active and the stage is not paused (`src/services/curation/region_stage_control.py:1-12`). |
| Segmenter-verify switch | `scripts/curation/worker/runtime.py:205-206` (`vlm_available`), `stage_sam.py:202-209` (`_queue_for_verification`: combined VLM call, else `accept_without_vlm` in `region_text_stage.py:270`) | **No flag.** VLM verify happens iff the project has an active VLM. `OP_SEGMENTER_SKIP_VERIFY_SCORE` (`verify.py:64`) skips verify for one confident candidate. |

### 1.2 A cluster

| Piece | Where | Notes |
|---|---|---|
| the route `/vlm/label_cluster/{cluster_id}` | `src/routers/curation/pipeline_control.py:69-104` | Path `cluster_id: int`; query `prompt_pack`, `vlm`, `acknowledge_external`. Wraps `pipeline_auto_label_start(train_clusters=False, run_vlm=True, run_auto_promote=False, max_vlm_crops=0, cluster_id=...)`. Labels **every** unvalidated, non-holdout, non-excluded member (`selection.py:70-76`, `:129-137`). No sample, no proposal, no sub-cluster. |
| `?cluster_id=` on auto-label start | same job | Same scope. |
| Frontend | `frontend/src/lib/api_autoLabel.ts:41` `runVlmOnCluster`; `clusterDetailActions.svelte.ts:130` `runVlm`; `ClusterDetailToolbar.svelte:198-226` ("Run VLM" + `VlmRunPicker`); dashboard modal `routes/p/[project]/dashboard/+page.svelte:131-160,285-330` | The server picks members; the UI sends no ids. |
| Human cluster labeling | `PUT /crops/batch_label` `src/routers/curation/crops_write.py:79`, `CropBatchLabelRequest(SelectionTargets)` `_common_models.py:103`, OCC `occ_bulk_human_relabel` `crop_bulk.py:16` | Human write. |
| Segmenter on a cluster | No route. Workaround: `POST /reprocess` with `targets.filter.cluster_id`, `scopes:['region']` (queues for the region worker). | |

### 1.3 A sub-cluster (AHC refine)

| Piece | Where | Notes |
|---|---|---|
| `POST /clusters/refine/{cluster_id}` | `src/routers/curation/clusters.py:425-455` (`distance_threshold` 0.05–1.0, `max_members` ≤50000) | Calls `refine_cluster` `src/services/curation/clustering/refine.py:348`. |
| Sub-id assignment | `refine.py:303-322`: `AgglomerativeClustering(... linkage='complete', metric='cosine')`, then `f'{cluster_id}{subcluster_label(idx)}'` (`:67`) | **Positional labels.** A re-refine relabels every member from `a`, so `"47a"` before and after a re-refine can be different sets. No partition id or run id is stored. |
| Write | `refine.py:132-189` `_bulk_update_subids`, guarded painless (`noop` if `cluster_id` changed) | Item field `cluster_subid` (keyword), `bodies_core.py:303-308`. |
| Cleared when the cluster changes | `src/services/curation/clustering/id_normalize.py:107-120,230-232`; `orchestrator.py:234,321,380` | |
| Region-box sub-clusters | `POST /regions/clusters/refine/{id}` `regions_fp.py:118`; box keys `cluster_id`/`cluster_subid`/`cluster_distance` in nested `region_boxes` (`bodies_core.py:196-198`) | `GET /regions?region_cluster_subid=` exists (`regions.py:73`). |
| Item filter | `src/services/curation/item_filter.py:39-79` `ItemFilter` has `cluster_id` but **no `cluster_subid`** | |
| Cluster cards | `GET /clusters` `clusters.py:125` returns `n_subclusters` (`:251,291,329`) | |
| Frontend sub-cluster tab | `frontend/src/lib/clusters/clusterDetailController.svelte.ts:122,261-273` | Filters the **loaded page in memory** (`cropPager.items.filter(c => c.cluster_subid === subTab)`). A sub-cluster larger than a page is never shown whole. This is a frontend decision that belongs in the backend. |
| VLM on a sub-cluster | **None.** | |

### 1.4 A user-selected set of crops

| Piece | Where | Notes |
|---|---|---|
| `POST /vlm/label_batch` | `src/routers/curation/vlm.py:113-289`; body `VlmLabelBatchRequest` `_vlm_route_models.py:9-12` | Synchronous, `crop_ids` ≤64 (route, `vlm.py:123`; the schema allows 5000). Query `vlm`, `acknowledge_external`; **no `prompt_pack`** (always the settings default, `vlm.py:64`). Skips locked/validated (`:180`) and holdout (`:254-264`). Does not write make/model (`label_batch_write.py:44`). Used by `vlm_worker`; the frontend no longer calls it (`api_autoLabel.ts:34`). |
| `POST /vlm/verify_regions` | `vlm.py:292-404`; body `crop_ids` ≤64 | Synchronous; every verifiable box of each crop gets one sequential `labeler.verify_region` call (`:353-364`); writes `verify_regions_update` (`src/services/curation/region_verify.py:49-122`). |
| `POST /vlm/verify_region_batch`, `/vlm/region_visible_batch` | `vlm.py:407-497`, `:500-562` | Write nothing. |
| `POST /reprocess` | `src/routers/curation/reprocess.py:81`; `ReprocessRequest` `src/services/curation/reprocess_models.py:84-97`; targets `:58` (exactly one of `image_ids`/`crop_ids`/`filter`, `limit`, `sample`, `seed`; `MAX_TARGET_IDS=5000`) | Scope `vlm` does **not** call the VLM; it clears VLM answers so the worker picks them up again (`reprocess_vlm.py:55-96`). Scope `region` re-queues for the region worker (`reprocess_region.py`, `region_requeue.py:388-406,433`). |
| Shared selection | `ItemSelection` `src/services/curation/item_selection.py:36-67` (`crop_ids` ≤5000 or `filter`, `limit`, `sample` random/largest, `seed`, `include_test`, `include_excluded`; resolver `:134`); `SelectionTargets` `src/routers/curation/_selection.py:25-58` (`MAX_BULK_SELECTION=20_000`) | Used by the crop write routes. Auto-label and reprocess have their own shapes. |
| Frontend multi-select | `frontend/src/lib/selection.svelte.ts:32`; bulk bar `ClusterDetailToolbar.svelte:106-235` (assign class, move, flag, Reprocess selection `:177`, Accept VLM page `:183`, Run VLM `:198`); filter selection `itemFilter/selectionActionController.svelte.ts:30,48` | No "ask the VLM about these" action. |

### 1.5 A single box

- **No route sends one chosen box to the VLM or the segmenter.** `verify_regions` sends all
  boxes of a crop; the worker's combined call sends all candidates; reprocess `reverify`
  re-proposes all unlocked rejected boxes of the item.
- Box ids appear only on human edit routes: `PATCH /crops/{crop_id}/regions/{box_id}`
  (`regions_boxes_edit.py:348`), `POST /regions/batch_box_state` (`:406`).
- `BoxVerdict` carries `box_id` (`region_verify.py:31-39`); the merge drops a verdict whose
  box moved (`same_box`, `:83`).
- The segmenter takes text prompts only (`docker/segmenter/main.py:292,310`, `SegmentRequest`
  `:94`): there is no box or point prompt. Running "the segmenter on a box" means
  re-segmenting the item crop with the profile's text prompt. The worker's text-hint
  re-pass crops a sub-region (`cascade.py:263-277`), but no route exposes it.
- Read-only preview: `POST /region_profiles/test` (`config_test.py:383`, one `crop_id`,
  `verify: bool`).

### 1.6 A single crop

- A one-element `crop_ids` on `/vlm/label_batch` or `/vlm/verify_regions`.
- `POST /crops/{crop_id}/reprocess` and `POST /images/{image_id}/reprocess`
  (`reprocess.py:123,93`, `ReprocessOneRequest` `reprocess_models.py:100-108`). In the
  frontend: `ReprocessControl` in `CropMetaPanel.svelte:210`, `CropCard.svelte:519`.
- Read-only previews: `/prompt_packs/test` (`config_test.py:195`, ≤64 crops),
  `/region_profiles/test`.

### 1.7 Prompt packs, reply parsing, storage, lock rule

- Pack: `PromptPack` `src/services/labeling/vlm_prompts.py:45-109`; wire `PromptPackBody`
  `src/routers/curation/_prompt_pack_models.py:17-62`. Fields: `class_*`, `open_class_*`,
  `combined_*`, `region_*`, `region_visible_*`, `class_descriptions`, `synonyms`,
  `proposal_denylist`, `detector_hint_min_confidence_pct`. `from_dict` drops unknown keys
  (`:116-127`). Reply contract `REPLY_KEY_CONTRACT` `:486-531`. Validation
  `src/services/config_store/pack_validation.py` (`MAX_FIELD_LEN=20000`,
  `MAX_MAP_ENTRIES=500`, `:29-31`). Call to writer map: `src/routers/curation/prompt_packs.py:113`.
- Parse: `VlmClassPrediction` `vlm_models.py:87-124` (make/model ≤48, parsed at
  `vlm_labeler_classify.py:236`); combined `vlm_reply_parse.py:111-170` (make at `:142`).
- Item class write: `prediction_class_update` `src/services/curation/vlm_class_attempt.py:116-180`.
- Lock rule: `src/clients/occ_locks.py:40-97` (`is_locked_class`: human source, validated
  import, `test_holdout`; `is_locked_box`; `is_locked_item`). VLM class writers also refuse
  any `class_validated` item (`src/services/curation/class_write_guard.py:65-68`), and
  `ClassWriteGuard.allows` needs the class unchanged since read.
- Box schema: `RegionBox` `src/domain/region_boxes.py:49-72` (fixed keys; guarded by
  `tests/test_no_legacy_region_scalars.py`).
- Cropwright display: VLM chip `CropCard.svelte:441-475`; `CropMetaPanel.svelte:116-170`;
  provenance `provenance/VlmProvenanceRows.svelte:14-16`; review `ReviewItemMeta.svelte`,
  `ReviewSlotPanel.svelte:121,223-246`. Wire types are hand-written (`frontend/src/lib/types*.ts`,
  mapper `api_cropMap.ts:40-81`) and pinned by contract tests in
  `frontend/src/lib/contract/*.test.ts` against `contracts/`.

### 1.8 A bug found during the trace (fix in PR 0)

`POST /reprocess` with `targets.filter` + `limit`/`sample` caps only the image scopes
(`src/services/curation/reprocess.py:140-147`). The `region` scope calls
`apply_region_filter(opensearch, filter, mode)` (`reprocess.py:371-376`), and the `vlm`
scope pages the whole filter (`reprocess.py:381-395`, `_plan_vlm` `:177-204` returns
`None` ids for a filter). A "sample 50" region or vlm reprocess runs on every match.

## 2. Gap table

| # | Scope | VLM classify / propose | VLM attributes | Segmenter | VLM verify of new boxes | Gap |
|---|---|---|---|---|---|---|
| G1 | Whole project | Yes (auto-label, `vlm=`) | No | Yes (region worker, profile active) | Implicit: iff the project VLM is active | No attribute output. No per-project verify switch other than deactivating the VLM. |
| G2 | Cluster | All members only (`label_cluster`) | No | Only via reprocess filter | Implicit | No sample choice, no proposal, no attributes, no per-run verify choice. |
| G3 | Sub-cluster | No | No | No (no `cluster_subid` filter) | No | Not addressable. Positional ids; no staleness check. |
| G4 | Selected crops | Sync ≤64 (`label_batch`), no pack choice | No | Via reprocess `crop_ids` | Via sync `verify_regions` ≤64, all boxes | No job, no pack, no attributes, no cluster context. |
| G5 | Single box | No | No | No | No | Missing entirely. |
| G6 | Single crop | One-element `label_batch` | No | `crops/{id}/reprocess` | One-element `verify_regions` | No attributes; verify not selectable. |
| G7 | Sampling | `representatives` policy (nearest to centroid, clustering-method dependent) | — | — | — | No size-scaled, stratified, seeded sample. |
| G8 | Output | Per-item labels only | Hard-coded make/model, not on the wire | — | — | No cluster proposal, no attribute display. |
| G9 | Reprocess caps | — | — | `limit`/`sample` ignored | — | Bug (1.8). |
| G10 | Frontend | Run VLM on a whole cluster | — | ReprocessControl | — | No assist dialog. Sub-cluster tab filters in memory. |

## 3. Design

### 3.1 Principles

- **One request shape for every scope.** `POST {P}/assist/runs`, with a `target` (what is in
  scope), a `sample` (what the VLM sees), `actions` (what to do) and the model choices.
  Whole-project runs stay the auto-label job (PR 9 gives it the same attribute output
  through the shared core). Assist runs are capped at 5000 VLM items, see 3.4.
- **Method-independent scopes.** A cluster or sub-cluster target is a filter on stored
  fields (`cluster_id`, `cluster_subid`). The run records the **resolved member ids and a
  membership digest**. Staleness is measured as membership overlap, never by trusting a
  sub-id string across a re-refine. Nothing in the assist layer imports
  `src/services/curation/clustering/`.
- **Suggestions, never authority.** Class suggestions go through the existing VLM class
  writer (`prediction_class_update` + `ClassWriteGuard` + OCC), so locked, validated and
  holdout items are never written. Attributes go to a separate `assist` object on the
  item, which no export, train or holdout path reads. A cluster proposal is stored on the
  run and is applied only by an explicit user call.
- **Fail closed.** No VLM (project off, no `vlm=`) → 409 `vlm_not_configured`. Unknown
  cluster → 404. Stale scope on apply → 409 `scope_changed`. Too large → 422. A segment
  action with no region profile, no segmenter leg, a paused region stage or a dead
  region worker → 409 with a specific code.
- **Class identity is the name.** Proposals, votes and apply requests carry `class_name`.
  The server resolves the name to the project registry at apply time.

### 3.2 Wire: `POST {P}/assist/runs`

Request `AssistRunRequest` (`extra='forbid'`), in `src/schemas/curation/assist.py`:

```jsonc
{
  "target": {                       // exactly one kind
    "kind": "crops",                // single crop = one id; user selection = many
    "crop_ids": ["..."],            // 1..5000, deduplicated, caller order kept
    "context": {"cluster_id": 10000, "cluster_subid": "10000b"}   // optional: attach the proposal to this (sub-)cluster
  },
  // or {"kind": "boxes", "boxes": [{"crop_id": "...", "box_id": "..."}]}   1..1000 pairs
  // or {"kind": "cluster", "cluster_id": 10000, "cluster_subid": null | "10000b"}
  // or {"kind": "filter", "filter": <ItemFilter>}                         non-empty filter required
  "sample": {
    "mode": "all" | "representative",   // "selected" is implied by kind crops/boxes: mode must be "all" there (422 otherwise)
    "size": null | 1..256,              // representative only; null = the formula in 3.4
    "seed": 0,                          // representative only
    "stratify_by": "auto" | "cluster_subid" | "none"   // auto = cluster_subid when the scope has >1 distinct sub-id, else none
  },
  "actions": ["classify", "attributes", "segment", "verify"],  // non-empty subset; see 3.3
  "vlm": null | "<endpoint name>",      // null = the project's active endpoint (409 vlm_not_configured when off)
  "prompt_pack": null | "<pack name>",  // null = the project's effective default pack
  "acknowledge_external": false,
  "segment": {"verify": "vlm" | "none"},  // required iff "segment" in actions
  "min_agreement": 0.6,                 // proposal gate, 0.5..1.0
  "dry_run": true
}
```

Validation (422 `invalid_request`, never truncation):

- `kind: crops` / `boxes` with `sample.mode: representative` → 422. Selection is the sample.
- `kind: filter` with an empty `ItemFilter` → 422 (`ItemFilter.is_empty`). Whole project = auto-label.
- `kind: boxes` with `classify` → 422 (boxes have no class).
- `actions` containing `segment` with `kind: boxes` → 422 (the segmenter takes a text
  prompt over the item crop; it cannot re-segment one box, see 1.5).
- `segment.verify: "vlm"` with a `vlm` that is not the project's active endpoint → 409
  `verify_endpoint_mismatch`. The region worker builds one VLM per project
  (`runtime.py:205`). To verify with another endpoint, run `segment` with `verify: none`,
  then a `verify` run on the new boxes with `vlm=<name>`. This is open decision D3.

Response `AssistRunState` (also from `GET {P}/assist/runs/{run_id}`):

```jsonc
{
  "run_id": "asr_...", "status": "dry_run" | "queued" | "running" | "waiting_region" | "completed" | "completed_with_errors" | "failed" | "cancelled" | "lost",
  "created_at": "...", "finished_at": null, "error": null,
  "request": { ...normalized request... },
  "resolved": {
    "scope_count": 412,             // members of the target after visibility rules
    "scope_digest": "sha256:...",   // over the sorted resolved ids
    "sampled": 41, "sample_seed": 0,
    "strata": [{"key": "10000a", "size": 120, "sampled": 12}, ...],   // [] when not stratified
    "skipped": {"holdout": 3, "excluded": 1, "class_locked": 27, "box_locked": 0, "not_found": 0, "unembedded": 0},
    "vlm": {"name": "env", "revision": 3, "model": "..."}, "prompt_pack": {"name": "...", "revision": 7},
    "estimated_vlm_calls": 6
  },
  "progress": {"processed": 0, "total": 41},
  "result": null | {
    "classify": {"predicted": 41, "updated": 30, "failed": 0, "unreadable": 0, "locked_suggest_only": 11},
    "attributes": {"items_written": 41, "invalid_values": 2},
    "verify": {"boxes": 0, "accepted": 0, "rejected": 0, "locked_skipped": 0},
    "segment": {"queued": 0, "done": 0, "pending": 0, "failed": 0},
    "proposal": null | {
      "class_name": "spoiler" | null, "in_registry": true,
      "agreement": 0.83, "answered": 41, "min_agreement": 0.6, "passes_gate": true,
      "distribution": [{"class_name": "spoiler", "votes": 34}, ...],
      "new_class_proposals": [{"name": "rear_wing", "votes": 3}],
      "attribute_summary": [{"key": "make", "values": [{"value": "toyota", "count": 12}, ...], "answered": 38}]
    }
  }
}
```

Other routes (all project-scoped, all through `src/services/curation/wire.py` for item
data, all added to `tests/curation/cross_project_leak/_route_tables.py`):

| Route | Purpose |
|---|---|
| `GET {P}/assist/runs/{run_id}` | State as above, plus `scope_now: {count, overlap, jaccard, changed}` when `target.kind` is `cluster` or `context` is set (members now vs `resolved`). |
| `GET {P}/assist/runs?cluster_id=&cluster_subid=&limit=20` | Newest runs whose target or context names that scope; the newest 100 runs per project are kept. |
| `POST {P}/assist/runs/{run_id}/cancel` | Cancel; `processed` stays. |
| `POST {P}/assist/runs/{run_id}/apply` | Body `{class_name, label_source: "human", allow_changed_scope: false, dry_run: true}`. Applies the class to the scope's **current** members that are not class-locked, not validated, not holdout and not excluded, through the same OCC human relabel service as `PUT /crops/batch_label` (`crop_bulk.py:16`, no second writer). 409 `scope_changed` when `jaccard < 0.9` and not `allow_changed_scope`. 409 `class_not_in_registry` when the name is not a class (the UI offers to create it). 409 `run_has_no_scope` for a run without a cluster target or context. Response `BatchRelabelResponse` (`_selection_write_models.py`) plus `skipped_locked`. |

Deleted (no parallel routes):

- The route `/vlm/label_cluster/{cluster_id}` (PR 4), replaced by `target.kind: cluster` +
  `sample.mode: all` + `actions: ["classify"]`.
- `POST /vlm/verify_regions` (PR 6), replaced by `actions: ["verify"]` on `crops` or `boxes`.
- `?cluster_id=` on `auto_label/start` stays. It is the whole-project job with a narrowing
  filter, not a duplicate of the assist run.
- `/vlm/label_batch` stays. It is the continuous worker's transport. The frontend does not
  call it (`api_autoLabel.ts:34`).

### 3.3 Actions

| Action | Targets | What runs | Writes |
|---|---|---|---|
| `classify` | crops, cluster, filter | The pack's `assist` call when `attributes` is also chosen and the pack declares item attributes, else the existing classify call (`label_or_propose_batch`) | Class suggestion through `prediction_class_update` + `ClassWriteGuard` + `occ_skip_on_conflict_bulk` (as `vlm_stage.py:201-226`), `writer_id='assist_run'`. **Locked or validated items:** class fields untouched; the answer goes to `assist.class_name` only (counted `locked_suggest_only`), so the UI can show "VLM disagrees". Every sampled answer is a vote for the proposal. |
| `attributes` | crops, boxes, cluster, filter | The `assist` call (with classify) or the `attributes` call (attributes only); box targets send the box sub-crop (`load_region_jpeg`, `src/services/curation/crop_bytes.py`) with the `applies_to: box` attributes | `assist` object on the item (3.6). Never touches class or box fields. Written even on locked items: attributes are suggestions, not labels. |
| `verify` | crops (every verifiable box), boxes (only the named boxes) | `labeler.verify_region` per box, as `vlm.py:353-364`, moved into the assist service with a `box_ids` filter | `verify_regions_update` (`region_verify.py:49-122`), which already skips locked boxes and human region sets. |
| `segment` | crops, cluster, filter | Re-queue the resolved items for the region worker with `region_mode: redetect` through `apply_requeue_ids` (`region_requeue.py:433`), stamping `region_run_id` and `region_run_verify` (3.6). The worker honours `region_run_verify: none` by taking `accept_without_vlm` even with a VLM available, and `vlm` by the combined call. The run waits in `waiting_region` until no item of the run is pending, then completes. | Through the worker's existing writer. `region_validated` items are skipped (`reprocess_locks.region_set_locked`). |

Order inside one run: `segment` (and wait), then `verify`, then `classify`/`attributes`.
`segment` + `verify` on the same run verifies the new boxes with the run's endpoint after
the worker wrote them with `verify: none`. That makes "SAM, then VLM validation" one run
whose endpoint is selectable.

Project-level switch (G1): add `region_verify: "vlm" | "none"` (default `"vlm"`) to
`VlmPolicyBody` (`src/services/curation/vlm_policy.py:32-39`). The region worker reads it
through the config snapshot it already hot-reloads (`runtime.py:112`), and `none` routes to
`accept_without_vlm`. This makes the whole-project verify selectable without deactivating
the VLM. A per-item `region_run_verify` wins over the policy.

### 3.4 Sampling (pure, method-independent)

New module `src/domain/assist_sampling.py` (imports only stdlib; `domain` may import
`config` and `utils` only):

```python
SAMPLE_FLOOR = 8        # below this a scope is sampled whole
SAMPLE_SQRT_FACTOR = 2.0
SAMPLE_DEFAULT_CAP = 64
SAMPLE_HARD_MAX = 256
MAX_ASSIST_ITEMS = 5000 # VLM items per run (sample size, or scope size for mode "all")

def representative_size(n: int, *, size: int | None = None) -> int:
    """size given: min(n, size). Else min(n, max(SAMPLE_FLOOR, min(SAMPLE_DEFAULT_CAP, ceil(SAMPLE_SQRT_FACTOR * sqrt(n)))))."""

def stratified_sample(ids_by_stratum: Mapping[str, Sequence[str]], k: int, *, seed: int) -> list[str]:
    """Allocation: if k >= number of non-empty strata, every stratum gets 1, the rest
    proportional to size by largest remainder (ties broken by stratum key). If k < strata,
    the k largest strata (ties by key) get 1 each. Within a stratum: sort ids, then
    random.Random(f'{seed}:{stratum}').sample. Returns the ids sorted."""
```

Sizes with the defaults: n=5 → 5, 10 → 8, 25 → 10, 100 → 20, 400 → 40, 1000 → 64,
50000 → 64. Items with no `cluster_subid` form the stratum `""`. `stratify_by: auto`
stratifies by `cluster_subid` when the resolved scope has more than one distinct sub-id.
The run stores the seed and the strata, so a run is reproducible: the same scope members,
seed and size give the same ids.

Resolution (in `src/services/curation/assist/scope.py`) reuses
`item_selection.resolve_ids`/`selection_query` (`item_selection.py:112-159`) for the
scope, with `include_test=False`, `include_excluded=False`. It fetches `cluster_subid` only
when stratifying. `MAX_SELECTION_SCAN` (500k) bounds the scope. Mode `all` over a scope
larger than `MAX_ASSIST_ITEMS` → 422 `assist_too_large` (the message points to the
auto-label job). Holdout and excluded items are never sent to the VLM. They are counted in
`skipped`.

### 3.5 Prompt pack: attributes and the two assist calls

`PromptPack` (`vlm_prompts.py:45-109`), `PromptPackBody` (`_prompt_pack_models.py:17-62`):

- `attributes: list[AttributeSpec]`, ≤16. `AttributeSpec`: `key` (`^[a-z][a-z0-9_]{0,31}$`,
  unique), `description` (≤300), `values: list[str]` (optional closed vocabulary, ≤50
  values, each ≤48 chars, lowercase normalised), `applies_to: "item" | "box"`.
- `assist_system`, `assist_user_template`: classify + attributes for items. The template
  placeholders are `{class_list}`, `{attribute_block}` and `{n_images}`. The server renders
  `{attribute_block}` from `attributes` (key, description, allowed values), so the pack
  author does not hand-format it.
- `attributes_system`, `attributes_user_template`: attributes only (items without classify, and box sub-crops).
- `REPLY_KEY_CONTRACT` (`:486-531`) gains `assist`: required `img`, `class`,
  `confidence`; optional `proposed_class`, `attributes`. It also gains `attributes`:
  required `img`, `attributes`. The `attributes` reply is an object
  `{key: {"value": str, "confidence": "high"|"medium"|"low"}}`.
- Parsing (new `src/services/labeling/vlm_reply_attributes.py`): it drops keys not declared
  in the pack. It normalises values (strip, lowercase, ≤48 chars). A value outside a closed
  `values` list is dropped and counted in `invalid_values`. A missing confidence becomes
  `low`. The reply never creates a class: `proposed_class` goes through the existing
  sanitiser and denylist.
- Validation (`pack_validation.py`): the placeholders exist, the keys are unique, the
  limits hold, and `assist_*` is required when `attributes` has an `item` entry.
- `prompt_packs.py:113` call→writer map: `assist`, `attributes` → `['assist_run']`.
  `/prompt_packs/test` `call` enum (`_config_test_models.py:39-54`) adds `assist` and
  `attributes`, so a user previews attributes on ≤64 crops before a run.
- The shipped `GENERIC_ITEM_PACK` gets default `assist_*`/`attributes_*` prose and
  `attributes: []`. The car example pack in `examples/` declares `make`, `model` and
  `body_style` (item) and `rim_material` (box, for the wheel region).
- **make/model fold (open decision D1, recommended):** delete the hard-coded `make`/`model`
  optional keys (`vlm_prompts.py:490`), their parse (`vlm_labeler_classify.py:236`,
  `vlm_reply_parse.py:142`), the writes (`vlm_stage.py:159-165`, `scripts/curation/worker/verify.py:197-200`)
  and the mapping fields `vlm_item_make`/`vlm_item_model` (`bodies_core.py:285-286`).
  The combined call's reply then carries `attributes`, and the worker writes them into the
  same `assist` object. They are not on the wire today, so no client loses anything.

### 3.6 Storage

Items mapping (`src/clients/curation_opensearch/bodies_core.py`, plus an additive
`ensure_fields.py` migration for existing indexes):

```text
assist: object
  run_id            keyword
  run_created_at    date
  vlm_endpoint      keyword   vlm_model keyword   prompt_pack keyword   prompt_pack_revision integer
  class_name        keyword   (the VLM's answer, also on locked items)
  class_confidence  keyword
  attributes: nested { key keyword, value keyword, confidence keyword }
  boxes: nested { box_id keyword, attributes: nested { key, value, confidence } }
region_run_id      keyword
region_run_verify  keyword   ("vlm" | "none"; removed by the worker when it writes the item)
```

- One `assist` slot per item; the newest run wins. The write is an OCC merge
  (`src/clients/occ.py` helpers) that skips when the stored `assist.run_created_at` is newer
  than the run's. A re-run after a re-refine therefore replaces only older suggestions and
  never touches class or box fields of locked items.
- Box attributes live under `assist.boxes[]` keyed by `box_id`, so the fixed `RegionBox`
  keys (`region_boxes.py:49-72`) are unchanged and `test_no_legacy_region_scalars.py`
  stays green. A box id that no longer exists on the item is dropped on read by the
  serializer.
- Run records: a `FileJob` (`src/services/curation/file_job.py:41`) per run under
  `<project_state_dir>/assist_runs/<run_id>/`, with `state.json`, `request.json` and
  `members.json` (the resolved scope ids, ≤500k; for `kind: crops`/`boxes` the ids
  themselves). The run executes as an in-API asyncio task like
  `reprocess_job.run_job` (`reprocess_job.py:197`), with a heartbeat ticker
  (`file_job.py:186`), and `reconcile_orphaned_jobs` (`reprocess_job.py:92`) marks it
  `lost` after an API restart. One active run per project → 409 `assist_busy`. It is
  independent of the auto-label lock; both writers go through OCC.
- Wire: `wire.py` maps `assist` to an `assist` object on the item wire (same names as
  above, `boxes` filtered to existing box ids). `make contracts` regenerates
  `contracts/json/item_wire.json`, `contracts/ts/itemWire.ts` and `contracts/openapi/curation.json`.

### 3.7 Cluster proposal

Computed by `src/services/curation/assist/proposal.py` from the sampled answers, never
from unsampled items:

- Votes per answered `class_name` (an empty answer is not a vote; `proposed_class` values
  are counted separately in `new_class_proposals`).
- `agreement = top_votes / answered`. `passes_gate = answered >= 3 and agreement >= min_agreement`.
- `class_name` is the top class when the gate passes, else `null` with the distribution
  still returned. `in_registry` is checked by name.
- `attribute_summary`: per key, value counts over the answers that gave the key, top 10.

The proposal is stored in the run result. It is recomputed for the response only for
`in_registry` (the registry can change). The proposal appears on the cluster page from
`GET {P}/assist/runs?cluster_id=&cluster_subid=`, with `scope_now.changed` deciding whether
the UI marks it stale. `apply` is the only path that turns it into labels.

How this covers "propose the label as the user refines the cluster": a refine rewrites
sub-ids, and the membership digest of the old run no longer matches. `scope_now` reports
the overlap and the UI shows "proposal from before the last refine (62 % overlap), re-run".
A new run on the new sub-cluster gets its own proposal. Items the user already labelled
are validated, so the new run sends them for votes only when the user selects them
explicitly. A locked item's answer still counts as a vote, which makes it a useful check
against the human label. Its class is never written.

### 3.8 Region worker change (segment action and project switch)

- `scripts/curation/worker/cascade.py:44-88` `_build_pending_query`: add `region_run_verify`
  to the fetched `_source`.
- `scripts/curation/worker/stage_sam.py:202-209` `_queue_for_verification` and the path-1
  route in `stage_a.py:128-167`: `effective_verify = item.region_run_verify or policy.region_verify`.
  `none` → `accept_without_vlm`. `vlm` without `rt.vlm_available` → write the item as
  `detection_failed` with reason `vlm_unavailable` (a new `region_rejection.py` reason) and
  never auto-accept. That is fail closed. Today the item would be accepted silently.
- The writer removes `region_run_verify` and keeps `region_run_id` (provenance; the assist
  run counts pending items by it).
- The skip-verify shortcut (`stage_sam.py:221-262`) still applies only when
  `effective_verify == "vlm"`.

### 3.9 Cropwright (renders backend data and decisions)

Wire types: add `frontend/src/lib/types_assist.ts` (hand-written to match
`contracts/openapi/curation.json`, as `types_vlm.ts` does) and a contract test
`frontend/src/lib/contract/assistContract.test.ts`. Extend `types.ts` and `api_cropMap.ts`
with the item `assist` object.

| File | Change |
|---|---|
| `frontend/src/lib/api_assist.ts` (new) | `createAssistRun`, `getAssistRun`, `listAssistRuns`, `cancelAssistRun`, `applyAssistRun`. |
| `frontend/src/lib/assistRun/assistRunController.svelte.ts` (new) | Dry run → confirm → poll (reuse the polling pattern of `pollAutoLabelJob`, `api_autoLabel.ts:87`). It holds no sampling or gating logic: it shows `resolved` and `result`. |
| `frontend/src/lib/components/assistRun/AssistRunDialog.svelte` (new) | Target summary (from the dry run), sample mode (all / representative, optional size and seed, stratify), action checkboxes, segment verify choice, `VlmRunPicker` (`components/vlm/VlmRunPicker.svelte`) and pack picker, and the server's counts and refusal messages. Named `assistRun` to avoid collision with the existing `assistScope.svelte.ts` / `AssistScopeBar` (auto-label scope bar). |
| `.../assistRun/AssistProposalCard.svelte` (new) | Proposal, distribution, attribute summary, stale banner from `scope_now`, and "Apply {class} to N unlocked members" (dry run count first) → `applyAssistRun`. |
| `.../assistRun/AssistAttributeChips.svelte` (new) | Attribute chips with confidence on `CropCard.svelte` (near `:441-475`), `CropMetaPanel.svelte` (new rows near `:152-170`), `ReviewItemMeta.svelte`, and per box in `ReviewSlotPanel.svelte`. It reads `crop.assist`; locked items show "VLM said X" from `assist.class_name`. |
| `ClusterDetailToolbar.svelte:198-226` | "Run VLM" becomes "Assist…", which opens the dialog with target `cluster` (+ the current sub-cluster tab), or `crops` + `context` when crops are selected. |
| `clusterDetailActions.svelte.ts:130`, `api_autoLabel.ts:41` | Delete `runVlm`/`runVlmOnCluster` (PR 4 swaps the call; PR 8 removes the leftovers and their tests `vlmClusterLabelJob.test.ts`, `dashboardVlmClusterLabelJob.test.ts`). |
| `routes/p/[project]/dashboard/+page.svelte:131-160,285-330` | Delete the "Run VLM on cluster" modal (the cluster page owns it). |
| `clusterDetailController.svelte.ts:261-273` | The sub-cluster tab passes `cluster_subid` to `GET /crops` (PR 1) instead of filtering in memory. |
| `CropMetaPanel.svelte:210`, box editor `MultiBoxCanvas.svelte` / `SlotBboxEditor.svelte` | "Assist…" for one crop (target `crops`, one id) and one box (target `boxes`, actions `verify`/`attributes`). |
| `itemFilter/selectionActionController.svelte.ts:30,48` | New action id `assist` that opens the dialog with target `filter` (the current filter). |
| `ingest/RegionStagePanel.svelte` / settings | `region_verify` switch on the VLM policy form (`api_labelConfirmation.ts:19,25` already calls `/vlm/policy`). |

E2E stubbed tests in `frontend/e2e/stubbed/` (pattern: `test_ingest_policy.py` in #329,
`test_vlm_job_poll_by_id.py`); `npm run check`, `npm run build`; full-page screenshots at
desktop and 430 px of the cluster page with the dialog and the proposal card, actually viewed.

## 4. Test plan (tests first, watched failing)

Offline suite (`.venv/bin/python -m pytest tests/ -q --no-cov -m 'not live'`), fakes as in
`tests/curation/test_vlm_cluster_scope.py` (`QueryFakeOpenSearch`, `vlm_env` fixture, and
from #329 `tests/curation/vlm_opt_in.py` to activate a VLM). Register every new labeler
call site in `tests/fixtures/labeler_call_sites.txt`. Fixture images: only
`tests/fixtures/sample_image.jpg` or COCO val2017 (cars, the wheel example). Never a
`baseline-*` project.

| PR | Test file | Asserts (each written first and seen failing) |
|---|---|---|
| 0 | `tests/curation/test_reprocess_filter_cap_region_vlm.py` | A filter with `limit=2, sample=random, seed=1` over 10 items re-queues exactly the same 2 ids for `region`, and clears exactly those 2 for `vlm`; dry run counts match. |
| 1 | `tests/curation/test_item_filter_cluster_subid.py` | `ItemFilter(cluster_subid='47b')` makes a term clause; `GET /crops?cluster_subid=` filters server-side; `cluster_subid` without `cluster_id` is accepted (sub-ids embed the cluster). Frontend: `clusterDetailController` test asserts the request carries `cluster_subid`. |
| 2 | `tests/curation/test_assist_sampling.py` | Size table from 3.4; `size` override; hard max 256 (422 at schema level); every stratum gets ≥1 when k ≥ strata; largest-remainder sums to k; the same seed gives the same ids, a different seed gives different ids; the result is independent of input order. |
| 3 | `tests/curation/test_pack_attributes.py`, `tests/curation/test_vlm_reply_attributes.py` | Pack round-trip keeps `attributes`; validation rejects duplicate keys, bad key regex, >16 attributes, a missing `{attribute_block}`; parser drops undeclared keys, drops out-of-vocabulary values and counts them, defaults confidence. `/prompt_packs/test` with `call: assist` returns parsed attributes and writes nothing. If D1: `make`/`model` are gone from the contract and a car pack declaring them yields them as attributes. |
| 4 | `tests/curation/test_assist_run_scope.py` | Cluster target resolves members, excludes holdout and excluded, digest stable under reordering; sub-cluster target uses `cluster_subid`; representative sample stratified by sub-id; `kind: crops` + representative → 422; empty filter → 422; mode all over 5001 → 422 `assist_too_large`; project VLM off and no `vlm=` → 409 `vlm_not_configured`; second run → 409 `assist_busy`; dry run writes nothing. |
| 4 | `tests/curation/test_assist_run_classify.py` | Fake labeler: unlocked unvalidated items get a VLM class suggestion through the guard; a human-labelled item and a validated import keep their class and get `assist.class_name` only; holdout never sent; a concurrent human edit between read and write wins (OCC); `/vlm/label_cluster` returns 404 (route deleted). |
| 4 | `tests/curation/test_assist_wire.py` | `assist` appears on the item wire through `wire.py`; `make contracts-check` clean. |
| 5 | `tests/curation/test_assist_proposal.py` | Votes, agreement, gate (`answered < 3` never passes), new-class proposals by name, attribute summary; `scope_now` after a simulated re-refine (subids rewritten) reports `changed` with the overlap; `apply` refuses 409 `scope_changed`, applies with `allow_changed_scope`, skips locked/validated/holdout, resolves the class by name, 409 `class_not_in_registry`. |
| 5 | `tests/curation/test_assist_rerun_after_refine.py` | Run, human-label 3 members, re-refine, re-run: the 3 keep their class and `class_source`, their `assist` slot is updated, and an older run's write after a newer one is skipped (`run_created_at` guard). |
| 6 | `tests/curation/test_assist_box_verify_attributes.py` | `kind: boxes` verifies only the named box ids (others untouched), skips locked boxes, writes box attributes under `assist.boxes[]`, a deleted box id is dropped on read; `kind: boxes` + `classify` → 422; `/vlm/verify_regions` returns 404. |
| 7 | `tests/curation/test_region_run_verify.py` | Worker with a VLM available and `region_run_verify: none` takes `accept_without_vlm`; with `vlm` and no VLM → `detection_failed` `vlm_unavailable`; policy `region_verify: none` applies to items without a per-item value; skip-verify only under `vlm`; the run moves `waiting_region` → `completed` when the items stop pending. `segment` with no region profile / paused stage / no segmenter leg → 409 codes; `segment.verify: vlm` with another endpoint → 409 `verify_endpoint_mismatch`. |
| 8 | Vitest `frontend/src/lib/assistRun/*.test.ts`, `contract/assistContract.test.ts`; Playwright `frontend/e2e/stubbed/test_assist_run_dialog.py` | Dialog posts exactly the server shape, shows dry-run counts and refusal codes, proposal card stale banner, apply posts `class_name`. |
| 9 | `tests/curation/test_auto_label_vlm_attributes.py` | `auto_label/start?run_vlm=true&vlm_attributes=true` writes `assist` attributes through the shared core; without it, no `assist` writes. |
| live | `tests/live/test_live_assist.py` (harness `docker/test/compose.yml`, `-p` name, GPUs 0/2) | On a throwaway project with ~40 COCO val2017 car images: ingest, cluster, refine, assist run on one sub-cluster with a representative sample and `attributes` (pack declares `make`, `body_style`), proposal returned, apply, re-refine, re-run, a human label survives. Segment + verify run on 3 selected cars with the wheel profile. |

The cross-project leak sweep (`tests/curation/cross_project_leak/test_every_scoped_route_sweep.py`)
must cover the new routes: add bodies to `_route_bodies.py` and rows to `_route_tables.py`.

## 5. PR sequence (each sized for one Sonnet agent)

**Delivery override (owner, 2026-10-15): do not open ten PRs.** Group the steps below into the
fewest waves that keep dependencies in order (about three: backend core, worker and storage,
frontend and docs). Each wave is one agent on one shared branch, committed in order, with only
the targeted tests and hooks run per wave. Open a single PR when the last wave is done, run the
full offline suite and the stubbed e2e once on that branch, and merge once. The step list below
stays as the work breakdown and the order inside a wave.

Each PR: conventional commit title, `.venv/bin/pre-commit run --all-files`, the targeted
tests, `make contracts` when the wire changes (commit `contracts/` in the same commit),
`scripts/docs/check_docs_vs_code.py`, a `CHANGELOG.md` `[Unreleased]` entry (and
`frontend/CHANGELOG.md` when the frontend changes), and the docs listed. Merge commits,
never squash. Nothing is pushed without the owner's yes.

**PR 0 — `fix(reprocess): honour limit and sample for the region and vlm scopes`**
- Files: `src/services/curation/reprocess.py` (`_plan_region` `:156`, `_plan_vlm` `:177`,
  apply branch `:371-396`: when `targets.limit` is set, resolve the capped ids once with
  `resolve_ids` and use the id paths `apply_region_ids` / `apply_vlm`).
- Test: PR 0 row. Docs: `docs-site/docs/guides/reprocess.mdx` (one sentence). Changelog: Fixed.

**PR 1 — `feat(curation): filter items by sub-cluster`**
- Files: `src/services/curation/item_filter.py` (`cluster_subid: str | None`, term clause),
  `src/routers/curation/_item_filter_params.py` (query param), `contracts/`,
  `frontend/src/lib/clusters/clusterDetailController.svelte.ts:261-273` (server filter,
  remove in-memory filter), its API call in `api_clusters.ts`/crop list client.
- Docs: `docs/design/curation_api_contract.md` filter table; `docs-site/docs/guides/pipeline-modes.mdx`
  recipe step 4 (`/crops?cluster_id=10000&cluster_subid=10000b`).

**PR 2 — `feat(curation): size-scaled, stratified, seeded representative sampling`**
- Files: new `src/domain/assist_sampling.py`; `tests/curation/test_assist_sampling.py`.
  No route yet: the function is pure and reused by PR 4. (The unused-code ratchet may flag
  it until PR 4 lands. If it does, merge PR 2 into PR 4 rather than adding a baseline line.)

**PR 3 — `feat(prompt-packs): pack-declared attributes and the assist and attributes calls`**
- Files: `src/services/labeling/vlm_prompts.py` (fields, defaults, contract),
  `src/routers/curation/_prompt_pack_models.py`, `src/services/config_store/pack_validation.py`,
  new `src/services/labeling/vlm_reply_attributes.py`, labeler methods `assist_batch` /
  `attributes_batch` next to `label_or_propose_batch` (`vlm_labeler_classify.py`; split a
  new module if it would pass the file-size ratchet), `src/routers/curation/prompt_packs.py:113`,
  `_config_test_models.py` + `src/services/curation/pack_test_run.py` (new call kinds),
  `examples/` car pack, `contracts/`.
- If D1 = yes, PR 3b, `refactor(vlm): fold make and model into pack attributes`: delete the
  make/model sites listed in 3.5 and the `vlm_item_make`/`_model` mapping. The combined reply
  carries `attributes`, and `scripts/curation/worker/verify.py:197-200` writes them to
  `assist.attributes`. This depends on the `assist` mapping, so PR 3b lands after PR 4.
- Docs: `docs-site/docs/guides/prompt-packs.mdx` (attributes section, example),
  `docs/design/curation_api_contract.md`.

**PR 4 — `feat(curation): assist runs for crops, clusters, sub-clusters and filters (classify, attributes)`**
- New: `src/schemas/curation/assist.py`; package `src/services/curation/assist/`
  (`scope.py` resolve + digest + sample, `job.py` FileJob run + in-API task + busy lock,
  `classify.py` sharing the class-write path with `autolabel/vlm_stage.py` (extract the
  shared `_label_chunk`/`_write_updates` into `src/services/curation/vlm_class_batch.py` and
  make `vlm_stage.py` call it, so there is no duplicate), `attributes_write.py` OCC `assist`
  merge); router `src/routers/curation/assist.py` registered in
  `src/routers/curation/__init__.py`; mapping in `bodies_core.py` + `ensure_fields.py`;
  `wire.py`; `contracts/`.
- Delete the route `/vlm/label_cluster/{cluster_id}` (`pipeline_control.py:69-104`) and its tests
  (`tests/curation/test_vlm_cluster_scope.py` route parts; keep the `cluster_id` job
  selection tests). Swap `frontend/src/lib/api_autoLabel.ts:41` `runVlmOnCluster` to
  `createAssistRun` (target cluster, sample all, actions classify) so no caller breaks.
- Cross-project leak tables. Docs: `curation_api_contract.md` (new section "Assist runs"),
  `docs/CURATION.md`, `docs-site/docs/guides/vlm-selection.mdx`, `pipeline-modes.mdx`
  (the selective-VLM table: one cluster → `POST $API/assist/runs`), `docs/ARCHITECTURE.md`
  (job list).

**PR 5 — `feat(curation): cluster and sub-cluster label proposals from an assist run`**
- Files: `src/services/curation/assist/proposal.py`, `scope_now` in the GET, `GET {P}/assist/runs`
  listing, `POST {P}/assist/runs/{run_id}/apply` (calls the `crop_bulk.py` human relabel
  service), `contracts/`. Tests: PR 5 rows. Docs: `docs/CURATION.md` labeling section,
  `pipeline-modes.mdx` recipe and section (d) cross-reference to #286.

**PR 6 — `feat(curation): assist on single boxes; verify through assist runs`**
- Files: `src/services/curation/region_verify.py` (`verifiable_boxes(..., box_ids=None)`),
  move the per-box verify loop from `vlm.py:292-404` to `src/services/curation/assist/verify.py`,
  delete `POST /vlm/verify_regions` and `VlmVerifyRegionsRequest`
  (`_vlm_route_models.py:15-18`), box attributes. Update `tests/live/test_live_vlm.py`,
  `tests/curation/test_vlm_verify_regions_*.py`, `test_vlm_prompt_pack_stamp_write_sites.py`,
  `pack_test_run.py`/`prompt_packs.py` references. Docs: `docs-site/docs/guides/multi-box-regions.mdx`.

**PR 7 — `feat(regions): selectable segmenter verification per run and per project`**
- Files: `src/services/curation/vlm_policy.py` (`region_verify`), router + contract;
  `scripts/curation/worker/cascade.py`, `stage_a.py`, `stage_sam.py`, writer; new rejection
  reason in `src/config/region_rejection.py`; `region_requeue.py` stamp; assist `segment`
  action and `waiting_region` (`src/services/curation/assist/segment.py`), 409 refusals
  (region profile, segmenter leg via `runtime.py:182-193` rules, `region_stage_control`
  paused, worker heartbeat stale via `worker_liveness`).
- Docs: `docs-site/docs/guides/region-profiles.mdx`, `wheel-example.mdx` (run SAM + VLM on
  three selected cars), `vlm-selection.mdx` (policy field).

**PR 8 — `feat(frontend): assist dialog, proposal card and attribute chips`**
- Files: the table in 3.9. Delete the dashboard modal, `runVlm`, `runVlmOnCluster` and
  their tests. Screenshots for docs (`docs-site` `Screenshot` names `cropwright/assist-dialog-1600.png`,
  `cropwright/assist-proposal-1600.png`). Docs: `docs-site/docs/cropwright/user-guide/` (new page
  `assist.md`), `frontend/CHANGELOG.md`.

**PR 9 — `feat(autolabel): pack attributes in whole-project VLM runs`**
- Files: `src/routers/curation/pipeline_start.py` (`vlm_attributes: bool = False`),
  `auto_label_run.py`, `autolabel/vlm_stage.py` (uses `assist_batch` + `attributes_write`
  when set), `scripts/curation/vlm_worker.py` stays classify-only. Docs: `pipeline-modes.mdx`
  mode (c), `vlm-selection.mdx`.

Dependencies: 0 and 1 are independent. 2 → 4. 3 → 4. 4 → 5, 6, 7, 3b, 9. 8 needs 4–7.

## 6. Risks and open decisions

Risks:

- **VLM cost on mode `all`.** `MAX_ASSIST_ITEMS = 5000` per run and the dry run's
  `estimated_vlm_calls` bound it. Assist runs ignore the project's daily budget, the same
  as `label_cluster` today (`selection.py:129-137`). This is documented.
- **In-API execution.** VLM calls are I/O bound, as `/vlm/label_batch` is today, but a
  restart loses a run, which ends `lost` and is re-run by the user. Concurrency per run
  uses the labeler's own batching (`max_images_per_call`) with a semaphore of 4.
- **Membership files.** `members.json` for a 500k-item filter is about 20 MB. The cap and
  the retention of 100 runs bound the disk use. Only cluster targets and contexts need the
  file for `scope_now`. For `kind: filter`, store only the digest and count.
- **Worker change (PR 7)** touches the region cascade. The per-item override must not change
  behaviour for items without it, which the test pins: the existing worker tests run unchanged.
- **Ratchets.** `vlm.py` (565 lines) and `vlm_stage.py` shrink. New modules stay small. Do
  not add baseline lines; split instead.

Owner decisions (D1 to D4 decided 2026-10-15: all four recommendations approved, as industry best practice):

- **D1. Fold `make`/`model` into pack attributes (delete the hard-coded keys and the
  `vlm_item_make`/`_model` fields)?** Recommend **yes**. They are vehicle-specific in a
  domain-agnostic product, invisible on the wire, and would duplicate the attribute path.
- **D2. Applying a cluster proposal writes `label_source: human` (validated, trainable), as
  `PUT /crops/batch_label` does today?** Recommend **yes**, behind the dry-run count and the
  `scope_changed` refusal. The user is the one confirming. The alternative is a new
  non-validated source that needs per-item review, which slows the "label quickly" goal.
- **D3. The `segment` action verifies only with the project's active VLM (another endpoint =
  a second `verify` run), rather than teaching the region worker per-item endpoints?**
  Recommend **yes** for this series. One run with `segment` + `verify` already gives a
  selectable endpoint.
- **D4. Sample defaults: floor 8, `ceil(2·√n)`, default cap 64, hard max 256, stratified
  by sub-cluster, `min_agreement` 0.6 with at least 3 answers?** Recommend **yes**. The
  numbers are constants in one module and per-request overridable within the hard max.

Out of scope: the iterative sample → agree → propagate → split loop (#286). This plan
supplies its parts: the stratified sample, the proposal with agreement, the
method-independent scope and the apply. #286 can then be a driver over assist runs.
Rework of clustering methods (owner, later) does not touch the assist layer: it reads only
`cluster_id`/`cluster_subid` fields and member ids.
