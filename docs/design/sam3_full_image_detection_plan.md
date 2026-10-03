# Full-image SAM 3 detection: design and implementation plan

Status: implemented for v0.4.0 (all eight waves; the implementation record,
including every deviation from this plan, is section 16, and the frontend
deltas as built are section 9). Issue: #30. Shares one gating layer with #46.
Related: #52 (default detector / store-all-detections / embedding-policy; its
plan is a sibling design doc under `docs/design/`, see section 12). The owner
decisions on the open questions (section 14) were: accept every
recommendation, and ship all eight waves in v0.4.0 (this supersedes the
release placement in section 15).

A fresh agent with no memory should be able to implement this from this file
alone. Paths are relative to the repo root. Line numbers are from main at
`837ca316` and will drift; re-find by symbol.

## 1. Goal

Let an operator define a project-level set of open-vocabulary targets (for
example "traffic cone", "wheel", "traffic light", "cup"). SAM 3 runs each
prompt on the WHOLE source image and every hit becomes a normal item (own
crop, embedding, cluster, label, export row), exactly as a primary-detector
item does. The prompt text is mapped to a class NAME. This discovers objects
the primary detector has no class for. Crop-scoped region SAM 3 (W8) is
unchanged and stays the quality refinement for small objects.

## 2. Current state (findings)

### 2.1 Segmenter service (`docker/segmenter/`)
- `docker/segmenter/main.py`: the segment endpoint (`SegmentRequest`, ~L94) takes
  `crop_jpeg_b64` (any JPEG, the field name says crop but it is just an
  image), `text_prompt` (required), `max_candidates` (default 4, cap 128 =
  `MAX_CANDIDATES_CAP` in `sam3_backend.py` ~L62), optional `min_score`,
  `return_masks`. The batch endpoint (`BatchSegmentRequest`, up to 64
  images, ONE shared prompt). Response candidates: `bbox_norm` in the
  SUBMITTED image frame, `score`, `mask_iou`, optional `mask_polygon`
  (largest external contour, <= 256 points, normalized, from W5).
  `GET /health` reports `instances`, `default_min_score` (0.5).
- `docker/segmenter/sam3_backend.py`: `ProcessorPool` (~L345) = N processors,
  one `asyncio.Lock` each, round-robin; N = in-flight forwards. `min_score`
  is applied per call and restored after. `Candidate` dataclass ~L66.
- Placement/concurrency (live stack, read-only check): one container on GPU 0
  (compose `device_ids` from the segmenter GPU setting), 2 instances, shared
  weights on, masks on, no `torch.compile`. GPU 0 shows ~43.8 of 49 GB used
  with the whole stack up, so adding instances is VRAM-limited; shared
  weights make each extra instance cheap compared with independent weights
  (see `docker/segmenter/README.md` settings table, ~L118-136).
- Latency: ~2.5-4 s per call measured on crops (issues #46/#30 text; a miss
  costs the same as a hit). Full-image input is larger, so assume the upper
  end until measured (section 10 measures it).

### 2.2 Worker client (`scripts/curation/worker/client.py`)
- `SegmenterUnavailable` (L63) with `SegmenterRequestFailed` (L68) and
  `SegmenterAllHostsDown` (L73). Contract: infrastructure failure is NEVER
  reported as "no candidate"; the caller leaves the work pending.
- `SegmenterClient.segment_multi(crop_jpeg)` (L372): sends
  `crop_jpeg_b64`, `text_prompt` (client-level `self.text_prompt`, one
  prompt per client), `max_candidates`; does NOT send `min_score` or
  `return_masks`, and DROPS `mask_polygon` (builds `RegionCandidate` with
  only bbox/score/source/`mask_iou`). Multi-host load balancing and health
  tracking (`_pick_healthy_url`, `_on_failure`, `_on_success`) already exist.
  `segment()` (L360) wraps it.
- Gap for this feature: per-call prompt, per-call `min_score`, per-call
  `return_masks`, and carrying `mask_polygon` through. The crop path must
  keep its current behavior.

### 2.3 Stage A secondary consumer (`scripts/curation/worker/runner.py`)
- `stage_a_sam_consumer` (L1143): pulls `_ItemTask` from `sam_q`, calls
  `rt.segmenter.segment_multi(t.crop_jpeg)` (L1192), on
  `SegmenterUnavailable` calls `leave_pending_segmenter_down` (L1157: item
  stays `pending_detection`, discard from `in_flight`, sleep 1 s), records
  `OP_STAGE_A_SEGMENTER_DURATION_SECONDS{outcome=hit|miss|error}`.
- `_select_candidates` (L~150-180): floor/NMS/cap via
  `select_region_candidates` (`src/services/detection/region_candidates.py`,
  greedy class-agnostic NMS, `min_score`, `iou`, `max_n`), then projects
  crop frame to source frame with `crop_norm_to_source_norm`. The full-image
  case needs NO projection (the frame is already the source frame).
- The crop path writes nested `RegionBox` entries on the parent item
  (`region_boxes`), NOT new items. Full-image output must not reuse that
  shape (issue #30 point 2).

### 2.4 Region profiles
- `src/config/detection_profile.py` `DetectionProfile` (L23): per-profile
  `segmenter_text_prompt`, `segmenter_name`/`segmenter_version` ('sam3'),
  `confidence_floor`, `region_nms_iou`, `max_regions_per_item`,
  `parent_classes` (L~137; empty = every item gets the region stage; this is
  the #46 cost problem), `class_ids`, `assigns_class`, `region_class_name`.
  One profile = one prompt. A prompt SET does not fit this dataclass.
- Config store: `src/services/config_store/index.py` L35
  `ConfigAxis = Literal['prompt_pack', 'detection_profile', 'vlm']`; records
  in `profiles.py`, `packs.py`; validation in `profile_validation.py`,
  `pack_validation.py`; activation in `activation_apply.py` /
  `activation_gate.py` / `activation_view.py`; per-project activation doc in
  `store.py`; clone in `src/services/projects/clone.py`,
  `src/services/curation/axis_copy.py`. Routes: `src/routers/curation/
  region_profiles.py`, `prompt_packs.py`, `settings.py` (activation).

### 2.5 Ingest and reprocess
- Ingest items come from `src/services/curation/ingest_detect.py`
  (`WholeImageDetector`, Triton end2end / raw detectors, bbox in pixel
  space -> `DetectedItem` in `item_doc.py` L54: `class_source`,
  `proposal_name`, `class_detector(_version)`, `label`, `extra_fields`).
  `ingest.py` orchestrates (`ingest_image` L329, `ingest_batch` L570).
- Reprocess: `src/services/curation/reprocess_models.py` L19
  `ReprocessScope = Literal['detect','region','vlm','embed']`;
  `ReprocessRequest` (targets exactly one of image_ids/crop_ids/filter,
  `dry_run` default true); implementation `reprocess.py`,
  `reprocess_images.py`; routes `src/routers/curation/reprocess.py`.
- Lock rule: `src/services/curation/reprocess_locks.py` (document form
  `class_locked`/`item_locked`, query form `class_locked_clause`; a test
  keeps both in agreement) over `src/clients/occ_locks.py`. Human and
  imported labels and boxes are never overwritten.
- Project isolation: index names resolve per bound project at call time
  (`src/routers/curation/_common.py` ~L74-81); every new read/write must go
  through that, never a module-level index name.

## 3. Target design

### 3.1 New concept: the prompt set (a config axis `open_vocab`)
A project-scoped, versioned document, same lifecycle as the other axes
(templates + user copies, validate, save with etag, revisions, activate,
rollback, clone with the project). Body:

```
name, display_name
targets: [ {
  prompt: str            # SAM 3 text prompt, e.g. "traffic cone"
  class_name: str        # registry class NAME this target creates/maps to
  min_score: float       # per-target confidence floor (default 0.5)
  min_area_frac: float   # drop boxes smaller than this fraction of the image
  max_area_frac: float   # drop near-whole-image boxes (default 0.9)
  max_instances: int     # cap per image per target (default 20)
  parent_classes: [str]  # optional Tier-1 rule; empty = whole-image only
  enabled: bool
  mask: bool             # also keep the polygon (default true)
} ]
image_max_side: int      # downscale before the call (default 1024)
dedup_iou: float         # vs existing boxes (default 0.5)
gating: { tier2_vlm_precheck: bool, tier3_hit_rate: {...} }   # section 5
```
Class identity is by NAME (the registry maps name to id; a missing name is
created through the existing class-registry growth path, never by id).
Validation rejects: duplicate `class_name`+`prompt`, empty prompt, a
`class_name` that collides with a primary-detector class unless
`merge_with_detector: true` is explicit, areas outside (0,1], more than a
configured number of enabled targets (cost guard, default 8).

Implementation choice: add the axis `open_vocab` next to the existing three
(do NOT overload `detection_profile`; it is one-prompt-per-profile and its
shape is already wide). Reuse `clone_shared.py`, the etag/revision helpers
and the activation gate; the gate's "impact" view reports how many images a
reprocess would touch.

### 3.2 Detector source and provenance
Each hit is written as a normal item doc through the same writer ingest
uses (`DetectedItem` -> `build_item_doc`), with:
- `detector = 'sam3'` (the profile-independent segmenter name constant),
  `detector_version`, `source_prompt` (the target prompt string),
  `open_vocab_set` + `open_vocab_revision` (provenance, enables "reprocess
  items not produced by this revision"), `bbox_norm` in the source frame,
  `score`, and `mask_polygon` (source-frame normalized) when `mask` is on.
- `class_name` = target `class_name`, `class_source` = a new unlabeled-
  proposal-style source ('open_vocab_proposal') so the lock rule treats it as
  machine output (replaceable) until a human or VLM confirms. Register the
  new source in `ingest_class_sources.py` so `unlabeled_proposal_class_
  sources()` includes it. Decision: the class is assigned by name at write
  time (the prompt IS the class intent); the "unclassified pending state" of
  the issue applies only to targets with an empty `class_name` (discovery
  mode: stored as unlabeled proposals with `proposal_name` = prompt, resolved
  later by VLM or human, same as generic proposers today).
- Item crop, crop embedding, clustering: identical to primary items; the
  embedding step honors the embedding policy from the sibling plan (detections
  are always stored; embedding is `all | selected | lazy`). Until that lands,
  default to `all` for these items.

### 3.3 Dedup against existing boxes
After per-target selection, drop a SAM 3 box when it overlaps an existing item
on the same image of the SAME class name with IoU >= `dedup_iou`, or of ANY
class when IoU >= 0.8 and the existing item is locked (human/imported):
- Same class name + high IoU: the existing detector item wins (keep its box);
  the SAM 3 hit is recorded only as a provenance chain entry on that item
  (`sam3:open_vocab_agree`) so agreement is visible but no duplicate exists.
- Different class name, IoU high, existing is machine-labeled: keep both
  (two interpretations are legitimate, e.g. "wheel" inside "car" is IoU low
  anyway); flag nothing.
- Locked existing item overlapping a candidate: candidate is skipped and
  counted (`skipped_locked`), never written, never edits the locked item.
- Within SAM 3 hits of one image: reuse `select_region_candidates` (NMS) for
  per-target, plus a cross-target class-agnostic NMS only for identical
  `class_name`.
IoU helper: `src/services/detection/geometry.py` `iou`; the pure selection
function is `region_candidates.select_region_candidates`. The dedup rule
lives in ONE new pure function (section 8, wave 2) shared by ingest-time and
reprocess-time.

### 3.4 Project isolation and idempotency
Every doc id for an open-vocab item is deterministic from
(image_id, class_name, quantized bbox) so re-running a pass is idempotent
(upsert, not duplicate). Reads/writes use the project-bound index resolver.
A reprocess of scope `open_vocab` removes only machine items whose
`open_vocab_set` matches and whose lock predicate is false, then rewrites.

## 4. Coordinate path
The full image is JPEG-encoded (optionally downscaled to `image_max_side` on
its long side) and sent unmodified in aspect. Returned `bbox_norm` and
`mask_polygon` are already source-normalized (downscale preserves
normalization), so there is no `crop_norm_to_source_norm`. Pixel bbox =
norm x original width/height. The crop path's call must stay byte-for-byte
unchanged.

## 5. Shared gating layer (one function, #46 + #30)

New module `src/services/detection/segmenter_gate.py` (pure where possible):

```
decide(target_or_profile, subject, state) -> GateDecision
GateDecision = run | skip(reason) | sample(prob)
```
`subject` is a crop (region stage) or an image (full-image pass). Tiers, in
order, each returning early:
1. Registry rules (free): for a crop, `parent_classes` match (case-insens.,
   same predicate as today at `DetectionProfile.parent_classes`); for a
   full-image target, `enabled`, and (if `parent_classes` set) that some item
   of those classes exists on the image. A target with empty `parent_classes`
   that is "not tied to any detector class" is routed to the full-image pass.
2. Optional VLM yes/no pre-check, only when a VLM endpoint is active:
   "Is there a <prompt> visible in this image?" on a downscaled image (or the
   crop). A "no" skips the SAM call and records `skip(vlm_no)`. A VLM error
   is NOT a "no": it falls through to `run` (fail toward running, but logged),
   mirroring the segmenter unavailable contract.
3. Learned hit-rate skip/sampling: per (project, target-or-class) rolling
   window of hit/miss counts (window N, miss threshold, sampling floor,
   default OFF or conservative). When the miss streak exceeds the threshold,
   sample at the floor rate rather than skipping outright, so recovery is
   possible. Counters live in the worker, persisted per project so restarts do
   not reset them.
Invariants: the lock rule (never skip-and-delete or override human/validated
state: the gate only decides whether to SPEND a SAM call, it never edits
labels); every skip is counted and reported (worker stats, review-queue
filter "skipped by gate", metric labels `tier`, `reason`); infrastructure
failure is never a skip. The crop stage (#46) and the full-image pass both
call `decide`; #46's part ships first in its own change, this plan consumes it.

## 6. Triggering
1. Batch: new reprocess scope `open_vocab` added to `ReprocessScope`
   (`POST /reprocess` with `scopes: ["open_vocab"]`, same targets/filter/
   dry_run contract; dry run returns counts: images selected, targets
   enabled, estimated SAM calls, estimated minutes from the section-7 model,
   `locked_skipped`). Also available via `POST /images/{id}/reprocess`.
   Default entry point for existing datasets.
2. Per-image test route for the UI: `POST /open_vocab/test` with an image id
   or uploaded image, a candidate target (not saved), returns
   boxes/scores/polygons and which tier decided, WITHOUT writing items
   (mirrors `POST /region_profiles/test`, `region_test_run.py`).
3. At ingest: opt-in per prompt set (`run_on_ingest: false` default). When on,
   the pass runs as a deferred worker stage after the item is written (NOT
   inline in the ingest request: 2.5-4 s per target would blow ingest
   latency). It uses the same pending/terminal pattern as the region stage
   (a new status field, `open_vocab_status`: pending / done / skipped_gate /
   failed_unavailable) so `SegmenterUnavailable` leaves it pending exactly
   as `leave_pending_segmenter_down` does.
All three call one service function `run_open_vocab(image, set, project)`.

## 7. Throughput and cost model
Cost per image = (enabled targets not skipped by the gate) x per-call
latency / effective parallelism. Parallelism = segmenter instance count (2 on
the live stack; each in-flight forward holds one instance lock).
- Per call ~2.5-4 s on crops (measured); assume 3 s as the planning number
  until section 10 measures full-image calls.
- Images/s ~= instances / (targets x per_call_s). Live stack: 2 / (1 x 3) =
  ~0.67 img/s for ONE target; 3 targets ~0.22 img/s (~800 images/hour). With
  8 instances (VRAM permitting with shared weights) ~4x.
- Levers, in order of payoff: (a) gate (Tier 2 VLM yes/no is far cheaper than
  a miss); (b) downscale to `image_max_side` (smaller input = faster, but the
  small-object quality tradeoff in #30 point 5; keep crop-region SAM 3 as
  the refinement); (c) early exit on empty: process targets in descending
  prior hit rate and stop an image when `stop_after_first_hit` is set (off by
  default, only valid for "does any exist" use); (d) batch endpoint: the
  shared-prompt batch endpoint batches many IMAGES for ONE target, so
  group work by target and send up to 64 images per call to amortize overhead
  (verify real gain in section 10; if per-image forward time dominates,
  batching only saves HTTP/decode); (e) reprocess ordering: group by target.
- Throttle: the pass must not starve the crop region stage; use a shared
  concurrency budget (a fraction of in-flight slots reserved for the crop
  stage) and surface queue depth.

## 8. Wave breakdown (small, independently shippable)

Each wave: red-first tests (watch them fail), a mutation check (break the
rule, see the test fail, restore), acceptance. All new code follows the
existing public-naming rules (section 11).

**Wave 1: client plumbing (no behavior change).**
Files: `scripts/curation/worker/client.py`, `src/services/detection/
cascade_detect.py` (`RegionCandidate` gains optional `mask_polygon`).
Add `segment_image(jpeg, prompt, *, min_score, max_candidates,
return_masks)` returning candidates with polygon; `segment_multi` becomes a
thin wrapper so the crop path is identical. Tests (red first):
`tests/curation/test_segmenter_client_full_image.py` - payload carries
prompt/min_score/return_masks; polygon is surfaced; unavailable raises
`SegmenterUnavailable` (not an empty list); crop path payload unchanged
(golden). Mutation: drop polygon, or swallow the failure as `[]`. Accept:
existing `tests/curation/test_segmenter_*` and `test_region_*` still pass.

**Wave 2: pure selection + dedup function.**
Files: new `src/services/detection/open_vocab_select.py`. Per-target
floor/min-max area/NMS/cap, then dedup vs existing boxes per section 3.3
(returns kept, dropped with reason). Tests: table-driven, includes locked
overlap skip, same-name agree, different-name keep both, determinism.
Mutation: flip IoU comparison, ignore lock. Accept: 100% branch coverage on
the module.

**Wave 3: prompt-set axis (config store).**
Files: `src/services/config_store/open_vocab.py` (record, etag, revisions),
`open_vocab_validation.py`, `index.py` (`ConfigAxis` adds `open_vocab`),
`store.py` (activation doc), `activation_*`, `clone.py`/`axis_copy.py`
(clone/copy with project), `project_usage.py`, router
`src/routers/curation/open_vocab.py` (list, get, create, save, delete, clone,
validate, revisions, activate, rollback, schema), wire models
`src/routers/curation/_open_vocab_models.py`, one shipped template under
`examples/open_vocab/` built from public classes only (traffic cone, cup,
traffic light). Regenerate and commit API contracts
(`scripts/codegen/export_api_contracts.py`, the `api-contracts-drift` hook)
and any TS contract. Tests: route + validation + etag conflict + project
isolation (set saved in project A is invisible in B) + clone carries it.
Mutation: remove project binding, skip validation. Accept: contracts drift
hook clean.

**Wave 4: full-image runner + item write.**
Files: new `src/services/curation/open_vocab_run.py` (`run_open_vocab`),
item writer reuse via `DetectedItem` + `build_item_doc`, `item_doc.py`
(new provenance fields), `ingest_class_sources.py` (new proposal source),
registry class-by-name ensure path. Idempotent ids, lock rule honored.
Tests: fake segmenter; hit -> item with `detector=sam3`, `source_prompt`,
source-frame bbox, polygon; rerun -> no duplicate; locked overlap -> no
write; unavailable -> nothing written and status stays pending. Mutation:
nondeterministic id, drop `source_prompt`. Accept: items appear in
queue/export with correct class name.

**Wave 5: triggers.**
Files: `reprocess_models.py` (`open_vocab` scope), `reprocess.py` /
`reprocess_images.py`, `src/routers/curation/reprocess.py`, new
`POST /open_vocab/test`, optional ingest hook + worker stage in
`scripts/curation/worker/` (status field via `RegionFields`-style config, so
the no-literal-field guard stays green). Tests: dry-run counts,
`locked_skipped`, scope validation, test route writes nothing. Mutation:
make the test route write. Accept: dry run estimate matches section 7 model.

**Wave 6: gate (shared with #46).**
Files: `src/services/detection/segmenter_gate.py`, hooks in
`runner.py` `stage_a_sam_consumer` (before L1192) and `run_open_vocab`.
Tier 1 first (this is also the #46 parent_classes work), then Tier 2, then
Tier 3 behind a default-off setting. Tests: each tier in isolation and
combined; VLM error does not skip; skip counted and visible; locked items
never touched. Mutation: treat VLM error as "no", drop the skip counter.

**Wave 7: frontend (Cropwright), section 9.** Separate repo/PR; backend
contract from wave 3/5 is the dependency.

**Wave 8: metrics + docs.** Prometheus: SAM calls by `scope`
(crop|image), `outcome`, `tier`; time spent on misses; docs page and
`docs/CURATION.md` section; example walkthrough.

Order: 1, 2 can ship any time (no behavior change). 3, 4, 5 are the
feature core. 6 is shared with #46 and may land earlier than 4.

## 9. Frontend deltas for Cropwright (numbered, as built)
All paths are project-relative (under `/curation/projects/{project}`). The
generated contracts (`contracts/openapi/curation.json`, `contracts/ts/*.ts`)
carry every shape below.
1. "Open-vocabulary targets" settings page over the `/open_vocab` routes:
   list (`GET /open_vocab`, `include_templates=true` for the shipped template),
   create, edit (`PUT` with `expected_revision`; 409 `revision_conflict`),
   clone (`POST /open_vocab/{name}/clone`, `from_project` optional), revisions,
   activate / deactivate / roll back (409 `active_conflict` carries `current`).
   Build the form from `GET /open_vocab/schema`: rows have `scope`
   (`set`, `target`, `gating`, `tier3_hit_rate`), `field`, `type`, `default`,
   `min`, `max`, `advanced`, `help`; no field list is hardcoded.
2. Target row editor: `prompt`, `class_name` (autocomplete from the class
   registry; empty means discovery mode, say so), `min_score`, `min_area_frac`,
   `max_area_frac`, `max_instances`, `parent_classes`, `mask`, `enabled`.
   Inline messages from `POST /open_vocab/validate` (`errors`, `warnings`,
   infos; codes `open_vocab_*`, `segmenter_prompt_*`, `parent_class_unknown`).
   Warn on `open_vocab_detector_class`, inform on `open_vocab_class_new`.
3. "Test on an image" panel: `POST /open_vocab/test` with `image_id` or
   `image_base64`, an unsaved `target` and optional `gating`. Draw
   `hits[].bbox_norm` and `hits[].mask_polygon` client-side; show
   `selected` / `drop_reason` per hit and `gate` (`run`, `tier`, `reason`).
   502 `segmenter_error` is an outage, not "nothing found".
4. Reprocess dialog: scope `open_vocab` with the dry-run `detail`
   (`enabled_targets`, `estimated_calls`, `segmenter_instances`,
   `segmenter_reachable`, `estimated_minutes`) and `locked_skipped`; a note
   that it uses the segmenter (slow). New filter fields `all_images` and
   `open_vocab_status` (image-level; not combined with item selectors or the
   `region` / `vlm` scopes). 422 `reprocess_targets_invalid` with "no
   open-vocabulary set is active" means activate one first. Over
   `OP_REPROCESS_SYNC_MAX` images the response carries a `job` (existing job UI).
5. Item provenance: item wire fields `source_prompt`, `open_vocab_set`,
   `open_vocab_revision`, `mask_polygon` (list responses send `mask_polygon`
   as null; `GET /crops/{id}` carries it), `class_detector` is `sam3`. New
   `GET /crops` filters `open_vocab_set` and `source_prompt`; the chip "from
   open-vocabulary pass" is `class_source=open_vocab_proposal`. The class-source
   catalog gains id `open_vocab_proposal` with role `open_vocab`.
6. "Skipped by gate" is an image state, not an item state: filter images with
   reprocess `open_vocab_status: ["skipped_gate"]` (and `["pending"]` for work
   ingest queued but did not finish). The "re-run these" action is a
   reprocess with that filter and scope `open_vocab`. Skip counts per tier are
   in the run's `detail` (`skipped_gate_tier<N>_<reason>`).
7. Hide the page with an explanatory empty state when no segmenter is
   configured (the activation 422 `segmenter_not_configured` / `segmenter_unreachable`
   and the validate warning say so; `GET /models/status` already reports it).
8. Visual check: desktop and narrow full-page screenshots, opened and
   inspected, per project convention.

## 10. Verification on the live stack (public COCO sets)
Use only the public COCO sample (already used for the live pass); the
running project is not modified except in a throwaway project created for
the test. Steps for the implementing agent:
1. Create a scratch project; ingest ~200 COCO images.
2. Save a set with targets "traffic light" and "cup" (class names equal);
   run `POST /open_vocab/test` on 5 images known to contain them; confirm
   boxes land on the objects (view the screenshot).
3. Dry-run then run `POST /reprocess` scope `open_vocab` over the 200
   images; record: wall-clock, images/s, SAM calls, hit rate, gate skips.
   Compare to the section-7 model (2 instances). Repeat with the instance
   count raised only if VRAM allows (GPU 0 currently ~44 of 49 GB; do not
   starve other services; stop and report instead).
4. Measure per-call latency on full image at 1024 and 1536 long side vs the
   2.5-4 s crop figure; test the batch endpoint with 8/32/64 images.
5. Precision check: sample 50 hits, count correct (human look at the grid);
   compare with COCO ground truth for the two categories to estimate recall
   vs the 80-class detector baseline.
6. Idempotency (re-run -> same count), lock rule (human-label one hit, rerun
   -> untouched), unavailable (stop segmenter briefly in the scratch setup
   only if allowed; otherwise unit-level only).
Acceptance: numbers recorded in the PR; no regressions in crop region stage
throughput (sample window rate before/after).

## 11. Config, contracts and leak-sweep registration
- Routes/wire models in section 8 wave 3; run the contract export and commit
  generated files so the drift hook passes.
- Naming guard (`scripts/codegen/check_naming_leaks.py`): the new code must
  not spell the scanned private/domain words; use the public example
  vocabulary only. If a legitimate test fixture must, add a reasoned entry to
  `scripts/codegen/naming_leak_allowlist.txt` (path + reason). Add the new
  files to any "ported paths" list in `check_no_literal_region_fields.py` if
  they touch region field names; new status field names come from the field
  config, not literals.
- Settings: any new tunable (concurrency share, gate defaults) must be read
  by code AND documented in the env template, or not added (docs checks reject
  advertised-but-unread settings). Prefer per-set body fields over new
  environment settings.
- File size limit hook (`check_file_size.py`): keep modules split by concern
  (select / run / gate / routes).

## 12. Relationship to the sibling plan (#52)
The sibling plan (store all detections, embedding policy `all|selected|lazy`,
full-vocabulary default detector) is tracked as GH #52 and a design doc in
this directory. This plan assumes: a hit is always stored as an item; whether
it is embedded follows the project embedding policy; a default full-vocab
detector reduces the cases where full-image SAM 3 is needed (targets already
in the detector vocabulary should use the detector, not SAM 3; the validator
warns when a target's class_name equals a detector class). If the sibling doc
lands first, reconcile section 3.2 and the dedup defaults with it.

## 13. Risks
- Cost: a pass is minutes per hundred images per target; mitigated by
  dry-run estimates, gate, per-set target cap, opt-in at ingest.
- Quality on small objects at full image resolution (known); crop-region SAM 3
  stays. Provide `image_max_side` and document the tradeoff.
- False positives become items and pollute clusters; mitigated by
  `source_prompt` provenance, a distinct proposal class source, confidence
  floors, and filter chips so a whole prompt's output can be bulk-removed
  (reprocess with the lock rule).
- VRAM: more instances compete with the rest of GPU 0.
- Duplicates with detector boxes: dedup rule (3.3), tested.
- Coordinate bugs from downscale: tests assert normalized invariance.
- Silent loss: gate skips and unavailable states are counted and visible.

## 14. Open owner questions (with recommendations)
1. Is a missing `class_name` (discovery mode, unlabeled proposals) in v1 or
   deferred? Recommend: v1 supports it only as stored proposals; labeling
   flows reuse the existing VLM/human path.
2. Default for `run_on_ingest`? Recommend off; batch reprocess is the entry.
3. Should a target whose name equals a detector class be an error or a
   warning? Recommend warning with the detector-first note (section 12).
4. Hard cap on enabled targets per set? Recommend 8 (configurable in the
   set), because cost is linear in targets.
5. Tier 3 default? Recommend off until measured; ship counters first.
6. New axis vs extending region profiles? Recommend new axis (section 3.1).

## 15. Release placement
Recommend AFTER v0.4.0. It adds a config axis, routes, a worker stage and a
frontend surface; none is trivial. Only wave 1 (client plumbing) and the Tier-1
`parent_classes` example-profile fix from #46 are small enough to consider for
v0.4.0, and #46's note already scopes v0.4.0 to the parent_classes fix only.
Target v0.5.0 with waves 1-2 anytime, 3-5 as the core, 6 shared with #46.

## 16. Implementation record (v0.4.0)

What was built, where it differs from the plan above, and what is left.

- **Wave 1.** `SegmenterClient.segment_image` (per-call prompt, `min_score`,
  `return_masks`, polygon) with `segment_multi` as the unchanged crop path;
  `RegionCandidate.mask_polygon`. The API process uses the plain
  `segment_image_http` (`src/services/detection/segmenter_http.py`, first
  configured host, no circuit breaker); both raise or map to
  `SegmenterCallError` / `SegmenterUnavailable`, and an outage is never "no hit".
- **Wave 2.** `src/services/detection/open_vocab_select.py`:
  `select_open_vocab_hits` (floor, area, NMS, cap, cross-target NMS for one class
  name, dedup against existing boxes). Locked overlap threshold 0.8, any class.
- **Wave 3.** Axis `open_vocab` (kind `open_vocab_set`, doc id prefix `ovset:`),
  decoder `src/services/detection/open_vocab_set.py`, validation
  `open_vocab_validation.py`, routes `open_vocab.py` / `_open_vocab_clone.py`,
  project clone axis `open_vocab` (`clone_open_vocab.py`; the stored-config copy
  is now `clone_stored.copy_stored_configs`, shared with `prompt_packs`). The
  activation tables are `activation_apply.AXIS_STORAGE`. Not on the settings
  defaults bridge (`PUT /settings`): sets have their own activate routes.
  `max_enabled_targets` is a set field (default 8) with a ceiling of 32. A target
  named like a detector class is a warning (`open_vocab_detector_class`).
  Class names are not forced to a slug by the validator.
- **Wave 4.** `open_vocab_run.run_open_vocab_image`. Items go through the same
  `index_items` writer as ingest and reprocess `detect` (the shared item-writer
  helper; a later embedding-policy change lands there once). `class_source` is
  `open_vocab_proposal` (registered in `unlabeled_proposal_class_sources()` and
  the class-source catalog); a named target creates its class by name
  (`ensure_class_by_name`, group `open_vocab`). Items are stamped
  `class_detector: sam3` (the wire's detector field) rather than a second
  `detector` key. Not built: recording an "agree" note on the surviving
  detector item (`sam3:open_vocab_agree`); agreement is counted as
  `dropped_agree_existing`.
- **Wave 5.** Scope `open_vocab` (dry-run estimate, locked count, outage trip
  after three images), image-level filter selectors, `POST /open_vocab/test`,
  and the ingest opt-in. The ingest-time pass runs as a background task of the
  API process (not a `scripts/curation/worker` stage: that process has no item
  writer or embedder), serialized per process, with a durable
  `open_vocab_status` on the image doc. A process restart leaves `pending`
  images for a reprocess with `open_vocab_status: ["pending"]`; there is no
  periodic sweeper.
- **Wave 6.** `src/services/detection/segmenter_gate.py`: `decide` with the
  three tiers, used by the full-image pass. The crop region stage already applies
  tier 1 (its `parent_classes` fetch and seed use the same predicate); wiring
  tiers 2 and 3 into `stage_a_sam_consumer` for crops is left to the #46 change
  that owns it (it needs a profile field for the opt-in). Defaults: tier 2 and
  tier 3 are both off.
- **Wave 7.** Section 9.
- **Wave 8.** Metrics `op_open_vocab_call_seconds`,
  `op_segmenter_gate_decisions_total`, `op_open_vocab_hits_dropped_total`,
  `op_open_vocab_items_written_total`; docs in the guide, `docs/CURATION.md`,
  README and the changelog.
- **Not measured.** Section 10 (live throughput, recall against COCO ground
  truth, batch-endpoint gain) needs the live stack; the dry-run estimate uses
  the 3 s per call planning figure and the segmenter's reported instance count.
