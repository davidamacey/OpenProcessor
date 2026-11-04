# v0.8.0 platform breadth: plan

| Field | Value |
|---|---|
| Status | plan, not started. Nothing in this file has been implemented or executed |
| Depends on | v0.7.0 tagged (B0-B5 and R-final of the 0.7.0 program), and the owner's go for each wave (section 3) |
| Release | one release, v0.8.0, after one acceptance run (R8, section 4). No intermediate tag |
| Sources | the 0.7.0 program set in `docs/design/program/` (00 index, 02b registry, 03 B2 overview, 04 B3, 05 B4, 09 GitHub actions, 10 standards and libraries, 12 prior plans); [`docs/VISION_AND_GOALS.md`](../VISION_AND_GOALS.md); [`docs/ARCHITECTURE.md`](../ARCHITECTURE.md); [`docs/CURATION.md`](../CURATION.md); [`SECURITY.md`](../../SECURITY.md); issues #8, #84, #348, #351, #365, #367, #368, #369, #370, #402, #435, #436 with their comments (read 2026-11-04); the code on `int/b1-correctness` at `5ca6c5b0` (read only, `rg`); the official docs cited per feature |
| Research items | #346, #347, #398, #400, #401, #406 and #416 are covered by the separate [clustering research plan](research_clustering_plan.md). This plan only consumes its result for #365 |

Read before you start: the program's `00_index.md` sections 4, 7 and 8.1 (waves, lanes, the test
and GPU budget). **The whole `docs/design/program/` folder is deleted after the v0.7.0 tag**
(program plan lifecycle). By the time this plan executes, read those rules from git history
(`git show v0.7.0:docs/design/program/00_index.md`) or from `docs/design/engineering_standards.md`,
where the lasting rules move. Section 3.1 restates the rules this plan relies on, so it does not
depend on the deleted files.

Paths: `<repo>` is the OpenProcessor checkout, `<repo>-wt-<name>` a sibling worktree. Line
numbers move; every site below is named by file and symbol, so re-find it with `rg`.

## 0. How to use this file

A fresh agent executes one wave (section 3) or one lane (one feature section) at a time. Each
feature section is self-contained: goal, user-visible outcome, design, library and doc, tasks with
red tests and done-when, size, risks, upgrade impact. Where this file and a source plan disagree,
this file wins for v0.8.0.

| Section | What it owns |
|---|---|
| 1 | scope, order, dependencies (between features and on v0.7.0) |
| 2.1-2.9 | one section per feature |
| 3 | waves, lanes, test and GPU budget |
| 4 | release acceptance additions (R8 rows) |
| 5 | GitHub actions: re-scopes, new issues, milestone moves |
| 6 | parked items with reason and revisit trigger |
| 7 | open questions for the owner |
| 8 | plan lifecycle and the keep-list |

## 1. Scope and order

### 1.1 Why breadth now and not earlier

Owner rule: no new features until the current ones and performance are done. v0.7.0 finishes
correctness, the jobs system, one Ultralytics version, the stateless API and speed. v0.8.0 then
widens the platform. Inside v0.8.0 the same rule applies in small: first finish the half-built
features (an import format, search follow-ups, the assist loop controls, training on more
families), then add access control, then the new surfaces (a second segmenter, video).

### 1.2 Feature list

| # | Feature | Issue | Size | Wave | Kind |
|---|---|---|---|---|---|
| F1 | COCO segmentation-only annotations as boxes | #370 (re-scoped) | S | W8.1 | finish |
| F2 | Search: one row per image (`group_by=image`); frame target decided | #436, #435 | S-M | W8.1 | finish |
| F3 | Assist loop: pause, event stream, token budgets | N16 (filed by the 0.7.0 program) | M | W8.1 (+ S0 in GPU window V1) | finish |
| F4 | Training on more model families | #368 remainder | M-L | W8.1 (CPU), V1 (GPU check) | finish |
| F5 | API authentication and authorization | #369 | XL | W8.2 | platform |
| F6 | Second segmenter backend behind `POST /segment` | #367 | L | W8.3 | breadth |
| F7 | Video files as an ingest source | N17 (new, split from #8) | L | W8.3 | breadth |
| F8 | Public wheel teacher-student example at scale | #84 | L (+ GPU) | W8.3 | proof |
| F9 | Embedding-model swap (register, build, evaluate, flip on the v0.7.0 registry) | #365 | M (was XL: the machinery ships in v0.7.0) | W8.4, **conditional** | breadth |
| F10 | S3-compatible storage backend for multi-host installs (D-46) | N21 (drafted in the 0.7.0 program's `00b_traceability.md` section 4.6) | L | W8.3 | platform |
| C | Carry-over from v0.7.0 (if still open at the tag) | N14 leftovers, N15 remainder, GPU-15 leftover | S each | W8.1 fill lanes | cleanup |

### 1.3 Order and the reasons for it

```
W8.0 kick-off (decisions, GH actions, spikes in GPU window V0)
  -> W8.1 finish: F1, F2, F3, F4-CPU, carry-over          (CPU sweep)
  -> W8.2 auth: F5 (four lanes)                            (CPU sweep + live harness run 1 + security review)
  -> W8.3 breadth: F6, F7, F8 (+ F4 GPU check, F3 S0)      (CPU sweep + GPU window V1)
  -> W8.4 F9 only if the research plan names a candidate   (CPU sweep + GPU window V-EM)
  -> W8R release prep + R8                                 (GPU window V2 + live harness run 2) -> tag v0.8.0 on the owner's word
```

- F1-F4 first: they finish features users already have. They are small and CPU-only.
- F5 before F6-F9: every route added after F5 is born with a declared permission, checked by the
  F5 ratchet. Adding the video upload route (F7) or migration routes (F9) before auth would mean
  classifying them twice.
- F6, F7 and F8 share one GPU window, so they sit in one wave.
- F9 is gated: building a migration with no model to migrate to is dead code. If the research plan
  has no candidate that passes its registered gain by the W8.3 sweep, F9 moves to parked
  (section 6) and v0.8.0 ships without it.

### 1.4 Dependencies on v0.7.0

| v0.7.0 piece | Used by | How |
|---|---|---|
| B1 job leases (OpenSearch lease docs, thread heartbeat, fencing) | F3, F7, F9, F5 bootstrap | one start per video, per migration, per loop pause; the first-admin create is create-only like a lease acquire |
| B1 class registry document with OCC, class by name | F1, F4, F7, F8 | imports, training, promote and the wheel example resolve classes by name through the registry |
| B1 upgrade-hook pattern (`registry_not_migrated`, `.migrated` rename, rollback step) | F5 basic-auth migration, F9 | same shape: dry run, migrate, rename, reverse step on rollback |
| B2 Celery 5.6 on Redis 8, beat, reaper, `queued` state | F3, F7, F8, F9 | new task families `op.jobs.video_extract`, `op.jobs.embedding_migrate`; a new `video` queue; F3 runs on the `assist_loop` consumer |
| B2 Redis (internal network, password, AOF) | F5 | sessions, login rate limits, token cache |
| B3 one Ultralytics version, fused `[batch,300,6]` contract | F4, F6, F8 | every trained family exports to the one contract; the YOLOE backend uses the same pin |
| B3 `src/domain/model_families.py` | F4 | training reads `trainable` and per-family settings from the registry module |
| B4 stateless API, `API_WORKERS` | F5 | no per-process session or permission state; the 1-vs-32 diff extends to auth |
| B4 Redis Streams events, `fastapi.sse` | F3, F7, F5 | loop and video progress events; SSE stays same-origin so the session cookie authenticates it |
| B5 GPU pipeline (decode once, resident crops) | F7 (later) | v0.8.0 video frames enter the normal ingest path as JPEG; feeding decoded frames straight into the resident pipeline is a follow-up, measured first (section 2.7, risks) |

### 1.5 Dependencies between features

| Feature | Needs | Reason |
|---|---|---|
| F5 | F2 merged (if both touch `search.py`) | avoid a conflict; otherwise none |
| F6 | F5 | new or changed routes get permissions; the `origin` filter rename is a wire change done once |
| F7 | F5 | the upload route is born with a permission |
| F8 | F4 not required; B3 required | the example trains YOLO26 only; it benefits from F6 (it can compare both segmenters) but does not wait for it |
| F9 | research plan result; F5 | a candidate encoder; admin-only migration routes |

## 2. Features

Common rules for every task (they are not repeated per task):

- T0 only in a lane: the task's red test seen failing on the lane base, plus a paired control that
  passes before and after; static checks on touched files. No full suite, no `--all-files`, no live
  or GPU run in a lane.
- A wire change runs `make contracts` and commits `contracts/` in the same commit. One serializer:
  storage-to-wire names go through `src/services/curation/wire.py`.
- A superseded route, field, env var or module is deleted in the same change. Retired env vars go
  into `src/config/retired_env.py`.
- Class identity by name at every boundary. Fail closed on an ambiguous or unbound state.
- No in-process or file lock for cross-process state; heavy work goes to a Celery job.
- New dependencies: a licence row in `ATTRIBUTION.md`, the Trivy licence scan, and an exact or
  floor pin as `docs/design/engineering_standards.md` (the library register) says.
- Docs updated per section 8.
- Public data only (COCO, Open Images, or data synthesized from them) in tests and docs.

---

### 2.1 F1: COCO segmentation-only annotations as boxes (#370)

**Goal.** A COCO annotation that carries `segmentation` but no usable `bbox` is imported as the
axis-aligned box of its shape, instead of being dropped.

**User-visible outcome.** Importing such a dataset gives boxes; the preview report shows an
info issue `coco_segmentation_to_box` with a count, like YOLO-seg's `yolo_polygon_to_box` today.

**Today.** `src/services/curation/dataset_import/coco.py::_annotation_box` reads only
`ann['bbox']`; a missing or empty `bbox` returns `None` and the image gets
`coco_bbox_out_of_image`. Crowd annotations (`iscrowd=1`) are skipped with `coco_crowd_skipped`.

**Design.**
- Use `bbox` when it is present and valid (no change).
- Else, polygon segmentation (a list of flat `[x1, y1, x2, y2, ...]` lists): the box is the min
  and max over all points of all parts. This is exact and is what the YOLO-seg import does. Share
  one pure helper in `src/domain/` with `yolo.py`'s polygon-to-box (no duplicate logic).
- Else, RLE segmentation (a dict with `counts` and `size`, compressed string or uncompressed
  list): `pycocotools.mask.toBbox`, after `mask.frPyObjects` for uncompressed RLE. Hand-decoding
  RLE is a library's job.
- `iscrowd=1` stays skipped (unchanged behaviour; COCO uses RLE for crowd regions).
- A segmentation that parses to an empty or out-of-image box keeps the existing
  `coco_bbox_out_of_image` code. A malformed one gets `coco_segmentation_invalid` (warning).
- Class by name: unchanged (`categories[].name`, never `category_id` against a registry id).

**Library and standard.**
- COCO data format, "Object Detection" annotation fields (`segmentation` polygon or RLE, `bbox`,
  `iscrowd`): https://cocodataset.org/#format-data
- pycocotools mask API (`frPyObjects`, `toBbox`), BSD-2-Clause:
  https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/mask.py
- pycocotools is in `requirements-test.txt` and the bake-off requirements, not in the API image.
  Add it to `requirements.txt` as a lazy import inside the RLE branch (open question Q-11).

**Tasks.**

| ID | Task | Files | Red test (fails today) | Control | Done when |
|---|---|---|---|---|---|
| IM-1 | shared polygon-to-box helper | new `src/domain/polygon_box.py`; `dataset_import/yolo.py` uses it | unit test: the helper on a multi-part polygon gives the min/max box | YOLO-seg import tests unchanged | `yolo.py` has no local polygon min/max |
| IM-2 | COCO polygon fallback | `dataset_import/coco.py::_annotation_box`; `dataset_import/issues.py` (new codes) | fixture: COCO JSON with one polygon-only annotation on a COCO val2017 image: today 0 boxes and `coco_bbox_out_of_image` | an annotation with `bbox` still uses `bbox` | 1 box, issue `coco_segmentation_to_box` |
| IM-3 | COCO RLE fallback | same; `requirements.txt` (`pycocotools`), `ATTRIBUTION.md` | fixture with a compressed and an uncompressed RLE: today dropped | `iscrowd=1` still skipped | boxes equal `toBbox` |
| IM-4 | wire and docs | `src/routers/curation/_dataset_issue_wire.py` (if the issue codes are a wire Literal), `contracts/`, `docs-site/docs/guides/dataset-import.mdx` | contracts-check fails until regenerated | | the guide lists both codes |

Fixtures: build the JSON in the test from a public COCO val2017 image id and synthetic shapes; no
annotation file is committed beyond a few lines.

**Size.** S. **Risks.** Low. The only behaviour change is boxes appearing where none did; a re-run
preview shows them before import. **Upgrade impact.** None (import-time only; past imports
unchanged). #370 closes on merge.

---

### 2.2 F2: search follow-ups (#436, #435)

**Goal.** A search page can show one row per source image (#436). #435 (`target=frame`) is decided,
not left open.

**Today.** `GET /curation/projects/{project}/search/text`, `GET .../search/similar` and
`POST .../search/image` return one card per item (`target=item|box`). Items carry `image_id`
(keyword). The project's images index already has a whole-frame `pe_embedding`
(`src/clients/curation_opensearch/bodies_core.py::_images_body`).

**Design (#436).**
- New query parameter `group_by` with values `none` (default) and `image` on the three routes.
- `group_by=image` adds an OpenSearch `collapse` on `image_id` to the existing kNN query and
  over-fetches (`k = page_size * GROUP_OVERFETCH`, a constant, start at 5). Each row is the best
  item (or, with `target=box`, the best item and its `matched_box_id`) of one image. No new
  response shape: the existing page model plus one optional field `image_hit_count` (items of that
  image in the candidate set), filled from collapse `inner_hits` only if the spike shows its cost
  is small; otherwise omitted.
- Paging with collapse uses `search_after` on the sort field, as the collapse docs require; the
  routes already page by cursor, so the lane checks the cursor field matches.
- Cropwright: a "one per image" toggle in the search box; the selection and bulk actions keep
  working on the returned items.

**Decision (#435), recommendation: close.** Whole-image search exists in the core API
(`/search/image`), curation search is item-centric, and #436 gives the per-image view. A frame
target would need its own response shape and its own filter set (the issue's own words). If the
owner wants it (Q-6), it is a separate route over the images index's `pe_embedding`, not a third
`target` value, and it is sized then.

**Library and standard.**
- OpenSearch collapse (read 2026-11-04): the field must be keyword or numeric, have doc values and
  hold one value per document; collapse cannot be combined with scroll or rescore; `total` counts
  hits before collapse; paging uses `search_after`:
  https://docs.opensearch.org/latest/search-plugins/searching-data/collapse-search/
- k-NN with filters (the existing filtered ANN query): https://docs.opensearch.org/latest/vector-search/filter-search-knn/
- The docs do not say whether collapse composes with a `knn` query with `rescore` on the Lucene
  engine at our version (OpenSearch 3.6.0 in `docker-compose.yml`). **Spike SP-1** (section 3.2)
  checks it before the lane starts.

**Tasks.**

| ID | Task | Files | Red test | Control | Done when |
|---|---|---|---|---|---|
| SR-0 | spike SP-1: collapse over the filtered kNN on OpenSearch 3.6.0 (CPU, OpenSearch container only, 10 min) | scratch script, results in the PR | n/a | n/a | it works with our query body, or the design changes to "over-fetch and group in the service" |
| SR-1 | `group_by` parameter and query | `src/routers/curation/search.py`, `search_image.py`, `src/services/curation/` search body builder (find with `rg -n "_knn_page" src`), `src/schemas/curation/` | fake OpenSearch records the body: no `collapse` today; three items in one image give three rows | `group_by=none` body unchanged byte for byte | one row per image; contracts regenerated |
| SR-2 | Cropwright toggle | `frontend/src/lib/searchBox.svelte.ts`, the review search component, `frontend/src/lib/api*.ts` | component test: toggle sends `group_by=image` | default request unchanged | browser check at 1440 and 430 px with screenshots viewed |
| SR-3 | close #435 per Q-6 | GitHub only | | | comment C-435 posted |

**Size.** S-M. **Risks.** Collapse and kNN `k` interact: too small an over-fetch returns fewer
than a page of images. The constant is tested on a fake with many items per image; the live check
is in R8 (R8-20). **Upgrade impact.** None (no mapping change; `image_id` is already a keyword).

---

### 2.3 F3: assist loop pause, events and token budgets (N16)

**Re-scoped by the 0.7.0 research integration (2026-11-04, pending its owner question RD-1).** The
0.7.0 rebuild now ships the loop's own record in PostgreSQL, pause and resume that never spend
retries, time slices, budgets with token reservations from the endpoint's `usage`, the event types
and the Cropwright pause and budget controls (B2 lanes 4D and 4E; the research plan
`docs/design/research_auto_label_assist_plan.md` section 5). What F3 keeps: the review queue and
metrics routes and the Cropwright review grid (`feat/loop-review`), the daily cap read from
`vlm_usage_hourly` and the optional price per million tokens (RD-74), and the `sweep` strategy
(RD-73). The text below is the earlier design and is replaced by that plan's section 7.3 at W8.0.

**Goal.** The assist loop gets a pause control, a live event log and token budgets for remote VLM
endpoints. (Scope as filed in N16 by the 0.7.0 program.)

**User-visible outcome.** In Cropwright a running loop has Pause and Resume buttons and a live
event list; a run with a token budget stops cleanly with `budget_exhausted` and shows the tokens
used.

**Today.** `LoopStore` has a `pause.flag` and the state has `pause_reason`, but no route serves
them; the VLM client does not read the response's `usage` object. After B2 the loop runs on the
`assist_loop` Celery consumer (`-c 1`); after B4 events go through Redis Streams.

**Design.**
- Routes: `POST /curation/projects/{project}/assist_loop/{run_id}/pause` and `.../resume`
  (confirm the exact path prefix against the existing assist-loop routes in
  `src/routers/curation/assist_loop.py`; reuse its run id parameter). Pause writes the flag through
  the job record; the consumer checks it between batches, so a pause takes effect at the next batch
  boundary, never mid-write. Resume re-enqueues on the same queue. Both are idempotent.
- Events: the worker publishes `assist_loop.batch_done`, `.paused`, `.resumed`,
  `.budget_exhausted`, `.finished` to the project stream (B4 `op:events:<project>`). Cropwright
  subscribes through the existing SSE route.
- Budgets: the run spec gains `token_budget` (optional, integer, total prompt + completion tokens).
  The VLM client reads `usage.prompt_tokens` and `usage.completion_tokens` from each
  OpenAI-compatible response and adds them to the run state with an OCC write. A response without
  `usage` counts as unknown: with a budget set, the run stops with `budget_unmeasurable` (fail
  closed) rather than running unbounded.
- **No default budget value** ships until S0 measures the label rate (owner rule: measure first).
  S0 runs in GPU window V1 on public COCO crops only; a default, if the owner wants one, is a
  one-line config commit after V1.

**Library and standard.**
- OpenAI chat completion object, `usage` fields: https://platform.openai.com/docs/api-reference/chat/object
- vLLM's OpenAI-compatible server returns `usage`: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
- Redis Streams and `fastapi.sse` as in B4 (library register).
- S0 method: `docs/design/auto_label_loop_plan.md` section 13 (stage S0), restricted to public
  COCO crops.

**Tasks.**

| ID | Task | Files | Red test | Control | Done when |
|---|---|---|---|---|---|
| AL-1 | pause and resume routes | `src/routers/curation/assist_loop.py`, `src/services/curation/assist_loop/` store and driver, `contracts/` | route test: `POST .../pause` is 404 today | start, watch, cancel, rollback unchanged | the consumer stops at the next batch boundary (fake clock) |
| AL-2 | events | assist-loop driver; the B4 event publisher | fake stream records no `assist_loop.*` event today | other event kinds unchanged | each state change publishes once |
| AL-3 | token usage and budget | `src/services/labeling/vlm_client.py`, the run spec model, run state | a fake VLM returning `usage`: tokens not recorded today; a budget of 100 with 3 x 50 tokens stops after the second call | a run without a budget is unchanged | `budget_exhausted` and `budget_unmeasurable` reasons on the wire |
| AL-4 | Cropwright controls and event log | `frontend/src/lib/api_assistLoop.ts`, the assist-loop page, `types_assistLoop.ts` | component test: no pause button today | | browser check 1440/430 px |
| AL-5 | S0 measurement | `scripts/bench/vlm_rate.py` (per the source plan) | n/a | n/a | rows in `docs/research/vlm_serving_sizing.md` and `docs/PERFORMANCE.md`, GPU window V1 |

**Size.** M. **Risks.** A pause that lands mid-batch must not leave a half-applied propagate; the
batch boundary rule and the existing rollback cover it (test with a fake that pauses between the
write of two items: the second item is never written). **Upgrade impact.** Additive wire fields;
old runs have no budget.

---

### 2.4 F4: training on more model families (#368 remainder)

**Goal.** Training, promote and serving work for every family the registry marks trainable, not
only YOLO26. The B3 slice already gave one registry module (`src/domain/model_families.py`) used
by the probe, the upload validator, the exporter and the trainer's export.

**User-visible outcome.** The train page offers a family picker (YOLO26 default; YOLO11 and YOLOv8
at least); a trained YOLO11 model promotes and serves through the same fused `[batch,300,6]`
contract.

**Today.** `src/services/training/job_models.py` has `model_family: Literal['yolo26']`;
`src/services/training/profiles.py` holds `PROFILES_YOLO26` and `RESERVED_OPTIMIZERS_YOLO26`;
`src/services/training/yolo_triton_config.py` is named and documented for YOLO26;
`docker/trainer/trainer_kwargs.py` builds `f'{family}{size}.pt'`, which is wrong for YOLOv5u
(`yolov5nu.pt`).

**Design.**
- Registry (`model_families.py`, stdlib only so the trainer can copy it): add `trainable: bool`,
  `weights_name(size) -> str`, `sizes`, and `train_defaults` (a small dict of the few arguments
  that differ per family; everything else stays the Ultralytics default). Trainable in v0.8.0:
  `yolo26`, `yolo11`, `yolov8` (Q-9 may add `yolo12`, `yolov5u`).
- Wire: `model_family` becomes a Literal generated from the registry's trainable names; a new
  `GET /curation/projects/{project}/train/families` lists them with sizes and defaults (or fold
  into the existing profiles route if one exists; check with `rg -n "profiles" src/routers/curation`).
- Profiles: the hyperparameter table becomes per family. YOLO26's recipe stays as is; other
  families start from Ultralytics defaults with the same augmentation presets.
- Promote and Triton config: one fused-contract writer for all families (B3 made the contract
  one); rename `Yolo26TritonConfig` and its module docstring to the family-neutral contract; the
  Triton model name prefix stays per project.
- Probe vs served head (U-R8): record the decision. Recommendation: the probe keeps its
  architecture list from the registry and states, in the registry, that YOLO26's probe uses the
  one-to-many head with NMS while serving uses the one-to-one head; no behaviour change.
- "Generic single-class exporter" (the second box of #368): `export_single_class.py` already
  exports any class subset with background frames and an integrity envelope, and says "nothing
  here is domain-specific". Recommendation: mark that box done (Q-10).
- Class by name: unchanged; the trainer's export snapshot of the registry is the only class source.

**Library and standard.**
- Ultralytics train mode and arguments: https://docs.ultralytics.com/modes/train/
- Export `nms` argument by head kind (B3, unchanged): https://docs.ultralytics.com/modes/export/
- Model pages for weight names: https://docs.ultralytics.com/models/yolo11/ ,
  https://docs.ultralytics.com/models/yolov8/ , https://docs.ultralytics.com/models/yolo26/

**Tasks.**

| ID | Task | Files | Red test | Control | Done when |
|---|---|---|---|---|---|
| FT-1 | registry training fields | `src/domain/model_families.py`, `tests/domain/test_model_families.py` | `weights_name('n')` for `yolov5u` is `yolov5nu.pt` (absent today) | B3 fields unchanged; the isolated-interpreter import test passes | one definition of trainable families |
| FT-2 | spec and preflight accept trainable families | `src/services/training/job_models.py`, `preflight_checks.py`, `contracts/` | `TrainSpec(model_family='yolo11')` is a 422 today | `yolo26` default unchanged | a non-trainable family is refused by name |
| FT-3 | trainer reads the registry | `docker/trainer/trainer_kwargs.py`, `trainer_fit.py` | recording fake: `yolov5u` asks for `yolov5un.pt` today | YOLO26 kwargs unchanged | weights and kwargs come from the registry |
| FT-4 | per-family profiles | `src/services/training/profiles.py` | `profiles_for('yolo11')` missing | YOLO26 profile values unchanged | |
| FT-5 | family-neutral Triton config writer | `src/services/training/yolo_triton_config.py`, `promote_run.py` | a YOLO11 promote fixture writes a config named for YOLO26 constants (rename test) | the YOLO26 config is byte-identical | no YOLO26-only name left in the writer |
| FT-6 | Cropwright family picker | train page, `frontend/src/lib/types_train.ts`, `api` client | component test | default submission unchanged | browser check |
| FT-7 | GPU check (window V1) | `scripts/release/` or `tests/live/` smoke | n/a | n/a | per trainable family: 3 epochs on Ultralytics `coco8` (public), export, promote, `/detect` serves `[300,6]`, class names equal the registry by name |

**Size.** M-L. **Risks.** Family defaults drift between Ultralytics releases (the B3 pin and freeze
rule cover it). YOLOv8 heads need `nms=True` export (registry `export_kwargs` already decides by
head). **Upgrade impact.** None for existing runs; promoted YOLO26 models are unchanged.

---

### 2.5 F5: API authentication and authorization (#369)

**Goal.** The API and Cropwright know who is calling and what they may do. This replaces the
single shared basic-auth login in front of Cropwright (#351, shipped in 0.7.0) with users, roles
and tokens that the API itself enforces, for every route, across all API workers.

**User-visible outcome.**
- A LAN install asks for a login. Each person has an account with a role. Cropwright shows a login
  page, the user's name, and (for admins) a user and token page.
- Scripts and the CLI use an API token from a file.
- A loopback-only install can keep running without login, as today.
- Behind an organisation's authenticating proxy, the API takes the user from the proxy.
- Label operations and history entries record the actor (the comment on #369 from #404/#419).

#### 2.5.1 Threat model summary (neutral)

This summary names assets, actors and controls. Mechanism-level detail of any past finding stays
in the maintainers' private notes, as for every security item.

| Item | Summary |
|---|---|
| Assets | project images and labels; trained and promoted models; biometric face data (deployment-wide, see SECURITY.md); configuration (VLM endpoint registry, prompt packs, profiles); routes that read server-side paths or control services |
| Actors | a person on the LAN with no account; an authenticated user acting beyond their role; a web page in the same browser making cross-site requests; an internal worker container; an operator with shell access (trusted, out of scope) |
| Trust boundaries | browser to Cropwright's nginx; nginx to the API (Docker network); scripts and workers to the API; the API to OpenSearch and Redis (internal network, unchanged) |
| Controls added | authentication on every route (default deny); roles per route; session cookies that scripts and other sites cannot read or send cross-site; CSRF protection for cookie requests; login attempt limits shared by all workers; revocable sessions and tokens; secrets only in files with mode 600; fail-closed startup checks |
| Not addressed (documented) | transport encryption (plain HTTP on a LAN exposes cookies and passwords: use a TLS-terminating proxy, as SECURITY.md already says); per-project access lists (Q-1); the other UIs' own logins beyond the gateway check (Grafana keeps its admin account) |

#### 2.5.2 Deployment modes

One setting, `OP_AUTH_MODE`, with three values. The installer chooses it; startup checks it.

| Mode | When | Who is the caller | Startup refuses when (fail closed) |
|---|---|---|---|
| `none` | API and Cropwright bound to loopback only (`--local-only`) | an implicit local admin; `actor` is `local` | any of the API or Cropwright binds is not loopback |
| `local` (installer default for a LAN bind) | homelab, small office | a user in the built-in user store | the user store has no admin (every route except `/live` and `/ready` answers 503 `auth_not_initialised`); Redis unreachable (authenticated routes answer 503, like job starts in PR-07) |
| `proxy` | behind an authenticating reverse proxy (for example oauth2-proxy in front of an OIDC provider) | the user named by the proxy's identity headers | `OP_AUTH_TRUSTED_PROXIES` empty; the API port published beyond the proxy network; a request with identity headers from a peer not in the trusted list is 401 |

OIDC inside the API is not in v0.8.0 (recommendation; Q-2): the proxy mode covers it with a
maintained proxy, and the API keeps a small surface.

#### 2.5.3 Design

**Identity store.** A deployment-wide OpenSearch index `{OP_PROJECT_INDEX_PREFIX}_auth` (not
project-owned; one more allowlisted shape in `src/services/projects/guard.py`, everything else
still refused). Documents:
- `user:<username>`: `username`, `display_name`, `role`, `password_hash`, `disabled`,
  `created_at`, `updated_at`; written with OCC (`if_seq_no`), created with `op_type=create`.
- `token:<sha256(token)>`: `token_id`, `owner`, `role` (at most the owner's), `name`,
  `expires_at`, `created_at`, `last_used_at` (updated at most once per hour per token).
- The first admin: created by a CLI command run in the `api` container, create-only, so two
  installers cannot both create it.

**Passwords.** `pwdlib` with Argon2id (`PasswordHash.recommended()`), as the FastAPI security
tutorial uses. Length 12 to 1024, no composition rules, a check against a short local list of
common passwords (NIST SP 800-63B guidance).

**Sessions (Cropwright).** An opaque random token (`secrets.token_urlsafe(32)`) in a cookie
`op_session`: `HttpOnly`, `SameSite=Strict`, `Path=/`, `Secure` when the request came over HTTPS
(read through the trusted proxy headers). The server side is a Redis key
`op:auth:session:<sha256(token)>` with the user name, role and timestamps; idle TTL 12 h (sliding),
absolute lifetime 7 days. A set `op:auth:user_sessions:<username>` lists a user's sessions so a role
change, disable or password change deletes them all at once. No JWT: an opaque token is revocable
at once on every worker, and nothing needs a self-contained token.

**API tokens (scripts, CLI).** `Authorization: Bearer op_<random>`; looked up by its SHA-256 (high
entropy, so a fast hash is fine) through a Redis cache key with a 60 s TTL that revocation deletes,
then OpenSearch on a miss. Tokens are read from files by every client (`--token-file`,
`OP_API_TOKEN_FILE`), never from an argv or a plain env value.

**Service token (internal workers).** One token with role `service`, generated by the installer
into `secrets/auth/service_token` (mode 600), mounted with Compose `secrets:` into the worker
containers that call the API (detection worker bulk writer, trainer campaign, auto-label, any
remaining event publisher). After B4 most events no longer go over HTTP; the lane lists the real
callers with `rg -n "OP_API_BASE_URL|OP_API\b|OP_EVENT_API_URL" docker-compose.yml scripts docker`.

**Authorization.**
- Roles: `viewer` (read), `editor` (curation writes, ingest, inference), `admin` (deployment
  config, users, project create and delete, model upload, export and delete, server-path reads,
  the VLM endpoint registry), `service` (what workers call, nothing else).
- One global dependency added in `create_app` (`FastAPI(dependencies=[Depends(authorize)])`) reads
  the matched route from the request scope and looks up its permission in one table,
  `src/services/auth/permissions.py`. Default by method (`GET`/`HEAD` -> viewer, others -> editor),
  plus an explicit override list (admin routes, public routes). A route missing from the table
  resolves to `admin` (fail closed).
- Public without login: `/live`, `/ready`, `/auth/login`, the Cropwright static assets.
  `/metrics` is `service` (Prometheus scrapes with `authorization.credentials_file`). `/health`,
  `/docs`, `/openapi.json` are `viewer`.
- Ratchet test: every route in the app resolves to exactly one permission; every override names
  an existing route; the admin list is reviewed in the PR (a security reviewer reads it).

**CSRF.** Cookie-authenticated unsafe requests must carry `X-OP-CSRF` equal to the
non-HttpOnly `op_csrf` cookie (double submit), and an `Origin` that matches the request's origin
(the existing `RequestOriginMiddleware` parses it). Bearer-token requests are exempt (a browser
cannot add the header cross-site). SSE (`EventSource`) is a same-origin GET and sends the cookie.

**Login limits.** `limits` with its Redis storage: per username and client address, and per
address; a refusal is 429 with `Retry-After`. Shared by all API workers. The message does not say
whether the user exists.

**Proxy mode.** Identity headers (`X-Forwarded-User`, `X-Forwarded-Email`, `X-Forwarded-Groups`,
as oauth2-proxy sets them) are read only when the direct peer is in `OP_AUTH_TRUSTED_PROXIES`
(CIDR list, matched against uvicorn's `--forwarded-allow-ips`). Roles come from group names
(`OP_AUTH_PROXY_ADMIN_GROUPS`, `OP_AUTH_PROXY_EDITOR_GROUPS`; any other authenticated user is a
viewer). No local users in this mode.

**Routes** (deployment-level, no `/v1` twin, like `/curation`):
`POST /auth/login`, `POST /auth/logout`, `GET /auth/me`, `GET /auth/check` (for nginx
`auth_request`: 204 or 401/403, with a role query), `GET /auth/users`, `POST /auth/users`,
`PATCH /auth/users/{username}`, `POST /auth/users/{username}/password`, `GET /auth/tokens`,
`POST /auth/tokens`, `DELETE /auth/tokens/{token_id}`.

**Cropwright.**
- `/login` page; the central fetch wrapper in `frontend/src/lib/api*.ts` turns a 401 into a
  redirect to `/login?next=<path>` and a 403 into a "not allowed" state; `GET /auth/me` fills a
  user store; admin pages (`/admin/users`, `/account/tokens`). The UI hides what a role cannot do,
  but the server is the only enforcement.
- nginx: the basic-auth include and the htpasswd handling are deleted. A `location /auth/` proxies
  to the API. Gateway sub-paths (Grafana, Prometheus, MLflow, Dashboards) are protected by
  `auth_request /auth/check?role=admin`, so the "whole origin behind a login" property of #351
  holds. The lane verifies that the `nginx-unprivileged` image has the `auth_request` module
  (`nginx -V`); if not, the gateway sub-paths stay refused when published beyond loopback.

**Audit.** `label_ops` entries, class history entries and job records gain `actor` (username,
`service`, or `local`). The wire adds the field; old entries have none.

#### 2.5.4 Migration from the existing basic auth

| Install before (0.7.0) | After `openprocessor upgrade` to 0.8.0 |
|---|---|
| Cropwright on loopback, basic auth off | `OP_AUTH_MODE=none`; nothing else changes |
| Cropwright on the LAN, basic auth on | `OP_AUTH_MODE=local`; the installer creates user `admin` (Q-4) with the password stored in `secrets/cropwright/password`, writes it to `secrets/auth/admin_password` (mode 600, never printed), then renames `secrets/cropwright/` to `secrets/cropwright.migrated/`; the service token is generated |
| Cropwright on the LAN, basic auth off (the user chose no login) | the upgrade stops and asks: `local` (recommended, a new admin password is generated) or loopback-only. Unattended runs choose `local` |
| Gateway mode on | as above; gateway sub-paths move from basic auth to `auth_request` |

- Retired: `OP_CROPWRIGHT_BASIC_AUTH` and `OP_CROPWRIGHT_BASIC_AUTH_FILE` (added to
  `src/config/retired_env.py`; the Cropwright entrypoint refuses them too). Deleted: the htpasswd
  parts of `scripts/lib/cropwright_auth.sh` (the bind chooser stays, moved if the file then holds
  one concern), the entrypoint's auth block, `include /etc/nginx/auth.d/*.conf`, the
  `openprocessor cropwright password` command, `tests/installer/test_cropwright_basic_auth.py`.
- New CLI: `openprocessor auth init`, `auth user add|passwd|role|disable|list`,
  `auth token create|revoke|list`, `auth mode show`. Passwords are read from a prompt or a file,
  never an argument.
- Rollback to 0.7.0 (`setup-openprocessor.sh --rollback`): renames `secrets/cropwright.migrated/`
  back and restores `OP_CROPWRIGHT_BASIC_AUTH` from the saved `.env`; 0.7.0 ignores the auth index.
- Re-upgrade: the admin user exists, the step is a no-op.

#### 2.5.5 Library and standard

- FastAPI security tutorial and `fastapi.security` classes (`APIKeyCookie`, `HTTPBearer`):
  https://fastapi.tiangolo.com/tutorial/security/ ; the JWT page (read 2026-11-04) recommends
  `pwdlib[argon2]` for hashing and PyJWT for JWTs: https://fastapi.tiangolo.com/tutorial/security/oauth2-jwt/
  (we use pwdlib; no JWT, see sessions)
- pwdlib: https://frankie567.github.io/pwdlib/
- FastAPI Users is in maintenance mode (read 2026-11-04: security fixes only, no new features,
  a successor announced): https://github.com/fastapi-users/fastapi-users . Not chosen.
- OWASP cheat sheets: Session Management https://cheatsheetseries.owasp.org/cheatsheets/Session_Management_Cheat_Sheet.html ;
  CSRF https://cheatsheetseries.owasp.org/cheatsheets/Cross-Site_Request_Forgery_Prevention_Cheat_Sheet.html ;
  Password Storage https://cheatsheetseries.owasp.org/cheatsheets/Password_Storage_Cheat_Sheet.html ;
  Authentication https://cheatsheetseries.owasp.org/cheatsheets/Authentication_Cheat_Sheet.html
- NIST SP 800-63B (passwords): https://pages.nist.gov/800-63-4/sp800-63b.html
- `limits` (Redis storage): https://limits.readthedocs.io/en/stable/
- Redis `SET ... EX`, sets: https://redis.io/docs/latest/commands/set/
- Cookies (`HttpOnly`, `Secure`, `SameSite`): https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Set-Cookie
- nginx `auth_request`: https://nginx.org/en/docs/http/ngx_http_auth_request_module.html
- oauth2-proxy (proxy mode headers): https://oauth2-proxy.github.io/oauth2-proxy/
- uvicorn `--forwarded-allow-ips`: https://www.uvicorn.org/settings/
- Docker Compose secrets: https://docs.docker.com/compose/how-tos/use-secrets/

Read the pages marked "read" again at kick-off; the others are to be read by the lane before
coding (owner rule: docs before hand-rolling).

#### 2.5.6 Tasks

Four lanes (section 3). Merge order: core, clients, cropwright, install.

| ID | Lane | Task | Files | Red test (fails today) | Control | Done when |
|---|---|---|---|---|---|---|
| AU-1 | core | identity store and guard shape | new `src/clients/auth_store.py` over `src/clients/occ.py`; `src/services/projects/guard.py`; `src/clients/curation_opensearch/` index body | guard test: a request to the auth index is refused today | every other refused shape still refused | create-only first admin; OCC user writes |
| AU-2 | core | passwords, sessions, tokens | new `src/services/auth/` (`passwords.py`, `sessions.py`, `tokens.py`), `requirements.txt` (`pwdlib[argon2]`, `limits`), `ATTRIBUTION.md` | session revoked through worker A is still accepted by worker B: test with two app instances on one fake Redis (red on a naive per-process cache) | login, logout round trip | revocation is immediate on all instances |
| AU-3 | core | `authorize` dependency and permission table | `src/app/factory.py`, new `src/services/auth/permissions.py` | ratchet test enumerating `app.routes`: no route has a permission today | `/live`, `/ready` public | every route resolves; unknown resolves to admin |
| AU-4 | core | modes and startup checks | `src/config/` (new `AuthConfig`), `src/app/lifespan.py` (check only, no side effect) | `OP_AUTH_MODE=none` with a non-loopback bind starts today | loopback `none` starts | the three refusal rows of 2.5.2 tested |
| AU-5 | core | CSRF and login limits | `src/services/auth/csrf.py`, the login route | cookie POST without `X-OP-CSRF` succeeds today (after AU-3) | bearer POST without it succeeds | 403 `csrf_required`; 429 with `Retry-After` |
| AU-6 | core | auth routes and wire | new `src/routers/auth.py`, `src/schemas/auth.py`, `contracts/` | route tests 404 today | | contracts regenerated; `docs/design/curation_api_contract.md` auth section |
| AU-7 | core | proxy mode | `src/services/auth/proxy.py` | identity headers from an untrusted peer are accepted (test on the dependency) | trusted peer accepted | group-to-role mapping tested |
| AU-8 | core | `actor` on label ops, history, job records | `src/services/curation/label_ops/`, `src/domain/history.py`, `file_job.py`, `wire.py`, `contracts/` | label op record has no actor | old records read fine (field optional on read, required on write) | actor present on every new write |
| AU-9 | clients | service token in workers; `--token-file` in scripts | `scripts/curation/worker/`, `scripts/curation/*.py` with `--api`, `docker/trainer/campaign.py`, `docker-compose.yml` (secrets) | a worker call without a token gets 401 once AU-3 is merged (red against the merged core) | | every internal caller authenticates; `rg` list in the PR |
| AU-10 | cropwright | login page, 401/403 handling, user and token pages | `frontend/src/routes/login/`, `frontend/src/routes/admin/`, `frontend/src/lib/api*.ts`, `frontend/src/lib/stores/` | e2e stubbed: an unauthenticated page load shows data today | logged-in flows unchanged | `npm run check`, `npm run build`, browser check 1440/430 px |
| AU-11 | cropwright | nginx: delete basic auth, add `/auth/` and `auth_request` | `frontend/nginx.conf`, `frontend/docker-entrypoint.sh`, `frontend/CHANGELOG.md` | config test: `auth_basic` present today | `/nginx-health` stays open | no htpasswd code left |
| AU-12 | install | installer modes, migration, CLI, retired env | `setup-openprocessor.sh`, `scripts/lib/cropwright_auth.sh`, `scripts/lib/cli_core.sh`, `openprocessor`, `env.template`, `src/config/retired_env.py`, `tests/installer/` | installer test: an upgrade with basic auth on leaves no API login today | loopback install unchanged | the migration table of 2.5.4 tested, rollback and re-upgrade included |
| AU-13 | install | docs | `SECURITY.md` (rewrite the "No authentication" section), `INSTALLATION.md`, `docs/ARCHITECTURE.md` (Security boundary), docs-site security and deployment pages | docs check | | `check_docs_vs_code.py` clean |

**Size.** XL (core L, clients S, cropwright M, install M). **Risks.**
- A missed internal caller breaks a worker silently: AU-9's red test runs against the merged core,
  and the W8.2 live run exercises every worker.
- Lockout of the only admin: `openprocessor auth user passwd admin` works from the host (it runs in
  the `api` container) without a login.
- Cookie without `Secure` on plain HTTP: documented; the installer prints a TLS proxy pointer for
  LAN installs.
- Performance: one Redis `GET` per request; the W8.2 live run records `/detect` p50/p99 with auth
  on against the v0.7.0 number (PQ-7 style).
- The permission table is security-relevant: an independent adversarial review (Sonnet, fresh
  context, given the code not this plan) before the wave PR, as the vision requires for security
  surfaces.

**Upgrade impact.** Breaking for LAN installs: a login replaces basic auth (CHANGELOG
"Breaking"). Scripts that call the API need a token file. Rollback documented.

---

### 2.6 F6: second segmenter backend (#367)

**Goal.** The segmenter service can run a second backend behind the same `POST /segment`,
`POST /segment/batch` and `GET /health` contract, so a deployment can choose a faster or
differently licensed model, and the code has a real adapter boundary.

**User-visible outcome.** `OP_SEGMENTER_BACKEND=yoloe` (or `sam3`, the default) at install; region
profiles and the open-vocabulary pass work unchanged; boxes record which backend found them; the
segmenter health card shows the backend.

**Today.** `docker/segmenter/main.py` serves `sam3_backend.py` only. The API and worker hard-code
`sam3` in several places: `DetectionProfile.segmenter_name = 'sam3'`
(`src/config/detection_profile.py`), the region profile model, `OPEN_VOCAB_DETECTOR = 'sam3'`
(`src/services/curation/open_vocab_run.py`), and the item filter origin `sam3`
(`src/services/curation/item_filter.py`), which really means "found by the open-vocabulary pass"
(field `open_vocab_set`).

**Design.**
- Service: a `SegmenterBackend` protocol in `docker/segmenter/` (`load()`,
  `segment(images, prompt, max_candidates, min_score, return_masks) -> list[list[Candidate]]`,
  `name`, `version`). `sam3_backend.py` implements it unchanged; new `yoloe_backend.py`. One backend
  per container, chosen by `OP_SEGMENTER_BACKEND` at start; an unknown value refuses to start. The
  Dockerfile gets one build target per backend (their Python dependencies differ); the YOLOE target
  installs `ultralytics==X`, the B3 pin, and the pin test reads that image too.
- Contract: request and response models unchanged. `GET /health` adds `backend` and
  `backend_version`. Candidates keep crop-frame normalized `bbox_norm`, `score`, `mask_iou`,
  `mask_polygon`.
- YOLOE specifics: the prompt is set per request with `set_classes([prompt])`; text embeddings
  are cached per prompt in the process (an LRU keyed by the prompt string; per-process cache is
  correct here, it is a pure function of the prompt). Served in PyTorch, not TensorRT: a TensorRT
  export bakes the classes in, and prompts change per request (YOLOE docs). `max_candidates` maps to
  `max_det`, `min_score` to `conf`.
- API and worker: the box `detector` value becomes the backend name reported by `/health` (`sam3`
  or `yoloe`), with `segmenter_version` from `backend_version`. A region profile may name an
  expected backend (`segmenter_backend`, optional); a mismatch with the running service refuses the
  profile test and pauses the region stage with `segmenter_backend_mismatch` (fail closed). The
  origin filter value `sam3` is renamed `open_vocab` (wire change, contracts, Cropwright filter
  labels); the stored field `open_vocab_set` is unchanged, so no data migration.
- One backend per deployment in v0.8.0 (recommendation; Q-5 asks about per-project choice, which
  would need two services and per-backend URLs).

**Library and standard.**
- YOLOE (read 2026-11-04): text prompts, instance masks on `-seg` checkpoints, `set_classes`,
  YOLOE-26 needs `ultralytics>=8.4.0`, exported models bake prompts in, AGPL-3.0 or an Ultralytics
  enterprise licence: https://docs.ultralytics.com/models/yoloe/
- Alternatives for Q-5: Grounding DINO in Transformers (Apache-2.0)
  https://huggingface.co/docs/transformers/model_doc/grounding-dino , OWLv2
  https://huggingface.co/docs/transformers/model_doc/owlv2 , SAM 2 for masks
  https://github.com/facebookresearch/sam2 . Each would be another backend behind the same protocol.
- SAM 3 backend unchanged (`docker/segmenter/README.md`).

**Tasks.**

| ID | Task | Files | Red test | Control | Done when |
|---|---|---|---|---|---|
| SG-0 | spike SP-2 (GPU window V0): YOLOE-26-seg per-request prompt latency, `set_classes` cost, cached-embedding reuse, quality on the public wheel set vs SAM 3 | scratch, result in the PR and `docs/research/` | n/a | n/a | numbers recorded; go or switch candidate (Q-5) |
| SG-1 | backend protocol, SAM 3 behind it | `docker/segmenter/main.py`, new `docker/segmenter/backend.py`, `sam3_backend.py` | contract test with a fake backend: `main.py` cannot take one today | SAM 3 responses byte-equal on a recorded fixture | |
| SG-2 | YOLOE backend and build target | `docker/segmenter/yoloe_backend.py`, `docker/segmenter/Dockerfile`, `tests/test_requirements_contract.py` (pin) | unit test on a stub YOLOE model: normalized boxes, `max_det`, `conf` mapping | | pin test covers the image |
| SG-3 | `/health` backend fields; API reads them | `docker/segmenter/main.py`, `src/services/detection/segmenter_health.py`, `segmenter_http.py` | health parse test: no `backend` field today | | |
| SG-4 | box `detector` from the backend; profile expectation; fail closed | `src/config/detection_profile.py`, `src/routers/curation/_region_profile_models.py`, `scripts/curation/worker/`, `open_vocab_run.py` | worker writes `sam3` while the service reports `yoloe` (red) | SAM 3 deployments write `sam3` as today | `segmenter_backend_mismatch` tested |
| SG-5 | origin `sam3` -> `open_vocab` | `item_filter.py`, `_item_filter_params.py`, `review_filter_specs.py`, Cropwright item filter, `contracts/` | `origin=open_vocab` is a 422 today | the clause body is unchanged | `rg -n "'sam3'" src` lists only the SAM 3 backend name |
| SG-6 | compose, installer, docs | `docker-compose.yml` (`segmenter` service env), `scripts/lib/image_keys.sh`, `docker/segmenter/README.md`, `docs-site` segmenter pages, `ATTRIBUTION.md` | | | |
| SG-7 | GPU parity (window V1) | `scripts/` bench | n/a | n/a | both backends pass the live `/segment` contract suite; per-crop latency and box recall vs SAM 3 on the public wheel set in `docs/PERFORMANCE.md` |

**Size.** L. **Risks.** YOLOE quality on a domain prompt may be well below SAM 3; the backend is an
option, not the default, and the numbers go to the owner. Two dependency stacks in one Dockerfile:
separate targets. **Upgrade impact.** The `origin=sam3` filter value is renamed (breaking for saved
links and scripts; CHANGELOG). Existing boxes keep `detector=sam3`, which stays true.

---

### 2.7 F7: video files as an ingest source (N17, first slice of #8)

**Goal.** A user can bring video files into a project. Frames are sampled on the GPU, deduplicated,
and enter the normal ingest path, so detection, regions, labeling, review, export and training work
on video-derived images with no new pipeline.

**Why this slice.** #8 describes a full live video analytics subsystem (RTSP, tracking, a message
bus, a SQL store, Helm). The hardened DeepStream image it relied on was removed from the repo
(issue comment of 2026-10-31), and that subsystem is a different product from dataset curation.
Curation users mostly need "turn my recordings into a dataset". That is this slice; the rest stays
parked in #8 (section 6).

**User-visible outcome.** On the ingest page, a Video tab: upload one or more files (or point at a
folder source with video files), choose sampling (every N seconds, a frame cap, a start and end,
skip near-identical frames), watch progress. Items show "from video V at 00:01:23". Exports keep all
frames of one video in one split.

**Design.**
- Entry points:
  - `POST /curation/projects/{project}/ingest/video` (multipart upload, size cap
    `OP_VIDEO_UPLOAD_MAX_MB`), answers 202 with a job id;
  - folder and S3 sources may opt in with a `video` sampling block; each video file found becomes
    a job (task VI-6, optional in v0.8.0, Q-7).
- Job: `op.jobs.video_extract` on a new queue `video`, consumer `video-worker` (profile `video`,
  GPU from `VIDEO_GPU_ID` in `.env`, `-c 1`). Lease kind `video_extract`, scope `<video_id>`.
  Auto-resume: yes. Frame ids are deterministic (`<video_id>:<pts_ms>`), so a resumed job never
  writes a frame twice; image writes are create-only.
- Decode: PyNvVideoCodec on the GPU (NVDEC). Its demuxer uses FFmpeg. A CPU backend with PyAV runs
  the offline tests and hosts without a supported NVDEC codec (Q-7: keep the CPU path as a
  supported fallback or as test-only). One `VideoFrameSource` interface, two backends, chosen by
  `OP_VIDEO_DECODE=gpu|cpu`; `gpu` refuses an unsupported codec rather than silently using the CPU.
- Sampling: `every_s` (default 1.0), `max_frames` (default 2000), `start_s`, `end_s`,
  `dedup_threshold` (optional). Near-identical frames are dropped by the existing frame dedup
  (`src/services/detection/frame_dedup.py::dedup_rows_by_embedding`) on PE embeddings, so there is
  one dedup rule in the codebase. Seek to keyframes where `every_s` is large.
- Frames: encoded to JPEG on the GPU (`torchvision.io.encode_jpeg` on a CUDA tensor), written to the
  project's upload root under `video/<video_id>/<pts_ms>.jpg`, then ingested through the same
  service the upload route uses (one ingest path). Feeding decoded frames straight into the B5
  resident pipeline, without the JPEG round trip, is a measured follow-up (risk row).
- Storage: images index gains `video_id`, `video_name`, `frame_pts_ms`, `frame_index` (mapping
  migration through the `curation_opensearch` mapping migrations); items inherit `video_id`.
- Leakage: the export splitter's group key becomes `video_id` when present, else `image_id`
  (`src/services/curation/export_split.py::DEFAULT_SPLIT_GROUP_KEY`). Frames of one video are near
  duplicates; a split across them leaks test into train. The holdout freeze uses the same key.
- Cropwright: Video tab on the ingest page; progress from the job record and the event stream;
  provenance on the item detail.
- Permissions (F5): upload is `editor`; a folder source with video is configured by `admin` like
  other sources.

**Library and standard.**
- PyNvVideoCodec (read 2026-11-04): NVIDIA's Python video decode and encode library, successor of
  VPF, MIT licence, `pip install PyNvVideoCodec`, FFmpeg for demuxing, DLPack frames:
  https://docs.nvidia.com/video-technologies/pynvvideocodec/ and https://developer.nvidia.com/pynvvideocodec
- NVIDIA Video Codec SDK (NVDEC) and the decode support matrix:
  https://developer.nvidia.com/video-codec-sdk ,
  https://developer.nvidia.com/video-encode-and-decode-gpu-support-matrix-new
- PyAV (FFmpeg bindings, CPU decode, used to build test clips): https://pyav.basswood-io.com/docs/stable/
- FFmpeg documentation and licensing (LGPL build in wheels; record it in ATTRIBUTION):
  https://ffmpeg.org/documentation.html , https://ffmpeg.org/legal.html
- `torchvision.io.encode_jpeg` (CUDA tensors supported):
  https://docs.pytorch.org/vision/stable/generated/torchvision.io.encode_jpeg.html
- DeepStream SDK is the NVIDIA product for live multi-stream analytics; it is the right tool for the
  parked live part of #8, not for this batch slice: https://docs.nvidia.com/metropolis/deepstream/dev-guide/

**Tasks.**

| ID | Task | Files | Red test | Control | Done when |
|---|---|---|---|---|---|
| VI-0 | spike SP-3 (GPU window V0): PyNvVideoCodec in the API base image, decode of H.264 and HEVC 1080p, frames as CUDA tensors, GPU JPEG encode | scratch, PR notes | n/a | n/a | go, with the wheel and driver versions recorded |
| VI-1 | frame source interface, CPU backend | new `src/services/video/frame_source.py`, `frame_source_cpu.py`; `requirements*.txt` (`av`) | fixture: a 10 s H.264 clip built with PyAV from 10 public COCO val2017 images (1 s each): `every_s=1` gives 10 frames with the right pts (module absent today) | | |
| VI-2 | GPU backend | `frame_source_gpu.py` (lazy import), `requirements.txt` (`PyNvVideoCodec`), `ATTRIBUTION.md` | unit test with a stub decoder: unsupported codec refuses with `video_codec_unsupported` | | live parity in V1 |
| VI-3 | sampling and dedup | `src/services/video/sampling.py` | clip with a repeated image: dedup on keeps one frame per distinct image | dedup off keeps all | |
| VI-4 | `video_extract` job, queue, worker service | `src/tasks/` (B2 layout), task routes, `docker-compose.yml` (`video-worker`, profile `video`), `tests/test_task_routes_complete.py`, `tests/test_queue_consumers_from_compose.py` | the B2 guards fail until routed and consumed | | lease: two starts for one video give one job; a killed worker resumes with no duplicate frame |
| VI-5 | upload route and wire | `src/routers/curation/ingest_video.py`, `src/schemas/curation/`, `contracts/`, F5 permission table | 404 today | | 202 + job id; size cap 413 |
| VI-6 | folder and S3 sources with a `video` block (optional, Q-7) | `src/services/curation/bulk_ingest/walk.py`, `sources.py`, `_ingest_sources_models.py` | a source with `.mp4` files ingests nothing today | image sources unchanged | |
| VI-7 | mapping and split key | `src/clients/curation_opensearch/` (images and items bodies, a mapping migration), `export_split.py`, holdout freeze | two frames of one video land in train and test today | non-video items still group by `image_id` | |
| VI-8 | Cropwright Video tab and provenance | ingest page, item detail, `frontend/src/lib/api_ingest.ts`, `types_ingest.ts` | component tests | | browser check 1440/430 px |
| VI-9 | GPU measurement (window V1) | `scripts/bench/` | n/a | n/a | decode fps and frames/s end to end on a 1080p clip synthesized from public images; CPU vs GPU frame parity (boxes from `/detect` within 1 px); numbers in `docs/PERFORMANCE.md` |

**Size.** L. **Risks.**
- NVDEC codec support differs by GPU generation: fail closed per codec, documented with the
  support matrix.
- The JPEG round trip costs time and storage; it keeps one ingest path. Measure in V1; a resident
  path is a follow-up with its own parity gate.
- Storage growth: `max_frames` default and the frame count shown before the job starts (a dry-run
  estimate from duration and `every_s`).
- Test media: only clips synthesized from public COCO images, built in fixtures, never committed.

**Upgrade impact.** New optional service and profile; a mapping migration adds four fields
(additive, no reindex). The split key change affects only items with `video_id`, which exist only
after the upgrade.

---

### 2.8 F8: public wheel teacher-student example at scale (#84)

**Goal.** A reproducible, public demonstration of the loop the product exists for: label with the
segmenter once, train a small detector, run that detector at scale, compare.

**Today.** The 60-photo wheel example, region profile, prompt pack, bake-off profile, live script
and guide shipped before 0.6.0 (`docs-site/docs/guides/wheel-example.mdx`,
`scripts/examples/wheel_example_live.py`, `make sample-coco-cars`).

**Design.** Exactly the issue's remaining boxes:
- scale the sample to a few thousand COCO cars (Q-13 sets the number; start at 3,000) with a pinned
  manifest (image ids and SHA-256), CC BY 4.0 annotations, images under the COCO terms;
- region stage with SAM 3 to propose and verify wheel boxes; export the validated boxes as a
  single-class dataset (`export_single_class`); train YOLO26n; bake off; promote;
- switch the region profile's detector leg to the trained model (segmenter-free) and compare with
  SAM 3 on a held-out COCO subset: images/s, calls/s, GPU time, box IoU and recall against the SAM 3
  labels;
- optional (if F6 merged): the same comparison with the YOLOE backend.
Class identity stays by name (`wheel`) through train, promote and the profile.

**Library and standard.** COCO dataset and licences: https://cocodataset.org/#termsofuse ;
Ultralytics train and val: https://docs.ultralytics.com/modes/train/ , https://docs.ultralytics.com/modes/val/

**Tasks.**

| ID | Task | Files | Red test | Control | Done when |
|---|---|---|---|---|---|
| WE-1 | manifest and fetch | `scripts/datasets/`, a `make sample-coco-cars-full` target (or a flag on the existing one) | manifest checksum test on a 10-image slice | the 60-photo sample unchanged | deterministic set |
| WE-2 | end-to-end script | `scripts/examples/wheel_example_live.py` (extend, do not fork) | offline dry run with fakes covers every step | existing live walk-through unchanged | one command from a clean project |
| WE-3 | the run (window V1) | none | n/a | n/a | the results table in `docs/PERFORMANCE.md`; raw output kept out of `/tmp` |
| WE-4 | guide | `docs-site/docs/guides/wheel-example.mdx` | docs check | | tutorial steps and the table |

**Size.** L, about half a GPU day in V1. **Risks.** GPU time; the run is announced and capped.
**Upgrade impact.** None.

---

### 2.9 F9: embedding-model swap (#365), conditional

**Re-scoped by the 0.7.0 research integration (2026-11-04, pending RD-1).** The project setting, the
dual field, the migrate job and the routes exist from v0.7.0 (lanes ES-A, ES-B, ES-C of the 0.7.0
program: the embedding registry, `embedding_reindex` on Celery and PostgreSQL with no lease, the
fenced flip). F9 becomes: register the winning model version in the catalog, build its space,
evaluate it in-product against the primary, flip; size M instead of XL. The gate is the embeddings
research plan's EP-4 verdict (research wave RW-2), no longer the clustering plan. Q-8 is answered by
RD-33 (copy matching spaces, defer the rest). F2's `group_by` composes with the search `space`
parameter. The design text below is the earlier one and is superseded at W8.0 by
`docs/design/research_embeddings_search_plan.md` sections 5 and 8.2.

**Gate.** Build only if `research_clustering_plan.md` (the #347 evaluation) names an encoder that
passes its registered gain over PE-Core-L on the public benchmarks by the W8.3 sweep. Without a
candidate the machinery would be dead code, so #365 then moves to parked (section 6).

**Goal.** Switch a project's encoder without losing search, clustering or training, by writing the
new vectors next to the old ones and flipping once.

**Design.**
- Project setting (config store, revisioned): `encoder` (today implicitly `pe_core_l`) and, during a
  migration, `encoder_next`. Every reader takes the vector field name from the bound project
  (`project.embedding_field`), never from the constant `ITEM_EMBEDDING_FIELD`; a ratchet test pins
  that (the existing `tests/curation/test_embedding_state_consumers.py` pattern).
- Dual field: put-mapping adds `emb_<encoder_id>` on items, images and the per-box sibling
  (`index.knn` is already true on these indexes; a new field may have another dimension).
- `op.jobs.embedding_migrate` (queue `jobs`, GPU via the encoder on Triton, lease
  `embedding_migrate`, auto-resume): re-embeds every item, box and image into the new field, then
  re-runs clustering and the cluster models on it. Search keeps using the old field until the flip.
- Flip: one OCC write of the setting; readers pick it up through the config store poll. Then a
  task clears the old field's values; the mapping keeps the field (OpenSearch cannot remove a
  mapped field; documented).
- Combine refuses projects with different encoders, or the preview warning (#55) becomes a refusal
  (fail closed; Q-8).
- Routes (admin): start, status, flip, abort. Cropwright: a settings panel with progress.

**Library and standard.** OpenSearch put mapping (add fields, no removal):
https://docs.opensearch.org/latest/api-reference/index-apis/put-mapping/ ; k-NN vector field:
https://docs.opensearch.org/latest/field-types/supported-field-types/knn-vector/ ; the encoder's own
model card and licence (from the research plan).

**Tasks.** EM-1 reader ratchet (all readers via the project field; red: the constant is read in
N places today); EM-2 setting and mapping; EM-3 migrate job with lease and resume (red: kill mid-job
resumes without re-embedding finished items); EM-4 flip and clear; EM-5 combine rule; EM-6 routes,
contracts, Cropwright; EM-7 GPU run (window V-EM): a public COCO project migrated, search recall and
cluster quality before and after recorded.

**Size.** XL. **Risks.** Disk: two vector fields during a migration (the size is shown before the
start, from the item count and dimension). **Upgrade impact.** None until a project migrates.

---

### 2.10 F10: S3-compatible storage backend for multi-host installs (N21; D-46)

**Goal.** API and worker replicas on different hosts share blobs through an S3-compatible object
store instead of one POSIX volume.

**User-visible outcome.** An install can set `OP_STORAGE_BACKEND=s3` with an endpoint, bucket and
credentials; uploads, crops, exports, training artifacts, model files and the upgrade bundle go to
the store; a migration command moves an existing install's blobs and verifies them; nothing else
changes for the user.

**Today (after v0.7.0).** Every blob goes through `src/clients/storage/` (`BlobStore`: `open_read`,
`write_atomic`, `exists`, `stat`, `list_prefix`, `delete_prefix`, `local_path`) with one backend,
`LocalFsStore`; database rows hold relative keys only (a ratchet enforces it); the roots are
`source`, `projects`, `models`, `cache`, `upgrade` (0.7.0 program `02c` section 5).

**Design.**
- A second backend `S3Store` behind the same protocol, chosen per deployment; the library is chosen
  at W8.0 from maintained S3 clients (the AWS SDK for Python `boto3` and `aiobotocore` are the
  documented candidates; MinIO's community distribution changed in 2025, so a server is not bundled:
  the operator brings an S3-compatible endpoint) with the licence read then.
- `write_atomic` maps to a single PUT (objects appear whole); multipart above a size threshold.
  `delete_prefix` lists and deletes in batches. `stat` and `exists` are HEAD requests.
- `local_path(key)` materializes the object into job-local scratch and returns the path; writes from
  code that needs a real file (Ultralytics training, engine builds, `faiss.write_index`, COLMAP's
  scratch database in a later spatial lane) go to scratch and are uploaded with `write_atomic` when
  the step ends; nothing reads a shared path.
- Triton reads the `models` root from the store natively (its cloud-storage model repository);
  the `source` root stays read-only (local or a read-only bucket); the `cache` root may stay local
  per host (regenerable derivatives, as Immich keeps them apart from originals).
- Migration: `openprocessor storage migrate --to s3` copies every key of every root, verifies size
  and SHA-256 per object against the source, and switches the setting only after a clean verify;
  `--dry-run` lists counts and bytes. No database row changes (keys are relative).
- Credentials from a secrets file, never in env values or logs; TLS on by default; the bucket is
  private (the API still has no authentication: SECURITY says so).

**Library and standard.** S3 API semantics (strong read-after-write consistency):
https://docs.aws.amazon.com/AmazonS3/latest/userguide/Welcome.html#ConsistencyModel ; Triton cloud
storage: https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_repository.html#cloud-storage-with-environment-variables ;
boto3: https://boto3.amazonaws.com/v1/documentation/api/latest/index.html .

**Tasks (lanes).**

| Lane | Owns | Red test (fails before) | Control |
|---|---|---|---|
| `v8/storage-s3-core` (ST-1..ST-4) | new `src/clients/storage/s3.py`, `src/config/storage.py` (the backend setting and its secrets), the shared `BlobStore` contract tests run against both backends | `tests/clients/storage/test_contract_s3.py` against a local S3-compatible test container: atomic visibility, prefix list and delete, `local_path` materialize and write-back | the same suite on `LocalFsStore` unchanged |
| `v8/storage-s3-consumers` (ST-5, ST-6; after core) | the call sites that need a real file (training, export, the FAISS index store) moved to scratch-and-upload; Triton's repository URL from the setting | `tests/storage/test_no_shared_path_reads.py`: an AST scan finds no direct filesystem read of a `projects` or `models` key outside the backend | a local install behaves as before |
| `v8/storage-s3-migrate` (ST-7) | the `storage migrate` CLI and its script; INSTALLATION multi-host section | `tests/storage/test_migrate_verify.py`: a corrupted object stops the switch | a clean copy switches |
| `v8/storage-s3-live` (ST-8, in the W8.3 sweep) | none (coordinator) | a two-host install (two compose projects on one host with the object store, `-p op-v8-s3a`, `-p op-v8-s3b`) ingests, trains, exports and promotes; R8 row | - |

**Size.** L. **Risks.** Small-object latency on the crop cache (keep `cache` local per host);
partial uploads (single PUT or completed multipart only); cost of listing large prefixes (delete in
batches, keys stay prefix-scoped). **Upgrade impact.** None until an install opts in; then the
migrate command.

### 2.11 Landings from the post-tag research waves

The 0.7.0 program's research waves RW-1..RW-6 (`docs/design/program/00e_research_track.md`
section 5) land these into v0.8.0, each only on its prototype's PASS: F9 reduced (RW-2); F2
`group_by` with `space` (RW-2); F3's review queue and metrics (RW-6); `feat/loop-strategies` and
`feat/loop-z5i` (RW-6); the faces lane L5 and the verification tables L6 with the Cropwright review
screens (RW-3); the geo-photo timeline L-T1 (RW-5, under the location policy); the two-card serving
default (RW-1). Each joins the W8.x wave whose files it touches, decided at W8.0.

### 2.12 Carry-over from v0.7.0

At the v0.7.0 tag the 0.7.0 program moves still-open items to v0.8.0. They are cleanup, not
features, and fill W8.1 lanes.

| Item | Source | Plan |
|---|---|---|
| N15 remainder (typed response models) | audit BA-6 | one router per lane commit; contracts in the same commit; no behaviour change |
| N14 leftovers (OS-8, FE-6, T-6) | audit evidence tasks | measure first; act only on a measured defect |
| GPU-15 leftover | audit 4.3 | refuse a model with an undeclared region head at profile validation (low) |
| LIB-6 | library register | Pillow `ImageOps.pad` for the training-export letterbox only if the golden test shows no pixel change, else record and keep |
| #42 | 0.7.0 side track | if still unrun at the tag, it stays a non-blocking side track on GPU 1 |

## 3. Waves

### 3.1 Rules (restated from the 0.7.0 program, which is deleted after its tag)

- A wave is closed by exactly one full sweep on its integration branch. While one sweep runs, the
  next wave's lanes are already coding from the frozen head.
- One lane = one agent in its own worktree `<repo>-wt-v8-<lane>` on branch `v8/<lane>`. Never two
  writers in one worktree. A lane commits (one commit per concern, hooks on, never `--no-verify`),
  never pushes, never merges, never checks out another branch.
- Lanes run T0 only. The coordinator merges lanes with `--no-ff` in the stated order, one merge
  commit per lane (a failing lane reverts alone with `git revert -m 1`), then runs the sweep.
- Agents: Opus plans, Sonnet implements and reviews. Every brief forbids `pgrep -f`, `pkill -f`
  and sleep or until loops, and names its compose project (`-p`), GPU, worktree and deadline.
- Exit gates per wave (the PQ list): CI reproduced locally first; ratchet baselines only shrink; no
  dead code or shims; contracts regenerated; docs check clean; red/green evidence in the PR;
  hot-path numbers where the wave touches the hot path; image audit and Trivy when an image is
  built; owner rules (no new lock for cross-process state, no request fan-out, class by name,
  fail closed).
- Host: check `uptime` and `nvidia-smi`; message the peer sessions before any heavy run or GPU use
  and wait for the reply. GPUs 0 and 2 for this repo; GPU 1 only when its owning session frees it.
- Releases: no VERSION bump, image push or GitHub release between waves. Breaking changes collect
  under `[Unreleased]` in `CHANGELOG.md` (root and frontend).

### 3.2 Wave table

| Wave | Integration branch | Lanes (merge order) | Codes in parallel | Sweep that closes it | Exit criteria (plus the gates above) |
|---|---|---|---|---|---|
| **W8.0** kick-off | none | none. The coordinator: checks v0.7.0 is tagged and `program/` removed; runs the GitHub actions of section 5 on the owner's go; gets answers to section 7; runs spike SP-1 (OpenSearch only, CPU, 10 min, `-p op-v8-sp1`) and GPU window **V0** for SP-2 and SP-3 (at most 3 h, GPU 0) | W8.1 lanes may start once their decisions are answered | none | decisions recorded in this file (a docs commit); spike results in the PRs that use them |
| **W8.1** finish | `int/v080-w1-finish` | `v8/coco-seg` (F1), `v8/search-group` (F2), `v8/assist-loop` (F3 code), `v8/train-families` (F4 CPU tasks FT-1..FT-6), `v8/carry-<item>` (one per carry-over row, if any) | all in parallel; W8.2 lanes start during this sweep | T1: pre-commit `--all-files`, offline suite `-n 8`, `make contracts-check`, docs check, changelog rules, frontend full set (`npm run check`, `npm run build`, e2e stubbed). No live, no GPU | #370 and #436 close on merge; #435 decided |
| **W8.2** auth | `int/v080-w2-auth` | `v8/auth-core` (AU-1..AU-8), `v8/auth-clients` (AU-9), `v8/auth-cropwright` (AU-10, AU-11), `v8/auth-install` (AU-12, AU-13) | core first; the other three branch from core's frozen head once AU-3 and AU-6 are committed; W8.3 lanes start during this sweep and branch from the merged auth head | T1 + frontend full set with e2e + **live harness run 1** (`-p op-v8-auth-live`, about 2 h): the three modes, the permission matrix per role, session revocation across `API_WORKERS=1` and `=32` (the B4 diff extended to auth), CSRF, login limits across workers, every worker with the service token, `/detect` p50/p99 with auth on vs the v0.7.0 number + an independent security review of the permission table and the store | #369 closes on merge; SECURITY.md rewritten; review findings fixed in the wave |
| **W8.3** breadth | `int/v080-w3-breadth` | `v8/segmenter` (F6), `v8/video` (F7), `v8/wheel` (F8 WE-1, WE-2, WE-4), `v8/storage-s3-core` then `v8/storage-s3-consumers` and `v8/storage-s3-migrate` (F10; the S3 lanes touch no F6-F8 file) | F9 lanes may code during this sweep only if its gate passed | T1 + frontend full set + image audit and Trivy for the new and changed images (segmenter targets, video worker) + **GPU window V1** in this order: F3 S0 (about 2 h), F6 SG-7 (about 3 h), F7 VI-9 (about 2 h), F4 FT-7 (about 3 h), F8 WE-3 (about 6 h). GPU 0, `-p op-v8-gpu` | #367, N17, #84 and #368 close on merge; numbers in `docs/PERFORMANCE.md` |
| **W8.4** (conditional) | `int/v080-w4-embed` | `v8/embed-swap` (F9, now register, build, evaluate, flip) | | T1 + GPU window **V-EM** (about half a day) | #365 closes; or the wave is skipped and #365 parked |
| **W8R** release | `chore/release-0.8.0-prep`, then R8 | release prep (VERSION 0.7.0 -> 0.8.0, CHANGELOG `[Unreleased]` -> `[0.8.0]`, release notes, `docs/releases/v0.8.0.md`); `fix/r8-findings` only if a row fails | nothing else merges | R8 itself (section 4), **GPU window V2** and **live harness run 2** inside it | every R8 row PASS; tag v0.8.0 only on the owner's word |

### 3.3 Test and GPU budget for v0.8.0

| Tier | Budget |
|---|---|
| T0 per task | the red/green test plus its control, static checks on touched files. Nothing else in a lane |
| Offline sweep | exactly one per wave (W8.1, W8.2, W8.3, W8.4 if run, release prep), on the merged integration branch |
| Spikes | SP-1 (OpenSearch only, CPU, 10 min) and SP-2 and SP-3 inside GPU window V0. A spike is throwaway code; its result is written into the PR that uses it |
| Live harness | exactly two runs: W8.2 (auth, about 2 h) and R8 (the full live suite inside the acceptance run) |
| GPU windows | V0 spikes (at most 3 h), V1 at the W8.3 sweep (about 1.5 GPU days, five measurements in sequence), V-EM only if F9 runs (about 1 day), V2 inside R8. Each window is announced to the peer sessions and runs on the agreed GPU |
| Re-runs | a failed window or live run is re-run only after the fix is merged to the integration branch, not per attempt |
| CI | each wave PR, the release-prep PR and the release commit |

Consequence: a lane that needs a GPU number waits for its window. Nothing in W8.1 or W8.2 needs a
GPU.

## 4. Release acceptance additions for v0.8.0 (R8)

R8 runs the v0.7.0 R-final procedure (kept after the tag in `docs/design/release_acceptance_plan.md`
and `RELEASE_CHECKLIST.md`; RF rows re-numbered as R8 equivalents by release prep) with the supported
upgrade **0.7.0 -> 0.8.0** and rollback to 0.7.0, plus these rows. Public data only.

| ID | Feature | Check | Expect |
|---|---|---|---|
| R8-01 | F5 | fresh LAN install (`OP_AUTH_MODE=local`) | Cropwright shows a login; the admin password is in `secrets/auth/admin_password` (mode 600) and in no log |
| R8-02 | F5 | the permission matrix: a scripted call of every route as anonymous, viewer, editor, admin, service | anonymous 401 except public routes; each role gets exactly its table; unknown routes refused |
| R8-03 | F5 | `OP_AUTH_MODE=none` with a LAN bind | the API refuses to start, naming the setting |
| R8-04 | F5 | proxy mode behind oauth2-proxy with a test OIDC provider in the harness; identity headers sent directly from an untrusted peer | proxy users work with mapped roles; direct headers 401 |
| R8-05 | F5 | log out in one browser; replay its cookie against several API workers | 401 on every worker |
| R8-06 | F5 | a disabled user's API token and session | both refused at once |
| R8-07 | F5 | cross-site POST with the session cookie | 403 `csrf_required` |
| R8-08 | F5 | wrong passwords from two clients across workers | 429 after the shared limit |
| R8-09 | F5 | upgrade a real 0.7.0 install with basic auth on | login as `admin` with the old stored password; `secrets/cropwright.migrated/` present; no basic auth in nginx; gateway sub-paths need an admin login |
| R8-10 | F5 | rollback to 0.7.0 after R8-09, then re-upgrade | 0.7.0 basic auth works; re-upgrade is a no-op; the user store is unchanged |
| R8-11 | F5 | every worker, the trainer and the scripts with `--token-file` | jobs, training, import client and VLM CLI work; no 401 in any log |
| R8-12 | F5 | label op and history entries after a labeling session | `actor` equals the user |
| R8-13 | F5 | SSE pages after an API restart | resume with the cookie, no lost or duplicated event |
| R8-20 | F2 | `group_by=image` on a project with many items per image | one row per image, a full page, stable cursor paging |
| R8-21 | F1 | import a COCO JSON with polygon-only and RLE-only annotations (public images) | boxes created; issue counts in the report |
| R8-22 | F3 | pause, resume and a token budget on a run against the local VLM | pause at a batch boundary; `budget_exhausted` with the token count |
| R8-23 | F4 | train, promote and serve one non-YOLO26 family | `/detect` serves `[300,6]`; class names by name |
| R8-24 | F6 | install with `OP_SEGMENTER_BACKEND=yoloe`; region profile test and an open-vocab pass | candidates returned; boxes carry `detector=yoloe`; a profile expecting `sam3` is refused |
| R8-25 | F6 | the default `sam3` install | unchanged behaviour; `origin=open_vocab` filter works |
| R8-26 | F7 | upload a clip synthesized from public images; kill the video worker mid-job | the job resumes; frame count equals the sampling plan; no duplicate frame; export keeps one video in one split |
| R8-27 | F7 | an unsupported codec with `OP_VIDEO_DECODE=gpu` | refused with `video_codec_unsupported` |
| R8-28 | F8 | the wheel example from a clean project | runs end to end; the table matches `docs/PERFORMANCE.md` within the stated spread |
| R8-29 | F9 (if shipped) | migrate a public project's encoder, flip, search | search and clustering work on the new field; the old field is cleared |
| R8-30 | all | image-vs-git audit and Trivy on every published image, the new segmenter target and the video worker included | fixable CRITICAL = 0; licence scan clean for the new dependencies |

Production-ready additions (to the v0.7.0 definition): every route authenticated and authorized by
one table (PR-A1); the auth state is shared across workers through Redis and OpenSearch, never per
process (PR-A2); the basic-auth upgrade and rollback work (PR-A3); new features are fail-closed on
an unknown backend, codec or encoder (PR-A4).

## 5. GitHub actions

Execute only on the owner's go (W8.0). Rules as in the 0.7.0 program: neutral security wording; no
private names, machine paths or other repositories in any text; fewer issues is better; edit, move
and close rather than delete. Record created numbers in `artifacts_local/gh/created.tsv` and then in
this file.

### 5.1 Milestone

Edit the v0.8.0 description:
`Platform breadth after the 0.7.0 program: finish import, search, assist-loop controls and training families; API authentication and authorization; a second segmenter backend; video files as an ingest source; the public wheel example at scale; the embedding swap only if the research picks a candidate. One release after one acceptance run. Plan: docs/design/v080_platform_breadth_plan.md`

### 5.2 New issues

| N | Title | Labels | Milestone | Brief |
|---|---|---|---|---|
| N17 | `feat(ingest): video files as an ingest source (sampled frames, GPU decode)` | enhancement, area:ingest, area:curation, effort:hard | v0.8.0 | Problem: users with recordings cannot curate them. Scope: section 2.7 (upload route, `video_extract` job on a `video` queue, PyNvVideoCodec decode with a PyAV CPU path, sampling and dedup, provenance fields, split by video, Cropwright Video tab). Out of scope: live streams, tracking, the rest of #8. Acceptance: R8-26, R8-27. Plan: this file |
| N18 | `chore(release): 0.8.0 close-out` | chore | v0.8.0 | the checklist of section 3.2 waves and R8; closes after the tag (like #411 for 0.7.0) |
| N21 | `feat(storage): an S3-compatible storage backend for multi-host installs` | enhancement, effort:hard | v0.8.0 | the exact text is in the 0.7.0 program's `00b_traceability.md` section 4.6 (filed by the 0.7.0 coordinator); plan section 2.10 |

No other new issues: the remaining work fits the existing issues.

### 5.3 Re-scope comments

| ID | Issue | Comment (summary to post) | Edits |
|---|---|---|---|
| C-369 | #369 | Basic auth (#351) shipped in 0.7.0; #349 is closed, drop it from the dependencies. Scope: users, roles, tokens and sessions enforced by the API on every route; three modes (loopback without login, built-in users, an authenticating proxy); the basic-auth login migrates on upgrade; `actor` on label operations. OIDC through the proxy mode. Plan: this file section 2.5 | drop the `blocked` label; scope line |
| C-367 | #367 | Design: a backend protocol inside the segmenter service, SAM 3 unchanged, a second backend (YOLOE-26 recommended, see the plan), `/health` reports the backend, boxes record it, the `origin=sam3` filter becomes `origin=open_vocab`. Plan: section 2.6 | scope line |
| C-368 | #368 | The 0.7.0 single-version work shipped the model-family module. Remaining: training, profiles and promote for every trainable family. The "generic single-class exporter" box: the single-class dataset export is already generic (`export_single_class`); proposed done unless the owner meant something else. Plan: section 2.4 | tick or edit the second box per Q-10 |
| C-370 | #370 | (posted by the 0.7.0 program) Design for the last box: polygon min/max, RLE through pycocotools. Plan: section 2.1 | none |
| C-436 | #436 | Design: `group_by=image` on the three search routes with an OpenSearch collapse on `image_id`, over-fetch, a Cropwright toggle. Plan: section 2.2 | add milestone v0.8.0 if missing |
| C-435 | #435 | Recommendation: close. Whole-frame search exists in the core API; curation search stays item-centric; #436 gives the per-image view. If wanted later, a separate route over the images index, not a third `target` | close as not planned, per Q-6 |
| C-84 | #84 | Scope unchanged; scheduled in wave W8.3 with half a GPU day; optional comparison with the second segmenter. Plan: section 2.8 | none |
| C-365 | #365 | Gated: built in v0.8.0 only if the clustering research plan names an encoder that passes its registered gain; otherwise moved to parked with that trigger. Design: section 2.9 | keep `blocked` until the gate |
| C-8 | #8 | Re-baseline: the first video slice is N17 (video files into curation). Live RTSP, tracking, the message bus and the SQL store stay parked here; the DeepStream groundwork the body cites was removed, so any restart begins with a fresh design and spikes. Remove machine paths from the old implementation-guide comments if any remain | add a "Re-baselined" note to the body; stays unmilestoned, `parked` |
| C-348 | #348 | Still parked; trigger in the plan section 6 | none |

### 5.4 Milestone moves

| Issue | Move |
|---|---|
| N16 (assist loop) | stays v0.8.0 (filed there by the 0.7.0 program) |
| #367, #368, #369, #370, #84, #365, #436 | stay or are set on v0.8.0 |
| #435 | closed (Q-6) or v0.8.0 if the owner keeps it |
| #347, #398, #406, #416 | per the clustering research plan (they are on v0.8.0 today) |
| #8, #346, #348, #400, #401 | stay unmilestoned and parked |
| N14, N15 leftovers | moved from v0.7.0 at its tag by the 0.7.0 program |

Verify after the steps: `gh issue list --milestone v0.8.0 --state open` lists exactly the issues of
section 1.2 plus N17, N18 and the research items; the unmilestoned open issues are the parked ones.

## 6. Parked items

| Item | Why parked | Revisit trigger |
|---|---|---|
| #348 spatial scene discovery and photogrammetry | research, not product; no experiment run; needs licence review of the 3D tools and a go/no-go report before any code (the issue's own rule); capture metadata at ingest already ships in v0.7.0 | the v0.7.0 tag exists, a research window has about 4 GPU hours on GPU 0 or 2, and the owner says go (the spatial research plan's J9, RD-2 of the 0.7.0 program); S1 is P-4..P-8 of `docs/design/research_spatial_attributes_plan.md`, run in research wave RW-5. The geo-photo timeline (L-T1) is a v0.8.0 candidate under the location policy (RD-51, RD-58) |
| #8 live video analytics (RTSP, tracking, message bus, SQL store, Helm) | a separate product from dataset curation; the DeepStream groundwork was removed; N17 covers the curation need | N17 shipped **and** a stated user need for live streams; then a fresh design doc and spikes, DeepStream as the documented NVIDIA path |
| #365 embedding swap (if its gate fails) | no candidate encoder; the machinery would be dead code | the research plan names a candidate that passes its registered gain |
| Segmenter on Triton / on resident crops (Wave 8 of `triton_pipeline_optimization_plan.md`, moved here when that plan is deleted after B5d) | speed work, needs measured evidence that the segmenter leg dominates worker time; needs its own plan | the B5 numbers or the F6 V1 measurement show the segmenter leg as a large share of worker time; F6's backend protocol makes a Triton backend possible |
| OIDC inside the API | the proxy mode covers it with a maintained proxy | a deployment that cannot run a proxy asks for it (Q-2) |
| Per-project access lists | not needed for the current homelab and small-office deployments | a multi-team deployment asks for it (Q-1); the permission table and `actor` make it additive |
| Video frames straight into the resident GPU pipeline | measure the JPEG round trip first | F7's V1 numbers show the round trip as a large share of video ingest time |

## 7. Open questions for the owner

Each has a recommendation. Needed by the wave named.

| ID | Question | Recommendation | Needed by |
|---|---|---|---|
| Q-1 | Roles per deployment only, or per project too? | per deployment in v0.8.0; per-project access lists later | W8.2 |
| Q-2 | OIDC inside the API, or only through an authenticating proxy? | proxy only | W8.2 |
| Q-3 | Keep a no-login mode on loopback-only installs? | yes (`none`, refused beyond loopback) | W8.2 |
| Q-4 | Basic-auth upgrade: keep the stored password for a new `admin` user, or generate a new one? Username `admin` or keep `cropwright`? | keep the stored password, user `admin` | W8.2 |
| Q-5 | Second segmenter: YOLOE-26 (AGPL, Ultralytics pin, fast) vs Grounding DINO + SAM 2 vs OWLv2; one backend per deployment or a choice per project? | YOLOE-26 after spike SP-2; one per deployment | W8.0 |
| Q-6 | #435 frame target: close, or build as a separate route? | close | W8.0 |
| Q-7 | Video: keep the PyAV CPU decode as a supported fallback or test-only? Include folder and S3 sources (VI-6) in v0.8.0? | supported fallback; VI-6 yes if W8.3 has room, else a follow-up | W8.0 |
| Q-8 | Embedding swap: combine with different encoders refused, or a warning? | refused (fail closed) | W8.4 |
| Q-9 | Trainable families in v0.8.0: YOLO26, YOLO11, YOLOv8 only, or also YOLO12 and YOLOv5u? | the three; add others only if FT-7 passes for them in the same window | W8.1 |
| Q-10 | #368 "generic single-class exporter": is the shipped `export_single_class` what was meant? | yes, tick the box | W8.0 |
| Q-11 | Add `pycocotools` to the API image for RLE, or refuse RLE-only annotations with an issue code? | add it (lazy import) | W8.1 |
| Q-12 | One release for v0.8.0 with one acceptance run, as for v0.7.0? | yes | W8.0 |
| Q-13 | Wheel example scale: how many COCO cars? | 3,000, capped by half a GPU day | W8.3 |
| Q-14 | F10: which S3 client library (`boto3` with `aiobotocore`, or another maintained client), and is the crop cache kept local per host? | the AWS SDK pair; cache local per host | W8.0 |

## 8. Plan lifecycle and keep-list

This file is a working paper. The PR that closes each wave moves the lasting knowledge and trims
the wave's rows here; the release PR deletes this file.

| What | Moves to |
|---|---|
| auth design, modes, threat model summary | `SECURITY.md`, `docs/ARCHITECTURE.md` (Security boundary), `docs/design/curation_api_contract.md` (auth routes and errors) |
| permission rules and the ratchet | `docs/design/engineering_standards.md` |
| segmenter backend protocol | `docker/segmenter/README.md`, `docs/ARCHITECTURE.md` |
| video ingest, split key | `docs/CURATION.md`, `docs/ARCHITECTURE.md`, docs-site ingest guide |
| training families | `docs/CURATION.md` (Export and training), docs-site training page |
| measured numbers | `docs/PERFORMANCE.md`, studies in `docs/research/` |
| user-facing changes | `CHANGELOG.md` (root and frontend), `docs/releases/v0.8.0.md` |
| new libraries | the library register in `docs/design/engineering_standards.md`, `ATTRIBUTION.md` |

The commit that adds this file also adds a row to `docs/design/README.md` (status: plan, v0.8.0)
and, after the research plan lands, turns its reference in the header into a link.

### Docs touched per wave

| Wave | ARCHITECTURE | CURATION | SECURITY | INSTALLATION | API contract + `contracts/` | PERFORMANCE | other | CHANGELOG |
|---|---|---|---|---|---|---|---|---|
| W8.1 | - | import, search, assist loop, training families | - | - | `group_by`, issue codes, pause routes, budgets, `model_family` | - | dataset-import guide, auto_label_loop_plan status | root, frontend |
| W8.2 | security boundary, auth | users and roles in review flows | rewritten | modes, upgrade, rollback, token files | auth routes, `actor`, error codes | auth overhead | docs-site security, deployment, CLI | root (breaking), frontend |
| W8.3 | segmenter backends, video worker | video ingest, split key | - | `video` profile, `OP_SEGMENTER_BACKEND` | video route, origin rename | S0, segmenter, video, training, wheel tables | segmenter README, wheel guide, ATTRIBUTION | root (breaking: origin rename), frontend |
| W8.4 | encoder per project | migration | - | - | migration routes | migration numbers | - | root |
| W8R | - | - | - | release upgrade notes | - | - | `docs/releases/v0.8.0.md`, `RELEASE_CHECKLIST.md`, `docs/design/README.md` | consolidated, then `[0.8.0]` |

### Corrections found while planning

- The 0.7.0 index's program table lists the #42 side track under v0.8.0 while its GitHub plan moves
  #42 to v0.7.0 (step GH-3). This plan treats #42 as the 0.7.0 side track carried over only if
  unrun.
- #369's body lists #349 as a dependency; #349 is closed and #351 (basic auth) shipped in 0.7.0.
- #8's body cites a hardened DeepStream image that was removed from the repo; its implementation
  guide comments carry machine paths and a GPU assignment that the host rules replace. C-8 covers it.
- `triton_pipeline_optimization_plan.md` puts its Waves 8 and 10 in v0.8.0. Wave 10 is #365 (F9);
  Wave 8 is speed work, parked here with a trigger (section 6).
- The item filter origin `sam3` is not a model name in meaning: it selects items found by the
  open-vocabulary pass (field `open_vocab_set`). F6 renames it.
