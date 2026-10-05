# Curation and Active Learning

> **Status: experimental for v0.4.1.** This is a working, tested subsystem,
> but it is new and its API can still change between releases. It ships
> opt-in, behind Docker Compose profiles, and is off by default.

## What this is

The curation subsystem is a domain-agnostic backend for building and
maintaining an image-labeling dataset with active learning. You ingest or
import images, detect items in them, optionally find regions inside each
item, cluster and browse the results, label them (by hand or with an
OpenAI-compatible vision-language model), review queues, export a dataset,
and train a detector.

Nothing in it assumes what an "item" or a "region" is. A pallet, a defect on
a part, a license plate, or the wheel on a car is configuration, not code.
Everything domain-specific lives in data you own: a class registry, a region
profile, a prompt pack and a VLM endpoint, all stored per project.

Companion documents:

- [`design/curation_api_contract.md`](design/curation_api_contract.md): the
  route-by-route wire contract.
- [`design/curation_design_rationale.md`](design/curation_design_rationale.md):
  why it is built this way.
- [`../contracts/openapi/curation.json`](../contracts/openapi/curation.json):
  the generated OpenAPI document, the source of truth for routes and schemas.

All curation routes are mounted under one prefix (`OP_API_PREFIX`, default
`/curation`) and have no `/v1` twin. Every route that touches data is scoped
to a project: `/curation/projects/{project}/...`. Only the project list and
lifecycle routes, the VLM endpoint registry, `GET /curation/health` and
`GET /curation/events` are deployment-wide.

The examples below use two shell variables:

```bash
BASE=http://localhost:4603/curation          # deployment-wide routes
API=$BASE/projects/<slug>                    # one project's routes
```

## Concepts

| Term | Meaning |
|---|---|
| Project | An isolated dataset: its own OpenSearch indexes, class registry, exports, uploads, settings and config. Nothing is shared between projects unless you clone or combine. |
| Image | One ingested source file. |
| Item (crop) | One detected object on an image. A route path calls it a `crop`; the wire object is an item. |
| Region | An optional sub-area inside an item (a wheel on a car, a tag on a pallet). An item holds a list of region boxes, `region_boxes`; one box is a list of one. |
| Class registry | The project's list of classes. Class identity is the class name; ids are local to the project. |
| Region profile | How regions are found for a project: detector, segmenter prompt, parent classes, box cap, whether text is read. |
| Prompt pack | The VLM prompts and vocabulary the labeler uses. |
| VLM endpoint | An OpenAI-compatible server the labeler calls. Registered once per deployment, activated per project. |
| Config store | Where profiles, packs and settings live. Every save is an immutable revision; activation is a separate step. |

## The workflow

1. Create a project and define its classes.
2. Activate a region profile and a prompt pack (optional if you only label
   whole items) and pick a VLM endpoint (optional).
3. Ingest images, or import an already-labeled dataset.
4. Let the workers run: item detection, region detection, VLM verification,
   embeddings, clustering.
5. Review and label in a frontend (Cropwright is the first consumer) or
   through the API.
6. Freeze a test holdout, export a YOLO dataset, run preflight, train.
7. Promote the model, compare it in the bake-off, and feed disagreements back
   into review.

## Use your own domain

This is the whole path for a new domain. The car-to-wheel example in
[`examples/`](../examples/) is used throughout; replace the names with yours.

**1. Create a project.**

```bash
curl -s -X POST $BASE/projects -H 'content-type: application/json' \
  -d '{"slug": "wheels", "display_name": "Wheels"}'
API=$BASE/projects/wheels
```

A slug is 2 to 32 characters, lowercase letters, digits and single hyphens,
starting with a letter. Pass `clone_settings_from` (and optionally
`clone_axes`) to start from another project's settings.

**2. Define classes.** Items are classed by name. Add only the classes you
care about.

```bash
for name in car wheel; do
  curl -s -X POST $API/classes -H 'content-type: application/json' \
    -d "{\"name\": \"$name\"}"
done
curl -s $API/classes | jq '.classes | length'
```

A duplicate name is a 409. To start from the ingest detector's own label
space (all 80 COCO classes for the stock detector), seed it by name:
`POST /curation/projects/{project}/classes/seed_from_detector` (dry run by
default; send `{"dry_run": false}` to write). Labels with spaces become slugs
(`traffic light` -> `traffic_light`), existing names are skipped, and ids are
appended, never aligned to the detector's. `GET /curation/projects/{project}/ingest/config`
now returns a `detector` block (model, label list with raw name and slug, env
class-id filter if set). `scripts/curation/seed_class_registry.py --model
<detector.onnx>` remains for a model that assigns registry classes by id.

**3. Write a region profile.** Start from
[`examples/region_profiles/vehicle_wheel.json`](../examples/region_profiles/vehicle_wheel.json).
Its key fields:

| Field | Value in the example |
|---|---|
| `parent_classes` | `["car"]`: the region stage runs only on items of this class name |
| `segmenter_text_prompt` | `wheel`: what the segmenter is asked to find |
| `detector_model` | empty: no detector leg until you train and promote one |
| `max_regions_per_item` | `4`: the box cap per item |
| `text_reader` | `none`: a text-free region |
| `region_class_name` | `wheel`: the class name used on export |

Save and activate it. `jq` drops the `_comment` and `name` keys, which are
not part of the body.

```bash
BODY=$(jq '{name: "wheel_example", body: del(._comment, .name)}' \
  examples/region_profiles/vehicle_wheel.json)
curl -s -X POST $API/region_profiles -H 'content-type: application/json' -d "$BODY"
curl -s -X POST $API/region_profiles/wheel_example/activate \
  -H 'content-type: application/json' -d '{"expected_active": null}'
```

Use `POST /curation/projects/{project}/region_profiles/validate` to check a
draft before saving it.

**4. Write a prompt pack.** Start from
[`examples/prompt_packs/vehicle_wheel.json`](../examples/prompt_packs/vehicle_wheel.json).
Its prompts ask the VLM to classify a car crop and to verify each numbered
candidate box. The reply keys it must keep are described by
`GET /curation/projects/{project}/prompt_packs/schema`.

```bash
BODY=$(jq '{name: "wheel_example", body: del(._comment, .name)}' \
  examples/prompt_packs/vehicle_wheel.json)
curl -s -X POST $API/prompt_packs -H 'content-type: application/json' -d "$BODY"
curl -s -X POST $API/prompt_packs/wheel_example/activate \
  -H 'content-type: application/json' -d '{"expected_active": null}'
```

**5. Pick a VLM endpoint.** Register one, probe it, activate it for the
project. See [Choosing a VLM](#choosing-a-vlm). Without an active endpoint
the worker keeps the segmenter's boxes unverified.

**6. Ingest or import.** Put images where the API can read them (see
[Mounting your image source](#mounting-your-image-source)), then:

```bash
curl -s -X POST $API/ingest/batch -H 'content-type: application/json' \
  -d '{"items": [{"path": "/data/source/coco_car/images/000000000001.jpg", "source": "coco_car"}]}'
curl -s $API/ingest/region_drain | jq     # repeat until drained
```

An already-labeled YOLO or COCO dataset goes through
[dataset import](#dataset-import) instead.

**7. Review.** Open the project in a frontend, or list items with
`GET /curation/projects/{project}/review/tabs` and
`GET /curation/projects/{project}/review/{tab}`. Confirm or edit boxes with
the [region routes](#multi-box-regions).

**8. Export and train.**

```bash
curl -s -X POST $API/export/single_class -H 'content-type: application/json' \
  -d '{"profile_name": "wheels", "box_source": "region", "region_class_name": "wheel",
       "image_mode": "item_crop", "class_ids": []}'
curl -s -X POST $API/train/preflight -H 'content-type: application/json' -d '{}'
```

Then start a run with `POST /curation/projects/{project}/train/start` (see
[Export and training](#export-and-training)).

### The worked example: wheels on cars

COCO has no wheel labels. The output of this example is machine-proposed
wheel boxes that a person reviews, which is the point. Everything it needs is
in the repository:

- [`examples/region_profiles/vehicle_wheel.json`](../examples/region_profiles/vehicle_wheel.json)
- [`examples/prompt_packs/vehicle_wheel.json`](../examples/prompt_packs/vehicle_wheel.json)
- [`examples/bakeoff/vehicle_wheel/profile.json`](../examples/bakeoff/vehicle_wheel/profile.json)
  (to compare wheel detectors you train later; set its `triton_model`)
- [`scripts/examples/wheel_example_live.py`](../scripts/examples/wheel_example_live.py)

```bash
make sample-coco-cars      # 60 CC BY car images into data/samples/coco_car
.venv/bin/python scripts/examples/wheel_example_live.py \
    --api http://localhost:4603 --project wheels \
    --container-dir /data/source/coco_car/images
```

The script creates the project and the `car` and `wheel` classes, saves and
activates the example profile and pack, ingests the images, waits for the
region queue to drain, and exports the wheel boxes cropped to their car. It
needs the live stack: Triton with the primary detector loaded, the segmenter,
the `curation` workers, and `data/samples/coco_car` mounted for the API
container. Without an active VLM endpoint the boxes stay unverified.

Items are chosen for the region stage by class name. The profile's
`parent_classes` (`car`) matches an item's `class_name` or the detector's own
label (`proposal_name`), never a class index. The same walk runs offline in CI
with fakes at the OpenSearch, Triton, segmenter and VLM boundaries:
`tests/integration/test_wheel_example_e2e.py`.

> Screenshot pending: Cropwright (the wheel project's review grid with
> numbered wheel boxes on a car).

### Cost control for the region stage

A segmenter call that finds nothing costs as much as one that does, so the
stage has three cheap checks before it:

1. `parent_classes` (free): only items of the named classes get the stage.
2. The batched vision-model visibility check, when a VLM is active: "is a
   region visible at all?" decides before the segmenter runs.
3. An optional learned hit rate per item class, off by default. With
   `gate_hit_rate` on in the region profile, a class whose last
   `gate_hit_window` segmenter calls had `gate_hit_miss_threshold` misses is only
   sampled at `gate_hit_sample_floor` so it can recover. A skipped item is not
   lost: it is written as `no_region_box` with `region_gate_skip` set (wire field
   of the same name, `tier3_hit_rate`) and `gate:tier3_hit_rate` in its detector
   chain. An item whose class a human owns or validated is never skipped. The
   windows live in the worker's memory, one per project.

Re-run the skipped items with `POST /curation/projects/{project}/reprocess`,
scope `region`, filter `{"region_status": ["no_region_box"],
"region_gate_skipped": true}`; the `rerun_skipped` field of
`GET /curation/projects/{project}/region_stage` is that request (dry run).

`POST /curation/projects/{project}/region_stage/pause` stops the region stage
for this project only: the worker fetches nothing for it and hands back items
it holds before their segmenter call, so they all stay `pending_detection`;
items already past the segmenter finish, and other stages keep running.
`POST /curation/projects/{project}/region_stage/resume` undoes it. The state
(`paused`, `paused_since`, `pipeline_paused`, `counts` of pending and
gate-skipped items) comes from `GET /curation/projects/{project}/region_stage`.
Like the other region routes these answer 409 until a region profile is active.

Metrics (worker): `op_region_segmenter_calls_total` and
`op_region_segmenter_seconds_total` by `profile`, `class_name` and `outcome`
(`hit` or `miss`; the `miss` seconds are the time spent on calls that found
nothing), and `op_segmenter_gate_decisions_total` with `scope="crop"`.

## Projects

A project owns its indexes (`<OP_PROJECT_INDEX_PREFIX><slug>__<role>`), its
files under `OP_PROJECTS_DATA_ROOT/<slug>/`, its uploads under
`OP_STATE_DIR/projects/<slug>/`, its class registry, its settings and its
config. A `default` project is created on first start; it is an ordinary
project that can be archived but never deleted.

| Action | Route |
|---|---|
| List, create | `GET /curation/projects`, `POST /curation/projects` |
| Read, rename | `GET /curation/projects/{project}`, `PATCH /curation/projects/{project}` |
| Archive, restore | `POST /curation/projects/{project}/archive`, `POST /curation/projects/{project}/unarchive` |
| Delete | `DELETE /curation/projects/{project}` |
| Pause the workers for one project | `POST /curation/projects/{project}/pause`, `POST /curation/projects/{project}/resume` |
| Pause only the region stage | `GET /curation/projects/{project}/region_stage`, `POST /curation/projects/{project}/region_stage/pause`, `POST /curation/projects/{project}/region_stage/resume` |

Rules:

- `PATCH`, archive and unarchive take `expected_revision`; a stale value is a
  409 `revision_conflict`.
- Archive moves `active` to `archived` and is refused while the project has
  running jobs (409 `project_busy`) and for the last active project. An
  archived project is read-only.
- Delete is a dry run with `?dry_run=true` (writes nothing and returns what
  would block it). A real delete needs `?confirm=<slug>` (it must equal the slug), drains the
  detection worker, answers 202 with a `deleting` record, then removes the
  indexes and directories in the background. The slug is retired afterwards.
- Reserved slugs: `all`, `combine`, `global`, `health`, `new`, `none`,
  `projects`, `settings`, `vlm`.
- A project that is `building` (a combine target filling up) or `failed` is
  listed with a status; deleting a failed project is a complete cleanup.

`POST /curation/projects/{project}/clone_settings` copies settings from
another project into this one. The body is `{"from": "<slug>", "axes": [...],
"expected_revision": N}`. Cloneable axes: `settings_defaults`, `classes`,
`activations`, `keymap`, `prompt_packs`, `vlm_activation`. A VLM external
acknowledgement is never copied.

## Settings and the config store

Per-project settings are one shared document, not per-browser state.

- `GET /curation/projects/{project}/settings` and
  `PUT /curation/projects/{project}/settings` read and merge
  `defaults`, an open map keyed by axis id. Settable axes: `cluster`, `sort`,
  `prompt_pack`, `detection_profile`, `vlm`.
- `GET /curation/projects/{project}/methods` lists every selectable
  strategy per axis, with the effective default flagged. A settings default
  is applied by the server wherever a request omits that axis.
- `GET /curation/projects/{project}/config/vocabulary` serves the labels,
  limits, model-choice roles and VLM block a frontend needs to render the
  config screens without its own tables.

Prompt packs, region profiles and VLM endpoints share one lifecycle:

- A save creates an immutable revision. Revision numbers are never reused.
- Activation is a separate step and takes an `expected_active` guard taken
  from the active record; a stale guard is a 409.
- Validation runs on every save and again, stricter, on activation. `force`
  bypasses only the warnings that are bypassable.
- Workers and API processes notice an activation within `OP_CONFIG_POLL_S`
  seconds and apply it without a restart. The detection worker swaps at a
  quiesce point (its queues drained).
- Rollback re-activates the previous activation.

## Class registry and class identity

Each project's registry is a JSON file at
`$OP_PROJECTS_DATA_ROOT/<slug>/class_registry.json`, created on the first
`POST /curation/projects/{project}/classes`. A worked example is
[`data/class_registry.example.json`](../data/class_registry.example.json).

```json
{
  "version": 1,
  "updated_at": "2026-01-01T00:00:00+00:00",
  "classes": [
    {"id": 0, "name": "cardboard_box", "group": "packaging", "sample_count": 0,
     "validated_count": 0, "added_at": "2026-01-01T00:00:00+00:00",
     "deprecated": false, "notes": "", "merged_into": null, "hotkey_letter": "b"}
  ]
}
```

**Class identity is the name.** A class id is a dense index local to one
project and one export. Anything that crosses a boundary pairs classes by
name, never by index: dataset import mapping, combine, bake-off class
mapping, model promotion (`labels.txt`) and the dense remap an export writes.
`tests/integration/test_class_identity_e2e.py` and
`tests/integration/test_class_identity_combine.py` walk import, export,
train, promote and predict and assert the `(class_id, class_name)` pairing at
every hop.

**By-name resolution has one rule** (`resolve_class_by_name`). Names compare
after normalization (case, spaces and hyphens fold to `_`). An *active* class
always wins over a deprecated one with the same name; a name that only a
deprecated class carries is reported as deprecated and never assigned to.
Ties break on the exact spelling, then the lowest id. Creating an active class
with a deprecated class's name is allowed; two active classes can never share
a name, and restoring a deprecated class is refused while an active class holds
its name. Dataset-import mapping, `ensure_class_by_name`, adopt-existing and
detector seeding all go through it. Stored-item `class_name` filters on list
routes match the name saved on each item, so use `class_id` to select one class
exactly.

Class routes: `GET /curation/projects/{project}/classes`,
`POST /curation/projects/{project}/classes`,
`PUT /curation/projects/{project}/classes/{class_id}`,
`POST /curation/projects/{project}/classes/merge` (supports `?dry_run=true`),
`POST /curation/projects/{project}/classes/{class_id}/deprecate` and
`POST /curation/projects/{project}/classes/{class_id}/restore`. A merged class
cannot be restored. Reserved hotkeys are rejected the same way in the API and
the UI.

## The lock rule

Automated writers never overwrite what a person or a trusted import decided.
An item is locked when any of these hold:

- a human set or confirmed its class;
- its class came from a dataset import with `label_trust: validated`;
- it is frozen in the test holdout;
- any of its region boxes was created, moved, verdicted or transcribed by a
  human, or came from an import and is not a mere suggestion;
- its region set was validated by a human or an import.

A `label_trust: suggestion` import writes unvalidated classes and `proposed`
boxes, which the machine pipeline may still replace. The VLM labeler,
re-ingest, the region worker, reprocess, clustering writers and the
false-positive auto-assign all skip locked items and report them as
`locked_skipped`. Imports, undo and reconcile use the same rule, so an undo
never deletes something a person touched after the import.

## Region profiles

A region profile describes how regions are found inside an item. It is data,
stored per project, and has no built-in default: with none active, region
detection is off and the worker idles.

Routes (all under `/curation/projects/{project}`):

| Purpose | Route |
|---|---|
| List, create | `GET /region_profiles`, `POST /region_profiles` |
| Field help | `GET /region_profiles/schema` |
| Validate a draft | `POST /region_profiles/validate`, `POST /region_profiles/validate_segmenter_prompt` |
| Read, save, delete | `GET /region_profiles/{name}`, `PUT /region_profiles/{name}`, `DELETE /region_profiles/{name}` |
| Revisions | `GET /region_profiles/{name}/revisions`, `GET /region_profiles/{name}/revisions/{revision}` |
| Copy | `POST /region_profiles/{name}/clone` |
| Activate | `POST /region_profiles/{name}/activate`, `POST /region_profiles/deactivate` |
| Active profile, impact, rollback | `GET /region_profiles/active`, `GET /region_profiles/active/impact`, `POST /region_profiles/active/rollback` |
| Test on a crop | `POST /region_profiles/test` |

Notes:

- Sources: `stored` (edited here), `env` (the boot default from
  `OP_REGION_PROFILE_PATH` / `OP_REGION_PROFILE` / `OP_REGION_DETECTION_*`,
  read-only), `registered` (from deployment code) and `template` (the files in
  [`examples/region_profiles/`](../examples/region_profiles/), listed with
  `?include_templates=true`). A template cannot be activated; clone it first.
- The activation response carries an impact summary: how many items were
  processed under another profile or an older revision, how many are
  validated, how many are pending, and a `suggested_reprocess` body to
  re-run them. `GET /region_profiles/active/impact` returns the same summary
  without activating.
- A profile changes only items processed after activation. Re-run older
  items with [`POST /reprocess`](#reprocess).
- `parent_classes` restricts the stage to items of those class names. Empty means every
  item gets the stage (and the segmenter call), so set it; the `license_plate` example
  names `car`, `truck`, `bus` and `motorcycle`.
  `max_regions_per_item` caps the boxes kept per item.
- `text_reader: "none"` makes a text-free profile: no OCR, no text fields.
  `text_hint_enabled` adds an optional OCR text hint to locate text.
- A `detector_model` must be a model this project owns or that was shared to
  it (`PUT /curation/projects/{project}/models/{model_name}/sharing`);
  validation reports `detector_model_not_shared` and
  `detector_model_classes_unmapped` otherwise.
- An empty `detector_model` skips the detector leg and never calls Triton.

## Multi-box regions

Every item holds its regions as one list, `region_boxes`. An item with one
region has a list of one. There are no single-box scalar fields on the item.

Per box: `box_id` (stable, never reused after a delete), `bbox_norm`, `state`,
`score`, `detector`, `source`, `confidence`, `rejection_reason`, `text` fields
when the profile reads text, and `cluster_id`, `cluster_subid`,
`cluster_distance`. Each box also carries `bbox_in_parent` and a
`thumbnail_url`. Each accepted or false-positive box has one embedding in
`region_box_embeddings`.

Box states are `proposed`, `accepted`, `rejected` and `false_positive`. The
item status is derived from its boxes with a fixed precedence: any accepted
box makes the item `detected`, then `false_positive`, then `proposed`, then
`rejected`. The item-level fields `region_count`, `region_rejected_count`,
`region_max_score`, `region_set_complete`, `region_revision` and
`region_validated` summarize the list.

How regions get boxes:

1. The region stage selects candidates from the detector and segmenter legs
   (score floor, NMS, then the `max_regions_per_item` cap).
2. The VLM sees the crop with each candidate numbered and returns one verdict
   per box plus whether the region is visible at all.
3. Verified boxes become `accepted`, the rest `rejected`. A VLM-reported
   incomplete set is recorded in `region_set_complete`.

Edit routes (all under `/curation/projects/{project}`):

| Purpose | Route |
|---|---|
| Replace the whole list | `PUT /crops/{crop_id}/regions` |
| Edit one box | `PATCH /crops/{crop_id}/regions/{box_id}` |
| Item-level metadata (status, rejection reason) | `PATCH /crops/{crop_id}/region_meta` |
| Several items at once | `PUT /crops/batch_regions`, `POST /regions/batch_status`, `POST /regions/batch_box_state` |
| Undo | `POST /crops/{crop_id}/region/undo`, `POST /crops/region/undo_batch` |
| Box thumbnail | `GET /crops/{crop_id}/region_thumbnail` |

`PUT /crops/{crop_id}/regions` takes the full list in display order: an
element with only `box_id` leaves that box alone, `box_id` plus `bbox_norm`
moves it (a moved box is human geometry), `box_id: null` adds a box (default
state `accepted`), and a stored box that is omitted is deleted.
`frame: "parent"` sends boxes in the source image frame. A stale
`expected_region_revision` is a 409 `region_conflict` with the current
revision. The list is capped by `OP_REGION_MAX_BOXES_PER_WRITE` per write.

Browse and cluster at box level:

- `GET /regions`, `GET /regions/statuses`, `GET /regions/vocabulary`,
  `GET /regions/training_candidates` and
  `GET /regions/suspected_false_positives` return rows. A box-selecting
  request returns one row per matching box, with `region_box_id`; `total`
  counts items and `total_rows` counts rows. All box filters apply to the
  same box.
- `POST /regions/cluster` partitions accepted boxes (poll
  `GET /regions/cluster/status`); `GET /regions/clusters` lists the cards and
  `POST /regions/clusters/refine/{cluster_id}` splits one.
- `POST /regions/fp_centroids/build` builds false-positive centroids from
  boxes marked `false_positive`; matching boxes are then flipped
  automatically and the item status is re-derived (a sibling accepted box
  keeps the item `detected`).
- Export strata and region clustering counts use boxes: `n_boxes`,
  `n_boxes_changed`, `n_items_written`.

## Prompt packs

A prompt pack holds the VLM prompt templates and two vocabulary tables
(`class_descriptions`, `synonyms`) and an optional `proposal_denylist` of
glob patterns (`blurry_*`, `*_scene`) whose matching new-class proposals are
dropped. It is separate from the region profile:
a profile says where to look, a pack says what to ask. Either can be used
without the other.

Routes (all under `/curation/projects/{project}`): `GET /prompt_packs`,
`POST /prompt_packs`, `GET /prompt_packs/schema`,
`POST /prompt_packs/validate`, `GET /prompt_packs/{name}`,
`PUT /prompt_packs/{name}`, `DELETE /prompt_packs/{name}`,
`GET /prompt_packs/{name}/revisions`,
`GET /prompt_packs/{name}/revisions/{revision}`,
`POST /prompt_packs/{name}/clone`, `POST /prompt_packs/{name}/activate`,
`GET /prompt_packs/active`, `POST /prompt_packs/active/rollback` and
`POST /prompt_packs/test`.

The built-in `generic_region_v1` pack is text-free. A pack whose region
prompts do not return a per-box list is a warning on save and an error on
activation when the active profile has `max_regions_per_item` above one.
`POST /curation/projects/{project}/pipeline/auto_label/start` takes
`?prompt_pack=` to use another pack for one run.

## Test on a crop

Test a draft before activating it. Both routes run on crops already in the
project, are read-only (nothing is indexed, updated or queued), and are
project scoped.

- `POST /curation/projects/{project}/prompt_packs/test` runs the production
  labeler on up to 64 stored crops and returns the exact request and reply,
  whether the reply parsed, and per crop the parsed answer and the item the
  worker would write.
- `POST /curation/projects/{project}/region_profiles/test` runs the detector
  and segmenter legs on one crop and returns every candidate with whether it
  was selected and why it was dropped (floor, NMS or cap), plus segmenter
  mask polygons. An optional `verify` leg sends the selected boxes to the VLM.
  The OCR text-hint and text-reading legs are not previewed.

Guards: at most 4 concurrent segmenter calls and 2 concurrent VLM runs (429
`test_busy`), a 60 second bound (504 `test_timeout`), and a crop cap (422
`too_many_crops`).

## Choosing a VLM

The VLM is a registry of endpoints, and each project runs one of them. The
`env` built-in (`OP_VLM_URL`, `OP_VLM_MODEL`, `OP_VLM_API_KEY`) is always
listed first and is read-only.

Ways to get a server behind an endpoint:

1. **The shipped `vlm` service** (`docker compose --profile vlm up -d`),
   with `OP_VLM_URL=http://vlm:8000/v1` and the served model name in
   `OP_VLM_MODEL`.
2. **A server on this host outside compose**:
   `OP_VLM_URL=http://host.docker.internal:<port>/v1`. `yolo-api` and the VLM
   worker carry `extra_hosts: host.docker.internal:host-gateway` for this.
3. **A server elsewhere on the network**: `OP_VLM_URL=http://<host>:<port>/v1`.

Client and server image caps are one contract: keep
`OP_VLM_MAX_IMAGES_PER_CALL` (default 8) numerically equal to the server's
per-prompt image limit (`VLM_LIMIT_MM_IMAGES` for the in-compose service).
A request above the smaller of the two gets a 400.

Registry workflow:

- **Register**: `POST /curation/vlm/endpoints` with a name and a body
  (`base_url`, `model`, optional `api_key_ref`, image cap, `json_mode`). A key
  is never sent through the API: write it on the host with
  `./openprocessor vlm key set <slug>` and reference it as `secret:<slug>`.
  `GET /curation/vlm/endpoints/schema` describes the fields.
- **Save**: `PUT /curation/vlm/endpoints/{name}` creates a new revision;
  `GET /curation/vlm/endpoints/{name}/revisions` lists them. Saving never
  changes what a project runs.
- **Probe**: `POST /curation/vlm/endpoints/{name}/probe` (or
  `POST /curation/vlm/endpoints/validate` with a draft). The probe sends
  synthetic images only and records the served model root, context length,
  image cap and JSON-mode support for that exact revision.
- **Activate** for a project:
  `POST /curation/projects/{project}/vlm/endpoints/{name}/activate`, or
  `defaults.vlm` in `PUT /curation/projects/{project}/settings`. Read the
  active one with `GET /curation/projects/{project}/vlm/endpoints/active`;
  roll back with `POST /curation/projects/{project}/vlm/endpoints/active/rollback`
  or turn it off with `POST /curation/projects/{project}/vlm/endpoints/deactivate`.
- **Per run**: the VLM and pipeline routes take `?vlm=<name>` (and
  `acknowledge_external`) to use another endpoint for that run only.

An endpoint that would send crops outside this deployment needs
`allow_external` on the endpoint and an acknowledgement when it is activated,
recorded per `name@revision`. `OP_VLM_EXTERNAL_POLICY=deny` refuses such
endpoints outright. The URL policy also refuses this stack's own services and
link-local or metadata addresses. Every labeler re-checks its endpoint at
most every 30 seconds and refuses (fails closed) if the host now resolves to a
denied address. See [`../SECURITY.md`](../SECURITY.md) for the residual risk.

Activation runs a pairing check: context size per call, the server's image
cap, multi-box and text-reading verification, and JSON mode. Items record
`vlm_endpoint` and `vlm_model` for each answer.

**Local models.** `examples/vlm/catalog.tsv` lists the models the in-compose
vLLM can serve. `GET /curation/vlm/catalog` and `GET /curation/vlm/local`
show them with a VRAM fit check. The host CLI changes the served model:

```bash
./openprocessor vlm list
./openprocessor vlm status
./openprocessor vlm use <id> [--force] [--yes]
./openprocessor vlm apply
./openprocessor vlm probe
./openprocessor vlm key set <slug>
```

`use` checks fit, rewrites `.env` (restoring it on failure), waits for the
new model, probes it and unpauses. The API records the desired local model
(`POST /curation/vlm/local/select`) but never restarts vLLM.

## Keymaps

Each project has a keymap for its labeling frontend. `GET /curation/projects/{project}/keymap`
returns the grammar, contexts, actions with defaults, overrides, reserved
hotkeys and an `etag`. `PUT /curation/projects/{project}/keymap` saves
overrides (with `expected_revision`),
`POST /curation/projects/{project}/keymap/validate` checks a draft, and
`POST /curation/projects/{project}/keymap/reset` clears overrides. A key that
collides with a class hotkey is reported with the class; pass
`unbind_conflicting_class_hotkeys` to unbind it.

## Ingest

| Purpose | Route |
|---|---|
| One image by path | `POST /curation/projects/{project}/ingest/image` |
| Many images by path | `POST /curation/projects/{project}/ingest/batch` |
| Bytes upload | `POST /curation/projects/{project}/ingest/upload` |
| Which paths are already known | `POST /curation/projects/{project}/ingest/path_lookup` |
| Limits and source roots | `GET /curation/projects/{project}/ingest/config` |
| Read or replace the ingest policy | `GET`, `PUT /curation/projects/{project}/ingest/policy` |
| What a policy would embed | `POST /curation/projects/{project}/ingest/policy/preview` |
| Status, region queue | `GET /curation/projects/{project}/ingest/status`, `GET /curation/projects/{project}/ingest/region_drain` |

Ingest does duplicate detection, a quality gate, crop-cache population, PE
embedding and bulk indexing. With a region profile active, every newly
created item is seeded `pending_detection`, which is the only way an item
enters the region worker's queue. An existing region status is never
overwritten on re-ingest. Items that exist before you activate a profile are
picked up with [`POST /reprocess`](#reprocess).

### Ingest policy

Each project has an ingest policy with two independent parts. An absent
policy changes nothing: every detection is stored and embedded.

- `detect`: a filter on the detector output (`classes` allow-list,
  `exclude_classes`, `min_confidence`, `min_box_area_frac`, `max_per_image`).
  Filtered detections are not stored; ingest results and the batch summary
  report them as `n_filtered`. A `reprocess` with scope `detect` applies the
  same filter, so it also removes stored, unlocked items the filter now
  excludes.
- `embedding.mode`: `all` (default), `selected` (embed only detections
  matching `classes`, `min_confidence`, `min_box_area_frac`, `max_per_image`;
  at least one criterion is required) or `lazy` (embed none at ingest). A
  detection that is stored without a vector gets `embedding_state`
  `not_selected` or `deferred`, keeps its class cluster when it has a class,
  and is not placed in a residual cluster. An item with a human or imported
  label always gets a vector.

- `detect.class_resolution`: `proposal` (default) leaves every detection an
  unlabeled proposal; `by_name` gives a detection the registry class whose name
  equals the detector's own label (`traffic light` becomes `traffic_light`,
  active classes only, never the region class). The class is written like a
  classifier's label (`class_source` `<detector>_model`, not validated), so the
  VLM stage skips it, the item sits in its class cluster and the detector's own
  label stays on the item as `proposal_name`.
- `detector`: this project's own ingest detector (`model`, optional `version`,
  `input_size`, `labels_path`), replacing the deployment's primary model for
  this project only. `PUT` refuses it (422) unless the model is loaded on
  Triton and serves the end2end outputs (`num_dets`, `det_boxes`, `det_scores`,
  `det_classes`); its class ids are never read as registry ids. Ingest, the
  `detector` block of `GET .../ingest/config`, `POST .../classes/seed_from_detector`
  and a dataset import's propose mode all use it, and unsharing a model a
  project runs this way is refused like any other use. Deleting the model is
  not blocked: change the project's policy first.

Class names match by name (case, spaces and hyphens are normalized, so
`traffic light` and `traffic_light` are one name) against an item's class name
or the detector's own label, never by model class id. Names the project does
not know yet are accepted and returned as `unknown_names`. `PUT` takes the
`expected_revision` you read (409 when stale) and affects future ingests and
embed runs only, never stored data. `POST .../ingest/policy/preview` counts
what a candidate policy would embed over the items already stored, with an
estimated vector size, and writes nothing. The policy is cloned with a
project's settings and is not merged by a combine.

### Items without an embedding

Every item carries `embedding_state`: `embedded` (it has a vector), `failed`
(the encoder raised at ingest; the item is still stored), `deferred` (a vector
was dropped because the target project could not use it, as in a combine) or
`not_selected` (the ingest policy's `selected` mode skipped it). `null` means the item was
written before the field existed. Whether an item has a vector is always the
`exists` test on its embedding; the state only says why not.

Items without a vector are stored and browsable, but clustering, kNN search,
the outlier and diverse orderings, the review queue's unclassed view and the VLM stage
all skip them, so the API says so instead of returning a silent gap:

- Ingest results and the batch summary carry `n_embedded` and `n_not_embedded`
  (items stored without a vector, which includes policy skips) and `n_embed_failed`
  (only the encoder failures: warn on this one); `ingest_walker.py` warns when any
  item was stored without a vector.
- `GET /curation/projects/{project}/search/text` returns `unembedded_in_scope`.
- `GET /curation/projects/{project}/crops` with `order=outliers` or
  `order=diverse` (and `core_first`) returns `n_unembedded` next to `n_pool`, plus
  `suggested_reprocess`: the dry-run embed request to POST for the items it could not rank.
  An empty `GET /review/tabs` queue offers the same request in `empty_state`.
- `GET /curation/projects/{project}/stats/dataset` returns an `embedding` block
  (`embedded`, `not_embedded`, `by_state`); the project counts return
  `items_embedded`; the auto-label `baseline`/`after` snapshots return
  `unembedded`.
- An empty review `all` queue explains when the cause is unembedded items.

The VLM stage works on embedded items only, so one setting (whether an item
is embedded) bounds both the embedding and the VLM work.

### Embedding use cases

Normal flow: every detection is embedded at ingest. Filtering, selecting,
searching and clustering are views over the embedded items. The cases below
are the ways an item needs a vector after ingest, and what happens today.

1. **A new object or box.** An item is created only by ingest (detector) or a
   dataset import; both embed it through the same code. A region box that the
   region worker writes (SAM 3 or a region profile) gets its box vector in the
   same pass. A box a person draws is embedded in the edit request itself
   (see 2).
2. **A moved or resized box.** An item's own box is never edited. A region box
   a person moves or deletes has its stored vector pruned at once (a vector
   records the geometry it was computed from, so a moved box counts as having
   none). Every human box-edit route (`PUT .../crops/{crop_id}/regions`,
   `PUT .../crops/batch_regions`, `PATCH .../crops/{crop_id}/regions/{box_id}`,
   `POST .../regions/batch_box_state`, `PATCH .../crops/{crop_id}/region_meta`
   and `POST .../regions/batch_status`) prunes what the edit invalidated and
   then embeds the accepted and false-positive boxes that have no valid vector,
   and returns `vector_refresh: {embedded, pending}`. A box it could not
   embed (encoder down, image unreadable) stays `pending`: the edit still
   succeeds and an `embed` run with `only_missing` picks the box up.
3. **An embedding failed at ingest.** The item is stored with
   `embedding_state: failed` and counted in `n_not_embedded` and `n_embed_failed`. Retry with
   `POST /curation/projects/{project}/reprocess` and scope `embed` on the
   item or its image; the item becomes `embedded`.
4. **An ingest policy skipped it** (`selected`, `lazy`, per-image caps). It is
   stored with `embedding_state` `not_selected` or `deferred`. Embed them with
   the `embed` scope and `only_missing: true`, selecting by `filter` (for
   example `embedding_state: [not_selected]` and `class_names`), by ids, with
   a `limit`, or all at once.
5. **An imported dataset** (YOLO, COCO or your own export). Import embeds each
   item through the ingest path, so imported items are `embedded` (or
   `failed` and retried as in 3). A project import that excludes vectors, and
   a combine that drops a vector the target cannot use, leave items
   without one (`deferred` for a dropped vector); embed them with the `embed`
   scope.
6. **The embedding model changed.** A full re-embed, not embed-missing: run the
   `embed` scope over every image without `only_missing`. It rewrites every crop, frame and box
   vector and keeps labels and locks untouched. The index mapping fixes the
   vector dimension, so a model with a different dimension needs a new
   project (re-ingest or combine), not an in-place re-embed.

For bulk work from a shell:

- `scripts/curation/ingest_walker.py`: walk a directory with a resumable
  progress file.
- `scripts/curation/ingest_upload.py`: read files locally and upload the
  bytes when the API cannot mount your storage (resume is server-side content
  dedup).
- `scripts/curation/import_labeled_dataset.py`: a thin client of dataset
  import.

Fetch public sample data with `make sample-coco`, `make sample-coco-cars` or
`make sample-plates`; `make sample-clean` removes it. On an installed stack,
`./openprocessor sample coco` fetches the COCO sample.

## Dataset import

Bring an already-labeled YOLO or COCO dataset, or an OpenProcessor export,
into a project. Under `/curation/projects/{project}`:

| Step | Route |
|---|---|
| Upload an archive | `POST /datasets/uploads` |
| Supported formats | `GET /datasets/formats` |
| Preview (writes nothing) | `POST /datasets/preview` |
| Start | `POST /datasets/imports` |
| List, status | `GET /datasets/imports`, `GET /datasets/imports/{import_id}` |
| Per-image rows, issues | `GET /datasets/imports/{import_id}/entries`, `GET /datasets/imports/{import_id}/issues` |
| Cancel, resume, undo | `POST /datasets/imports/{import_id}/cancel`, `POST /datasets/imports/{import_id}/resume`, `POST /datasets/imports/{import_id}/undo` |

The preview returns the detected format, totals, splits, per-class
suggestions and an `import_key`. Every dataset class needs one decision, by
name:

- `map` to an existing class (`class_id`),
- `create` a new class (`new_class_name`),
- `skip` it, or
- `region`: the class is a region of its parent item, not an item.

```bash
curl -s -X POST $API/datasets/preview -H 'content-type: application/json' \
  -d '{"source": {"path": "/data/source/import_fixture/yolo"}}' | jq '{format, totals, classes}'
curl -s -X POST $API/datasets/imports -H 'content-type: application/json' -d '{
  "source": {"path": "/data/source/import_fixture/yolo"},
  "mapping": [{"dataset_class": "car", "action": "map", "class_id": 0}],
  "options": {"processing": "none", "freeze_test_split": true, "name": "coco_yolo"}
}' | jq '{import_id, status}'
```

A COCO layout (`images/` next to `annotations/*.json`, no `labels/`) is
detected without naming its format; `"format": "coco"` forces it. Options: `label_trust` (`validated` or
`suggestion`), `missing_label` (`unlabeled` or `negative`), `processing`
(`none` or `propose` to run the detector), `parents`, `region_containment`,
`freeze_test_split`, `source_tag`, `force`.

Properties:

- Imports are chunked, persisted and resumable. One import runs per project at
  a time. A repeated request with the same source, mapping and options is
  idempotent (`reused: true`).
- A human edit made between plan and write still wins (the lock rule).
- Undo restores class snapshots and box history, deletes the items and
  images the import created, and deprecates classes it created. A second
  undo reports zeros.
- Importing a newer dataset version over an earlier import removes the items
  the new version no longer has, except the ones a person or a holdout freeze
  touched.
- Backpressure on the region worker: `OP_DATASET_IMPORT_MAX_PENDING`.
- Import fixture for testing: `make sample-coco-import` builds four layouts
  (`yolo/`, `coco/`, `yolo_region/`, `yolo_region_only/`) from 88 CC BY COCO
  images with a `FIXTURE.json` of expected counts. The YOLO layout numbers
  classes unlike any registry, spells `Car` differently and adds a synonym,
  and injects label problems; the wheel boxes in the `yolo_region*` layouts
  are synthetic geometry, not annotations.

### Importing other formats

The importer reads YOLO (`data.yaml` plus `images/` and `labels/`), COCO
(`images/` next to `annotations/*.json`) and OpenProcessor exports. Pascal
VOC, CVAT, LabelMe, Label Studio and segmentation, OBB or keypoint-specific
formats are not read directly. Convert them to YOLO or COCO first, then
import the result:

- Pascal VOC, CVAT (XML or "CVAT for images"), LabelMe, Label Studio:
  convert with a general converter such as `fiftyone` (`fiftyone.utils`
  importers plus `export(dataset_type=fo.types.YOLOv5Dataset)`) or
  Roboflow's `supervision` (`sv.DetectionDataset.from_pascal_voc(...)` then
  `.as_yolo(...)`). Keep one class name per class: the import decides by name.
- Polygon, OBB or keypoint labels: YOLO rows with more than five columns are
  imported as their bounding box (`yolo_polygon_to_box`); the extra geometry
  is not kept.
- After converting, check the result with `POST /datasets/preview` before
  starting the import. A `format_undetected` error means no YOLO, COCO or
  export layout was found at the path.

Items from an import carry `import_ids` and an `imported` review tab.
`GET /curation/projects/{project}/crops` and `GET /curation/projects/{project}/review/{tab}`
filter by `import_id`, `dataset_split` and `on_negative_frame`.

To score the region cascade against region ground truth, import with
`region` classes and run
`scripts/curation/eval_regions_vs_gt.py --dataset <data.yaml> --state-dir <dir> --wait-pending 1800`
(recall, precision, F1, mean IoU, a background false-positive gate and a
worst-first miss list).

## Reprocess

`POST /curation/projects/{project}/reprocess` re-runs one or more scopes over
images, items or a filter. It is a dry run by default (`dry_run: true`) and
reports per scope what is selected and what the lock rule skips.

```bash
curl -s -X POST $API/reprocess -H 'content-type: application/json' -d '{
  "targets": {"filter": {"profile_not": "wheel_example", "include_detected": true}},
  "scopes": ["region"], "region_mode": "redetect", "dry_run": true
}' | jq
```

- `scopes`: any of `detect`, `open_vocab`, `region`, `vlm`, `embed`.
- `targets`: `crop_ids`, `image_ids` or a `filter`. The filter is the item
  filter every list route takes (`class_names`, `exclude_class_names`,
  `conf_min`, `conf_max`, `min_area`, `max_area`, `max_rank`, `origin`,
  `embedding_state`, `review_status`, `source`, `import_id`, ...) plus the
  reprocess-only selectors (`region_status`, `missing_status`, `profile_not`,
  `profile_revision_below`, ...). A filter can be capped with `limit` and
  `sample` (`largest` boxes or a seeded `random` draw, with `seed`). The
  image-level selectors `all_images` and `open_vocab_status` also reach images
  with no item yet (they take no `limit`).
- `embed`: the options of the `embed` scope. `only_missing: true` embeds only
  the items that have no vector (crop and region boxes, frame vector left
  alone) and skips the rest; without it every selected vector is rewritten.
  A crop-id or filter target embeds exactly the items it names, not the whole
  image. A newly embedded item with no class gets the residual cluster ingest
  would have given it; class fields are never written. The dry run reports
  `to_embed`, `without_vector`, `region_boxes_to_embed` and
  `estimated_vector_kb` in `detail`.
- `region_mode`: `redetect` (clear and re-run the cascade) or `reverify`
  (re-run VLM verification on existing boxes).
- Detect and embed over many images return a job; poll
  `GET /curation/projects/{project}/reprocess/jobs/{job_id}` and stop it with
  `POST /curation/projects/{project}/reprocess/jobs/{job_id}/cancel`. The
  synchronous limit is `OP_REPROCESS_SYNC_MAX`.
- `POST /curation/projects/{project}/crops/{crop_id}/reprocess` and
  `POST /curation/projects/{project}/images/{image_id}/reprocess` run one
  target.

## Open-vocabulary detection

A prompt set (config axis `open_vocab`) lists text prompts such as "traffic
cone". SAM 3 runs each prompt on the WHOLE source image and every hit becomes a
normal item with `class_detector: sam3`, `source_prompt`, `open_vocab_set`,
`open_vocab_revision` and (with `mask`) a `mask_polygon`. The target's
`class_name` is the registry class, by name, and the item's `class_source` is
`open_vocab_target`: the VLM never relabels it (its answer is kept as the
name-only suggestion `vlm_proposed_class_name`, see the lock rule below). An
empty `class_name` stores the hit as an unlabeled proposal named by the prompt
(`class_source: open_vocab_proposal`), which the VLM may label. It needs the
segmenter (`OP_SEGMENTER_URL`). Items written before `open_vocab_target`
existed keep `open_vocab_proposal`; `source_prompt` and `open_vocab_set` filter
both.

| Operation | Route |
|---|---|
| Sets: list, create, read, save, delete, clone | `GET /curation/projects/{project}/open_vocab`, `POST /curation/projects/{project}/open_vocab`, `GET /curation/projects/{project}/open_vocab/{name}`, `PUT /curation/projects/{project}/open_vocab/{name}`, `DELETE /curation/projects/{project}/open_vocab/{name}`, `POST /curation/projects/{project}/open_vocab/{name}/clone` |
| List with the shipped example sets | `GET /curation/projects/{project}/open_vocab?include_templates=true` (templates are omitted unless you pass it) |
| Validate, form schema | `POST /curation/projects/{project}/open_vocab/validate`, `GET /curation/projects/{project}/open_vocab/schema` |
| Activate, deactivate, roll back | `POST /curation/projects/{project}/open_vocab/{name}/activate`, `POST /curation/projects/{project}/open_vocab/deactivate`, `POST /curation/projects/{project}/open_vocab/active/rollback`, `GET /curation/projects/{project}/open_vocab/active` |
| Try one unsaved target on one image | `POST /curation/projects/{project}/open_vocab/test` |
| Run over images | `POST /curation/projects/{project}/reprocess` with scope `open_vocab` |

- A pass writes through the same item writer as ingest, so a hit gets its crop,
  embedding, cluster and region seed like any detector item. Ids are
  deterministic in (image, box): a re-run upserts, and output of the same set
  that a re-run no longer produces is removed.
- Lock rule: a hit overlapping a locked item (IoU 0.8 or more, any class) is
  skipped and counted; an existing item with the SAME label that overlaps at
  `dedup_iou` or more wins. A hit's label is its target's class name (or its
  prompt in discovery mode); an existing item's label is its class name or, for
  an unlabeled detector proposal, the detector's own label. A box with no label
  at all, or another label, never absorbs a hit: both are kept. A locked item is
  never overwritten, replaced or deleted. A VLM answer for an `open_vocab_target`
  item is stored as a suggestion and the class, class source and cluster stay.
- A segmenter outage is never "no hit": the image is left as it was. A target
  named like a primary-detector class is a warning. At most
  `max_enabled_targets` (default 8, ceiling 32) targets may be enabled.
- `run_on_ingest` (off by default) queues newly ingested images in a background
  task and stamps them `open_vocab_status: pending` until done. A sweeper in the
  API (every `OP_OPEN_VOCAB_SWEEP_S` seconds, default 120, 0 = off) finishes
  images a restart or a segmenter outage left `pending` for longer than
  `OP_OPEN_VOCAB_STALE_S` seconds (default 240, two sweep ticks, floor 30), while the active set has `run_on_ingest` on.
- Throughput: up to `OP_OPEN_VOCAB_CONCURRENCY` images (default 4) are in
  flight at once, each fanning out one segmenter call per target. The dry run's
  `estimated_minutes` uses the measured per-call latency (a moving average of
  this API process's calls, 0.36 s until it has measured one) divided by the
  smaller of the segmenter's instance count and that fan-out. A job over many
  images reports `images_done` and `updated_at` after every image.
- One gate decides whether to spend a segmenter call: registry rules
  (`parent_classes`), an optional vision-model yes/no
  (`gating.tier2_vlm_precheck`) and an optional hit-rate sampler
  (`gating.tier3_hit_rate`); both optional tiers are off by default and a
  skipped target keeps its earlier output.
- Metrics: `op_open_vocab_call_seconds`, `op_segmenter_gate_decisions_total`,
  `op_open_vocab_hits_dropped_total`, `op_open_vocab_items_written_total`.

See the [Open-vocabulary detection guide](../docs-site/docs/guides/open-vocabulary.mdx).

## Combine projects

`POST /curation/projects/combine` builds a new project from 1 to 8 existing
ones. The sources are only read.

1. `POST /curation/projects/combine/preview` writes nothing and returns
   errors, warnings, a `suggested_mapping`, counts, duplicate and conflict
   numbers, bytes to link and a `preview_sha`.
2. `POST /curation/projects/combine` starts the job with
   `expected_preview_sha` and answers 202 with `{job_id, target}`. The target
   is `building` while it fills, then `active` (`failed` on an error).
3. Follow it with `GET /curation/projects/combine/{job_id}`,
   `POST /curation/projects/combine/{job_id}/cancel` and
   `POST /curation/projects/combine/{job_id}/resume`. Progress is also
   published as the `combine.progress` event.

```bash
curl -s -X POST $BASE/projects/combine/preview -H 'content-type: application/json' -d '{
  "target": {"slug": "fleet", "display_name": "Fleet"},
  "sources": [{"project": "cars"}, {"project": "trucks"}]
}' | jq '{ok, errors, preview_sha, suggested_mapping}'
```

Rules: each source class maps to a target class by name (`map`, `create`,
`skip` or `region`, same completeness rule as dataset import); the target owns
its ids and nothing numbered in a source crosses. Byte-identical images are
copied once and their boxes merge by IoU and target class (human beats import
beats VLM beats model); a box with a conflicting class keeps the priority
label and is flagged `combine_conflict` (filter
`GET /curation/projects/{project}/review/{tab}?combine_conflict=true`). Items
record `origin_project`, `origin_item_id` and `origin_image_id`. Frozen test
splits are kept by union (`holdout: preserve_union`) or recomputed. Deleting
a failed or unwanted target is a complete undo. Jobs live under
`OP_COMBINE_JOBS_DIR` and are chunked by `OP_COMBINE_PAGE_SIZE`.

## Review, search and clustering

- Queues: `GET /curation/projects/{project}/review/tabs` lists them with
  their filters; `GET /curation/projects/{project}/review/{tab}` pages one and
  `GET /curation/projects/{project}/review/{tab}/locate` finds an item's page.
  With the region profile off, the `regions` tab is an empty queue whose
  `empty_reason` says so (rows written under an earlier profile are not served).
- Search: `GET /curation/projects/{project}/search/text` (semantic, needs the
  PE text encoder and `OP_SEMANTIC_SEARCH_ENABLED`).
- Item clustering: `GET /curation/projects/{project}/clusters`,
  `POST /curation/projects/{project}/clusters/refine/{cluster_id}`,
  `POST /curation/projects/{project}/clusters/auto_promote`.
- Auto-label job: `POST /curation/projects/{project}/pipeline/auto_label/start`
  clusters by default; pass `run_vlm=true` to label with the VLM, and
  `class_id` to scope the run to one class. Poll
  `GET /curation/projects/{project}/pipeline/auto_label/status`.
- Scores, diverse selection and the projection:
  `POST /curation/projects/{project}/scores/compute`,
  `POST /curation/projects/{project}/select/diverse`,
  `POST /curation/projects/{project}/viz/projection/rebuild`.

### Filter, select, act

One item filter is shared by every route that lists, searches or counts
items, and by every route that acts on a selection of them. Class identity is
by name, never by model class id.

| Parameter (query) / field (body) | Meaning |
|---|---|
| `class_name`, `exclude_class_name` | Class names, repeatable. Match an item's class name or the detector's own label; case, spaces and hyphens are normalized (`traffic light` is `traffic_light`) |
| `conf_min`, `conf_max` | Inclusive confidence band |
| `min_area`, `max_area` | Box area as a fraction of its image |
| `max_rank` | The N largest boxes per image |
| `origin` | `detector`, `sam3`, `human` or `import`, repeatable (how the item came to exist; `source` is the separate ingest tag) |
| `embedding_state` | `embedded` (has a vector), `not_selected`, `deferred` or `failed`, repeatable |
| `review_status` | `pending`, `validated`, `dismissed` or `excluded`, repeatable |

Routes that take it as query parameters: `GET .../crops`,
`GET .../review/{tab}` (and its `/locate`), `GET .../search/text`,
`GET .../stats/classes`, `GET .../stats/dataset`, `GET .../clusters`,
`GET .../regions` and `GET .../detections/summary`. Each keeps its own
route-specific parameters on top. A malformed band (`conf_min` above
`conf_max`, `min_area` above `max_area`) is a 400.

`GET /curation/projects/{project}/detections/summary` answers "what did the
detector store, and what is embedded?": a count per detector label with its
embedding breakdown, over the items the filter selects, plus a
`suggested_reprocess` body that embeds the missing ones.

Acting on a selection, in two steps (filter, then act):

- **Bulk writes** (`POST .../crops/batch_exclude`, `POST .../crops/batch_unexclude`,
  `PUT .../crops/batch_label`, `POST .../crops/move`) take `crop_ids` or a
  `selection`: `{filter, limit, sample, seed}`. `limit` caps the selection to
  its `limit` largest boxes (`sample: largest`) or a seeded random draw
  (`sample: random`). `dry_run: true` returns `{dry_run, selected}` and writes
  nothing; the write then changes exactly those ids. A selection above 20000
  items is refused (422), never truncated. Hold-out and excluded items are not
  selected unless `include_test` / `include_excluded` (or
  `review_status: [excluded]`) say so.
- **Embedding**: `POST .../reprocess` with scope `embed` takes the same filter
  (see [Reprocess](#reprocess)); `embedding_state` plus `only_missing` embeds
  what the policy skipped.
- **VLM labeling and the lazy embed trigger**:
  `POST .../pipeline/auto_label/start` takes the filter as query parameters
  and scopes the VLM stage and its unvalidated count to it (with `class_id`
  and `cluster_id`). `embed_missing=true` adds a first stage that embeds the
  in-scope items stored without a vector, so they are clustered and labeled
  in the same run; the worker builds its encoder from `OP_TRITON_URL` or
  `TRITON_URL` (default `triton-server:8001`). Clustering itself is index-wide
  by design (the centroids are shared by ingest placement); the primary
  subject gate (`gate_max_rank`, `gate_min_blur_ratio`) is its only scope.
- **Export**: `POST .../export/yolo` takes an `item_filter` (the same filter
  as a JSON object) that narrows the validated items written; the manifest
  records it. The single-class export is scoped by `class_ids`. To hide items
  from any export, exclude them (`POST .../crops/batch_exclude` with a
  selection, reversible with `POST .../crops/batch_unexclude`).

Cluster ids: items cluster on `pe_embedding` by default
(`OP_RESIDUAL_EMBEDDING_FIELD`); region boxes cluster on their own box
embedding.

## Export and training

Freeze a test holdout first with
`POST /curation/projects/{project}/test_holdout/freeze`, check it with
`GET /curation/projects/{project}/test_holdout/stats`.

| Purpose | Route |
|---|---|
| Multi-class export | `POST /curation/projects/{project}/export/yolo` |
| One class or a class subset | `POST /curation/projects/{project}/export/single_class` |
| Status, list | `GET /curation/projects/{project}/export/status`, `GET /curation/projects/{project}/export/datasets` |
| Frozen artifacts | `GET /curation/projects/{project}/export/registry/{artifact}` |

`export/yolo` always exports every class and rejects unknown body keys (422).
To train a subset use `export/single_class`, or the top-level
`include_classes` field of the training spec (not `hyperparameters`, which
rejects it with a 422). Both exporters record
`dataset_sha`, a hash of the written label content plus the ordered class
list, and flip their `current` symlink atomically. They split by source
image, so items cut from one image never straddle splits; an image with a
frozen holdout item goes to `test`. The multi-class export writes one image
and one label file per source image, one `cls cx cy w h` line per validated
object. Reviewed-negative frames are included by default
(`include_negative_frames`).

An image can hold objects that are not labeled yet. By default it is still
exported with its validated objects, and the manifest records
`unlabeled_items_on_exported_images`; preflight then warns
(`export_unlabeled_objects`). Pass `require_fully_labeled_images: true` to
leave such images out.

`export/single_class` with `box_source: "region"` and `image_mode:
"item_crop"` writes one box per accepted region, cropped to its parent item.
It adds background and hard-negative frames and a `frozen_test_sha`.

Training is a control plane over a shared-volume file protocol
(`src/services/training/jobs.py`). Routes under
`/curation/projects/{project}`:

| Purpose | Route |
|---|---|
| Check a spec | `POST /train/preflight` |
| Start a run, a multi-size campaign | `POST /train/start`, `POST /train/start_campaign` |
| Status, run list, log tail | `GET /train/status`, `GET /train/status/{job_id}`, `GET /train/runs`, `GET /train/log/tail/{job_id}` |
| Cancel | `POST /train/cancel/{job_id}`, `POST /train/cancel_campaign/{campaign_id}` |
| Profiles, presets, GPUs | `GET /train/profiles`, `GET /train/presets`, `GET /train/gpus` |
| Lineage, promote, reload | `GET /train/manifest/{job_id}`, `POST /train/promote/{job_id}` (202 + `promote_id`: use it, and poll, for UIs and anything behind a proxy that times out near 120 s; `?wait=true` blocks 2-3 minutes and is for direct scripting with a >= 300 s client timeout; `force` bypasses the promote gate and goes in the JSON body or as `?force=true`), `GET /train/promote/{job_id}/jobs/{promote_id}`, `POST /train/reload_promoted` |

```bash
docker compose --profile training up -d curation-trainer
curl -s -X POST $API/train/preflight -H 'content-type: application/json' -d '{}'
curl -s -X POST $API/train/start -H 'content-type: application/json' -d '{
  "model_family": "yolo26", "model_size": "s", "profile": "small",
  "hyperparameters": {"epochs": 70, "imgsz": 640, "batch": 16, "optimizer": "MuSGD"}
}'
```

- `dataset_export_dir` defaults to the current export. With no export yet,
  preflight reports one blocking check telling you to export first.
- `start` does not merge profile defaults: copy the values from
  `GET /train/profiles` into `hyperparameters`. A bare top-level `epochs` is
  a 422 (`extra='forbid'`).
- The trainer downloads the family checkpoint on first use unless
  `hyperparameters.model` points to a local file. On an air-gapped host,
  download it first and set that path.
- Promote exports the model to Triton under the project. A subset or
  single-class run needs its `class_remap`; promote refuses (422) rather than
  serve the full registry as if it were the trained subset, unless
  `force=true`.
- A promoted YOLO26 model has a fused `[300, 6]` output. It serves through
  `POST /detect?model_name=<promoted>`, but is not a drop-in ingest detector
  (`OP_INGEST_PRIMARY_DETECTOR_MODEL` expects the end2end four-tensor
  response).
- Model comparison (bake-off) is per project under
  `/curation/projects/{project}/bakeoff/` (`eval_datasets`, `trained_models`,
  `baseline_models`, `profiles`, `run`, `runs`, `status/{job_id}`,
  `results/{job_id}`, `matrix/{job_id}`). Start the evaluator with
  `docker compose --profile curation up -d curation-evaluator`. It is a
  long-lived watcher; do not use `docker compose run --rm` for it. See the
  [design rationale](design/curation_design_rationale.md#8-the-detector-bake-off-harness-bakeoffprofile).
- Active-learning probe: `POST /curation/projects/{project}/probe/run`
  scores validated items with the new model;
  `GET /curation/projects/{project}/review/{tab}` has a model-disagreement
  view.

## Models you must supply

Nothing here ships a pretrained region detector, VLM or trainer weights.

**Rebuild `yolo-api` before exporting any model.** A pulled image can predate
the source tree. If `make export-pe` fails with
`ModuleNotFoundError: No module named 'core'`, the `perception_models` package
is missing from a stale image: run `docker compose build yolo-api && docker
compose up -d --force-recreate yolo-api`, then re-run the export.

- **PE-Core-L14-336 encoders.** The image tower `pe_image_encoder` is
  required and its Triton name is fixed (`src/clients/pe_encoder.py`: input
  `images` `[B, 3, 336, 336]`, output `image_embeddings` `[B, 1024]`). The
  result is the `pe_embedding` field that semantic search, near-duplicate
  detection, clustering and the projection run on. The text tower
  (`pe_text_encoder`) is needed for semantic search. `OP_PE_TEXT_BACKEND=auto`
  prefers Triton and falls back to a lazily loaded in-process PyTorch backend;
  `onnx` is an explicit opt-in because every worker loads its own session.
  Build: `make pe-download`, `make pe-export-image`, `make pe-build-trt` (or
  `make pe-build-ort`), `make pe-export-text-triton`, or the whole chain with
  `make export-pe`; then load both models in Triton and restart the API.
  `make pe-text-status` confirms the text backend. Details:
  [`../export/README.md`](../export/README.md#pe-core-encoders-curation-embeddings).
- **An item detector for ingest**: an end2end Triton model named by
  `OP_INGEST_PRIMARY_DETECTOR_MODEL` (plus other `OP_INGEST_PRIMARY_<FIELD>`).
  A stock COCO checkpoint proposes all 80 classes and ingest stores them all;
  narrowing is the per-project ingest policy (see "Ingest policy" below), not
  an env var (`OP_INGEST_PRIMARY_CLASS_IDS` is retired and ignored with a
  warning). To switch detectors set
  `OP_INGEST_PRIMARY_DETECTOR_MODEL` (and `OP_INGEST_PRIMARY_LABELS_PATH` when
  the model directory has no `labels.txt`) and recreate `yolo-api`; the model
  must serve the end2end four-tensor output. The choice is deployment-wide.
  `OP_INGEST_PRIMARY_CONFIDENCE_FLOOR` only applies when the primary assigns
  classes; a proposer stores every detection the engine emits. A region profile
  should keep `parent_classes` set, because with a full-vocabulary detector an
  empty list matches every class. By default the primary is a
  proposer (`OP_INGEST_PRIMARY_ASSIGNS_CLASS=false`): detections are unlabeled
  `<name>_proposal` items carrying the model's own label, never a registry
  class looked up by id. Set it true only when the primary was trained on your
  registry. Ingest returns 503 until a detector is configured and loaded. An
  optional raw-output secondary detector
  (`OP_INGEST_SECONDARY_DETECTOR_MODEL`, `OP_INGEST_SECONDARY_<FIELD>`)
  overrides the primary's class on IoU-matched boxes.
  `GET /curation/projects/{project}/class_sources` lists the `class_source`
  values a deployment can write.
- **A region profile and segmenter, if you want regions.** The segmenter is
  any service at `OP_SEGMENTER_URL` speaking the segment wire protocol
  (`scripts/curation/worker/client.py`). `docker/segmenter/` is a reference
  implementation behind the `segmenter` compose profile; it needs a GPU and a
  Hugging Face token. With `OP_SEGMENTER_URL` empty the leg is skipped.
- **A dual-head detector, if you want `backbone_embedding`.** It is the
  detector's backbone feature map RoI-pooled over each box. A stock export
  emits only the detection tensor; re-export with
  [`export/export_detector_dual_head.py`](../export/export_detector_dual_head.py)
  (`output0` plus `sppf_feat`). Optional: residual clustering reduces
  `pe_embedding` by default. `OP_BACKBONE_EMBEDDING_DIM` sets the mapping only
  when the items index is created.
- **An OCR model, if your region has readable text**
  (`ocr_rec_model` or `ocr_pipeline_model` in the profile).
- **A VLM**: see [Choosing a VLM](#choosing-a-vlm).
- **A dataset and base weights** for training.

## Workers and compose profiles

The HTTP API (browse, label, cluster, export) works with no workers. The
asynchronous half is opt-in:

```bash
docker compose --profile curation up -d
```

| Service | What it does |
|---|---|
| `curation-detection-worker` | Runs the region cascade over `pending_detection` items for every active project. |
| `curation-vlm-worker` | Verifies and labels items through each project's active VLM endpoint. |
| `curation-auto-label-worker` | Drives the auto-label protocol. Clusters only unless a caller passes `run_vlm=true`. |
| `curation-cluster-refresh` | Periodically refreshes the clustering. |
| `curation-evaluator` | Long-lived bake-off watcher; mounts `./data` read-only. |

None needs Triton or a GPU to start; they idle or error per call until a
detector and VLM are configured. Workers serve every active project and skip
a paused one. Each worker writes a heartbeat file under `OP_HEARTBEAT_DIR`
that its container health check reads.

The segmenter and the trainer are separate profiles because they need a GPU:

```bash
OP_SEGMENTER_URL=http://segmenter:8000 \
  docker compose --profile curation --profile segmenter up -d
docker compose --profile training up -d curation-trainer
```

`docker/trainer/` implements the job-file protocol. Without it running,
preflight's `trainer_reachable` check is a warning, never a block. The probe
reads the trainer's heartbeat on the shared `/jobs` volume, so it needs no
Docker socket.

### GPU arbiter

`OP_GPU_ARBITER_CONTAINERS` lets a training job stop and restart named sibling
containers so they do not compete for GPU memory. That needs the Docker socket,
which is not mounted by default:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu-arbiter.yml \
  --profile training up -d
```

Mounting `/var/run/docker.sock` gives that container root-equivalent control
of the Docker host. Read the header of `docker-compose.gpu-arbiter.yml`
first. Without the overlay, the stop and start calls are no-ops and the API
logs one `arbiter_docker_unavailable` warning per outage.

## Mounting your image source

Ingest resolves paths under `OP_SOURCE_ROOT` (container default
`/data/source`). Both `yolo-api` and `curation-detection-worker` must mount the
same host directory at the same container path:

```yaml
volumes:
  - ${OP_SOURCE_ROOT_HOST:-./data/source}:/data/source:ro
```

Set `OP_SOURCE_ROOT_HOST` in `.env`. If only `yolo-api` has the mount, ingest
succeeds but every later worker read fails with
`detection_failed` / `reason=image_unavailable`. `GET
/curation/projects/{project}/ingest/config` lists the accepted source roots.
Because the mount is read-only, run the dataset fetch scripts on the host,
not through `docker compose exec`.

## Wiring up Cropwright

Cropwright, or any frontend that consumes `/curation`, needs:

| Cropwright env var | Value | Why |
|---|---|---|
| `API_UPSTREAM` | `http://op-api:8000` | `yolo-api` has the network alias `op-api`; `http://yolo-api:8000` works too. |
| `PUBLIC_API_PREFIX` | `/curation` | Must equal `OP_API_PREFIX`. |
| `PUBLIC_TRITON_API_URL` | empty in Docker | The frontend talks to Triton only through the API. |
| Docker network | `${COMPOSE_PROJECT_NAME:-openprocessor}_triton_net` | Join it as an external network. |

## Quick start for a fresh install

1. Start the API (`docker compose up -d`). Indexes are created on startup.
2. Create a project and classes ([Use your own domain](#use-your-own-domain)).
3. Build the PE encoders and load them in Triton ([Models you must supply](#models-you-must-supply)).
4. Set `OP_INGEST_PRIMARY_DETECTOR_MODEL`.
5. Ingest or import, start `--profile curation`, review, export, train.

A per-deployment checklist of the `OP_*` variables the compose file reads:

| Purpose | Vars |
|---|---|
| Ingest / detector | `OP_INGEST_PRIMARY_DETECTOR_MODEL`, `OP_SOURCE_ROOT_HOST` |
| Region detection | `OP_REGION_PROFILE_PATH`, `OP_SEGMENTER_URL` / `OP_SEGMENTER_URLS` |
| VLM | `OP_VLM_URL`, `OP_VLM_MODEL`, `OP_VLM_API_KEY` |
| Feature flags | `OP_SCORES_ENABLED`, `OP_SCORES_SHADOW`, `OP_SEMANTIC_SEARCH_ENABLED`, `OP_VIZ_PROJECTION_ENABLED`, `OP_SELECT_DIVERSE_ENABLED` |
| GPU and training | `OP_GPU_ALLOWED_IDS`, `OP_GPU_LABELS`, `OP_GPU_ARBITER_CONTAINERS`, `OP_GPU_ARBITER_TRAINER_CONTAINER`, `OP_TRAIN_DEFAULT_GPUS`, `OP_TRAIN_GPU_ORDER` |
| Projects and API | `OP_PROJECTS_DATA_ROOT`, `OP_PROJECT_INDEX_PREFIX`, `OP_API_PREFIX` |
| Image build (compose only) | `OP_IMAGE_REPO`, `OP_IMAGE_TAG`, `OP_BUILD_SHA` |

## Environment variables

All curation `OP_*` variables are optional. The authoritative list with
defaults and comments is [`../env.template`](../env.template); its
"Curation quick-config" block gathers the ones every tier needs. See
[INSTALLATION.md](../INSTALLATION.md#curation-quick-config) and the
[README Quick Start](../README.md#quick-start).

Curation routers build their mount prefix and index names at import time, so
set these before `src.main` is imported; they cannot change at runtime.
Profiles, packs, VLM endpoints and settings are not environment: they live in
the config store and change at runtime.

| Area | Vars |
|---|---|
| Projects | `OP_PROJECT_INDEX_PREFIX`, `OP_PROJECTS_INDEX`, `OP_PROJECTS_DATA_ROOT` |
| Filesystem roots | `OP_SOURCE_ROOT`, `OP_SOURCE_PATH_ALIASES`, `OP_STATE_DIR`, `OP_CROP_CACHE_DIR` |
| Prompt pack files | `OP_PROMPT_PACK_PATH` (the boot default pack), `OP_PROMPT_PACK_PATHS` (extra packs, comma-separated) |
| Config store | `OP_CONFIG_POLL_S` |
| API surface | `OP_API_PREFIX`, `OP_API_TAG` |
| Embedding and HNSW | `OP_EMBEDDING_DIM`, `OP_ENCODER_EMBEDDING_DIM`, `OP_BACKBONE_EMBEDDING_DIM`, `OP_HNSW_EF_CONSTRUCTION`, `OP_HNSW_M` |
| Region field-name overrides | `OP_REGION_FIELD_<ATTR>` (see `RegionFields`) |
| Ingest detectors | `OP_INGEST_PRIMARY_<FIELD>`, `OP_INGEST_SECONDARY_<FIELD>` (tuple and frozenset fields take comma-separated values) |
| Region profile boot default | `OP_REGION_PROFILE_PATH` (a JSON profile file), `OP_REGION_PROFILE` (a name registered by deployment code), `OP_REGION_DETECTION_<FIELD>` (per-field overrides) |
| Region limits | `OP_REGION_MAX_BOXES_PER_WRITE` |
| Ingest and upload | `OP_MAX_INGEST_CONCURRENCY`, `OP_UPLOAD_MAX_IMAGES_PER_REQUEST`, `OP_UPLOAD_MAX_BYTES_PER_REQUEST`, `OP_UPLOAD_ACCEPTED_EXTENSIONS` |
| Dataset import | `OP_DATASET_IMPORTS_DIR`, `OP_DATASET_IMPORT_CHUNK`, `OP_DATASET_IMPORT_MAX_PENDING`, `OP_DATASET_IMPORT_MAX_FAILED_CHUNKS` |
| Reprocess | `OP_REPROCESS_JOBS_DIR`, `OP_REPROCESS_SYNC_MAX`, `OP_OPEN_VOCAB_CONCURRENCY`, `OP_OPEN_VOCAB_SWEEP_S`, `OP_OPEN_VOCAB_STALE_S` |
| Combine | `OP_COMBINE_JOBS_DIR`, `OP_COMBINE_PAGE_SIZE` |
| PE text encoder | `OP_PE_TEXT_BACKEND`, `OP_PE_TEXT_ONNX_PATH`, `OP_PE_TEXT_TRITON_MODEL`, `OP_PE_TEXT_ORT_THREADS` |
| Feature flags (off by default) | `OP_SEMANTIC_SEARCH_ENABLED`, `OP_VIZ_PROJECTION_ENABLED`, `OP_SELECT_DIVERSE_ENABLED`, `OP_SCORES_ENABLED`, `OP_SCORES_SHADOW` |
| Item scores | `OP_SCORES_KNN_K`, `OP_SCORES_NPROBE`, `OP_SCORES_STATE_DIR`, `OP_CROP_DUP_THRESHOLD`, `OP_FIELD_COVERAGE_TTL_S` |
| Probe | `OP_PROBE_JOBS_DIR`, `OP_PROBE_ACTIONABLE_MIN_CONFIDENCE` |
| Diverse selection | `OP_SELECT_JOBS_DIR`, `OP_SELECT_JOB_MAX_N`, `OP_SELECT_MAX_N`, `OP_SELECT_SYNC_MAX_OPS`, `OP_SELECT_CACHE_TTL_S` |
| Clustering and IVF | `OP_IVF_RETRAIN_CHECK_S`, `OP_IVF_RETRAIN_GROWTH`, `OP_IVF_RETRAIN_MIN_INTERVAL_S`, `OP_MAX_REFINE_MEMBERS`, `OP_OUTLIER_CACHE_TTL_S`, `OP_OUTLIER_MAX_MEMBERS`, `OP_RESIDUAL_EMBEDDING_FIELD` |
| Training | `OP_TRAIN_JOBS_DIR`, `OP_TRAIN_RUNS_ROOT`, `OP_TRAIN_STAGING`, `OP_PREFLIGHT_SCAN_CAP`, `OP_MLFLOW_PUBLIC_URL` |
| GPU arbiter | `OP_GPU_ALLOWED_IDS`, `OP_GPU_ARBITER_CONTAINERS` (`name` or `name@ids`, e.g. `segmenter@0/2`), `OP_GPU_ARBITER_TRAINER_CONTAINER`, `OP_GPU_LABELS` (`id=label` pairs), `OP_TRAIN_DEFAULT_GPUS`, `OP_BAKEOFF_JOBS_DIR` |
| Export | `OP_BUILD_SHA` |
| Bake-off | `OP_BAKEOFF_OUT_DIR`, `OP_BAKEOFF_CONCURRENCY`, `OP_BAKEOFF_GPUS`, `OP_BAKEOFF_BASELINES_PATH`, `OP_BAKEOFF_PROFILE`, `OP_BAKEOFF_PROFILE_<FIELD>`. Jobs queue per project in `$OP_STATE_DIR/projects/<slug>/bakeoff_jobs`; the evaluator must watch `$OP_STATE_DIR/bakeoff_jobs` on the same path as the API's state dir. |
| Worker and pipeline | `OP_API`, `OP_AUTO_LABEL_STATE_DIR`, `OP_EVENT_API_URL`, `OP_EVENT_BUS` (`file` or `process`), `OP_EVENT_LOG_MAX_BYTES`, `OP_HEARTBEAT_DIR`, `OP_PAUSE_SENTINEL`, `OP_WORKER_PAUSE_SENTINEL`, `OP_VIZ_JOBS_DIR`, `OP_VIZ_MAX_N` |
| VLM built-in endpoint | `OP_VLM_URL`, `OP_VLM_MODEL` (required when `OP_VLM_URL` is set), `OP_VLM_API_KEY`, `OP_VLM_MAX_IMAGES_PER_CALL`, `OP_VLM_OPEN_IMAGES_PER_CALL`, `OP_VLM_EXTERNAL_POLICY` (`ack` or `deny`), `OP_VLM_HTTPX_MAX_CONNECTIONS`, `OP_VLM_HTTPX_KEEPALIVE` |
| Segmenter | `OP_SEGMENTER_URL`, `OP_SEGMENTER_URLS`, `OP_SEGMENTER_HTTPX_MAX_CONNECTIONS`, `OP_SEGMENTER_HTTPX_KEEPALIVE` |

## Wire naming

Every OpenSearch field defaults to a `region_*`, `vlm_*` or `classifier_*`
name, and the wire uses the same generic vocabulary whatever the storage
names are (see the key-invariant section of
[`design/curation_api_contract.md`](design/curation_api_contract.md)). A
deployment with existing data under other field names builds its own
`RegionFields` through `OP_REGION_FIELD_<ATTR>` with no reindex.
Retired environment-variable prefixes are rejected at startup by
`src/config/retired_env.py`.

Item wire fields that were renamed or replaced in 0.4.0:

| Old | Now |
|---|---|
| Single-box item fields (`region_bbox_norm`, `region_score`, `region_detector`, `region_text*`, `region_cluster_*`, `region_candidate_*`) | An element of `region_boxes[]` (`bbox_norm`, `score`, `detector`, `text*`, `cluster_*`) |
| `region_thumbnail_url` on the item | `thumbnail_url` on each box |
| Region cluster counts `n_regions`, `assigned` | `n_boxes`, `n_boxes_changed`, `n_items_written` |
| Item refine counts `n_members`, `n_updated` | `n_items`, `n_items_updated` |
| FP centroid build `n_members`; auto pull `n_scanned`, `n_moved` | `n_boxes`; `n_boxes_scanned`, `n_boxes_moved` |
| `GET /models/status` VLM rows with `kind: "external"` | One row per registered endpoint with `kind: "vlm"` |

## Known limits

- No authentication on the API. Do not expose it to the internet; see
  [`../SECURITY.md`](../SECURITY.md).
- BYO models: the encoders, item detector, segmenter, VLM and base weights are
  yours to supply.
- A region profile is per project, but one detection worker process serves
  every project, and a project runs one active profile at a time.
- Text reading and the OCR text hint are not previewed by the profile test
  route.
- The trainer needs internet access on first use of a base checkpoint unless
  you point `hyperparameters.model` at a local file.
- Coverage is uneven across the surface; the least-tested routers are the
  older ones.

## Resource links

`GET /curation/projects/{project}/settings` includes `resource_links` (API docs plus the
monitoring and MLflow UIs, each with `status` and `reachable`). The rules and
env vars are in `docs-site/docs/operations/monitoring.mdx`.
