# Curation subsystem — design rationale

Status: **living reference doc**, owned by this backend. This is the
canonical answer to "why is it built this way" for the `curation`
subsystem: the configuration dataclasses, the storage/wire split, the
project and config-store model, multi-box regions, the lock rule, class
identity by name, the pre-commit ratchet exemptions carried by the files
that introduced them, and the gaps that are known and tracked rather than
accidental. It complements, and deliberately does not duplicate,
[`curation_api_contract.md`](curation_api_contract.md) (the HTTP wire
contract itself) and [`../ARCHITECTURE.md`](../ARCHITECTURE.md#curation-subsystem)
(the component map).

## 1. Where this subsystem came from

The `curation` subsystem — active-learning review queues, clustering,
VLM-assisted labeling, dataset export, and the training-job API under
`CurationConfig.api_prefix` (default `/curation`) — was genericized out
of an earlier, domain-specific internal implementation built for one
deployment. That earlier implementation hardcoded its domain everywhere:
OpenSearch field names, detector model names and geometry heuristics, index names,
filesystem paths. None of that is inherent to "curate a stream of
detector crops with human review and active learning" — it's one
deployment's parameter values. The genericization's job was to find the
seam between "generic curation mechanics" and "one deployment's
parameters" and turn every hardcoded value on the wrong side of that
seam into configuration.

That is the through-line for everything below: **the mechanics are
generic Python; the domain lives entirely in data**. Deployment-level data
is a set of frozen dataclasses built from environment variables. Domain-level
data (classes, region profile, prompt pack, VLM endpoint, settings) is stored
per project and edited at runtime, so changing domain never means forking or
restarting.

## 2. The configuration dataclasses

### 2.1 `CurationConfig` (`src/config/curation.py`)

Holds index names, filesystem roots, and API-surface constants: which
OpenSearch indexes back images/items/labels/classes/clusters, where the
class registry and exported datasets live on disk, the crop-cache
directory, the API mount prefix, and embedding-dimension/HNSW tuning.

Before genericization, these were a hardcoded, company-prefixed index
enum and a scatter of module-level path constants. Two deployments never share
index names, cache paths, or a mount prefix by coincidence — they're
deployment data, not code — so they moved onto a dataclass with an
`IndexRole` + `index_name()` lookup (typo-proof; a role can't resolve to
the wrong deployment's index by a copy-paste mistake) and a
`from_env(prefix='OP_')` classmethod so an operator overrides one field
via environment variable without touching source. `get_curation_config()`
is the process-wide default, built through `from_env()` exactly once and
memoized; anything needing a *different* configuration (an overlay for
an existing deployment with different index names) constructs its own
instance and injects it rather than mutating the default.

### 2.2 `RegionFields` (`src/config/region_fields.py`)

See §4 below — it's substantial enough to warrant its own section.

### 2.3 `DetectionProfile` (`src/config/detection_profile.py`)

Describes one detectable "region of interest" type as data: aspect-ratio
and area heuristics for auto-confirmation, a text-hint pattern and
length range, which Triton models (detector/segmenter/OCR) back it, and
their input sizes and confidence floors. The earlier internal cascade
hardcoded all of this for exactly one region type.
A deployment describing a different region — a barcode on a package, a
tag on livestock — constructs its own `DetectionProfile` instance
instead of branching or forking the cascade code that consumes it.

`DetectionProfile.from_env(prefix)` sets every field from environment
variables under three prefixes: `OP_REGION_DETECTION_*` (the region cascade,
optionally on top of a profile selected by name with `OP_REGION_PROFILE` or
loaded from a file with `OP_REGION_PROFILE_PATH`), `OP_INGEST_PRIMARY_*` and
`OP_INGEST_SECONDARY_*` (the ingest item detectors). The environment profile
is now only the boot default and one read-only source in the config store
(§9): a project activates a stored profile over it. No region profile ships
built in, so an unconfigured deployment advertises an empty
`detection_profile` axis and the region worker idles. The dataclass default
field values still describe the reference deployment's tuning, so a new
profile should set every field it relies on.

### 2.4 `RegionStatus` (`src/config/region_state.py`)

The canonical state-machine enum for the region-of-interest pipeline
(`pending_detection` → detector cascade → `detected` /
`verify_rejected` / `no_region_box`; `pending_verification` → VLM
verify → `detected` / `no_region_visible`; any path can short-circuit to
the terminal `detection_failed`, and a human reviewer can additionally
mark a detected box `false_positive` without deleting it, preserving
provenance for hard-negative training). On-disk string values are kept
byte-identical to what earlier code wrote directly as literals, matching the
same no-reindex reasoning as `RegionFields` (§4). The status is item-level
and derived from the item's boxes (§10); the box-level vocabulary is
`proposed`, `accepted`, `rejected`, `false_positive`.

## 3. The frozen wire-contract split

The subsystem draws a hard line between two independent naming systems
that happen to look similar:

- **HTTP JSON field names** on the Pydantic wire models under
  `src/routers/curation/_common.py` — the contract every API consumer
  (the labeling frontend, any future service) speaks. These are frozen:
  once shipped, a field is not renamed, repurposed, or dropped on the
  wire.
- **OpenSearch document field names** — how the backend actually stores
  a value internally. These are configurable per §4.

A router handler can read `doc[region_fields.status]` internally while
returning a response whose Pydantic attribute is a fixed name — a
storage-field rename never has to ripple into a wire-format change, and
a wire-format decision never constrains which OpenSearch field name a
deployment's existing data happens to use. See
[`curation_api_contract.md`](curation_api_contract.md) for the full
route-by-route contract and the specific model attributes this applies
to; this doc only states the principle because §4 depends on it.

## 4. `RegionFields` and why there is no reindex

`RegionFields` is the indirection that makes the wire/storage split in
§3 actually hold at the OpenSearch layer. Every place the backend reads
or writes a per-item "region of interest" sub-annotation (the box list,
status, verification state, counts, revision) goes through a `RegionFields`
instance rather than a literal string. The generic OSS defaults are
`region_*` names (`region_status`, `region_boxes`, `region_count`, …); a deployment whose existing
OpenSearch index already has data under different field names (the
reference deployment's were prefixed for its domain) constructs its own
`RegionFields` instance with those names instead.

**Why this matters enough to be its own module:** an existing
deployment adopting this generic code does not want to reindex however
many million documents just to satisfy a generic module's naming
opinion. Reindexing is a real-money, real-downtime operation on a
production dataset; renaming a Python-level indirection is not. By
routing every field access through `RegionFields`, a rename becomes a
config flip — point the dataclass at the existing field names — with
zero data migration and zero reindex risk. The alternative (hardcode
`region_*` everywhere and require every adopter to migrate their data to
match) would make the generic code effectively unusable by the one
deployment it was extracted from.

`RegionFields.from_env(env_prefix='OP_REGION_FIELD_')` gives the same
per-field environment-variable override that `CurationConfig` has, and
`get_region_fields()` is the equivalent memoized process-wide default.
Two things worth stating plainly because they have been a source of
confusion before:

- This indirection governs OpenSearch query bodies, `_source` lists,
  bulk-update documents, painless scripts, and index mapping bodies —
  **not** the HTTP wire contract (§3), which is frozen independently,
  and **not** enum *values*, which are untouched.
- `get_region_fields()` **does** call `from_env()` — every
  `OP_REGION_FIELD_*` override is live. (An earlier internal audit
  claimed otherwise, confusing this module with a since-fixed bug in
  `CurationConfig.get_curation_config()` that briefly had the same
  symptom. Do not "fix" this again; both are the correct, already-fixed
  form.)

`scripts/codegen/check_no_literal_region_fields.py` is the automated
half of this guarantee: a pre-commit hook that bans a raw
`'region_...'`-shaped domain literal (the exemplar the check was built
against was the reference deployment's own field prefix) from appearing
in any file it has been told to police, via a growing `PORTED_PATHS`
allowlist. Each wave that ports or writes new curation code appends its
paths to that allowlist in the same commit — once a module is covered,
it can never silently regress to hardcoding a storage field name again.
Two categories are deliberately exempt from the ban and documented
in-file: `RegionFields`' own docstrings (which must name the
illustrative override example) and the frozen Pydantic wire-model
attribute declarations from §3 (which are a different naming system
entirely and were never in this guard's scope).

## 5. The pre-commit ratchet exemptions

`.pre-commit-config.yaml`'s `max-file-size` hook caps source files at
700 LOC to keep modules reviewable and discourage grab-bag files. Five
files that arrived with the curation port are grandfathered past that
cap:

- `src/clients/curation_opensearch.py`
- `src/services/training/triton_promote.py`
- `src/routers/curation_train.py`
- `src/services/training/jobs.py`
- `scripts/curation/worker/runner.py`

The reasoning is the same for all five and worth stating once instead
of once per exclude-list comment: each corresponds to a genuinely
cohesive reference-implementation module that was already over the cap
*before* any porting work touched it. Splitting a file correctly
requires understanding its internal seams well enough to draw a
sensible boundary; doing that split *while simultaneously* genericizing
the file's contents would have produced a diff that mixed structural
reorganization with semantic changes in the same commit — exactly the
kind of diff a reviewer cannot meaningfully check line-by-line. The
porting decision was: land the file whole (readable, semantically
diffable against its reference counterpart), grandfather it explicitly
with an in-file comment naming which port chunk added it, and treat the
split as separate, tracked follow-up work rather than silently deferring
it.

Practical consequence for anyone extending one of these five files: add
functionality to the existing file rather than treating the grandfathering
as license to keep growing it indefinitely, and do not add an eighth
undocumented exemption — a genuinely new oversize file should be split
before it is committed, following the same reasoning that will
eventually retire these five. Two have already gone, each now split into
modules per concern that are all under the cap and no longer exempt:
`src/services/detection/cascade_detect.py` became the
`src/services/detection/cascade_detect/` package, and
`src/services/curation/clustering/orchestrator.py` became `orchestrator.py`
(residual-pool clustering) plus `refine.py`, `retrain_policy.py`,
`residual_gate.py` and `cluster_write_guard.py` in the same package.

## 6. Known gaps

These are documented so they read as "known and tracked," not
discovered-in-production surprises.

- **Ingest does not carry a domain policy.** `POST
  /curation/projects/{project}/ingest/image` and `/ingest/batch` create items
  (duplicate detection, quality gate, crop-cache population, bulk indexing),
  and `POST /curation/projects/{project}/datasets/imports` brings in labeled
  YOLO, COCO or OpenProcessor-export datasets. What is deliberately not
  included is a domain-specific detector ensemble, a fixed class allowlist or a
  region-status assignment policy: those are the region profile, the class
  registry and your detector models.
- **The asynchronous half is opt-in and needs your models.** `docker compose
  --profile curation up -d` starts the detection, VLM, auto-label and
  cluster-refresh workers. The trainer (`docker/trainer/`, profile `training`)
  and the segmenter (`docker/segmenter/`, profile `segmenter`) ship as
  reference containers speaking the file and wire protocols the API already
  uses. Both need a GPU; you bring weights and a dataset.
- **One detection worker serves every project and each project runs one
  active region profile and one active VLM endpoint at a time.** Switching is
  a runtime activation, not a restart, but two profiles are never active in
  one project at once.
- **`DetectionProfile` dataclass defaults are the reference deployment's
  tuned numbers** (§2.3). A stored profile should set every field it relies
  on; the profile `validate` route reports what a draft leaves implicit.
- **No authentication of any kind on the API.** See `SECURITY.md`. Several
  routes are destructive (`DELETE /curation/projects/{project}/models/{model_name}`)
  or read server-side paths (`POST /curation/projects/{project}/ingest/batch`,
  dataset import). Do not expose the service directly to the internet.
- **Coverage is uneven across the surface.** Some routers carry thorough
  suites; the older ones carry thinner ones.

None of this blocks the core loop of ingest or import, browse, cluster,
review, label and export. It limits how turnkey the subsystem is for an
arbitrary deployment, which is why it is labelled experimental for v0.4.0;
see `docs/CURATION.md`.

## 7. Labeling-assist selection: `PromptPack`, `DetectionProfile`, and the frontend's annotation-slot model

The frontend's labeling-assist UX lets an operator pick which items or
classes they want assistance with and scope a run to that selection. That
needed a fourth member of the configuration family (§2): `PromptPack`
(`src/services/labeling/vlm_prompts.py`), discoverable over the wire next to
`DetectionProfile`, without conflating either with the frontend's own concept
of an "annotation slot."

**Why `PromptPack` is separate from the deployment config.** `CurationConfig`
holds names and paths: small, uniformly typed data. A `PromptPack` is a dozen
multi-paragraph prompt templates plus two vocabulary tables
(`class_descriptions`, `synonyms`), all specific to one labeling domain. A
deployment swaps its vocabulary independently of its index names or its
detection heuristics. A pack is now stored per project in the config store
(§9) and activated at runtime; `OP_PROMPT_PACK_PATH` and
`OP_PROMPT_PACK_PATHS` remain as the file-based boot defaults, and a missing or
malformed file degrades to the built-in generic pack with a logged warning
rather than crashing the labeler.

**Why `PromptPack` and `DetectionProfile` are not the same concept.**
`DetectionProfile` describes a region *cascade*: which detector and
segmenter, which thresholds, which classes it applies to, whether text is
read. `PromptPack` describes a *VLM conversation*: what to ask and what
vocabulary to expect back. They correlate per domain, but nothing ties them
structurally. A project can run a profile with no VLM stage, or a pack with no
region profile (the VLM classifies whole-item crops). The one place they must
agree is the reply contract: a profile that keeps several boxes per item needs
a pack whose region prompts return a per-box list. The pairing check enforces
that on save (warning) and on activation (error).

**Why neither is the frontend's "annotation slot."** A slot is a UI-level
grouping of what a curator sees and edits for one class. A profile and a pack
are backend pipeline concepts that influence what shows up for review. The
join between them is data (a shared class name), not a code dependency, and a
run scopes on `class_id` directly: `POST
/curation/projects/{project}/pipeline/auto_label/start?class_id=<id>`.

**Discovery.** `GET /curation/projects/{project}/methods` advertises the
selectable strategies on the `cluster`, `score`, `sort`, `overlay`, `export`,
`detection_profile`, `prompt_pack` and `vlm` axes in one
`{id, axis, label, status, default}` shape, with a per-entry `settable` flag.
The effective default for an axis comes from the project's settings document
when it names a still-advertised id, else from the active config-store record,
else from a built-in constant; the same resolution function serves every
endpoint that applies a default, so `/methods` and real behaviour cannot
disagree. A template profile is not advertised until a project clones and
activates it, which keeps an unconfigured deployment's axis empty.

**Worked example: a pallet labeling-assist setup.**

1. Create a project and add the pallet classes
   (`POST /curation/projects/{project}/classes`; see
   `data/class_registry.example.json` for a warehouse registry).
2. Save and activate a region profile for the sub-region the cascade should
   find (a pallet ID tag): `POST /curation/projects/{project}/region_profiles`
   then `POST /curation/projects/{project}/region_profiles/{name}/activate`.
   `POST /curation/projects/{project}/region_profiles/test` previews the
   result on a stored crop before activating.
3. Save and activate a prompt pack with the pallet vocabulary (start from
   `data/prompt_pack.example.json`): `POST
   /curation/projects/{project}/prompt_packs` then `POST
   /curation/projects/{project}/prompt_packs/{name}/activate`.
4. `GET /curation/projects/{project}/methods` now reports the pack's `name`
   on the `prompt_pack` axis.
5. `POST /curation/projects/{project}/pipeline/auto_label/start?class_id=<id>&run_vlm=true`
   scopes the run's unvalidated-item query to that class instead of the whole
   pool.

## 8. The detector bake-off harness: `BakeoffProfile`

`scripts/curation/bakeoff/` scores any number of detectors on the same
frozen YOLO test split with one shared metric (pycocotools COCOeval plus a
fixed-threshold operating point), so a new training run, earlier runs and
external baselines are compared on identical ground, per class.
`src/routers/curation/bakeoff.py` does not score anything: it resolves a
request into a job spec (`src/services/curation/bakeoff_jobs.py`) and drops
`<job_id>.job.json` into a shared directory; the evaluator container
(`docker/evaluator/Dockerfile`, running `bakeoff_runner --watch`) does the
scoring and writes `status.json`, one `comparison.json` per dataset and a
`matrix.json` back. The harness lives under `scripts/` rather than `src/`
because it needs a newer detection stack than the API image pins and never
runs inside the API process. The rest of this section covers the shipped
design; see [`docs/design/curation_api_contract.md`](curation_api_contract.md)
for the exact `/bakeoff/*` wire shapes.

**The flow.** Eval datasets are the exports themselves: every export under
`CurationConfig.export_root` with a labelled test split is listed by
`GET /curation/projects/{project}/bakeoff/eval_datasets` (id `export:<path>`), next to optional
frozen third-party sets (`external:<group>/<name>`,
`src/services/curation/eval_datasets.py`). Contenders are finished training
runs (`GET /curation/projects/{project}/bakeoff/trained_models`, no weight upload), external
models from a baseline registry, or a custom model such as a deployed
Triton model. `POST /curation/projects/{project}/bakeoff/run` re-scores every selected model on
every selected dataset in one job; two hashes pin the test split, identity
(`frozen_test_sha`, which frames) and content (`test_label_sha`, which
boxes), and the evaluator refuses a dataset whose content hash changed
since enqueue. The trainer's opt-in auto-quantize posts the same request
for a finished run, so its ONNX variants land in the same comparison.

**Multi-class scoring and class mapping.** A model's class ids are mapped
onto the eval dataset's ids per (model, dataset): by an explicit name map,
by a run's `class_remap` or its training export's registry ids (resolved by
the API at enqueue, the same way promote resolves them), else by normalized
class names inside the evaluator. Eval classes a model does not cover are
reported as not covered with null metrics, and predictions of unmapped
model classes are counted; nothing is dropped silently. Each row carries
`overall` metrics over the model's own covered classes and `common` metrics
over the classes every model covers; ranking uses `common` (competition
ranks, ties listed together), falling back to `overall` with a warning when
no class is common. A run whose training split shares images with the eval
test split gets a leakage warning, never a block. Results are
informational: promote does not read them.

**Why a profile.** The harness was ported from the same earlier internal
deployment as the rest of this subsystem (§1), and it carried
that domain in code: a hardcoded single-class id, a YOLO writer that
always emitted a fixed label, hardcoded classes baked in as the
crop-mode coarse stage, and domain-specific benchmark converters and
backends as built-ins. Following the `DetectionProfile` pattern (§2.3), everything that
decides *what* is measured lives on a frozen `BakeoffProfile` dataclass
(`scripts/curation/bakeoff/profile.py`):

| Field(s) | Replaces |
|---|---|
| `class_filter` | the hardcoded single target class (a profile now scores every class in the split, optionally narrowed by name) |
| `class_names` | the fixed single-class `data.yaml` written by dataset converters |
| `context_class_ids`, `context_weights/imgsz/conf` | the hardcoded COCO context-class ids used as the crop-mode coarse stage |
| `triton_model` (empty by design) | a default Triton model name — `--backend triton` now requires one from the profile or the request |
| `default_backend`, `imgsz` | CLI defaults (a run's own imgsz still wins) |
| `conf_floor`, `nms_iou`, `op_conf`, `op_iou`, `rank_metric` | metric thresholds and the comparison's hardcoded ranking metric |
| `converter_modules`, `backend_modules`, `baselines_path` | plate-benchmark converters, plate-only backends and baseline models shipped as built-ins |

A request's optional `profile` (a registered name or a `.json` path)
selects one; `GET /curation/projects/{project}/bakeoff/profiles` lists the registered profiles
and flags the default (`default: true`, `kind: registered` or
`configured` when `OP_BAKEOFF_PROFILE` names a `.json` path). With no
profile the neutral `generic` profile (every class, no context classes, no
Triton model) applies, overridable per field via `OP_BAKEOFF_PROFILE_*` env
vars or wholesale via `OP_BAKEOFF_PROFILE`.

**Datasets.** `datasets.py` is a converter registry around a generic
`YoloWriter` whose `data.yaml` names come from the profile. Only
domain-neutral formats are built in (`yolo` passthrough, Pascal `voc` with
object-name → class-id mapping); a single-class profile collapses every
source box onto class 0, a multi-class one keeps/maps ids and drops
anything outside its label space. Domain formats register themselves from
a profile's `converter_modules`.

**The `examples/bakeoff/license_plate/` example.** This directory keeps an
original example configuration as an opt-in reference, never a default and
never listed by the API: its `profile.json` (fixed context classes), the
domain-specific benchmark-dataset converters, and detector backends
(registered from the profile's
`backend_modules` through `scripts/curation/bakeoff/backends/registry.py`),
and a `baselines.json` of example detectors, each with a `class_map`.
It is loaded only by path (`profile: "<path>/profile.json"` or
`OP_BAKEOFF_PROFILE`), so `examples/` must be mounted into the API and the
evaluator to use it. Nothing in `src/` or `scripts/` imports it. The
default baseline registry is empty: previous runs are the baselines.

**The quantize leg.** A job's optional `quantize` block (`{run_id, formats,
n_calib, calib_split, throughput}`) runs `scripts/curation/bakeoff/quantize.py`
(Ultralytics FP32/FP16 ONNX export plus ONNX Runtime static QDQ INT8,
calibrated on the run's training export) and scores the variants in the
same job as `run:<id>:<format>`; failures are recorded as failed stages in
`status.json`, and a job with nothing left to score ends `state: error`.
An earlier internal deployment's CoreML leg drove a macOS host through a
proprietary driver and is not shipped: the request has no field for it.

**Scope.** The harness is a measurement tool. Scripts that only produce one
paper's tables and figures (threshold sweeps, number generators, figure
prototypes) are outside it and not shipped; the reusable lean-angle core lives
in `src/services/detection/region_lean.py`.
`tests/curation/test_bakeoff_harness.py` guards the harness core against
domain vocabulary creeping back in.

## 9. Projects and the config store

**Why projects.** A curation dataset is a set of indexes, a class registry,
exports, uploads and domain configuration. Sharing those across unrelated
datasets means a class id, a validated label or a prompt pack from one domain
can leak into another. A project is the isolation boundary: every
data-touching route is `/curation/projects/{project}/...`, every project owns
indexes named `<OP_PROJECT_INDEX_PREFIX><slug>__<role>`, and a project-bound
request fails closed when it cannot resolve its project instead of falling
back to a default. `default` is an ordinary project created at first start,
which can be archived but never deleted. A cross-project leak sweep test
issues every route against two projects and asserts neither sees the other.

**Lifecycle.** `active` is the working state. Archive makes a project
read-only and is refused while it has running jobs and for the last active
project. Delete is a guarded, background operation: a dry run lists what would
block it, a real delete needs `confirm=<slug>`, drains the detection worker,
removes indexes and files, and retires the slug forever so an old URL can
never resolve to a new project. `building` and `failed` exist for combine
targets (§14) so a half-built project is never selectable. Every mutation
takes `expected_revision`, so two operators cannot silently overwrite each
other.

**Why a config store.** Domain configuration as environment variables and files means that
changing a prompt is a file edit and a restart. Operators
need to edit, test and roll back configuration while workers run. The store
keeps three kinds of document (prompt packs, region profiles, VLM endpoints)
plus the settings defaults, with the same rules:

- Saves are immutable revisions with audit fields; a revision number is never
  reused, so "profile `wheels` r3" always means the same body.
- Activation is separate from saving. Saving a draft changes nothing a worker
  does. Activation takes an `expected_active` guard so two editors cannot
  activate over each other, and re-validates more strictly than save.
- Rollback is a first-class route, because the fastest fix for a bad
  activation is the previous one.
- Cross-process visibility is a poll (`OP_CONFIG_POLL_S`) against the stored
  document, not a signal, so API workers and the detection worker converge
  without coordination. The detection worker applies a new profile or pack
  only at a quiesce point, so one item is never processed under two profiles.
- An activation response reports its impact (items under another profile or
  an older revision, validated and pending counts) and a suggested reprocess
  body, because a profile only affects items processed after it. Re-running
  old items is an explicit act (§14), never a side effect of activating.
- Deployment-wide documents (the VLM endpoint registry) live in a separate
  global store; project documents are bound read-only when cloned from another
  project.

The environment profile and the file-based pack remain as the boot default and
as read-only sources in the store, so an existing deployment keeps working
until a project activates its own.

## 10. Multi-box regions

**Why a list.** The first design stored one region box per item as scalar
fields. That fits a license plate on a car and fails for wheels (a car has up
to four), tags on a pallet or defects on a part. Retrofitting a second box
onto scalars means parallel fields and a representative-box convention that
every reader has to know. The decision was a nested list, `region_boxes`, with
one element per box, and no scalar fallback: an item with one region has a
list of one. The scalar fields, routes and writers were deleted rather than
kept in parallel, so there is one code path and one `derive_status`.

What follows from the list:

- **Per-box state, verdict, text, cluster and embedding.** Anything a person
  can say about a region is said per box (`accepted`, `rejected`,
  `false_positive`, `proposed`), and a box has its own `box_id`, which is
  never reused after a delete. The item status is derived from the boxes with
  a fixed precedence (accepted, false_positive, proposed, rejected, empty), so
  a rejected box beside an accepted one no longer hides the accepted one.
- **Replace-the-list edits with an optimistic revision.** `PUT
  /curation/projects/{project}/crops/{crop_id}/regions` sets the whole list
  and carries `expected_region_revision`. The revision advances on any write
  that changes a box's state, geometry or text; cluster-only writes (partition,
  refine, false-positive sub-typing) do not advance it, so a background
  recluster does not 409 an open editor.
- **Candidate selection is an explicit step.** The worker gathers candidates
  from the detector and segmenter legs, applies a score floor, NMS and the
  profile's `max_regions_per_item`, then asks the VLM about numbered boxes.
  `region_set_complete` records the VLM's "all of them are here" answer. The
  profile test route returns every candidate with its drop reason so the cap
  is tunable.
- **Per-box embeddings and clustering.** Boxes are embedded and clustered, not
  items. A box carries `bbox_norm` in its embedding entry so a moved box is
  recognised as stale and re-embedded. False-positive sub-typing works on
  boxes: a matching box flips to `false_positive` and the item status is
  re-derived, so a sibling accepted box keeps the item `detected`.
- **Row shapes.** Box-selecting queries return one row per matching box with
  `region_box_id`; `total` counts items and `total_rows` counts rows. Every
  box filter applies to one and the same box, so `detector=a` AND
  `min_score=0.8` cannot match an item whose a-box scored low and whose b-box
  scored high.
- **Explicit mapping.** The nested list and its embeddings have an explicit
  OpenSearch mapping and a raised `index.max_inner_result_window`, and a
  per-write cap (`OP_REGION_MAX_BOXES_PER_WRITE`) bounds abuse. Element keys
  inside the list are fixed strings, not `RegionFields`-indirected: the list is
  new, so no deployment has legacy names for them.

## 11. The lock rule

An automated writer must never undo a human decision or a trusted import. The
rule is one definition (`src/clients/occ_locks.py`), consulted by every
automated writer, so no writer carries its own narrower copy:

- A class is locked when a human set it, when it is a validated imported label,
  or when the item is in the test holdout.
- A box is locked when a human created, moved, verdicted or transcribed it, or
  when it came from an import and is not a suggestion.
- An item is locked when its class is locked, any box is locked, or a human or
  import validated its region set.

`label_trust: suggestion` is the escape hatch: an import writes unvalidated
classes and `proposed` boxes, which the pipeline may replace. The rule is
enforced inside the optimistic-concurrency write itself, not only at the
decision point, so a human edit made between a worker's read and write still
wins. Reprocess reports what it skipped as `locked_skipped`. Imports and undo
share one "is an import still the sole owner" test, so undo and reconcile can
never delete an item a person touched after the import. The test holdout
belongs to the rule because a frozen test item whose class changes silently
invalidates every evaluation that used it.

## 12. Class identity is the name

A class id is a dense index into one project's registry at one moment. It is
not stable across projects, datasets, exports or models, and treating it as
identity is a silent-corruption bug: two YOLO datasets that agree on class
names but disagree on index order, merged by index, swap labels without an
error. So everything that crosses a boundary pairs by name:

- Dataset import maps each dataset class to a registry class by name through an
  explicit `map`, `create`, `skip` or `region` decision, and the preview shows
  what an index-based guess would have done wrong.
- Combine maps each source class by name; the target owns its ids and nothing
  numbered in a source crosses.
- Export writes a dense remap and the promoted model's `labels.txt` carries the
  trained subset; promote refuses a subset run it cannot remap.
- Bake-off maps model classes onto eval classes by explicit name map, a run's
  remap, or normalised names, and reports uncovered classes instead of
  dropping them.
- Region profiles select items by `parent_classes` names, matched against the
  item's class name or the detector's own label, never a detector index.

End-to-end tests walk import, export, train, promote and predict (and combine)
and assert the `(class_id, class_name)` pairing at every hop.

## 13. The VLM endpoint registry

**Why a registry.** One process-wide `OP_VLM_URL` cannot serve projects that
need different models, cannot be switched without a restart, and cannot be
tested before use. Endpoints are deployment-wide documents (a server is a
deployment fact) and activation is per project.

- **Probe before trust.** A probe sends synthetic images only and records the
  served model root, context length, image cost, image cap and JSON-mode
  support. It belongs to the revision and body it tested; a re-save is a new
  revision that needs a new probe.
- **One gate.** Every VLM-calling route, including per-run `?vlm=`, passes one
  function (`enforce_vlm_gate`), and a test walks each route with an input that
  must be refused. A second path with its own checks is how a safety rule
  rots.
- **Keys are references.** `secret:<slug>` points at a file on the host
  written by `openprocessor vlm key set`; the API never stores, serves or logs
  a key.
- **Outbound safety.** A URL policy refuses this stack's own services and
  link-local or metadata addresses for the literal host and every resolved
  address, no client follows a redirect, and an endpoint that sends crops
  outside the deployment needs an explicit acknowledgement recorded per
  `name@revision` (`OP_VLM_EXTERNAL_POLICY=deny` refuses them). DNS can change
  after validation, so each labeler re-checks at most every 30 seconds and
  fails closed. The residual risk is recorded in `SECURITY.md`.
- **Pairing.** A model that cannot read the prompt size, handle the image
  count, return JSON or verify several boxes is a bad pairing for the active
  profile and pack, so activation of any of the three runs the same check.
- **Provenance.** Items record `vlm_endpoint` and `vlm_model` per answer, and
  `region_verifier` records the resolved model root, so a label's origin
  survives a later switch.
- **Local models are a catalog, not an API action.** `examples/vlm/catalog.tsv`
  is read by the installer, the CLI and the API. The API records the desired
  local model; only the host CLI (`openprocessor vlm use`) restarts vLLM,
  because that needs a fit check, the training lock and an `.env` rewrite that
  restores itself on failure.

## 14. Dataset import, reprocess and combine

These three are bulk writers over data a person may have touched, so they
share one design.

- **Preview, then start, then undo.** Preview writes nothing and returns issues,
  suggestions and an `import_key` hashing the source, the name-based mapping
  and the write-affecting options. A repeated start with the same key is
  idempotent. Imports are chunked, persisted and resumable, with a write-ahead
  ledger so `created` versus `updated` stays truthful across a crash, and undo
  is the inverse of the ledger. Backpressure on the region worker
  (`OP_DATASET_IMPORT_MAX_PENDING`) keeps an import from burying the queue.
- **Dry run by default.** Reprocess (`POST
  /curation/projects/{project}/reprocess`) is the single re-run route for the
  `detect`, `region`, `vlm` and `embed` scopes. It counts what it would do and
  what the lock rule skips before it does anything, and large detect or embed
  runs are file-backed jobs that survive a restart. It replaced a handful of
  separate requeue, clear and retry paths whose differing guards were the
  source of inconsistent behaviour.
- **Combine reuses import.** `POST /curation/projects/combine` reads a source
  project as a dataset (the same scan, mapping and completeness rule as
  import), dedups byte-identical images by content hash, merges boxes by IoU
  with a fixed source priority (human, import, VLM, model), flags real class
  conflicts as `combine_conflict` for review instead of picking silently, and
  builds into a `building` target that is `active` only when it is complete.
  The preview carries a `preview_sha` the start must echo, so a source that
  changed since the preview is refused. Sources are bound read-only; deleting
  the target is a complete undo.
- **Delete-time re-check.** Deleting what an import or reprocess created
  re-reads each document and deletes with a sequence-number guard, so a human
  edit made after the decision survives.
