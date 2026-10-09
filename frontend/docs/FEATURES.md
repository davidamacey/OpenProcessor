# Feature tour

A route-by-route walkthrough of what Cropwright does. See the main
[README](../README.md) for setup and configuration, and
[CLAUDE.md](../CLAUDE.md) for the authoritative, detailed route and
keyboard-shortcut tables.

Screenshots are not published yet. Every public screenshot will be
captured from a fresh-start run on public datasets only (COCO val2017 for
the main flow, license-checked Open Images "Vehicle registration plate"
images for the region example), which the backend fetches and Cropwright
never bundles.

Everything below that depends on an optional backend feature is **absent,
not broken**, when the backend doesn't serve it: the nav link or page
section simply doesn't render.

## Dashboard (`/dashboard`)

Pipeline health at a glance: live dataset stats (polled every 10 s),
recent labels, and quick actions. **Run Clustering Now** starts an
auto-label run with stage progress, shared with the backend's own
scheduled run. When the backend advertises a VLM prompt pack, an optional
assist scope points the VLM-assisted sweep at a single class instead of
the whole pool. With a served region profile, a detections panel shows
region coverage.

## Ingest (`/ingest`)

Bring images into the pool. Browser upload (files, folders, drag and
drop) is chunked to the served per-request cap with bounded concurrency,
pre-filters images the backend already has, and shows a per-file result
(ingested / duplicate / failed, with the served reason and a stable error
code). Runs can pause, resume and cancel. A server-path mode appears only
when the backend advertises batch source roots. The page also shows ingest
status by source, a region-drain panel (with a region profile), and a
clustering handoff that waits for the served "drained" verdict.

## Cluster grid (`/clusters`)

Every cluster as a card: dominant class, cohesion (the served
nearest-centroid purity), size and representative crops. Filter by
cohesion band or class, search by embedding similarity or by OCR text,
list the **Ignored** bucket with a restore action, and (when the backend
serves a projection) open an embedding plot to lasso a set of points and
preview them side by side. Filtering on the region class swaps the grid
for the region gallery.

## Cluster detail (`/clusters/[id]`)

The core triage surface: a crop grid for one cluster with pointer-based
drag and drop onto class rows, bulk select / confirm / move / discard /
ignore, per-class hotkeys, **Run VLM** and **Accept VLM** for the page,
**Refine** (sub-cluster with AHC), a strategy bar (sort, score chips,
diverse-selection overlay) and a core-member cut line.

## Crop detail

The info button on any crop opens its full provenance: the source image
with every item and region box drawn client-side (toggleable), the crop,
class / label source / confidence / cluster metadata, OCR text lines,
label-write history and sibling crops from the same image.

## Review queue (`/review`)

A one-item-at-a-time, keyboard-driven queue. Tabs: **All**,
**Uncertainty**, **Model Disagreements**, **Classifier Blind Spots**,
**New Class Proposals**, and one region tab when the backend serves a
region profile (labelled by the profile's display name). The All tab adds
quick-filter preset chips (VLM mismatches, VLM low-conf, primary
low-conf). Filters (class, source, confidence and any served per-tab
enum filter) and deep links (`?crop_id=`) resolve on the server. An empty
queue shows the served reason and a link to the step that fills it (run a
probe, compute scores). Keys: `Enter` confirm, `/` class search, `N`
skip, `D` discard, `Z` undo, `←`/`→` navigate, plus class letters.

### Region tab

Driven entirely by the served region profile: a sub-box editor (`E` to
edit), confirm / reject / false positive (`F`), the backend's chosen text
reading with reader-disagreement and choice badges, a detector-provenance
chip strip, served rejection reasons, and verifier-rejected candidates
drawn dashed so a human can accept them.

## Class management (`/classes`)

Add, rename, merge, deprecate and restore classes, and bind a per-class
hotkey letter (reserved action keys are rejected). A merged class shows
what it was merged into instead of a Restore button. The **Proposals**
section lists VLM-proposed terms the registry doesn't have yet: create a
class from one, or map it onto an existing class, resolving every pending
item that proposed it (dry-run count first, `Z` to undo).

## Export (`/export`)

Per-class balance and trainability gaps, a one-shot test-holdout freeze,
and YOLO export to a versioned directory, with the served image, object
and per-split counts and downloadable `class_registry.json`, `data.yaml`
and `manifest.json`. An option exports only images whose every object is
labeled.

## Train cockpit (`/train`)

Submit a training job against an export: class selection (defaulting to
classes with enough data), model size, augmentation preset (served list),
GPU allocation and live preflight checks. Live progress and log tail
while it runs. Past runs show results (test-split evaluation, per-class
table, MLflow link, lineage), **Promote** to Triton (with the served gate
report), **Reproduce**, and **Run probe predictions**, which feeds the
Uncertainty and Model Disagreements queues. A training-cohorts picker
previews served cohorts per class.

## Bake-off (`/bakeoff`)

Compare trained and baseline models on evaluation datasets in one run:
pick datasets, models and a scoring profile, then read a model × dataset
matrix (served winners bolded) and per-dataset ranked results with a
per-class table. Present only when the backend mounts its bake-off
router.

## Model registry (`/models`)

Live status of every inference service the backend uses (Triton models
and external services such as the VLM and segmenter), with inference
counts and latency. Models the backend marks unloadable get an Unload
button; pipeline-protected models show a "protected" chip instead.

## Settings (`/settings`)

Deployment-wide curation defaults (clustering method, review sort, VLM
prompt pack), each shown only when the backend marks it settable, with a
confirm step before every save. The curation scores card shows per-scorer
coverage and starts, tracks and cancels score computation.

## Keyboard shortcuts overlay

Press `` ` `` anywhere to see every shortcut for the current page and the
per-class hotkey table, with an inline editor to rebind a class letter.
