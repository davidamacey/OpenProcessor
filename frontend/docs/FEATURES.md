# Feature tour

Screenshots below are captured live against a running instance (real
data, dark theme, 1600×1000 viewport). See the main [README](../README.md)
for setup/config; this doc is purely a visual walkthrough of what each
route does.

## Dashboard

Pipeline health at a glance: data-integrity recluster controls (AHC
residual clustering, auto-promote, Gemma pass), live pipeline stats
(labeled-by-source breakdown, plate coverage, unlabeled backlog,
in-flight queue depth), and one-click actions (run Gemma labeling,
export dataset, snapshot indexes).

![Dashboard](screenshots/dashboard.png)

## Cluster grid (`/clusters`)

Every cluster (class or candidate) as a card: dominant class, purity,
size, representative crops. Filter by purity band, "unlabeled only",
sub-clustered, search by text, or toggle the embedding-plot lasso-select
overlay for a 2D view of the pool.

![Clusters](screenshots/clusters.png)

## Cluster detail (`/clusters/[id]`)

The core triage surface: a full crop grid for one cluster, pointer-based
drag-and-drop onto class rows, bulk select/confirm/move/ignore, per-class
hotkeys, "Run Gemma" / "Accept Gemma" for the whole page, and "Refine
(AHC)" to re-split a messy cluster on the spot.

![Cluster detail](screenshots/cluster-detail.png)

## Crop detail modal

Click the ⓘ on any crop card for full provenance: source image with the
detection bbox drawn in, the cropped region, class/label-source/
confidence/cluster metadata, and the raw source path — everything needed
to sanity-check a label without leaving the grid.

![Crop detail modal](screenshots/crop-detail-modal.png)

## Review queue (`/review`)

A single-item-at-a-time keyboard-driven queue for the harder cases: five
tabs (All, Uncertainty, Model Disagreements, COCO Blind Spots, Plates),
quick-filter preset chips on All (Gemma mismatches / Gemma low-conf /
Primary low-conf), fuzzy class search, and full keyboard shortcuts
(`Enter` confirm, `N` skip, `D` discard, `Z` undo, class-letter assign).

![Review](screenshots/review.png)

## Review — Plates tab

The one genuinely domain-specific workflow in this repo today: a bbox
editor, an OCR plate-text field (Gemma-read), a detector-provenance
chip strip showing the full cascade (`LPR:HIT` → `Gemma verify`), and a
shape-warning gate for implausible boxes. See
[GH issue #1](https://github.com/example-org/openprocessor/issues/1)
for the plan to generalize this into a per-class opt-in
specialized-annotation plugin (secondary bbox detection + free-text
field + provenance chain + validity gate), rather than a one-off
hardcoded to `license_plate`.

![Review regions](screenshots/review-regions.png)

## Class management (`/classes`)

Add/rename/regroup classes, assign per-class hotkey letters, merge a
class into another (bulk-relabels matching crops), and see validated
vs. total crop counts per class at a glance. Deprecated classes are
listed separately (restore isn't wired up yet — see README "Limits").

![Classes](screenshots/classes.png)

## Export (`/export`)

Per-class augmentation-target gap tracking, one-shot test-holdout
freeze, and YOLO-format export to a versioned staging directory with
downloadable `class_registry.json` / `data.yaml` / `manifest.json`
artifacts.

![Export](screenshots/export.png)

## Train cockpit (`/train`)

Submit a training job against a frozen export: dataset/version picker,
model size + profile (nano through xlarge), GPU allocation, live
preflight checks (blocks on real issues before you waste a run), and a
log tail once it's running. Past runs support Promote (to Triton) and
Reproduce (same spec/lineage, fresh job).

![Train](screenshots/train.png)

## Bake-off (`/bakeoff`)

Score every selected trained model against every selected frozen
dataset in one matrix run (via the on-demand evaluator container),
bolding the best model per dataset.

![Bakeoff](screenshots/bakeoff.png)

## Model registry (`/models`)

Live status of every inference service backing the pipeline — Triton
models (detector, classifier, plate detector, image encoder) and the
external Gemma VLM — with per-model inference counts, latency, and a
force-unload control.

![Models](screenshots/models.png)

## Keyboard shortcuts overlay

Press `` ` `` anywhere to see every shortcut for the current page,
plus the full per-class hotkey table (also an inline hotkey editor —
click a letter box and press a new key to rebind).

![Shortcuts](screenshots/shortcut-overlay.png)
