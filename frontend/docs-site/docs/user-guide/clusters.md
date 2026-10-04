---
sidebar_position: 3
title: Clusters
---

# Clusters (`/p/<project>/clusters`, `/p/<project>/clusters/[id]`)

## Cluster grid (`/p/<project>/clusters`)

Every cluster as a card: dominant class, **cohesion** (the served
nearest-centroid purity — labelled "cohesion" rather than "purity" so a
mixed-labels-but-visually-coherent cluster doesn't read as "bad"), size and
representative crops.

- Filter by cohesion band or class; search by embedding similarity
  (semantic search, when enabled) or literal OCR text.
- An **Ignored** toggle lists the excluded bucket with a restore action.
- When the backend serves an embedding projection, an overlay lets you
  lasso a set of points on a 2-D plot and preview them side by side —
  visualization only, it never feeds a clustering decision.
- Filtering on the region class swaps the grid for the region gallery
  (browsing the region queue directly, with detector/verified/status/score/
  text filters, plus a **Box state** filter; see
  [Multi-box regions](./multi-box-regions.md)).
- One synthetic inventory card per registered annotation slot with a browse
  endpoint is pinned to the unfiltered grid.

<Screenshot name="clusters-1600.png" alt="Cropwright cluster grid" caption="Cluster grid with cohesion badges" />

<Screenshot name="wheels-inventory-card-1600.png" alt="Cluster grid with a Wheels inventory card pinned first, showing the number of region items listed" caption="Wheels inventory card — pinned first, it opens the slot's region gallery" />

The filter bar above the grid, and the **Matching items** mode that lists and
bulk-edits everything the filter matches, are described in
[Item filter and Matching items](./item-filter.md).

## Cluster detail (`/p/<project>/clusters/[id]`)

The core triage surface: a crop grid for one cluster with pointer-based
drag and drop onto class rows, bulk select/confirm/move/discard/ignore,
per-class hotkeys, **Run VLM** (with an optional per-run endpoint picker, see
[VLM models](./vlm-models.md#choosing-a-vlm-per-run)) and **Accept VLM** for
the page, **Reprocess** for the selection (see
[Reprocess](./dataset-import.md#reprocess)), **Refine**
(sub-cluster with agglomerative clustering), a strategy bar (sort, score
chips, diverse-selection overlay) and a core-member cut line.

See [Keyboard shortcuts](./keyboard-shortcuts.md) for the full key table.

## Crop detail

The info button on any crop opens its full provenance: the source image
with every item and region box drawn client-side (toggleable), the crop
itself, class/label-source/confidence/cluster metadata, OCR text lines,
label-write history, and sibling crops from the same source image. It also
shows provenance when recorded (import, combine origin, VLM endpoint, model
and prompt pack), and a **lock** icon on a label a human has set.
