---
sidebar_position: 3
title: Clusters
---

# Clusters (`/clusters`, `/clusters/[id]`)

## Cluster grid (`/clusters`)

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
  text filters).
- One synthetic inventory card per registered annotation slot with a browse
  endpoint is pinned to the unfiltered grid.

<Screenshot name="clusters-1600.png" alt="Cropwright cluster grid" caption="Cluster grid with cohesion badges" />

## Cluster detail (`/clusters/[id]`)

The core triage surface: a crop grid for one cluster with pointer-based
drag and drop onto class rows, bulk select/confirm/move/discard/ignore,
per-class hotkeys, **Run VLM** and **Accept VLM** for the page, **Refine**
(sub-cluster with agglomerative clustering), a strategy bar (sort, score
chips, diverse-selection overlay) and a core-member cut line.

See [Keyboard shortcuts](./keyboard-shortcuts.md) for the full key table.

## Crop detail

The info button on any crop opens its full provenance: the source image
with every item and region box drawn client-side (toggleable), the crop
itself, class/label-source/confidence/cluster metadata, OCR text lines,
label-write history, and sibling crops from the same source image.
