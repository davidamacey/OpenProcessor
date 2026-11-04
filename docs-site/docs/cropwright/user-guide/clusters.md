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

- The sort and the filters are the server's. The toolbar offers the sort
  options and flags the backend serves and filters by dominant class,
  detector class, size range, validation state, flags and needs review. The
  choice goes out as query parameters, stays in the URL and the session, and
  **Clear filters** resets it. Cards also show the detector's top class with
  its share.
- Search by embedding similarity (semantic search, when enabled) or literal
  OCR text. A **Crops / Regions** switch next to the search box applies to
  text, **More like this** and picture search; with Regions each result
  names the region that matched.
- **Search by image** (semantic search only) takes a dropped or chosen
  picture, optionally with a region dragged on its preview, and lists the
  nearest crops.
- **More like this** (the `similar` action on a crop card, shown when semantic search is
  enabled) lists the crops nearest to that crop, with their match scores, on the Clusters page
  (`?similar_to=<crop id>`). A crop without a vector shows the server's message instead. Choosing
  it on a result re-runs the search from that crop; "Back to clusters" leaves it.
- The **auto-label loop** panel runs the VLM over the whole project: a dry run
  first, then live status and budget meters, the node table, cancel and
  resume (also after a run that stopped because every VLM call failed), and a
  two-step rollback. Clusters the budget could not cover are shown as not
  visited, each with the backend's reason. See the
  [Auto-label loop](../../guides/auto-label-loop.mdx) guide.
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

<Screenshot name="cropwright/clusters-1600.png" alt="Cropwright cluster grid" caption="Cluster grid with cohesion badges" />

<Screenshot name="cropwright/wheels-inventory-card-1600.png" alt="Cluster grid with a Wheels inventory card pinned first, showing the number of region items listed" caption="Wheels inventory card — pinned first, it opens the slot's region gallery" />

The filter bar above the grid, and the **Matching items** mode that lists and
bulk-edits everything the filter matches, are described in
[Item filter and Matching items](./item-filter.md).

## Cluster detail (`/p/<project>/clusters/[id]`)

The core triage surface: a crop grid for one cluster with pointer-based
drag and drop onto class rows, bulk select/confirm/move/discard/ignore,
per-class hotkeys, **Assist…** (ask the VLM about the selection, the cluster
or the open sub-cluster, with a per-run endpoint and a proposal for the
cluster's label, see [Assist (VLM)](./assist.md)) and **Accept VLM** for
the page, **Reprocess** for the selection (see
[Reprocess](./dataset-import.md#reprocess)), **Refine**
(sub-cluster with agglomerative clustering; a sub-cluster tab lists that
sub-cluster's members from the server), a strategy bar (sort, score
chips, diverse-selection overlay) and a core-member cut line.

A facet bar on the page shows the served counts by detector class, validation,
review tab and detector-confidence band, and the crops can be sorted by
detector confidence.

See [Keyboard shortcuts](./keyboard-shortcuts.md) for the full key table.

## Crop detail

The info button on any crop opens its full provenance: the source image
with every item and region box drawn client-side (toggleable), the crop
itself, class/label-source/confidence/cluster metadata, OCR text lines,
label-write history, and sibling crops from the same source image. It also
shows provenance when recorded (import, combine origin, VLM endpoint, model
and prompt pack), and a **lock** icon on a label a human has set.

A thumbnail whose source file cannot be read shows "Source not mounted" or
"Image file missing" with the backend's message, instead of only a dimmed tile.

## Label operations (`/p/<project>/label-ops`)

**Label ops** in the top bar lists the bulk labels and moves the backend has
recorded for the project, newest first (**Load older** pages back through the
kept history). **Details** opens an operation with its served fields. **Undo**
first shows the backend's dry run (what it would restore, what it would keep,
the conflicts); only **Confirm undo** changes anything.
