---
sidebar_position: 3
title: Workspace
---

# Workspace (`/p/<project>/workspace`)

The Workspace is where you look at a project's items and label them in
batches. One page holds the **filter row**, the **view bar**, the **rail**
(queues and classes), the **view** itself, a **details** panel and a **bulk
bar**. The filter, the queue, the open cluster and the selection stay when
you switch view, and Back returns to the previous view and filter.

<Screenshot name="cropwright/workspace-clusters-1600.png" alt="The Workspace, Clusters view" caption="Clusters view with the rail and the filter row" />

## Views

- **Clusters.** Every cluster as a card: dominant class, **cohesion**, size
  and representative crops. Choosing a card opens the cluster in place (the
  scope chip in the filter row shows which one); **All clusters** or the
  chip's close leaves it. The sort and card filters, similarity and text
  search, the auto-label loop panel, the embedding overlay, the **Ignored**
  list, the region gallery for a region class and the pinned slot inventory
  cards work as before. Filtering by a class lists the clusters whose
  dominant class it is.
- **Detections.** Every item the filter, queue and scope match, as a flat
  grid with the total and what the filter hides.

The view bar lists the views; under 768 px it is a menu. The first visit
opens the view the server suggests for the project's state, and the
**Next** chip in the filter row names the next step of the curation loop.

## Filter row and rail

The filter row is the shared [item filter](./item-filter.md) plus chips for
the open cluster and the queue; under 768 px it sits behind one **Filters**
button. The **rail** lists the review queues with their counts, then the
classes with theirs, as the server serves them. A click on a queue or class
filters every view; drag a card onto a class to assign it. **+ New class**
creates one (a refused name offers the server's suggestion). Under 1024 px
the rail is a drawer (**Queues & classes**).

## Labeling a batch

1. Open a cluster, or pick the Detections view and set the filter.
2. Select the cards (click, Shift+click for a range, `A` for the page).
3. Read the bulk bar: it states what the selection is, in the server's
   words, including how many items are locked and will be skipped.
4. **Assign class…** (or **Move…**, **Ignore** with a reason, **Restore…**).
   Each asks the server for a dry run first and shows the count it would
   change.
5. **Apply** writes it. `Z` or **Undo** reverts the last one.

In a cluster, the bulk bar also offers **Flag**, **Reprocess** (see
[Reprocess](./dataset-import.md#reprocess)), **Accept VLM** and **Assist…**
(see [Assist (VLM)](./assist.md)). A facet bar shows the served counts by
detector class, validation, review tab and detector-confidence band.

The **Details** panel shows the focused detection, else the selection's
served summary, else the open cluster. Under 1280 px it is a drawer.

See [Keyboard shortcuts](./keyboard-shortcuts.md) for the keys.

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
kept history). **Details** opens an operation with its served fields. A
label operation on many items is accepted as a background job (202); the page
follows it, and **Undo** targets the operation by its id. **Undo**
first shows the backend's dry run (what it would restore, what it would keep,
the conflicts); only **Confirm undo** changes anything.

## Map

**Map** is a view in the view bar when the project has photos with a stored
position. See [Map](./map.md).
