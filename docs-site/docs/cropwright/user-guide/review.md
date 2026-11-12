---
sidebar_position: 4
title: Review
---

# Review (a Workspace view)

Review is one item at a time, keyboard-driven, inside the
[Workspace](./workspace.md) (`/p/<project>/workspace?view=review`). It shares
the Workspace's filter row, rail and details panel. There is no separate
`/review` page: that address is gone.

## Queues

The rail lists the review queues the server serves, each with its count under
the current filter. A click on a queue opens it in the Review view.

- **To label**: items with no class yet, highest detector confidence first.
- **Check machine labels**: items whose class a model set and no human has
  confirmed, least confident first.
- **Disagreements**: items where the detector, the VLM or the probe's model
  disagree, most contested first.
- **New class proposals**: items flagged as needing a class that does not exist.
- **Audit sample**: the crops drawn by an [accuracy audit](./audit.md). Listed
  only while a sample is pending; the project's next step points at it
  otherwise.
- A **region** queue, labelled by the region profile's display name, when a
  region profile is active.

A queue is a preset of the shared [item filter](./item-filter.md), so the
class, area, origin and review-state filters narrow it, and any queue-only
filter the server advertises (for example a region-status filter on the region
queue) shows beside it. An empty queue shows the served reason and a direct
link to the step that fills it.

## Sort

The **Sort** menu in the Review header lists the orders the server offers for
the open queue, with the one it applied selected (outlier-first, representativeness,
uncertainty, mistakenness and uniqueness appear once the project has the data
they need). Each queue opens in its own default order, and a sort chosen in one
queue does not carry to another. If the server applied a different order than
the one asked for, it says why beside the menu. The score and blur thresholds
and "hide near-duplicates" are filters the queue serves; they appear beside the
filter row when the queue has them.

<Screenshot name="cropwright/review-1600.png" alt="The Review view on a queue" caption="Review: one item at a time, keyboard-first" />

## From the grid and back

In the Detections view, `Enter` on a focused card opens Review at that item.
`Esc` returns to the grid with the card you reached focused, so a confirmed
item's next neighbour is where you land. A Review opened from a link returns
to the Detections grid. A `focus=item:<id>` in the address jumps straight to
the item's real position in the queue, or shows the server's reason it is not
there.

## Labeling

`Enter` confirms the proposed class and moves on; when there is no proposal it
opens the class picker, the same picker the Workspace uses (`/` opens it at
any time). `D` discards (reversible, not through `Z`), `N` skips, `Z` undoes,
and the arrow keys move between items. [Keyboard
shortcuts](./keyboard-shortcuts.md) has the table.

## More like this

A **More like this** link under the item's details (shown when semantic search
is enabled) opens the Clusters view with the crops nearest to the current one,
so you can label look-alikes together. A crop without a vector shows the
server's message there.

## Item details

The **Details** panel shows the item's label history, its source image with
every box drawn over it, and provenance rows: how it was imported, which
project it was combined from, and which VLM endpoint, model and prompt pack
labeled it, each shown only when recorded. **Reprocess** is available there. A
**lock** icon on a label or box means a human set it and machine proposals
won't overwrite it.

A [dataset import](./dataset-import.md)'s **Review imported labels** link lists
the imported items in the Detections view, narrowed to that import (`import_id`
is part of the shared filter, so Review honours it too). A
[combine job](./combine-projects.md)'s **Review flagged conflicts** opens the
Check machine labels queue with the combine-conflict filter on.

## Region queue

Driven entirely by the served region profile: a multi-box editor (see
[Multi-box regions](./multi-box-regions.md)), per-box accept / reject,
confirm and false-positive actions, the backend's chosen text reading with
reader-disagreement and choice badges, a detector-provenance chip strip,
served rejection reasons, and rejected boxes drawn dashed so a human can
accept them.

The queue, its buttons and the related dashboard and `/train` panels take
their names from the profile. The screenshots here come from a backend
running the example license-plate profile, which is why they read
"Plates" and "Confirm Plate"; a profile for another region type (a tag, a
wheel, a defect) renders the same screens under its own name.

<Screenshot name="cropwright/review-regions-1600.png" alt="The Review view on the region queue" caption="Region queue: region box, provenance chips and text reading" />
