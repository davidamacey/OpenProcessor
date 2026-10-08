---
sidebar_position: 8
title: Multi-box regions
---

# Multi-box regions

An item can carry **any number of region boxes**, not just one: a part with
several tags, a vehicle with a front and a rear plate, a surface with several
defects. Every box has its own state, score, detector, verdict and (when the
profile reads text) text. There is no client-side limit on how many boxes an
item has; the only cap is the backend's per-write limit on how many boxes one
save may carry, which is shown as "N / max" next to **Add box**.

## Box states

Boxes are drawn in the colour of their state, using the vocabulary the backend
serves: accepted boxes are green, proposed boxes amber, and rejected or
false-positive boxes dashed grey. Cards, the details panel and the source
image overlay all show every box.

## Reviewing a region item

On the region tab of [Review](./review.md):

| Action | Default key | Effect |
| --- | --- | --- |
| Accept selected box | `Y` | Saves that box as accepted immediately; the queue does not advance. |
| Reject selected box | `R` | Saves that box as rejected immediately; the queue does not advance. |
| Select next box | `Tab` | Cycles through the item's boxes. |
| Confirm | `Enter` | Confirms every box that is still **proposed**, and any pending geometry edit, in one write. Boxes you already rejected or marked false-positive are left as they are. |

A whole-item confirm never overrides a per-box decision you made. The
on-screen buttons do the same as the keys. The keys are
[configurable](./keyboard-shortcuts.md#customizing-shortcuts), and a key
reserved by an action can't be bound to a class.

## Editing boxes

Edit mode (`E`) lets you select, move, resize, **add** and **delete** boxes
on the canvas, with the arrow keys nudging the selected box and `[` / `]`
adjusting its edge. Save writes the whole set in one request; cancel
discards your changes. The same editor opens from the pencil on a crop card,
where saving doesn't imply confirming anything. A **lock** icon marks a box a
human set that the backend won't overwrite.

Saves are checked against the item's current revision. If the item changed
under you (another reviewer, a reprocess), the backend rejects the write,
Cropwright adopts the current item it returns, and you retry from there. An
item whose set of boxes is incomplete shows a chip.

## The region gallery

The region gallery on `/clusters` shows item and box counts side by side, and
adds a **Box state** filter. Opening a region cluster and triaging in bulk
changes the **box** state of the selected boxes, not their sibling boxes on
the same item. A "rows truncated" chip appears when the backend cut the
listing short.

<Screenshot name="review-regions-multibox-1600.png" alt="Region review with several boxes on one item in different states" caption="Region review — several boxes on one item, each with its own state" />

<Screenshot name="box-editor-1600.png" alt="Multi-box editor with a selected box, Add box and the N of max counter" caption="Box editor — select, move, add and delete boxes" />

<Screenshot name="region-gallery-boxes-1600.png" alt="Region gallery with the boxes-listed versus items count in the toolbar" caption="Region gallery — read the toolbar count as boxes listed versus items (12 / 12 boxes listed, 8 items)" />
