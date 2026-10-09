---
sidebar_position: 7
title: Item filter and Matching items
---

# Item filter and Matching items

One filter drives three pages: `/clusters`, `/review` and `/export`. Classes
in it are always chosen **by name**.

## What you can filter by

A filter bar offers whichever of these the page and the backend serve; a
control the page doesn't support is absent, not disabled.

- classes, and classes to exclude
- confidence and area bands, and the largest N items per image
- origin, embedding state and review state
- open-vocabulary set and prompt

Nothing is validated in the browser: a malformed band is refused by the
backend and its message is shown under the bar.

## `/clusters`

The bar above the cluster grid filters the clusters (and the semantic search
box uses the same filter). A sidebar class chosen on the left counts only
when the bar names no class.

<Screenshot name="cropwright/clusters-1600.png" alt="Cropwright clusters grid" caption="Clusters grid, with the item-filter bar above it" />

### Matching items

Switch the page to **Matching items** (`?mode=matching`) to list every item
the filter matches, page by page, instead of clusters. A link from an item's
open-vocabulary provenance opens this mode already filtered. You can **Ignore**,
**Restore**, **Label** or **Move** all matching items at once.

Each action first asks the backend for a dry run, and the dialog states how
many items the backend would touch. You can limit the count, sample, and pick
a seed for a random sample. **Apply** repeats the write against the filter as
it was when the dialog opened. A refusal (an empty filter without a limit, or
too many items) is the backend's own message. The written ids go on the undo
stack, so `Z` reverts the whole action.

## `/review`

The filter bar sits in the filter row, limited per tab to what that tab
serves, and the choice is kept in the URL and cleared when you change tab.
Every other filter the backend lists for the tab is drawn from its own
description: a select, chips, a number box or a text box. A select with no
backend default reads **any** and sends nothing; it never shows a value that
isn't applied.

## `/export`

A collapsed **Only items matching a filter** limits the YOLO export to the
filter. While the filter names something, the line "Matching items: N" shows
the backend's own count, and the export request carries the filter only then.
