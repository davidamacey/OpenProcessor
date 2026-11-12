---
sidebar_position: 5
title: Embedding
---

# Embedding (Workspace view)

**Embedding** is a view of the [Workspace](./workspace.md): a 2-d scatter in which items
that look alike sit close together. It appears in the view bar only when the deployment
serves the embedding projection (the server reports it in `/methods`).

## What the plot shows

Every projected item is a dot. The colours are the server's class legend: one colour per
class, listed under the plot with the number of dots of that class. The projection is
fitted by a background job and only read here: **Rebuild** refits it over the current
pool, and a project that has none yet offers **Build projection**. When the server says
the picture is approximate, the plot says so in a note.

The plot follows the [filter](./item-filter.md). Dots the filter leaves out stay on the
plot but dimmed, so a class filter shows where that class sits and what lies around it.
Dots of one class inside the cloud of another are the likely mislabels.

## Selecting, opening and filtering

- **Drag** to lasso a group of dots. The matching items become the Workspace selection;
  the bulk bar above the plot assigns a class, moves them to a cluster or ignores them,
  and a strip of thumbnails shows what the lasso caught (click one to enlarge it).
- **Click a dot** to open its item in the details panel.
- **Shift+drag**, or **Draw a box** and then drag, draws a box. It becomes the chip
  **Area: embedding box** and filters every view to the items inside it; remove the chip
  to lift it. The box is part of the page address, so it survives a reload and Back.

There are no keyboard shortcuts for the plot.
