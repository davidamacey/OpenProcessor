---
sidebar_position: 4
title: Map
---

# Map (Workspace view)

**Map** is a view of the [Workspace](./workspace.md): it shows where the photos
that match the current filter and queue were taken. It appears in the view bar
only when the project has photos with a stored position and a location policy
that serves them. With the policy `drop` no position is stored, so the view
says the project does not store where photos were taken.

<Screenshot name="cropwright/map-view-1600.png" alt="The Map view with cells and the cell details" caption="Cells over the map; the details panel shows the selected cell" />

## What the header says

The server counts the photos: for example **1,920 located** and **88 without
location**. The second count is a link; it narrows the Workspace to photos
without a position. The header also states the location policy: **exact**, or
**coarse (about 1 km)** when the policy is `store_coarse`.

## Cells and pins

The server groups the matching photos into grid cells for the visible area, so
a large project never sends its points. Colour and size grow with the count.
Where a cell holds exactly one photo at the finest zoom the policy allows, it
is drawn as a **pin**; clicking a pin opens that photo in the details panel.
If more cells match than fit, the view says so: zoom in or narrow the filter.
Panning and zooming never change the filter.

## A cell

Click a cell to focus it. The details panel shows the server's sentence about
the photos in it (for example "9 detections in 14 images") and its first 12
image cards. **Show in Gallery** applies the cell's area as a filter and
switches the view; **Close** leaves the cell.

## Selecting an area

- **Shift+drag** draws a rectangle.
- **Lasso** (the button on the map) then drag draws a free shape.

The area becomes a chip in the filter row (**Area: box** or **Area: drawn
shape**) and applies to every view, so Gallery, Clusters, Detections and Table
show only what was taken there. Remove the chip to clear it. A new area
replaces the previous one.

## Keys

The Map has no letter keys, so a class hotkey can never label an area by
accident. The Workspace keys (`[` and `]` for the view, `\` for the details)
work as in every view.

## Without WebGL2

The map needs WebGL2. If the browser cannot start it, the view says so and
still shows the counts; the rest of the Workspace works.
