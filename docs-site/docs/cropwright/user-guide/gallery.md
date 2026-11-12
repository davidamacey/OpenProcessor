---
sidebar_position: 2
title: Gallery
---

# Gallery

**Gallery** shows whole images, not crops: the picture with every box drawn on
it, so you can see what came in and find the images that still need work. The
images, their order, their counts and every box coordinate come from the
backend (`/curation/projects/{project}/images`); the page only draws them.

Gallery is the view the Workspace opens on when the project has no partition
yet.

## The cards

Each card shows:

- the thumbnail with the item boxes over it, drawn from the coordinates the
  server sent. Boxes keep their review colour (validated, machine, unlabelled,
  held out, rejected). Labels are hidden on small thumbnails, and a card says
  **+ more boxes** when the image has more items than are drawn;
- three counts: **items**, **labelled** and **unlabelled**. Unlabelled is
  orange when it is above zero;
- a **Near-dup** badge when the image is in a near-duplicate group, with the
  group size.

The order is the one you pick in the sort menu; the options and the default
come from the server.

## Find images that need work

Turn on **Has unlabelled objects** to keep only the images with at least one
item that has no class. The filter bar's other filters (class, review state,
capture date, location) apply to the same list: an image is listed when at
least one of its items matches.

## Select and act

Click a card to select it; Shift-click selects a range and Ctrl or Cmd-click
adds one. A bulk action (assign a class, confirm, ignore) acts on the matching
items of the selected images, and the confirmation shows the item count the
server will change.

## Open an image

**Details** on a card opens the image details panel: the image's facts, its
counts and the items on it. From there, open the image in the editor to draw,
move or delete boxes.
