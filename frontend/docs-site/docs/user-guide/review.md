---
sidebar_position: 4
title: Review
---

# Review queue (`/p/<project>/review`)

A one-item-at-a-time, keyboard-driven queue.

## Tabs

**All**, **Uncertainty**, **Model Disagreements**, **Classifier Blind
Spots**, **New Class Proposals**, an **Imported** tab (only when the backend
serves it), and one region tab (labelled by the
backend's own display name) when a region profile is served. The All tab
adds quick-filter preset chips (VLM mismatches, VLM low-confidence, primary
low-confidence) layered on top of the plain All view.

## Filters and deep links

Class, source, confidence, and any per-tab enum filter the backend
advertises (e.g. a region-status filter on the region tab) resolve
server-side. `?crop_id=` deep links jump straight to the item's real
position in the queue, or show the backend's reason it isn't there.

The shared item filter (class, area, origin, embedding and review state) and
the backend's per-tab filters are described in
[Item filter and Matching items](./item-filter.md).

## Empty queues

An empty queue shows the served reason (e.g. "no probe predictions — run a
probe") and a direct link to the step that fills it.

<Screenshot name="review-1600.png" alt="Cropwright review queue" caption="Review — one item at a time, keyboard-first" />

<Screenshot name="review-regions-1600.png" alt="Cropwright region review tab" caption="Region tab — region box, provenance chips and text reading" />

## Imported labels

After a [dataset import](./dataset-import.md), the **Imported** tab lists
items that came from imports, with filters for the dataset split and for items
on negative frames. It opens from the import's **Review imported labels**
link already filtered to that import, shown as a removable chip ("Import
`<id>`"). An empty Imported tab links to the import page. A similar chip,
**Combine conflicts only**, appears when you arrive from a
[combine job](./combine-projects.md)'s **Review flagged conflicts**. A
**lock** icon on a label or box means a human set it and machine proposals
won't overwrite it.

## Item details

A collapsed **Details** panel shows the item's label history, its source image
with every box drawn over it, and provenance rows: how it was imported, which
project it was combined from, and which VLM endpoint, model and prompt pack
labeled it, each shown only when recorded. **Reprocess** is available there.

## Region tab

Driven entirely by the served region profile: a multi-box editor (see
[Multi-box regions](./multi-box-regions.md)), per-box accept / reject,
confirm and false-positive actions, the backend's chosen text reading with
reader-disagreement and choice badges, a detector-provenance chip strip,
served rejection reasons, and rejected boxes drawn dashed so a human can
accept them.

The tab, its buttons and the related dashboard and `/train` panels take
their names from the profile. The screenshots here come from a backend
running the example license-plate profile, which is why they read
"Plates" and "Confirm Plate"; a profile for another region type (a tag, a
wheel, a defect) renders the same screens under its own name.

See [Keyboard shortcuts](./keyboard-shortcuts.md) for the review-queue key
table.

<Screenshot name="review-imported-1600.png" alt="Review Imported tab with an import filter chip" caption="Imported tab — items from an import, filtered by import" />
