---
sidebar_position: 4
title: Review
---

# Review queue (`/p/<project>/review`)

A one-item-at-a-time, keyboard-driven queue.

## Tabs

**All**, **Uncertainty**, **Model Disagreements**, **Classifier Blind
Spots**, **New Class Proposals**, and one region tab (labelled by the
backend's own display name) when a region profile is served. The All tab
adds quick-filter preset chips (VLM mismatches, VLM low-confidence, primary
low-confidence) layered on top of the plain All view.

## Filters and deep links

Class, source, confidence, and any per-tab enum filter the backend
advertises (e.g. a region-status filter on the region tab) resolve
server-side. `?crop_id=` deep links jump straight to the item's real
position in the queue, or show the backend's reason it isn't there.

## Empty queues

An empty queue shows the served reason (e.g. "no probe predictions — run a
probe") and a direct link to the step that fills it.

<Screenshot name="review-1600.png" alt="Cropwright review queue" caption="Review — one item at a time, keyboard-first" />

<Screenshot name="review-regions-1600.png" alt="Cropwright region review tab" caption="Region tab — region box, provenance chips and text reading" />

## Region tab

Driven entirely by the served region profile: a sub-box editor, confirm /
reject / false-positive actions, the backend's chosen text reading with
reader-disagreement and choice badges, a detector-provenance chip strip,
served rejection reasons, and verifier-rejected candidates drawn dashed so
a human can accept them.

The tab, its buttons and the related dashboard and `/train` panels take
their names from the profile. The screenshots here come from a backend
running the example license-plate profile, which is why they read
"Plates" and "Confirm Plate"; a profile for another region type (a tag, a
wheel, a defect) renders the same screens under its own name.

See [Keyboard shortcuts](./keyboard-shortcuts.md) for the review-queue key
table.
