---
sidebar_position: 3
title: Detector and ingest policy
---

# Detector and ingest policy

The backend can run a generic object detector when images are ingested, and
a policy decides what an ingest keeps and which detections get an embedding
vector. This page covers where you see that and change it. Every number,
list and refusal shown is served by the backend; Cropwright computes none of
them.

## The detector card (`/ingest`)

The [Ingest](./ingest.md) page shows a card for the served detector: model,
version, input size, whether it assigns classes, and a collapsed table of its
labels. It also gives a one-line summary of the current policy with a link to
the policy page. With no detector reported, the card reads "No detector
reported."

<Screenshot name="ingest-1600.png" alt="Cropwright ingest page" caption="Ingest page, where the detector card and the policy link appear" />

## The ingest policy page (`/settings/ingest-policy`)

The form starts from the policy the backend serves.

- **Embedding mode**, always open: `all` (every detection), `selected` (only
  those matching your criteria) or `lazy` (none until asked for).
- **Detect filter**, collapsed under advanced: classes, excluded classes,
  confidence, area, a per-image cap and how classes are resolved.
- **Per-project detector override**, also advanced.

Nothing is checked in the browser, so for example `selected` mode with no
criterion is refused by the backend when you save.

### Cost preview

A moment after any edit the draft goes to the backend, which answers "N of M
stored detections would be embedded, about X MB". The panel says outright
that this describes detections **already stored** and that a policy change
only affects **future** ingests. When the backend sampled, it says
"estimated from S detections".

<Screenshot name="ingest-policy-preview-1600.png" alt="Ingest policy page in selected mode with the cost preview table of would-embed and would-not counts per class" caption="Ingest policy — selected mode, and the served cost preview: how many stored detections would be embedded and about how much space" />

### Saving

**Save** asks for confirmation, then writes the policy. If someone else saved
first you are offered **Reload** (drops your edits) or **Keep my edits**
(rebases them on the latest policy). A list of unknown class names is shown
as a warning; the policy is saved anyway. A detector that cannot be served
shows the backend's message and reasons.

## Create classes from the detector (`/classes`)

When a detector is reported, **Classes** has a collapsed panel that creates
classes from the detector's labels. Pick labels (none means all), run
**Preview** (a dry run), read which classes would be created, skipped or
conflict, and only then confirm **Create N classes**.

See [Embedding state and detections summary](./embedding-state.md) for what
you can see afterwards, and how to fill in missing vectors.
