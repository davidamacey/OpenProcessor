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

What each field means, the API calls and the cost of each mode are in the
[Ingest policy](../../guides/ingest-policy.mdx) guide. Set the policy before a
[large ingest](../../guides/ingest-large-folder.mdx): it changes future ingests
only.

## The detector card (`/ingest`)

The [Ingest](./ingest.md) page shows a card for the served detector: model,
version, input size, whether it assigns classes, and a collapsed table of its
labels. It also gives a one-line summary of the current policy with a link to
the policy page. With no detector reported, the card reads "No detector
reported."

<Screenshot name="cropwright/ingest-1600.png" alt="Cropwright ingest page" caption="Ingest page, where the detector card and the policy link appear" />

## The ingest policy page (`/settings/ingest-policy`)

The page shows, top to bottom, the **Detector (ingest profile)** card, the
**Ingest profiles** list and the policy form. The form starts from the policy
the backend serves.

- **Embedding mode**, always open: `all` (every detection), `selected` (only
  those matching your criteria), `lazy` (no detection vector until asked for;
  the whole-frame vector of each image is still computed) or `none` (detection
  only: no vector of any kind).
- **Clustering**, a checkbox: "Cluster automatically". Unchecked, new items are not
  placed in clusters and the refresh and idle retrains skip the project.
- **Detect filter**, collapsed under advanced: classes, excluded classes,
  confidence, area, a per-image cap and how classes are resolved.

The detector is not part of the policy. It comes from the project's active
ingest profile (see below).

Nothing is checked in the browser, so for example `selected` mode with no
criterion is refused by the backend when you save.

### Ingest profiles

The detector a project ingests with is chosen by its active ingest profile.

- The **Detector (ingest profile)** card names the active profile and
  revision. With none active it says ingest uses the deployment's default
  detector and names it.
- The **Ingest profiles** list marks the active one with an **Active** chip
  and has **Activate** on the others.
- A refused activation shows the backend's message and each issue it served
  (for example `detector_model_not_ready`, `detector_model_unusable`,
  `triton_unreachable`) as sent. **Activate anyway** appears only when the
  backend says the refusal can be overridden. Warnings on a successful
  activation are listed under "Activated, with warnings."

Creating and editing profiles is done through the API; see the
[Ingest profiles](../../guides/ingest-profiles.mdx) guide.

### Cost preview

A moment after any edit the draft goes to the backend, which answers "N of M
stored detections would be embedded, about X MB". The panel says outright
that this describes detections **already stored** and that a policy change
only affects **future** ingests. When the backend sampled, it says
"estimated from S detections".

<Screenshot name="cropwright/ingest-policy-preview-1600.png" alt="Ingest policy page in selected mode with the cost preview table of would-embed and would-not counts per class" caption="Ingest policy — selected mode, and the served cost preview: how many stored detections would be embedded and about how much space" />

### Saving

**Save policy** asks for confirmation, then writes the policy. If someone else saved
first you are offered **Reload** (drops your edits) or **Keep my edits**
(rebases them on the latest policy). A list of unknown class names is shown
as a warning; the policy is saved anyway. 
## Create classes from the detector (`/classes`)

When a detector is reported, **Classes** has a collapsed panel that creates
classes from the detector's labels. Pick labels (none means all), run
**Preview** (a dry run), read which classes would be created, skipped or
conflict, and only then confirm **Create N classes**.

See [Embedding state and detections summary](./embedding-state.md) for what
you can see afterwards, and how to fill in missing vectors.
