---
sidebar_position: 1
title: Dashboard
---

# Dashboard (`/p/<project>/dashboard`)

Pipeline health at a glance.

- **Headline numbers**: objects, validated, detector labels not yet checked and
  objects with no class, from the live dataset stats.
- **Next steps**: the curation loop in the order the backend serves it (ingest,
  classes, cluster, activate, label, hotkeys, near-duplicates, export, train, review
  disagreements, audit).
  A done step is one muted line; the first undone step is open with the one
  primary button, and each step links to the page that does it. The empty Workspace shows the same
  list. The backend decides which steps exist and which are done.
- **Refresh** is in the page header. The YOLO export is on **Model › Export**.
- **Clusters**: says which partition is serving (clusters, items, method, when it was
  fitted) and starts a clustering run with **Cluster items**. A finished partition that is
  not serving shows **Activate it**.
- **Auto-label**: **Auto-label now** (asks for confirmation) embeds the items that have no
  vector when **Embed missing vectors first** is ticked, optionally runs the VLM over the
  unvalidated items (**Run VLM labeling stage**, off by default), then finalizes. The
  classifier's confident labels and human validations are never overwritten. A run shows its
  stage progress and can be paused, resumed or cancelled. Grouping items into clusters is the
  separate Clusters run above.
- **Assist scope** (`AssistScopeBar`) — when the backend advertises a usable
  VLM prompt pack, an optional per-class scope lets you point the
  VLM-assisted sweep at a single class instead of the whole pool. Absent
  when the backend doesn't advertise one.
- The **Assist with the VLM** quick action links to the Workspace Clusters view: a VLM run on a
  cluster, selection, crop or box is started from there with
  **Assist…** (see [Assist](./assist.md)). With no VLM active for the project
  it links to the Models settings instead.
- Below the stats: the class balance chart, the last 20 recent labels and,
  with a served region profile, a detections panel with region coverage.

<Screenshot name="cropwright/dashboard-1600.png" alt="Cropwright dashboard" caption="Dashboard — live stats and the clustering and auto-label controls" />

`/` itself is not a route — it redirects to the default project's dashboard,
`/p/<default-project>/dashboard`. See [Projects](./projects.md) for how the
active project is chosen and how old bare URLs (without a `/p/<project>`
prefix) resolve.

## Embedding and detections

The stats include an **Embedding** card, and a **Detections summary** panel
below them, with an **Embed N detections** action when the backend suggests
one. See [Embedding state and detections summary](./embedding-state.md).
