---
sidebar_position: 1
title: Dashboard
---

# Dashboard (`/p/<project>/dashboard`)

Pipeline health at a glance.

- **Live dataset stats**, polled every 10 seconds.
- **Run Clustering Now** — starts an auto-label run with stage progress,
  shared with the backend's own scheduled run.
- **Assist scope** (`AssistScopeBar`) — when the backend advertises a usable
  VLM prompt pack, an optional per-class scope lets you point the
  VLM-assisted sweep at a single class instead of the whole pool. Absent
  when the backend doesn't advertise one.
- With a served region profile, a detections panel shows region coverage.

<Screenshot name="dashboard-1600.png" alt="Cropwright dashboard" caption="Dashboard — live stats and the clustering/assist trigger" />

`/` itself is not a route — it redirects to the default project's dashboard,
`/p/<default-project>/dashboard`. See [Projects](./projects.md) for how the
active project is chosen and how old bare URLs (without a `/p/<project>`
prefix) resolve.

## Embedding and detections

The stats include an **Embedding** card, and a **Detections summary** panel
below them, with an **Embed N detections** action when the backend suggests
one. See [Embedding state and detections summary](./embedding-state.md).
