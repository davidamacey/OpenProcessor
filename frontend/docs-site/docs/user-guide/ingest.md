---
sidebar_position: 2
title: Ingest
---

# Ingest (`/p/<project>/ingest`)

Bring images into the current project's pool. The ingest router is always
mounted, so the nav link always appears — the page itself reads the
backend's served ingest config before showing any upload UI.

## Served on/off switches

Both upload paths are switches the backend controls per project, not
something Cropwright decides on its own:

- If browser upload is disabled on the deployment, the upload panel is
  replaced with a single line saying so.
- Server-path batch ingest only appears when the backend both enables it
  and has at least one configured source root.

## Browser upload

Files, folders, and drag-and-drop, chunked to the backend's served
per-request cap with bounded concurrency (default 2 in flight). Already-
indexed identifiers are pre-filtered via a lookup call before anything
uploads. Each file's result — ingested, duplicate, or failed with a stable
error code and the backend's own reason — shows in a table with pause,
resume and cancel.

An amber "uploads aren't kept" banner appears only when the backend reports
that it doesn't persist uploaded bytes server-side. A stock deployment
persists them (content-addressed), so this banner is normally absent.

## Server-path batch ingest

Points at a folder the backend already has mounted (for example,
OpenProcessor's sample-data output) rather than uploading bytes through the
browser.

## Importing an already-labeled dataset

When the backend supports it, a link on this page opens the
[dataset import wizard](./dataset-import.md), for images that already come
with labels.

## Ingest status and region drain

A status table by source, and — with a served region profile — a
region-drain panel showing whether the backend has finished processing
everything ingested so far. The clustering handoff below waits for that
served "drained" verdict before treating the pool as ready.

## Clustering handoff

Reuses the same `AutoLabelPanel` as the dashboard, gated on the drain
verdict above.

<Screenshot name="ingest-1600.png" alt="Cropwright ingest page" caption="Ingest — upload run panel and status table" />
