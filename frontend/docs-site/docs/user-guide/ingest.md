---
sidebar_position: 2
title: Ingest
---

# Ingest (`/ingest`)

Bring images into the pool. Absent entirely when the backend doesn't mount
the ingest router.

## Browser upload

Files, folders, and drag-and-drop, chunked to the backend's served
per-request cap with bounded concurrency (default 2 in flight). Already-
indexed identifiers are pre-filtered via a lookup call before anything
uploads. Each file's result — ingested, duplicate, or failed with a stable
error code and the backend's own reason — shows in a table with pause,
resume and cancel.

## Server-path batch ingest

Shown only when the backend advertises at least one configured source root.
Points at a folder the backend already has mounted (for example,
OpenProcessor's sample-data output) rather than uploading bytes through the
browser.

## Ingest status and region drain

A status table by source, and — with a served region profile — a
region-drain panel showing whether the backend has finished processing
everything ingested so far. The clustering handoff below waits for that
served "drained" verdict before treating the pool as ready.

## Clustering handoff

Reuses the same `AutoLabelPanel` as the dashboard, gated on the drain
verdict above.

<Screenshot name="ingest-1600.png" alt="Cropwright ingest page" caption="Ingest — upload run panel and status table" />
