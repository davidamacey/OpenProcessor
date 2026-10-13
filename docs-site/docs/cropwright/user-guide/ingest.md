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
- Server-path ingest (bulk ingest and the paste-a-few-paths panel) only appears
  when the backend both enables it and has at least one configured source root.

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

## Bulk ingest (folder or manifest)

Starts a background run over a folder, or a manifest listing image paths, that
the backend already has mounted under a source root. The roots are listed
above the form. Mounting a share, trial runs, timing and disk are in the
[Ingest a large folder](../../guides/ingest-large-folder.mdx) guide.

1. Choose **Folder** or **Manifest**.
2. Enter the path as the backend sees it (it starts with the root, for example
   `/data/source`).
3. Press **Start bulk ingest**. **Advanced** has file extensions, images per
   batch and batches in flight; the placeholders are the backend's defaults.

The run belongs to the backend, the same one `scripts/curation/bulk_ingest.py`
drives, so it keeps going if you leave the page and the page picks it up when
you return. While it runs the panel shows what the backend serves: images found
so far (a folder is scanned while it runs), ingested, duplicate and failed
counts, the speed over the last 30 seconds and over the whole run, the time left
and a "Where the time went" table of seconds per stage.

- **Cancel** stops after the batches in flight; images already stored stay
  stored.
- **Resume** appears after a cancel, a backend restart or a run that finished
  with failed images. It starts the same source again and skips what is already
  stored.
- Failed images are listed with the reason for each.
- Only one run per project can be live; a second start shows the backend's
  answer.

<Screenshot name="cropwright/ingest-bulk-panel-1600.png" alt="Cropwright ingest page with the bulk ingest panel and the list of source roots" caption="The bulk ingest panel: the backend's source roots, folder or manifest, and Start" />

<Screenshot name="cropwright/ingest-bulk-run-1600.png" alt="A bulk ingest run in progress with counts, speed and the stage table" caption="A run in progress" />

## Server-path batch ingest

Points at a folder the backend already has mounted (for example,
OpenProcessor's sample-data output) rather than uploading bytes through the
browser.

## Importing an already-labeled dataset

When the backend supports it, a link on this page opens the
[dataset import wizard](./dataset-import.md), for images that already come
with labels.

## Detector and ingest policy

When the backend reports a detector, the page shows a card for it, and a
policy decides what an ingest keeps and which detections get a vector. See
[Detector and ingest policy](./ingest-policy.md). The run and batch panels
also show how many detections were embedded, not embedded or filtered out.

## Ingest status and region drain

A status table by source, and — with a served region profile — a
region-drain panel showing whether the backend has finished processing
everything ingested so far. The clustering handoff below waits for that
served "drained" verdict before treating the pool as ready.

<Screenshot name="cropwright/region-stage-panel-1600.png" alt="Region detection worklog and Region stage panel on the ingest page with a Pause button" caption="Region stage — the worklog counts, the running or paused state and the Pause control" />

## Clustering handoff

Reuses the same `AutoLabelPanel` as the dashboard, gated on the drain
verdict above.

<Screenshot name="cropwright/ingest-1600.png" alt="Cropwright ingest page" caption="Ingest — upload run panel and status table" />
