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

## Sources (S3 buckets and watched folders)

The **Sources** panel keeps a project fed from an S3 bucket prefix or a server folder.
What may be read is set on the deployment (the bucket allow-list and credentials live
in the API's environment, see [Ingest from S3 and watched sources](../../guides/ingest-sources.mdx));
Cropwright never asks for or shows a credential. If S3 is not allowed, the form says so
and only **Server folder** is offered.

1. Press **Add a source**, name it, and choose **S3 bucket** (bucket and prefix) or
   **Server folder**.
2. Pick triggers: **Watch every** N seconds, **On a schedule** (five-field cron, UTC),
   and for S3 **Accept S3 event notifications (webhook)**. With none, the source runs
   only when you press **Run now**. **Options** has file extensions, batch size and
   concurrency.
3. Save. Each source then shows what the backend serves: counts (pending, ingested,
   retrying, gave up), last and next poll or fire, the webhook path and its event
   counts, the scan in progress, the last run and the objects that failed with the
   reason.

- **Run now** scans the source and drains its queue. When the backend cannot run it
  (a run of this project is live, S3 not allowed), the button is disabled and the
  backend's reason is shown next to it.
- **Retry the N that gave up** queues failed objects again.
- **Edit** changes the name, triggers and options; the location cannot change.
- **Remove** asks first. Images already ingested stay in the project.

The panel refreshes on the interval the backend serves.

<Screenshot name="cropwright/ingest-sources-panel-1600.png" alt="The Sources panel with one S3 source card and the Add a source form open" caption="The Sources panel: a source card and the create form (S3 bucket, triggers, options)" />

<Screenshot name="cropwright/ingest-source-card-1600.png" alt="A source card with counts, watch and schedule times, the webhook path and event counts, and the last run" caption="A source card as the backend serves it" />

**Verified live.** On the dev stack with a real MinIO and public COCO images: adding an S3
source (bucket `op-s3-test`, prefix `ui/`), **Save** and **Run now** ended with
"Last run completed · ingested 5 · duplicates 0 · failed 0" and counts `pending 0,
ingested 5, retrying 0, gave up 0`. **Remove** asked for confirmation and the 5 images
stayed in the project. With the allow-list narrowed to another bucket the card showed
**Run now** disabled with "This bucket or prefix is outside OP_S3_ALLOWED" and the watch
error with its retry time. The Ingest page at 430 px wide had no horizontal overflow
(`scrollWidth` 430, `clientWidth` 430; 1600/1600 at desktop width). The server-side
checks (idempotent re-run, webhook, poll, cron, a rejected credential, project isolation)
are in [Verified on a live stack](../../guides/ingest-sources.mdx#verified-on-a-live-stack).
On a second pass: a **Server folder** source (path under a source root, watch every 10 s)
ingested its files, picked up a file added later without a click and ingested 0 on **Run now**
again. For objects that gave up (a text file named `.jpg`, an empty file), the card showed **Retry the 2
that gave up**; after the click it read `pending 2, gave up 0` and the button disappeared (the
button only queues them, so a source with no trigger needs **Run now**), and **Run now** ended with
`ingested 2`. The button is not shown when nothing gave up. With S3 off on the deployment the S3
sources show **Run now** disabled with "S3 is off on this deployment: OP_S3_ALLOWED is empty" and
the form offers only **Server folder**; if the list cannot be loaded the panel shows "Could not load
sources: ..." and no cards, and recovers by itself.

<Screenshot name="cropwright/ingest-source-gave-up-1300.png" alt="A source card with gave up 2 and the Retry the 2 that gave up button" caption="A source with objects that gave up: the Retry button" />

<Screenshot name="cropwright/ingest-source-folder-1300.png" alt="A Server folder source card watching every 10 seconds" caption="A Server folder source" />

<Screenshot name="cropwright/ingest-source-s3-off-1300.png" alt="The add source form with S3 off: only Server folder is offered" caption="S3 off on the deployment: the form offers only Server folder" />

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
