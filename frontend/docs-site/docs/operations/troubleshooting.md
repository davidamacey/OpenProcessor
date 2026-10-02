---
sidebar_position: 4
title: Troubleshooting
---

# Troubleshooting

## The status chip stays red / "API OK" never shows

- Confirm `PUBLIC_API_PREFIX` matches the backend's own `OP_API_PREFIX`
  exactly.
- Confirm `API_UPSTREAM` resolves — Cropwright's container must be on the
  same docker network as the backend (`OP_DOCKER_NETWORK`).
- `curl http://localhost:<CROPWRIGHT_PORT>/curation/health` (swap in your
  prefix) from the host to bypass the browser entirely.

## A route/section I expect isn't showing up

Nearly every optional feature is gated on the backend advertising it at
runtime (see [Backend feature flags](../configuration/backend-feature-flags.md)).
Absent is the expected behavior for a backend that hasn't enabled that
feature — it is not a bug.

## `/ingest` shows only an error

The page renders from the served `GET {API_PREFIX}/ingest/config`; the
error line is that read's own failure. Check the backend is reachable
and on a current OpenProcessor release.

## Upload fails with a 413

Either the server-path batch's own item cap, or nginx's
`client_max_body_size` (`CROPWRIGHT_INGEST_MAX_REQUEST_MB`). The upload
run controller retries once at half the chunk size on the backend's own
JSON 413; an nginx HTML 413 stops the run outright rather than retrying —
raise the cap and re-run.

## A class hotkey binding was silently rejected

The letter is in the reserved set (global actions + every registered
slot's keymap). Pick a different letter — `/classes` shows the backend's
detail text verbatim when a bind is rejected server-side.

## A page I expect (Datasets, Prompt packs, Region profiles, Models registry, Combine) isn't there

Each is probed once and shown only when the backend serves it (see
[Backend feature flags](../configuration/backend-feature-flags.md)). A
backend that predates the feature, or has it unmounted, gives exactly this
result. A probe that fails for any other reason shows an error with a
**Retry** instead.

## "Project not found" or "not available"

The URL's slug isn't in the backend's project list, or the project is
building, failed or being deleted. Use the links on that page to reach
`/projects` or the default project. Nothing project-scoped is requested
until a project is selectable.

## A region screen didn't change after I activated a profile

Region screens aren't hot-swapped. Cropwright shows a **reload to apply**
notice after an activation, rollback or turn-off; reload the page.

## A save was refused with a conflict

Prompt packs, region profiles, VLM endpoints, keymaps, project edits and box
edits are all saved against the revision you loaded. Someone (or something,
such as a reprocess) changed it first. Reload, or choose **Keep my edits**
where offered, then save again.

## Activating a VLM endpoint is refused

An endpoint that sends crops outside your deployment needs the
acknowledgement checkbox ticked, and a deployment-wide default needs one
recorded first. Use **Settings → Models** to acknowledge. See
[VLM models](../user-guide/vlm-models.md#sending-images-outside-your-deployment).

## A dataset archive upload fails

A file larger than the smaller of the backend's limit and
`CROPWRIGHT_DATASET_UPLOAD_MAX_MB` is refused before it is sent. Raise the
env var and restart the container, or import from a folder path the backend
can read.

## Reprocess skipped items

Items a human has labeled or edited are locked and skipped by design; the
result shows how many.
