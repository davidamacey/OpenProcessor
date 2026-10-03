---
sidebar_position: 4
title: Dataset import and Reprocess
---

# Dataset import and Reprocess (`/p/<project>/datasets`)

Bring an **already-labeled** dataset into the current project, review what it
brought, and re-run the backend's machine proposals on items without
overwriting human work. These pages exist only when the backend supports
dataset import; otherwise there is no link to them and nothing is requested.
Compare with [Ingest](./ingest.md), which brings in unlabeled images.

## Formats

The backend decides which formats it can read and serves the list. A typical
backend serves **Detect automatically**, **YOLO** (a `data.yaml` layout),
**COCO JSON** and an **OpenProcessor export** (the dataset the app's own
[Export](./export.md) page writes). The wizard only offers what the backend
serves.

## The import wizard (`/datasets/import`)

1. **Source.** Type a folder path the backend can read (the allowed roots
   are listed under the field, and the preview tells you if the path isn't
   usable), or upload an archive. An archive larger than the smaller of the
   backend's own limit and the deployment's `CROPWRIGHT_DATASET_UPLOAD_MAX_MB`
   is refused before it is sent. Pick a format, or leave **Detect
   automatically**.
2. **Preview.** A moment after any change, the backend previews exactly what
   you've chosen: splits, totals, an estimate, facts about an OpenProcessor
   export (such as whether a test split is present), and a list of issues
   grouped by severity with samples (file and line). If an issue blocks the
   import, Start is disabled; a **Start anyway** checkbox appears only when
   the backend says forcing is allowed for those issues.
3. **Class mapping.** One row per class in the dataset, matched to your
   project's classes **by name only**. Each row shows the dataset class, its
   box and image counts, the backend's suggestion and how it matched, and the
   target the current choices resolve to. For each row choose to map it to an
   existing class, create a new class, treat it as a region, or skip it. A
   row that has boxes but no target yet is highlighted. **Use suggestion**
   copies the suggestion into a row; **Accept suggestions** lets the backend
   fill every row it can; and **Use the mapping from a previous import**
   copies an earlier completed import's choices by class name. Two dataset
   classes merge by picking the same target class on both rows.
4. **Options.** What to do with each image (import as-is, or also run the
   backend's detectors and VLM, with each mode's description shown), how far
   to trust the imported labels, how to treat region parents, whether to
   freeze the dataset's test split as the project's holdout, and some
   advanced settings. An option you leave alone is not sent, so the
   backend's own default applies.
5. **Start.** A confirmation shows the previewed totals and the request is
   tied to that preview. Refusals use the backend's own words: an
   already-imported dataset links to the earlier job, an interrupted one
   offers **Resume**, a dataset that changed on disk asks you to preview
   again, and an incomplete mapping highlights the rows still missing.

To import into a brand-new project, create the project first on
[`/projects`](./projects.md); the wizard always imports into the project
you're in.

## Imports and the job view

`/datasets/imports` lists past imports with their status and a **New import**
button. An import's page (`/datasets/imports/<id>`) shows:

- the status, a progress bar with throughput and ETA, and whatever the job is
  waiting for;
- the report counts, the issues and the per-image entries (paged, filterable);
- the class mapping the job actually used;
- the error, if the job failed;
- **Next steps** the backend suggests, each behind a confirmation;
- **Review imported labels**, which opens the [Imported tab](./review.md#imported-labels).

The page follows the job on the backend's own polling cadence and wakes
immediately when the backend announces progress. **Cancel**, **Resume** and
**Undo** each ask for confirmation, and are offered only when the backend says
they are allowed; when one is not, the backend's reason is listed under the
buttons. **Undo** does a dry run first and shows
what would change, including that **your own edits are kept**, then lets you
choose whether to also remove the imported images and deprecate the classes
the import created.

## Reprocess

**Reprocess** re-runs machine proposals on items that are already in the
project. It respects the lock rule: anything a human has labeled or edited is
locked and is skipped, not overwritten, and the result counts how many items
were skipped for that reason.

- **One item.** In an item's **Details** panel, choose **Reprocess**, tick
  the scopes to re-run (detect, region, VLM, embed; for regions, redetect or
  reverify), and confirm. The updated item replaces what you were looking at.
- **One image.** On a card's expanded view and on a region card, **Reprocess
  image…** re-runs the proposals for every item cut from that image.
- **A selection.** On `/clusters/<id>`, select crops and use **Reprocess** in
  the toolbar. The backend does a dry run first, you see the per-scope counts
  (selected, locked and skipped, queued, failed, not found), and then confirm
  to apply. A large request becomes a job you can follow and cancel.
- **After a profile change.** Activating a region profile can suggest a
  reprocess; see [Region profiles](./region-profiles.md#activation).

## Lock badges

A **lock** icon marks a label (on a crop card) or a box (on a region card or
in the box editor) that a human has set and the backend will not overwrite.
Hover it to read the backend's lock rule (what makes a label locked). The
Reprocess scope names come from the backend as well.

## Import provenance

An item's **Details** panel shows how an imported item got here: whether its
label is locked, the dataset split, which import(s) brought it (linking to the
import), when, what the import proposed, and whether it sits on a negative
frame. A row appears only when it has a value.

<Screenshot name="import-wizard-1600.png" alt="Dataset import wizard with a preview and a class mapping table" caption="Import wizard — served preview and by-name class mapping" />

<Screenshot name="import-job-1600.png" alt="Dataset import job view with progress, report counts and next steps" caption="Import job — progress, report, and Undo with a dry run" />

<Screenshot name="reprocess-dialog-1600.png" alt="Reprocess dialog with scopes and the locked-and-skipped counts" caption="Reprocess — scopes, dry run and locked-and-skipped counts" />
