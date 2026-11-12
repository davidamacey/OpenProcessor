---
sidebar_position: 5
title: Quality
---

# Quality (`/p/<project>/quality`)

**Quality** is the one place for dataset hygiene. It has three tabs, each with
its own address so you can link to it:

- **Accuracy audit** (`/quality/audit`): how often the detector and the VLM are right.
- **Near-duplicates** (`/quality/near-duplicates`): which images are almost the same picture.
- **Label history** (`/quality/history`): the bulk labels and moves, and undo.

Open **Quality** in the top bar. `/quality` alone opens the audit tab.

## Accuracy audit

The audit answers one question: how often are the detector and the VLM right on
this project? A VLM label is only a suggestion until a human confirms it, so the
page draws a sample of machine-labelled crops, you label them, and the backend
reports precision per class.

Everything on the page is served by the backend. Cropwright does not count,
average or judge anything. The audit routes are `POST /curation/projects/{project}/audit/start`,
`GET /curation/projects/{project}/audit/queue` and `GET /curation/projects/{project}/audit/report`;
how the sample is drawn and what the report means is in the
[curation workflow](../../operations/curation-workflow.mdx#5-label-vlm-suggestions-and-review-queues).

<Screenshot name="cropwright/audit-1600.png" alt="Accuracy audit page with the draw-sample form, detector and VLM precision tables, the confusion matrix and the queue of crops waiting for a label" caption="Accuracy audit: draw a sample, label it, read the precision tables" />

### 1. Draw a sample

1. Open **Quality** in the top bar; **Accuracy audit** is the first tab.
2. Under **Draw a sample**, optionally set **Sample size** (the total number of
   crops, your human-time budget) and **Crops per class**. An empty field is not
   sent, so the backend default applies.
3. Click **Draw sample**.

The sample is machine-labelled crops that no human has validated, spread over the
detector's classes. Drawing a sample never changes a class.

Expected result: a line "Drew N of the M requested crops", and one chip per
detector class showing `class: sampled of available`. An amber chip means the
budget left that class fewer crops than **Crops per class**, although more exist.
Raise **Sample size** or lower **Crops per class** and draw again.

If there is nothing to draw, the backend answers `409 audit_no_candidates` and the
form shows its message.

### 2. Label the sampled crops

The **queue** at the bottom of the page lists the drawn crops that still wait for
a human label. Each row links to [Review](./review.md) on that crop. Confirm or
change the class there. Your label is what measures the detector and the VLM,
and automated writers never overwrite it afterwards.

### 3. Read the report

Click **Refresh** after labeling. The page shows:

- **labeled by a human** and **still waiting**: how many drawn crops are done and
  how many are pending, plus the outcome counts (agree, detector wrong, VLM wrong,
  both wrong).
- **Detector precision**: per detector class, how often the human agreed with the
  detector, with a 95% interval.
- **VLM precision**: per class, how often the human agreed with the VLM, over crops
  whose label came from the VLM.
- **Confusion matrix**: rows are the detector's class, columns are the human's
  class, cells are crop counts. A large off-diagonal cell is a class pair the
  detector confuses.

A class with too few audited crops is dimmed and marked as an insufficient sample.
Do not trust its precision yet; label more crops of that class.

### Why it matters

Automatic promotion of cluster labels is gated on this audit. A class needs enough
audited crops and a high enough detector precision before
the cluster auto-promote step will validate it; otherwise
the backend answers `409 audit_required` or `409 audit_precision_low`. Cropwright
has no screen for that call, so run the audit here first and let the backend
promote. See [Workspace](./workspace.md) and the
[curation workflow](../../operations/curation-workflow.mdx#4-cluster).

### Troubleshooting

| Symptom | Cause |
| --- | --- |
| The page shows an error and **Retry** | The report could not be read from the backend. Check the status chip, then click **Retry**. |
| "Draw sample" shows `audit_no_candidates` | No unvalidated machine-labelled crop with a detector class exists. Ingest and let the detector run first. |
| Report shows 0 labeled | You drew a sample but have not labeled any of it yet. |

## Near-duplicates

**Near-duplicates** shows which images (and which crops) are almost the same
picture, so a test split does not leak and an export does not repeat itself.
The grouping, the cuts and every count come from the backend; this page only
shows them and lets you decide pairs. The page reads
`/curation/projects/{project}/near_duplicates/status`, `/runs`, `/groups` and
`/decisions`, and the thresholds from `/near_dup_profiles`.

### Status and runs

Each level has a card:

- **Image level**: whole frames, compared by their image vectors.
- **Item level**: crops inside the image-level leak groups.

A card shows the current run (when it ran, how many groups it made) and, when
the groups are out of date, why: **has items not yet grouped**, **needs a full
rebuild** (the thresholds changed), or **has never been built**.

- **Group new items** appears when only new items are pending. It runs the
  incremental pass.
- **Full rebuild** regroups the whole level.

Both start a job and follow it; the card shows the job's status. If the server
refuses (a run is already active, or the item level needs a current image-level
run first), its message appears under the title and nothing starts.

**Run history** lists the last runs of both levels with their counts, the
profile revision they used, and how many items the test holdout absorbed.

### Groups

Choose the level (**Images** or **Items**) and the kind:

- **Duplicates**: members at or above the duplicate cut. Keep the
  representative and drop the rest, for example in an export that collapses
  near-duplicates.
- **Leak groups**: members at or above the leak cut. They must stay on one
  train/test side, so freezing the test holdout moves a whole group together.

Groups are listed largest first with a picture of the representative, the
lowest cosine between linked members and the lowest cosine to the
representative. **In test holdout** marks a group with a held-out member.
**Load more** pages on.

An empty list means nothing is at or above the cut, or the level has no current
run yet; the status card says which.

### Deciding a pair

Click a group to see every member with its cosine to the representative.

- **Not a duplicate** pins the member apart from the representative.
- **Keep together** pins it to the representative.

**Open in Workspace** in the group dialog opens the Workspace scoped to that
group, to look at its members next to the rest of your items.

A decision is kept for every later run and applies at the next run of that
level (the card then shows items pending). **Undo** appears on a member you
decided in this dialog and removes the decision. The API serves no list of
older decisions, so they cannot be undone from here.

### Advanced: thresholds

The **Advanced: thresholds** panel edits the project's near-duplicate profile:
the duplicate cosine, the leak cosine and the largest group per level, the
neighbour count for images, and "only group crops of the same detector class"
for items. The seeded defaults (0.98 cosine) are shown beside each field, and
**Use the seeded defaults** puts them in the form.

**Save and activate** writes the next revision of the active profile and makes
it active. Both levels then need a **Full rebuild**; nothing regroups on its
own. The server checks the values (the leak cut cannot exceed the duplicate
cut) and its reasons are shown as sent. Few projects need to change them.

## Label history

The **Label history** tab lists the bulk labels and moves the backend has
recorded for the project, newest first (**Load older** pages back through the
kept history). **Details** opens an operation with its served fields. A
label operation on many items is accepted as a background job (202); the page
follows it, and **Undo** targets the operation by its id. **Undo**
first shows the backend's dry run (what it would restore, what it would keep,
the conflicts); only **Confirm undo** changes anything.
