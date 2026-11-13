---
sidebar_position: 5
title: Quality
---

# Quality (`/p/<project>/quality`)

**Data quality** is the last look at the dataset before you train. It has three
tabs, each with its own address so you can link to it:

- **Ready to train** (`/quality/ready`, the default): a checklist of enough
  checked examples per class, a fair test set, machine labels nobody has
  checked, the accuracy audit and the near-duplicate groups. Each line says
  whether it blocks a run or only warns, in words and an icon, and links to the
  items or page that need work. The backend serves every line
  (`GET /curation/projects/{project}/quality/readiness`).
- **Accuracy audit** (`/quality/audit`): how often the detector and the VLM are right.
- **Label history** (`/quality/history`): the bulk labels and moves, and undo.

Open **Data quality** in the top bar. `/quality` alone opens Ready to train.
Near-duplicates are no longer a tab here: see
[Near-duplicates](#near-duplicates) below.

## Accuracy audit

The audit answers one question: how often are the detector and the VLM right on
this project? A VLM label is only a suggestion until a human confirms it, so the
page draws a sample of machine-labelled crops, you label them, and the backend
reports precision per class.

Everything on the page is served by the backend. Cropwright does not count,
average or judge anything. The audit routes are `POST /curation/projects/{project}/audit/start`,
`GET /curation/projects/{project}/audit/report`; the drawn crops are the `audit` queue of Review (`GET /curation/projects/{project}/review/{tab}` with `audit`);
how the sample is drawn and what the report means is in the
[curation workflow](../../operations/curation-workflow.mdx#5-label-vlm-suggestions-and-review-queues).

<Screenshot name="cropwright/audit-1600.png" alt="Accuracy audit page with the draw-sample form, detector and VLM precision tables, the confusion matrix and the link to the audit queue" caption="Accuracy audit: draw a sample, label it, read the precision tables" />

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

**Label the sample in Review** at the bottom of the page opens the **Audit sample**
queue of [Review](./review.md), the drawn crops that still wait for a human label.
The queue exists only while a sample is pending. Confirm or change the class
there. Your label is what measures the detector and the VLM,
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

Near-duplicates are photos that look almost the same. Nothing is ever deleted:
they are grouped so the Gallery can show one photo per group and so a test
photo never has a near-copy in training. Collapsing near-duplicates in an
export or a training run is a choice you make for that run and is off unless
you turn it on.

Working with them happens in two places:

- **Workspace**: the Gallery's Duplicates view (`near_dup=collapsed`) shows one
  card per group, and the details panel of an image opens its group.
- **Settings › Near-duplicate profiles**: the thresholds (the profile
  documents), then the **Runs** (status per level, **Full rebuild**, history),
  the **Pinned pairs** you decided and the object-level **leak groups**, each
  linking to the matching items in Detections.

The grouping, the cuts and every count come from the backend
(`/curation/projects/{project}/near_duplicates/*` and `/near_dup_profiles`).
Both levels need a **Full rebuild** after a threshold change; nothing regroups on
its own.

## Label history

The **Label history** tab lists the bulk labels and moves the backend has
recorded for the project, newest first (**Load older** pages back through the
kept history). **Details** opens an operation with its served fields. A
label operation on many items is accepted as a background job (202); the page
follows it, and **Undo** targets the operation by its id. **Undo**
first shows the backend's dry run (what it would restore, what it would keep,
the conflicts); only **Confirm undo** changes anything.
