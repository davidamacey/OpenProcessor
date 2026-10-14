---
sidebar_position: 5
title: Accuracy audit
---

# Accuracy audit (`/p/<project>/audit`)

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

## 1. Draw a sample

1. Open **Accuracy audit** in the top bar.
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

## 2. Label the sampled crops

The **queue** at the bottom of the page lists the drawn crops that still wait for
a human label. Each row links to [Review](./review.md) on that crop. Confirm or
change the class there. Your label is what measures the detector and the VLM,
and automated writers never overwrite it afterwards.

## 3. Read the report

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

## Why it matters

Automatic promotion of cluster labels is gated on this audit. A class needs enough
audited crops and a high enough detector precision before
`POST /curation/projects/{project}/clusters/auto_promote` will validate it; otherwise
the backend answers `409 audit_required` or `409 audit_precision_low`. Cropwright
has no screen for that call, so run the audit here first and let the backend
promote. See [Clusters](./clusters.md) and the
[curation workflow](../../operations/curation-workflow.mdx#4-cluster).

## Troubleshooting

| Symptom | Cause |
| --- | --- |
| The page shows an error and **Retry** | The report could not be read from the backend. Check the status chip, then click **Retry**. |
| "Draw sample" shows `audit_no_candidates` | No unvalidated machine-labelled crop with a detector class exists. Ingest and let the detector run first. |
| Report shows 0 labeled | You drew a sample but have not labeled any of it yet. |
