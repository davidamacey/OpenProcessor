---
sidebar_position: 4
title: Embedding state and detections summary
---

# Embedding state and detections summary

Not every detection gets an embedding vector: the
[ingest policy](./ingest-policy.md) can embed all, some or none at ingest
time. An item with no vector cannot be ranked, clustered by similarity or
matched by a text search. This page lists where Cropwright shows that and
how to repair it.

## Where the state appears

- **Crop cards and the detail panel** show a badge when an item has no
  vector, and why: encoder failed, deferred, or not selected by the policy.
  Items with a vector, and items written before the field existed, show no
  badge.
- **Dashboard** has an **Embedding** card in the dataset stats (embedded and
  not embedded, with the per-state breakdown), and a **Detections summary**
  panel below it: totals, an embedding breakdown and a per-label table.
- **Projects** has an "Embedded" column on each row.
- **Ingest** results list, per file and in total, how many detections were
  embedded, not embedded, failed to embed or were filtered out.

<Screenshot name="dashboard-1600.png" alt="Cropwright dashboard" caption="Dashboard, which carries the Embedding card and the Detections summary" />

## Filling in missing vectors

Wherever missing vectors matter there is an **Embed** action, and each one
runs a [Reprocess](./dataset-import.md) dialog: a dry run first, then the real
run behind a confirmation.

- The dashboard's Detections summary offers **Embed N detections** when the
  backend suggests it.
- Semantic-search results and ordered views of a cluster show "N items in
  scope have no vector and are not ranked" with the same action.
- An empty review queue whose reason mentions vectors offers it too.
- Reprocessing a chosen batch of crops with the `embed` scope offers
  "Only items without a vector" and which parts to embed.
- The **auto-label** run on the dashboard has an **Embed missing vectors
  first** checkbox, and its progress shows an "embedding items without a
  vector" stage.

## Region writes

After you edit or accept region boxes, the backend reports how many boxes
still lack a vector. `/review` shows "N boxes have no vector yet" with an
**Embed now** action while that number is above zero.
