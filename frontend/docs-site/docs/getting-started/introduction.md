---
sidebar_position: 1
title: Introduction
---

# Introduction

Cropwright is a **keyboard-first, cluster-assisted web app for labeling image
crops at scale** — the human-in-the-loop frontend for
[OpenProcessor](https://github.com/example-org/OpenProcessor).

It is a **pure frontend**: no database of its own, no offline or mock mode.
Every piece of data it shows — clusters, crops, classes, training runs — comes
from a running OpenProcessor backend over one HTTP API prefix. Cropwright has
nothing useful to display without that backend.

## What it does

Cropwright manages a high-volume labeling workflow over hundreds of thousands
of image crops:

- **Cluster-based assisted labeling** — crops the backend has grouped by
  visual similarity are triaged as a batch, not one at a time.
- **VLM-assisted suggestions** — an optional vision-language model proposes
  a class per crop or cluster; a human confirms, corrects, or rejects.
- **A keyboard-first review queue** — class letters, undo, discard and skip
  are single keystrokes, tuned for a labeling session that runs for hours.
- **A full pipeline cockpit** — ingest, cluster, review, manage classes,
  export a YOLO dataset, train, compare models, and promote — all from one
  app, in that order, in a loop.

## Domain-agnostic by design

Cropwright ships with **no domain built in**. What you're labeling — vehicles,
manufacturing defects, aircraft tail numbers, or nothing with a sub-region at
all — is defined entirely by the OpenProcessor backend's **served region
profile**, plus, optionally, a small JSON config dropped in at deploy time.
See [Annotation profiles](../configuration/annotation-profiles.md).

## Requirements

- A reachable **OpenProcessor** backend (the data and model backend — a
  separate project).
- Docker with Compose v2 (the supported deployment path), or Node 20+ for a
  source build.

There is no way to "try Cropwright" without a backend — there's no sample
data or mock mode baked into the app itself. OpenProcessor's own sample-data
tooling (`make sample-coco`, a small public COCO val2017 subset) is the
fastest way to get something to label.

## Where to go next

- [Quick start](./quick-start.md) — get a Cropwright instance talking to a
  backend.
- [Architecture overview](./architecture-overview.md) — how the pieces fit
  together.
- The [User Guide](../user-guide/dashboard.md) — a page for every route.
