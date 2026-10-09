---
sidebar_position: 9
title: Models
---

# Model registry (`/p/<project>/models`)

Live status of every inference service the backend uses — Triton models and
external services such as the VLM and segmenter — with inference counts and
latency.

- Models the backend marks unloadable get an **Unload** button.
- Pipeline-protected models (the region detector, ingest proposer/
  classifier, OCR det/rec pair) show a "protected: in use by the pipeline"
  chip instead, never a button that would 403.
- A service the backend marks not configured renders "—" for its
  inference/latency fields, never a false 0.

## Sharing models between projects

A promoted model belongs to the project that trained it. The page also lists
models **other projects have shared**, each with a "from `<project>`" chip.

- **Class mapping.** A model that serves a class list shows "N classes map"
  and the class names that don't, and a **Class mapping** table that matches
  the model's classes to your project's classes by name (with how each one
  matched). Class ids are never shown.
- **Share / stop sharing.** On a model your project owns, the owner can
  choose **Share with other projects** or **Stop sharing**, after a
  confirmation. The button needs the model's current sharing revision from
  the backend; if the model changed meanwhile, the list reloads and you retry.
  A model you don't own has no such button and can't be unloaded here.
- **Stopping while in use.** The backend refuses to unshare a model that
  another project's active region profile uses, and the dialog lists each
  project and profile that does. **Unshare anyway** is behind a second
  confirmation and forces it; use it only when you've checked with those
  projects. If the backend couldn't check every project, you see its message
  and may retry or force the same way.
- A refused unload (a model in use) shows the backend's own reason.

## VLM endpoints

Each registered VLM endpoint is its own row, with its resolved model, an
**active** chip, and a link to [VLM models](./vlm-models.md) when the
registry is available.

<Screenshot name="cropwright/models-1600.png" alt="Cropwright Model registry page" caption="Models — live status of every inference service" />

<Screenshot name="cropwright/models-sharing-1600.png" alt="Models page showing the owner's model shared with other projects" caption="Models — a model shared with other projects, with Stop sharing" />

<Screenshot name="cropwright/models-unshare-force-1600.png" alt="Unshare dialog listing the projects whose profile uses the model" caption="Unshare — the server refuses while another project's profile uses the model" />
