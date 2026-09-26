---
sidebar_position: 9
title: Models
---

# Model registry (`/models`)

Live status of every inference service the backend uses — Triton models and
external services such as the VLM and segmenter — with inference counts and
latency.

- Models the backend marks unloadable get an **Unload** button.
- Pipeline-protected models (the region detector, ingest proposer/
  classifier, OCR det/rec pair) show a "protected: in use by the pipeline"
  chip instead, never a button that would 403.
- A service the backend marks not configured renders "—" for its
  inference/latency fields, never a false 0.
