---
sidebar_position: 7
title: Train
---

# Train cockpit (`/train`)

Submit a training job against the current export.

- **Class selection** defaults to classes with enough data, with an
  "exclude classes without enough data" option.
- **Model size**, **augmentation preset** (served list), **GPU
  allocation**, and a debounced **preflight** report — every check renders
  generically by name/severity/message, so a new backend check needs no
  frontend change.
- **Live progress** and a **log tail** while a run is active; multi-size
  campaigns get an auto-promote-best + stop-when threshold.

## Past runs

- **Results** — test-split evaluation (labelled by its actual split, "val"
  vs "test"), per-class table, MLflow link, and full lineage (export
  identity, class remap, code versions), fetched lazily on first open.
- **Promote ↑** — to the Triton model registry, with the served gate report
  and, when the backend allows it, a "promote anyway" override.
- **Reproduce** — resubmits the same spec/lineage as a fresh job.
- **Run probe predictions** — scores the pool with the run's checkpoint,
  which is what feeds the Uncertainty and Model Disagreements review
  queues.

<Screenshot name="train-1600.png" alt="Cropwright training cockpit" caption="Train — preflight, live progress, past-run results" />

## Training cohorts

A per-class cohort picker sourced from the backend
(`GET {API_PREFIX}/training_cohorts`): 4 class-agnostic cohorts plus, with a
region profile, 5 region-training-candidate modes. Thresholds shown are the
server's own, never a client constant. This is a curation preview only — it
never filters the actual training run.
