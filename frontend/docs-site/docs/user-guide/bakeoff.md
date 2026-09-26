---
sidebar_position: 8
title: Model comparison (bake-off)
---

# Model comparison (`/bakeoff`)

Compare trained and baseline models on evaluation datasets in one run.
Present only when the backend mounts its bake-off router — absent, not
disabled, otherwise.

1. Pick evaluation datasets (export test splits, or external frozen sets
   grouped by the backend's own grouping).
2. Pick models (finished training runs, baseline models, or a custom
   reference), each showing per-dataset facts and a train/test overlap
   warning when relevant.
3. Pick a scoring profile.
4. Confirm and run — the job polls with progress and per-stage failures.

## Results

A model × dataset matrix with every served tied winner bolded, plus ranked
per-dataset results and a per-class table (an uncovered class reads "not
covered", not a blank). A pre-v2 result shows a compatibility note instead
of a broken render. Nothing here computes a metric, mapping, rank or winner
client-side — it's all served.
