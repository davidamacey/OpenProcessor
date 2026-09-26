---
sidebar_position: 6
title: Export
---

# Export (`/export`)

- Per-class balance and trainability gaps.
- A one-shot **test-holdout freeze** — deterministic selection (a hash of
  each crop id per class), no seed input.
- **YOLO export** to a versioned directory, with served image/object/
  per-split counts and downloadable `class_registry.json`, `data.yaml` and
  `manifest.json`.
- An option to export only images whose every object is labeled, with the
  served partial-frame counts shown when present.

Every count is the server's own — a field the backend hasn't started
recording yet renders as "—", never a misleading 0.

<Screenshot name="export-1600.png" alt="Cropwright Export page" caption="Export — holdout freeze, YOLO export and split counts" />
