---
sidebar_position: 3
title: Annotation profiles
---

# Configuring your own domain

A domain — what class of object you're labeling, and whether it has a
sub-region like a defect zone or a tag — is **backend configuration, not a
Cropwright rebuild**.

1. The backend's served region profile (if any) drives the built-in region
   tab, gallery and sub-box editor automatically. See
   [Architecture overview](../getting-started/architecture-overview.md#the-served-region-profile).
2. A deployment can further customize labels, keymap, or cohorts, or add a
   **second** annotation slot the backend doesn't know about, by dropping
   an `annotation-profiles.json` file next to the built app — no fork, no
   rebuild.
   - `static/annotation-profiles.example.json` in the Cropwright repo is a
     worked example that customizes a served profile.
   - `examples/annotation-profiles/` has three more: license plate,
     aircraft tail number, and defect code.
   - The schema is documented in
     `docs/annotation-slots-contract-draft.md` in the repo.

A missing or malformed profile file never crashes the app — it's dropped
with a console warning and a toast, and Cropwright falls back to whatever
the backend serves.

## The region-profile rule

A tier-2 slot that declares any region route (browse path, endpoints,
thumbnail, or a cohort path under `/regions` or `/crops/{id}/region*`) is
kept only when its key matches the backend's own served profile name — in
which case it *replaces* the synthesized slot wholesale. Any other region
slot is dropped with a warning. A slot that touches no region route is
always kept.
